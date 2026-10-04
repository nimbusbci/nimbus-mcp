"""Thin HTTP client for the Nimbus backend (X-MCP-Key locally, Bearer when hosted)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .config import McpConfig, load_config, missing_credential_error
from .errors import McpToolError
from .setup_mode import SetupRequired, setup_guidance, token_rejected_guidance

__all__ = ["McpToolError", "NimbusClient", "NullClient"]

_BACKEND_DOWN_HINT = (
    "Cannot reach the Nimbus backend at {url}. "
    "Start the desktop app (or the dev backend on :8080) and retry."
)
_AUTH_HINT = (
    "Nimbus rejected the MCP key ({status}). Check that MCP_LOCAL_KEY on the backend "
    "matches NIMBUS_MCP_KEY for this server, and that the backend runs locally "
    "(desktop app or DEBUG=1) — the key is never accepted on Fly deployments."
)

# Quota denials (run_pipeline / run_experiment 403s) that carry the pricing
# deep-link; the single place the link is appended is _backend_error (v0.5 T5).
_QUOTA_ERROR_CODES = frozenset(
    {"FREEMIUM_MONTHLY_QUOTA_EXCEEDED", "RUNTIME_PAID_TIER_REQUIRED"}
)
QUOTA_PRICING_URL = "https://studio.nimbusbci.com/pricing?reason=mcp-quota"


class NimbusClient:
    """Blocking HTTP client; tools run in FastMCP's worker threads.

    Auth mode follows the config: a ``nimbus_token`` (env token, token file,
    or the login store) sends ``Authorization: Bearer <token>`` (hosted
    backend, no X-MCP-Key header); otherwise the local ``mcp_key`` — the env
    key, the key file, or the desktop-app auto-discovered key — is sent as
    ``X-MCP-Key`` (the desktop key is local, so the header mode is right).
    """

    def __init__(self, config: McpConfig, transport: httpx.BaseTransport | None = None) -> None:
        if config.nimbus_token:
            headers = {"Authorization": f"Bearer {config.nimbus_token}"}
            self._token_mode = True
        elif config.mcp_key:
            headers = {"X-MCP-Key": config.mcp_key}
            self._token_mode = False
        else:
            raise missing_credential_error()
        self.config = config
        self._http = httpx.Client(
            base_url=config.api_url,
            headers=headers,
            timeout=300.0,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def ensure_ready(self) -> None:
        """Preflight hook for tools that do work outside the request path
        (campaign's ``run_experiment`` spawns a worker thread that would
        otherwise swallow a missing credential). A real client always has
        credentials attached, so this is a no-op; :class:`NullClient` raises
        :class:`SetupRequired` instead."""
        return None

    def _auth_rejection(self, status: int) -> McpToolError:
        """The exception for a 401 (see _request).

        Token mode: the token is expired or revoked — guidance with a re-login
        action and the day counts decoded locally (SetupRequired, converted to
        the guidance dict by the server middleware). Key mode: a wrong local
        key, which no login fixes — keep the MCP_LOCAL_KEY mismatch hint.
        """
        if self._token_mode:
            return SetupRequired(token_rejected_guidance(self.config.nimbus_token, status))
        return McpToolError(_AUTH_HINT.format(status=status))

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=json)

    def put(self, path: str, json: dict[str, Any] | None = None) -> Any:
        """PUT that tolerates 409 conflicts: returns a conflict payload instead
        of raising, so callers can implement optimistic-revision retries."""
        return self._request("PUT", path, json=json, allow_409=True)

    def get_bytes(self, path: str) -> bytes:
        return self._request("GET", path, content=True)

    def post_bytes(self, path: str, json: dict[str, Any] | None = None) -> bytes:
        return self._request("POST", path, json=json, content=True)

    def post_file(
        self, path: str, file_path: Path, fields: dict[str, str] | None = None
    ) -> Any:
        """POST a multipart/form-data request with one ``file`` part plus text fields."""
        data_fields = dict(fields or {})
        try:
            with open(file_path, "rb") as fh:
                resp = self._http.post(path, data=data_fields, files={"file": fh})
        except httpx.HTTPError as err:
            raise McpToolError(_BACKEND_DOWN_HINT.format(url=self.config.api_url)) from err
        if resp.status_code == 401:
            raise self._auth_rejection(resp.status_code)
        if resp.status_code >= 400:
            raise self._backend_error(resp)
        return self._json(resp)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        content: bool = False,
        allow_409: bool = False,
    ) -> Any:
        try:
            resp = self._http.request(method, path, params=params, json=json)
        except httpx.HTTPError as err:
            raise McpToolError(_BACKEND_DOWN_HINT.format(url=self.config.api_url)) from err
        # 401 = the credential was rejected by the auth layer (stale API token in
        # hosted mode, wrong key locally). 403 is a policy denial (e.g. freemium
        # limits) — let it fall through to the generic path so the backend's
        # detail message surfaces (with the pricing link for quota codes).
        if resp.status_code == 401:
            raise self._auth_rejection(resp.status_code)
        # 409 with allow_409 (PUT doc saves): an optimistic-revision conflict,
        # not a hard failure — hand it back so the caller can re-read the
        # revision and retry. Other >=400 statuses still raise below.
        if resp.status_code == 409 and allow_409:
            body = self._json(resp)
            # The backend's 409 is a flat RFC7807 problem body: top-level
            # ``code`` and a string ``detail`` (see backend errors.py
            # user_facing_problem_content). A nested detail.code is only a
            # fallback for non-problem payloads.
            code = body.get("code") if isinstance(body, dict) else None
            if not code and isinstance(body, dict) and isinstance(body.get("detail"), dict):
                code = body["detail"].get("code")
            return {"ok": False, "code": code or "conflict", "status": 409}
        if resp.status_code >= 400:
            raise self._backend_error(resp)
        return resp.content if content else self._json(resp)

    @staticmethod
    def _json(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            return {"ok": False, "raw": resp.text[:500]}

    @staticmethod
    def _backend_error(resp: httpx.Response) -> McpToolError:
        """The exception for a >=400 backend response.

        The message is the human-readable line the agent sees; the HTTP
        ``status_code`` and the backend's stable dotted ``code`` (top-level
        RFC7807 key, nested ``detail.code`` fallback) ride along as
        attributes so tools can branch on them (e.g. inspect_file's
        hosted-backend guidance) without parsing message text.
        """
        body: Any = None
        try:
            body = resp.json()
        except ValueError:
            pass
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, dict):
            msg = detail.get("detail") or detail.get("title") or str(detail)
        else:
            msg = str(detail) if detail else resp.text[:300]
        code = body.get("code") if isinstance(body, dict) else None
        if not code and isinstance(detail, dict):
            code = detail.get("code")
        # Quota denials (the two codes the backend returns from run_pipeline /
        # run_experiment) get the pricing deep-link appended — THIS is the single
        # place the link is added (v0.5 Task 5), so every tool benefits.
        if code in _QUOTA_ERROR_CODES:
            msg = f"{msg} Upgrade or manage your plan: {QUOTA_PRICING_URL}"
        return McpToolError(
            f"Nimbus API error {resp.status_code}: {msg}",
            status_code=resp.status_code,
            code=code if isinstance(code, str) else None,
        )


class NullClient:
    """Setup-mode stand-in for :class:`NimbusClient` (no credential resolved).

    Every request method raises :class:`~nimbus_mcp.setup_mode.SetupRequired`
    carrying the guidance payload; ``SetupGuidanceMiddleware`` (server.py)
    converts that into the structured setup dict, so the server starts,
    lists every tool, and each call answers with onboarding guidance instead
    of crashing. ``config`` is a real :class:`McpConfig` so tool code that
    reads ``client.config.export_dir`` (artifacts tools) keeps working.
    """

    def __init__(
        self,
        config: McpConfig | None = None,
        guidance: dict[str, Any] | None = None,
    ) -> None:
        self.config = load_config() if config is None else config
        self._guidance = guidance if guidance is not None else setup_guidance()

    def _setup_required(self) -> SetupRequired:
        return SetupRequired(self._guidance)

    def ensure_ready(self) -> None:
        """Preflight counterpart of :meth:`NimbusClient.ensure_ready`: no
        credential resolved, so raise — the caller fails fast with the
        guidance payload instead of dead-ending in a background thread."""
        raise self._setup_required()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        raise self._setup_required()

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        raise self._setup_required()

    def put(self, path: str, json: dict[str, Any] | None = None) -> Any:
        raise self._setup_required()

    def get_bytes(self, path: str) -> bytes:
        raise self._setup_required()

    def post_bytes(self, path: str, json: dict[str, Any] | None = None) -> bytes:
        raise self._setup_required()

    def post_file(
        self,
        path: str,
        file_path: Path,
        fields: dict[str, str] | None = None,
    ) -> Any:
        raise self._setup_required()

    def close(self) -> None:
        pass
