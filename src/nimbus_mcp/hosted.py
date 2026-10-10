"""Hosted gateway: streamable-HTTP transport with per-request credentials.

``nimbus-mcp serve`` runs the same 48 tools over streamable HTTP so remote
MCP clients (and registries such as Smithery) can connect by URL. The
gateway process itself holds NO credential: each request's Nimbus API token
arrives as an MCP client header (``X-Nimbus-Token: nimb_…`` or
``Authorization: Bearer nimb_…`` — an Authorization-style ``Bearer ``
prefix on ``X-Nimbus-Token`` is stripped) and is resolved per call, so one
shared gateway serves many users as themselves. Headerless calls get the
hosted flavor of the setup guidance (add the header, or install locally via
uvx).

If the gateway process itself was started with ``NIMBUS_TOKEN`` in the
environment (single-tenant self-hosting), that token is the fallback for
headerless calls instead of guidance.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastmcp.server.dependencies import get_http_headers

from .client import NimbusClient, _credential_fingerprint, _mask_secret
from .config import McpConfig
from .setup_mode import SetupRequired

__all__ = [
    "HOSTED_SETUP_MESSAGE",
    "HOSTED_SETUP_OPTIONS",
    "HeaderCredentialClient",
    "hosted_setup_guidance",
]

HOSTED_SETUP_MESSAGE = (
    "This nimbus-mcp gateway is hosted, so it needs your Nimbus API token as a "
    'request header: add "X-Nimbus-Token": "nimb_…" to this server\'s '
    "headers in your MCP client config (mint one at Nimbus Studio → Account → "
    "API tokens, or via a local `nimbus-mcp login` flow). Prefer running the "
    "tools locally instead? Install with `uvx nimbus-mcp` — zero-config with "
    "the Nimbus desktop app, or one `nimbus-mcp login` for the hosted API."
)

HOSTED_SETUP_OPTIONS: list[dict[str, str]] = [
    {
        "action": 'add the "X-Nimbus-Token" header to this MCP server config',
        "detail": "Remote/gateway mode: put your hosted API token "
        "(nimb_…, minted in Nimbus Studio → Account → API tokens) in the "
        'server\'s headers — e.g. "headers": {"X-Nimbus-Token": "nimb_…"} '
        "in the mcpServers entry. Every request then acts as your account.",
    },
    {
        "action": "run locally with `uvx nimbus-mcp` instead",
        "detail": "Local mode keeps the token on your machine: "
        "`pip install nimbus-mcp` (or uvx), then `nimbus-mcp login` for the "
        "one-click device flow — or just run the Nimbus Studio desktop app "
        "and its key is auto-discovered.",
    },
]


def hosted_setup_guidance(**extra: Any) -> dict[str, Any]:
    """Setup payload for a hosted gateway call without a credential header."""
    guidance: dict[str, Any] = {
        "ok": False,
        "setupRequired": True,
        "message": HOSTED_SETUP_MESSAGE,
        "options": [dict(option) for option in HOSTED_SETUP_OPTIONS],
    }
    guidance.update(extra)
    return guidance


def _default_headers_getter() -> dict[str, str]:
    """fastmcp's get_http_headers strips credential headers (authorization,
    cookie) by default, which would make the Bearer form below unreachable on
    the real gateway — request authorization explicitly. ``include`` is
    absent on fastmcp <2.10: degrade to the plain call there (X-Nimbus-Token
    resolution is unaffected)."""
    try:
        return get_http_headers(include={"authorization"})
    except TypeError:
        return get_http_headers()


class HeaderCredentialClient:
    """Request-scoped credential resolution over one shared config.

    Duck-types :class:`nimbus_mcp.client.NimbusClient` (the interface the
    tool modules consume). Per request: read the MCP HTTP headers, build or
    reuse a :class:`NimbusClient` for that token (bounded cache — tools run
    in FastMCP worker threads), and translate any mid-session setup payload
    from the per-token client into the hosted flavor (the stock text tells
    people to run ``nimbus-mcp login``, which cannot fix a client-side
    header). ``config`` stays the gateway's base config so tools reading
    ``client.config.export_dir`` keep working.
    """

    def __init__(
        self,
        config: McpConfig,
        client_factory: Callable[[McpConfig], NimbusClient] = NimbusClient,
        headers_getter: Callable[[], dict[str, str]] = _default_headers_getter,
        max_clients: int = 32,
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._headers_getter = headers_getter
        self._max_clients = max_clients
        self._lock = threading.Lock()
        self._clients: dict[str, NimbusClient] = {}
        # Evicted-but-unclosed clients (see _client_for): closed by the NEXT
        # eviction, giving in-flight requests a full cycle to drain.
        self._retired: list[NimbusClient] = []

    # ── credential resolution ─────────────────────────────────────────────

    def _resolve_token(self) -> tuple[str | None, bool]:
        """``(token, from_header)`` — the request's credential and whether a
        request header supplied it (vs the startup fallback).

        Header keys arrive lowercase (Starlette convention) — check both
        spellings. ``X-Nimbus-Token`` values sometimes carry an
        Authorization-style ``Bearer `` prefix (MCP clients that template one
        header shape into both fields); it is stripped here,
        case-insensitively, so the backend receives
        ``Authorization: Bearer <token>`` — never a double-prefixed scheme
        the auth layer would reject.
        """
        try:
            headers = self._headers_getter() or {}
        except Exception:
            headers = {}
        token = (headers.get("x-nimbus-token") or headers.get("X-Nimbus-Token") or "").strip()
        if token[:7].lower() == "bearer ":
            token = token[7:].strip()
        elif token.lower() == "bearer":
            token = ""  # scheme with no credential is no credential
        if token:
            return token, True
        auth = headers.get("authorization") or headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if token:
                return token, True
        return self.config.nimbus_token or None, False

    def _current_token(self) -> str | None:
        """The request's token: X-Nimbus-Token, or Authorization: Bearer ….

        Returns the startup ``NIMBUS_TOKEN`` fallback (single-tenant
        self-hosting) when the request carries neither header.
        """
        return self._resolve_token()[0]

    def credential_report(self) -> dict[str, Any]:
        """Per-request counterpart of
        :meth:`nimbus_mcp.client.NimbusClient.credential_report`: which token
        carried THIS call (masked) and where it came from — ``"header"`` for
        request-header tokens, otherwise the base config's source (the
        startup ``NIMBUS_TOKEN`` fallback). No resolvable token → ``None``.
        """
        token, from_header = self._resolve_token()
        source = "header" if from_header else (self.config.credential_meta.get("source") or "env")
        return {"source": source, "token": _mask_secret(token) if token else None}

    def credential_fingerprint(self) -> str:
        """Per-request counterpart of
        :meth:`nimbus_mcp.client.NimbusClient.credential_fingerprint`: the
        fingerprint of the token carrying THIS call, so process-local state
        (campaign's experiment registry) is scoped per gateway user, not per
        gateway process. ``"anon"`` when no token is resolvable — campaign
        tools hit that only after their setup preflight already raised."""
        token = self._current_token()
        return _credential_fingerprint(token) if token else "anon"

    def _build_token_client(self, token: str) -> NimbusClient:
        return self._client_factory(
            replace(
                self.config,
                nimbus_token=token,
                mcp_key="",
                credential_meta={"source": "header"},
            )
        )

    def _client_for(self, token: str) -> NimbusClient:
        with self._lock:
            client = self._clients.get(token)
            if client is None:
                # Bounded: an unbounded map would leak a client (and socket)
                # per distinct token. Eviction must NOT close the current
                # set's clients outright — another request may be mid-flight
                # on them — so they are retired (references dropped) now and
                # closed by the NEXT eviction, which gives them a full cycle
                # to drain.
                if len(self._clients) >= self._max_clients:
                    previously_retired = self._retired
                    self._retired = list(self._clients.values())
                    self._clients.clear()
                    for stale in previously_retired:
                        stale.close()
                client = self._build_token_client(token)
                self._clients[token] = client
            return client

    def bound_client(self) -> Any:
        """A context manager yielding a NimbusClient bound to the CURRENT
        request's token, for code that outlives the request context (e.g.
        campaign's worker thread): fastmcp's header lookup answers {} there,
        so per-request resolution cannot run. The yielded client is fresh and
        worker-owned — never shared through the cache, so eviction cannot
        close it mid-experiment — and is closed when the context exits.
        Raises SetupRequired (hosted guidance) when no token is resolvable."""
        token = self._current_token()
        if not token:
            raise SetupRequired(hosted_setup_guidance())
        return closing(self._build_token_client(token))

    def _translate(self, err: SetupRequired) -> SetupRequired:
        """Re-flavor a per-token client's setup payload for gateway users.

        Keeps situation-specific extras (``daysExpired`` / ``daysLeft`` from
        a rejected token); only the message and options change, because the
        stock guidance's fix (``nimbus-mcp login``) cannot update a remote
        client's header.
        """
        guidance = hosted_setup_guidance(
            **{
                key: value
                for key, value in err.guidance.items()
                if key not in ("ok", "setupRequired", "message", "options")
            }
        )
        return SetupRequired(guidance)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        token = self._current_token()
        if not token:
            raise SetupRequired(hosted_setup_guidance())
        try:
            return getattr(self._client_for(token), method)(*args, **kwargs)
        except SetupRequired as err:
            raise self._translate(err) from err

    # ── NimbusClient interface ────────────────────────────────────────────

    def ensure_ready(self) -> None:
        if not self._current_token():
            raise SetupRequired(hosted_setup_guidance())

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._call("get", path, params)

    def post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        return self._call("post", path, json)

    def put(self, path: str, json: dict[str, Any] | None = None) -> Any:
        return self._call("put", path, json)

    def get_bytes(self, path: str) -> bytes:
        return self._call("get_bytes", path)

    def post_bytes(self, path: str, json: dict[str, Any] | None = None) -> bytes:
        return self._call("post_bytes", path, json)

    def post_file(
        self,
        path: str,
        file_path: Path,
        fields: dict[str, str] | None = None,
    ) -> Any:
        return self._call("post_file", path, file_path, fields)

    def close(self) -> None:
        with self._lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()
            for stale in self._retired:
                stale.close()
            self._retired.clear()
