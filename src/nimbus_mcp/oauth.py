"""Hosted-gateway OAuth 2.1 authorization server (PR-O3, scope doc §1 Option A).

Opt-in with ``NIMBUS_MCP_OAUTH=1`` on ``nimbus-mcp serve``. MCP clients
(Claude, Cursor, ChatGPT, Inspector) then discover the gateway's metadata,
register (DCR), send the user to Nimbus Studio to Allow/Deny, and receive a
normal ``nimb_`` API token (30 days, no refresh) as the bearer access token.

The gateway keeps no state: client registrations and one-time codes live in
the backend behind ``MCP_OAUTH_GATEWAY_SECRET`` routes, and the token IS the
existing hosted token — revocation, quotas, scopes and the Account → API
tokens list all keep working. Requests with ``X-Nimbus-Token`` (or a pasted
``Authorization: Bearer nimb_…``) are accepted exactly as before.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from typing import Any

import httpx
from fastmcp.server.auth.auth import AccessToken, OAuthProvider
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    RefreshToken,
    TokenError,
)
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .credentials import decode_token_claims

logger = logging.getLogger("nimbus_mcp.oauth")

OAUTH_ENV_FLAG = "NIMBUS_MCP_OAUTH"
GATEWAY_SECRET_ENV = "MCP_OAUTH_GATEWAY_SECRET"
GATEWAY_SECRET_HEADER = "X-Nimbus-Gateway-Secret"
AUTHORIZE_REQUEST_ISSUER = "nimbus-mcp-gateway"
AUTHORIZE_REQUEST_AUDIENCE = "nimbus-mcp-authorize"
AUTHORIZE_REQUEST_TTL_SECONDS = 1800  # 30 min: room for a slow sign-up
DEFAULT_PUBLIC_URL = "https://nimbus-mcp.fly.dev"
DEFAULT_STUDIO_URL = "https://studio.nimbusbci.com"
NIMBUS_SCOPES = ["read", "run", "write", "leaderboard:submit"]
# Access-token verification cache (positive results only; backend is authoritative).
_VERIFY_TTL_SECONDS = 60.0
_CLIENT_CACHE_MAX = 512


def oauth_enabled(env: dict[str, str] | None = None) -> bool:
    value = (env if env is not None else os.environ).get(OAUTH_ENV_FLAG, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def sign_authorize_request(payload: dict[str, Any], secret: str) -> str:
    """HS256 JWT the Studio consent page posts back to the backend.

    Binds the gateway-validated client_id, redirect_uri, PKCE challenge and
    state, so Studio can only ever redirect to a registered redirect_uri.
    """
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url(sig)}"


# Public clients (PKCE, no secret) first: that is what MCP hosts register as.
TOKEN_ENDPOINT_AUTH_METHODS = ("none", "client_secret_post", "client_secret_basic")


class NimbusOAuthProvider(OAuthProvider):
    """fastmcp ``OAuthProvider`` backed by the Nimbus backend (stateless gateway)."""

    def __init__(
        self,
        *,
        base_url: str,
        backend_url: str,
        studio_url: str,
        gateway_secret: str,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            client_registration_options=ClientRegistrationOptions(
                enabled=True,
                valid_scopes=list(NIMBUS_SCOPES),
                # DCR clients that omit `scope` (e.g. Smithery) would otherwise be
                # stored with scope=None, and the SDK then rejects every
                # /authorize?scope=… with invalid_scope. The user still approves
                # on the Studio consent page.
                default_scopes=list(NIMBUS_SCOPES),
            ),
            revocation_options=RevocationOptions(enabled=False),
        )
        if len(gateway_secret) < 32:
            raise ValueError(f"{GATEWAY_SECRET_ENV} must be at least 32 characters")
        self._backend_url = backend_url.rstrip("/")
        self._studio_url = studio_url.rstrip("/")
        self._secret = gateway_secret
        self._transport = transport
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._verified: dict[str, tuple[float, AccessToken]] = {}
        self._lock = threading.Lock()

    def get_middleware(self) -> list:
        """Prepend the X-Nimbus-Token shim so it runs before the bearer gate."""
        from starlette.middleware import Middleware

        return [Middleware(XNimbusTokenAuthMiddleware), *super().get_middleware()]

    def get_routes(self, mcp_path: str | None = None) -> list:
        """Serve AS metadata that matches what the gateway actually accepts.

        The SDK's ``build_metadata`` hard-codes
        ``token_endpoint_auth_methods_supported=[client_secret_post, client_secret_basic]``
        and advertises ``refresh_token``. Public clients (Claude, Cursor, Claude Code:
        DCR with ``token_endpoint_auth_method=none`` + PKCE) are accepted by DCR and
        the token endpoint, and we never issue refresh tokens, so advertise
        ``none`` and only ``authorization_code``.
        """
        from mcp.server.auth.handlers.metadata import MetadataHandler
        from mcp.server.auth.routes import build_metadata, cors_middleware
        from starlette.routing import Route

        routes = super().get_routes(mcp_path)
        out: list = []
        for route in routes:
            if isinstance(route, Route) and route.path == "/.well-known/oauth-authorization-server":
                assert self.base_url is not None
                metadata = build_metadata(
                    self.base_url,
                    self.service_documentation_url,
                    self.client_registration_options or ClientRegistrationOptions(),
                    self.revocation_options or RevocationOptions(),
                )
                metadata.issuer = self.issuer_url  # type: ignore[assignment]
                metadata.token_endpoint_auth_methods_supported = list(TOKEN_ENDPOINT_AUTH_METHODS)
                metadata.grant_types_supported = ["authorization_code"]
                handler = MetadataHandler(metadata)
                out.append(
                    Route(
                        path=route.path,
                        endpoint=cors_middleware(handler.handle, ["GET", "OPTIONS"]),
                        methods=route.methods or ["GET", "OPTIONS"],
                        name=route.name,
                        include_in_schema=route.include_in_schema,
                    )
                )
            else:
                out.append(route)
                # Also serve protected-resource metadata at the root form
                # (E2E 2026-10-21: only /.well-known/oauth-protected-resource/mcp
                # answered; some clients probe the bare path first).
                prm = "/.well-known/oauth-protected-resource"
                if (
                    isinstance(route, Route)
                    and route.path.startswith(prm + "/")
                    and not any(isinstance(r, Route) and r.path == prm for r in routes)
                ):
                    out.append(
                        Route(
                            path=prm,
                            endpoint=route.endpoint,
                            methods=route.methods or ["GET", "OPTIONS"],
                            name=f"{route.name}-root" if route.name else None,
                            include_in_schema=route.include_in_schema,
                        )
                    )
        return out

    # ── backend plumbing ────────────────────────────────────────────────────

    def _http(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {"base_url": self._backend_url, "timeout": 15.0}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    def _gw_headers(self) -> dict[str, str]:
        return {GATEWAY_SECRET_HEADER: self._secret}

    # ── clients (DCR) ───────────────────────────────────────────────────────

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        cached = self._clients.get(client_id)
        if cached is not None:
            return cached
        async with self._http() as http:
            resp = await http.get(f"/api/mcp/oauth/clients/{client_id}", headers=self._gw_headers())
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        info = OAuthClientInformationFull.model_validate(resp.json()["clientInfo"])
        self._remember_client(info)
        return info

    def _remember_client(self, info: OAuthClientInformationFull) -> None:
        if info.scope is None:
            # Clients registered before default_scopes existed were stored
            # without a scope; treat them as registered for all supported scopes.
            info.scope = " ".join(NIMBUS_SCOPES)
        with self._lock:
            if len(self._clients) >= _CLIENT_CACHE_MAX:
                self._clients.clear()
            if info.client_id:
                self._clients[info.client_id] = info

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        payload = {
            "clientId": client_info.client_id,
            "clientInfo": client_info.model_dump(mode="json", exclude_none=True),
        }
        async with self._http() as http:
            resp = await http.post(
                "/api/mcp/oauth/clients", json=payload, headers=self._gw_headers()
            )
        resp.raise_for_status()
        self._remember_client(client_info)

    # ── authorize → Studio consent ──────────────────────────────────────────

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if params.scopes:
            unknown = set(params.scopes) - set(NIMBUS_SCOPES)
            if unknown:
                raise AuthorizeError(
                    error="invalid_scope",
                    error_description=f"Unknown scope(s): {', '.join(sorted(unknown))}",
                )
        now = int(time.time())
        request = sign_authorize_request(
            {
                "iss": AUTHORIZE_REQUEST_ISSUER,
                "aud": AUTHORIZE_REQUEST_AUDIENCE,
                "iat": now,
                "exp": now + AUTHORIZE_REQUEST_TTL_SECONDS,
                "cid": client.client_id,
                "cname": (client.client_name or "")[:128] or None,
                "ruri": str(params.redirect_uri),
                "rexp": params.redirect_uri_provided_explicitly,
                "cc": params.code_challenge,
                "st": params.state,
                "scp": params.scopes or None,
                "res": params.resource,
            },
            self._secret,
        )
        return f"{self._studio_url}/mcp/authorize?request={request}"

    # ── code → token ────────────────────────────────────────────────────────

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        async with self._http() as http:
            resp = await http.post(
                "/api/mcp/oauth/code",
                json={"code": authorization_code, "clientId": client.client_id},
                headers=self._gw_headers(),
            )
        if resp.status_code in (400, 404):
            return None
        resp.raise_for_status()
        data = resp.json()
        return AuthorizationCode(
            code=authorization_code,
            scopes=data.get("scopes") or [],
            expires_at=float(data["expiresAt"]),
            client_id=data["clientId"],
            code_challenge=data["codeChallenge"],
            redirect_uri=data["redirectUri"],
            redirect_uri_provided_explicitly=bool(data["redirectUriProvidedExplicitly"]),
            resource=data.get("resource"),
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        async with self._http() as http:
            resp = await http.post(
                "/api/mcp/oauth/token",
                json={
                    "code": authorization_code.code,
                    "clientId": client.client_id,
                    "redirectUri": str(authorization_code.redirect_uri),
                },
                headers=self._gw_headers(),
            )
        if resp.status_code in (400, 404):
            raise TokenError(error="invalid_grant", error_description="authorization code invalid")
        resp.raise_for_status()
        data = resp.json()
        scopes = data.get("scopes")
        return OAuthToken(
            access_token=data["accessToken"],
            token_type="Bearer",
            expires_in=int(data["expiresIn"]),
            scope=" ".join(scopes) if scopes else None,
        )

    # No refresh tokens: the access token is a 30-day nimb_ token (Q3 default).
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        return None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        raise TokenError(error="invalid_grant", error_description="refresh tokens are not issued")

    async def revoke_token(self, token: Any) -> None:  # revocation endpoint disabled
        return None

    # ── bearer verification ─────────────────────────────────────────────────

    async def load_access_token(self, token: str) -> AccessToken | None:
        """Accept any live ``nimb_`` token (OAuth-issued or pasted).

        A cheap local decode rejects malformed/expired tokens; the backend's
        ``/api/me/profile`` confirms it is not revoked (cached 60 s). A 403
        scope error still means the token is valid (e.g. a run-only key).
        """
        if not token.startswith("nimb_"):
            return None
        claims = decode_token_claims(token)
        exp = claims.get("exp")
        if not isinstance(exp, (int, float)) or exp <= time.time():
            return None
        key = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.monotonic()
        hit = self._verified.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        try:
            async with self._http() as http:
                resp = await http.get(
                    "/api/me/profile", headers={"Authorization": f"Bearer {token}"}
                )
        except httpx.HTTPError as err:
            # Backend down: don't bounce a well-formed token into a re-auth loop;
            # the tool call itself will surface the outage.
            logger.warning("OAuth bearer verification skipped (backend unreachable): %s", err)
            return self._access_token(token, claims)
        if resp.status_code == 401:
            return None
        if resp.status_code not in (200, 403):
            return self._access_token(token, claims)
        access = self._access_token(token, claims)
        with self._lock:
            if len(self._verified) >= 4096:
                self._verified.clear()
            self._verified[key] = (now + _VERIFY_TTL_SECONDS, access)
        return access

    @staticmethod
    def _access_token(token: str, claims: dict[str, Any]) -> AccessToken:
        scp = claims.get("scp")
        scopes = [s for s in scp if isinstance(s, str)] if isinstance(scp, list) else []
        exp = claims.get("exp")
        return AccessToken(
            token=token,
            client_id="nimbus",
            scopes=scopes,
            expires_at=int(exp) if isinstance(exp, (int, float)) else None,
            subject=str(claims.get("sub") or "") or None,
        )


class XNimbusTokenAuthMiddleware:
    """ASGI shim: a request carrying only ``X-Nimbus-Token`` (today's header
    contract) gets the same value as ``Authorization: Bearer`` so the OAuth
    bearer gate accepts it. When both headers are sent, Authorization (the one
    the gate verifies) wins and X-Nimbus-Token is dropped."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            headers = list(scope.get("headers") or [])
            names = {k.lower() for k, _ in headers}
            if b"authorization" in names and b"x-nimbus-token" in names:
                # The bearer gate verifies Authorization; tools prefer
                # X-Nimbus-Token. Drop the unverified one so the token that
                # was checked is the token that is used (a stale pasted
                # header must not shadow a fresh OAuth token).
                headers = [(k, v) for k, v in headers if k.lower() != b"x-nimbus-token"]
                scope = dict(scope, headers=headers)
            elif b"authorization" not in names:
                for k, v in headers:
                    if k.lower() == b"x-nimbus-token":
                        token = v.decode("latin-1").strip()
                        if token[:7].lower() == "bearer ":
                            token = token[7:].strip()
                        if token:
                            headers.append((b"authorization", f"Bearer {token}".encode("latin-1")))
                            scope = dict(scope, headers=headers)
                        break
        await self.app(scope, receive, send)


def provider_from_env(env: dict[str, str] | None = None) -> NimbusOAuthProvider:
    """Build the provider from the gateway environment (flag already checked)."""
    e = env if env is not None else os.environ
    secret = (e.get(GATEWAY_SECRET_ENV) or "").strip()
    if not secret:
        raise RuntimeError(f"{OAUTH_ENV_FLAG}=1 requires {GATEWAY_SECRET_ENV}")
    return NimbusOAuthProvider(
        base_url=(e.get("NIMBUS_MCP_PUBLIC_URL") or DEFAULT_PUBLIC_URL).rstrip("/"),
        backend_url=(e.get("NIMBUS_API_URL") or "https://nimbus-studio.fly.dev").rstrip("/"),
        studio_url=(e.get("NIMBUS_STUDIO_URL") or DEFAULT_STUDIO_URL).rstrip("/"),
        gateway_secret=secret,
    )
