"""Gateway OAuth (PR-O3): NimbusOAuthProvider + flag wiring, backend mocked."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from starlette.testclient import TestClient

from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import HeaderCredentialClient
from nimbus_mcp.oauth import (
    AUTHORIZE_REQUEST_AUDIENCE,
    GATEWAY_SECRET_HEADER,
    NIMBUS_SCOPES,
    NimbusOAuthProvider,
    oauth_enabled,
    provider_from_env,
)
from nimbus_mcp.product_telemetry import Telemetry
from nimbus_mcp.server import build_server

SECRET = "unit-gateway-secret-0123456789abcdef0123456789"
GW = "https://gw.test"


def _b64(obj: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def _token(**claims: object) -> str:
    payload = {"sub": "user_1", "exp": int(time.time()) + 3600, **claims}
    return "nimb_" + f"{_b64({'alg': 'HS256'})}.{_b64(payload)}.sig"


class FakeBackend:
    def __init__(self) -> None:
        self.clients: dict[str, dict] = {}
        self.profile_status = 200
        self.calls: list[str] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(f"{req.method} {req.url.path}")
        if req.url.path.startswith("/api/mcp/oauth/"):
            assert req.headers.get(GATEWAY_SECRET_HEADER) == SECRET
        if req.url.path == "/api/mcp/oauth/clients" and req.method == "POST":
            body = json.loads(req.content)
            self.clients[body["clientId"]] = body["clientInfo"]
            return httpx.Response(200, json={"ok": True, **body})
        if req.url.path.startswith("/api/mcp/oauth/clients/"):
            cid = req.url.path.rsplit("/", 1)[1]
            if cid not in self.clients:
                return httpx.Response(404, json={"code": "nimbus.mcp.oauth_client_not_found"})
            return httpx.Response(200, json={"clientId": cid, "clientInfo": self.clients[cid]})
        if req.url.path == "/api/mcp/oauth/code":
            return httpx.Response(400, json={"code": "nimbus.mcp.oauth_code_invalid"})
        if req.url.path == "/api/me/profile":
            return httpx.Response(self.profile_status, json={"userId": "user_1"})
        return httpx.Response(404)


def _provider(backend: FakeBackend) -> NimbusOAuthProvider:
    return NimbusOAuthProvider(
        base_url=GW,
        backend_url="http://backend.test",
        studio_url="http://studio.test",
        gateway_secret=SECRET,
        transport=httpx.MockTransport(backend),
    )


def _app(provider: NimbusOAuthProvider | None):
    base = McpConfig(api_url="http://backend.test", mcp_key="", export_dir=Path("/tmp/nx"))
    server = build_server(
        HeaderCredentialClient(base), product_telemetry=Telemetry.disabled(), auth=provider
    )
    return server.http_app(path="/mcp")


_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
_ACCEPT = {"Accept": "application/json, text/event-stream"}


def test_flag_parsing():
    assert oauth_enabled({"NIMBUS_MCP_OAUTH": "1"})
    assert oauth_enabled({"NIMBUS_MCP_OAUTH": "true"})
    assert not oauth_enabled({})
    assert not oauth_enabled({"NIMBUS_MCP_OAUTH": "0"})


def test_provider_requires_secret():
    with pytest.raises(RuntimeError):
        provider_from_env({"NIMBUS_MCP_OAUTH": "1"})
    with pytest.raises(ValueError):
        provider_from_env({"MCP_OAUTH_GATEWAY_SECRET": "short"})


def test_flag_off_keeps_headerless_behavior():
    """No provider → no metadata routes and no 401 challenge (today's gateway)."""
    with TestClient(_app(None), base_url=GW) as c:
        assert c.get("/.well-known/oauth-protected-resource/mcp").status_code == 404
        resp = c.post("/mcp", json=_LIST, headers=_ACCEPT)
        assert resp.status_code != 401
        assert "www-authenticate" not in resp.headers


def test_flag_on_metadata_and_challenge():
    with TestClient(_app(_provider(FakeBackend())), base_url=GW) as c:
        prm = c.get("/.well-known/oauth-protected-resource/mcp")
        assert prm.status_code == 200
        # Root form served too (E2E 2026-10-21: it was a 404).
        root = c.get("/.well-known/oauth-protected-resource")
        assert root.status_code == 200 and root.json() == prm.json()
        assert prm.json()["authorization_servers"] == [f"{GW}/"]
        asm = c.get("/.well-known/oauth-authorization-server").json()
        assert asm["registration_endpoint"] == f"{GW}/register"
        assert "S256" in asm["code_challenge_methods_supported"]
        assert set(asm["scopes_supported"]) == {"read", "run", "write", "leaderboard:submit"}
        resp = c.post("/mcp", json=_LIST, headers=_ACCEPT)
        assert resp.status_code == 401
        assert 'resource_metadata="' in resp.headers["www-authenticate"]


@pytest.mark.parametrize("header", ["X-Nimbus-Token", "Authorization"])
def test_flag_on_existing_header_tokens_still_accepted(header):
    tok = _token()
    value = f"Bearer {tok}" if header == "Authorization" else tok
    with TestClient(_app(_provider(FakeBackend())), base_url=GW) as c:
        resp = c.post("/mcp", json=_LIST, headers={**_ACCEPT, header: value})
        assert resp.status_code != 401, resp.text


def test_flag_on_revoked_token_is_challenged():
    backend = FakeBackend()
    backend.profile_status = 401
    with TestClient(_app(_provider(backend)), base_url=GW) as c:
        resp = c.post("/mcp", json=_LIST, headers={**_ACCEPT, "X-Nimbus-Token": _token()})
        assert resp.status_code == 401


async def test_load_access_token_rules():
    backend = FakeBackend()
    p = _provider(backend)
    assert await p.load_access_token("not-nimbus") is None
    assert await p.load_access_token(_token(exp=int(time.time()) - 5)) is None
    ok = await p.load_access_token(_token(scp=["run"]))
    assert ok is not None and ok.scopes == ["run"] and ok.subject == "user_1"
    backend.profile_status = 403  # scope error = still a valid token
    assert await p.load_access_token(_token(jti="other")) is not None
    backend.profile_status = 401
    assert await p.load_access_token(_token(jti="revoked")) is None


async def test_load_access_token_caches_success():
    backend = FakeBackend()
    p = _provider(backend)
    tok = _token()
    await p.load_access_token(tok)
    await p.load_access_token(tok)
    assert backend.calls.count("GET /api/me/profile") == 1


async def test_register_and_get_client_roundtrip_via_backend():
    backend = FakeBackend()
    p = _provider(backend)
    info = OAuthClientInformationFull(
        client_id="cid-1", redirect_uris=["http://127.0.0.1:9/cb"], client_name="Claude"
    )
    await p.register_client(info)
    fresh = _provider(backend)  # another machine: no local cache
    got = await fresh.get_client("cid-1")
    assert got is not None and got.client_name == "Claude"
    assert await fresh.get_client("missing") is None


async def test_authorize_redirects_to_studio_with_signed_request():
    p = _provider(FakeBackend())
    client = OAuthClientInformationFull(
        client_id="cid-1", redirect_uris=["http://127.0.0.1:9/cb"], client_name="Claude"
    )
    url = await p.authorize(
        client,
        AuthorizationParams(
            state="st",
            scopes=["read"],
            code_challenge="abc",
            redirect_uri="http://127.0.0.1:9/cb",
            redirect_uri_provided_explicitly=True,
            resource=f"{GW}/mcp",
        ),
    )
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == "http://studio.test/mcp/authorize"
    raw = parse_qs(parts.query)["request"][0]
    header, body, sig = raw.split(".")
    expected = hmac.new(SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    assert base64.urlsafe_b64encode(expected).decode().rstrip("=") == sig
    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    assert claims["aud"] == AUTHORIZE_REQUEST_AUDIENCE
    assert claims["cid"] == "cid-1" and claims["ruri"] == "http://127.0.0.1:9/cb"
    assert claims["cc"] == "abc" and claims["st"] == "st" and claims["scp"] == ["read"]
    assert claims["exp"] - claims["iat"] == 1800


async def test_unknown_code_is_none():
    p = _provider(FakeBackend())
    client = OAuthClientInformationFull(client_id="cid-1", redirect_uris=["http://x/cb"])
    assert await p.load_authorization_code(client, "nope") is None


async def test_no_refresh_tokens():
    p = _provider(FakeBackend())
    client = OAuthClientInformationFull(client_id="cid-1", redirect_uris=["http://x/cb"])
    assert await p.load_refresh_token(client, "r") is None


# ── X-Nimbus-Token ↔ Authorization shim ─────────────────────────────────────


def _run_shim(headers: list[tuple[bytes, bytes]]) -> dict[bytes, bytes]:
    import asyncio

    from nimbus_mcp.oauth import XNimbusTokenAuthMiddleware

    seen: dict[bytes, bytes] = {}

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        seen.update({k.lower(): v for k, v in scope["headers"]})

    asyncio.run(XNimbusTokenAuthMiddleware(app)({"type": "http", "headers": headers}, None, None))
    return seen


def test_shim_maps_x_nimbus_token_to_bearer() -> None:
    seen = _run_shim([(b"x-nimbus-token", b"Bearer nimb_abc")])
    assert seen[b"authorization"] == b"Bearer nimb_abc"


def test_shim_verified_authorization_wins_over_stale_x_nimbus_token() -> None:
    seen = _run_shim([(b"authorization", b"Bearer nimb_fresh"), (b"x-nimbus-token", b"nimb_stale")])
    assert seen[b"authorization"] == b"Bearer nimb_fresh"
    assert b"x-nimbus-token" not in seen


def test_metadata_advertises_public_clients_and_no_refresh():
    """MCP hosts register public clients (auth method ``none`` + PKCE)."""
    with TestClient(_app(_provider(FakeBackend())), base_url=GW) as c:
        asm = c.get("/.well-known/oauth-authorization-server").json()
    assert asm["token_endpoint_auth_methods_supported"][0] == "none"
    assert set(asm["token_endpoint_auth_methods_supported"]) == {
        "none",
        "client_secret_post",
        "client_secret_basic",
    }
    assert asm["grant_types_supported"] == ["authorization_code"]
    assert asm["issuer"] == f"{GW}/"
    assert asm["token_endpoint"] == f"{GW}/token"


def test_public_client_dcr_and_token_endpoint_without_secret():
    """DCR with ``none`` gets no secret; /token authenticates it without one."""
    backend = FakeBackend()
    with TestClient(_app(_provider(backend)), base_url=GW) as c:
        reg = c.post(
            "/register",
            json={
                "client_name": "Claude",
                "redirect_uris": ["http://127.0.0.1:9/cb"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert reg.status_code == 201, reg.text
        body = reg.json()
        assert body["token_endpoint_auth_method"] == "none"
        assert not body.get("client_secret")
        tok = c.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "client_id": body["client_id"],
                "code": "bogus",
                "redirect_uri": "http://127.0.0.1:9/cb",
                "code_verifier": "v" * 50,
            },
        )
    # Client auth passed (no invalid_client); the unknown code is rejected.
    # (fastmcp maps an unknown code to 4xx invalid_grant; never invalid_client).
    assert 400 <= tok.status_code < 500, tok.text
    assert tok.json()["error"] == "invalid_grant"


def test_dcr_without_scope_can_authorize_scope_read():
    """Smithery-style DCR omits `scope`; /authorize?scope=read must not be invalid_scope."""
    backend = FakeBackend()
    with TestClient(_app(_provider(backend)), base_url=GW) as c:
        reg = c.post(
            "/register",
            json={
                "client_name": "Smithery",
                "redirect_uris": ["https://smithery.test/cb"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert reg.status_code == 201, reg.text
        assert set(reg.json()["scope"].split()) == set(NIMBUS_SCOPES)
        r = c.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": reg.json()["client_id"],
                "redirect_uri": "https://smithery.test/cb",
                "scope": "read",
                "state": "s",
                "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
                "code_challenge_method": "S256",
            },
            follow_redirects=False,
        )
    assert r.status_code == 302, r.text
    assert "invalid_scope" not in r.headers["location"]


async def test_legacy_client_stored_without_scope_gets_all_scopes():
    """Clients registered before the default (scope=None) are treated as all-scope."""
    backend = FakeBackend()
    p = _provider(backend)
    await p.register_client(
        OAuthClientInformationFull(client_id="old", redirect_uris=["https://x.test/cb"])
    )
    got = await _provider(backend).get_client("old")
    assert got is not None
    assert got.validate_scope("read") == ["read"]


def _dcr_and_authorize(c, reg_scope, auth_scope):
    body = {
        "client_name": "Edge",
        "redirect_uris": ["https://edge.test/cb"],
        "grant_types": ["authorization_code"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if reg_scope is not None:
        body["scope"] = reg_scope
    reg = c.post("/register", json=body)
    assert reg.status_code == 201, reg.text
    params = {
        "response_type": "code",
        "client_id": reg.json()["client_id"],
        "redirect_uri": "https://edge.test/cb",
        "state": "s",
        "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
        "code_challenge_method": "S256",
    }
    if auth_scope is not None:
        params["scope"] = auth_scope
    r = c.get("/authorize", params=params, follow_redirects=False)
    assert r.status_code == 302, r.text
    return reg.json(), r.headers["location"]


@pytest.mark.parametrize(
    ("reg_scope", "auth_scope", "ok"),
    [
        ("read", "read", True),
        ("read", "write", False),  # explicit subset still binds the client
        ("read", "read write", False),
        ("read run", "run", True),
        (None, "read write run leaderboard:submit", True),
        (None, None, True),  # no scope requested: unchanged pre-PR behaviour
        ("", "read", True),  # empty registration scope == omitted -> defaults
        (None, "", False),  # SDK rejects an empty `scope=` (fails closed)
    ],
)
def test_dcr_scope_edge_cases(reg_scope, auth_scope, ok):
    with TestClient(_app(_provider(FakeBackend())), base_url=GW) as c:
        _reg, loc = _dcr_and_authorize(c, reg_scope, auth_scope)
    assert ("invalid_scope" not in loc) is ok, loc


def test_dcr_unknown_scope_rejected_at_registration():
    with TestClient(_app(_provider(FakeBackend())), base_url=GW) as c:
        reg = c.post(
            "/register",
            json={
                "redirect_uris": ["https://edge.test/cb"],
                "token_endpoint_auth_method": "none",
                "scope": "read admin",
            },
        )
    assert reg.status_code == 400
    assert reg.json()["error"] == "invalid_client_metadata"


async def test_backfill_keeps_explicit_scopes():
    backend = FakeBackend()
    await _provider(backend).register_client(
        OAuthClientInformationFull(
            client_id="narrow", redirect_uris=["https://x.test/cb"], scope="read"
        )
    )
    got = await _provider(backend).get_client("narrow")
    assert got is not None and got.scope == "read"
