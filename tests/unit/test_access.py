"""X9/X10 acceptance: Local Free auth model + tier gating (plan section 5.1).

Anonymous fresh installs reach local results with no login; login prompts
fire only at the listed cloud triggers; verified leaderboard submissions
are Pro-gated on hosted while Account-Free unverified is explicit (not
faked); exports are never gated.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastmcp import Client

from nimbus_mcp import access
from nimbus_mcp.access import (
    LOGIN_TRIGGERS,
    TierUnverifiable,
    account_tier,
    client_mode,
    login_prompt,
    tier_gate,
)
from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.server import build_server

CODE = "class M:\n    def fit(self, X, y, info=None): ...\n    def predict(self, X): ...\n"


def _client(tmp_path: Path, handler, *, token: str = "", key: str = "") -> NimbusClient:
    cfg = McpConfig(
        api_url="http://test",
        mcp_key=key,
        export_dir=tmp_path,
        nimbus_token=token,
        credential_meta={},
    )
    return NimbusClient(cfg, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _fresh_tier_cache():
    access.clear_tier_cache()
    yield
    access.clear_tier_cache()


def _no_http(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP call: {request.method} {request.url}")


# ── modes / prompt shapes ────────────────────────────────────────────────────


def test_client_mode_matrix(tmp_path):
    anon = _client(tmp_path, _no_http)
    local = _client(tmp_path, _no_http, key="k")
    hosted = _client(tmp_path, _no_http, token="t")
    assert (client_mode(anon), client_mode(local), client_mode(hosted)) == (
        "anonymous",
        "local",
        "hosted",
    )
    assert (account_tier(anon), account_tier(local)) == ("anonymous", "local")


def test_login_triggers_are_exactly_the_plans_list():
    assert set(LOGIN_TRIGGERS) == {
        "leaderboard_submit",
        "verified_history",
        "hosted_compute",
        "cloud_sync",
        "collaboration",
    }
    prompt = login_prompt("leaderboard_submit", feature="f", reason="r")
    assert prompt["loginRequired"] is True and prompt["trigger"] == "leaderboard_submit"
    gate = tier_gate("f", current_tier="account-free", detail="d")
    assert gate["tierGate"] is True and gate["requiredTier"] == "pro"


# ── anonymous: first local result with zero credentials ─────────────────────


async def test_anonymous_fresh_install_reaches_first_local_result(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/templates"
        assert "authorization" not in request.headers
        assert "x-mcp-key" not in request.headers
        return httpx.Response(200, json={"templates": [{"id": "csp_lda"}]})

    server = build_server(_client(tmp_path, handler))
    async with Client(server) as c:
        result = await c.call_tool("catalog.templates", {})
        assert not result.is_error
        assert result.data["templates"][0]["id"] == "csp_lda"
        who = await c.call_tool("account.whoami", {})
        assert who.data["authenticated"] is False
        assert who.data["loginRequired"] is False


async def test_anonymous_whoami_makes_no_http_call(tmp_path):
    server = build_server(_client(tmp_path, _no_http))
    async with Client(server) as c:
        who = await c.call_tool("account.whoami", {})
        assert who.data["tier"] == "local-free"


# ── leaderboard.submit gating ────────────────────────────────────────────────


def _submit_handler(posts: list, *, is_pro: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/me/profile":
            return httpx.Response(
                200, json={"capabilities": {"isPro": is_pro, "hasPioneerAccess": False}}
            )
        if request.url.path == "/api/leaderboard/submissions":
            posts.append(json.loads(request.content))
            return httpx.Response(200, json={"submissionId": "s1", "status": "queued"})
        raise AssertionError(f"unexpected {request.url.path}")

    return handler


async def test_submit_anonymous_gets_login_prompt_only_there(tmp_path):
    posts: list = []
    server = build_server(_client(tmp_path, _submit_handler(posts, is_pro=False)))
    async with Client(server) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "m", "code": CODE},
        )
        assert result.data["loginRequired"] is True
        assert result.data["trigger"] == "leaderboard_submit"
        assert posts == []  # nothing reached the backend


async def test_submit_hosted_free_verified_is_pro_gated(tmp_path):
    posts: list = []
    client = _client(tmp_path, _submit_handler(posts, is_pro=False), token="t")
    server = build_server(client)
    async with Client(server) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "m", "code": CODE},
        )
        assert result.data["tierGate"] is True
        assert result.data["requiredTier"] == "pro"
        assert result.data["currentTier"] == "account-free"
        assert posts == []


async def test_submit_hosted_free_unverified_is_explicit_not_faked(tmp_path):
    posts: list = []
    client = _client(tmp_path, _submit_handler(posts, is_pro=False), token="t")
    server = build_server(client)
    async with Client(server) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {
                "dataset_id": "BNCI2014_001",
                "name": "m",
                "code": CODE,
                "verification": "unverified_local",
            },
        )
        assert result.data["ok"] is False
        assert result.data["unsupported"] == ("unverified_local_submissions_not_yet_supported")
        assert posts == []  # no fake row was created


async def test_submit_hosted_pro_verified_proceeds_and_is_labelled(tmp_path):
    posts: list = []
    client = _client(tmp_path, _submit_handler(posts, is_pro=True), token="t")
    server = build_server(client)
    async with Client(server) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "m", "code": CODE},
        )
        assert result.data["submissionId"] == "s1"
        assert result.data["verification"] == "verified"
        assert result.data["scoring"] == "server-side"
        assert len(posts) == 1


async def test_submit_local_mode_is_not_tier_gated(tmp_path):
    posts: list = []
    client = _client(tmp_path, _submit_handler(posts, is_pro=False), key="k")
    server = build_server(client)
    async with Client(server) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "m", "code": CODE},
        )
        assert result.data["submissionId"] == "s1"
        assert len(posts) == 1  # no /api/me/profile gate fetch needed locally
        # local-backend row: never labelled as the hosted verified board
        assert result.data["verification"] == "local"


# ── exports are never gated ──────────────────────────────────────────────────


async def test_bids_export_anonymous_is_not_gated(tmp_path):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "nimbus_export_manifest.json",
            json.dumps({"tier": "bids", "subjects": [], "dataFiles": []}),
        )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/bids/export"
        return httpx.Response(
            200,
            content=buf.getvalue(),
            headers={"content-type": "application/zip"},
        )

    server = build_server(_client(tmp_path, handler))
    async with Client(server) as c:
        result = await c.call_tool("bids.export_dataset", {"dataset": "BNCI2014_001"})
        assert not result.is_error
        assert result.data.get("loginRequired") is not True
        assert result.data.get("tierGate") is not True


# ── foundation.compare: hosted Pro gate, local ungated ──────────────────────


async def test_foundation_compare_hosted_free_is_gated(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/me/profile":
            return httpx.Response(200, json={"capabilities": {"isPro": False}})
        raise AssertionError("compare must not run for hosted free accounts")

    client = _client(tmp_path, handler, token="t")
    server = build_server(client)
    async with Client(server) as c:
        with pytest.raises(Exception, match="Pro feature"):
            await c.call_tool(
                "foundation.compare", {"source": "dataset", "dataset": "BNCI2014_001"}
            )


async def test_foundation_compare_local_mode_not_gated(tmp_path):
    import base64

    png = base64.b64encode(b"\x89PNG fake").decode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/foundation/compare"
        return httpx.Response(200, json={"image": png, "title": "t", "meta": {"results": []}})

    client = _client(tmp_path, handler, key="k")
    server = build_server(client)
    async with Client(server) as c:
        result = await c.call_tool(
            "foundation.compare", {"source": "dataset", "dataset": "BNCI2014_001"}
        )
        assert not result.is_error


# ── fix pass: per-credential cache, TTL, 401 re-raise, campaigns ────────────


def _gateway(tmp_path: Path, pro_tokens: set[str], posts: list, current: dict):
    from nimbus_mcp.hosted import HeaderCredentialClient

    def factory(cfg: McpConfig) -> NimbusClient:
        token = cfg.nimbus_token

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/me/profile":
                return httpx.Response(200, json={"capabilities": {"isPro": token in pro_tokens}})
            posts.append(token)
            return httpx.Response(200, json={"submissionId": "s", "status": "queued"})

        return NimbusClient(cfg, transport=httpx.MockTransport(handler))

    cfg = McpConfig(api_url="http://test", mcp_key="", export_dir=tmp_path, credential_meta={})
    return HeaderCredentialClient(
        cfg,
        client_factory=factory,
        headers_getter=lambda: {"x-nimbus-token": current["token"]},
    )


@pytest.mark.parametrize("order", [("pro", "free"), ("free", "pro")])
async def test_gateway_tier_never_leaks_between_users(tmp_path, order):
    posts: list = []
    current = {"token": ""}
    proxy = _gateway(tmp_path, {"nimb_pro"}, posts, current)
    server = build_server(proxy)
    args = {"dataset_id": "BNCI2014_001", "name": "m", "code": CODE}
    async with Client(server) as c:
        for who in order:
            current["token"] = f"nimb_{who}"
            result = await c.call_tool("leaderboard.submit", args)
            if who == "pro":
                assert result.data["verification"] == "verified"
            else:
                assert result.data["tierGate"] is True
                assert result.data["currentTier"] == "account-free"
    assert posts == ["nimb_pro"]  # the free user never posted


def test_tier_cache_expires_after_ttl(tmp_path, monkeypatch):
    state = {"pro": False}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": {"isPro": state["pro"]}})

    client = _client(tmp_path, handler, token="t")
    assert account_tier(client) == "account-free"
    state["pro"] = True
    assert account_tier(client) == "account-free"  # cached
    real = access.time.monotonic
    monkeypatch.setattr(
        access.time, "monotonic", lambda: real() + access.TIER_CACHE_TTL_SECONDS + 1
    )
    assert account_tier(client) == "pro"  # upgrade visible after TTL


async def test_expired_token_gets_relogin_not_upgrade(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "expired"})

    server = build_server(_client(tmp_path, handler, token="t"))
    async with Client(server) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "m", "code": CODE},
            raise_on_error=False,
        )
    assert result.structured_content["setupRequired"] is True
    assert "tierGate" not in result.structured_content


def test_profile_failure_is_unverifiable_and_not_cached(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json={"capabilities": {"isPro": True}})

    client = _client(tmp_path, handler, token="t")
    with pytest.raises(TierUnverifiable) as exc:
        account_tier(client)
    assert exc.value.code == "tier_unverifiable"
    assert "not a plan limit" in str(exc.value)
    assert account_tier(client) == "pro"


def test_unreachable_backend_is_unverifiable_not_free(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(tmp_path, handler, token="t")
    with pytest.raises(TierUnverifiable):
        account_tier(client)


def test_profile_without_capabilities_is_unverifiable(tmp_path):
    client = _client(tmp_path, lambda r: httpx.Response(200, json={"id": "u"}), token="t")
    with pytest.raises(TierUnverifiable):
        account_tier(client)


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


async def test_leaderboard_submit_unreachable_backend_no_upgrade_nag(tmp_path):
    server = build_server(_client(tmp_path, _unreachable, token="t"))
    async with Client(server) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "m", "code": "class M: pass"},
            raise_on_error=False,
        )
    text = " ".join(getattr(b, "text", "") for b in result.content)
    data = result.structured_content or {}
    assert not data.get("tierGate")
    assert "Upgrade" not in text and "free tier" not in text
    assert "Cannot verify your Nimbus plan" in text


async def test_foundation_compare_unreachable_backend_no_upgrade_nag(tmp_path):
    server = build_server(_client(tmp_path, _unreachable, token="t"))
    async with Client(server) as c:
        result = await c.call_tool(
            "foundation.compare",
            {"source": "dataset", "dataset": "BNCI2014_001", "subject": "1"},
            raise_on_error=False,
        )
    text = " ".join(getattr(b, "text", "") for b in result.content)
    assert result.is_error
    assert "free tier" not in text
    assert "Cannot verify your Nimbus plan" in text


GRAPH = {
    "nodes": [{"id": "d", "type": "public_data", "config": {"dataset": "BNCI2014_001"}}],
    "connections": [],
}


async def test_hosted_free_big_campaign_is_gated(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/me/profile":
            return httpx.Response(200, json={"capabilities": {"isPro": False}})
        raise AssertionError(f"no run may start: {request.url.path}")

    server = build_server(_client(tmp_path, handler, token="t"))
    runs = [{"name": f"r{i}", "train_graph": GRAPH} for i in range(6)]
    async with Client(server) as c:
        result = await c.call_tool("experiment.run", {"runs": runs})
    assert result.data["tierGate"] is True
    assert result.data["feature"] == "experiment.run"


async def test_local_big_campaign_is_not_tier_gated(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        # Local backends are never tier-gated, whatever the profile says.
        if request.url.path == "/api/me/profile":
            return httpx.Response(200, json={"capabilities": {"isPro": False}})
        return httpx.Response(404, json={"detail": "x"})

    server = build_server(_client(tmp_path, handler, key="k"))
    runs = [{"name": f"r{i}", "train_graph": GRAPH} for i in range(6)]
    async with Client(server) as c:
        result = await c.call_tool("experiment.run", {"runs": runs})
    assert result.data.get("tierGate") is not True


def test_anonymous_guidance_does_not_lead_with_login():
    from nimbus_mcp.setup_mode import local_mode_guidance

    actions = [o["action"] for o in local_mode_guidance(401)["options"]]
    assert actions[0] != "nimbus-mcp login"
    assert actions[-1] == "nimbus-mcp login"
    # Unreachable (no desktop app): install first, login never first (L1).
    actions = [o["action"] for o in local_mode_guidance(None)["options"]]
    assert actions[0] == "install the Nimbus Studio desktop app"
    assert actions[0] != "nimbus-mcp login"
