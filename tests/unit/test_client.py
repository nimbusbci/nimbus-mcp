from pathlib import Path

import httpx
import pytest

from nimbus_mcp.client import McpToolError, NimbusClient
from nimbus_mcp.config import McpConfig


def make_client(handler) -> NimbusClient:
    cfg = McpConfig(api_url="http://test", mcp_key="k", export_dir=Path("/tmp/nx"))
    return NimbusClient(cfg, transport=httpx.MockTransport(handler))


def make_token_client(handler) -> NimbusClient:
    cfg = McpConfig(
        api_url="http://test", mcp_key="", export_dir=Path("/tmp/nx"), nimbus_token="nimb_test"
    )
    return NimbusClient(cfg, transport=httpx.MockTransport(handler))


def test_get_returns_json():
    c = make_client(lambda req: httpx.Response(200, json={"ok": True, "nodeTypes": []}))
    assert c.get("/api/node-types") == {"ok": True, "nodeTypes": []}


def test_ensure_ready_is_noop_with_credentials():
    """A real client always has credentials attached: the preflight hook (used
    by run_experiment/get_experiment before spawning background work) must not
    raise and must not touch HTTP."""
    c = make_client(lambda req: pytest.fail("ensure_ready must not touch HTTP"))
    assert c.ensure_ready() is None


def test_mcp_key_header_sent():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["key"] = req.headers.get("X-MCP-Key")
        return httpx.Response(200, json={"ok": True})

    make_client(handler).get("/api/templates")
    assert seen["key"] == "k"


def test_token_mode_sends_bearer_and_no_mcp_key_header():
    """Hosted mode: Authorization: Bearer replaces X-MCP-Key entirely."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        seen["key"] = req.headers.get("X-MCP-Key")
        return httpx.Response(200, json={"ok": True})

    make_token_client(handler).get("/api/templates")
    assert seen["auth"] == "Bearer nimb_test"
    assert seen["key"] is None


def test_key_mode_sends_mcp_key_and_no_bearer():
    """Local mode is unchanged: X-MCP-Key only, no Authorization header."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("Authorization")
        seen["key"] = req.headers.get("X-MCP-Key")
        return httpx.Response(200, json={"ok": True})

    make_client(handler).get("/api/templates")
    assert seen["key"] == "k"
    assert seen["auth"] is None


def test_missing_key_raises_actionable_error():
    cfg = McpConfig(api_url="http://test", mcp_key="", export_dir=Path("/tmp/nx"))
    with pytest.raises(McpToolError, match="NIMBUS_MCP_KEY"):
        NimbusClient(cfg)


def test_401_raises_auth_hint():
    c = make_client(lambda req: httpx.Response(401, json={"detail": "no"}))
    with pytest.raises(McpToolError, match="MCP_LOCAL_KEY"):
        c.get("/api/templates")


def test_token_mode_401_raises_hosted_hint():
    """A rejected hosted token becomes SetupRequired guidance pointing at
    `nimbus-mcp login` (v0.5), not the local MCP_LOCAL_KEY hint."""
    from nimbus_mcp.setup_mode import SetupRequired

    c = make_token_client(lambda req: httpx.Response(401, json={"detail": "no"}))
    with pytest.raises(SetupRequired) as excinfo:
        c.get("/api/templates")
    guidance = excinfo.value.guidance
    assert guidance["ok"] is False
    assert guidance["setupRequired"] is True
    assert "nimbus-mcp login" in guidance["message"]
    assert "MCP_LOCAL_KEY" not in guidance["message"]


def test_403_surfaces_backend_detail_not_auth_hint():
    """403 is a policy denial (e.g. freemium limits), not a rejected key —
    the backend's detail must surface instead of the MCP-key hint."""
    c = make_client(
        lambda req: httpx.Response(
            403, json={"detail": "This action requires a Pro subscription."}
        )
    )
    with pytest.raises(McpToolError, match="requires a Pro subscription") as excinfo:
        c.post("/api/execute", json={})
    assert "MCP_LOCAL_KEY" not in str(excinfo.value)


# ── v0.5: quota 403s carry the pricing deep-link (appended in _backend_error) ──


@pytest.mark.parametrize(
    "code",
    ["FREEMIUM_MONTHLY_QUOTA_EXCEEDED", "RUNTIME_PAID_TIER_REQUIRED"],
)
def test_quota_403_appends_pricing_link(code):
    """The backend's quota denials are flat RFC7807 bodies with a top-level
    code — both get the ?reason=mcp-quota link appended to the message."""
    c = make_client(
        lambda req: httpx.Response(
            403,
            json={
                "code": code,
                "title": "Forbidden",
                "detail": "Monthly free training quota exceeded.",
            },
        )
    )
    with pytest.raises(McpToolError) as excinfo:
        c.post("/api/execute", json={})
    msg = str(excinfo.value)
    assert "Monthly free training quota exceeded." in msg
    assert "https://studio.nimbusbci.com/pricing?reason=mcp-quota" in msg


def test_quota_403_nested_detail_code_also_gets_link():
    c = make_client(
        lambda req: httpx.Response(
            403,
            json={"detail": {"code": "FREEMIUM_MONTHLY_QUOTA_EXCEEDED", "detail": "over"}},
        )
    )
    with pytest.raises(McpToolError, match=r"pricing\?reason=mcp-quota"):
        c.post("/api/experiment/run", json={})


def test_non_quota_403_gets_no_pricing_link():
    c = make_client(lambda req: httpx.Response(403, json={"code": "SOME_OTHER_CODE"}))
    with pytest.raises(McpToolError) as excinfo:
        c.post("/api/execute", json={})
    assert "pricing" not in str(excinfo.value)


def test_backend_error_detail_surfaced():
    c = make_client(lambda req: httpx.Response(422, json={"detail": "bad graph"}))
    with pytest.raises(McpToolError, match="bad graph"):
        c.post("/api/validate-pipeline", json={"train": {}})


def test_put_returns_conflict_payload_instead_of_raising_on_409():
    """allow_409 maps 409 to a returned dict so callers can retry on revision;
    the code comes from the flat RFC7807 problem body's top-level ``code``."""
    c = make_client(
        lambda req: httpx.Response(
            409, json={"code": "nimbus.project.doc_conflict", "detail": "stale"}
        )
    )
    assert c.put("/api/projects/p1/doc", json={}) == {
        "ok": False, "code": "nimbus.project.doc_conflict", "status": 409,
    }


def test_put_409_nested_detail_code_still_supported():
    """Fallback: a nested detail.code still wins when there is no top-level code."""
    c = make_client(
        lambda req: httpx.Response(409, json={"detail": {"code": "nimbus.project.doc_conflict"}})
    )
    assert c.put("/api/projects/p1/doc", json={})["code"] == "nimbus.project.doc_conflict"


def test_put_409_without_detail_code_maps_to_conflict():
    c = make_client(lambda req: httpx.Response(409, json={"detail": "doc_conflict"}))
    assert c.put("/api/projects/p1/doc", json={}) == {
        "ok": False, "code": "conflict", "status": 409,
    }


def test_put_still_raises_on_other_client_errors():
    c = make_client(lambda req: httpx.Response(422, json={"detail": "bad graph"}))
    with pytest.raises(McpToolError, match="bad graph"):
        c.put("/api/projects/p1/doc", json={})


def test_connection_error_raises_backend_down_hint():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    c = make_client(handler)
    with pytest.raises(McpToolError, match="Cannot reach the Nimbus backend"):
        c.get("/api/templates")


# ── v0.6: backend >=400 errors carry structured status_code + code attrs ─────
# inspect_file branches on these to convert the hosted local-path refusal into
# guidance, instead of parsing the human-readable message text.


def test_backend_error_carries_status_and_code_attributes():
    c = make_client(
        lambda req: httpx.Response(
            403,
            json={
                "type": "urn:nimbus:error:Forbidden",
                "title": "Forbidden",
                "status": 403,
                "detail": "Local file inspection is only available on a local backend.",
                "code": "nimbus.data.describe_local_forbidden",
            },
        )
    )
    with pytest.raises(McpToolError) as excinfo:
        c.get("/api/data/describe", params={"source": "local", "path": "/x.csv"})
    assert excinfo.value.status_code == 403
    assert excinfo.value.code == "nimbus.data.describe_local_forbidden"


def test_backend_error_code_is_none_without_problem_body():
    c = make_client(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(McpToolError, match="Nimbus API error 500") as excinfo:
        c.get("/api/data/describe")
    assert excinfo.value.status_code == 500
    assert excinfo.value.code is None
