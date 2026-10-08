"""Setup mode: structured guidance from every tool; 401-expired → guidance.

No tool module knows setup mode exists — the null client raises
``SetupRequired`` and ``SetupGuidanceMiddleware`` (server.py) converts it, so
these tests drive the real FastMCP server end-to-end via the in-memory client.
"""

import base64
import json
import time
from pathlib import Path

import httpx
import pytest
from fastmcp import Client

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.server import build_server
from nimbus_mcp.setup_mode import SetupRequired, setup_guidance, token_rejected_guidance

TRAIN_GRAPH = {
    "nodes": [
        {"id": "d", "type": "public_data", "config": {"dataset": "BNCI2014_001"}},
        {"id": "m", "type": "rxlda_sdk", "config": {}},
    ],
    "connections": [{"from": "d", "to": "m"}],
}

ENV_VARS = (
    "NIMBUS_TOKEN",
    "NIMBUS_TOKEN_FILE",
    "NIMBUS_MCP_KEY",
    "NIMBUS_MCP_KEY_FILE",
    "NIMBUS_API_URL",
    "NIMBUS_CREDENTIALS_FILE",
    "XDG_CONFIG_HOME",
    "APPDATA",
)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Hermetic HOME + no NIMBUS_* / desktop-key leakage from the dev shell."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("NIMBUS_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _b64url(obj: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def fake_token(payload: dict) -> str:
    """A locally-decodable nimb_ JWT (decode_token_claims verifies nothing)."""
    return "nimb_" + f"{_b64url({'alg': 'none'})}.{_b64url(payload)}.sig"


def _token_config(payload: dict, meta: dict | None = None) -> McpConfig:
    return McpConfig(
        api_url="http://t",
        mcp_key="",
        export_dir=Path("/tmp/nx"),
        nimbus_token=fake_token(payload),
        credential_meta=meta if meta is not None else {"source": "store"},
    )


def assert_is_guidance(data: dict) -> None:
    """Core shape (spec keys); the 401 variants adapt the message wording."""
    assert data["ok"] is False
    assert data["setupRequired"] is True
    assert "nimbus-mcp login" in data["message"]
    actions = [option["action"] for option in data["options"]]
    assert actions == [
        "nimbus-mcp login",
        "start the Nimbus Studio desktop app",
        "set NIMBUS_MCP_KEY / NIMBUS_MCP_KEY_FILE",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# setup mode: no credential → every tool returns the guidance dict
# ─────────────────────────────────────────────────────────────────────────────


async def test_setup_mode_guidance_from_three_tool_families():
    """catalog.templates (discovery), execution.run (run), device.list (live) —
    no HTTP happens; each call returns the structured guidance dict."""
    server = build_server()  # isolated env resolves nothing → NullClient
    async with Client(server) as c:
        tools = {tool.name for tool in (await c.list_tools())}
    assert "account.whoami" in tools  # every tool is registered in setup mode
    assert len(tools) >= 30

    async with Client(server) as c:
        for name, args in (
            ("catalog.templates", {}),
            ("execution.run", {"train_graph": TRAIN_GRAPH}),
            ("device.list", {}),
            ("account.whoami", {}),
        ):
            result = await c.call_tool(name, args)
            assert not result.is_error
            assert_is_guidance(result.data)
            # the default (no-credential) message names all three paths
            assert "NIMBUS_TOKEN" in result.data["message"]
            assert "NIMBUS_MCP_KEY" in result.data["message"]
            assert "desktop app" in result.data["message"]


def test_null_client_every_method_raises_setup_required():
    from nimbus_mcp.client import NullClient

    null = NullClient()
    for call in (
        lambda: null.ensure_ready(),
        lambda: null.get("/api/me/profile"),
        lambda: null.post("/api/execute", json={}),
        lambda: null.put("/x", json={}),
        lambda: null.get_bytes("/x"),
        lambda: null.post_bytes("/x"),
        lambda: null.post_file("/x", Path("/tmp/f")),
    ):
        with pytest.raises(SetupRequired) as excinfo:
            call()
        assert_is_guidance(excinfo.value.guidance)
    null.close()  # no-op, must not raise
    assert null.config.export_dir  # real config: tool code touching config works


async def test_setup_mode_experiment_tools_preflight_before_worker():
    """experiment.run preflights the credential BEFORE registering state or
    spawning the worker thread — the worker's catch-all would otherwise
    swallow SetupRequired into per-run errors and the tool would answer a
    misleading {status: "running"} instead of the guidance dict. experiment.get
    for symmetry: in setup mode no experiment can exist, so guidance beats the
    confusing "Unknown experiment" error."""
    from nimbus_mcp.tools import campaign

    before = set(campaign.EXPERIMENTS._states)
    server = build_server()  # isolated env resolves nothing → NullClient
    async with Client(server) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        assert not result.is_error
        assert_is_guidance(result.data)

        poll = await c.call_tool("experiment.get", {"experiment_id": "anything"})
        assert not poll.is_error
        assert_is_guidance(poll.data)
    # nothing was registered and no worker thread was spawned
    assert set(campaign.EXPERIMENTS._states) == before


# ─────────────────────────────────────────────────────────────────────────────
# 401 mid-session: expired / revoked hosted token → guidance with day counts
# ─────────────────────────────────────────────────────────────────────────────


async def test_401_expired_token_returns_guidance_with_days_expired():
    exp = int(time.time()) - 3 * 86400  # expired exactly 3 days ago
    client = NimbusClient(
        _token_config({"exp": exp}),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "no"})),
    )
    server = build_server(client)
    async with Client(server) as c:
        result = await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH})
    assert_is_guidance(result.data)
    assert result.data["daysExpired"] == 3
    assert "expired 3 day" in result.data["message"]
    assert result.data["options"][0]["action"] == "nimbus-mcp login"


async def test_401_revoked_token_reports_days_left():
    exp = int(time.time()) + 10 * 86400  # unexpired → revoked/plan-changed story
    client = NimbusClient(
        _token_config({"exp": exp}),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "no"})),
    )
    server = build_server(client)
    async with Client(server) as c:
        result = await c.call_tool("catalog.templates", {})
    assert_is_guidance(result.data)
    assert result.data["daysLeft"] == 10
    assert "revoked" in result.data["message"]


def test_401_undecodable_token_still_guides_to_login():
    client = NimbusClient(
        McpConfig(
            api_url="http://t",
            mcp_key="",
            export_dir=Path("/tmp/nx"),
            nimbus_token="nimb_not-a-jwt",
        ),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})),
    )
    with pytest.raises(SetupRequired) as excinfo:
        client.get("/api/templates")
    assert_is_guidance(excinfo.value.guidance)
    assert "daysLeft" not in excinfo.value.guidance
    assert "daysExpired" not in excinfo.value.guidance


def test_key_mode_401_keeps_local_key_hint():
    """A rejected local key is a config mismatch, not a login problem — the
    MCP_LOCAL_KEY hint must survive (back-compat with the v0.3 copy)."""
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "no"})),
    )
    with pytest.raises(Exception) as excinfo:
        client.get("/api/templates")
    assert not isinstance(excinfo.value, SetupRequired)
    assert "MCP_LOCAL_KEY" in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────────
# quota 403 error copy surfaces through a real tool
# ─────────────────────────────────────────────────────────────────────────────


async def test_run_pipeline_quota_403_carries_pricing_link():
    """The pricing link is appended in NimbusClient._backend_error (the single
    place); through the tool layer it surfaces in the raised error message."""
    from fastmcp.exceptions import ToolError

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "code": "nimbus.freemium.monthly_quota_exceeded",
                "title": "Forbidden",
                "detail": "Monthly free training quota exceeded.",
            },
        )

    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = build_server(client)
    async with Client(server) as c:
        with pytest.raises(ToolError) as excinfo:
            await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH})
    message = str(excinfo.value)
    assert "Monthly free training quota exceeded." in message
    assert "https://studio.nimbusbci.com/pricing?reason=mcp-quota" in message


# ─────────────────────────────────────────────────────────────────────────────
# guidance payload unit checks
# ─────────────────────────────────────────────────────────────────────────────


def test_setup_guidance_shape_is_spec_verbatim():
    guidance = setup_guidance()
    assert set(guidance) == {"ok", "setupRequired", "message", "options"}
    assert len(guidance["options"]) == 3


def test_token_rejected_guidance_expiry_day_math():
    exp = time.time() - 0.5 * 86400  # half a day ago → at least 1 day reported
    guidance = token_rejected_guidance(fake_token({"exp": exp}), 401)
    assert guidance["daysExpired"] == 1


def test_build_server_reports_package_version():
    """The initialize handshake (serverInfo.version) must report THIS
    package's version, not fastmcp's — registries, Smithery scans and support
    triage all read it. Guards the FastMCP(version=...) wiring."""
    from nimbus_mcp import __version__

    server = build_server()  # isolated env → NullClient; version is static
    assert server.version == __version__
