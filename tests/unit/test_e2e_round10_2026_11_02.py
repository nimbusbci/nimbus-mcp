"""E2E round 10 (2026-11-02): L1 protocol header, L2 timestamps, usability."""

from pathlib import Path

import httpx
from fastmcp import Client, FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import ProtocolVersionHeaderMiddleware, supported_protocol_versions
from nimbus_mcp.prompts import EXAMPLE_PROMPTS
from nimbus_mcp.tools import discovery, stats
from nimbus_mcp.tools._time import iso_utc
from nimbus_mcp.tools.run import _trim_summary


def _pv_app():
    async def ep(request):
        return JSONResponse({"ok": True})

    return ProtocolVersionHeaderMiddleware(Starlette(routes=[Route("/mcp", ep, methods=["POST"])]))


def test_unsupported_protocol_header_is_named_clearly():
    with TestClient(_pv_app()) as c:
        r = c.post(
            "/mcp",
            headers={"mcp-session-id": "s", "mcp-protocol-version": "1999-01-01"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["message"].startswith("Unsupported protocol version")
    assert "_meta" not in err["message"]
    assert err["data"]["requested"] == "1999-01-01"
    assert "2025-11-25" in err["data"]["supported"]


def test_supported_header_and_initialize_pass_through():
    with TestClient(_pv_app()) as c:
        ok = c.post("/mcp", headers={"mcp-session-id": "s", "mcp-protocol-version": "2025-11-25"})
        assert ok.status_code == 200
        # No session yet (initialize): the SDK negotiates the version itself.
        assert c.post("/mcp", headers={"mcp-protocol-version": "1999-01-01"}).status_code == 200
    assert "2024-11-05" in supported_protocol_versions()


def test_iso_utc_normalizes_naive_backend_timestamps():
    assert iso_utc("2026-11-02 12:56:53.00891") == "2026-11-02T12:56:53.008910Z"
    assert iso_utc("2026-11-02T09:56:53-03:00") == "2026-11-02T12:56:53Z"
    assert iso_utc("2026-11-02T12:56:53Z") == "2026-11-02T12:56:53Z"
    assert iso_utc("not a date") == "not a date"
    assert iso_utc(None) is None


def test_execution_summary_timestamps_are_iso_z():
    out = _trim_summary(
        {
            "executionId": "e",
            "startedAt": "2026-11-02 12:56:53.00891",
            "completedAt": "2026-11-02 12:57:01",
        }
    )
    assert out["startedAt"] == "2026-11-02T12:56:53.008910Z"
    assert out["completedAt"] == "2026-11-02T12:57:01Z"


TEMPLATES = {
    "ok": True,
    "templates": [
        {
            "id": "mi_bciiv2a_csp_lda",
            "name": "MI",
            "category": "Motor Imagery",
            "datasetId": "BNCI2014_001",
            "paradigm": "motor_imagery",
        },
        {
            "id": "p300_x",
            "name": "P300",
            "category": "P300",
            "datasetId": "BNCI2014_009",
            "paradigm": "p300",
        },
        {"id": "custom", "name": "Custom", "category": "General"},
    ],
}


def _server() -> FastMCP:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=TEMPLATES)

    cfg = McpConfig(api_url="http://test", mcp_key="k", export_dir=Path("/tmp/nx"))
    client = NimbusClient(cfg, transport=httpx.MockTransport(handler))
    server = FastMCP("nimbus-test")
    discovery.register(server, client)
    stats.register(server, client)
    return server


async def test_templates_expose_dataset_and_filter():
    async with Client(_server()) as c:
        all_ = (await c.call_tool("catalog.templates", {})).structured_content
        assert all_["count"] == 3
        assert all_["templates"][0]["datasetId"] == "BNCI2014_001"
        by_ds = (
            await c.call_tool("catalog.templates", {"dataset": "bnci2014_001"})
        ).structured_content
        assert [t["id"] for t in by_ds["templates"]] == ["mi_bciiv2a_csp_lda"]
        by_par = (await c.call_tool("catalog.templates", {"paradigm": "P300"})).structured_content
        assert [t["id"] for t in by_par["templates"]] == ["p300_x"]


async def test_stats_grouped_description_explains_loso_setup():
    async with Client(_server()) as c:
        tools = {t.name: t for t in await c.list_tools()}
    desc = tools["stats.grouped"].description or ""
    assert "evaluation_plan" in desc and "loso" in desc


def test_every_prompt_names_tools_and_eeg_prompt_explains_paths():
    for name, (_t, _d, text, tools) in EXAMPLE_PROMPTS.items():
        assert tools, name
        assert all(tool in text for tool in tools), name
    text = EXAMPLE_PROMPTS["inspect_my_eeg"][2]
    assert "hosted" in text and "Nimbus Studio" in text
