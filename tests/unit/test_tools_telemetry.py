from pathlib import Path

import httpx
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import telemetry


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    telemetry.register(server, client)
    return server


async def test_get_live_session_passes_params_and_payload():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/hardware/session-live"
        assert req.url.params["sessionId"] == "s1"
        assert req.url.params["window"] == "10"
        return httpx.Response(200, json={
            "ok": True, "sessionId": "s1", "status": "active",
            "chunksProcessed": 5,
            "latestPrediction": {"class": 1, "confidence": 0.9},
            "recentPredictions": [{"index": 4, "class": 1, "confidence": 0.9}],
            "quality": {"snrDb": 6.0},
            "stats": {"n": 5, "confidenceMean": 0.8},
        })

    async with Client(make_server(handler)) as c:
        r = await c.call_tool("get_live_session", {"session_id": "s1", "window": 10})
    assert r.data["latestPrediction"]["confidence"] == 0.9
    assert r.data["stats"]["n"] == 5


async def test_get_live_session_rejects_bad_session_id():
    async with Client(make_server(lambda r: httpx.Response(200))) as c:
        try:
            await c.call_tool("get_live_session", {"session_id": "../evil"})
            raised = False
        except Exception as e:
            raised = "Invalid session id" in str(e)
        assert raised
