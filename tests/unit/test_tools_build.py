from pathlib import Path

import httpx
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import build

TRAIN_GRAPH = {
    "nodes": [{"id": "d", "type": "public_data", "config": {}}],
    "connections": [],
}


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    build.register(server, client)
    return server


async def test_validate_pipeline_posts_train_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/validate-pipeline":
            seen["body"] = req.read()
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "valid": False,
                    "validationErrors": [{"message": "no model"}],
                    "warnings": [],
                },
            )
        raise AssertionError(f"unexpected path {req.url.path}")

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("pipeline.validate", {"train_graph": TRAIN_GRAPH})
    assert b'"train"' in seen["body"]
    assert result.data["valid"] is False
    assert result.data["validationErrors"][0]["message"] == "no model"


async def test_validate_node_config_camel_case_body():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True, "valid": True, "errors": [], "warnings": []})

    async with Client(make_server(handler)) as c:
        await c.call_tool(
            "pipeline.validate_node",
            {"node_type": "bandpass_filter", "config": {"lowCut": 8}},
        )
    assert b'"nodeType"' in seen["body"] and b'"lowCut"' in seen["body"]
