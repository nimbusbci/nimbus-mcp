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
            {
                "node_type": "bandpass_filter",
                "config": {"lowCut": 8},
                "dataset": "BNCI2014_001",
            },
        )
    assert b'"nodeType"' in seen["body"] and b'"lowCut"' in seen["body"]
    assert b'"dataset"' in seen["body"] and b"BNCI2014_001" in seen["body"]


async def test_validate_node_unknown_type_is_tool_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "ok": True,
                "valid": False,
                "errors": ["Unknown node type: nope"],
                "warnings": [],
            },
        )

    # E2E r8 B5: a tool error (isError), like catalog.node_schema's not-found.
    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "pipeline.validate_node", {"node_type": "nope", "config": {}}, raise_on_error=False
        )
    assert result.is_error
    text = result.content[0].text
    assert "nimbus.catalog.node_type_not_found" in text
    assert "correlationId" in text


async def test_validate_node_value_error_has_no_code():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": True, "valid": False, "errors": ["low > high"], "warnings": []}
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "pipeline.validate_node", {"node_type": "bandpass_filter", "config": {}}
        )
    assert "code" not in result.data
