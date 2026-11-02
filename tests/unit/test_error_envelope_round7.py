"""Round 7 (2026-10-30): every tool error carries a code + correlationId."""

from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.errors import McpToolError
from nimbus_mcp.server import ErrorEnvelopeMiddleware
from nimbus_mcp.tools import campaign, plots, stats


def _server(handler=lambda req: httpx.Response(500)) -> FastMCP:
    cfg = McpConfig(api_url="http://test", mcp_key="k", export_dir=Path("/tmp/nx"))
    client = NimbusClient(cfg, transport=httpx.MockTransport(handler))
    server = FastMCP("t")
    server.add_middleware(ErrorEnvelopeMiddleware())
    plots.register(server, client)
    stats.register(server, client)

    @server.tool(name="bare.raise")
    def bare() -> dict:
        raise McpToolError("something odd")

    @server.tool(name="replaced.raise")
    def replaced() -> dict:
        try:
            raise McpToolError("swallowed")
        except McpToolError:
            raise ToolError("deliberate message") from None

    return server


async def test_bad_plot_kind_is_coded():
    async with Client(_server()) as c:
        with pytest.raises(Exception) as ei:
            await c.call_tool("plots.dataset", {"kind": "nope", "source": "dataset"})
    assert "nimbus.mcp.invalid_argument" in str(ei.value)
    assert "correlationId" in str(ei.value)


async def test_empty_stats_grouped_is_coded():
    async with Client(_server()) as c:
        with pytest.raises(Exception) as ei:
            await c.call_tool("stats.grouped", {"execution_ids": []})
    assert "nimbus.mcp.invalid_argument" in str(ei.value)
    assert "correlationId" in str(ei.value)


async def test_middleware_tags_bare_tool_error():
    async with Client(_server()) as c:
        with pytest.raises(Exception) as ei:
            await c.call_tool("bare.raise", {})
    msg = str(ei.value)
    assert "something odd" in msg
    assert "nimbus.mcp.tool_error" in msg
    assert "correlationId" in msg


def test_pooling_label_flags_mixed_kinds():
    label = campaign._pooling_label([{"kind": "per_subject", "n": 3}, {"kind": "loso", "n": 1}])
    assert label["mixedEvaluationKinds"] is True
    assert "loso" in label["poolingNote"]
    single = campaign._pooling_label([{"kind": "loso", "n": 2}])
    assert single == {"pooledEvaluationKinds": ["loso"], "mixedEvaluationKinds": False}


async def test_middleware_keeps_deliberately_replaced_error():
    async with Client(_server()) as c:
        with pytest.raises(Exception) as ei:
            await c.call_tool("replaced.raise", {})
    assert "deliberate message" in str(ei.value)
    assert "swallowed" not in str(ei.value)
