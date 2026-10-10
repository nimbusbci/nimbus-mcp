"""Unit tests for the plots.* and stats.* MCP tools (v0.12).

Asserts the image-content contract: every plots tool returns an
``ImageContent`` block carrying the backend's PNG plus a JSON sidecar with
the figure meta, and ``save=true`` writes the PNG into
NIMBUS_EXPORT_DIR/plots/.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ImageContent, TextContent

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import plots, stats

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1200).decode("ascii")


def _plot_payload(meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, "image": _PNG, "mimeType": "image/png", "title": "t", "meta": meta or {}}


def make_server(handler, tmp_path, module: Any = plots) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=tmp_path),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    module.register(server, client)
    return server


def _image_and_info(result: Any) -> tuple[ImageContent, dict[str, Any]]:
    images = [b for b in result.content if isinstance(b, ImageContent)]
    texts = [b for b in result.content if isinstance(b, TextContent)]
    assert images, "no ImageContent block in tool result"
    assert texts, "no JSON sidecar in tool result"
    info = json.loads(texts[0].text)
    return images[0], info


async def test_confusion_returns_image_and_meta(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/plots/confusion"
        assert json.loads(req.content) == {"executionId": "exec_9", "matrix": "eval"}
        return httpx.Response(200, json=_plot_payload({"nClasses": 2, "totalTrials": 100}))

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool("plots.confusion", {"execution_id": "exec_9"})
    image, info = _image_and_info(result)
    assert image.mimeType == "image/png"
    assert base64.b64decode(image.data)[:8] == b"\x89PNG\r\n\x1a\n"
    assert info["meta"]["nClasses"] == 2


async def test_confusion_saves_png(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_plot_payload())

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "plots.confusion", {"execution_id": "exec_9", "save": True}
        )
    _, info = _image_and_info(result)
    saved = tmp_path / "plots" / "exec_9-eval-confusion.png"
    assert saved.exists()
    assert info["savedPath"] == str(saved)


async def test_confusion_rejects_bad_matrix(tmp_path):
    async with Client(make_server(lambda req: httpx.Response(500), tmp_path)) as c:
        with pytest.raises(ToolError, match="eval.*train"):
            await c.call_tool(
                "plots.confusion", {"execution_id": "exec_9", "matrix": "bogus"}
            )


async def test_dataset_forwards_source_params(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/plots/dataset"
        body = json.loads(req.content)
        assert body["kind"] == "topomap"
        assert body["source"] == "dataset"
        assert body["dataset"] == "BNCI2014_001"
        assert body["subject"] == "S01"
        assert body["band"] == "alpha"
        return httpx.Response(200, json=_plot_payload({"layout": "standard_1005"}))

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "plots.dataset",
            {
                "kind": "topomap",
                "source": "dataset",
                "dataset": "BNCI2014_001",
                "subject": "S01",
            },
        )
    _, info = _image_and_info(result)
    assert info["meta"]["layout"] == "standard_1005"


async def test_dataset_rejects_bad_kind_and_band(tmp_path):
    async with Client(make_server(lambda req: httpx.Response(500), tmp_path)) as c:
        with pytest.raises(ToolError, match="kind must be one of"):
            await c.call_tool("plots.dataset", {"kind": "heatmap", "source": "dataset"})
        with pytest.raises(ToolError, match="band must be one of"):
            await c.call_tool(
                "plots.dataset",
                {"kind": "topomap", "source": "dataset", "band": "gamma2"},
            )


async def test_leaderboard_track_and_erp_passthrough(tmp_path):
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json=_plot_payload())

    async with Client(make_server(handler, tmp_path)) as c:
        await c.call_tool(
            "plots.leaderboard",
            {"dataset_id": "BNCI2014_001", "track": "crossSubject"},
        )
        await c.call_tool(
            "plots.erp",
            {"source": "dataset", "dataset": "BNCI2014_001", "subject": "S03"},
        )
    paths = [p for p, _ in seen]
    assert paths == ["/api/plots/leaderboard", "/api/plots/erp"]
    assert seen[0][1] == {"datasetId": "BNCI2014_001", "track": "crossSubject"}
    assert seen[1][1]["maxChannels"] == 16


async def test_stats_grouped_forwards_ids_and_chance(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/stats/grouped"
        assert json.loads(req.content) == {
            "executionIds": ["a", "b"],
            "chance": 0.25,
        }
        return httpx.Response(
            200,
            json={"ok": True, "executions": [], "overall": {"nGroups": 6}},
        )

    async with Client(make_server(handler, tmp_path, module=stats)) as c:
        result = await c.call_tool(
            "stats.grouped", {"execution_ids": ["a", "b"], "chance": 0.25}
        )
    assert result.data["overall"]["nGroups"] == 6


async def test_stats_grouped_requires_ids(tmp_path):
    async with Client(make_server(lambda req: httpx.Response(500), tmp_path, stats)) as c:
        with pytest.raises(ToolError, match="at least one id"):
            await c.call_tool("stats.grouped", {"execution_ids": []})
