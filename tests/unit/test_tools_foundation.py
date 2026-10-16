"""Unit tests for the foundation.* MCP tool (v0.13.1).

Mirrors the plots.* contract: foundation.compare returns an ImageContent
block carrying the backend's ranked-chart PNG plus a JSON sidecar with the
per-backbone results, and forwards the describe-style source params.
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
from nimbus_mcp.tools import foundation

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 1200).decode("ascii")


def _payload(meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": True, "image": _PNG, "mimeType": "image/png", "title": "t", "meta": meta or {}}


def make_server(handler, tmp_path) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=tmp_path),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    foundation.register(server, client)
    return server


def _image_and_info(result: Any) -> tuple[ImageContent, dict[str, Any]]:
    images = [b for b in result.content if isinstance(b, ImageContent)]
    assert images, "no ImageContent block in tool result"
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return images[0], structured
    texts = [b for b in result.content if isinstance(b, TextContent)]
    assert texts, "no JSON sidecar in tool result"
    return images[0], json.loads(texts[0].text)


async def test_compare_returns_image_and_results_meta(tmp_path):
    meta = {
        "best": {"name": "cbramod", "meanAccuracyPct": 78.9},
        "results": [{"backbone": "cbramod", "accuracyPct": 78.9}],
        "skipped": [{"name": "labram", "error": "window too long"}],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/foundation/compare"
        assert json.loads(req.content) == {
            "source": "dataset",
            "dataset": "BNCI2014_001",
            "subject": "S01",
            "mode": "all",
            "path": None,
        }
        return httpx.Response(200, json=_payload(meta))

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "foundation.compare",
            {"source": "dataset", "dataset": "BNCI2014_001", "subject": "S01"},
        )
    image, info = _image_and_info(result)
    assert image.mimeType == "image/png"
    assert base64.b64decode(image.data)[:8] == b"\x89PNG\r\n\x1a\n"
    assert info["meta"]["best"]["name"] == "cbramod"
    assert info["meta"]["results"][0]["backbone"] == "cbramod"
    assert info["meta"]["skipped"][0]["name"] == "labram"


async def test_compare_forwards_upload_and_local_sources(tmp_path):
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_payload())

    async with Client(make_server(handler, tmp_path)) as c:
        await c.call_tool("foundation.compare", {"source": "upload", "path": "u/1/eeg.set"})
        await c.call_tool(
            "foundation.compare", {"source": "local", "path": "/tmp/eeg.set", "mode": "training"}
        )
    assert seen[0] == {
        "source": "upload",
        "dataset": None,
        "subject": None,
        "mode": "all",
        "path": "u/1/eeg.set",
    }
    assert seen[1]["mode"] == "training"
    assert seen[1]["path"] == "/tmp/eeg.set"


async def test_compare_rejects_bad_source_without_calling_backend(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler, tmp_path)) as c:
        with pytest.raises(ToolError, match="source must be one of"):
            await c.call_tool("foundation.compare", {"source": "bogus"})


async def test_compare_is_read_only_and_documented(tmp_path):
    server = make_server(lambda req: httpx.Response(500), tmp_path)
    tool = await server.get_tool("foundation.compare")
    assert tool is not None
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    # Agents pick tools from descriptions: the skip-not-silently contract and
    # the ranked-chart promise must be visible there.
    assert "which wins" in tool.description
    assert "noted, not silently skipped" in tool.description
    assert "CI whiskers" in tool.description
