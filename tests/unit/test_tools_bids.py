"""Unit tests for the bids.* MCP tools (zip save + manifest summary)."""

from __future__ import annotations

import io
import json
import zipfile

import httpx
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import bids


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("root/dataset_description.json", "{}")
        zf.writestr(
            "root/nimbus_export_manifest.json",
            json.dumps(
                {
                    "tier": "derivative",
                    "subjects": ["S01"],
                    "dataFiles": [{"path": "a", "bytes": 1, "sha256": "x"}],
                }
            ),
        )
    return buf.getvalue()


def _zip_bytes_no_manifest() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("root/dataset_description.json", "{}")
    return buf.getvalue()


def make_server(handler, tmp_path) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=tmp_path),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    bids.register(server, client)
    return server


async def test_export_dataset_saves_zip_and_summarizes(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/bids/export"
        body = json.loads(req.content)
        assert body == {"source": {"kind": "auto", "id": "BNCI2014_001"}, "subjects": ["S01"]}
        return httpx.Response(200, content=_zip_bytes())

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "bids.export_dataset", {"dataset": "BNCI2014_001", "subjects": ["S01"]}
        )
    saved = tmp_path / "bids" / "BNCI2014_001-bids.zip"
    assert saved.read_bytes() == _zip_bytes()
    assert result.data["tier"] == "derivative"
    assert result.data["subjects"] == ["S01"]
    assert result.data["files"] == 1


async def test_export_execution_uses_execution_kind(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body == {"source": {"kind": "execution", "id": "exec_9"}}
        return httpx.Response(200, content=_zip_bytes())

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool("bids.export_execution", {"execution_id": "exec_9"})
    assert result.data["path"].endswith("exec_9-bids.zip")


async def test_export_dataset_accepts_slash_upload_id(tmp_path):
    """Upload ids ARE slash-containing relative paths (user_<sha>/<file>) — the
    id must reach the backend verbatim while the LOCAL save stem is sanitized."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["source"] == {"kind": "auto", "id": "user_abc123/data.edf"}  # verbatim
        return httpx.Response(200, content=_zip_bytes())

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool("bids.export_dataset", {"dataset": "user_abc123/data.edf"})
    saved = tmp_path / "bids" / "user_abc123_data.edf-bids.zip"  # "/" → "_"
    assert saved.read_bytes() == _zip_bytes()
    assert result.data["tier"] == "derivative"


async def test_export_dataset_rejects_empty_id(tmp_path):
    async with Client(make_server(lambda r: httpx.Response(200), tmp_path)) as c:
        try:
            await c.call_tool("bids.export_dataset", {"dataset": ""})
            raised = False
        except Exception:
            raised = True
        assert raised


async def test_export_dataset_missing_manifest_raises(tmp_path):
    """A zip without nimbus_export_manifest.json fails with a clear tool error."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_zip_bytes_no_manifest())

    async with Client(make_server(handler, tmp_path)) as c:
        try:
            await c.call_tool("bids.export_dataset", {"dataset": "BNCI2014_001"})
            raised = False
        except Exception as e:
            raised = True
            assert "manifest" in str(e)
        assert raised
