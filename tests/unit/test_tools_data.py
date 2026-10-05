from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import data


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    data.register(server, client)
    return server


async def test_upload_data_posts_multipart(tmp_path):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["ct"] = req.headers.get("content-type", "")
        seen["body"] = req.read()
        return httpx.Response(
            200,
            json={
                "ok": True,
                "file": {
                    "path": "user_x/rec.csv",
                    "filename": "rec.csv",
                    "originalFilename": "rec.csv",
                    "size": 4,
                    "format": "csv",
                },
                "metadata": {"channels": 4},
            },
        )

    src = tmp_path / "rec.csv"
    src.write_text("a,b\n1,2\n")
    async with Client(make_server(handler)) as c:
        result = await c.call_tool("data.upload", {"file_path": str(src), "dataset_name": "my-rec"})
    assert seen["path"] == "/api/upload"
    assert "multipart/form-data" in seen["ct"]
    assert b'name="dataset_name"' in seen["body"]
    assert b'name="file"' in seen["body"]
    assert b"a,b\n1,2\n" in seen["body"]
    assert result.data["file"]["path"] == "user_x/rec.csv"
    assert "custom_data" in result.data["usage"]


async def test_upload_data_missing_file_errors(tmp_path):
    async with Client(make_server(lambda r: httpx.Response(200))) as c:
        with pytest.raises(Exception, match="File not found"):
            await c.call_tool("data.upload", {"file_path": str(tmp_path / "missing.csv")})


async def test_upload_data_sends_config_form_field_for_sampling_rate(tmp_path):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True, "file": {"path": "x/r.csv"}})

    src = tmp_path / "rec.csv"
    src.write_text("a,b\n1,2\n")
    async with Client(make_server(handler)) as c:
        await c.call_tool(
            "data.upload",
            {"file_path": str(src), "sampling_rate": 512.5, "format": "csv"},
        )
    # The config form field carries a JSON body with the backend's camelCase
    # custom_data schema keys (uploads.py validates against that schema and
    # rejects unknown keys with 400 — "sampling_rate" would never pass).
    assert b'name="config"' in seen["body"]
    assert b'"samplingRate": 512.5' in seen["body"]
    assert b'"format": "csv"' in seen["body"]
    assert b'"sampling_rate"' not in seen["body"]


async def test_upload_data_omits_null_config_keys(tmp_path):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True, "file": {"path": "x/r.csv"}})

    src = tmp_path / "rec.csv"
    src.write_text("a,b\n1,2\n")
    async with Client(make_server(handler)) as c:
        await c.call_tool("data.upload", {"file_path": str(src), "format": "edf"})
    assert b'name="config"' in seen["body"]
    assert b'"format": "edf"' in seen["body"]
    assert b"samplingRate" not in seen["body"]


async def test_upload_data_without_config_sends_no_config_field(tmp_path):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True, "file": {"path": "x/r.csv"}})

    src = tmp_path / "rec.csv"
    src.write_text("a,b\n1,2\n")
    async with Client(make_server(handler)) as c:
        await c.call_tool("data.upload", {"file_path": str(src)})
    assert b'name="config"' not in seen["body"]


async def test_upload_data_rejects_non_positive_sampling_rate_before_http(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    src = tmp_path / "rec.csv"
    src.write_text("a,b\n1,2\n")
    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="sampling_rate"):
            await c.call_tool("data.upload", {"file_path": str(src), "sampling_rate": 0.0})
