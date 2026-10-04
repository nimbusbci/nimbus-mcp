import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import artifacts

ZIP_BYTES = b"PK\x03\x04fakezip"


def make_server(handler, tmp_path) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=tmp_path),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    artifacts.register(server, client)
    return server


async def test_list_artifacts(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/executions/exec_1/artifacts"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "executionId": "exec_1",
                "artifacts": [
                    {"name": "nimbus_lda.pkl", "size": 1234, "availableLocally": True}
                ],
            },
        )

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool("list_artifacts", {"execution_id": "exec_1"})
    assert result.data["artifacts"][0]["name"] == "nimbus_lda.pkl"


async def test_download_artifact_saves_file(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/executions/exec_1/artifacts/nimbus_lda.pkl"
        return httpx.Response(200, content=b"pickle-bytes")

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "download_artifact", {"execution_id": "exec_1", "artifact_name": "nimbus_lda.pkl"}
        )
    saved = tmp_path / "executions" / "exec_1" / "nimbus_lda.pkl"
    assert saved.read_bytes() == b"pickle-bytes"
    assert result.data["path"] == str(saved)
    assert result.data["size"] == len(b"pickle-bytes")


async def test_download_artifact_rejects_path_traversal(tmp_path):
    async with Client(make_server(lambda r: httpx.Response(200), tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid artifact name"):
            await c.call_tool(
                "download_artifact",
                {"execution_id": "exec_1", "artifact_name": "../evil.pkl"},
            )


async def test_download_artifact_rejects_bad_execution_id(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler, tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid execution id"):
            await c.call_tool(
                "download_artifact",
                {"execution_id": "../evil", "artifact_name": "nimbus_lda.pkl"},
            )
    assert not (tmp_path / "evil").exists()


async def test_list_artifacts_rejects_bad_execution_id(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler, tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid execution id"):
            await c.call_tool("list_artifacts", {"execution_id": "../evil"})


async def test_download_artifact_rejects_dot_artifact_name(tmp_path):
    async with Client(make_server(lambda r: httpx.Response(200), tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid artifact name"):
            await c.call_tool(
                "download_artifact",
                {"execution_id": "exec_1", "artifact_name": "."},
            )


async def test_export_python_saves_zip(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/export/python"
        return httpx.Response(200, content=ZIP_BYTES)

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "export_python",
            {"train_graph": {"nodes": [], "connections": []}},
        )
    path = tmp_path / result.data["path"].split(str(tmp_path) + "/")[-1]
    assert path.read_bytes() == ZIP_BYTES
    assert result.data["size"] == len(ZIP_BYTES)


async def test_export_python_no_silent_overwrite_on_collision(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ZIP_BYTES)

    async with Client(make_server(handler, tmp_path)) as c:
        await c.call_tool("export_python", {"train_graph": {"nodes": [], "connections": []}})
        await c.call_tool("export_python", {"train_graph": {"nodes": [], "connections": []}})
    # Distinct stamps or the -1 collision suffix: either way two zips survive.
    assert len(list((tmp_path / "exports").glob("*.zip"))) == 2
