import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import _hosted_files, artifacts

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
                "artifacts": [{"name": "nimbus_lda.pkl", "size": 1234, "availableLocally": True}],
            },
        )

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool("execution.artifacts", {"execution_id": "exec_1"})
    assert result.data["artifacts"][0]["name"] == "nimbus_lda.pkl"


async def test_download_artifact_saves_file(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/executions/exec_1/artifacts/nimbus_lda.pkl"
        return httpx.Response(200, content=b"pickle-bytes")

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "execution.download_artifact",
            {"execution_id": "exec_1", "artifact_name": "nimbus_lda.pkl"},
        )
    saved = tmp_path / "executions" / "exec_1" / "nimbus_lda.pkl"
    assert saved.read_bytes() == b"pickle-bytes"
    assert result.data["path"] == str(saved)
    assert result.data["size"] == len(b"pickle-bytes")


async def test_download_artifact_rejects_path_traversal(tmp_path):
    async with Client(make_server(lambda r: httpx.Response(200), tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid artifact name"):
            await c.call_tool(
                "execution.download_artifact",
                {"execution_id": "exec_1", "artifact_name": "../evil.pkl"},
            )


async def test_download_artifact_rejects_bad_execution_id(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler, tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid execution id"):
            await c.call_tool(
                "execution.download_artifact",
                {"execution_id": "../evil", "artifact_name": "nimbus_lda.pkl"},
            )
    assert not (tmp_path / "evil").exists()


async def test_list_artifacts_rejects_bad_execution_id(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler, tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid execution id"):
            await c.call_tool("execution.artifacts", {"execution_id": "../evil"})


async def test_download_artifact_rejects_dot_artifact_name(tmp_path):
    async with Client(make_server(lambda r: httpx.Response(200), tmp_path)) as c:
        with pytest.raises(Exception, match="Invalid artifact name"):
            await c.call_tool(
                "execution.download_artifact",
                {"execution_id": "exec_1", "artifact_name": "."},
            )


async def test_export_python_saves_zip(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/export/python"
        return httpx.Response(200, content=ZIP_BYTES)

    async with Client(make_server(handler, tmp_path)) as c:
        result = await c.call_tool(
            "pipeline.export",
            {"train_graph": {"nodes": [], "connections": []}},
        )
    path = tmp_path / result.data["path"].split(str(tmp_path) + "/")[-1]
    assert path.read_bytes() == ZIP_BYTES
    assert result.data["size"] == len(ZIP_BYTES)


async def test_export_python_no_silent_overwrite_on_collision(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=ZIP_BYTES)

    async with Client(make_server(handler, tmp_path)) as c:
        await c.call_tool("pipeline.export", {"train_graph": {"nodes": [], "connections": []}})
        await c.call_tool("pipeline.export", {"train_graph": {"nodes": [], "connections": []}})
    # Distinct stamps or the -1 collision suffix: either way two zips survive.
    assert len(list((tmp_path / "exports").glob("*.zip"))) == 2


def _zip_with(files: dict[str, str]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


async def test_export_python_surfaces_export_notes_locally(tmp_path):
    data = _zip_with({"pipeline.py": "x", "EXPORT_NOTES.md": "skipped interactive_application"})

    async with Client(make_server(lambda req: httpx.Response(200, content=data), tmp_path)) as c:
        result = await c.call_tool(
            "pipeline.export", {"train_graph": {"nodes": [], "connections": []}}
        )
    assert "path" in result.data
    assert "interactive_application" in result.data["exportNotes"]


async def test_export_python_hosted_returns_inline_base64(tmp_path, monkeypatch):
    import base64

    data = _zip_with({"pipeline.py": "x", "EXPORT_NOTES.md": "skipped interactive_application"})
    monkeypatch.setattr(artifacts, "_is_hosted_gateway", lambda _c: True)
    async with Client(make_server(lambda req: httpx.Response(200, content=data), tmp_path)) as c:
        result = await c.call_tool(
            "pipeline.export", {"train_graph": {"nodes": [], "connections": []}}
        )
    assert "path" not in result.data
    assert base64.b64decode(result.data["contentBase64"]) == data
    assert result.data["exportNotes"]
    assert not (tmp_path / "exports").exists()  # nothing written on the gateway


async def test_export_python_hosted_over_cap_has_no_inline_content(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "_is_hosted_gateway", lambda _c: True)
    monkeypatch.setattr(_hosted_files, "HOSTED_INLINE_MAX_BYTES", 4)
    async with Client(
        make_server(lambda req: httpx.Response(200, content=ZIP_BYTES), tmp_path)
    ) as c:
        result = await c.call_tool(
            "pipeline.export", {"train_graph": {"nodes": [], "connections": []}}
        )
    assert "contentBase64" not in result.data and "Export" in result.data["note"]


@pytest.mark.parametrize(("raw", "inline"), [(b"abc", True), (b"abcd", False)])
async def test_export_inline_cap_is_on_encoded_size(tmp_path, monkeypatch, raw, inline):
    # 3 raw bytes -> 4 base64 chars (fits a cap of 4); 4 raw bytes -> 8 (does not).
    monkeypatch.setattr(artifacts, "_is_hosted_gateway", lambda _c: True)
    monkeypatch.setattr(_hosted_files, "HOSTED_INLINE_MAX_BYTES", 4)
    async with Client(make_server(lambda req: httpx.Response(200, content=raw), tmp_path)) as c:
        result = await c.call_tool(
            "pipeline.export", {"train_graph": {"nodes": [], "connections": []}}
        )
    assert ("contentBase64" in result.data) is inline


def test_hosted_detection_uses_the_real_class():
    from nimbus_mcp.hosted import HeaderCredentialClient

    fake = HeaderCredentialClient.__new__(HeaderCredentialClient)
    assert artifacts._is_hosted_gateway(fake)
    assert not artifacts._is_hosted_gateway(object())


async def test_download_artifact_hosted_returns_inline_and_writes_nothing(tmp_path, monkeypatch):
    """E2E round 2: hosted returned a gateway-local /tmp/nimbus-exports path."""
    import base64

    monkeypatch.setattr(artifacts, "_is_hosted_gateway", lambda _c: True)
    async with Client(
        make_server(lambda req: httpx.Response(200, content=b"model-bytes"), tmp_path)
    ) as c:
        result = await c.call_tool(
            "execution.download_artifact",
            {"execution_id": "exec_1", "artifact_name": "nimbus_lda.pkl"},
        )
    assert "path" not in result.data
    assert base64.b64decode(result.data["contentBase64"]) == b"model-bytes"
    assert result.data["filename"] == "nimbus_lda.pkl"
    assert not (tmp_path / "executions").exists()
