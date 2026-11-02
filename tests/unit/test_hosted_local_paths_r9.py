"""E2E round 9 N1: the hosted gateway never resolves caller paths on its own disk."""

from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.config import McpConfig
from nimbus_mcp.errors import McpToolError
from nimbus_mcp.hosted import HeaderCredentialClient
from nimbus_mcp.tools import data, inspect
from nimbus_mcp.tools._hosted_files import LOCAL_PATH_UNSUPPORTED

CFG = McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx"))


def _gateway(seen: list) -> FastMCP:
    def factory(cfg):  # any backend call would be recorded here
        seen.append(cfg)
        raise AssertionError("gateway must not build a backend client for a local path")

    proxy = HeaderCredentialClient(CFG, client_factory=factory, headers_getter=lambda: {})
    server = FastMCP("t")
    data.register(server, proxy)
    inspect.register(server, proxy)
    return server


async def _text(server: FastMCP, tool: str, args: dict) -> str:
    async with Client(server) as c:
        result = await c.call_tool(tool, args, raise_on_error=False)
    assert result.is_error
    return result.content[0].text


@pytest.mark.parametrize(
    "file_path", ["/etc/passwd", "/nonexistent/a.csv", "/tmp/a.csv", "~/x.edf", "rel/a.edf"]
)
async def test_gateway_upload_refuses_every_local_path_identically(file_path, monkeypatch):
    # Even if the path exists on the server, nothing may be stat'ed or read.
    monkeypatch.setattr(Path, "is_file", lambda self: pytest.fail("stat on gateway"))
    seen: list = []
    text = await _text(
        _gateway(seen), "data.upload", {"file_path": file_path, "sampling_rate": 250.0}
    )
    assert LOCAL_PATH_UNSUPPORTED in text
    assert "correlationId" in text
    # No existence oracle: the message never differs by path and never echoes it.
    assert file_path not in text
    assert "File not found" not in text and "Unsupported format" not in text
    assert seen == []


async def test_gateway_upload_messages_are_identical_for_existing_and_missing_paths():
    a = await _text(_gateway([]), "data.upload", {"file_path": "/etc/passwd"})
    b = await _text(_gateway([]), "data.upload", {"file_path": "/nonexistent/a.csv"})
    strip = lambda t: t.split("correlationId")[0]  # noqa: E731
    assert strip(a) == strip(b)


async def test_gateway_inspect_file_absolute_path_is_tool_error():
    seen: list = []
    text = await _text(_gateway(seen), "data.inspect_file", {"path": "/etc/passwd"})
    assert LOCAL_PATH_UNSUPPORTED in text
    assert "data.upload" in text  # still carries the upload-first guidance
    assert seen == []


def test_gateway_client_post_file_is_refused(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("1,2\n")
    proxy = HeaderCredentialClient(
        CFG, client_factory=lambda c: pytest.fail("no client"), headers_getter=lambda: {}
    )
    with pytest.raises(McpToolError) as exc:
        proxy.post_file("/api/upload", f, {})
    assert exc.value.code == LOCAL_PATH_UNSUPPORTED


async def test_local_upload_missing_file_has_specific_code(tmp_path):
    from nimbus_mcp.client import NimbusClient

    client = NimbusClient(CFG, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    server = FastMCP("t")
    data.register(server, client)
    text = await _text(server, "data.upload", {"file_path": str(tmp_path / "missing.edf")})
    assert "nimbus.mcp.file_not_found" in text
    assert "File not found" in text


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("plots.dataset", {"source": "local", "kind": "psd", "path": "/etc/passwd"}),
        ("plots.erp", {"source": "local", "path": "/etc/passwd"}),
        ("foundation.compare", {"source": "local", "path": "/etc/passwd"}),
    ],
)
async def test_gateway_refuses_local_source_figures(tool, args):
    """source=local on the gateway is refused before any backend call (r9 review)."""
    from nimbus_mcp.tools import foundation, plots

    seen: list = []
    proxy = HeaderCredentialClient(
        CFG, client_factory=lambda c: seen.append(c), headers_getter=lambda: {}
    )
    server = FastMCP("t")
    plots.register(server, proxy)
    foundation.register(server, proxy)
    text = await _text(server, tool, args)
    assert LOCAL_PATH_UNSUPPORTED in text
    assert "/etc/passwd" not in text
    assert seen == []
