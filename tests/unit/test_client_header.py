"""Every backend request from nimbus-mcp (local or gateway) self-identifies."""

from pathlib import Path

from nimbus_mcp import __version__
from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import HeaderCredentialClient


def _cfg(tmp_path: Path, **kw) -> McpConfig:
    return McpConfig(api_url="http://backend.test", mcp_key=None, export_dir=tmp_path, **kw)


def test_local_client_sends_x_nimbus_client(tmp_path: Path) -> None:
    c = NimbusClient(_cfg(tmp_path))
    try:
        assert c._http.headers["X-Nimbus-Client"] == f"nimbus-mcp/{__version__}"
    finally:
        c.close()


def test_gateway_header_credential_client_sends_x_nimbus_client(tmp_path: Path) -> None:
    hc = HeaderCredentialClient(
        _cfg(tmp_path), headers_getter=lambda: {"x-nimbus-token": "nimb_test"}
    )
    c = hc._client_for("nimb_test")
    try:
        assert c._http.headers["X-Nimbus-Client"] == f"nimbus-mcp/{__version__}"
        assert c._http.headers["Authorization"] == "Bearer nimb_test"
    finally:
        c.close()
