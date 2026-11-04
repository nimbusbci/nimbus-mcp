"""Anonymous 401 from the HOSTED API asks for a login, not a local identity.

Regression for the 0.14.20 post-release check (Nimbus MCP Test): a stdio
config with NIMBUS_API_URL=https://nimbus-studio.fly.dev and no credential
got "The local Nimbus backend rejected this request (HTTP 401)... No Nimbus
Studio login is needed for local work" from catalog.datasets.
"""

from __future__ import annotations

import httpx
import pytest
from fastmcp import Client

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.server import build_server
from nimbus_mcp.setup_mode import hosted_login_guidance, is_loopback_api_url


def _anon(tmp_path, api_url: str) -> NimbusClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(401, json={"detail": "Not authenticated"})

    cfg = McpConfig(
        api_url=api_url, mcp_key="", export_dir=tmp_path, nimbus_token="", credential_meta={}
    )
    return NimbusClient(cfg, transport=httpx.MockTransport(handler))


async def _datasets(client: NimbusClient) -> dict:
    async with Client(build_server(client)) as c:
        result = await c.call_tool("catalog.datasets", {}, raise_on_error=False)
        return result.structured_content or {}


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:8080", True),
        ("http://localhost:8080", True),
        ("http://[::1]:8080", True),
        ("https://nimbus-studio.fly.dev", False),
        ("https://api.example.com", False),
        ("", True),
    ],
)
def test_is_loopback_api_url(url, expected):
    assert is_loopback_api_url(url) is expected


async def test_hosted_anonymous_401_returns_login_required(tmp_path):
    data = await _datasets(_anon(tmp_path, "https://nimbus-studio.fly.dev"))
    blob = str(data)
    assert data.get("loginRequired") is True
    assert data.get("reason") == "hosted_login_required"
    assert "nimbus-mcp login" in data["message"]
    assert "NIMBUS_TOKEN" in data["message"]
    assert "https://nimbus-mcp.fly.dev/mcp" in data["message"]
    assert "local Nimbus backend" not in blob
    assert "No Nimbus Studio login is needed" not in blob
    assert data["options"][0]["action"] == "nimbus-mcp login"


async def test_local_anonymous_401_keeps_local_guidance(tmp_path):
    data = await _datasets(_anon(tmp_path, "http://127.0.0.1:8080"))
    assert "The local Nimbus backend rejected this request (HTTP 401)" in data["message"]
    assert not data.get("loginRequired")
    assert data["options"][-1]["action"] == "nimbus-mcp login"


def test_hosted_login_guidance_shape():
    g = hosted_login_guidance(401, "https://nimbus-studio.fly.dev")
    assert g["ok"] is False and g["setupRequired"] is True and g["loginRequired"] is True
    assert g["httpStatus"] == 401
    assert g["message"].startswith("Login required: the hosted Nimbus API")
