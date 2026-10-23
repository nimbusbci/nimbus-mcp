"""L1: local mode with no desktop app → actionable install/hosted guidance."""

from __future__ import annotations

import httpx
import pytest

from nimbus_mcp.client import McpToolError, NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.setup_mode import (
    DESKTOP_DOWNLOAD_MACOS,
    DESKTOP_DOWNLOAD_WINDOWS,
    HOSTED_GATEWAY_URL,
    LOCAL_BACKEND_UNREACHABLE,
    SetupRequired,
    local_mode_guidance,
)

URL = "http://127.0.0.1:8080"


def _down(req: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=req)


def _client(*, key: str = "", token: str = "", meta: dict | None = None, handler=_down):
    cfg = McpConfig(
        api_url=URL,
        mcp_key=key,
        nimbus_token=token or None,  # type: ignore[arg-type]
        export_dir=None,
        credential_meta=meta or {},
    )
    return NimbusClient(cfg, transport=httpx.MockTransport(handler))


def _assert_unreachable_payload(g: dict) -> None:
    assert g["ok"] is False and g["setupRequired"] is True
    assert g["reason"] == LOCAL_BACKEND_UNREACHABLE
    for needle in (
        DESKTOP_DOWNLOAD_MACOS,
        DESKTOP_DOWNLOAD_WINDOWS,
        HOSTED_GATEWAY_URL,
        "nimbus-mcp login",
        "NIMBUS_MCP_KEY",
    ):
        assert needle in g["message"], needle
    actions = [o["action"] for o in g["options"]]
    assert actions == [
        "install the Nimbus Studio desktop app",
        "use the hosted gateway",
        "nimbus-mcp login",
        "set NIMBUS_MCP_KEY / NIMBUS_MCP_KEY_FILE",
    ]
    assert g["options"][0]["url_macos"] == DESKTOP_DOWNLOAD_MACOS
    assert g["options"][0]["url_windows"] == DESKTOP_DOWNLOAD_WINDOWS
    assert g["options"][1]["url"] == HOSTED_GATEWAY_URL


def test_unreachable_guidance_has_links_hosted_login_and_reason():
    g = local_mode_guidance(None, URL)
    _assert_unreachable_payload(g)
    assert f"nothing answered at {URL}" in g["message"]


def test_rejected_variant_has_no_download_links():
    g = local_mode_guidance(403)
    assert "HTTP 403" in g["message"]
    assert DESKTOP_DOWNLOAD_MACOS not in g["message"]
    assert "reason" not in g


def test_anonymous_connect_error_is_setup_required_with_reason():
    with pytest.raises(SetupRequired) as exc:
        _client().get("/api/templates")
    _assert_unreachable_payload(exc.value.guidance)


def test_anonymous_ensure_ready_uses_same_payload(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(SetupRequired) as exc:
        _client().ensure_ready()
    _assert_unreachable_payload(exc.value.guidance)


def test_stale_desktop_key_says_app_not_running():
    c = _client(key="desk", meta={"source": "desktop", "path": "/x/mcp-key.json"})
    with pytest.raises(SetupRequired) as exc:
        c.get("/api/templates")
    g = exc.value.guidance
    _assert_unreachable_payload(g)
    assert g["message"].startswith("The Nimbus Studio desktop app isn't running")
    assert g["desktopKeyFound"] is True
    # Still a status-less McpToolError → experiment.run stays fatal pre-id (N3).
    assert isinstance(exc.value, McpToolError)
    assert getattr(exc.value, "status_code", None) is None


def test_dev_env_key_keeps_plain_backend_down_error():
    c = _client(key="dev", meta={"source": "env"})
    with pytest.raises(McpToolError) as exc:
        c.get("/api/templates")
    assert not isinstance(exc.value, SetupRequired)
    assert "Cannot reach the Nimbus backend" in str(exc.value)


def test_hosted_connect_error_unchanged():
    c = _client(token="nimb_x", meta={"source": "env"})
    with pytest.raises(McpToolError) as exc:
        c.get("/api/templates")
    assert not isinstance(exc.value, SetupRequired)
    assert "Cannot reach hosted Nimbus" in str(exc.value)
