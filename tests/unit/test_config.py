import json
import re
import sys
from pathlib import Path

import pytest

from nimbus_mcp.client import McpToolError, NimbusClient
from nimbus_mcp.config import DEFAULT_API_URL, load_config
from nimbus_mcp.credentials import desktop_key_path, write_store

NIMBUS_ENV_VARS = (
    "NIMBUS_TOKEN",
    "NIMBUS_TOKEN_FILE",
    "NIMBUS_MCP_KEY",
    "NIMBUS_MCP_KEY_FILE",
    "NIMBUS_API_URL",
    "NIMBUS_CREDENTIALS_FILE",
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Keep resolution hermetic: no real ~/.nimbus store and no real desktop
    mcp-key.json (a dev machine running the desktop app HAS one at the macOS
    userData path, which would otherwise leak into every env={} test)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    for var in NIMBUS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_defaults():
    cfg = load_config(env={})
    assert cfg.api_url == DEFAULT_API_URL
    assert cfg.mcp_key == ""
    assert cfg.nimbus_token == ""
    assert cfg.export_dir == Path.home() / "nimbus-exports"


def test_env_overrides_and_trailing_slash_strip():
    cfg = load_config(
        env={
            "NIMBUS_API_URL": "http://localhost:9999/",
            "NIMBUS_MCP_KEY": "k1",
            "NIMBUS_EXPORT_DIR": "~/exports",
        }
    )
    assert cfg.api_url == "http://localhost:9999"
    assert cfg.mcp_key == "k1"
    assert cfg.export_dir == Path("~/exports").expanduser()


def _write_key_file(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "mcp-key.json"
    body = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(body, encoding="utf-8")
    return path


def test_key_file_only_resolution(tmp_path):
    key_file = _write_key_file(tmp_path, {"key": "from-file"})
    cfg = load_config(env={"NIMBUS_MCP_KEY_FILE": str(key_file)})
    assert cfg.mcp_key == "from-file"


def test_env_key_wins_over_key_file(tmp_path):
    key_file = _write_key_file(tmp_path, {"key": "from-file"})
    cfg = load_config(
        env={"NIMBUS_MCP_KEY": "from-env", "NIMBUS_MCP_KEY_FILE": str(key_file)}
    )
    assert cfg.mcp_key == "from-env"


def test_key_file_missing_raises(tmp_path):
    missing = tmp_path / "absent.json"
    with pytest.raises(McpToolError, match="Cannot read NIMBUS_MCP_KEY_FILE") as excinfo:
        load_config(env={"NIMBUS_MCP_KEY_FILE": str(missing)})
    assert str(missing) in str(excinfo.value)


def test_key_file_malformed_json_raises(tmp_path):
    key_file = _write_key_file(tmp_path, "{not json")
    with pytest.raises(McpToolError, match="Cannot read NIMBUS_MCP_KEY_FILE") as excinfo:
        load_config(env={"NIMBUS_MCP_KEY_FILE": str(key_file)})
    assert str(key_file) in str(excinfo.value)


@pytest.mark.parametrize("payload", ['"just-a-string"', "[1, 2]", '{"other": "x"}', '{"key": 42}'])
def test_key_file_wrong_shape_raises(tmp_path, payload):
    key_file = _write_key_file(tmp_path, payload)
    pattern = rf"Cannot read NIMBUS_MCP_KEY_FILE at {re.escape(str(key_file))}"
    with pytest.raises(McpToolError, match=pattern):
        load_config(env={"NIMBUS_MCP_KEY_FILE": str(key_file)})


def test_token_takes_precedence_over_key_and_key_file(tmp_path):
    key_file = _write_key_file(tmp_path, {"key": "from-file"})
    cfg = load_config(
        env={
            "NIMBUS_TOKEN": "nimbabc.def.ghi",
            "NIMBUS_MCP_KEY": "from-env",
            "NIMBUS_MCP_KEY_FILE": str(key_file),
        }
    )
    assert cfg.nimbus_token == "nimbabc.def.ghi"
    assert cfg.mcp_key == ""


def test_token_only_without_any_key():
    cfg = load_config(env={"NIMBUS_TOKEN": "nimbabc.def.ghi"})
    assert cfg.nimbus_token == "nimbabc.def.ghi"
    assert cfg.mcp_key == ""


def test_neither_token_nor_key_lists_both_options():
    cfg = load_config(env={})
    assert cfg.nimbus_token == "" and cfg.mcp_key == ""
    with pytest.raises(McpToolError) as excinfo:
        NimbusClient(cfg)
    msg = str(excinfo.value)
    assert "NIMBUS_TOKEN" in msg
    assert "NIMBUS_MCP_KEY" in msg


# ── v0.5 additions ───────────────────────────────────────────────────────────


def test_token_file_resolution(tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text(json.dumps({"token": "nimb_from_file"}), encoding="utf-8")
    cfg = load_config(env={"NIMBUS_TOKEN_FILE": str(token_file)})
    assert cfg.nimbus_token == "nimb_from_file"
    assert cfg.mcp_key == ""


def test_store_resolution_uses_store_api_url():
    write_store(token="nimb_stored", api_url="http://store.api/", name="my token")
    cfg = load_config(env={})
    assert cfg.nimbus_token == "nimb_stored"
    assert cfg.mcp_key == ""
    assert cfg.api_url == "http://store.api"


def test_store_api_url_loses_to_env():
    write_store(token="nimb_stored", api_url="http://store.api")
    cfg = load_config(env={"NIMBUS_API_URL": "http://env.api"})
    assert cfg.nimbus_token == "nimb_stored"
    assert cfg.api_url == "http://env.api"


def test_desktop_autodiscovery_fills_mcp_key(_isolated_home):
    """Zero-config desktop local mode: the desktop mcp-key.json at the current
    OS's userData path resolves into mcp_key with the local default API URL."""
    path = desktop_key_path(sys.platform, env={}, home=_isolated_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": "desk-key", "createdAt": "2026-10-04"}), encoding="utf-8")
    cfg = load_config(env={})
    assert cfg.mcp_key == "desk-key"
    assert cfg.nimbus_token == ""
    assert cfg.api_url == DEFAULT_API_URL


def test_env_key_beats_store_and_desktop(_isolated_home):
    write_store(token="nimb_stored", api_url="http://store.api")
    path = desktop_key_path(sys.platform, env={}, home=_isolated_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": "desk-key"}), encoding="utf-8")
    cfg = load_config(env={"NIMBUS_MCP_KEY": "env-key"})
    assert cfg.mcp_key == "env-key"
    assert cfg.nimbus_token == ""
