"""Resolution chain, credential store, and desktop auto-discovery tests."""

import base64
import json
import stat
import time
from pathlib import Path

import pytest

from nimbus_mcp.credentials import (
    DEFAULT_API_URL,
    ResolvedCredential,
    credentials_store_path,
    decode_token_claims,
    desktop_key_path,
    discover_desktop_key,
    linux_desktop_key_path,
    logout,
    macos_desktop_key_path,
    read_store,
    resolve_credential,
    windows_desktop_key_path,
    write_store,
)
from nimbus_mcp.errors import McpToolError

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def _make_desktop_key(home: Path, platform: str, key: str = "desk-key") -> Path:
    path = desktop_key_path(platform, env={}, home=home)
    return _write_json(path, {"key": key, "createdAt": "2026-10-04T00:00:00Z"})


def _b64url(obj: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def _fake_jwt_token(payload: dict) -> str:
    return "nimb_" + f"{_b64url({'alg': 'none'})}.{_b64url(payload)}.sig"


# ─────────────────────────────────────────────────────────────────────────────
# Chain precedence (full chain populated, first hit wins)
# ─────────────────────────────────────────────────────────────────────────────


def test_env_token_wins_over_everything(tmp_path):
    _make_desktop_key(tmp_path, "darwin")
    token_file = _write_json(tmp_path / "token.json", {"token": "nimb_file"})
    key_file = _write_json(tmp_path / "key.json", {"key": "file-key"})
    write_store(token="nimb_stored", api_url="http://store.api", home=tmp_path)
    cred = resolve_credential(
        env={
            "NIMBUS_TOKEN": "nimb_env",
            "NIMBUS_TOKEN_FILE": str(token_file),
            "NIMBUS_MCP_KEY": "env-key",
            "NIMBUS_MCP_KEY_FILE": str(key_file),
        },
        home=tmp_path,
        platform="darwin",
    )
    assert cred == ResolvedCredential("token", DEFAULT_API_URL, "nimb_env", {"source": "env"})


def test_token_file_beats_key_store_and_desktop(tmp_path):
    _make_desktop_key(tmp_path, "darwin")
    token_file = _write_json(tmp_path / "token.json", {"token": "nimb_file"})
    key_file = _write_json(tmp_path / "key.json", {"key": "file-key"})
    write_store(token="nimb_stored", api_url="http://store.api", home=tmp_path)
    cred = resolve_credential(
        env={"NIMBUS_TOKEN_FILE": str(token_file), "NIMBUS_MCP_KEY_FILE": str(key_file)},
        home=tmp_path,
        platform="darwin",
    )
    assert cred.kind == "token"
    assert cred.secret == "nimb_file"
    assert cred.meta["source"] == "token_file"
    assert cred.meta["path"] == str(token_file)


def test_env_key_beats_key_file_store_and_desktop(tmp_path):
    _make_desktop_key(tmp_path, "linux")
    key_file = _write_json(tmp_path / "key.json", {"key": "file-key"})
    write_store(token="nimb_stored", api_url="http://store.api", home=tmp_path)
    cred = resolve_credential(
        env={"NIMBUS_MCP_KEY": "env-key", "NIMBUS_MCP_KEY_FILE": str(key_file)},
        home=tmp_path,
        platform="linux",
    )
    assert cred.kind == "key"
    assert cred.secret == "env-key"
    assert cred.meta == {"source": "env"}


def test_key_file_beats_store_and_desktop(tmp_path):
    _make_desktop_key(tmp_path, "win32")
    key_file = _write_json(tmp_path / "key.json", {"key": "file-key"})
    write_store(token="nimb_stored", api_url="http://store.api", home=tmp_path)
    cred = resolve_credential(
        env={"NIMBUS_MCP_KEY_FILE": str(key_file)}, home=tmp_path, platform="win32"
    )
    assert cred.kind == "key"
    assert cred.secret == "file-key"
    assert cred.meta["source"] == "key_file"


def test_store_beats_desktop_and_supplies_api_url(tmp_path):
    _make_desktop_key(tmp_path, "darwin")
    write_store(token="nimb_stored", api_url="http://store.api/", name="my token", home=tmp_path)
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "token"
    assert cred.secret == "nimb_stored"
    assert cred.api_url == "http://store.api"
    assert cred.meta["source"] == "store"
    assert cred.meta["path"] == str(tmp_path / ".nimbus" / "credentials.json")
    assert cred.meta["name"] == "my token"
    assert cred.meta["saved_at"]


def test_desktop_only_resolves_local_default_url(tmp_path):
    _make_desktop_key(tmp_path, "darwin", key="desk-secret")
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "desktop_key"
    assert cred.secret == "desk-secret"
    assert cred.api_url == DEFAULT_API_URL == "http://127.0.0.1:8080"
    assert cred.meta["source"] == "desktop"
    assert cred.meta["path"].endswith("mcp-key.json")
    assert cred.meta["created_at"] == "2026-10-04T00:00:00Z"


# ─────────────────────────────────────────────────────────────────────────────
# Desktop key file optional "port" field (the desktop app may bind 8081/8082)
# ─────────────────────────────────────────────────────────────────────────────


def _write_desktop_key(home: Path, payload: dict) -> Path:
    return _write_json(desktop_key_path("darwin", env={}, home=home), payload)


def test_desktop_key_port_overrides_default_api_url(tmp_path):
    _write_desktop_key(tmp_path, {"key": "desk-key", "port": 8081})
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "desktop_key"
    assert cred.api_url == "http://127.0.0.1:8081"


def test_desktop_key_without_port_keeps_default_api_url(tmp_path):
    _write_desktop_key(tmp_path, {"key": "desk-key"})
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "desktop_key"
    assert cred.api_url == "http://127.0.0.1:8080"


def test_env_api_url_beats_desktop_key_port(tmp_path):
    _write_desktop_key(tmp_path, {"key": "desk-key", "port": 8081})
    cred = resolve_credential(
        env={"NIMBUS_API_URL": "https://hosted.api"}, home=tmp_path, platform="darwin"
    )
    assert cred.api_url == "https://hosted.api"


def test_desktop_key_non_int_port_falls_back_to_default(tmp_path):
    _write_desktop_key(tmp_path, {"key": "desk-key", "port": "8081"})
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "desktop_key"
    assert cred.api_url == "http://127.0.0.1:8080"


@pytest.mark.parametrize("port", [-1, 0, 65536, 99999])
def test_desktop_key_out_of_range_port_falls_back_to_default(tmp_path, port):
    """Only real TCP ports (1-65535) override the default; anything else
    keeps 8080."""
    _write_desktop_key(tmp_path, {"key": "desk-key", "port": port})
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "desktop_key"
    assert cred.api_url == "http://127.0.0.1:8080"


@pytest.mark.parametrize("port", [1, 65535])
def test_desktop_key_boundary_ports_are_honored(tmp_path, port):
    _write_desktop_key(tmp_path, {"key": "desk-key", "port": port})
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.api_url == f"http://127.0.0.1:{port}"


def test_desktop_key_bool_port_falls_back_to_default(tmp_path):
    """JSON true/false decode to Python bools (a subclass of int) — never a
    port; the 8080 default stands."""
    _write_desktop_key(tmp_path, {"key": "desk-key", "port": True})
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "desktop_key"
    assert cred.api_url == "http://127.0.0.1:8080"


def test_nothing_resolves_to_none(tmp_path):
    cred = resolve_credential(env={}, home=tmp_path, platform="darwin")
    assert cred.kind == "none"
    assert cred.secret is None
    assert cred.api_url == DEFAULT_API_URL


def test_nimbus_api_url_env_beats_store_url(tmp_path):
    write_store(token="nimb_stored", api_url="http://store.api", home=tmp_path)
    cred = resolve_credential(
        env={"NIMBUS_API_URL": "http://env.api/"}, home=tmp_path, platform="darwin"
    )
    assert cred.api_url == "http://env.api"
    assert cred.secret == "nimb_stored"


def test_empty_env_values_are_skipped(tmp_path):
    cred = resolve_credential(
        env={"NIMBUS_TOKEN": "  ", "NIMBUS_MCP_KEY": "", "NIMBUS_TOKEN_FILE": " "},
        home=tmp_path,
        platform="darwin",
    )
    assert cred.kind == "none"


# ─────────────────────────────────────────────────────────────────────────────
# Explicit-file error behavior (loud, mirroring the v0.3 key-file contract)
# ─────────────────────────────────────────────────────────────────────────────


def test_token_file_missing_raises(tmp_path):
    with pytest.raises(McpToolError, match="Cannot read NIMBUS_TOKEN_FILE"):
        resolve_credential(
            env={"NIMBUS_TOKEN_FILE": str(tmp_path / "absent.json")},
            home=tmp_path,
            platform="darwin",
        )


@pytest.mark.parametrize("payload", ['{"token": 42}', '{"other": 1}', "[1]", "{bad json"])
def test_token_file_wrong_shape_raises(tmp_path, payload):
    path = _write_json(tmp_path / "token.json", payload)
    with pytest.raises(McpToolError, match="Cannot read NIMBUS_TOKEN_FILE"):
        resolve_credential(env={"NIMBUS_TOKEN_FILE": str(path)}, home=tmp_path, platform="darwin")


def test_key_file_missing_raises(tmp_path):
    with pytest.raises(McpToolError, match="Cannot read NIMBUS_MCP_KEY_FILE"):
        resolve_credential(
            env={"NIMBUS_MCP_KEY_FILE": str(tmp_path / "absent.json")},
            home=tmp_path,
            platform="darwin",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Credential store
# ─────────────────────────────────────────────────────────────────────────────


def test_store_roundtrip(tmp_path):
    path = write_store(token="nimb_x", api_url="http://api.test/", name="dev box", home=tmp_path)
    assert path == tmp_path / ".nimbus" / "credentials.json"
    payload = read_store(home=tmp_path)
    assert payload is not None
    assert payload["token"] == "nimb_x"
    assert payload["api_url"] == "http://api.test"
    assert payload["name"] == "dev box"
    assert payload["saved_at"]


def test_store_written_0600(tmp_path):
    path = write_store(token="nimb_x", api_url="http://api.test", home=tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_store_tightens_preexisting_wide_permissions(tmp_path):
    """A pre-existing 0644 store must be 0600 before the new secret lands."""
    path = write_store(token="nimb_old", api_url="http://api.test", home=tmp_path)
    path.chmod(0o644)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    write_store(token="nimb_new", api_url="http://api.test", home=tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert read_store(home=tmp_path)["token"] == "nimb_new"


def test_store_env_override_location(tmp_path):
    target = tmp_path / "elsewhere" / "creds.json"
    path = write_store(
        token="nimb_x", api_url="http://api.test", env={"NIMBUS_CREDENTIALS_FILE": str(target)}
    )
    assert path == target
    assert read_store(env={"NIMBUS_CREDENTIALS_FILE": str(target)})["token"] == "nimb_x"
    assert credentials_store_path(env={"NIMBUS_CREDENTIALS_FILE": str(target)}) == target


def test_read_store_absent_and_malformed_return_none(tmp_path):
    assert read_store(home=tmp_path) is None
    bad = tmp_path / ".nimbus" / "credentials.json"
    _write_json(bad, "{not json")
    assert read_store(home=tmp_path) is None
    _write_json(bad, '{"api_url": "http://x"}')  # no token
    assert read_store(home=tmp_path) is None


def test_logout_removes_store(tmp_path):
    write_store(token="nimb_x", api_url="http://api.test", home=tmp_path)
    assert logout(home=tmp_path) is True
    assert read_store(home=tmp_path) is None
    assert logout(home=tmp_path) is False  # absent the second time


# ─────────────────────────────────────────────────────────────────────────────
# Desktop auto-discovery: path builders + discovery per OS
# ─────────────────────────────────────────────────────────────────────────────


def test_macos_path_builder(tmp_path):
    assert macos_desktop_key_path(tmp_path) == (
        tmp_path / "Library" / "Application Support" / "Nimbus Studio" / "mcp-key.json"
    )


def test_linux_path_builder_default_and_xdg(tmp_path):
    assert linux_desktop_key_path(None, tmp_path) == (
        tmp_path / ".config" / "Nimbus Studio" / "mcp-key.json"
    )
    assert linux_desktop_key_path("/custom/xdg", tmp_path) == (
        Path("/custom/xdg") / "Nimbus Studio" / "mcp-key.json"
    )


def test_windows_path_builder_default_and_appdata(tmp_path):
    assert windows_desktop_key_path(None, tmp_path) == (
        tmp_path / "AppData" / "Roaming" / "Nimbus Studio" / "mcp-key.json"
    )
    assert windows_desktop_key_path("C:\\Users\\u\\AppData\\Roaming", tmp_path) == (
        Path("C:\\Users\\u\\AppData\\Roaming") / "Nimbus Studio" / "mcp-key.json"
    )


def test_desktop_discovery_all_three_os_schemes(tmp_path):
    """The exact desktop userData shape on each OS scheme resolves the key."""
    # macOS
    mac = _make_desktop_key(tmp_path, "darwin", key="mac-key")
    found = discover_desktop_key(env={}, home=tmp_path, platform="darwin")
    assert found == {
        "key": "mac-key",
        "createdAt": "2026-10-04T00:00:00Z",
        "path": str(mac),
    }
    # Linux (XDG default)
    _make_desktop_key(tmp_path, "linux", key="linux-key")
    assert discover_desktop_key(env={}, home=tmp_path, platform="linux")["key"] == "linux-key"
    # Linux (XDG_CONFIG_HOME override)
    xdg_home = tmp_path / "xdg-home"
    _write_json(
        linux_desktop_key_path(str(tmp_path / "custom-xdg"), xdg_home),
        {"key": "xdg-key"},
    )
    found = discover_desktop_key(
        env={"XDG_CONFIG_HOME": str(tmp_path / "custom-xdg")}, home=xdg_home, platform="linux"
    )
    assert found is not None and found["key"] == "xdg-key"
    # Windows (APPDATA)
    win_home = tmp_path / "win-home"
    _write_json(
        windows_desktop_key_path(str(tmp_path / "win-appdata"), win_home),
        {"key": "win-key", "createdAt": "x"},
    )
    found = discover_desktop_key(
        env={"APPDATA": str(tmp_path / "win-appdata")}, home=win_home, platform="win32"
    )
    assert found is not None and found["key"] == "win-key"
    # Windows (default %APPDATA% derivation)
    _make_desktop_key(win_home, "win32", key="win-default-key")
    found = discover_desktop_key(env={}, home=win_home, platform="win32")
    assert found is not None and found["key"] == "win-default-key"


def test_desktop_discovery_missing_or_broken_returns_none(tmp_path):
    assert discover_desktop_key(env={}, home=tmp_path, platform="darwin") is None
    broken = desktop_key_path("darwin", env={}, home=tmp_path)
    _write_json(broken, "{not json")
    assert discover_desktop_key(env={}, home=tmp_path, platform="darwin") is None
    _write_json(broken, '{"key": ""}')
    assert discover_desktop_key(env={}, home=tmp_path, platform="darwin") is None
    _write_json(broken, '{"other": 1}')
    assert discover_desktop_key(env={}, home=tmp_path, platform="darwin") is None


# ─────────────────────────────────────────────────────────────────────────────
# Local JWT decode
# ─────────────────────────────────────────────────────────────────────────────


def test_decode_token_claims_exp():
    exp = int(time.time()) + 30 * 86400
    claims = decode_token_claims(_fake_jwt_token({"exp": exp, "email": "u@x.test"}))
    assert claims["exp"] == exp
    assert claims["email"] == "u@x.test"


def test_decode_token_claims_tolerant_of_garbage():
    assert decode_token_claims("nimb_not-a-jwt") == {}
    assert decode_token_claims("") == {}
    assert decode_token_claims("nimb_###.???._ _") == {}
    # payload that is not a JSON object
    assert decode_token_claims(f"nimb_{_b64url({'a': 1})}.{_b64url([1, 2])}.sig") == {}


def test_write_store_surfaces_write_error_not_double_close(tmp_path, monkeypatch):
    """A failing write must surface the ORIGINAL error. The fd is consumed by
    fdopen, so the finally-branch closing it again raises EBADF — which used
    to replace the real cause (e.g. disk full) with 'Bad file descriptor'."""
    import os

    class FailingFile:
        def __init__(self, fd: int) -> None:
            self.fd = fd

        def write(self, data: str) -> int:
            raise OSError("disk full (simulated)")

        def close(self) -> None:
            os.close(self.fd)  # the real with-block closes the fd

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

    monkeypatch.setattr(os, "fdopen", lambda fd, *a, **k: FailingFile(fd))
    with pytest.raises(OSError, match="disk full"):
        write_store(token="nimb_x", api_url="http://t", home=tmp_path)
