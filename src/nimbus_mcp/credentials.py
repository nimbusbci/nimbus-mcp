"""Credential resolution for nimbus-mcp: env → token file → store → desktop key.

The full chain (first hit wins), implementing Nimbus MCP v0.5 "Auth v2":

1. ``NIMBUS_TOKEN`` env — a hosted API token (``nimb_…``), Bearer mode.
2. ``NIMBUS_TOKEN_FILE`` env — a JSON file ``{"token": "…"}`` (secret kept out
   of process env / MCP configs; the docker/CI-friendly variant of #1).
3. ``NIMBUS_MCP_KEY`` / ``NIMBUS_MCP_KEY_FILE`` env — the v0.3 local-key mode,
   kept verbatim for back-compat. ``NIMBUS_MCP_KEY_FILE`` is the JSON shape
   ``{"key": "…"}`` written by the desktop app.
4. The credential store at ``~/.nimbus/credentials.json`` (0600), written by
   ``nimbus-mcp login`` after the device-code flow. Carries the token plus the
   ``api_url`` it was minted for.
5. Desktop-app auto-discovery: the desktop's ``mcp-key.json`` at the
   per-OS Electron userData path (zero-config local mode — the key is local so
   it authenticates with ``X-MCP-Key`` against ``http://127.0.0.1:8080``).

Explicit env configuration always beats files on disk; a store written by
``login`` beats desktop auto-discovery (a user who logged in wants the hosted
token). Unreadable/malformed files that the user *named explicitly* (the two
``*_FILE`` vars) raise :class:`McpToolError`; files we *discover* (store,
desktop key) are skipped silently — auto-discovery must never break startup.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import McpToolError

__all__ = [
    "DEFAULT_API_URL",
    "DESKTOP_APP_DIR",
    "DESKTOP_KEY_FILENAME",
    "ResolvedCredential",
    "credentials_store_path",
    "decode_token_claims",
    "desktop_key_path",
    "discover_desktop_key",
    "linux_desktop_key_path",
    "logout",
    "macos_desktop_key_path",
    "read_store",
    "resolve_credential",
    "windows_desktop_key_path",
    "write_store",
]

DEFAULT_API_URL = "http://127.0.0.1:8080"

# The desktop app's Electron userData folder name and key file (desktop/lib/
# mcp-config.js writes {"key": "...", "createdAt": "..."} there, 0600).
DESKTOP_APP_DIR = "Nimbus Studio"
DESKTOP_KEY_FILENAME = "mcp-key.json"

# Env override for the store location (tests, multi-account setups).
STORE_FILE_ENV = "NIMBUS_CREDENTIALS_FILE"

# Nimbus API tokens are "<prefix><JWT>"; only the JWT part decodes.
NIMBUS_TOKEN_PREFIX = "nimb_"

_VALID_KINDS = ("token", "key", "desktop_key", "none")


@dataclass(frozen=True)
class ResolvedCredential:
    """The outcome of the resolution chain.

    ``kind``:

    - ``"token"`` — hosted API token (env, token file, or the login store);
      the client sends ``Authorization: Bearer``.
    - ``"key"`` — explicit local key (``NIMBUS_MCP_KEY``/``_KEY_FILE``);
      ``X-MCP-Key`` header.
    - ``"desktop_key"`` — auto-discovered desktop app key; local, so the same
      ``X-MCP-Key`` header mode.
    - ``"none"`` — no credential anywhere (setup mode in v0.5 Task 5).

    ``meta`` distinguishes provenance: ``source`` is one of ``env``,
    ``token_file``, ``key_file``, ``store``, ``desktop``; the rest is
    source-specific (``path``, ``name``, ``saved_at``, ``created_at``).
    """

    kind: str
    api_url: str | None
    secret: str | None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unknown credential kind: {self.kind!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Credential store (~/.nimbus/credentials.json)
# ─────────────────────────────────────────────────────────────────────────────


def credentials_store_path(env: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """Location of the credential store (``NIMBUS_CREDENTIALS_FILE`` wins)."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    override = env.get(STORE_FILE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return home / ".nimbus" / "credentials.json"


def read_store(
    env: Mapping[str, str] | None = None, home: Path | None = None
) -> dict[str, Any] | None:
    """Read the store; ``None`` when absent or malformed (never raises)."""
    path = credentials_store_path(env, home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("token"), str):
        return None
    return payload


def write_store(
    token: str,
    api_url: str,
    name: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Persist the login result as ``{"api_url","token","name?","saved_at"}`` 0600.

    The file is opened with mode 0600 and re-``fchmod``-ed before writing, so a
    pre-existing file with wider permissions is tightened *before* the secret
    lands in it (and the mode can never depend on the umask).
    """
    path = credentials_store_path(env, home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "token": token,
        "api_url": api_url.rstrip("/"),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if name:
        payload["name"] = name
    data = json.dumps(payload, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        if hasattr(os, "fchmod"):  # POSIX; Windows keeps os.open's mode intent
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fd = -1  # consumed by fdopen
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def logout(env: Mapping[str, str] | None = None, home: Path | None = None) -> bool:
    """Remove the credential store. Returns whether a store existed."""
    path = credentials_store_path(env, home)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Desktop mcp-key.json auto-discovery (per-OS Electron userData paths)
# ─────────────────────────────────────────────────────────────────────────────


def macos_desktop_key_path(home: Path) -> Path:
    """``~/Library/Application Support/Nimbus Studio/mcp-key.json``."""
    return home / "Library" / "Application Support" / DESKTOP_APP_DIR / DESKTOP_KEY_FILENAME


def linux_desktop_key_path(xdg_config_home: str | None, home: Path) -> Path:
    """``$XDG_CONFIG_HOME/Nimbus Studio/mcp-key.json`` (default ``~/.config/…``)."""
    base = Path(xdg_config_home).expanduser() if xdg_config_home else home / ".config"
    return base / DESKTOP_APP_DIR / DESKTOP_KEY_FILENAME


def windows_desktop_key_path(appdata: str | None, home: Path) -> Path:
    """``%APPDATA%/Nimbus Studio/mcp-key.json`` (default ``~\\AppData\\Roaming/…``)."""
    base = Path(appdata).expanduser() if appdata else home / "AppData" / "Roaming"
    return base / DESKTOP_APP_DIR / DESKTOP_KEY_FILENAME


def desktop_key_path(
    platform: str,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """The desktop key path for ``platform`` (``sys.platform`` naming)."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    if platform == "darwin":
        return macos_desktop_key_path(home)
    if platform == "win32":
        return windows_desktop_key_path(env.get("APPDATA"), home)
    # Everything else follows the XDG convention (Linux, the BSDs).
    return linux_desktop_key_path(env.get("XDG_CONFIG_HOME"), home)


def discover_desktop_key(
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> dict[str, Any] | None:
    """Read the desktop key file if present; ``None`` when absent/broken.

    Auto-discovery is best-effort by design: any error (unreadable, bad JSON,
    wrong shape) resolves to ``None`` so the next chain link runs.
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    platform = sys.platform if platform is None else platform
    path = desktop_key_path(platform, env=env, home=home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("key"), str):
        return None
    key = payload["key"]
    if not key:
        return None
    return {"key": key, "createdAt": payload.get("createdAt"), "path": str(path)}


# ─────────────────────────────────────────────────────────────────────────────
# Resolution chain
# ─────────────────────────────────────────────────────────────────────────────


def resolve_credential(
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> ResolvedCredential:
    """Run the full chain (see the module docstring for order and semantics).

    ``api_url`` resolution: ``NIMBUS_API_URL`` env > the store's ``api_url`` >
    the desktop default (``http://127.0.0.1:8080``, i.e. ``DEFAULT_API_URL``).
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    platform = sys.platform if platform is None else platform
    env_api = env.get("NIMBUS_API_URL", "").strip().rstrip("/")

    def _api_url(store_api: str | None = None) -> str:
        if env_api:
            return env_api
        if store_api:
            return store_api.rstrip("/")
        return DEFAULT_API_URL

    # 1. NIMBUS_TOKEN env — hosted token, wins over everything (v0.3 rule kept).
    token = env.get("NIMBUS_TOKEN", "").strip()
    if token:
        return ResolvedCredential("token", _api_url(), token, {"source": "env"})

    # 2. NIMBUS_TOKEN_FILE — {"token": "..."}; explicit, so errors are loud.
    token_file = env.get("NIMBUS_TOKEN_FILE", "").strip()
    if token_file:
        path = Path(token_file).expanduser()
        payload = _read_explicit_json(path, "NIMBUS_TOKEN_FILE")
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token:
            raise _explicit_file_error(
                path, "NIMBUS_TOKEN_FILE", 'expected a JSON object like {"token": "..."}'
            )
        return ResolvedCredential(
            "token", _api_url(), token, {"source": "token_file", "path": str(path)}
        )

    # 3. Local key mode (v0.3 back-compat): NIMBUS_MCP_KEY > NIMBUS_MCP_KEY_FILE.
    key = env.get("NIMBUS_MCP_KEY", "").strip()
    if key:
        return ResolvedCredential("key", _api_url(), key, {"source": "env"})
    key_file = env.get("NIMBUS_MCP_KEY_FILE", "").strip()
    if key_file:
        path = Path(key_file).expanduser()
        payload = _read_explicit_json(path, "NIMBUS_MCP_KEY_FILE")
        key = payload.get("key") if isinstance(payload, dict) else None
        if not isinstance(key, str) or not key:
            raise _explicit_file_error(
                path, "NIMBUS_MCP_KEY_FILE", 'expected a JSON object like {"key": "..."}'
            )
        return ResolvedCredential("key", _api_url(), key, {"source": "key_file", "path": str(path)})

    # 4. The login store (silent skip when absent/broken).
    store = read_store(env=env, home=home)
    if store and store.get("token"):
        meta: dict[str, Any] = {
            "source": "store",
            "path": str(credentials_store_path(env, home)),
        }
        for optional in ("name", "saved_at"):
            if store.get(optional):
                meta[optional] = store[optional]
        return ResolvedCredential("token", _api_url(store.get("api_url")), store["token"], meta)

    # 5. Desktop auto-discovery (silent skip) — local key, local default URL.
    desktop = discover_desktop_key(env=env, home=home, platform=platform)
    if desktop:
        return ResolvedCredential(
            "desktop_key",
            _api_url(),
            desktop["key"],
            {"source": "desktop", "path": desktop["path"], "created_at": desktop.get("createdAt")},
        )

    return ResolvedCredential("none", _api_url(), None, {})


def _read_explicit_json(path: Path, var_name: str) -> Any:
    """Read a user-named JSON file, raising the actionable error on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise _explicit_file_error(path, var_name, str(err)) from err
    except json.JSONDecodeError as err:
        raise _explicit_file_error(path, var_name, f"invalid JSON: {err}") from err


def _explicit_file_error(path: Path, var_name: str, reason: str) -> McpToolError:
    return McpToolError(f"Cannot read {var_name} at {path}: {reason}")


# ─────────────────────────────────────────────────────────────────────────────
# Local JWT helpers (no verification — claims display only)
# ─────────────────────────────────────────────────────────────────────────────


def decode_token_claims(token: str) -> dict[str, Any]:
    """Decode a Nimbus token's payload locally (``nimb_`` prefix stripped).

    Signature is NOT verified — this is for display (expiry countdowns in
    ``status``/``whoami``), never for an auth decision. Any malformed input
    yields ``{}``.
    """
    jwt_part = token.removeprefix(NIMBUS_TOKEN_PREFIX)
    parts = jwt_part.split(".")
    if len(parts) != 3:
        return {}
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (binascii.Error, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
