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
   it authenticates with ``X-MCP-Key`` against ``http://127.0.0.1:<port>``
   from the key file's optional ``port`` field, default ``:8080``).

Explicit env configuration always beats files on disk; a store written by
``login`` beats desktop auto-discovery (a user who logged in wants the hosted
token). Unreadable/malformed files that the user *named explicitly* (the two
``*_FILE`` vars) raise :class:`McpToolError`; files we *discover* (store,
desktop key) are skipped silently — auto-discovery must never break startup.

Two credential/URL pinning guards ride on top of the chain:

- ``NIMBUS_API_URL`` env overriding the store-recorded ``api_url`` with a
  different origin prints ONE warning naming both hosts — the login token
  would silently be sent to the env host otherwise.
- Resolving an ``X-MCP-Key``-mode credential (``key``/``desktop_key``)
  whose ``api_url`` host is not loopback raises :class:`McpToolError` unless
  ``NIMBUS_ALLOW_REMOTE_MCP_KEY=1``: the local key is a loopback-only
  credential, and sending it to a remote host would leak it.
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
from urllib.parse import urlsplit

from .errors import McpToolError

__all__ = [
    "DEFAULT_API_URL",
    "DESKTOP_APP_DIR",
    "DESKTOP_KEY_FILENAME",
    "REMOTE_MCP_KEY_OVERRIDE_ENV",
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
# mcp-config.js writes {"key": "...", "createdAt": "...", "port": <int>?} there,
# 0600; "port" is the backend port the app actually bound).
DESKTOP_APP_DIR = "Nimbus Studio"
DESKTOP_KEY_FILENAME = "mcp-key.json"

# Env override for the store location (tests, multi-account setups).
STORE_FILE_ENV = "NIMBUS_CREDENTIALS_FILE"

# Opt-in escape hatch for sending the local X-MCP-Key to a non-loopback
# backend (e.g. a dev box reached over the LAN behind a tunnel). Exactly "1".
REMOTE_MCP_KEY_OVERRIDE_ENV = "NIMBUS_ALLOW_REMOTE_MCP_KEY"

# Hostnames that count as "this machine" for the local-key policy. No subnet
# wildcard: anything not listed (incl. 127.0.0.2+, ::ffff:127.0.0.1) is
# treated as remote and refused until the override is set.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

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
        fh = os.fdopen(fd, "w", encoding="utf-8")
        fd = -1  # consumed by fdopen — the finally must never close it again
        with fh:
            fh.write(data)
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
    found = {"key": key, "createdAt": payload.get("createdAt"), "path": str(path)}
    # Optional "port" (the backend port the desktop app actually bound) rides
    # along only when it is a real TCP port; anything else keeps the 8080
    # default below (bools are ints in Python — excluded explicitly).
    port = payload.get("port")
    if isinstance(port, int) and not isinstance(port, bool) and 1 <= port <= 65535:
        found["port"] = port
    return found


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
    the desktop key file's ``port`` (``http://127.0.0.1:<port>``) > the
    desktop default (``http://127.0.0.1:8080``, i.e. ``DEFAULT_API_URL``).

    ``X-MCP-Key`` results (``key``/``desktop_key``) are additionally checked
    against the resolved ``api_url``: a non-loopback host raises
    :class:`McpToolError` unless ``NIMBUS_ALLOW_REMOTE_MCP_KEY=1`` (see
    :func:`_guard_local_key_scope`).
    """
    env = os.environ if env is None else env
    cred = _resolve_chain(env, home, platform)
    _guard_local_key_scope(cred, env)
    return cred


def _resolve_chain(
    env: Mapping[str, str],
    home: Path | None,
    platform: str | None,
) -> ResolvedCredential:
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
        _warn_env_api_override(env_api, store.get("api_url"))
        meta: dict[str, Any] = {
            "source": "store",
            "path": str(credentials_store_path(env, home)),
        }
        for optional in ("name", "saved_at"):
            if store.get(optional):
                meta[optional] = store[optional]
        return ResolvedCredential("token", _api_url(store.get("api_url")), store["token"], meta)

    # 5. Desktop auto-discovery (silent skip) — local key; the key file's
    # optional ``port`` (the app binds 8081/8082 when 8080 is taken) overrides
    # the default local URL.
    desktop = discover_desktop_key(env=env, home=home, platform=platform)
    if desktop:
        port_url = f"http://127.0.0.1:{desktop['port']}" if desktop.get("port") else None
        return ResolvedCredential(
            "desktop_key",
            _api_url(port_url),
            desktop["key"],
            {"source": "desktop", "path": desktop["path"], "created_at": desktop.get("createdAt")},
        )

    return ResolvedCredential("none", _api_url(), None, {})


def _url_origin(url: str) -> tuple[str, str, int] | None:
    """``(scheme, host, port)`` with scheme-default ports normalized — two
    URLs share an origin iff these triples match. Unparseable/relative input
    (no host) yields ``None``, which never equals any real origin."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host:
            return None
        scheme = parts.scheme.lower()
        port = parts.port
        if port is None:
            port = 443 if scheme == "https" else 80
        return (scheme, host, port)
    except ValueError:  # e.g. a non-numeric port
        return None


def _host_label(url: str) -> str:
    """``host`` or ``host:port`` — how a URL's host is named in messages.

    Never raises on malformed input (mirroring :func:`_url_origin`):
    ``urlsplit`` itself rarely raises, but its ``.port`` access does so lazily
    for a non-numeric/out-of-range port — that degrades to the hostname alone,
    and an unparsable URL to the raw string, so guards and warnings built on
    this label always name something useful instead of crashing.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:  # e.g. http://host:badport — the host is still nameable
        return host or url
    if port is not None:
        return f"{host}:{port}"
    return host or url


def _warn_env_api_override(env_api: str, store_api: Any) -> None:
    """One stderr line when ``NIMBUS_API_URL`` points the STORE's token at a
    different origin than the one it was minted for — without this the token
    silently travels to the env host. Same origin with a different path is
    fine (no line); an env/explicit token with a store merely sitting on disk
    never reaches here (the chain short-circuits before the store is read)."""
    if not env_api or not isinstance(store_api, str) or not store_api.strip():
        return
    store_api = store_api.rstrip("/")
    if not store_api or _url_origin(env_api) == _url_origin(store_api):
        return
    print(
        f"WARNING: NIMBUS_API_URL ({_host_label(env_api)}) overrides the API URL "
        f"recorded at login ({_host_label(store_api)}): the login token will be "
        "sent to the env host, not the one it was minted for.",
        file=sys.stderr,
    )


def _guard_local_key_scope(cred: ResolvedCredential, env: Mapping[str, str]) -> None:
    """Refuse ``X-MCP-Key`` credentials aimed at a non-loopback backend.

    The local key authenticates the desktop app / a dev backend on this
    machine; sending it to a remote host leaks it off-box for no benefit
    (remote deployments reject it anyway — see the client's auth hint). Set
    ``NIMBUS_ALLOW_REMOTE_MCP_KEY=1`` for the rare deliberate case (e.g. a
    tunneled dev box). Hosted tokens are exempt: remote URLs are their point.
    """
    api_url = cred.api_url or DEFAULT_API_URL
    if cred.kind not in ("key", "desktop_key"):
        return
    try:
        host = (urlsplit(api_url).hostname or "").lower()
    except ValueError:
        host = ""
    if host in _LOOPBACK_HOSTS:
        return
    if str(env.get(REMOTE_MCP_KEY_OVERRIDE_ENV, "")).strip() == "1":
        return
    raise McpToolError(
        f"Refusing to send the local X-MCP-Key to a non-loopback backend "
        f"({_host_label(api_url)} from {api_url}): MCP keys are credentials for "
        "the desktop app or a dev backend on this machine, not for remote "
        "hosts. Use a hosted API token (NIMBUS_TOKEN / NIMBUS_TOKEN_FILE, or "
        "`nimbus-mcp login`) for remote backends, or set "
        f"{REMOTE_MCP_KEY_OVERRIDE_ENV}=1 to allow it explicitly."
    )


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
