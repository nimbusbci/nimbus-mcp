"""Environment-driven configuration for the nimbus-mcp server."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .credentials import DEFAULT_API_URL, resolve_credential
from .errors import McpToolError

__all__ = [
    "DEFAULT_API_URL",
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_SERVE_HOST",
    "McpConfig",
    "default_export_dir",
    "load_config",
    "missing_credential_error",
]


def default_export_dir() -> Path:
    """``$HOME/nimbus-exports``, resolved at CALL time so a HOME that changed
    after import (tests, service users) is honored — one definition; the
    constant below and load_config both derive from it."""
    return Path.home() / "nimbus-exports"


# Import-time snapshot kept for backwards compat; prefer default_export_dir().
DEFAULT_EXPORT_DIR = default_export_dir()

# `nimbus-mcp serve` binds loopback by default: the gateway accepts any token
# the request carries, so an accidental LAN bind would expose an unauthenticated
# MCP surface. Binding 0.0.0.0 (to expose the gateway) is explicit opt-in via
# --host / NIMBUS_MCP_HOST and must be combined with per-request tokens.
DEFAULT_SERVE_HOST = "127.0.0.1"


@dataclass(frozen=True)
class McpConfig:
    api_url: str
    mcp_key: str
    export_dir: Path
    nimbus_token: str = ""
    # Resolution provenance (``source``/``name``/``path`` …) from
    # ResolvedCredential.meta — consumed by the whoami tool. Field names of
    # the v0.3 config are unchanged; this is additive.
    credential_meta: dict[str, Any] = field(default_factory=dict)


def load_config(env: Mapping[str, str] | None = None) -> McpConfig:
    """Resolve config from ``env`` (defaults to ``os.environ``).

    Credentials resolve through :func:`nimbus_mcp.credentials.resolve_credential`
    (Nimbus MCP v0.5): ``NIMBUS_TOKEN`` → ``NIMBUS_TOKEN_FILE`` → the v0.3
    local-key pair ``NIMBUS_MCP_KEY`` / ``NIMBUS_MCP_KEY_FILE`` → the login
    store ``~/.nimbus/credentials.json`` → desktop-app ``mcp-key.json``
    auto-discovery. The ``McpConfig`` field names are unchanged for compat:
    a resolved token lands in ``nimbus_token`` (Bearer mode); a local key —
    env, key file, or the desktop-discovered key — lands in ``mcp_key``
    (``X-MCP-Key`` mode). Config with no credential is returned as-is;
    ``build_server`` enters setup mode with a ``NullClient`` (direct
    ``NimbusClient`` construction still rejects it via
    :func:`missing_credential_error`).
    """
    env = os.environ if env is None else env
    cred = resolve_credential(env=env)
    mcp_key = (cred.secret or "") if cred.kind in ("key", "desktop_key") else ""
    nimbus_token = (cred.secret or "") if cred.kind == "token" else ""
    return McpConfig(
        api_url=(cred.api_url or DEFAULT_API_URL).rstrip("/"),
        mcp_key=mcp_key,
        export_dir=Path(
            env.get("NIMBUS_EXPORT_DIR") or str(default_export_dir())
        ).expanduser(),
        nimbus_token=nimbus_token,
        credential_meta=dict(cred.meta),
    )


def missing_credential_error() -> McpToolError:
    """Error for "no credential resolved anywhere"."""
    return McpToolError(
        "No Nimbus credential configured. Run `nimbus-mcp login` for the device "
        "flow (recommended), or set NIMBUS_TOKEN / NIMBUS_TOKEN_FILE to a hosted "
        "API token (mint one in Nimbus Studio → Account → API tokens), or set "
        "NIMBUS_MCP_KEY / NIMBUS_MCP_KEY_FILE to the same value as MCP_LOCAL_KEY "
        "on a local backend (desktop app or dev server)."
    )
