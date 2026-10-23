"""Setup mode: structured guidance when no credential resolves (or one dies).

Spec keys are verbatim: ``{ok: false, setupRequired: true, message, options}``.
Every tool call in setup mode — or after a hosted token is rejected
mid-session — returns this dict instead of failing, so the agent can walk the
user through onboarding without the server crashing at startup.
"""

from __future__ import annotations

import math
import time
from typing import Any

from .credentials import decode_token_claims
from .errors import McpToolError

__all__ = [
    "DESKTOP_DOWNLOAD_MACOS",
    "DESKTOP_DOWNLOAD_WINDOWS",
    "HOSTED_GATEWAY_URL",
    "LOCAL_BACKEND_UNREACHABLE",
    "SETUP_OPTIONS",
    "desktop_app_not_running_guidance",
    "SETUP_MESSAGE",
    "SetupRequired",
    "local_mode_guidance",
    "setup_guidance",
    "token_rejected_guidance",
]

# Public "latest" aliases (docs/DESKTOP_APP.md) and the hosted MCP gateway.
DESKTOP_DOWNLOAD_MACOS = "https://nimbus-desktop-releases.fly.storage.tigris.dev/latest.dmg"
DESKTOP_DOWNLOAD_WINDOWS = "https://nimbus-desktop-releases.fly.storage.tigris.dev/latest-win.exe"
HOSTED_GATEWAY_URL = "https://nimbus-mcp.fly.dev/mcp"
# Stable machine-readable reason for "nothing answered on the local backend".
LOCAL_BACKEND_UNREACHABLE = "local_backend_unreachable"

SETUP_MESSAGE = (
    "No Nimbus credential is configured, so this tool cannot run yet. Pick one "
    "of three paths: (1) run `nimbus-mcp login` to connect this machine to the "
    "hosted Nimbus API via a one-click browser approval (or set NIMBUS_TOKEN / "
    "NIMBUS_TOKEN_FILE to a hosted API token minted in Nimbus Studio → Account "
    "→ API tokens); (2) Nimbus Studio desktop app users need no credential at "
    "all — just have the app running and its local key is auto-discovered; "
    "(3) local development: set NIMBUS_MCP_KEY (or NIMBUS_MCP_KEY_FILE) to the "
    "same value as MCP_LOCAL_KEY on the local backend. Then retry this tool."
)

SETUP_OPTIONS: list[dict[str, str]] = [
    {
        "action": "nimbus-mcp login",
        "detail": "Device-code login to the hosted Nimbus API (recommended): "
        "run it in a terminal, approve in the browser, and the token is stored "
        "at ~/.nimbus/credentials.json (0600). NIMBUS_TOKEN / NIMBUS_TOKEN_FILE "
        "are the env-var equivalents if you prefer pasting a minted token.",
    },
    {
        "action": "start the Nimbus Studio desktop app",
        "detail": "Zero-config local mode: while the desktop app runs, its "
        "mcp-key.json is auto-discovered and authenticates against the local "
        "backend (port 8080, or whatever port the key file records when the "
        "app bound another one) — nothing to configure.",
    },
    {
        "action": "set NIMBUS_MCP_KEY / NIMBUS_MCP_KEY_FILE",
        "detail": "Local development with a manually started backend: use the "
        "same value as MCP_LOCAL_KEY on the backend (desktop app or DEBUG=1 "
        "dev server; the key is never accepted on hosted deployments).",
    },
]


def setup_guidance(message: str | None = None, **extra: Any) -> dict[str, Any]:
    """The structured setup-mode payload (spec keys verbatim, plus ``extra``).

    ``extra`` carries situation-specific data, e.g. ``daysExpired`` for a token
    that died mid-session.
    """
    guidance: dict[str, Any] = {
        "ok": False,
        "setupRequired": True,
        "message": message if message is not None else SETUP_MESSAGE,
        "options": [dict(option) for option in SETUP_OPTIONS],
    }
    guidance.update(extra)
    return guidance


def _unreachable_options() -> list[dict[str, str]]:
    return [
        {
            "action": "install the Nimbus Studio desktop app",
            "detail": "It runs the Nimbus backend on this machine and nimbus-mcp "
            "finds it automatically — no login needed. Open the app, then retry.",
            "url_macos": DESKTOP_DOWNLOAD_MACOS,
            "url_windows": DESKTOP_DOWNLOAD_WINDOWS,
        },
        {
            "action": "use the hosted gateway",
            "detail": "No install: point your MCP client at the remote server "
            f"{HOSTED_GATEWAY_URL} with a Nimbus API token (Nimbus Studio → "
            "Account → API tokens), or run `nimbus-mcp login` here to use the "
            "hosted API from this server.",
            "url": HOSTED_GATEWAY_URL,
        },
        {
            "action": "nimbus-mcp login",
            "detail": "Connect this server to the hosted Nimbus API via a "
            "one-click browser approval (token stored at "
            "~/.nimbus/credentials.json, 0600).",
        },
        {
            "action": "set NIMBUS_MCP_KEY / NIMBUS_MCP_KEY_FILE",
            "detail": "Developers: start the dev backend on :8080 and set "
            "NIMBUS_MCP_KEY to its MCP_LOCAL_KEY.",
        },
    ]


def _unreachable_message(lead: str) -> str:
    return (
        f"{lead} Pick one: (1) install and open the Nimbus Studio desktop app — "
        "it runs the backend locally and nimbus-mcp finds it automatically, no "
        f"login needed. macOS: {DESKTOP_DOWNLOAD_MACOS} · Windows: "
        f"{DESKTOP_DOWNLOAD_WINDOWS} · Linux: no desktop build yet — use option (2) "
        "or (3). (2) or use hosted Nimbus with no install: "
        f"point your MCP client at {HOSTED_GATEWAY_URL}, or run `nimbus-mcp "
        "login` here to use the hosted API from this server. (3) Developers: "
        "start the dev backend on :8080 and set NIMBUS_MCP_KEY to its "
        "MCP_LOCAL_KEY. Then retry this tool."
    )


def local_mode_guidance(status: int | None = 401, api_url: str | None = None) -> dict[str, Any]:
    """Guidance when an ANONYMOUS local call fails against the local backend.

    Local Free (plan X9) needs no Nimbus Studio login: the missing piece is
    a local identity (the desktop app's auto-discovered key, or a dev
    backend key), never an account. ``status=None`` means nothing answered at
    all (no desktop app): the payload then carries the desktop download
    links, the hosted gateway and ``reason: local_backend_unreachable``.
    """
    if status is None:
        where = f" (nothing answered at {api_url})" if api_url else ""
        guidance = setup_guidance(
            message=_unreachable_message(f"No Nimbus backend is running on this machine{where}."),
            reason=LOCAL_BACKEND_UNREACHABLE,
        )
        guidance["options"] = _unreachable_options()
        return guidance
    guidance = setup_guidance(
        message=(
            f"The local Nimbus backend rejected this request (HTTP {status}) "
            "and no local identity was presented. No Nimbus Studio login "
            "is needed for local work: start the Nimbus Studio desktop app "
            "(its local key is auto-discovered), or set NIMBUS_MCP_KEY to the "
            "local backend's MCP_LOCAL_KEY for a dev backend. Logging in "
            "(`nimbus-mcp login`) is only needed for cloud features "
            "(leaderboard, verified history, hosted compute, sync)."
        )
    )
    # Local options first; login is last and labelled cloud-only (X9).
    options = guidance["options"]
    local_first = [o for o in options if o["action"] != "nimbus-mcp login"]
    login = [dict(o) for o in options if o["action"] == "nimbus-mcp login"]
    for o in login:
        o["detail"] = "Only for cloud features (not needed locally). " + o["detail"]
    guidance["options"] = local_first + login
    return guidance


def desktop_app_not_running_guidance(api_url: str) -> dict[str, Any]:
    """A desktop-app key file was found but nothing answers: the app is
    installed but closed (or was uninstalled with a stale key left behind)."""
    guidance = setup_guidance(
        message=_unreachable_message(
            "The Nimbus Studio desktop app isn't running (its key file was "
            f"found, but nothing answered at {api_url}). Open the app and "
            "retry. If it isn't installed:"
        ),
        reason=LOCAL_BACKEND_UNREACHABLE,
        desktopKeyFound=True,
    )
    guidance["options"] = _unreachable_options()
    return guidance


def token_rejected_guidance(token: str, status: int) -> dict[str, Any]:
    """Guidance for a hosted token rejected mid-session (401 expired/revoked).

    ``decode_token_claims`` is a LOCAL, no-verify decode — purely to say how
    many days are left (``daysLeft``) or how long ago the token expired
    (``daysExpired``); the raw secret never appears anywhere.
    """
    exp = decode_token_claims(token).get("exp")
    marks = {"tokenRejected": True, "httpStatus": status}
    if not isinstance(exp, (int, float)) or isinstance(exp, bool):
        return setup_guidance(
            **marks,
            message=(
                f"Your Nimbus API token was rejected (HTTP {status}) — run "
                "`nimbus-mcp login` to mint a fresh one (or update NIMBUS_TOKEN "
                "if you set it manually)."
            ),
        )
    remaining = exp - time.time()
    if remaining >= 0:
        days_left = max(0, math.ceil(remaining / 86400))
        return setup_guidance(
            message=(
                f"Your Nimbus API token was rejected (HTTP {status}) but looks "
                f"unexpired ({days_left} day(s) left) — it was probably revoked "
                "or your plan changed. Run `nimbus-mcp login` to mint a fresh "
                "token."
            ),
            daysLeft=days_left,
            **marks,
        )
    days_expired = max(1, math.floor(-remaining / 86400))
    return setup_guidance(
        message=(
            f"Your Nimbus API token expired {days_expired} day(s) ago "
            f"(HTTP {status}) — run `nimbus-mcp login` to mint a fresh one "
            "(or update NIMBUS_TOKEN if you set it manually)."
        ),
        daysExpired=days_expired,
        **marks,
    )


class SetupRequired(McpToolError):
    """No usable credential: convert to the guidance dict, do not fail.

    Raised by :class:`nimbus_mcp.client.NullClient` (nothing resolved at
    startup) and by :class:`nimbus_mcp.client.NimbusClient` when a hosted
    token is rejected mid-session; ``SetupGuidanceMiddleware`` (server.py)
    converts it into ``guidance`` returned as the tool's result.
    """

    def __init__(self, guidance: dict[str, Any]) -> None:
        message = str(guidance.get("message", "Nimbus setup required"))
        super().__init__(message)
        self.guidance = guidance
