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
    "SETUP_OPTIONS",
    "SETUP_MESSAGE",
    "SetupRequired",
    "setup_guidance",
    "token_rejected_guidance",
]

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
        "backend at http://127.0.0.1:8080 — nothing to configure.",
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


def token_rejected_guidance(token: str, status: int) -> dict[str, Any]:
    """Guidance for a hosted token rejected mid-session (401 expired/revoked).

    ``decode_token_claims`` is a LOCAL, no-verify decode — purely to say how
    many days are left (``daysLeft``) or how long ago the token expired
    (``daysExpired``); the raw secret never appears anywhere.
    """
    exp = decode_token_claims(token).get("exp")
    if not isinstance(exp, (int, float)) or isinstance(exp, bool):
        return setup_guidance(
            message=(
                f"Your Nimbus API token was rejected (HTTP {status}) — run "
                "`nimbus-mcp login` to mint a fresh one (or update NIMBUS_TOKEN "
                "if you set it manually)."
            )
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
        )
    days_expired = max(1, math.floor(-remaining / 86400))
    return setup_guidance(
        message=(
            f"Your Nimbus API token expired {days_expired} day(s) ago "
            f"(HTTP {status}) — run `nimbus-mcp login` to mint a fresh one "
            "(or update NIMBUS_TOKEN if you set it manually)."
        ),
        daysExpired=days_expired,
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
