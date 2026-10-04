"""Whoami tool: the authenticated identity, plan, quota, and token countdown."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ..credentials import decode_token_claims
from ._annotations import READ_ONLY


def _token_info(client: NimbusClient) -> dict[str, Any] | None:
    """Display info for the active hosted token — LOCAL decode only.

    ``name`` comes from the credential store metadata (written by
    ``nimbus-mcp login``; the JWT itself carries no name claim), the expiry
    from the token's own ``exp`` via :func:`decode_token_claims` (no
    verification, no extra endpoint). The raw secret never appears. In
    local-key modes there is no token: ``None``.
    """
    token = client.config.nimbus_token
    if not token:
        return None
    info: dict[str, Any] = {}
    name = client.config.credential_meta.get("name")
    if isinstance(name, str) and name:
        info["name"] = name
    exp = decode_token_claims(token).get("exp")
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        info["expiresAt"] = datetime.fromtimestamp(exp, timezone.utc).isoformat()
        info["daysLeft"] = max(0, math.ceil((exp - time.time()) / 86400))
    return info or None


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(annotations=READ_ONLY)
    def whoami() -> dict[str, Any]:
        """Who you are authenticated as: account email, plan (isPro / pioneer),
        this month's free-run quota, and — with a hosted token — the token name
        and days until it expires. Call this first when setup guidance appears
        or to check which credential a session uses."""
        profile = client.get("/api/me/profile")
        caps = profile.get("capabilities") or {}
        return {
            "userId": profile.get("userId"),
            "email": profile.get("email"),
            "plan": {
                "isPro": bool(caps.get("isPro")),
                "pioneerAccess": bool(caps.get("hasPioneerAccess")),
            },
            "freeRuns": {
                "monthlyLimit": caps.get("freeTrainingRunsMonthlyLimit"),
                "remaining": caps.get("freeTrainingRunsRemaining"),
            },
            "token": _token_info(client),
            "source": client.config.credential_meta.get("source") or "env",
        }
