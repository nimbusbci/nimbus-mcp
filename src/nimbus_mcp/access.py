"""Free Local Tier access model (development plan X9/X10, section 5.1).

Three access modes, derived from how the client is credentialed:

- ``anonymous`` — no credential at all. This is the Local Free tier: tools
  run against the local backend (the desktop app / a dev backend on
  127.0.0.1:8080) with NO Nimbus Studio login. Nothing here pre-blocks a
  tool call; cloud-only features answer with a structured login prompt
  (:func:`login_prompt`) at the moment they are first needed — only then.
- ``local`` — an MCP key (env/file, or the desktop app's auto-discovered
  key). Same local backend, machine-local identity.
- ``hosted`` — a Nimbus token (``nimbus-mcp login`` / ``NIMBUS_TOKEN``):
  the hosted backend, account-based. Tier gates (X10) apply here, resolved
  from the profile the backend already exposes (``GET /api/me/profile`` ->
  ``capabilities.isPro``, the same signal ``account.whoami`` reports — no
  parallel tier store).

Login prompts fire ONLY at first cloud value: leaderboard identity,
verified/cross-machine experiment history, hosted compute, cloud sync,
collaboration. Exports (BIDS etc.) are never gated at any tier.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .setup_mode import SetupRequired

MODE_ANONYMOUS = "anonymous"
MODE_LOCAL = "local"
MODE_HOSTED = "hosted"

TIER_ANONYMOUS = "anonymous"
TIER_LOCAL = "local"
TIER_ACCOUNT_FREE = "account-free"
TIER_PRO = "pro"

# The only moments a login may be requested (plan section 5.1: prompts fire
# at first cloud value, never as a blanket gate).
LOGIN_TRIGGERS = (
    "leaderboard_submit",
    "verified_history",
    "hosted_compute",
    "cloud_sync",
    "collaboration",
)

__all__ = [
    "LOGIN_TRIGGERS",
    "MODE_ANONYMOUS",
    "MODE_HOSTED",
    "MODE_LOCAL",
    "TIER_ACCOUNT_FREE",
    "TIER_ANONYMOUS",
    "TIER_LOCAL",
    "TIER_PRO",
    "HOSTED_FREE_MAX_CAMPAIGN_RUNS",
    "TIER_CACHE_TTL_SECONDS",
    "account_tier",
    "clear_tier_cache",
    "client_mode",
    "login_prompt",
    "tier_gate",
]


def client_mode(client: Any) -> str:
    """Access mode for a tool-layer client (see module docstring)."""
    mode = getattr(client, "access_mode", None)
    if mode in (MODE_ANONYMOUS, MODE_LOCAL, MODE_HOSTED):
        return str(mode)
    config = getattr(client, "config", None)
    if config is not None:
        if getattr(config, "nimbus_token", None):
            return MODE_HOSTED
        if getattr(config, "mcp_key", None):
            return MODE_LOCAL
        return MODE_ANONYMOUS
    # Unknown stub clients (unit-test fakes) behave as the pre-tier local
    # path: no gating.
    return MODE_LOCAL


def login_prompt(
    trigger: str,
    *,
    feature: str,
    reason: str,
    local_alternative: str | None = None,
) -> dict[str, Any]:
    """Structured, actionable login prompt — fired only at cloud-value triggers."""
    prompt: dict[str, Any] = {
        "ok": False,
        "loginRequired": True,
        "trigger": trigger,
        "feature": feature,
        "reason": reason,
        "howToAuthenticate": [
            "Run `nimbus-mcp login` in a terminal and approve in the browser "
            "(stores a hosted token at ~/.nimbus/credentials.json), or set "
            "NIMBUS_TOKEN to a token minted in Nimbus Studio -> Account -> "
            "API tokens.",
            "Nothing local needs an account: templates, your own data, local "
            "execution/plots/stats, BIDS export and local-only storage all "
            "keep working anonymously.",
        ],
    }
    if local_alternative:
        prompt["localAlternative"] = local_alternative
    return prompt


def tier_gate(
    feature: str,
    *,
    current_tier: str,
    detail: str,
    local_alternative: str | None = None,
) -> dict[str, Any]:
    """Structured response when a hosted feature needs a higher tier (X10)."""
    gate: dict[str, Any] = {
        "ok": False,
        "tierGate": True,
        "feature": feature,
        "requiredTier": TIER_PRO,
        "currentTier": current_tier,
        "reason": detail,
        "howToUpgrade": (
            "Verified leaderboard submissions, hosted compute and team "
            "features are Pro ($29). Upgrade in Nimbus Studio -> Account; "
            "the change is visible to this client on the next call (the "
            "backend profile is the source of truth)."
        ),
    }
    if local_alternative:
        gate["localAlternative"] = local_alternative
    return gate


# Tier cache: keyed by CREDENTIAL fingerprint (never by client object — the
# gateway shares one HeaderCredentialClient across all users), with a short
# TTL so an upgrade/downgrade is visible within minutes, matching the
# tier-gate copy.
TIER_CACHE_TTL_SECONDS = 300.0
HOSTED_FREE_MAX_CAMPAIGN_RUNS = 5
_tier_cache: dict[str, tuple[str, float]] = {}
_tier_cache_lock = threading.Lock()


def _credential_key(client: Any) -> str | None:
    fingerprint = getattr(client, "credential_fingerprint", None)
    if not callable(fingerprint):
        return None
    try:
        key = fingerprint()
    except Exception:  # noqa: BLE001 - no resolvable credential -> no caching
        return None
    return key if isinstance(key, str) and key and key != "anon" else None


def clear_tier_cache() -> None:
    with _tier_cache_lock:
        _tier_cache.clear()


def account_tier(client: Any) -> str:
    """Tier for gating: anonymous | local | account-free | pro.

    Hosted mode resolves Pro from the backend profile
    (``capabilities.isPro`` — backend truth, mirrored not invented), cached
    per credential for :data:`TIER_CACHE_TTL_SECONDS`. A rejected/expired
    credential (``SetupRequired``) re-raises so the caller gets re-login
    guidance; any other profile failure fails CLOSED to account-free and is
    not cached.
    """
    mode = client_mode(client)
    if mode == MODE_ANONYMOUS:
        return TIER_ANONYMOUS
    if mode == MODE_LOCAL:
        return TIER_LOCAL
    key = _credential_key(client)
    now = time.monotonic()
    if key is not None:
        with _tier_cache_lock:
            hit = _tier_cache.get(key)
        if hit is not None and now - hit[1] < TIER_CACHE_TTL_SECONDS:
            return hit[0]
    try:
        profile = client.get("/api/me/profile")
    except SetupRequired:
        raise
    except Exception:  # noqa: BLE001 - fail closed to account-free, uncached
        return TIER_ACCOUNT_FREE
    caps = profile.get("capabilities") if isinstance(profile, dict) else None
    tier = TIER_PRO if isinstance(caps, dict) and caps.get("isPro") else TIER_ACCOUNT_FREE
    if key is not None:
        with _tier_cache_lock:
            _tier_cache[key] = (tier, now)
    return tier
