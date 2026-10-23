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

from .errors import McpToolError, local_tool_error
from .setup_mode import SetupRequired

TIER_UNVERIFIABLE_CODE = "tier_unverifiable"


class TierUnverifiable(McpToolError):
    """The account tier could not be read (backend unreachable / erroring).

    Raised instead of guessing: failing "closed" to account-free told real
    Pro users on a network blip to upgrade (B2). The gate cannot be decided
    either way, so the call stops with an honest connectivity error.
    """


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
    "verify_hosted_credential",
    "require_token_scope",
    "TOKEN_SCOPE_INSUFFICIENT_CODE",
    "clear_tier_cache",
    "client_mode",
    "login_prompt",
    "tier_gate",
    "TierUnverifiable",
    "TIER_UNVERIFIABLE_CODE",
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
# key -> (tier, monotonic stamp, profile body) — the profile rides along so
# experiment.run's computeOrigin stamp reuses it instead of a second GET.
_tier_cache: dict[str, tuple[str, float, dict[str, Any]]] = {}
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
    guidance; any other profile failure (unreachable backend, timeout, 5xx,
    unexpected body) raises :class:`TierUnverifiable` — never a guessed
    tier, never cached — so nobody is told to upgrade because the network
    blinked.
    """
    mode = client_mode(client)
    if mode == MODE_ANONYMOUS:
        return TIER_ANONYMOUS
    if mode == MODE_LOCAL:
        return TIER_LOCAL
    return _hosted_tier_and_profile(client)[0]


def _hosted_tier_and_profile(client: Any) -> tuple[str, dict[str, Any]]:
    """One ``GET /api/me/profile`` (or a fresh cache hit) -> (tier, profile).
    Failure semantics are :func:`account_tier`'s."""
    key = _credential_key(client)
    now = time.monotonic()
    if key is not None:
        with _tier_cache_lock:
            hit = _tier_cache.get(key)
        if hit is not None and now - hit[1] < TIER_CACHE_TTL_SECONDS:
            return hit[0], hit[2]
    try:
        profile = client.get("/api/me/profile")
    except SetupRequired:
        raise
    except Exception as err:  # noqa: BLE001 - cannot verify: say so, uncached
        api_url = getattr(getattr(client, "config", None), "api_url", None)
        where = f" at {api_url}" if api_url else ""
        raise TierUnverifiable(
            f"Cannot verify your Nimbus plan: the hosted backend{where} could "
            f"not be reached or returned an error ({_short(err)}). This is a "
            "connectivity problem, not a plan limit — retry shortly, check "
            "NIMBUS_API_URL / your network, or run account.whoami to see the "
            "connection status.",
            status_code=getattr(err, "status_code", None),
            code=TIER_UNVERIFIABLE_CODE,
        ) from err
    if not isinstance(profile, dict) or "capabilities" not in profile:
        raise TierUnverifiable(
            "Cannot verify your Nimbus plan: the backend profile response had "
            "no capabilities block. Retry shortly; if it persists, report it.",
            code=TIER_UNVERIFIABLE_CODE,
        )
    caps = profile.get("capabilities")
    tier = TIER_PRO if isinstance(caps, dict) and caps.get("isPro") else TIER_ACCOUNT_FREE
    if key is not None:
        with _tier_cache_lock:
            _tier_cache[key] = (tier, now, profile)
    return tier, profile


def _short(err: BaseException) -> str:
    text = str(err).strip().splitlines()[0] if str(err).strip() else ""
    name = type(err).__name__
    return f"{name}: {text[:160]}" if text else name


def verify_hosted_credential(client: Any) -> dict[str, Any] | None:
    """Fail fast on a rejected hosted token before work starts (B5).

    Hosted mode only: resolves the tier through :func:`account_tier`, so a
    401 raises the re-login :class:`SetupRequired` (gateway: the
    rejected-token guidance) and an unreachable backend raises
    :class:`TierUnverifiable`. Success is cached per credential for
    :data:`TIER_CACHE_TTL_SECONDS` (the same cache tier gates read), so a
    repeat call — or the >5-run gate right after it — costs no request.
    Anonymous/local clients are a no-op.

    Returns the hosted profile body (fresh or cached) so callers such as
    experiment.run's computeOrigin stamp reuse it instead of fetching the
    profile a second time; ``None`` outside hosted mode.
    """
    if client_mode(client) == MODE_HOSTED:
        return _hosted_tier_and_profile(client)[1]
    return None


TOKEN_SCOPE_INSUFFICIENT_CODE = "nimbus.auth.token_scope_insufficient"


def _active_token_claims(client: Any) -> dict[str, Any]:
    """Unverified claims of the token this call runs under (``{}`` if none)."""
    from .credentials import decode_token_claims

    claims_fn = getattr(client, "current_token_claims", None)
    if callable(claims_fn):
        claims = claims_fn()
        return claims if isinstance(claims, dict) else {}
    config = getattr(client, "config", None)
    token = getattr(config, "nimbus_token", "") if config is not None else ""
    return decode_token_claims(token) if isinstance(token, str) and token else {}


def require_token_scope(client: Any, scope: str, action: str) -> None:
    """Fail fast when a scoped API token lacks ``scope`` (N4).

    Background work (experiment.run's worker) would otherwise answer
    ``status: running`` and die seconds later on the backend's 403. This
    mirrors the backend's rule (``api_token_scopes.enforce_api_token_scope``):
    a token without an ``scp`` claim is unrestricted; one with ``scp`` must
    list ``scope``. The claims are decoded locally WITHOUT verification —
    this is only an early, side-effect-free refusal; the backend still
    enforces every request, so a forged claim gains nothing.
    """
    if client_mode(client) != MODE_HOSTED:
        return
    claims = _active_token_claims(client)
    if "scp" not in claims:
        return
    raw = claims.get("scp")
    if isinstance(raw, str):
        raw = raw.split()
    granted = {s for s in raw if isinstance(s, str)} if isinstance(raw, list) else set()
    if scope in granted:
        return
    raise local_tool_error(
        f"Nimbus API error 403: This API token lacks the '{scope}' scope required for "
        f"{action}. Mint a token with that scope in Nimbus Studio → Account → API "
        "tokens.",
        status_code=403,
        code=TOKEN_SCOPE_INSUFFICIENT_CODE,
    )
