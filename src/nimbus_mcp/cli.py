"""``nimbus-mcp`` command line: ``login`` / ``logout`` / ``status`` + the server.

Invocation shapes:

- ``nimbus-mcp login [--api-url URL] [--ttl-days N] [--name NAME]`` — the
  device-code flow: create a grant (public endpoint, no auth header), open the
  verification URL in the browser (printed as fallback), poll until approved /
  denied / expired (honoring ``slow_down`` intervals), then save the minted
  token to the credential store.
- ``nimbus-mcp logout`` — remove the store.
- ``nimbus-mcp status`` — doctor: which credential source resolved, the API
  URL, the secret prefix (12 chars max), and a live health probe; exit 0 when
  healthy, 1 otherwise.
- ``nimbus-mcp`` (bare) or any unknown argument — the stdio MCP server,
  exactly as pre-v0.5 (``server.main``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import webbrowser
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import httpx

from .credentials import (
    DEFAULT_API_URL,
    ResolvedCredential,
    credentials_store_path,
    decode_token_claims,
    resolve_credential,
    write_store,
)
from .credentials import logout as remove_store

# Mint bounds mirrored from the backend device flow (7..90, default 30).
TTL_DAYS_MIN = 7
TTL_DAYS_MAX = 90

# Overall login give-up bound: the grant TTL plus slack for one slow poll.
LOGIN_TIMEOUT_SLACK_SECONDS = 120

_SECRET_PREFIX_LEN = 12

_KNOWN_COMMANDS = ("login", "logout", "status")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (console script). Bare/unknown → stdio server."""
    argv = list(sys.argv[1:] if argv is None else argv)
    known = _KNOWN_COMMANDS + ("-h", "--help")  # help must not start the server
    if not argv or argv[0] not in known:
        # Import lazily so login/status never pay the fastmcp import cost.
        from .server import main as server_main

        server_main()
        return 0
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "login":
        return cmd_login(args)
    if args.command == "logout":
        return cmd_logout(args)
    return cmd_status(args)


def _poll_interval(value: Any, default: int) -> int:
    """Parse a server-suggested poll interval, clamped to 1..60 seconds.

    Guarded ``int()``: a malformed value (string, None, float-garbage) falls
    back to ``default`` instead of crashing the login loop; the 60 s cap keeps
    a hostile/buggy server from stretching polls past the grant TTL.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(60, parsed))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nimbus-mcp",
        description="Nimbus Studio MCP server and account CLI.",
    )
    sub = parser.add_subparsers(dest="command")
    login = sub.add_parser("login", help="log in via the Nimbus Studio device flow")
    login.add_argument(
        "--api-url", default=None, metavar="URL", help="backend to log in to (default: env "
        "NIMBUS_API_URL, else http://127.0.0.1:8080)"
    )
    login.add_argument(
        "--ttl-days",
        type=int,
        default=None,
        metavar="N",
        help=f"minted-token lifetime in days ({TTL_DAYS_MIN}..{TTL_DAYS_MAX}, default 30)",
    )
    login.add_argument(
        "--name", default=None, metavar="NAME", help="display name for the minted token"
    )
    sub.add_parser("logout", help="remove the stored credential")
    sub.add_parser("status", help="show credential source and probe the backend")
    return parser


def _sleep(seconds: float) -> None:
    """Indirection so tests can fast-forward the poll loop."""
    time.sleep(seconds)


# ─────────────────────────────────────────────────────────────────────────────
# login
# ─────────────────────────────────────────────────────────────────────────────


def cmd_login(args: argparse.Namespace, transport: httpx.BaseTransport | None = None) -> int:
    """Run the device-code flow and store the minted token."""
    api_url = (
        args.api_url or os.environ.get("NIMBUS_API_URL") or DEFAULT_API_URL
    ).rstrip("/")
    if args.ttl_days is not None and not (TTL_DAYS_MIN <= args.ttl_days <= TTL_DAYS_MAX):
        print(f"--ttl-days must be {TTL_DAYS_MIN}..{TTL_DAYS_MAX}, got {args.ttl_days}.")
        return 1

    body: dict[str, Any] = {}
    if args.name:
        body["name"] = args.name
    if args.ttl_days is not None:
        body["ttlDays"] = args.ttl_days

    with httpx.Client(base_url=api_url, timeout=30.0, transport=transport) as http:
        try:
            resp = http.post("/api/mcp/device-code", json=body)
        except httpx.HTTPError as err:
            print(f"Cannot reach the Nimbus backend at {api_url}: {err}")
            return 1
        if resp.status_code != 200:
            print(f"Login failed: backend returned {resp.status_code} {_error_detail(resp)}.")
            return 1
        try:
            created = resp.json()
            device_code = created["deviceCode"]
            user_code = created["userCode"]
            verification_url = created["verificationUrl"]
            interval = _poll_interval(created.get("interval"), 5)
            expires_in = int(created.get("expiresIn", 600))
        except (ValueError, KeyError) as err:
            print(f"Login failed: malformed device-code response ({err!r}).")
            return 1

        print(f"Requesting a device code from {api_url} …")
        print(f"  1) open: {verification_url}")
        print(f"  2) confirm the code shown there matches: {user_code}")
        print("Waiting for approval (Ctrl-C to abort) …")
        _open_browser(verification_url)

        deadline = time.monotonic() + expires_in + LOGIN_TIMEOUT_SLACK_SECONDS
        while True:
            if time.monotonic() >= deadline:
                print("Timed out waiting for approval. Run nimbus-mcp login again.")
                return 1
            _sleep(interval)
            try:
                resp = http.post("/api/mcp/device-token", json={"deviceCode": device_code})
            except httpx.HTTPError as err:
                print(f"Lost contact with the backend: {err}")
                return 1
            if resp.status_code == 404:
                # The no-oracle answer: unknown / timed-out / already used.
                print("The device code expired or was already used. Run nimbus-mcp login again.")
                return 1
            if resp.status_code != 200:
                print(f"Login failed: backend returned {resp.status_code} {_error_detail(resp)}.")
                return 1
            try:
                poll = resp.json()
            except ValueError:
                print("Login failed: malformed poll response.")
                return 1
            status = poll.get("status")
            if status == "pending":
                interval = _poll_interval(poll.get("interval"), interval)
                continue
            if status == "slow_down":
                interval = _poll_interval(poll.get("interval"), interval + 1)
                continue
            if status == "denied":
                print("The request was denied in the web app. Nothing was saved.")
                return 1
            if status == "approved":
                return _finish_login(api_url, poll)


def _finish_login(api_url: str, poll: dict[str, Any]) -> int:
    token = poll.get("token")
    if not isinstance(token, str) or not token:
        print("Login failed: approved response carried no token.")
        return 1
    name = poll.get("name")
    expires_at = poll.get("expiresAt")
    path = write_store(token=token, api_url=api_url, name=name if isinstance(name, str) else None)
    print(f"Logged in as {name or 'your Nimbus account'}.")
    if expires_at:
        print(f"Token expires {expires_at}.")
    print(f"Credential saved to {path} (0600). Run nimbus-mcp status to verify.")
    return 0


def _open_browser(url: str) -> None:
    """Best-effort browser open; the printed URL is the fallback either way."""
    try:
        webbrowser.open(url)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# logout
# ─────────────────────────────────────────────────────────────────────────────


def cmd_logout(args: argparse.Namespace) -> int:
    try:
        removed = remove_store()
    except PermissionError:
        print(f"Cannot remove {credentials_store_path()} — permission denied.")
        return 1
    if removed:
        print(f"Removed {credentials_store_path()}.")
        return 0
    print(f"No stored credential at {credentials_store_path()} — nothing to log out.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# status (doctor)
# ─────────────────────────────────────────────────────────────────────────────


def cmd_status(
    args: argparse.Namespace,
    transport: httpx.BaseTransport | None = None,
    cred: ResolvedCredential | None = None,
) -> int:
    """Print the resolved credential and probe the backend. 0 healthy / 1 not."""
    cred = resolve_credential() if cred is None else cred
    print(f"credential source: {_describe_source(cred)}")
    print(f"api url: {cred.api_url}")
    if cred.secret:
        label = "token" if cred.kind == "token" else "key"
        # A secret shorter than the 12-char prefix would be printed WHOLE by a
        # naive slice — those get a 4-char prefix instead (still identifiable,
        # never reconstructable).
        prefix_len = _SECRET_PREFIX_LEN if len(cred.secret) > _SECRET_PREFIX_LEN else 4
        print(f"{label} prefix: {cred.secret[:prefix_len]}…")
    if cred.kind == "none":
        print(
            "No Nimbus credential found. Run `nimbus-mcp login`, or set "
            "NIMBUS_TOKEN / NIMBUS_TOKEN_FILE (hosted), or NIMBUS_MCP_KEY / "
            "NIMBUS_MCP_KEY_FILE (local). Desktop app users need no setup."
        )
        return 1
    if cred.kind == "token":
        return _probe_profile(cred, transport)
    return _probe_health(cred, transport)


def _describe_source(cred: ResolvedCredential) -> str:
    source = cred.meta.get("source")
    if cred.kind == "token":
        return {
            "env": "NIMBUS_TOKEN (environment)",
            "token_file": f"NIMBUS_TOKEN_FILE ({cred.meta.get('path', '?')})",
            "store": f"credential store ({cred.meta.get('path', '~/.nimbus/credentials.json')})",
        }.get(source, "token")
    if cred.kind == "key":
        return {
            "env": "NIMBUS_MCP_KEY (environment)",
            "key_file": f"NIMBUS_MCP_KEY_FILE ({cred.meta.get('path', '?')})",
        }.get(source, "local key")
    if cred.kind == "desktop_key":
        return f"desktop app key (auto-discovered at {cred.meta.get('path', '?')})"
    return "none"


def _probe_profile(cred: ResolvedCredential, transport: httpx.BaseTransport | None) -> int:
    """Token mode: verify against GET /api/me/profile and show plan/quota/expiry."""
    assert cred.secret is not None
    claims = decode_token_claims(cred.secret)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        days_left = max(0, math.ceil((exp - time.time()) / 86400))
        exp_iso = datetime.fromtimestamp(exp, timezone.utc).isoformat()
        print(f"token expires in {days_left} day(s) (exp {exp_iso})")
    try:
        with httpx.Client(
            base_url=cred.api_url or DEFAULT_API_URL,
            headers={"Authorization": f"Bearer {cred.secret}"},
            timeout=15.0,
            transport=transport,
        ) as http:
            resp = http.get("/api/me/profile")
    except httpx.HTTPError as err:
        print(f"backend unreachable: {err}")
        return 1
    if resp.status_code != 200:
        hint = ""
        if resp.status_code in (401, 403):
            hint = " — the token was rejected; run nimbus-mcp login again"
        print(f"profile probe failed: HTTP {resp.status_code} {_error_detail(resp)}{hint}.")
        return 1
    try:
        profile = resp.json()
    except ValueError:
        print("profile probe failed: malformed response body (not JSON).")
        return 1
    if not isinstance(profile, dict):
        print("profile probe failed: malformed response body (not an object).")
        return 1
    capabilities = profile.get("capabilities") or {}
    if capabilities.get("isPro"):
        print("plan: pro (unlimited training runs)")
    else:
        remaining = capabilities.get("freeTrainingRunsRemaining")
        limit = capabilities.get("freeTrainingRunsMonthlyLimit")
        quota = f"{remaining}/{limit}" if remaining is not None and limit is not None else "?"
        print(f"plan: free — training runs remaining this month: {quota}")
    if profile.get("email"):
        print(f"account: {profile['email']}")
    print("healthy: token accepted.")
    return 0


def _probe_health(cred: ResolvedCredential, transport: httpx.BaseTransport | None) -> int:
    """Local-key modes (key / desktop_key): the backend itself is the probe."""
    try:
        with httpx.Client(
            base_url=cred.api_url or DEFAULT_API_URL, timeout=10.0, transport=transport
        ) as http:
            resp = http.get("/health")
    except httpx.HTTPError as err:
        print(f"local backend unreachable: {err}. Start the desktop app (or the dev backend).")
        return 1
    if resp.status_code != 200:
        print(f"local backend health: HTTP {resp.status_code} {_error_detail(resp)}.")
        return 1
    print("local backend health: ok.")
    return 0


def _error_detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:200]
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("detail") or detail.get("title") or detail)[:200]
        if detail is not None:
            return str(detail)[:200]
    return str(body)[:200]
