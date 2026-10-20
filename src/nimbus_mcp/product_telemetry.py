"""Product telemetry for nimbus-mcp (development plan X12) + consent (X11).

DECISION (flagged for review): events go to the EXISTING PostHog project
'Nimbus' (the one Studio/web already capture into) with an ``mcp_`` event
prefix — one product view, MCP adoption readable next to web/Studio
funnels. Events: ``mcp_first_run``, ``mcp_first_local_result``,
``mcp_tool_completed``, ``mcp_login_prompt_shown``, ``mcp_login_completed``,
``mcp_sync_opt_in`` (reserved — no sync surface exists yet to fire it),
``mcp_version_drift_warned``.

Privacy contract (see PRIVACY.md):
- OFF until the user opts in. Consent is asked once, as a
  ``telemetryNotice`` field on the first tool response of a fresh install
  (MCP has no terminal); ``nimbus-mcp telemetry on|off|status`` or
  ``NIMBUS_MCP_TELEMETRY=on|off`` controls it anytime. Local Free works
  fully with telemetry off, and telemetry OFF = zero network calls.
- Identity is a random install UUID (``~/.nimbus/telemetry-state.json``);
  no person profile is created (``$process_person_profile: false``) and
  no account linkage happens until a future opt-in sync.
- IP/geo discarded: ``$ip`` is sent empty and ``$geoip_disable`` set.
- NEVER collected: dataset contents, file paths/names, upload metadata,
  user code, free-text error messages. Errors travel as stable CODES
  only, mapped from the structured error types.
- Fire-and-forget: sends happen on a daemon thread with a 2 s timeout;
  failures are swallowed. No SDK is constructed (raw capture POSTs), so
  nothing phones home before consent — there is no flag fetching at all.

Config: the public PostHog project key + host ship as client config with
env overrides (``NIMBUS_MCP_POSTHOG_KEY`` / ``NIMBUS_MCP_POSTHOG_HOST``).
``DEFAULT_POSTHOG_KEY`` is the PUBLIC client-side token of the existing
PostHog project 'Nimbus' (id 275296) — safe to ship, it only permits event
capture. Host is US (``https://us.i.posthog.com``), confirmed for that
project. Shipping the key changes nothing before consent: capture still
requires an explicit opt-in.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__

DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"
DEFAULT_POSTHOG_KEY = "phc_zmAy1HGn7pGiytU6eOoX1zMD0rSd9WmT1NWOymhmxK9"  # public project token

EVENTS = (
    "first_run",
    "first_local_result",
    "tool_completed",
    "login_prompt_shown",
    "login_completed",
    "sync_opt_in",
    "version_drift_warned",
)

_CONSENT_GRANTED = "granted"
_CONSENT_DENIED = "denied"


def _state_path() -> Path:
    override = os.environ.get("NIMBUS_MCP_TELEMETRY_FILE")
    if override:
        return Path(override)
    return Path.home() / ".nimbus" / "telemetry-state.json"


def _env_override() -> str | None:
    raw = os.environ.get("NIMBUS_MCP_TELEMETRY", "").strip().lower()
    if raw in ("on", "off"):
        return raw
    return None


def duration_bucket(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 5:
        return "1-5s"
    if seconds < 30:
        return "5-30s"
    return "30s+"


class Telemetry:
    """Consent-gated, fire-and-forget PostHog capture (see module docstring)."""

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        sender: Callable[[dict[str, Any]], None] | None = None,
        environ: dict[str, str] | None = None,
        disabled: bool = False,
    ) -> None:
        self._state_path = state_path or _state_path()
        self._sender = sender or self._default_sender
        self._environ = os.environ if environ is None else environ
        self._disabled = disabled
        self._lock = threading.RLock()
        self._state: dict[str, Any] | None = None

    @classmethod
    def disabled(cls) -> Telemetry:
        """A permanently-off instance (hosted gateway: no install identity)."""
        return cls(disabled=True)

    # -- state ---------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Load (or create) state. Race-safe: an RLock serializes threads,
        and a fresh install claims its UUID with an exclusive create, so a
        concurrent process that lost the race adopts the winner's id."""
        with self._lock:
            if self._state is not None:
                return self._state
            state = self._read_state()
            if not state.get("install_id"):
                state = {**state, "install_id": uuid.uuid4().hex}
                if not self._exclusive_create(state):
                    winner = self._read_state()
                    if winner.get("install_id"):
                        state = winner
            self._state = state
            return self._state

    def _read_state(self) -> dict[str, Any]:
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - unreadable state = fresh install
            return {}
        return raw if isinstance(raw, dict) else {}

    def _exclusive_create(self, state: dict[str, Any]) -> bool:
        """True if we created the file; False if it already existed.

        Content is written to a private temp file first and then hard-linked
        into place: ``os.link`` fails if the target exists, so the claim is
        atomic AND a loser never observes a half-written winner file.
        """
        tmp = self._state_path.with_name(
            f"{self._state_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(state), encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.link(tmp, self._state_path)
            return True
        except FileExistsError:
            return False
        except Exception:  # noqa: BLE001 - unwritable home: keep in-memory id
            return True
        finally:
            try:
                tmp.unlink()
            except Exception:  # noqa: BLE001 - best-effort temp cleanup
                pass

    def _save(self) -> None:
        """Atomic replace so readers never see a torn file."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(
                f"{self._state_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            tmp.write_text(json.dumps(self._state, indent=1), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except Exception:  # noqa: BLE001 - telemetry must never break tools
            pass

    # -- consent ---------------------------------------------------------------

    def consent(self) -> str:
        """'granted' | 'denied' | 'unknown' (env override wins)."""
        if self._disabled:
            return _CONSENT_DENIED
        env = _env_override() if self._environ is os.environ else (
            "on" if self._environ.get("NIMBUS_MCP_TELEMETRY", "").strip().lower() == "on"
            else "off" if self._environ.get("NIMBUS_MCP_TELEMETRY", "").strip().lower() == "off"
            else None
        )
        if env == "off":
            return _CONSENT_DENIED
        if env == "on":
            return _CONSENT_GRANTED
        with self._lock:
            value = self._load().get("consent")
        return str(value) if value in (_CONSENT_GRANTED, _CONSENT_DENIED) else "unknown"

    def set_consent(self, granted: bool) -> None:
        with self._lock:
            self._load()["consent"] = _CONSENT_GRANTED if granted else _CONSENT_DENIED
            self._save()
        if granted:
            self.capture("first_run")

    def enabled(self) -> bool:
        """Capture allowed: consent granted AND a project key configured."""
        return self.consent() == _CONSENT_GRANTED and bool(self._api_key())

    def _api_key(self) -> str:
        return self._environ.get("NIMBUS_MCP_POSTHOG_KEY", "") or DEFAULT_POSTHOG_KEY

    def _host(self) -> str:
        return self._environ.get("NIMBUS_MCP_POSTHOG_HOST", "") or DEFAULT_POSTHOG_HOST

    # -- notice (X11 consent surfacing) -----------------------------------------

    def pending_notice(self) -> dict[str, Any] | None:
        """The one-time consent notice for the first tool response, or None.

        Shown only while consent is undecided and no env control is set;
        marked shown exactly once (persisted), whatever the tool outcome.
        """
        if self._disabled or _env_override() is not None:
            return None
        with self._lock:
            state = self._load()
            if state.get("consent") in (_CONSENT_GRANTED, _CONSENT_DENIED):
                return None
            if state.get("notice_shown"):
                return None
            state["notice_shown"] = True
            self._save()
        return {
            "message": (
                "Anonymous usage telemetry is OFF. nimbus-mcp can send "
                "anonymous, content-free usage events (tool names, "
                "success/failure and error codes, duration buckets — never "
                "your data, file paths/names, or code) to PostHog to improve "
                "the tools. Enable with `nimbus-mcp telemetry on` (or "
                "NIMBUS_MCP_TELEMETRY=on); refuse or revoke anytime with "
                "`nimbus-mcp telemetry off`. Everything local works fully "
                "either way. Statement: PRIVACY.md in the nimbus-mcp package."
            ),
            "telemetry": "off-until-consent",
            "control": "nimbus-mcp telemetry on|off|status",
        }

    # -- capture -----------------------------------------------------------------

    def _default_sender(self, payload: dict[str, Any]) -> None:
        import httpx

        httpx.post(
            f"{self._host()}/capture/",
            json=payload,
            timeout=2.0,
        )

    def capture(self, event: str, properties: dict[str, Any] | None = None) -> None:
        """Fire-and-forget; returns immediately, swallows every failure."""
        if event not in EVENTS:
            return
        try:
            if not self.enabled():
                return
            with self._lock:
                state = self._load()
                if event != "first_run" and not state.get("first_run_sent"):
                    state["first_run_sent"] = True
                    self._save()
                    self._dispatch("first_run", {})
                if event == "first_run":
                    if state.get("first_run_sent"):
                        return
                    state["first_run_sent"] = True
                    self._save()
                if event == "first_local_result":
                    if state.get("first_local_result_sent"):
                        return
                    state["first_local_result_sent"] = True
                    self._save()
            self._dispatch(event, properties or {})
        except Exception:  # noqa: BLE001 - telemetry must never break tools
            pass

    def _dispatch(self, event: str, properties: dict[str, Any]) -> None:
        props: dict[str, Any] = {
            "$ip": "",
            "$geoip_disable": True,
            "$process_person_profile": False,
            "client_version": __version__,
            "os_family": platform.system().lower(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        props.update(properties)
        payload = {
            "api_key": self._api_key(),
            "event": f"mcp_{event}",
            "distinct_id": str(self._load().get("install_id")),
            "properties": props,
        }
        thread = threading.Thread(
            target=self._safe_send, args=(payload,), daemon=True
        )
        thread.start()

    def _safe_send(self, payload: dict[str, Any]) -> None:
        try:
            self._sender(payload)
        except Exception:  # noqa: BLE001 - fire-and-forget
            pass

    # -- semantic helpers used by the middleware ---------------------------------

    def tool_completed(
        self,
        *,
        tool: str,
        mode: str,
        success: bool,
        error_code: str | None,
        duration_s: float,
    ) -> None:
        if success and mode in ("anonymous", "local"):
            self.capture("first_local_result", {"mode": mode})
        self.capture(
            "tool_completed",
            {
                "tool": tool,
                "mode": mode,
                "success": success,
                "error_code": error_code,
                "duration_bucket": duration_bucket(duration_s),
            },
        )

    def login_prompt_shown(self, *, trigger: str, mode: str) -> None:
        self.capture("login_prompt_shown", {"trigger": trigger, "mode": mode})

    def login_completed(self, *, trigger: str) -> None:
        self.capture("login_completed", {"trigger": trigger})

    def version_drift_warned(self, *, current: str, latest: str, mode: str) -> None:
        self.capture(
            "version_drift_warned",
            {"client_version": current, "latest_version": latest, "mode": mode},
        )


_SINGLETON: Telemetry | None = None


def get_telemetry() -> Telemetry:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Telemetry()
    return _SINGLETON


def reset_telemetry_for_tests() -> None:
    global _SINGLETON
    _SINGLETON = None
