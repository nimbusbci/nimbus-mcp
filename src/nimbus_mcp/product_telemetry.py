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

# Tools whose success is a REAL local result (execution/plot/stats/export) —
# the only ones that may fire ``mcp_first_local_result`` (B3). Discovery,
# whoami, validation and listing calls succeed without producing a result.
RESULT_TOOLS = frozenset(
    {
        "execution.results",
        "execution.download_artifact",
        "experiment.get",
        "plots.confusion",
        "plots.dataset",
        "plots.erp",
        "plots.leaderboard",
        "stats.grouped",
        "foundation.compare",
        "bids.export_dataset",
        "bids.export_execution",
        "pipeline.export",
        "protocol.export_stats",
    }
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
        self._first_run_inflight = False
        # One mcp_first_run attempt per process: a failed send is retried by
        # the next process (first_run_sent stays False), not on every tool call.
        self._first_run_attempted = False
        self._first_local_result_attempted = False
        self._state: dict[str, Any] | None = None

    @classmethod
    def disabled(cls) -> Telemetry:
        """A permanently-off instance (hosted gateway: no install identity)."""
        return cls(disabled=True)

    # -- state ---------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Current state, RE-READ from disk on every call (H1).

        The file is the source of truth shared with other processes: a
        long-running server must observe ``nimbus-mcp telemetry off`` (and
        ``*_sent`` flags written by a sibling process) instead of trusting a
        cached copy. The in-memory copy is only a fallback for an unreadable
        or unwritable home. Race-safe first install: an RLock serializes
        threads, and a fresh install claims its UUID with an exclusive
        create, so a concurrent process that lost the race adopts the
        winner's id."""
        with self._lock:
            state = self._read_state()
            if not state.get("install_id"):
                if self._state is not None and self._state.get("install_id"):
                    # Unreadable/unwritable home: keep this process's view.
                    return self._state
                state = {**state, "install_id": uuid.uuid4().hex}
                if not self._exclusive_create(state):
                    winner = self._read_state()
                    if winner.get("install_id"):
                        state = winner
            self._state = state
            return self._state

    def _update(self, **changes: Any) -> None:
        """Merge-on-save (H1): re-read the file under the lock and apply ONLY
        ``changes`` — never write back cached keys (notably ``consent``) that
        another process may have changed since this one last looked."""
        with self._lock:
            state = dict(self._load())
            state.update(changes)
            self._state = state
            self._save()

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
        env = (
            _env_override()
            if self._environ is os.environ
            else (
                "on"
                if self._environ.get("NIMBUS_MCP_TELEMETRY", "").strip().lower() == "on"
                else "off"
                if self._environ.get("NIMBUS_MCP_TELEMETRY", "").strip().lower() == "off"
                else None
            )
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
            self._update(consent=_CONSENT_GRANTED if granted else _CONSENT_DENIED)
        if granted:
            # Synchronous: the CLI exits right after this call (B1).
            self.capture("first_run", sync=True)

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
            self._update(notice_shown=True)
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

        resp = httpx.post(
            f"{self._host()}/capture/",
            json=payload,
            timeout=2.0,
        )
        # Delivery means a 2xx from the capture endpoint. A 4xx/5xx raises, so
        # _safe_send reports delivered=False and one-shot events (first_run,
        # first_local_result) stay unsent and are retried by the next process.
        resp.raise_for_status()

    def capture(
        self,
        event: str,
        properties: dict[str, Any] | None = None,
        *,
        sync: bool = False,
    ) -> None:
        """Fire-and-forget; returns immediately, swallows every failure.

        ``sync=True`` sends inline (bounded by the sender's 2 s timeout) for
        short-lived processes — the ``nimbus-mcp telemetry on`` CLI would
        otherwise exit before a daemon-thread send completes (B1).

        ``first_run_sent`` is persisted only AFTER ``mcp_first_run`` was
        delivered: a lost send is retried by the next capture/session rather
        than silently marked done.
        """
        if event not in EVENTS:
            return
        try:
            if not self.enabled():
                return
            with self._lock:
                state = self._load()
                need_first_run = (
                    not state.get("first_run_sent")
                    and not self._first_run_inflight
                    and not self._first_run_attempted
                )
                if need_first_run:
                    self._first_run_inflight = True
                    self._first_run_attempted = True
                if event == "first_local_result":
                    if state.get("first_local_result_sent") or self._first_local_result_attempted:
                        return
                    # Marked sent only on a 2xx (see _first_local_result_done);
                    # one attempt per process, retried by the next one.
                    self._first_local_result_attempted = True
            if need_first_run:
                self._dispatch("first_run", {}, sync=sync, on_done=self._first_run_done)
            if event == "first_local_result":
                self._dispatch(
                    event, properties or {}, sync=sync, on_done=self._first_local_result_done
                )
            elif event != "first_run":
                self._dispatch(event, properties or {}, sync=sync)
        except Exception:  # noqa: BLE001 - telemetry must never break tools
            pass

    def _first_local_result_done(self, delivered: bool) -> None:
        if not delivered:
            return
        with self._lock:
            self._update(first_local_result_sent=True)

    def _first_run_done(self, delivered: bool) -> None:
        with self._lock:
            self._first_run_inflight = False
            if delivered:
                self._update(first_run_sent=True)

    def _dispatch(
        self,
        event: str,
        properties: dict[str, Any],
        *,
        sync: bool = False,
        on_done: Callable[[bool], None] | None = None,
    ) -> None:
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
        if sync:
            self._safe_send(payload, on_done)
            return
        thread = threading.Thread(target=self._safe_send, args=(payload, on_done), daemon=True)
        thread.start()

    def _safe_send(
        self,
        payload: dict[str, Any],
        on_done: Callable[[bool], None] | None = None,
    ) -> None:
        delivered = False
        try:
            self._sender(payload)
            delivered = True
        except Exception:  # noqa: BLE001 - fire-and-forget
            pass
        if on_done is not None:
            try:
                on_done(delivered)
            except Exception:  # noqa: BLE001
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
        if success and mode in ("anonymous", "local") and tool in RESULT_TOOLS:
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
