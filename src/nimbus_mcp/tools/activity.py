"""Process-level idle-session watchdog shared by live and telemetry tools.

A streaming session registered here is stopped (stop-stream + disconnect via
the caller's ``stop_fn``) when no activity has been noted for ``timeout_sec``.
Activity is noted by ``stream_status`` and ``get_live_session`` polls, so any
agent (or human UI) actively watching a session keeps it alive; an abandoned
one is shut down instead of streaming from the user's head indefinitely.

``CHECK_INTERVAL_SEC`` is read fresh on every checker cycle, so tests can
shrink it with monkeypatch.setattr(activity, "CHECK_INTERVAL_SEC", 0.5).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

CHECK_INTERVAL_SEC = 15.0


@dataclass
class _WatchState:
    stop_fn: Callable[[], object]
    timeout_sec: float
    last_activity: float = field(default_factory=time.monotonic)
    stop_called: bool = False


_LOCK = threading.Lock()
_SESSIONS: dict[str, _WatchState] = {}
_WAKE = threading.Event()
_CHECKER_THREAD: threading.Thread | None = None


def register_started(session_id: str, stop_fn: Callable[[], object], timeout_sec: float) -> None:
    """Track a live session; timeout_sec <= 0 disables the watchdog for it."""
    if timeout_sec is None or timeout_sec <= 0:
        return
    with _LOCK:
        _SESSIONS[session_id] = _WatchState(
            stop_fn=stop_fn, timeout_sec=float(timeout_sec)
        )
    # Wake the checker so a fresh registration is observed promptly (and so a
    # checker mid-sleep on a stale interval re-reads CHECK_INTERVAL_SEC).
    _WAKE.set()
    _ensure_checker()


def note_activity(session_id: str) -> None:
    """Record that someone is still watching the session (no-op if unknown)."""
    with _LOCK:
        state = _SESSIONS.get(session_id)
        if state is not None:
            state.last_activity = time.monotonic()


def unregister(session_id: str) -> None:
    """Drop a session from the watchdog (idempotent; used by stop_stream)."""
    with _LOCK:
        _SESSIONS.pop(session_id, None)


def _ensure_checker() -> None:
    global _CHECKER_THREAD
    with _LOCK:
        if _CHECKER_THREAD is not None and _CHECKER_THREAD.is_alive():
            return
        _CHECKER_THREAD = threading.Thread(
            target=_checker_loop, name="nimbus-idle-watchdog", daemon=True
        )
        _CHECKER_THREAD.start()


def _checker_loop() -> None:
    while True:
        _WAKE.wait(CHECK_INTERVAL_SEC)
        _WAKE.clear()
        _check_once()


def _check_once() -> None:
    now = time.monotonic()
    expired: list[tuple[str, _WatchState]] = []
    with _LOCK:
        for session_id, state in list(_SESSIONS.items()):
            if not state.stop_called and now - state.last_activity > state.timeout_sec:
                # Flip the guard under the lock so a concurrent check (or a
                # second checker thread) can never double-stop.
                state.stop_called = True
                expired.append((session_id, state))
    for session_id, state in expired:
        try:
            state.stop_fn()  # best-effort: stop-stream then disconnect
        except Exception:
            pass  # the session is abandoned either way; never crash the checker
        finally:
            unregister(session_id)
