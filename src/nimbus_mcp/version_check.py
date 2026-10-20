"""Startup version check (X11; gateway-drift lesson from #519).

Non-blocking: a daemon thread asks PyPI for the latest ``nimbus-mcp``
release (2 s timeout, result cached under ``~/.nimbus`` for 24 h). When
the installed client is behind, the next tool response carries a
``versionNotice`` field (once per process) and ``mcp_version_drift_warned``
is captured (if telemetry is consented). Silent when current, when the
cache is fresh, or when offline — never a startup failure, and Local
Free is never blocked by it. ``NIMBUS_MCP_VERSION_CHECK=off`` disables it
(no PyPI request at all).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__

PYPI_URL = "https://pypi.org/pypi/nimbus-mcp/json"
CACHE_TTL_SECONDS = 24 * 3600

_started = False
_lock = threading.Lock()
_notice: dict[str, Any] | None = None
_notice_consumed = False


def _cache_path() -> Path:
    override = os.environ.get("NIMBUS_MCP_VERSION_CHECK_FILE")
    if override:
        return Path(override)
    return Path.home() / ".nimbus" / "version-check.json"


def _parse(version: str) -> tuple[int, ...]:
    parts = []
    for piece in str(version).replace("-", ".").split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def _default_fetcher() -> str:
    import httpx

    resp = httpx.get(PYPI_URL, timeout=2.0)
    resp.raise_for_status()
    return str(resp.json()["info"]["version"])


def _check(fetcher: Callable[[], str], cache_path: Path) -> None:
    global _notice
    try:
        latest: str | None = None
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if time.time() - float(cache.get("checked_at", 0)) < CACHE_TTL_SECONDS:
                latest = str(cache.get("latest") or "") or None
        except Exception:  # noqa: BLE001 - no/stale cache -> fetch below
            latest = None
        if latest is None:
            latest = fetcher()
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({"checked_at": time.time(), "latest": latest}),
                    encoding="utf-8",
                )
            except Exception:  # noqa: BLE001 - cache is best-effort
                pass
        if latest and _parse(latest) > _parse(__version__):
            with _lock:
                _notice = {
                    "type": "version_drift",
                    "current": __version__,
                    "latest": latest,
                    "message": (
                        f"nimbus-mcp {latest} is available (you run "
                        f"{__version__}). Older clients can drift from the "
                        "hosted gateway's tool surface; upgrade with "
                        "`uvx nimbus-mcp@latest` or `pip install -U nimbus-mcp`."
                    ),
                }
    except Exception:  # noqa: BLE001 - offline/timeout: silent by design
        pass


def start_background_check(
    *,
    fetcher: Callable[[], str] | None = None,
    cache_path: Path | None = None,
) -> None:
    """Kick the one-shot check on a daemon thread (idempotent per process)."""
    global _started
    if os.environ.get("NIMBUS_MCP_VERSION_CHECK", "").strip().lower() == "off":
        return  # explicit opt-out: no PyPI request at all
    with _lock:
        if _started:
            return
        _started = True
    thread = threading.Thread(
        target=_check,
        args=(fetcher or _default_fetcher, cache_path or _cache_path()),
        daemon=True,
    )
    thread.start()


def pending_notice() -> dict[str, Any] | None:
    """The drift notice, once per process (None when current/offline)."""
    global _notice_consumed
    with _lock:
        if _notice is None or _notice_consumed:
            return None
        _notice_consumed = True
        return dict(_notice)


def reset_for_tests() -> None:
    global _started, _notice, _notice_consumed
    with _lock:
        _started = False
        _notice = None
        _notice_consumed = False
