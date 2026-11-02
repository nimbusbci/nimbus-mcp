"""Timestamp normalization for tool outputs (E2E round 10 L2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def iso_utc(value: Any) -> Any:
    """ISO-8601 UTC with ``Z`` for backend timestamps; naive values are UTC.

    The backend stored ``started_at`` without a zone and serialized it with a
    space (``2026-11-02 12:56:53.00891``). Anything unparseable passes through.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
