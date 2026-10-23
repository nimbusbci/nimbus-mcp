"""Shared input guards for tool parameters interpolated into request paths."""

from __future__ import annotations

from pathlib import Path

from ..errors import local_tool_error

INVALID_ARGUMENT = "nimbus.mcp.invalid_argument"


def safe_segment(value: str, *, label: str = "execution id") -> str:
    """Validate a value used as a single path/URL segment (e.g. execution id)."""
    # Path(value).name != value also rejects "." and any residual separator trick.
    if not value or "/" in value or "\\" in value or ".." in value or Path(value).name != value:
        raise local_tool_error(f"Invalid {label} '{value}'.", code=INVALID_ARGUMENT)
    return value


def safe_upload_path(value: str, *, label: str = "upload path") -> str:
    """Validate a relative upload path (may contain ``/``, never ``..`` or abs)."""
    cleaned = (value or "").replace("\\", "/").strip()
    if (
        not cleaned
        or cleaned.startswith("/")
        or ".." in cleaned.split("/")
        or cleaned.startswith("~")
    ):
        raise local_tool_error(f"Invalid {label} '{value}'.", code=INVALID_ARGUMENT)
    return cleaned
