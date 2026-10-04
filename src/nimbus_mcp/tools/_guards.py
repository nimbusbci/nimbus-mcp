"""Shared input guards for tool parameters interpolated into request paths."""

from __future__ import annotations

from pathlib import Path

from ..client import McpToolError


def safe_segment(value: str, *, label: str = "execution id") -> str:
    """Validate a value used as a single path/URL segment (e.g. execution id)."""
    # Path(value).name != value also rejects "." and any residual separator trick.
    if not value or "/" in value or "\\" in value or ".." in value or Path(value).name != value:
        raise McpToolError(f"Invalid {label} '{value}'.")
    return value
