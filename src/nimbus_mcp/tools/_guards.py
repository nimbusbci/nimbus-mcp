"""Shared input guards for tool parameters interpolated into request paths."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

from ..errors import local_tool_error

INVALID_ARGUMENT = "nimbus.mcp.invalid_argument"

#: Fixed-choice vocabularies shared by several tools (E2E r9 N2).
DATA_SOURCES = ("dataset", "upload", "local")
PACK_MODES = ("training", "evaluation", "all")
EXECUTION_STATUSES = ("running", "completed", "failed", "cancelled")

#: Longest id accepted in a URL path segment. Real ids are uuids or short
#: slugs; a 200 KB value otherwise reached httpx as "URL too long" (E2E r8 B2).
MAX_ID_LENGTH = 256


def _preview(value: str, limit: int = 40) -> str:
    """A short, printable rendering of ``value`` for error messages."""
    shown = value if len(value) <= limit else value[:limit] + "…"
    return repr(shown)


def _has_control_chars(value: str) -> bool:
    return any(unicodedata.category(ch) in ("Cc", "Cs") for ch in value)


def safe_segment(value: str, *, label: str = "execution id") -> str:
    """Validate a value used as a single path/URL segment (e.g. execution id)."""
    if not isinstance(value, str) or not value.strip():
        raise local_tool_error(
            f"Invalid {label}: must be a non-empty string.", code=INVALID_ARGUMENT
        )
    if len(value) > MAX_ID_LENGTH:
        raise local_tool_error(
            f"Invalid {label}: too long ({len(value)} chars, max {MAX_ID_LENGTH}).",
            code=INVALID_ARGUMENT,
        )
    # NUL and other control characters reached httpx as a raw "Invalid
    # non-printable ASCII character in URL" (E2E r8 B1). Only reject control
    # (Cc) and surrogate (Cs) code points: str.isprintable() would also refuse
    # NBSP, zero-width joiners in emoji and other legit Unicode in user-named
    # datasets / artifact files.
    if _has_control_chars(value) or value != value.strip():
        raise local_tool_error(
            f"Invalid {label} {_preview(value)}: "
            "has leading/trailing whitespace or control characters.",
            code=INVALID_ARGUMENT,
        )
    # Path(value).name != value also rejects "." and any residual separator trick.
    if "/" in value or "\\" in value or ".." in value or Path(value).name != value:
        raise local_tool_error(f"Invalid {label} {_preview(value)}.", code=INVALID_ARGUMENT)
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


def choice(values: tuple[str, ...], *, optional: bool = False) -> Any:
    """Type alias for a fixed-choice string param that publishes a JSON-schema
    ``enum`` (E2E r9 N2). Validation stays in :func:`require_choice` so a bad
    value gets the stable ``nimbus.mcp.invalid_argument`` code rather than a
    generic schema-validation error."""
    from typing import Annotated

    from pydantic import Field

    enum: list[Any] = list(values) + ([None] if optional else [])
    base: Any = str | None if optional else str
    return Annotated[base, Field(json_schema_extra={"enum": enum})]


def require_choice(
    value: Any, values: tuple[str, ...], *, label: str, optional: bool = False
) -> None:
    """Raise invalid_argument unless ``value`` is one of ``values``."""
    if optional and value is None:
        return
    if value not in values:
        raise local_tool_error(
            f"{label} must be one of {values}; got {value!r}.", code=INVALID_ARGUMENT
        )
