"""Shared MCP tool-annotation policy (client-visible behavior hints).

MCP clients (Claude, Cursor, …) read these hints to decide which tool calls
need explicit user confirmation. Four policies cover every nimbus tool:

- ``READ_ONLY`` — pure reads: no backend state changes, no local side
  effects, safe to repeat.
- ``LOCAL_WRITE`` — downloads/exports: no backend state changes, but the
  tool writes files into the user's NIMBUS_EXPORT_DIR, so it is not a pure
  read of its environment (readOnlyHint=false); safe to repeat.
- ``MUTATING`` — creates/starts things (compute, uploads, live sessions).
- ``DESTRUCTIVE`` — terminates or overwrites existing work (a running
  execution, a live EEG/protocol session, a saved pipeline doc); clients
  should confirm before calling.
- ``IDEMPOTENT_MUTATING`` — a mutation that is safe to repeat (rename).
"""

from __future__ import annotations

from typing import Any

from fastmcp.tools.function_tool import ToolAnnotations

__all__ = [
    "DESTRUCTIVE",
    "FIGURE_OUTPUT_SCHEMA",
    "IDEMPOTENT_MUTATING",
    "LOCAL_WRITE",
    "MUTATING",
    "READ_ONLY",
]

READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

LOCAL_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

MUTATING = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)

# Mutations where repeating the same call has no further effect (e.g. a
# rename to the same name) (E2E r9 N3).
IDEMPOTENT_MUTATING = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

DESTRUCTIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)

# Shared structured-output schema for the figure tools (plots.*,
# foundation.compare). MCP outputSchema must be an OBJECT schema (spec
# limitation; fastmcp enforces it), so the PNG itself rides an MCP image
# content block the assistant can see, while this schema types the JSON
# sidecar returned alongside as structuredContent (ToolResult).
FIGURE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "title": {"type": ["string", "null"]},
        "mimeType": {"type": "string"},
        "meta": {
            "type": "object",
            "additionalProperties": True,
            "description": (
                "Figure sidecar. foundation.compare includes results/best/skipped "
                "(also as a JSON text content block)."
            ),
        },
        "savedPath": {"type": "string", "description": "Set when save=true."},
        "sizeBytes": {"type": "integer", "description": "Set when save=true."},
    },
    "required": ["ok"],
}
