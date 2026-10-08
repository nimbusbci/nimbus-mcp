"""Shared MCP tool-annotation policy (client-visible behavior hints).

MCP clients (Claude, Cursor, …) read these hints to decide which tool calls
need explicit user confirmation. Four policies cover every nimbus tool:

- ``READ_ONLY`` — pure reads: no backend state changes, no local side
  effects, safe to repeat.
- ``LOCAL_WRITE`` — downloads/exports: no backend state changes, but the
  tool writes files into the user's NIMBUS_EXPORT_DIR, so it is not a pure
  read of its environment (readOnlyHint=false); safe to repeat.
- ``MUTATING`` — creates/starts things (compute, uploads, live sessions).
- ``DESTRUCTIVE`` — terminates existing work (a running execution, a live
  EEG session); clients should confirm before calling.
"""

from __future__ import annotations

from fastmcp.tools.function_tool import ToolAnnotations

__all__ = ["DESTRUCTIVE", "LOCAL_WRITE", "MUTATING", "READ_ONLY"]

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

DESTRUCTIVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
    open_world_hint=True,
)
