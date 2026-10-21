"""Shared error types for nimbus-mcp.

Lives outside ``client``/``config`` so both can raise it without an import
cycle; ``client`` re-exports it for the established ``from ..client import
McpToolError`` import path used by the tools.
"""

from __future__ import annotations

import logging
import uuid

logger = logging.getLogger("nimbus_mcp.errors")


class McpToolError(Exception):
    """Actionable error surfaced to the MCP client (agent-visible).

    Backend HTTP failures additionally carry the response ``status_code`` and
    the backend's stable dotted ``code`` (e.g.
    ``nimbus.data.describe_local_forbidden``) as attributes, so tools can
    branch on them programmatically instead of parsing message text. The
    optional ``correlation_id`` is the backend's per-request id (also in the
    message) to quote in support reports.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.correlation_id = correlation_id


def local_tool_error(
    message: str,
    *,
    code: str,
    status_code: int | None = None,
) -> McpToolError:
    """Build an error raised by nimbus-mcp itself (no backend request was made).

    Backend errors carry the backend's ``correlationId``; a locally raised
    error has none, so support could not tie a user report to a log line.
    Generate one here (uuid4 hex), append it in the same
    ``(code: …, correlationId: …)`` shape the backend path uses, and log it so
    the id appears in the MCP server's own logs.
    """
    correlation_id = uuid.uuid4().hex
    logger.warning(
        "Local tool error (code=%s, status=%s, correlationId=%s): %s",
        code,
        status_code,
        correlation_id,
        message,
    )
    return McpToolError(
        f"{message} (code: {code}, correlationId: {correlation_id})",
        status_code=status_code,
        code=code,
        correlation_id=correlation_id,
    )
