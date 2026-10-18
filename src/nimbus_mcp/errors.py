"""Shared error types for nimbus-mcp.

Lives outside ``client``/``config`` so both can raise it without an import
cycle; ``client`` re-exports it for the established ``from ..client import
McpToolError`` import path used by the tools.
"""


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
