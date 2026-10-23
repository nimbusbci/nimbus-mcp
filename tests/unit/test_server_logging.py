"""Expected setup conditions log one stderr line, not a traceback."""

from __future__ import annotations

import logging

from fastmcp import Client

from nimbus_mcp.client import McpToolError
from nimbus_mcp.server import ExpectedConditionLogFilter, build_server
from nimbus_mcp.setup_mode import SetupRequired, setup_guidance


def _record(exc: BaseException) -> logging.LogRecord:
    import sys

    try:
        raise exc
    except BaseException:  # noqa: BLE001
        info = sys.exc_info()
    return logging.LogRecord(
        "fastmcp.server.server",
        logging.ERROR,
        __file__,
        1,
        "Error calling tool %r",
        ("catalog.templates",),
        info,
    )


def test_setup_required_is_one_line_without_traceback():
    rec = _record(SetupRequired(setup_guidance(message="backend down\nmore")))
    assert ExpectedConditionLogFilter().filter(rec) is True
    assert rec.exc_info is None
    msg = rec.getMessage()
    assert "\n" not in msg and "backend down" in msg and "catalog.templates" in msg
    assert rec.levelno == logging.WARNING


def test_actionable_tool_error_is_one_line():
    rec = _record(McpToolError("bad arg", code="x.y"))
    ExpectedConditionLogFilter().filter(rec)
    assert rec.exc_info is None and "x.y" in rec.getMessage()


def test_real_bug_inside_except_mcp_tool_error_keeps_traceback_at_error():
    """A KeyError raised while handling an McpToolError has it only as
    __context__: that is a real bug, so the traceback and ERROR level stay."""
    import sys

    try:
        try:
            raise McpToolError("backend said no")
        except McpToolError:
            {}["bug"]
    except KeyError:
        info = sys.exc_info()
    rec = logging.LogRecord(
        "fastmcp.server.server",
        logging.ERROR,
        __file__,
        1,
        "Error calling tool %r",
        ("x",),
        info,
    )
    assert ExpectedConditionLogFilter().filter(rec) is True
    assert rec.exc_info is not None
    assert rec.levelno == logging.ERROR and rec.levelname == "ERROR"


def test_unexpected_error_keeps_traceback():
    rec = _record(ZeroDivisionError("boom"))
    ExpectedConditionLogFilter().filter(rec)
    assert rec.exc_info is not None and rec.levelno == logging.ERROR


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def test_live_setup_condition_logs_no_traceback(monkeypatch):
    monkeypatch.setenv("NIMBUS_API_URL", "http://127.0.0.1:9")
    server = build_server()
    logger = logging.getLogger("fastmcp.server.server")
    cap = _Capture()
    logger.addHandler(cap)
    try:
        async with Client(server) as c:
            await c.call_tool("catalog.templates", {}, raise_on_error=False)
    finally:
        logger.removeHandler(cap)
    tool_records = [r for r in cap.records if "Error calling tool" in r.getMessage()]
    assert tool_records, "fastmcp should still log the condition once"
    assert all(r.exc_info is None for r in tool_records)
    assert all("\n" not in r.getMessage() for r in tool_records)
