"""Locally raised coded errors carry (and log) a generated correlationId."""

from __future__ import annotations

import logging
import re

from nimbus_mcp.errors import McpToolError, local_tool_error


def test_local_tool_error_shape_and_attrs() -> None:
    err = local_tool_error("Nope.", code="nimbus.x.y", status_code=403)
    assert isinstance(err, McpToolError)
    assert err.code == "nimbus.x.y" and err.status_code == 403
    assert re.fullmatch(r"[0-9a-f]{32}", err.correlation_id or "")
    assert str(err) == f"Nope. (code: nimbus.x.y, correlationId: {err.correlation_id})"


def test_local_tool_error_ids_are_unique() -> None:
    a = local_tool_error("x", code="c")
    b = local_tool_error("x", code="c")
    assert a.correlation_id != b.correlation_id


def test_local_tool_error_is_logged(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="nimbus_mcp.errors"):
        err = local_tool_error("Scope missing.", code="nimbus.auth.token_scope_insufficient")
    assert err.correlation_id in caplog.text
    assert "nimbus.auth.token_scope_insufficient" in caplog.text
