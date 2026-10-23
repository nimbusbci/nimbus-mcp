"""429 queue_full surfaces retryAfter to the agent (E2E round 2, 2026-10-21)."""

from __future__ import annotations

import httpx

from nimbus_mcp.client import _retry_after_seconds


def test_retry_after_from_body_or_header():
    assert _retry_after_seconds({"retryAfter": 15}, None) == 15
    assert _retry_after_seconds({}, "7") == 7
    assert _retry_after_seconds({}, "Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert _retry_after_seconds(None, None) is None


def test_error_message_includes_retry_after():
    from nimbus_mcp.client import NimbusClient

    resp = httpx.Response(
        429,
        json={
            "detail": "All 3 concurrent execution slots are busy.",
            "code": "nimbus.execution.queue_full",
            "retryAfter": 15,
            "correlationId": "abc123",
        },
        headers={"Retry-After": "15"},
    )
    err = NimbusClient._backend_error(resp)
    assert err.retry_after == 15 and err.status_code == 429
    assert "retryAfter: 15s" in str(err) and "nimbus.execution.queue_full" in str(err)
