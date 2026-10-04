import asyncio
import time
from collections.abc import Callable

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import activity, live


@pytest.fixture(autouse=True)
def _reset_watchdog_registry():
    """Keep the process-level watchdog registry isolated between tests."""
    with activity._LOCK:
        activity._SESSIONS.clear()
    yield
    with activity._LOCK:
        activity._SESSIONS.clear()


def make_server(handler, calls=None) -> FastMCP:
    transport = httpx.MockTransport(handler)
    if calls is not None:
        def tracking(req: httpx.Request) -> httpx.Response:
            calls.append(req.url.path)
            return handler(req)
        transport = httpx.MockTransport(tracking)
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=None),  # type: ignore[arg-type]
        transport=transport,
    )
    server = FastMCP("t")
    live.register(server, client)
    return server


async def test_list_devices():
    async with Client(make_server(
        lambda r: httpx.Response(
            200,
            json={"ok": True, "devices": [{"id": "brainbit", "name": "BrainBit", "channels": 4}]},
        )
    )) as c:
        result = await c.call_tool("list_devices", {})
    assert result.data["devices"][0]["id"] == "brainbit"


async def test_test_device_camel_case_body():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(
            200, json={"ok": True, "status": "success", "available": True, "message": "ok"}
        )

    async with Client(make_server(handler)) as c:
        await c.call_tool(
            "test_device", {"device_type": "brainbit", "connection_type": "bluetooth"}
        )
    assert b'"deviceType"' in seen["body"] and b'"connectionType"' in seen["body"]


async def test_start_stream_refuses_without_confirm_and_makes_no_calls():
    calls: list[str] = []
    async with Client(make_server(
        lambda r: httpx.Response(200, json={"ok": True}), calls=calls
    )) as c:
        result = await c.call_tool("start_stream", {"device_type": "brainbit"})
    assert result.data["started"] is False
    assert "confirm" in result.data["message"].lower()
    assert calls == []  # rail: zero HTTP calls without confirm


async def test_start_stream_chains_connect_then_start():
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={"ok": True, "status": "success", "sessionId": "sess-1",
                      "connected": True, "deviceType": "brainbit", "channels": 4},
            )
        if req.url.path == "/api/hardware/start-stream":
            body = req.read()
            assert b'"sessionId"' in body and b'"chunkSize"' in body
            return httpx.Response(
                200, json={"ok": True, "started": True, "sessionId": "sess-1", "updateRateHz": 2.0}
            )
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "start_stream", {"device_type": "brainbit", "confirm": True, "chunk_size": 250}
        )
    assert calls == ["/api/hardware/connect", "/api/hardware/start-stream"]
    assert result.data["started"] is True and result.data["sessionId"] == "sess-1"


async def test_stream_status_query_param():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/hardware/stream-status"
        assert req.url.params["sessionId"] == "sess-1"
        return httpx.Response(
            200,
            json={"ok": True, "running": True, "deviceConnected": True, "sessionId": "sess-1"},
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("stream_status", {"session_id": "sess-1"})
    assert result.data["running"] is True


async def test_stop_stream_stops_then_disconnects_best_effort():
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/api/hardware/stop-stream":
            return httpx.Response(200, json={"ok": True, "stopped": True, "sessionId": "sess-1"})
        if req.url.path == "/api/hardware/disconnect":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("stop_stream", {"session_id": "sess-1"})
    assert calls == ["/api/hardware/stop-stream", "/api/hardware/disconnect"]
    assert result.data["stopped"] is True and result.data["disconnectWarning"] is None


async def test_stream_status_rejects_bad_session_id():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="Invalid session id"):
            await c.call_tool("stream_status", {"session_id": "../evil"})


def _streaming_backend(calls: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    """Handler that accepts connect/start/stream-status/stop/disconnect."""

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        path = req.url.path
        if path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={"ok": True, "status": "success", "sessionId": "sess-1",
                      "connected": True, "deviceType": "brainbit", "channels": 4},
            )
        if path == "/api/hardware/start-stream":
            return httpx.Response(200, json={"ok": True, "started": True, "updateRateHz": 2.0})
        if path == "/api/hardware/stream-status":
            return httpx.Response(200, json={"ok": True, "running": True})
        if path in ("/api/hardware/stop-stream", "/api/hardware/disconnect"):
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(path)

    return handler


async def test_watchdog_stops_idle_session(monkeypatch):
    """(a) idle_timeout_sec=1: no activity for ~2s => stop-stream + disconnect,
    exactly once each, and the session is unregistered afterwards."""
    monkeypatch.setattr(activity, "CHECK_INTERVAL_SEC", 0.5)
    calls: list[str] = []
    async with Client(make_server(_streaming_backend(calls))) as c:
        result = await c.call_tool(
            "start_stream", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 1}
        )
        assert result.data["started"] is True
        time.sleep(2.5)
    assert calls.count("/api/hardware/stop-stream") == 1
    assert calls.count("/api/hardware/disconnect") == 1
    with activity._LOCK:
        assert "sess-1" not in activity._SESSIONS


async def test_watchdog_zero_timeout_never_registers(monkeypatch):
    """(b) idle_timeout_sec=0 disables the watchdog: no registration, no stops."""
    monkeypatch.setattr(activity, "CHECK_INTERVAL_SEC", 0.5)
    calls: list[str] = []
    async with Client(make_server(_streaming_backend(calls))) as c:
        await c.call_tool(
            "start_stream", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 0}
        )
        with activity._LOCK:
            assert "sess-1" not in activity._SESSIONS
        time.sleep(2.5)
    assert "/api/hardware/stop-stream" not in calls
    assert "/api/hardware/disconnect" not in calls


async def test_stream_status_polling_keeps_session_alive(monkeypatch):
    """(c) stream_status notes activity, so a polled session is never stopped."""
    monkeypatch.setattr(activity, "CHECK_INTERVAL_SEC", 0.5)
    calls: list[str] = []
    async with Client(make_server(_streaming_backend(calls))) as c:
        await c.call_tool(
            "start_stream", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 1}
        )
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            await c.call_tool("stream_status", {"session_id": "sess-1"})
            await asyncio.sleep(0.3)
    assert "/api/hardware/stop-stream" not in calls
    assert "/api/hardware/disconnect" not in calls
    with activity._LOCK:
        assert "sess-1" in activity._SESSIONS  # still registered, still alive
