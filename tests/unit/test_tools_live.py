import asyncio
import threading
import time
from collections.abc import Callable

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import HeaderCredentialClient
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


def make_header_server(handler) -> FastMCP:
    """live server behind a HeaderCredentialClient (hosted gateway) whose
    header getter answers everywhere EXCEPT the idle-watchdog checker thread
    (where fastmcp reports no request — the production failure mode: the
    checker fires the stop callback outside any MCP request context)."""

    def getter() -> dict[str, str]:
        if threading.current_thread().name == "nimbus-idle-watchdog":
            return {}
        return {"x-nimbus-token": "t1"}

    def factory(cfg):
        return NimbusClient(cfg, transport=httpx.MockTransport(handler))

    proxy = HeaderCredentialClient(
        McpConfig(api_url="http://t", mcp_key="", export_dir=None),  # type: ignore[arg-type]
        client_factory=factory,
        headers_getter=getter,
    )
    server = FastMCP("t")
    live.register(server, proxy)
    return server


async def test_list_devices():
    async with Client(
        make_server(
            lambda r: httpx.Response(
                200,
                json={
                    "ok": True,
                    "devices": [{"id": "brainbit", "name": "BrainBit", "channels": 4}],
                },
            )
        )
    ) as c:
        result = await c.call_tool("device.list", {})
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
            "device.test", {"device_type": "brainbit", "connection_type": "bluetooth"}
        )
    assert b'"deviceType"' in seen["body"] and b'"connectionType"' in seen["body"]


async def test_start_stream_refuses_without_confirm_and_makes_no_calls():
    calls: list[str] = []
    async with Client(
        make_server(lambda r: httpx.Response(200, json={"ok": True}), calls=calls)
    ) as c:
        result = await c.call_tool("stream.start", {"device_type": "brainbit"})
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
                json={
                    "ok": True,
                    "status": "success",
                    "sessionId": "sess-1",
                    "connected": True,
                    "deviceType": "brainbit",
                    "channels": 4,
                },
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
            "stream.start", {"device_type": "brainbit", "confirm": True, "chunk_size": 250}
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
        result = await c.call_tool("stream.status", {"session_id": "sess-1"})
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
        result = await c.call_tool("stream.stop", {"session_id": "sess-1"})
    assert calls == ["/api/hardware/stop-stream", "/api/hardware/disconnect"]
    assert result.data["stopped"] is True and result.data["disconnectWarning"] is None


async def test_stream_status_rejects_bad_session_id():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="Invalid session id"):
            await c.call_tool("stream.status", {"session_id": "../evil"})


async def test_start_stream_disconnects_when_start_stream_fails():
    """start-stream failing AFTER a successful connect must not leave the
    device connected with no handle: the tool best-effort disconnects the
    just-created session and says so in the error (with the sessionId for
    reference)."""
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "status": "success",
                    "sessionId": "sess-1",
                    "connected": True,
                    "deviceType": "brainbit",
                    "channels": 4,
                },
            )
        if req.url.path == "/api/hardware/start-stream":
            return httpx.Response(404, json={"detail": "device dropped between calls"})
        if req.url.path == "/api/hardware/disconnect":
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="sess-1 was disconnected"):
            await c.call_tool("stream.start", {"device_type": "brainbit", "confirm": True})
    assert calls == [
        "/api/hardware/connect",
        "/api/hardware/start-stream",
        "/api/hardware/disconnect",
    ]


async def test_start_stream_failed_disconnect_failure_names_recovery_path():
    """If even the cleanup disconnect fails, the error must carry the
    sessionId and the stream.stop recovery hint instead of a silent orphaned
    device."""
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if req.url.path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={"ok": True, "status": "success", "sessionId": "sess-1", "connected": True},
            )
        if req.url.path == "/api/hardware/start-stream":
            return httpx.Response(500, json={"detail": "stream start failed"})
        if req.url.path == "/api/hardware/disconnect":
            return httpx.Response(500, json={"detail": "disconnect failed too"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="stream.stop"):
            await c.call_tool("stream.start", {"device_type": "brainbit", "confirm": True})
    assert calls.count("/api/hardware/disconnect") == 1


def _streaming_backend(calls: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    """Handler that accepts connect/start/stream-status/stop/disconnect."""

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        path = req.url.path
        if path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "status": "success",
                    "sessionId": "sess-1",
                    "connected": True,
                    "deviceType": "brainbit",
                    "channels": 4,
                },
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
            "stream.start", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 1}
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
            "stream.start", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 0}
        )
        with activity._LOCK:
            assert "sess-1" not in activity._SESSIONS
        time.sleep(2.5)
    assert "/api/hardware/stop-stream" not in calls
    assert "/api/hardware/disconnect" not in calls


async def test_stream_status_polling_keeps_session_alive(monkeypatch):
    """(c) stream.status notes activity, so a polled session is never stopped."""
    monkeypatch.setattr(activity, "CHECK_INTERVAL_SEC", 0.5)
    calls: list[str] = []
    async with Client(make_server(_streaming_backend(calls))) as c:
        await c.call_tool(
            "stream.start", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 1}
        )
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            await c.call_tool("stream.status", {"session_id": "sess-1"})
            await asyncio.sleep(0.3)
    assert "/api/hardware/stop-stream" not in calls
    assert "/api/hardware/disconnect" not in calls
    with activity._LOCK:
        assert "sess-1" in activity._SESSIONS  # still registered, still alive


async def test_watchdog_on_gateway_stops_via_request_bound_client(monkeypatch):
    """(hosted regression) the checker thread fires the stop callback with no
    MCP request context, so per-request token resolution cannot run there:
    stream.start must capture a request-bound client (X-Nimbus-Token: t1 →
    Authorization: Bearer t1) for the watchdog, and its stop+disconnect must
    arrive authenticated as that user instead of being swallowed as a
    SetupRequired by the checker's catch-all."""
    monkeypatch.setattr(activity, "CHECK_INTERVAL_SEC", 0.5)
    calls: list[str] = []
    auth_headers: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        calls.append(path)
        if path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "status": "success",
                    "sessionId": "sess-gw",
                    "connected": True,
                    "deviceType": "brainbit",
                    "channels": 4,
                },
            )
        if path == "/api/hardware/start-stream":
            return httpx.Response(
                200, json={"ok": True, "started": True, "sessionId": "sess-gw", "updateRateHz": 2.0}
            )
        if path in ("/api/hardware/stop-stream", "/api/hardware/disconnect"):
            auth_headers.append(req.headers.get("authorization", ""))
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(path)

    async with Client(make_header_server(handler)) as c:
        result = await c.call_tool(
            "stream.start",
            {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 1},
        )
        assert result.data["started"] is True
        time.sleep(2.5)
    assert calls.count("/api/hardware/stop-stream") == 1
    assert calls.count("/api/hardware/disconnect") == 1
    # the regression: the watchdog's requests authenticate as the STARTING
    # request's user, not as the credential-less gateway process
    assert auth_headers == ["Bearer t1", "Bearer t1"]


async def test_stream_stop_rejected_keeps_watchdog_armed(monkeypatch):
    """A stop the backend rejects (403: another user's session) must NOT
    disarm the idle watchdog — the session may still be streaming, and the
    error has to propagate to the caller. Unregister only happens once the
    backend accepts the stop."""
    monkeypatch.setattr(activity, "CHECK_INTERVAL_SEC", 0.5)
    state = {"reject_stop": False}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "status": "success",
                    "sessionId": "sess-1",
                    "connected": True,
                    "deviceType": "brainbit",
                    "channels": 4,
                },
            )
        if path == "/api/hardware/start-stream":
            return httpx.Response(200, json={"ok": True, "started": True, "updateRateHz": 2.0})
        if path == "/api/hardware/stop-stream" and state["reject_stop"]:
            return httpx.Response(403, json={"detail": "session belongs to another user"})
        if path in ("/api/hardware/stop-stream", "/api/hardware/disconnect"):
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(path)

    async with Client(make_server(handler)) as c:
        await c.call_tool(
            "stream.start", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 60}
        )
        with activity._LOCK:
            assert "sess-1" in activity._SESSIONS  # armed by stream.start
        state["reject_stop"] = True
        with pytest.raises(Exception, match="403"):
            await c.call_tool("stream.stop", {"session_id": "sess-1"})
        with activity._LOCK:
            assert "sess-1" in activity._SESSIONS  # rejected stop leaves it armed
        state["reject_stop"] = False
        result = await c.call_tool("stream.stop", {"session_id": "sess-1"})
        assert result.data["stopped"] is True  # return contract unchanged
        with activity._LOCK:
            assert "sess-1" not in activity._SESSIONS  # accepted stop disarms


# ── binding discard: replaced / explicitly stopped watches release their client ──


def test_watchdog_replacement_discards_previous_binding():
    """Re-registering the same session replaces the watch and releases the
    replaced watch's binding (it never fired and never will); an explicit
    unregister releases the current one. Idempotent."""
    closed: list[str] = []

    def stop() -> None:
        return None

    activity.register_started("s1", stop, 60, on_discard=lambda: closed.append("old"))
    activity.register_started("s1", stop, 60, on_discard=lambda: closed.append("new"))
    assert closed == ["old"]
    activity.unregister("s1")
    assert closed == ["old", "new"]
    activity.unregister("s1")  # idempotent: no double discard
    assert closed == ["old", "new"]


def test_watchdog_discard_failure_never_propagates():
    """A throwing discard (broken client) must not crash unregister."""
    activity.register_started("s1", lambda: None, 60, on_discard=lambda: 1 / 0)
    activity.unregister("s1")  # must not raise


def test_local_bound_client_exposes_no_discard():
    """Local mode: bound_client() is nullcontext over the SHARED client — no
    discard attribute, so the watchdog lifecycle can never close it."""
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=None),  # type: ignore[arg-type]
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    assert getattr(client.bound_client(), "discard", None) is None


async def test_stream_start_hands_discard_to_watchdog(monkeypatch):
    """(hosted wiring) stream.start must pass the binding's discard into the
    watchdog so an explicit stream.stop / replacement releases the fresh
    per-token client; calling it standalone closes without error."""
    calls: list[str] = []
    captured: dict = {}
    real_register = activity.register_started

    def spy(session_id, stop_fn, timeout_sec, on_discard=None):
        captured["on_discard"] = on_discard
        return real_register(session_id, stop_fn, timeout_sec, on_discard=on_discard)

    monkeypatch.setattr(activity, "register_started", spy)
    async with Client(make_header_server(_streaming_backend(calls))) as c:
        result = await c.call_tool(
            "stream.start", {"device_type": "brainbit", "confirm": True, "idle_timeout_sec": 60}
        )
    assert result.data["started"] is True
    assert callable(captured["on_discard"])  # hosted binding owns its client
    captured["on_discard"]()  # standalone release: closes without error
