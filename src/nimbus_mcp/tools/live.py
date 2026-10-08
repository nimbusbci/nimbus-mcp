"""Live tools: device discovery/testing and gated streaming control.

stream.start requires an explicit confirm=True — it connects an EEG device and
starts a live streaming session on the user's head.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ..setup_mode import SetupRequired
from ._annotations import DESTRUCTIVE, MUTATING, READ_ONLY
from ._guards import safe_segment

_DEVICE_FIELDS = (
    "connection_type",
    "port",
    "ip_address",
    "ip_port",
    "stream_name",
    "source_id",
    "mac_address",
    "serial_number",
)

_CAMEL = {
    "connection_type": "connectionType",
    "ip_address": "ipAddress",
    "ip_port": "ipPort",
    "stream_name": "streamName",
    "source_id": "sourceId",
    "mac_address": "macAddress",
    "serial_number": "serialNumber",
}


def _device_body(device_type: str, values: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"deviceType": device_type}
    for field in _DEVICE_FIELDS:
        value = values.get(field)
        if value is not None:
            body[_CAMEL.get(field, field)] = value
    return body


def register(mcp: FastMCP, client: NimbusClient) -> None:
    def _stop_chain(sess_id: str, via_client: NimbusClient = client) -> dict[str, Any]:
        """stop-stream then best-effort disconnect — the single stop sequence
        shared by the stream.stop tool and the idle-timeout watchdog.
        via_client defaults to the server's own client (the stream.stop
        path); the watchdog passes the request-bound client from below."""
        stopped = via_client.post("/api/hardware/stop-stream", json={"sessionId": sess_id})
        disconnect_warning = None
        try:
            via_client.post("/api/hardware/disconnect", json={"sessionId": sess_id})
        except Exception as err:  # best-effort: stream is already stopped
            disconnect_warning = f"stopped, but disconnect failed: {err}"
        return {
            "stopped": bool(stopped.get("stopped", True)),
            "disconnectWarning": disconnect_warning,
        }

    def _stop_chain_via(
        binding: AbstractContextManager[NimbusClient], sess_id: str
    ) -> dict[str, Any]:
        """Watchdog stop path: enter the request-bound client and run the stop
        chain through it. The binding is created inside the stream.start
        request because the watchdog's checker thread fires with no MCP
        request context — a hosted gateway cannot resolve its per-request
        token there. Hosted mode yields a fresh per-token client that is
        ALWAYS closed on exit (the with-block exits even when the stop call
        raises; the checker swallows the error afterwards); local mode yields
        the shared client and never closes it."""
        with binding as bound:
            return _stop_chain(sess_id, bound)

    @mcp.tool(name="device.list", annotations=READ_ONLY)
    def list_devices() -> dict[str, Any]:
        """EEG devices supported by this backend (OpenBCI, Muse, BrainBit, LSL, PiEEG...)."""
        return client.get("/api/deployment/devices")

    @mcp.tool(name="device.test", annotations=READ_ONLY)
    def test_device(
        device_type: str,
        connection_type: str | None = None,
        port: str | None = None,
        ip_address: str | None = None,
        ip_port: int | None = None,
        stream_name: str | None = None,
        source_id: str | None = None,
        mac_address: str | None = None,
        serial_number: str | None = None,
    ) -> dict[str, Any]:
        """Test a device connection WITHOUT starting a stream (safe, no confirm needed).

        Args:
            device_type: Device id from device.list (e.g. "brainbit", "muse").
            connection_type: Device-specific selector when several exist (e.g. serial vs wifi).
            port: Serial/COM port for wired devices.
            ip_address: Device IP for network/wifi devices.
            ip_port: Port for network devices.
            stream_name: LSL stream name (LSL devices).
            source_id: LSL source id.
            mac_address: Bluetooth MAC (BT devices).
            serial_number: Device serial (some BLE stacks).
        """
        values = {
            "connection_type": connection_type,
            "port": port,
            "ip_address": ip_address,
            "ip_port": ip_port,
            "stream_name": stream_name,
            "source_id": source_id,
            "mac_address": mac_address,
            "serial_number": serial_number,
        }
        return client.post("/api/deployment/test-device", json=_device_body(device_type, values))

    @mcp.tool(name="stream.start", annotations=MUTATING)
    def start_stream(
        device_type: str,
        confirm: bool = False,
        session_id: str | None = None,
        chunk_size: int = 125,
        n_channels: int = 8,
        connection_type: str | None = None,
        port: str | None = None,
        ip_address: str | None = None,
        ip_port: int | None = None,
        stream_name: str | None = None,
        source_id: str | None = None,
        mac_address: str | None = None,
        serial_number: str | None = None,
        idle_timeout_sec: int = 900,
    ) -> dict[str, Any]:
        """Connect an EEG device and START a live streaming session on the user's head.
        Requires confirm=True; call device.test first. Track with stream.status().
        Idle watchdog: if no stream.status()/stream.telemetry() poll happens for
        idle_timeout_sec (default 900), the session is stopped and the device
        disconnected automatically — an abandoned stream never keeps running on
        the user's head. Any poll resets the timer; idle_timeout_sec=0 disables
        the watchdog.

        Args:
            device_type: Device id from device.list (e.g. "brainbit").
            confirm: MUST be true to start — the explicit user go-ahead for a live
                session on their head; anything else is refused with zero requests.
            session_id: Optional existing session to resume/reuse.
            chunk_size: Samples per streamed chunk (default 125).
            n_channels: Channel count to open (default 8).
            connection_type: Device-specific selector (e.g. serial vs wifi).
            port: Serial/COM port for wired devices.
            ip_address: Device IP for network/wifi devices.
            ip_port: Port for network devices.
            stream_name: LSL stream name (LSL devices).
            source_id: LSL source id.
            mac_address: Bluetooth MAC (BT devices).
            serial_number: Device serial (some BLE stacks).
            idle_timeout_sec: Watchdog: stop+disconnect after this many seconds
                without a status poll (default 900; 0 disables).
        """
        if not confirm:
            return {
                "started": False,
                "message": "Refusing to start a live session: pass confirm=true to proceed. "
                "Call device.test(device_type=...) first to check the connection.",
            }
        # After the confirm check (so the refusal stays zero-HTTP) and before any request:
        # session_id is interpolated into request bodies downstream.
        if session_id is not None:
            session_id = safe_segment(session_id, label="session id")
        values = {
            "connection_type": connection_type,
            "port": port,
            "ip_address": ip_address,
            "ip_port": ip_port,
            "stream_name": stream_name,
            "source_id": source_id,
            "mac_address": mac_address,
            "serial_number": serial_number,
        }
        body = _device_body(device_type, values)
        if session_id is not None:
            body["sessionId"] = session_id
        connect = client.post("/api/hardware/connect", json=body)
        session_id = connect.get("sessionId")
        started = client.post(
            "/api/hardware/start-stream",
            json={"sessionId": session_id, "chunkSize": chunk_size, "nChannels": n_channels},
        )
        if session_id is not None and idle_timeout_sec > 0:
            from . import activity  # imported here to keep the module graph acyclic

            # Bind BEFORE registering (same pattern as experiment.run's
            # worker thread): the idle-watchdog checker fires the stop
            # callback with no MCP request context, so a request-scoped
            # client — the hosted gateway — cannot resolve its per-request
            # token there. bound_client() runs here, inside the request.
            try:
                binding = client.bound_client()
            except SetupRequired:
                # Unreachable in practice (the posts above already refused in
                # setup mode), but never let the safety net crash the start.
                binding = None
            if binding is not None:
                watch_id = str(session_id)
                activity.register_started(
                    watch_id,
                    lambda: _stop_chain_via(binding, watch_id),
                    float(idle_timeout_sec),
                )
        return {
            "started": bool(started.get("started")),
            "sessionId": session_id,
            "deviceType": connect.get("deviceType"),
            "channels": connect.get("channels"),
            "updateRateHz": started.get("updateRateHz"),
            "hint": "Poll stream.status(session_id); stop with stream.stop(session_id). "
            "Live telemetry (stream.telemetry) requires a DEPLOYED model session "
            "(hub deploy / playback with a classifier); modelless hardware streams "
            "have no telemetry — use stream.status for those.",
        }

    @mcp.tool(name="stream.status", annotations=READ_ONLY)
    def stream_status(session_id: str) -> dict[str, Any]:
        """Live snapshot of a streaming session (running, deviceConnected).
        Polling this also feeds the idle watchdog: each call resets the
        session's idle timer (see stream.start's idle_timeout_sec).

        Args:
            session_id: The streaming session to inspect.
        """
        # session_id is interpolated into a query param — validate before the request.
        sess_id = safe_segment(session_id, label="session id")
        from . import activity

        activity.note_activity(sess_id)
        return client.get("/api/hardware/stream-status", params={"sessionId": sess_id})

    @mcp.tool(name="stream.stop", annotations=DESTRUCTIVE)
    def stop_stream(session_id: str) -> dict[str, Any]:
        """Stop a streaming session and disconnect the device (always safe to call).
        Also removes the session from the idle watchdog so it cannot fire after
        an explicit stop.

        Args:
            session_id: The streaming session to stop.
        """
        sess_id = safe_segment(session_id, label="session id")
        from . import activity

        # Unregister ONLY when the backend accepted the stop: on failure the
        # error propagates to the caller (they see the 403) and the watchdog
        # stays armed — a rejected stop must not disarm the safety net for a
        # session that may still be streaming on the user's head.
        outcome = _stop_chain(sess_id)
        activity.unregister(sess_id)
        return {
            "stopped": outcome["stopped"],
            "sessionId": sess_id,
            "disconnectWarning": outcome["disconnectWarning"],
        }
