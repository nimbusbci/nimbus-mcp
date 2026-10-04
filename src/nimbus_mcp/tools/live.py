"""Live tools: device discovery/testing and gated streaming control.

start_stream requires an explicit confirm=True — it connects an EEG device and
starts a live streaming session on the user's head.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
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
    def _stop_chain(sess_id: str) -> dict[str, Any]:
        """stop-stream then best-effort disconnect — the single stop sequence
        shared by the stop_stream tool and the idle-timeout watchdog."""
        stopped = client.post("/api/hardware/stop-stream", json={"sessionId": sess_id})
        disconnect_warning = None
        try:
            client.post("/api/hardware/disconnect", json={"sessionId": sess_id})
        except Exception as err:  # best-effort: stream is already stopped
            disconnect_warning = f"stopped, but disconnect failed: {err}"
        return {
            "stopped": bool(stopped.get("stopped", True)),
            "disconnectWarning": disconnect_warning,
        }

    @mcp.tool
    def list_devices() -> dict[str, Any]:
        """EEG devices supported by this backend (OpenBCI, Muse, BrainBit, LSL, PiEEG...)."""
        return client.get("/api/deployment/devices")

    @mcp.tool
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
        """Test a device connection WITHOUT starting a stream (safe, no confirm needed)."""
        values = {
            "connection_type": connection_type, "port": port, "ip_address": ip_address,
            "ip_port": ip_port, "stream_name": stream_name, "source_id": source_id,
            "mac_address": mac_address, "serial_number": serial_number,
        }
        return client.post("/api/deployment/test-device", json=_device_body(device_type, values))

    @mcp.tool
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
        Requires confirm=True; call test_device first. Track with stream_status().
        Idle watchdog: if no stream_status()/get_live_session() poll happens for
        idle_timeout_sec (default 900), the session is stopped and the device
        disconnected automatically — an abandoned stream never keeps running on
        the user's head. Any poll resets the timer; idle_timeout_sec=0 disables
        the watchdog."""
        if not confirm:
            return {
                "started": False,
                "message": "Refusing to start a live session: pass confirm=true to proceed. "
                "Call test_device(device_type=...) first to check the connection.",
            }
        # After the confirm check (so the refusal stays zero-HTTP) and before any request:
        # session_id is interpolated into request bodies downstream.
        if session_id is not None:
            session_id = safe_segment(session_id, label="session id")
        values = {
            "connection_type": connection_type, "port": port, "ip_address": ip_address,
            "ip_port": ip_port, "stream_name": stream_name, "source_id": source_id,
            "mac_address": mac_address, "serial_number": serial_number,
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

            watch_id = str(session_id)
            activity.register_started(
                watch_id,
                lambda: _stop_chain(watch_id),
                float(idle_timeout_sec),
            )
        return {
            "started": bool(started.get("started")),
            "sessionId": session_id,
            "deviceType": connect.get("deviceType"),
            "channels": connect.get("channels"),
            "updateRateHz": started.get("updateRateHz"),
            "hint": "Poll stream_status(session_id); stop with stop_stream(session_id). "
            "Live telemetry (get_live_session) requires a DEPLOYED model session "
            "(hub deploy / playback with a classifier); modelless hardware streams "
            "have no telemetry — use stream_status for those.",
        }

    @mcp.tool
    def stream_status(session_id: str) -> dict[str, Any]:
        """Live snapshot of a streaming session (running, deviceConnected).
        Polling this also feeds the idle watchdog: each call resets the
        session's idle timer (see start_stream's idle_timeout_sec)."""
        # session_id is interpolated into a query param — validate before the request.
        sess_id = safe_segment(session_id, label="session id")
        from . import activity

        activity.note_activity(sess_id)
        return client.get(
            "/api/hardware/stream-status", params={"sessionId": sess_id}
        )

    @mcp.tool
    def stop_stream(session_id: str) -> dict[str, Any]:
        """Stop a streaming session and disconnect the device (always safe to call).
        Also removes the session from the idle watchdog so it cannot fire after
        an explicit stop."""
        sess_id = safe_segment(session_id, label="session id")
        from . import activity

        try:
            outcome = _stop_chain(sess_id)
        finally:
            activity.unregister(sess_id)
        return {
            "stopped": outcome["stopped"],
            "sessionId": sess_id,
            "disconnectWarning": outcome["disconnectWarning"],
        }
