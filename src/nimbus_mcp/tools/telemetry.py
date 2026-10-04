"""Live telemetry: what the streaming session is doing right now."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._guards import safe_segment


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool
    def get_live_session(session_id: str, window: int = 50) -> dict[str, Any]:
        """Live snapshot of a streaming session: latest prediction + recent window,
        signal quality (meanChannelQuality, snrDb, artifactProbability), indicators,
        running stats. Poll this while a session runs. Live telemetry requires a
        DEPLOYED model session (hub deploy / playback with a classifier); modelless
        hardware streams have no telemetry — use stream_status for those. Expect low
        confidence during filter/ASR warm-up (first seconds); quality < 0.5 or high
        artifactProbability means the signal is poor. 404 => session not active in
        this backend. Each poll also feeds the idle watchdog (see start_stream's
        idle_timeout_sec), keeping an actively watched session alive."""
        sess = safe_segment(session_id, label="session id")
        from . import activity

        activity.note_activity(sess)
        return client.get(
            "/api/hardware/session-live", params={"sessionId": sess, "window": window}
        )
