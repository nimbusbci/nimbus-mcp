"""Live telemetry: what the streaming session is doing right now."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._annotations import READ_ONLY
from ._guards import safe_segment


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="stream.telemetry", annotations=READ_ONLY)
    def get_live_session(session_id: str, window: int = 50) -> dict[str, Any]:
        """Live snapshot of a streaming session: latest prediction + recent window,
        signal quality (meanChannelQuality, snrDb, artifactProbability), indicators,
        running stats. Poll this while a session runs. Works with OR without a
        deployed model: modelless hardware streams (no classifier) report live
        signal quality + focus/relaxation band-power indicators with
        latestPrediction null; deployed sessions additionally carry predictions
        and confidence stats. Expect low
        confidence during filter/ASR warm-up (first seconds); quality < 0.5 or high
        artifactProbability means the signal is poor. 404 => session not active in
        this backend. Each poll also feeds the idle watchdog (see stream.start's
        idle_timeout_sec), keeping an actively watched session alive.

        Args:
            session_id: The streaming session to read telemetry for.
            window: How many recent predictions/chunks to include (default 50).
        """
        sess = safe_segment(session_id, label="session id")
        from . import activity

        activity.note_activity(sess)
        return client.get(
            "/api/hardware/session-live", params={"sessionId": sess, "window": window}
        )
