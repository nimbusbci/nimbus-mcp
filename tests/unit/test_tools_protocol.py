"""protocol.* tools — MockTransport tests (house style of test_tools_calibration.py).

Asserts against what the handler RECEIVES (captured request bodies), not mock
internals. Backend contracts pinned here (verified against nimbus_backend):
- POST /api/protocols/validate: 200 {"valid": true, "spec"} OR a RAW 422 body
  {"valid": false, "errors": [...]} — intentionally NOT a ProblemDetail; the
  tool must surface the boolean verdict, never raise on the 422.
- POST /api/protocols {"spec"} → 201 {"protocolId", "version"}.
- POST /api/protocol-assignments body is {protocolId, protocolVersion,
  subjectLabel, arm, yokedDonorUploadId?} (extra="forbid").
- Deploy-graph persistence rides POST /api/projects/{uuid}/prepare-zero-cal-deploy
  ({"nChannels", "samplingRate", "execGraph", "layout"}): a train-stage run of
  a hardware_device-headed graph cannot execute offline, so the zero-cal mint
  writes the completed execution + deploy artifacts from geometry alone
  (accepted sinks: interactive_application OR protocol_session).
- The template path binds the simulated device via POST /api/hardware/connect
  ({"deviceType": "simulated", ...}); lifecycle: /api/hardware/pause-stream,
  /api/hardware/start-stream (resume), /api/hardware/stop-stream +
  /api/hardware/disconnect (stop chain — same as tools/live.py).
"""

import json
from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import protocol as protocol_tools

SPEC = {
    "schemaVersion": 1,
    "id": "focus-nf-v1",
    "name": "Focus NF",
    "indicator": {"name": "focus", "transform": "raw"},
    "armDefaults": {"active": {"feedbackSource": "live"}, "sham": {"feedbackSource": "yoked"}},
    "blocks": [
        {"id": "pre", "kind": "assessment", "durationSec": 30},
        {"id": "train", "kind": "training", "durationSec": 30, "feedback": True},
        {"id": "post", "kind": "assessment", "durationSec": 30},
    ],
    "threshold": {"source": "baseline-quantile", "quantile": 0.7, "direction": "increase"},
    "sessions": 3,
    "breakCriteria": {"signalQualityFloor": 0.0},
}

# Mirrors the real focus_meter_deploy template (the built-in protocol graph
# head): simulated hardware_device → bandpass 4–40 Hz → bandpower_indicator.
# The app/output tail is dropped; protocol.save/start_assignment append a
# protocol_session node carrying the protocol refs.
TEMPLATE = {
    "ok": True,
    "template": {
        "id": "focus_meter_deploy",
        "name": "Focus Meter — Bandpower → Deploy",
        "version": "3",
        "train": {
            "nodes": [
                {
                    "id": "data",
                    "type": "hardware_device",
                    "position": {"x": 0, "y": 80},
                    "config": {
                        "buildMode": "live_record",
                        "deviceType": "simulated",
                        "channels": 4,
                        "samplingRate": 250,
                    },
                },
                {
                    "id": "bandpass",
                    "type": "bandpass_filter",
                    "position": {"x": 320, "y": 80},
                    "config": {"lowCut": 4.0, "highCut": 40.0},
                },
                {
                    "id": "indicator",
                    "type": "bandpower_indicator",
                    "position": {"x": 640, "y": 80},
                    "config": {"smoothing": 0.3, "indicators": ["focus"]},
                },
                {
                    "id": "app",
                    "type": "interactive_application",
                    "position": {"x": 960, "y": 80},
                    "config": {"preset": "indicator_focus_meter"},
                },
                {"id": "output", "type": "results_output", "config": {}},
            ],
            "connections": [
                {"from": "data", "to": "bandpass"},
                {"from": "bandpass", "to": "indicator"},
                {"from": "indicator", "to": "app"},
                {"from": "app", "to": "output"},
            ],
        },
    },
}

PIPELINE = {
    "ok": True,
    "executionId": "exec_src",
    "pipeline": {
        "train": {
            "nodes": [
                {"id": "src", "type": "custom_data", "config": {"samplingRate": 250}},
                {"id": "ind", "type": "bandpower_indicator", "config": {}},
                {
                    "id": "proto",
                    "type": "protocol_session",
                    "config": {"protocolId": "old", "protocolVersion": 1},
                },
            ],
            "connections": [
                {"from": "src", "to": "ind"},
                {"from": "ind", "to": "proto"},
            ],
        },
        "layout": {"nodes": {}},
    },
}


def make_server(handler, calls=None) -> FastMCP:
    transport = httpx.MockTransport(handler)
    if calls is not None:

        def tracking(req: httpx.Request) -> httpx.Response:
            calls.append(req.url.path)
            return handler(req)

        transport = httpx.MockTransport(tracking)
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=transport,
    )
    server = FastMCP("t")
    protocol_tools.register(server, client)
    return server


# ── validate ────────────────────────────────────────────────────────────────


async def test_validate_read_only_and_valid_shape():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/protocols/validate"
        assert json.loads(req.read()) == {"spec": SPEC}
        return httpx.Response(200, json={"valid": True, "spec": SPEC})

    async with Client(make_server(handler)) as c:
        tools = await c.list_tools()
        validate = next(t for t in tools if t.name == "protocol.validate")
        assert validate.annotations.read_only_hint is True
        result = await c.call_tool("protocol.validate", {"spec": SPEC})
    assert result.data["valid"] is True
    assert result.data["spec"]["id"] == "focus-nf-v1"


async def test_validate_surfaces_raw_422_verdict_body():
    """The invalid-spec 422 is a raw {"valid": false, "errors"} body, NOT a
    ProblemDetail — the tool returns the verdict instead of raising."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"valid": False, "errors": [{"loc": ["blocks", 0], "msg": "bad"}]}
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("protocol.validate", {"spec": {"id": "x"}})
    assert result.data["valid"] is False
    assert result.data["errors"][0]["msg"] == "bad"


# ── save / list ─────────────────────────────────────────────────────────────


async def test_save_refuses_without_confirm():
    calls: list[str] = []
    async with Client(
        make_server(lambda r: (_ for _ in ()).throw(AssertionError("HTTP!")), calls=calls)
    ) as c:
        result = await c.call_tool("protocol.save", {"spec": SPEC})
    assert result.data["saved"] is False and "confirm" in result.data["message"]
    assert calls == []  # rail: zero HTTP calls without confirm


async def test_save_posts_spec_and_returns_version():
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/protocols":
            return httpx.Response(201, json={"ok": True, "protocolId": "focus-nf-v1", "version": 2})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "protocol.save", {"spec": SPEC, "confirm": True, "name": "run 1"}
        )
    assert json.loads(seen["/api/protocols"].read()) == {"spec": SPEC}
    assert result.data["protocolId"] == "focus-nf-v1"
    assert result.data["version"] == 2


async def test_list_gets_protocols():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/protocols"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "protocols": [{"protocolId": "focus-nf-v1", "version": 2, "name": "Focus NF"}],
                "count": 1,
            },
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("protocol.list", {})
    assert result.data["count"] == 1
    assert result.data["protocols"][0]["protocolId"] == "focus-nf-v1"


# ── start_assignment ────────────────────────────────────────────────────────


async def test_start_assignment_refuses_without_confirm():
    calls: list[str] = []
    async with Client(
        make_server(lambda r: (_ for _ in ()).throw(AssertionError("HTTP!")), calls=calls)
    ) as c:
        result = await c.call_tool(
            "protocol.start_assignment",
            {"protocol_id": "focus-nf-v1", "protocol_version": 1, "subject_label": "S01",
             "arm": "sham"},
        )
    assert result.data["started"] is False and "confirm" in result.data["message"]
    assert calls == []  # rail: zero HTTP calls without confirm


async def test_start_assignment_rejects_unknown_arm():
    async with Client(make_server(lambda r: None)) as c:
        with pytest.raises(Exception, match="arm"):
            await c.call_tool(
                "protocol.start_assignment",
                {"protocol_id": "focus-nf-v1", "protocol_version": 1,
                 "subject_label": "S01", "arm": "placebo", "confirm": True},
            )


def _assignment_backend(
    seen: dict[str, list[httpx.Request]], donor: str | None = None
):
    """Template-path handler: assignment → template → mint → simulated connect."""

    def handler(req: httpx.Request) -> httpx.Response:
        seen.setdefault(req.url.path, []).append(req)
        if req.url.path == "/api/protocol-assignments":
            return httpx.Response(
                201, json={"ok": True, "assignmentId": "asg_1", "sessionIndex": 0}
            )
        if req.url.path == "/api/templates/focus_meter_deploy":
            return httpx.Response(200, json=TEMPLATE)
        if req.url.path.startswith("/api/projects/") and req.url.path.endswith(
            "/prepare-zero-cal-deploy"
        ):
            return httpx.Response(200, json={"ok": True, "executionId": "exec_mint",
                                             "projectId": "p"})
        if req.url.path == "/api/hardware/connect":
            return httpx.Response(
                200,
                json={"ok": True, "sessionId": "sess_9", "deviceType": "simulated",
                      "channels": 4, "samplingRate": 250.0},
            )
        raise AssertionError(req.url.path)

    return handler


async def test_start_assignment_template_path_mints_and_connects():
    seen: dict[str, list[httpx.Request]] = {}
    async with Client(make_server(_assignment_backend(seen))) as c:
        result = await c.call_tool(
            "protocol.start_assignment",
            {
                "protocol_id": "focus-nf-v1",
                "protocol_version": 2,
                "subject_label": "S01",
                "arm": "sham",
                "yoked_donor_upload_id": "up_7",
                "confirm": True,
                "name": "S01 sham run",
            },
        )
    assert result.data["started"] is True
    assert result.data["assignmentId"] == "asg_1"
    assert result.data["executionId"] == "exec_mint"
    assert result.data["sessionId"] == "sess_9"
    assert result.data["channels"] == 4 and result.data["samplingRate"] == 250.0
    assert result.data["statusUrl"]

    # Assignment body: exact camelCase wire shape (extra="forbid").
    body = json.loads(seen["/api/protocol-assignments"][0].read())
    assert body == {
        "protocolId": "focus-nf-v1",
        "protocolVersion": 2,
        "subjectLabel": "S01",
        "arm": "sham",
        "yokedDonorUploadId": "up_7",
    }

    # The mint graph: template head (data/bandpass/indicator, app/output tail
    # dropped, position keys stripped) + protocol_session patched with the
    # protocol refs; geometry from the template device node.
    mint_path = next(p for p in seen if p.endswith("/prepare-zero-cal-deploy"))
    mint = json.loads(seen[mint_path][0].read())
    assert mint["nChannels"] == 4
    assert mint["samplingRate"] == 250
    graph = mint["execGraph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert set(nodes) == {"data", "bandpass", "indicator", "protocol"}
    assert all(set(n) == {"id", "type", "config"} for n in graph["nodes"])
    assert nodes["protocol"]["config"] == {
        "protocolId": "focus-nf-v1",
        "protocolVersion": 2,
        "assignmentId": "asg_1",
    }
    assert {f["from"] for f in graph["connections"]} == {"data", "bandpass", "indicator"}
    assert {"from": "indicator", "to": "protocol"} in graph["connections"]
    assert set(mint["layout"]["nodes"]) == set(nodes)

    # Simulated device bound to the session the deploy WS will reuse.
    connect = json.loads(seen["/api/hardware/connect"][0].read())
    assert connect["deviceType"] == "simulated"
    assert connect["sessionId"].startswith("proto-")
    assert connect["channels"] == 4
    assert connect["samplingRate"] == 250


async def test_start_assignment_omits_donor_key_when_absent():
    seen: dict[str, list[httpx.Request]] = {}
    async with Client(make_server(_assignment_backend(seen))) as c:
        await c.call_tool(
            "protocol.start_assignment",
            {"protocol_id": "focus-nf-v1", "protocol_version": 1,
             "subject_label": "S02", "arm": "active", "confirm": True},
        )
    body = json.loads(seen["/api/protocol-assignments"][0].read())
    assert "yokedDonorUploadId" not in body


async def test_start_assignment_from_execution_graph():
    """An explicit execution_id reuses ITS stored graph (protocol_session node
    patched with the fresh refs) — no template fetch, no device connect (the
    deploy session's data source stays the caller's)."""
    seen: dict[str, list[httpx.Request]] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.setdefault(req.url.path, []).append(req)
        if req.url.path == "/api/protocol-assignments":
            return httpx.Response(201, json={"ok": True, "assignmentId": "asg_2",
                                             "sessionIndex": 0})
        if req.url.path == "/api/executions/exec_src/pipeline":
            return httpx.Response(200, json=PIPELINE)
        if req.url.path.startswith("/api/projects/"):
            return httpx.Response(200, json={"ok": True, "executionId": "exec_mint2",
                                             "projectId": "p"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "protocol.start_assignment",
            {"protocol_id": "focus-nf-v1", "protocol_version": 3,
             "subject_label": "S03", "arm": "active", "execution_id": "exec_src",
             "confirm": True},
        )
    assert result.data["started"] is True
    assert result.data["executionId"] == "exec_mint2"
    assert result.data["sessionId"] is None  # caller-owned data source
    mint_path = next(p for p in seen if p.endswith("/prepare-zero-cal-deploy"))
    graph = json.loads(seen[mint_path][0].read())["execGraph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    # The stored graph's own nodes survive; only the protocol refs are patched.
    assert set(nodes) == {"src", "ind", "proto"}
    assert nodes["proto"]["config"] == {
        "protocolId": "focus-nf-v1",
        "protocolVersion": 3,
        "assignmentId": "asg_2",
    }
    assert "/api/templates/focus_meter_deploy" not in seen
    assert "/api/hardware/connect" not in seen


async def test_start_assignment_execution_graph_requires_protocol_node():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/protocol-assignments":
            return httpx.Response(201, json={"ok": True, "assignmentId": "asg_3",
                                             "sessionIndex": 0})
        if req.url.path == "/api/executions/exec_src/pipeline":
            no_node = json.loads(json.dumps(PIPELINE))
            no_node["pipeline"]["train"]["nodes"] = no_node["pipeline"]["train"]["nodes"][:2]
            no_node["pipeline"]["train"]["connections"] = [
                {"from": "src", "to": "ind"}
            ]
            return httpx.Response(200, json=no_node)
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="protocol_session"):
            await c.call_tool(
                "protocol.start_assignment",
                {"protocol_id": "focus-nf-v1", "protocol_version": 1,
                 "subject_label": "S04", "arm": "active", "execution_id": "exec_src",
                 "confirm": True},
            )


# ── session lifecycle ───────────────────────────────────────────────────────


async def test_session_status_reads_session_live():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/hardware/session-live"
        assert dict(req.url.params) == {"sessionId": "sess_9", "window": "50"}
        return httpx.Response(
            200,
            json={"ok": True, "sessionId": "sess_9", "chunksProcessed": 12,
                  "protocol": {"sessionIndex": 0, "blockIndex": 1, "kind": "training"}},
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "protocol.session_status", {"session_id": "sess_9", "window": 50}
        )
    assert result.data["protocol"]["blockIndex"] == 1


async def test_pause_resume_stop_call_lifecycle_routes():
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/hardware/pause-stream":
            return httpx.Response(200, json={"ok": True, "sessionId": "sess_9",
                                             "paused": True})
        if req.url.path == "/api/hardware/start-stream":
            return httpx.Response(200, json={"ok": True, "sessionId": "sess_9",
                                             "started": True})
        if req.url.path == "/api/hardware/stop-stream":
            return httpx.Response(200, json={"ok": True, "sessionId": "sess_9",
                                             "stopped": True})
        if req.url.path == "/api/hardware/disconnect":
            return httpx.Response(200, json={"ok": True, "sessionId": "sess_9"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        paused = await c.call_tool("protocol.pause", {"session_id": "sess_9"})
        resumed = await c.call_tool("protocol.resume", {"session_id": "sess_9"})
        stopped = await c.call_tool("protocol.stop", {"session_id": "sess_9"})
    assert json.loads(seen["/api/hardware/pause-stream"].read()) == {"sessionId": "sess_9"}
    resume = json.loads(seen["/api/hardware/start-stream"].read())
    assert resume["sessionId"] == "sess_9" and resume["chunkSize"] == 125
    assert json.loads(seen["/api/hardware/stop-stream"].read()) == {"sessionId": "sess_9"}
    assert paused.data["paused"] is True
    assert resumed.data["started"] is True
    assert stopped.data["stopped"] is True
    assert stopped.data["disconnectWarning"] is None


async def test_stop_survives_disconnect_failure():
    """stop-stream succeeded, disconnect failed: the stop is still reported
    with a warning (live.py's best-effort stop-chain contract)."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/hardware/stop-stream":
            return httpx.Response(200, json={"ok": True, "sessionId": "sess_9",
                                             "stopped": True})
        if req.url.path == "/api/hardware/disconnect":
            return httpx.Response(500, json={"detail": "boom"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        stopped = await c.call_tool("protocol.stop", {"session_id": "sess_9"})
    assert stopped.data["stopped"] is True
    assert "boom" in (stopped.data["disconnectWarning"] or "")


# ── export_stats ────────────────────────────────────────────────────────────


async def test_export_stats_posts_assignment_id():
    stats = {
        "ok": True,
        "assignmentId": "asg_1",
        "arm": "sham",
        "indicator": "focus",
        "status": "completed",
        "sessionIndex": 3,
        "sessionsPlanned": 3,
        "descriptive": {},
        "inferential": {"nSessions": 3, "inferential": True},
        "specificity": {},
        "sessionBySession": [],
        "thresholdTrajectory": {},
        "figure": {"image": "aGk=", "mimeType": "image/png"},
        "csv": "sessionIndex,blockId,kind,mean,median,iqr,n",
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/stats/protocol"
        assert json.loads(req.read()) == {"assignmentId": "asg_1"}
        return httpx.Response(200, json=stats)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("protocol.export_stats", {"assignment_id": "asg_1"})
    assert result.data["inferential"]["inferential"] is True
    assert result.data["figure"]["image"] == "aGk="
    assert result.data["csv"].startswith("sessionIndex")
