"""Protocol tools: declarative neurofeedback protocols + assignment sessions (0.15).

Backend contracts (verified against nimbus_backend):
- POST /api/protocols/validate ``{"spec"}`` → ``200 {"valid": true, "spec"}`` OR
  a RAW 422 body ``{"valid": false, "errors": [...]}`` — deliberately NOT a
  ProblemDetail, so agents branch on the boolean; the tool surfaces that body
  instead of raising (opt-in 422 passthrough on the client).
- POST /api/protocols ``{"spec"}`` → ``201 {"protocolId", "version"}`` (validates
  first; an invalid spec is a semantic 422 ProblemDetail that raises).
- GET /api/protocols → latest version per id.
- POST /api/protocol-assignments ``{protocolId, protocolVersion, subjectLabel,
  arm, yokedDonorUploadId?}`` → ``201 {"assignmentId", "sessionIndex"}``. The
  backend validates the donor against the arm's yoked blocks (owned upload,
  deploy-HDF5 indicators group, coverage ≥ ~90% of yoked-seconds at the
  donor's derived indicator rate).
- Deploy-graph persistence rides ``POST /api/projects/{uuid}/prepare-zero-cal-deploy``
  (``{"nChannels", "samplingRate", "execGraph", "layout"}``): a train-stage run
  of a hardware_device-headed graph cannot execute offline (bandpass needs
  eeg_data), so the zero-cal mint writes the COMPLETED execution — snapshot +
  ``training_metadata.pkl`` + empty ``pipeline_state.pkl`` — from channel
  geometry alone (accepted sinks: interactive_application OR protocol_session).
  The project id only has to be uuid-shaped; the row need not exist.
- POST /api/hardware/connect ``{"deviceType": "simulated", ...}`` binds the
  simulated board to the session id the deploy WS then reuses.
- Lifecycle over HTTP: ``/api/hardware/pause-stream`` (device loop stopped,
  session kept — the WS pause semantics), ``/api/hardware/start-stream``
  (resume), ``/api/hardware/stop-stream`` + best-effort ``/api/hardware/disconnect``
  (the stop chain; the stop-path close-out saves the block-annotated recording
  and bumps the assignment when the scheduler reached its natural end).
- GET /api/hardware/session-live ``?sessionId=`` carries the live protocol
  scheduler snapshot under ``protocol``.
- POST /api/stats/protocol ``{"assignmentId"}`` → descriptive/inferential/
  specificity/sessionBySession/figure (base64 PNG)/csv.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import MUTATING, READ_ONLY
from ._guards import safe_segment
from .run import synth_layout

# The built-in protocol graph head: the focus-meter deploy template's
# simulated source → bandpass → bandpower_indicator chain. The app/output tail
# is dropped and a protocol_session node (Task 6) is appended with the protocol
# refs — the calibration.start template-clone-patch pattern.
PROTOCOL_TEMPLATE = "focus_meter_deploy"
_PROTOCOL_NODE_ID = "protocol"
# Canonical exec-graph node shape; canvas extras (position) ride template
# snapshots but not the execute/mint contracts.
_NODE_KEYS = ("id", "type", "config")
_DEFAULT_SAMPLING_RATE = 250


def _template_protocol_graph(client: NimbusClient, template_id: str) -> dict[str, Any]:
    """Clone the deploy template's graph head (data → filter → indicator).

    The interactive_application / results_output tail is dropped: the protocol
    session's feedback panel is mounted by the deploy runtime from the
    protocol snapshot, not by an app node.
    """
    template = client.get(f"/api/templates/{safe_segment(template_id, label='template id')}").get(
        "template"
    )
    if not isinstance(template, dict):
        raise McpToolError(f"Template '{template_id}' not found.")
    raw = template.get("train")
    if not isinstance(raw, dict) or not raw.get("nodes"):
        raise McpToolError(f"Template '{template_id}' has no train graph section.")
    keep: list[str] = []
    nodes: list[dict[str, Any]] = []
    for node in raw.get("nodes") or []:
        if not isinstance(node, dict) or not node.get("id"):
            continue
        if str(node.get("type") or "") in ("interactive_application", "results_output", "output"):
            continue
        nodes.append({key: node[key] for key in _NODE_KEYS if key in node})
        keep.append(str(node["id"]))
    if len(nodes) < 2:
        raise McpToolError(f"Template '{template_id}' has no indicator graph head to reuse.")
    connections = [
        {"from": str(c["from"]), "to": str(c["to"])}
        for c in raw.get("connections") or []
        if isinstance(c, dict) and c.get("from") in keep and c.get("to") in keep
    ]
    if not connections:
        raise McpToolError(f"Template '{template_id}' head has no connections.")
    return {"nodes": nodes, "connections": connections}


def _stored_execution_graph(client: NimbusClient, execution_id: str) -> dict[str, Any]:
    """The stored train graph of a completed execution (GET .../pipeline)."""
    payload = client.get(
        f"/api/executions/{safe_segment(execution_id)}/pipeline"
    ).get("pipeline")
    train = payload.get("train") if isinstance(payload, dict) else None
    if not isinstance(train, dict) or not train.get("nodes"):
        raise McpToolError(
            f"Execution {execution_id} has no stored train graph — run it to completion first."
        )
    graph = {"nodes": train["nodes"], "connections": train.get("connections") or []}
    return json.loads(json.dumps(graph))


def _patch_protocol_refs(
    graph: dict[str, Any],
    *,
    protocol_id: str,
    protocol_version: int,
    assignment_id: str,
    tail_from: str | None = None,
) -> dict[str, Any]:
    """Stamp the protocol refs onto the graph's protocol_session node.

    An existing node is patched in place; on the template path (``tail_from``,
    the indicator node id) a fresh node is appended and wired in.
    """
    refs = {
        "protocolId": protocol_id,
        "protocolVersion": int(protocol_version),
        "assignmentId": assignment_id,
    }
    for node in graph["nodes"]:
        if str(node.get("type") or "") == "protocol_session":
            cfg = dict(node.get("config") or {})
            cfg.update(refs)
            node["config"] = cfg
            return graph
    if tail_from is None:
        raise McpToolError(
            "Execution graph has no protocol_session node — add one in Build (or omit "
            "execution_id to use the built-in simulated protocol template)."
        )
    graph["nodes"] = [
        *graph["nodes"],
        {"id": _PROTOCOL_NODE_ID, "type": "protocol_session", "config": dict(refs)},
    ]
    connections = list(graph.get("connections") or [])
    connections.append({"from": tail_from, "to": _PROTOCOL_NODE_ID})
    graph["connections"] = connections
    return graph


def _graph_geometry(graph: dict[str, Any]) -> tuple[int, float]:
    """(channels, samplingRate) from the graph's data-source nodes (4ch/250Hz default)."""
    channels: int | None = None
    rate: float | None = None
    for node in graph["nodes"]:
        cfg = node.get("config") or {}
        if node.get("type") == "hardware_device":
            channels = int(cfg.get("channels") or channels or 4)
            rate = float(cfg.get("samplingRate") or rate or _DEFAULT_SAMPLING_RATE)
        elif node.get("type") in ("custom_data", "public_data") and rate is None:
            rate = float(cfg.get("samplingRate") or _DEFAULT_SAMPLING_RATE)
    return (channels or 4, rate or _DEFAULT_SAMPLING_RATE)


def _mint_deploy_execution(
    client: NimbusClient,
    graph: dict[str, Any],
    *,
    channels: int,
    sampling_rate: float,
) -> str:
    """Persist the graph as a completed deploy execution (zero-cal mint)."""
    mint_id = str(uuid.uuid4())  # uuid-shaped; the projects row need not exist
    payload = client.post(
        f"/api/projects/{mint_id}/prepare-zero-cal-deploy",
        json={
            "nChannels": channels,
            "samplingRate": sampling_rate,
            "execGraph": graph,
            "layout": synth_layout(graph),
        },
    )
    execution_id = payload.get("executionId") if isinstance(payload, dict) else None
    if not execution_id:
        raise McpToolError(f"Deploy mint returned no executionId: {payload!r}")
    return str(execution_id)


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="protocol.validate", annotations=READ_ONLY)
    def validate(spec: dict[str, Any]) -> dict[str, Any]:
        """Validate a neurofeedback protocol document WITHOUT saving it.
        Returns {valid: true, spec} on success, or {valid: false, errors: [...]}
        with the per-violation list (loc/msg) when the document is invalid —
        the verdict is data, never an exception.

        Args:
            spec: Wire-format protocol document (camelCase): id, name,
                indicator {name, transform}, armDefaults, blocks
                [{id, kind: assessment|training, durationSec, feedback?}],
                threshold, sessions, breakCriteria.
        """
        payload = client.post("/api/protocols/validate", json={"spec": spec}, allow_422=True)
        if not isinstance(payload, dict):
            return {"valid": False, "errors": [{"loc": [], "msg": str(payload)}]}
        out: dict[str, Any] = {"valid": bool(payload.get("valid"))}
        if payload.get("spec"):
            out["spec"] = payload["spec"]
        if payload.get("errors"):
            out["errors"] = payload["errors"]
        return out

    @mcp.tool(name="protocol.save", annotations=MUTATING)
    def save(spec: dict[str, Any], confirm: bool = False) -> dict[str, Any]:
        """Validate and persist the NEXT VERSION of a protocol document
        (confirm-gated: each call appends a version row; accidental calls
        clutter the version history). Edits create a new version — the
        assignment pins the version it runs.

        Args:
            spec: Wire-format protocol document (validate first to iterate).
            confirm: MUST be true to write.
        """
        if not confirm:
            return {
                "saved": False,
                "message": "Refusing to save: pass confirm=true (this appends the next "
                "protocol version). Call protocol.validate first while iterating.",
            }
        payload = client.post("/api/protocols", json={"spec": spec})
        return {
            "saved": True,
            "protocolId": payload.get("protocolId"),
            "version": payload.get("version"),
        }

    @mcp.tool(name="protocol.list", annotations=READ_ONLY)
    def list_protocols() -> dict[str, Any]:
        """Every saved protocol id at its latest version."""
        return client.get("/api/protocols")

    @mcp.tool(name="protocol.start_assignment", annotations=MUTATING)
    def start_assignment(
        protocol_id: str,
        protocol_version: int,
        subject_label: str,
        arm: str,
        confirm: bool = False,
        yoked_donor_upload_id: str | None = None,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Enroll a subject and prepare their neurofeedback deploy session
        (confirm-gated — a session runs on a human). Creates the assignment
        (yoked sham arms REQUIRE yoked_donor_upload_id: a recorded simulated
        session's upload — mint one by running an active-arm session and
        data.upload-ing its recording), then persists the protocol graph as a
        deployable execution. Requires a Pro plan for LIVE hardware sessions;
        simulated-source sessions (the default template) are free.

        Graph source: execution_id reuses THAT run's stored graph (it must
        contain a protocol_session node; no device is pre-connected — the
        deploy session's data source stays yours). Without execution_id the
        built-in template graph is cloned — simulated source → bandpass →
        bandpower_indicator → protocol_session stamped with the protocol refs —
        and a simulated device is connected to the returned session id.

        The session starts when a client opens {statusUrl} (the deploy WS),
        sends init_deployment with modelExecutionId={executionId}, then start;
        the Studio app shows the feedback panel automatically. Poll
        protocol.session_status; stop with protocol.stop (the recording's
        block annotations + assignment progression are written on stop).

        Args:
            protocol_id: Saved protocol id (protocol.save).
            protocol_version: Immutable version the assignment pins.
            subject_label: Subject identifier (e.g. "S01").
            arm: active | sham.
            confirm: MUST be true — explicit go-ahead for a session on a human.
            yoked_donor_upload_id: Donor recording upload (required when the
                arm's blocks resolve to yoked — the backend validates coverage).
            execution_id: Optional completed execution whose stored graph
                carries a protocol_session node; default clones the template.
        """
        if not confirm:
            return {
                "started": False,
                "message": "Refusing to start an assignment session: pass confirm=true "
                "(a neurofeedback session on a human). Validate the protocol and the "
                "donor first (protocol.validate).",
            }
        if arm not in ("active", "sham"):
            raise McpToolError(f"arm must be 'active' or 'sham' (got {arm!r}).")
        pid = safe_segment(protocol_id, label="protocol id")
        body: dict[str, Any] = {
            "protocolId": pid,
            "protocolVersion": int(protocol_version),
            "subjectLabel": subject_label,
            "arm": arm,
        }
        if yoked_donor_upload_id is not None:
            body["yokedDonorUploadId"] = yoked_donor_upload_id
        assignment = client.post("/api/protocol-assignments", json=body)
        assignment_id = assignment.get("assignmentId")
        if not assignment_id:
            raise McpToolError(f"Assignment create returned no id: {assignment!r}")
        try:
            session_id: str | None = None
            if execution_id is not None:
                graph = _stored_execution_graph(client, execution_id)
                graph = _patch_protocol_refs(
                    graph,
                    protocol_id=pid,
                    protocol_version=int(protocol_version),
                    assignment_id=str(assignment_id),
                )
            else:
                template = _template_protocol_graph(client, PROTOCOL_TEMPLATE)
                tail_from = str(template["nodes"][-1]["id"])
                graph = _patch_protocol_refs(
                    template,
                    protocol_id=pid,
                    protocol_version=int(protocol_version),
                    assignment_id=str(assignment_id),
                    tail_from=tail_from,
                )
            channels, sampling_rate = _graph_geometry(graph)
            execution = _mint_deploy_execution(
                client, graph, channels=channels, sampling_rate=sampling_rate
            )
            if execution_id is None:
                # Template path: bind the simulated board so the deploy WS
                # init reuses it (simulated sources are free-tier deploys).
                connect = client.post(
                    "/api/hardware/connect",
                    json={
                        "deviceType": "simulated",
                        "sessionId": f"proto-{uuid.uuid4().hex[:12]}",
                        "channels": channels,
                        "samplingRate": sampling_rate,
                    },
                )
                session_id = connect.get("sessionId")
        except McpToolError as err:
            raise McpToolError(
                f"Assignment {assignment_id} created, but session preparation failed "
                f"({err}). The assignment row remains usable — fix the cause and "
                "retry with the same protocol refs.",
                status_code=getattr(err, "status_code", None),
                code=getattr(err, "code", None),
            ) from err
        return {
            "started": True,
            "assignmentId": str(assignment_id),
            "sessionIndex": assignment.get("sessionIndex"),
            "executionId": execution,
            "sessionId": session_id,
            "channels": channels,
            "samplingRate": sampling_rate,
            "statusUrl": f"/ws/realtime/{session_id}" if session_id else None,
            "dashboardHint": (
                "Open the deploy session over WS: auth → init_deployment(modelExecutionId="
                f"{execution}, dataSource simulated"
                + (f", reuses the connected simulated device on session {session_id}"
                   if session_id else " — connect your data source first")
                + ") → start. The Studio app mounts the protocol feedback panel; "
                "poll protocol.session_status; protocol.stop saves the recording and "
                "advances the assignment."
            ),
        }

    @mcp.tool(name="protocol.session_status", annotations=READ_ONLY)
    def session_status(session_id: str, window: int = 50) -> dict[str, Any]:
        """Live snapshot of a protocol deploy session: the scheduler block /
        session progress + threshold state (under ``protocol``) beside the
        telemetry window (recent indicators, quality, stats).

        Args:
            session_id: The deploy session to inspect (from start_assignment).
            window: How many recent predictions to include (1-200, default 50).
        """
        sess_id = safe_segment(session_id, label="session id")
        return client.get(
            "/api/hardware/session-live", params={"sessionId": sess_id, "window": window}
        )

    @mcp.tool(name="protocol.pause", annotations=MUTATING)
    def pause(session_id: str) -> dict[str, Any]:
        """Pause a running protocol session (blocks freeze; the session and
        device stay alive — resume with protocol.resume). Simulated and
        playback sessions only while paused: no data is recorded.

        Args:
            session_id: The deploy session to pause.
        """
        sess_id = safe_segment(session_id, label="session id")
        return client.post("/api/hardware/pause-stream", json={"sessionId": sess_id})

    @mcp.tool(name="protocol.resume", annotations=MUTATING)
    def resume(session_id: str, chunk_size: int = 125, n_channels: int = 4) -> dict[str, Any]:
        """Resume a paused protocol session (restarts the server-driven device
        loop). Live-hardware resume requires a Pro plan; simulated sessions are
        free. Pass the session's channel count (start_assignment reports it).

        Args:
            session_id: The deploy session to resume.
            chunk_size: Samples per streamed chunk (default 125).
            n_channels: Session channel count (the built-in template opens 4).
        """
        sess_id = safe_segment(session_id, label="session id")
        return client.post(
            "/api/hardware/start-stream",
            json={"sessionId": sess_id, "chunkSize": chunk_size, "nChannels": n_channels},
        )

    @mcp.tool(name="protocol.stop", annotations=MUTATING)
    def stop(session_id: str) -> dict[str, Any]:
        """Stop a protocol session (always safe to call): stops streaming,
        saves the block-annotated recording, advances the assignment when the
        scheduler reached its natural end, and disconnects the device
        (best-effort — a failed disconnect is reported, never raised).

        Args:
            session_id: The deploy session to stop.
        """
        sess_id = safe_segment(session_id, label="session id")
        stopped = client.post("/api/hardware/stop-stream", json={"sessionId": sess_id})
        disconnect_warning = None
        try:
            client.post("/api/hardware/disconnect", json={"sessionId": sess_id})
        except Exception as err:  # best-effort: the stream is already stopped
            disconnect_warning = f"stopped, but disconnect failed: {err}"
        return {
            "stopped": bool(stopped.get("stopped", True)),
            "sessionId": sess_id,
            "disconnectWarning": disconnect_warning,
            "nextHint": "protocol.export_stats(assignment_id) analyzes the recorded "
            "sessions once enough have completed.",
        }

    @mcp.tool(name="protocol.export_stats", annotations=READ_ONLY)
    def export_stats(assignment_id: str) -> dict[str, Any]:
        """Paired pre/post analysis for one assignment's recorded sessions:
        per-block aggregates (descriptive + CSV), paired t / Wilcoxon / dz
        (per-statistic minimums: paired t and Cohen's dz need ≥ 3 completed
        sessions, Wilcoxon needs ≥ 5), specificity notes, the
        adaptive-threshold trajectory, and a pre/post figure (base64 PNG —
        render or save it to look at the effect). An assignment with no
        recordings yet returns the empty payload with a note, never an error.

        Args:
            assignment_id: The assignment to analyze (from start_assignment).
        """
        asg_id = safe_segment(assignment_id, label="assignment id")
        return client.post("/api/stats/protocol", json={"assignmentId": asg_id})
