"""Calibration tools: agent-guided subject calibration + train handoff.

Backend contracts (verified against nimbus_backend):
- POST /api/execute body is ``{"train": graph, "stage": ..., "layout": ...}`` —
  ``PipelineExecuteRequest`` is ``extra="forbid"`` and has NO ``calibrate``
  key; the calibration graph rides the same ``train`` field with
  ``stage="calibrate"`` (a stage absent from the request would be derived
  server-side from graph content, but MCP always sends it explicitly).
- POST /api/pause-calibration / /api/resume-calibration take ``{"executionId"}``.
- GET /api/calibration-live/{id}: raw engine phases while running, ``complete``
  + ``calibration {uploadId, path, filename, format}`` once terminal.
- The MI default train template is ``mi_headband_csp_lda`` (the app catalog
  pairs it with ``mi_hardware_calibration``; its first node is a
  ``custom_data`` node pinning the latest calibration run).
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import MUTATING, READ_ONLY
from ._guards import safe_segment
from .run import synth_layout

PARADIGM_TEMPLATES = {
    "mi": "mi_hardware_calibration",
    "p300": "p300_hardware_calibration",
    "sart": "sart_calibration",
    "target_hit": "target_hit_calibration",
}

# Verified against nimbus_backend/storage/templates/motor_imagery/
# mi_headband_csp_lda.yaml: first node (id "data") is a custom_data node whose
# source pins the latest calibration run — calibration.train only swaps the
# source ref for the pinned upload.
MI_DEFAULT_TRAIN_TEMPLATE = "mi_headband_csp_lda"

_DEFAULT_SAMPLING_RATE = 250

# Graph node types that load data; the first one in a train graph is the node
# calibration.train rewires onto the recorded calibration upload. The rewire
# PRESERVES the node's non-data config keys (mode/paradigm/nClasses/classLabels/
# channelNames — the training engine needs them) and replaces only the data
# placement: source → pinned upload, filePath cleared, format/samplingRate from
# the recording geometry. custom_data has NO "channels" config key (unknown
# keys are a 400) — the channel count rides inside the HDF5 recording.
_DATA_NODE_TYPES = ("custom_data", "public_data")

# snake_case tool params → device-node config keys (mirrors live.py's mapping).
_DEVICE_KEYS = (
    ("connection_type", "connectionType"),
    ("port", "port"),
    ("ip_address", "ipAddress"),
    ("ip_port", "ipPort"),
    ("stream_name", "streamName"),
    ("source_id", "sourceId"),
    ("mac_address", "macAddress"),
    ("serial_number", "serialNumber"),
)


def _fetch_template(client: NimbusClient, template_id: str) -> dict[str, Any]:
    """Fetch a template snapshot (GET /api/templates/{id}) as a deep copy."""
    tid = safe_segment(template_id, label="template id")
    template = client.get(f"/api/templates/{tid}").get("template")
    if not isinstance(template, dict):
        raise McpToolError(f"Template '{tid}' not found.")
    return json.loads(json.dumps(template))  # deep copy before patching


def _graph_section(template: dict[str, Any], section: str, template_id: str) -> dict[str, Any]:
    """A template's graph section ({nodes, connections}), inner layout dropped.

    The inner layout key must not ride the posted graph — the execute request's
    ``train`` field (ExecGraphSnapshot) is extra="forbid".
    """
    raw = template.get(section)
    if not isinstance(raw, dict) or not raw.get("nodes"):
        if section == "calibrate":
            # Some backend builds (e.g. desktop distribution) store calibration
            # graphs in the template's primary 'train' section.
            raw = template.get("train")
        if not isinstance(raw, dict) or not raw.get("nodes"):
            raise McpToolError(f"Template '{template_id}' has no {section} graph section.")
    return {
        "nodes": [n for n in (raw.get("nodes") or []) if isinstance(n, dict)],
        "connections": [c for c in (raw.get("connections") or []) if isinstance(c, dict)],
    }


def _layout_or_none(layout: Any) -> dict[str, Any] | None:
    """A canvas layout ({nodes: {id: {x, y}}}) when the candidate is one."""
    if isinstance(layout, dict) and layout.get("nodes"):
        return layout
    return None


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="calibration.start", annotations=MUTATING)
    def start(
        paradigm: str = "mi",
        confirm: bool = False,
        trials_per_class: int | None = None,
        classes: list[dict[str, Any]] | None = None,
        name: str | None = None,
        device_type: str | None = None,
        connection_type: str | None = None,
        port: str | None = None,
        ip_address: str | None = None,
        ip_port: int | None = None,
        stream_name: str | None = None,
        source_id: str | None = None,
        mac_address: str | None = None,
        serial_number: str | None = None,
    ) -> dict[str, Any]:
        """Start a guided subject calibration session (NON-BLOCKING; confirm-gated —
        the device goes on a human's head). The Nimbus Studio app shows the cues on
        its calibration dashboard automatically; poll calibration.status.
        Requires a Pro plan (hosted token or Pro session): calibration nodes and
        custom-data training are gated by the freemium node policy; local
        X-MCP-Key principals get 403 by policy.

        Args:
            paradigm: mi | p300 | sart | target_hit.
            confirm: MUST be true — explicit user go-ahead for a session on their head.
            trials_per_class: Override the template's trial count (minimum 5).
            classes: Override class list [{id,label,cue}] (MI default: left/right hand).
            name: Display name recorded on the execution.
            device_type: Device id from device.list (omit → template default synthetic).
            connection_type: Device selector when several exist (e.g. serial vs wifi).
            port: Serial/COM port for wired devices.
            ip_address: Device IP for network/wifi devices.
            ip_port: Port for network devices.
            stream_name: LSL stream name (LSL devices).
            source_id: LSL source id.
            mac_address: Bluetooth MAC (BT devices).
            serial_number: Device serial (some BLE stacks).
        """
        if not confirm:
            return {
                "started": False,
                "message": "Refusing to start calibration: pass confirm=true "
                "(a device session on a human). Call device.test first.",
            }
        if trials_per_class is not None and trials_per_class < 5:
            raise McpToolError(
                f"trials_per_class must be at least 5 (protocol minimum; got {trials_per_class})."
            )
        template_id = PARADIGM_TEMPLATES.get(paradigm)
        if template_id is None:
            raise McpToolError(
                f"Unknown paradigm '{paradigm}'. Valid: {sorted(PARADIGM_TEMPLATES)}."
            )
        template = _fetch_template(client, template_id)
        graph = _graph_section(template, "calibrate", template_id)
        device_values = {
            "connection_type": connection_type,
            "port": port,
            "ip_address": ip_address,
            "ip_port": ip_port,
            "stream_name": stream_name,
            "source_id": source_id,
            "mac_address": mac_address,
            "serial_number": serial_number,
        }
        protocol_cfg: dict[str, Any] | None = None
        for node in graph["nodes"]:
            cfg = node.get("config") or {}
            if node.get("type") == "hardware_device" and device_type is not None:
                cfg["deviceType"] = device_type
                for param, key in _DEVICE_KEYS:
                    value = device_values[param]
                    if value is not None:
                        cfg[key] = value
                node["config"] = cfg
            if node.get("type") == "trial_protocol":
                if trials_per_class is not None:
                    cfg["trialsPerClass"] = trials_per_class
                if classes is not None:
                    cfg["classes"] = classes
                node["config"] = cfg
                protocol_cfg = node["config"]
        # Planned trial count from the (patched or template) protocol node:
        # trialsPerClass × number of classes; None when no protocol node rides
        # the graph (should not happen for the shipped templates).
        trials_planned: int | None = None
        if protocol_cfg is not None:
            tpc = protocol_cfg.get("trialsPerClass")
            class_list = protocol_cfg.get("classes")
            if isinstance(tpc, int) and isinstance(class_list, list) and class_list:
                trials_planned = tpc * len(class_list)
        body: dict[str, Any] = {
            "train": graph,
            "stage": "calibrate",
            # The calibrate section's own canvas layout travels inside the
            # section (verbatim from the template yaml), or top-level layout.
            "layout": _layout_or_none((template.get("calibrate") or {}).get("layout"))
            or _layout_or_none(template.get("layout"))
            or synth_layout(graph),
        }
        if name is not None:
            body["name"] = name
        payload = client.post("/api/execute", json=body)
        return {
            "started": True,
            "executionId": payload.get("executionId"),
            "paradigm": paradigm,
            "trialsPlanned": trials_planned,
            "dashboardHint": "The Studio app shows the cues. Poll calibration.status.",
        }

    @mcp.tool(name="calibration.status", annotations=READ_ONLY)
    def status(execution_id: str) -> dict[str, Any]:
        """Live snapshot of a calibration session (phase, current trial, progress,
        paused). Once complete, carries the recorded upload — call
        calibration.train to turn it into the subject's own classifier.

        Args:
            execution_id: The calibration run to inspect (from calibration.start).
        """
        exec_id = safe_segment(execution_id)
        payload = client.get(f"/api/calibration-live/{exec_id}")
        out = dict(payload) if isinstance(payload, dict) else {"raw": payload}
        if out.get("phase") == "complete":
            upload = out.get("calibration") or {}
            if upload.get("uploadId"):
                out["nextHint"] = (
                    f"Calibration complete (upload {upload.get('uploadId')}). "
                    f"Call calibration.train(execution_id={exec_id}) "
                    "to train the subject's classifier."
                )
            else:
                # Completed without a resolvable recording (none linked, or a
                # desktop-local backend whose uploads the API can't resolve) —
                # do NOT point at train; it needs the uploadId.
                out["nextHint"] = (
                    "Calibration complete, but no recording is available from this "
                    "backend — rerun calibration.start on a Postgres-backed backend "
                    "to get a trainable recording."
                )
        return out

    @mcp.tool(name="calibration.pause", annotations=MUTATING)
    def pause(execution_id: str) -> dict[str, Any]:
        """Pause a running calibration between trials (cues hold; resume anytime).

        Args:
            execution_id: The calibration run to pause.
        """
        exec_id = safe_segment(execution_id)
        return client.post("/api/pause-calibration", json={"executionId": exec_id})

    @mcp.tool(name="calibration.resume", annotations=MUTATING)
    def resume(execution_id: str) -> dict[str, Any]:
        """Resume a paused calibration session.

        Args:
            execution_id: The calibration run to resume.
        """
        exec_id = safe_segment(execution_id)
        return client.post("/api/resume-calibration", json={"executionId": exec_id})

    @mcp.tool(name="calibration.train", annotations=MUTATING)
    def train(
        execution_id: str,
        paradigm: str = "mi",
        template_id: str | None = None,
        train_graph: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Train the subject's own classifier from a COMPLETED calibration session
        (NON-BLOCKING). Fetches the recorded upload, wires it into a train
        pipeline as a custom_data source, and starts the run. Requires a Pro
        plan (custom_data training is freemium-gated). The calibrate→train
        handoff requires a Postgres-backed backend (hosted or local dev); a
        desktop-local session completes and records, but its upload can't be
        resolved by MCP train today.

        Args:
            execution_id: The COMPLETED calibration run (from calibration.start).
            paradigm: Paradigm of the recording (mi | p300 | sart | target_hit);
                only mi has a default train template.
            template_id: Train template to use (e.g. from catalog.templates);
                required for non-mi paradigms (mi defaults to mi_headband_csp_lda).
            train_graph: Explicit train graph instead of a template; its first
                data node (custom_data/public_data) is rewired onto the recording.
            name: Display name recorded on the training run.
        """
        exec_id = safe_segment(execution_id)
        if PARADIGM_TEMPLATES.get(paradigm) is None:
            raise McpToolError(
                f"Unknown paradigm '{paradigm}'. Valid: {sorted(PARADIGM_TEMPLATES)}."
            )
        if train_graph is None and template_id is None and paradigm != "mi":
            raise McpToolError(
                f"No default train template for paradigm '{paradigm}' — pass an explicit "
                f"template_id (from catalog.templates). Only mi has one "
                f"({MI_DEFAULT_TRAIN_TEMPLATE})."
            )
        live = client.get(f"/api/calibration-live/{exec_id}")
        if live.get("phase") != "complete":
            raise McpToolError(
                f"Calibration {exec_id} is not complete (phase: {live.get('phase') or 'unknown'}) "
                "— poll calibration.status until phase is complete."
            )
        upload = live.get("calibration") or {}
        upload_id = upload.get("uploadId")
        if not upload_id:
            raise McpToolError(
                f"Calibration {exec_id} completed without a recording — rerun calibration.start."
            )
        # Recording geometry from the paradigm's calibration template (device
        # node): the sampling rate the train node must carry. The channel count
        # is NOT a custom_data config key — it rides inside the HDF5 recording.
        # That HTTP fetch is only needed for the default-template path; an
        # explicit train_graph keeps its own data-node samplingRate (default
        # 250 when it carries none).
        sampling_rate = _DEFAULT_SAMPLING_RATE
        if train_graph is None:
            cal_template = _fetch_template(client, PARADIGM_TEMPLATES[paradigm])
            cal_graph = _graph_section(cal_template, "calibrate", PARADIGM_TEMPLATES[paradigm])
            for node in cal_graph["nodes"]:
                if node.get("type") == "hardware_device":
                    cfg = node.get("config") or {}
                    sampling_rate = cfg.get("samplingRate") or _DEFAULT_SAMPLING_RATE
                    break
        # Train graph: explicit graph, or a template whose first data node gets
        # replaced by a custom_data node pinned to the calibration upload.
        if train_graph is not None:
            graph = json.loads(json.dumps(train_graph))
            layout: dict[str, Any] = synth_layout(graph)
        else:
            tid = template_id if template_id is not None else MI_DEFAULT_TRAIN_TEMPLATE
            template = _fetch_template(client, tid)
            graph = _graph_section(template, "train", tid)
            # Train snapshots carry their canvas layout at the top level.
            layout = _layout_or_none(template.get("layout")) or synth_layout(graph)
        replaced = False
        for index, node in enumerate(graph["nodes"]):
            if node.get("type") in _DATA_NODE_TYPES:
                # Preserve the node's non-data config keys (mode, paradigm,
                # nClasses, classLabels, channelNames, …) — the training engine
                # needs them: custom_data defaults to mode "both", which errors
                # without an eval split, and nClasses defaults to 4 (E2E-proven
                # 400s before this preservation). Only the data-placement keys
                # are rewired onto the recorded upload. Note custom_data has NO
                # "channels" config key (unknown keys are a 400) — the recording's
                # channel count rides inside the HDF5 itself; samplingRate is the
                # one geometry key the node carries.
                cfg = json.loads(json.dumps(node.get("config") or {}))
                cfg["source"] = {"kind": "calibration_run", "uploadId": upload_id}
                cfg["filePath"] = ""
                cfg["format"] = upload.get("format") or "hdf5"
                # Explicit graphs keep their own samplingRate (default 250 when
                # absent); template paths take the recording geometry fetched
                # from the calibration template's device node.
                cfg["samplingRate"] = (
                    (cfg.get("samplingRate") or _DEFAULT_SAMPLING_RATE)
                    if train_graph is not None
                    else sampling_rate
                )
                graph["nodes"][index] = {**node, "type": "custom_data", "config": cfg}
                replaced = True
                break
        if not replaced:
            raise McpToolError(
                "Train graph has no data node (custom_data/public_data) to rewire "
                "onto the calibration recording."
            )
        body: dict[str, Any] = {"train": graph, "stage": "train", "layout": layout}
        if name is not None:
            body["name"] = name
        payload = client.post("/api/execute", json=body)
        return {
            "trainExecutionId": payload.get("executionId"),
            "calibrationUploadId": upload_id,
            "pollHint": "Call execution.get(execution_id) to track training; "
            "execution.results(execution_id) once completed.",
        }
