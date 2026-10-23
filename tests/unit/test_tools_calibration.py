"""calibration.* tools — MockTransport tests (house style of test_tools_live.py).

Asserts against what the handler RECEIVES (captured request bodies), not mock
internals. Backend contracts pinned here (verified against nimbus_backend):
- POST /api/execute body is {"train": graph, "stage": ..., "layout": ...} —
  PipelineExecuteRequest is extra="forbid" and has NO "calibrate" key; the
  calibrate graph rides the same "train" field with stage="calibrate".
- POST /api/pause-calibration / resume-calibration take {"executionId"}.
- GET /api/calibration-live/{id} terminal shape: phase "complete" +
  calibration {uploadId, path, filename, format}.
"""

import json
from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import calibration

PARADIGM_TEMPLATE = {
    "template": {
        "id": "mi_hardware_calibration",
        "name": "MI hardware calibration",
        "version": "3",
        "calibrate": {
            "nodes": [
                {
                    "id": "device",
                    "type": "hardware_device",
                    "config": {
                        "buildMode": "live_record",
                        "deviceType": "synthetic",
                        "channels": 8,
                        "samplingRate": 250,
                    },
                },
                {
                    "id": "protocol",
                    "type": "trial_protocol",
                    "config": {
                        "paradigm": "motor_imagery",
                        "trialsPerClass": 40,
                        "classes": [{"id": 0, "label": "Left Hand", "cue": "← LEFT"}],
                    },
                },
                {
                    "id": "recorder",
                    "type": "calibration_recorder",
                    "config": {"datasetName": "calibration", "format": "hdf5"},
                },
            ],
            "connections": [
                {"from": "device", "to": "protocol"},
                {"from": "protocol", "to": "recorder"},
            ],
        },
    }
}

# Mirrors the real mi_headband_csp_lda template snapshot: the first node is a
# custom_data node pinning the latest calibration run. The non-data config keys
# (mode/paradigm/nClasses/classLabels) are the ones calibration.train must
# PRESERVE when rewiring — custom_data defaults to mode "both" (errors without
# an eval split) and nClasses 4 (E2E-proven failures before preservation).
TRAIN_TEMPLATE = {
    "ok": True,
    "template": {
        "id": "mi_headband_csp_lda",
        "name": "MI headband CSP+LDA",
        "version": "3",
        "train": {
            "nodes": [
                {
                    "id": "data",
                    "type": "custom_data",
                    "config": {
                        "mode": "all",
                        "paradigm": "motor_imagery",
                        "nClasses": 2,
                        "classLabels": ["left", "right"],
                        "format": "hdf5",
                        "samplingRate": 250.0,
                        "source": {"kind": "calibration_last_run"},
                    },
                },
                {"id": "csp", "type": "csp", "config": {"cspComponents": 4}},
                {"id": "lda", "type": "rxlda_sdk", "config": {}},
            ],
            "connections": [
                {"from": "data", "to": "csp"},
                {"from": "csp", "to": "lda"},
            ],
        },
        "layout": {
            "nodes": {
                "data": {"x": 0, "y": 80},
                "csp": {"x": 320, "y": 80},
                "lda": {"x": 640, "y": 80},
            }
        },
        "calibrate": None,
    },
}

# Mirrors the real p300_hardware_calibration template (storage/templates/
# calibration/p300_hardware_calibration.yaml): hardware_device node with the
# recording geometry calibration.train reads for the default-template path.
P300_CALIBRATION_TEMPLATE = {
    "template": {
        "id": "p300_hardware_calibration",
        "name": "P300 hardware calibration",
        "version": "3",
        "calibrate": {
            "nodes": [
                {
                    "id": "device",
                    "type": "hardware_device",
                    "config": {
                        "buildMode": "live_record",
                        "deviceType": "synthetic",
                        "channels": 8,
                        "samplingRate": 250,
                    },
                },
                {
                    "id": "protocol",
                    "type": "trial_protocol",
                    "config": {"paradigm": "p300"},
                },
                {
                    "id": "recorder",
                    "type": "calibration_recorder",
                    "config": {"format": "hdf5"},
                },
            ],
            "connections": [],
        },
    }
}


def _p300_calibration_yaml_node_types() -> list[str]:
    yaml_path = (
        Path(__file__).resolve().parents[3]
        / "backend-py"
        / "nimbus_backend"
        / "storage"
        / "templates"
        / "calibration"
        / "p300_hardware_calibration.yaml"
    )
    text = yaml_path.read_text(encoding="utf-8")
    calibrate, _, _rest = text.partition("\ntrain:")
    types: list[str] = []
    in_nodes = False
    for line in calibrate.splitlines():
        if line.startswith("  nodes:"):
            in_nodes = True
            continue
        if in_nodes and line.startswith("  connections:"):
            break
        if in_nodes and line.strip().startswith("type:"):
            types.append(line.split("type:", 1)[1].strip())
    return types


def test_p300_calibration_mock_node_types_match_template():
    mock_types = [
        node["type"] for node in P300_CALIBRATION_TEMPLATE["template"]["calibrate"]["nodes"]
    ]
    assert mock_types == _p300_calibration_yaml_node_types()
    assert mock_types == ["hardware_device", "trial_protocol", "calibration_recorder"]


# Mirrors the real p300_hardware_xdawn_lda template (storage/templates/p300/
# p300_hardware_xdawn_lda.yaml): the first node (id "data") is a custom_data
# node pinning the latest calibration run, carrying the p300 non-data config
# (mode: all, paradigm: p300, nClasses: 2, classLabels non_target/target) the
# rewire must PRESERVE.
P300_TRAIN_TEMPLATE = {
    "ok": True,
    "template": {
        "id": "p300_hardware_xdawn_lda",
        "name": "P300 (Headband, Adaptive) - xDAWN → Nimbus LDA",
        "version": "3",
        "train": {
            "nodes": [
                {
                    "id": "data",
                    "type": "custom_data",
                    "config": {
                        "mode": "all",
                        "paradigm": "p300",
                        "nClasses": 2,
                        "classLabels": ["non_target", "target"],
                        "format": "hdf5",
                        "samplingRate": 250.0,
                        "source": {"kind": "calibration_last_run"},
                    },
                },
                {"id": "xdawn", "type": "xdawn", "config": {"nComponents": 2, "reg": 0}},
                {"id": "lda", "type": "rxlda_sdk", "config": {}},
            ],
            "connections": [
                {"from": "data", "to": "xdawn"},
                {"from": "xdawn", "to": "lda"},
            ],
        },
        "layout": {
            "nodes": {
                "data": {"x": 0.0, "y": 80.0},
                "xdawn": {"x": 640.0, "y": 80.0},
                "lda": {"x": 800.0, "y": 80.0},
            }
        },
        "calibrate": None,
    },
}

COMPLETED_LIVE = {
    "ok": True,
    "executionId": "exec_1",
    "status": "completed",
    "phase": "complete",
    "calibration": {
        "uploadId": "up_9",
        "path": "/uploads/cal.h5",
        "filename": "cal.h5",
        "format": "hdf5",
    },
}

RUNNING_LIVE = {
    "ok": True,
    "executionId": "exec_2",
    "status": "running",
    "phase": "imagery",
    "currentTrial": {"index": 3, "total": 10, "class": "left", "cue": "← LEFT"},
    "progress": {"trialsDone": 2, "trialsPlanned": 10, "perClass": {"left": 1, "right": 1}},
    "paused": False,
    "paradigm": "motor_imagery",
    "deviceConnected": True,
    "elapsedSec": 12.0,
    "estimatedRemainingSec": 28.0,
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
    calibration.register(server, client)
    return server


async def test_start_refuses_without_confirm():
    calls: list[str] = []
    async with Client(
        make_server(lambda r: (_ for _ in ()).throw(AssertionError("HTTP!")), calls=calls)
    ) as c:
        result = await c.call_tool("calibration.start", {"paradigm": "mi"})
    assert result.data["started"] is False and "confirm" in result.data["message"]
    assert calls == []  # rail: zero HTTP calls without confirm


async def test_start_patches_device_and_trials():
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/templates/mi_hardware_calibration":
            return httpx.Response(200, json=PARADIGM_TEMPLATE)
        if req.url.path == "/api/execute":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_1"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "calibration.start",
            {
                "paradigm": "mi",
                "confirm": True,
                "device_type": "brainbit",
                "trials_per_class": 5,
                "name": "smoke",
            },
        )
    body = json.loads(seen["/api/execute"].read())
    assert body["stage"] == "calibrate"
    # The graph rides the "train" key (PipelineExecuteRequest has no calibrate key).
    graph = body["train"]
    assert "layout" not in graph  # inner layout stripped — ExecGraphSnapshot is extra=forbid
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["device"]["type"] == "hardware_device"
    assert nodes["device"]["config"]["deviceType"] == "brainbit"
    assert nodes["device"]["config"]["buildMode"] == "live_record"
    assert nodes["protocol"]["config"]["trialsPerClass"] == 5
    assert body["layout"]["nodes"]
    assert set(body["layout"]["nodes"]) == {"device", "protocol", "recorder"}
    assert body["name"] == "smoke"
    assert result.data["started"] is True and result.data["executionId"] == "exec_1"
    # trialsPlanned = patched trialsPerClass (5) x len(template classes) (1).
    assert result.data["trialsPlanned"] == 5


async def test_start_reports_trials_planned_from_template():
    """No overrides: trialsPlanned reads the TEMPLATE protocol node (40 x 1)."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/templates/mi_hardware_calibration":
            return httpx.Response(200, json=PARADIGM_TEMPLATE)
        if req.url.path == "/api/execute":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_1"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("calibration.start", {"paradigm": "mi", "confirm": True})
    assert result.data["trialsPlanned"] == 40  # 40 trialsPerClass x 1 template class


async def test_start_with_train_only_template_fallback():
    """Backend serving calibration graph in 'train' section (desktop app builds) works."""
    template_without_calibrate = {
        "template": {
            "id": "mi_hardware_calibration",
            "name": "MI hardware calibration",
            "version": "3",
            "train": PARADIGM_TEMPLATE["template"]["calibrate"],
            "layout": {
                "nodes": {
                    "device": {"x": 0.0, "y": 80.0},
                    "protocol": {"x": 380.0, "y": 80.0},
                    "recorder": {"x": 760.0, "y": 80.0},
                }
            },
        }
    }
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/templates/mi_hardware_calibration":
            return httpx.Response(200, json=template_without_calibrate)
        if req.url.path == "/api/execute":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_1"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "calibration.start",
            {"paradigm": "mi", "confirm": True, "trials_per_class": 5},
        )
    assert result.data["started"] is True
    assert result.data["executionId"] == "exec_1"
    body = json.loads(seen["/api/execute"].read())
    assert body["stage"] == "calibrate"
    assert len(body["train"]["nodes"]) == 3
    assert body["layout"]["nodes"]["device"] == {"x": 0.0, "y": 80.0}


async def test_start_refuses_low_trials_per_class():
    async with Client(make_server(lambda r: None)) as c:
        with pytest.raises(Exception, match="at least 5"):
            await c.call_tool(
                "calibration.start",
                {"paradigm": "mi", "confirm": True, "trials_per_class": 3},
            )


async def test_status_merges_complete_upload():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/calibration-live/exec_1":
            return httpx.Response(200, json=COMPLETED_LIVE)
        if req.url.path == "/api/calibration-live/exec_2":
            return httpx.Response(200, json=RUNNING_LIVE)
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        done = await c.call_tool("calibration.status", {"execution_id": "exec_1"})
        running = await c.call_tool("calibration.status", {"execution_id": "exec_2"})
    assert done.data["phase"] == "complete"
    assert done.data["calibration"]["uploadId"] == "up_9"
    assert "calibration.train" in done.data["nextHint"]
    assert running.data["phase"] == "imagery"
    assert running.data["currentTrial"]["cue"] == "← LEFT"
    assert "nextHint" not in running.data  # hint only once the recording exists


async def test_status_complete_without_recording_hints_unavailable():
    """Complete but calibration null (nothing linked, or a pre-linkage
    desktop-local recording): the hint must say the recording is unavailable —
    NOT point at train, which needs the uploadId."""
    no_upload_live = dict(COMPLETED_LIVE, calibration=None)

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/calibration-live/exec_1"
        return httpx.Response(200, json=no_upload_live)

    async with Client(make_server(handler)) as c:
        done = await c.call_tool("calibration.status", {"execution_id": "exec_1"})
    assert done.data["phase"] == "complete"
    assert done.data["calibration"] is None
    assert "calibration.train" not in done.data["nextHint"]
    assert "no recording" in done.data["nextHint"]


async def test_pause_resume_post_camelcase():
    seen: dict[str, bytes] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req.read()
        if req.url.path == "/api/pause-calibration":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_1", "paused": True})
        if req.url.path == "/api/resume-calibration":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_1", "paused": False})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        paused = await c.call_tool("calibration.pause", {"execution_id": "exec_1"})
        resumed = await c.call_tool("calibration.resume", {"execution_id": "exec_1"})
    assert json.loads(seen["/api/pause-calibration"]) == {"executionId": "exec_1"}
    assert json.loads(seen["/api/resume-calibration"]) == {"executionId": "exec_1"}
    assert paused.data["paused"] is True and resumed.data["paused"] is False


def _train_backend(seen: dict[str, httpx.Request]):
    """Handler for calibration.train's default path: live → cal template → train
    template → execute."""

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/calibration-live/exec_1":
            return httpx.Response(200, json=COMPLETED_LIVE)
        if req.url.path == "/api/templates/mi_hardware_calibration":
            return httpx.Response(200, json=PARADIGM_TEMPLATE)
        if req.url.path == "/api/templates/mi_headband_csp_lda":
            return httpx.Response(200, json=TRAIN_TEMPLATE)
        if req.url.path == "/api/execute":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_t"})
        raise AssertionError(req.url.path)

    return handler


async def test_train_wires_calibration_run_ref():
    seen: dict[str, httpx.Request] = {}

    async with Client(make_server(_train_backend(seen))) as c:
        result = await c.call_tool(
            "calibration.train", {"execution_id": "exec_1", "name": "subject model"}
        )
    body = json.loads(seen["/api/execute"].read())
    assert body["stage"] == "train"
    graph = body["train"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["data"]["type"] == "custom_data"
    # The rewire PRESERVES the template node's non-data config (mode/paradigm/
    # nClasses/classLabels — the training engine needs them) and replaces only
    # the data placement: source pinned to the upload, filePath cleared,
    # format/samplingRate from the recording geometry. No "channels" key —
    # custom_data has none (unknown keys are a 400); the count rides in the HDF5.
    assert nodes["data"]["config"] == {
        "mode": "all",
        "paradigm": "motor_imagery",
        "nClasses": 2,
        "classLabels": ["left", "right"],
        "format": "hdf5",
        "samplingRate": 250,  # recording geometry (calibrate template device node)
        "source": {"kind": "calibration_run", "uploadId": "up_9"},
        "filePath": "",
    }
    # Connections and node ids survive the data-node replacement.
    assert graph["connections"] == TRAIN_TEMPLATE["template"]["train"]["connections"]
    assert body["layout"]["nodes"]["data"] == {"x": 0.0, "y": 80.0}
    assert body["name"] == "subject model"
    assert result.data["trainExecutionId"] == "exec_t"
    assert "execution.get" in result.data["pollHint"]


async def test_train_p300_default_template():
    """p300 train WITHOUT template_id resolves the default: fetches the p300
    calibration template (recording geometry), the p300_hardware_xdawn_lda
    train template, rewires its first custom_data node onto the recorded
    upload (preserving the p300 non-data config), and posts execute."""
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/calibration-live/exec_1":
            return httpx.Response(200, json=COMPLETED_LIVE)
        if req.url.path == "/api/templates/p300_hardware_calibration":
            return httpx.Response(200, json=P300_CALIBRATION_TEMPLATE)
        if req.url.path == "/api/templates/p300_hardware_xdawn_lda":
            return httpx.Response(200, json=P300_TRAIN_TEMPLATE)
        if req.url.path == "/api/execute":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_p"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "calibration.train", {"execution_id": "exec_1", "paradigm": "p300"}
        )
    # Exactly the default-path fetches: live → calibration template → default
    # train template → execute.
    assert list(seen) == [
        "/api/calibration-live/exec_1",
        "/api/templates/p300_hardware_calibration",
        "/api/templates/p300_hardware_xdawn_lda",
        "/api/execute",
    ]
    body = json.loads(seen["/api/execute"].read())
    assert body["stage"] == "train"
    graph = body["train"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["data"]["type"] == "custom_data"
    # Rewire: source pinned to the upload; non-data config (mode/paradigm/
    # nClasses/classLabels) preserved; samplingRate from the calibration
    # template's device node (250), not the yaml's own 250.0 float.
    assert nodes["data"]["config"] == {
        "mode": "all",
        "paradigm": "p300",
        "nClasses": 2,
        "classLabels": ["non_target", "target"],
        "format": "hdf5",
        "samplingRate": 250,
        "source": {"kind": "calibration_run", "uploadId": "up_9"},
        "filePath": "",
    }
    assert graph["connections"] == P300_TRAIN_TEMPLATE["template"]["train"]["connections"]
    assert body["layout"]["nodes"]["data"] == {"x": 0.0, "y": 80.0}
    assert result.data["trainExecutionId"] == "exec_p"
    assert result.data["calibrationUploadId"] == "up_9"


async def test_train_sart_without_template_names_defaulted_paradigms():
    """sart/target_hit have NO train templates anywhere (their calibration
    templates ship empty train: sections) — they keep the explicit-template
    requirement. The error message must truthfully name BOTH defaulted
    paradigms (mi AND p300)."""
    async with Client(make_server(lambda r: (_ for _ in ()).throw(AssertionError("HTTP!")))) as c:
        with pytest.raises(Exception) as ei:
            await c.call_tool("calibration.train", {"execution_id": "exec_1", "paradigm": "sart"})
        with pytest.raises(Exception) as ei_target:
            await c.call_tool(
                "calibration.train", {"execution_id": "exec_1", "paradigm": "target_hit"}
            )
    message = str(ei.value)
    assert "No default train template for paradigm 'sart'" in message
    assert "template_id" in message
    assert "mi_headband_csp_lda" in message
    assert "p300_hardware_xdawn_lda" in message
    assert "p300_hardware_xdawn_lda" in str(ei_target.value)


async def test_train_explicit_graph_skips_calibration_template_fetch():
    """An explicit train_graph carries its own geometry — the paradigm
    calibration template fetch (recording sampling rate) is only needed for
    the default-template path. Exactly two HTTP calls: live snapshot, execute."""
    calls: list[str] = []
    execute_body: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/calibration-live/exec_1":
            return httpx.Response(200, json=COMPLETED_LIVE)
        if req.url.path == "/api/execute":
            execute_body.update(json.loads(req.read()))
            return httpx.Response(200, json={"ok": True, "executionId": "exec_t"})
        raise AssertionError(f"unexpected template fetch: {req.url.path}")

    explicit = {
        "nodes": [
            # Non-custom data node type is rewired to custom_data; the node's
            # own samplingRate (500) must survive the rewire.
            {"id": "src", "type": "public_data", "config": {"mode": "all", "samplingRate": 500}},
            {"id": "csp", "type": "csp", "config": {"cspComponents": 4}},
        ],
        "connections": [{"from": "src", "to": "csp"}],
    }

    async with Client(make_server(handler, calls=calls)) as c:
        result = await c.call_tool(
            "calibration.train", {"execution_id": "exec_1", "train_graph": explicit}
        )
    assert calls == ["/api/calibration-live/exec_1", "/api/execute"]
    assert result.data["trainExecutionId"] == "exec_t"
    graph = execute_body["train"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    assert nodes["src"]["type"] == "custom_data"
    assert nodes["src"]["config"]["source"] == {"kind": "calibration_run", "uploadId": "up_9"}
    assert nodes["src"]["config"]["filePath"] == ""
    # The explicit graph's own samplingRate survives (no template fetch to
    # override it with).
    assert nodes["src"]["config"]["samplingRate"] == 500
    assert nodes["src"]["config"]["mode"] == "all"


async def test_train_refuses_incomplete_session():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/calibration-live/exec_2"
        return httpx.Response(200, json=RUNNING_LIVE)

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="calibration.status"):
            await c.call_tool("calibration.train", {"execution_id": "exec_2"})


# ── connection canonicalization ────────────────────────────────────────────
# The execute request's GraphConnection is extra="forbid" with only {from, to}
# (backend contracts/pipeline.py), but canvas-style templates (sart,
# target_hit) carry sourceHandle/targetHandle on their connections. The graph
# section must canonicalize or calibration.start 422s before any route code.

HANDLE_TEMPLATE = {
    "template": {
        "id": "sart_calibration",
        "name": "SART calibration",
        "version": "3",
        "calibrate": {
            "nodes": [
                {"id": "device", "type": "hardware_device", "config": {}},
                {"id": "recorder", "type": "calibration_recorder", "config": {}},
            ],
            "connections": [
                {
                    "from": "device",
                    "to": "recorder",
                    "sourceHandle": "sidecar_meta",
                    "targetHandle": "behavioral_sidecar",
                }
            ],
        },
    }
}


async def test_start_posts_clean_connections_for_handle_templates():
    """sart/target_hit-style templates (canvas handles on connections) must
    start: the posted train graph carries connections as {from, to} only."""
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/templates/sart_calibration":
            return httpx.Response(200, json=HANDLE_TEMPLATE)
        if req.url.path == "/api/execute":
            return httpx.Response(200, json={"ok": True, "executionId": "exec_sart"})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("calibration.start", {"paradigm": "sart", "confirm": True})
    assert result.data["started"] is True and result.data["executionId"] == "exec_sart"
    graph = json.loads(seen["/api/execute"].read())["train"]
    assert graph["connections"] == [{"from": "device", "to": "recorder"}]


def test_graph_section_strips_connection_handles():
    section = calibration._graph_section(
        HANDLE_TEMPLATE["template"], "calibrate", "sart_calibration"
    )
    assert section["connections"] == [{"from": "device", "to": "recorder"}]


def test_graph_section_rejects_connection_missing_endpoints():
    """A connection without from/to is a broken template: raise rather than
    silently drop the edge — a missing edge would train a wrong pipeline."""
    broken = {
        "nodes": [{"id": "a", "type": "x", "config": {}}],
        "connections": [{"from": "a", "sourceHandle": "h"}],
    }
    with pytest.raises(calibration.McpToolError, match="connection"):
        calibration._graph_section({"calibrate": broken}, "calibrate", "t")
