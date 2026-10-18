from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import run

TRAIN_GRAPH = {
    "nodes": [
        {"id": "d", "type": "public_data", "config": {"dataset": "BNCI2014_001", "subject": "S01"}},
        {"id": "m", "type": "rxlda_sdk", "config": {}},
    ],
    "connections": [{"from": "d", "to": "m"}],
}

COMPUTE_ORIGIN = {
    "mode": "hosted",
    "nimbusVersion": "9.9.9-test",
    "desktopVersion": None,
    "mcpLocalBridge": False,
}

SUMMARY_ROW = {
    "executionId": "exec_1",
    "status": "completed",
    "startedAt": "2026-10-01T00:00:00Z",
    "pipelineName": "mi",
    "metrics": {"accuracyPct": 70.0, "kappa": 0.5},
    "hasArtifacts": True,
    "computeOrigin": COMPUTE_ORIGIN,
}

FULL_RESULT = {
    "ok": True,
    "executionId": "exec_1",
    "result": {
        "metrics": {
            "accuracyPct": 70.0,
            "evalAccuracyPct": 68.0,
            "kappa": 0.55,
            "itr": 12.5,
            "pValue": 1e-9,
            "latencyMs": 3.2,
        },
        "confusionMatrix": [[10, 2], [3, 15]],
        "confusionMatrixLabels": ["left", "right"],
        "perClassAccuracy": [{"class": "left", "accuracyPct": 83.3}],
        "artifactFiles": [{"name": "nimbus_lda.pkl"}],
        "metadata": {"big": "x" * 10_000, "computeOrigin": COMPUTE_ORIGIN},
    },
}


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    run.register(server, client)
    return server


async def test_run_pipeline_posts_train_and_returns_poll_hint():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True, "executionId": "exec_1", "message": "started"})

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH, "name": "mi test"})
    assert seen["path"] == "/api/execute"
    assert b'"train"' in seen["body"] and b'"name"' in seen["body"]
    assert result.data["executionId"] == "exec_1"
    assert "execution.get" in result.data["pollHint"]


async def test_run_pipeline_synthesizes_layout_with_positions_for_all_nodes():
    """POST /api/execute rejects bodies without layout.nodes positions for every
    node — the tool must synthesize them when the caller has no canvas."""
    import json

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.read())
        return httpx.Response(200, json={"ok": True, "executionId": "exec_1"})

    async with Client(make_server(handler)) as c:
        await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH})
    layout = seen["body"]["layout"]
    assert set(layout["nodes"]) == {"d", "m"}
    for pos in layout["nodes"].values():
        assert isinstance(pos["x"], float) and isinstance(pos["y"], float)


async def test_run_pipeline_passes_explicit_layout_through():
    import json

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.read())
        return httpx.Response(200, json={"ok": True, "executionId": "exec_1"})

    explicit = {"nodes": {"d": {"x": 1.0, "y": 2.0}, "m": {"x": 3.0, "y": 4.0}}}
    async with Client(make_server(handler)) as c:
        await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH, "layout": explicit})
    assert seen["body"]["layout"] == explicit


async def test_get_execution_passes_compute_origin_when_present():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": True, "executionId": "exec_1", "summary": SUMMARY_ROW}
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("execution.get", {"execution_id": "exec_1"})
    assert result.data["computeOrigin"] == COMPUTE_ORIGIN


async def test_get_execution_returns_summary_row():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/executions/exec_1/summary"
        return httpx.Response(
            200, json={"ok": True, "executionId": "exec_1", "summary": SUMMARY_ROW}
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("execution.get", {"execution_id": "exec_1"})
    assert result.data["status"] == "completed"
    assert result.data["metrics"]["kappa"] == 0.5


async def test_list_executions_passes_query_params():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params["status"] == "completed"
        assert req.url.params["limit"] == "5"
        return httpx.Response(200, json={"ok": True, "executions": [SUMMARY_ROW], "total": 1})

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("execution.list", {"limit": 5, "status": "completed"})
    assert result.data["executions"][0]["executionId"] == "exec_1"


async def test_get_results_trimmed_by_default():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/result/exec_1"
        return httpx.Response(200, json=FULL_RESULT)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("execution.results", {"execution_id": "exec_1"})
    assert result.data["metrics"]["kappa"] == 0.55
    assert result.data["confusionMatrix"] == [[10, 2], [3, 15]]
    assert "metadata" not in result.data  # trimmed
    assert result.data["computeOrigin"] == COMPUTE_ORIGIN


async def test_get_results_full_includes_everything():
    async with Client(make_server(lambda r: httpx.Response(200, json=FULL_RESULT))) as c:
        result = await c.call_tool("execution.results", {"execution_id": "exec_1", "full": True})
    assert "metadata" in result.data


async def test_cancel_execution_body():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True, "executionId": "exec_1", "message": None})

    async with Client(make_server(handler)) as c:
        await c.call_tool("execution.cancel", {"execution_id": "exec_1"})
    assert b'"executionId"' in seen["body"]


async def test_get_execution_rejects_bad_execution_id():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="Invalid execution id"):
            await c.call_tool("execution.get", {"execution_id": "../evil"})


async def test_get_results_rejects_bad_execution_id():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="Invalid execution id"):
            await c.call_tool("execution.results", {"execution_id": "../evil"})


async def test_cancel_execution_rejects_bad_execution_id():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="Invalid execution id"):
            await c.call_tool("execution.cancel", {"execution_id": "../evil"})


# ── duplicate in-flight run warning (non-blocking) ───────────────────────────


def _dup_handler(running_rows, snapshots=None, posts=None):
    snapshots = snapshots or {}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if req.method == "POST" and path == "/api/execute":
            if posts is not None:
                posts.append(1)
            return httpx.Response(200, json={"ok": True, "executionId": "exec_new"})
        if path == "/api/executions":
            assert req.url.params.get("status") == "running"
            return httpx.Response(
                200, json={"executions": running_rows, "total": len(running_rows)}
            )
        if path.endswith("/pipeline"):
            exec_id = path.split("/")[-2]
            if exec_id in snapshots:
                return httpx.Response(
                    200, json={"ok": True, "executionId": exec_id, "pipeline": snapshots[exec_id]}
                )
            return httpx.Response(404, json={"detail": "nope"})
        raise AssertionError((req.method, path))

    return handler


def test_graph_fingerprint_is_order_and_layout_insensitive():
    shuffled = {
        "nodes": list(reversed(TRAIN_GRAPH["nodes"])),
        "connections": TRAIN_GRAPH["connections"],
        "layout": {"nodes": {"d": {"x": 9, "y": 9}}},
    }
    assert run.graph_fingerprint(shuffled) == run.graph_fingerprint(TRAIN_GRAPH)
    changed = {
        "nodes": [
            TRAIN_GRAPH["nodes"][0],
            {"id": "m", "type": "rxlda_sdk", "config": {"x": 1}},
        ],
        "connections": TRAIN_GRAPH["connections"],
    }
    assert run.graph_fingerprint(changed) != run.graph_fingerprint(TRAIN_GRAPH)


async def test_run_without_running_executions_has_no_warning():
    async with Client(make_server(_dup_handler([]))) as c:
        r = await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH})
    assert "warning" not in r.data and r.data["executionId"] == "exec_new"


async def test_run_warns_when_identical_graph_running_per_backend_snapshot():
    snaps = {"exec_old": {"train": TRAIN_GRAPH, "subject": "S01"}}
    posts: list[int] = []
    handler = _dup_handler([{"executionId": "exec_old"}], snaps, posts)
    async with Client(make_server(handler)) as c:
        r = await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH})
    assert posts == [1], "duplicate must not block the run"
    assert r.data["executionId"] == "exec_new"
    assert r.data["duplicateOf"] == ["exec_old"]
    assert "exec_old" in r.data["warning"]


async def test_run_no_warning_for_different_graph():
    other = {"nodes": [{"id": "d", "type": "public_data", "config": {}}], "connections": []}
    handler = _dup_handler([{"executionId": "exec_old"}], {"exec_old": {"train": other}})
    async with Client(make_server(handler)) as c:
        r = await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH})
    assert "warning" not in r.data


async def test_second_identical_run_in_same_session_warns_via_registry():
    """Runs this process started are matched from memory (no snapshot fetch)."""
    graph = {"nodes": [{"id": "z", "type": "public_data", "config": {"u": 1}}], "connections": []}
    running: list[dict] = []
    handler = _dup_handler(running)
    async with Client(make_server(handler)) as c:
        first = await c.call_tool("execution.run", {"train_graph": graph})
        assert "warning" not in first.data
        running.append({"executionId": "exec_new"})  # the first run is now in flight
        second = await c.call_tool("execution.run", {"train_graph": graph})
    assert second.data["duplicateOf"] == ["exec_new"]


async def test_duplicate_preflight_failure_never_blocks_run():
    posts: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/execute":
            posts.append(1)
            return httpx.Response(200, json={"ok": True, "executionId": "exec_new"})
        return httpx.Response(500, text="boom")

    async with Client(make_server(handler)) as c:
        r = await c.call_tool("execution.run", {"train_graph": TRAIN_GRAPH})
    assert posts == [1] and "warning" not in r.data
