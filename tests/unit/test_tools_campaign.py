import asyncio
import base64
import json
import time

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import campaign

TRAIN_GRAPH = {
    "nodes": [
        {"id": "d", "type": "public_data", "config": {"dataset": "BNCI2014_001", "subject": "S01"}},
        {"id": "m", "type": "rxlda_sdk", "config": {}},
    ],
    "connections": [{"from": "d", "to": "m"}],
}


class FakeBackend:
    """Simulates the three endpoints the campaign orchestrator drives.

    - POST /api/execute: 500 "Execution queue full" on the very first attempt
      when queue_full_once is set, then {executionId} forever.
    - GET /api/executions/{id}/summary: 500 for the first summary_fail_first
      attempts across all executions, then completed.
    - GET /api/result/{id}: {result: {metrics}} from the metrics map, or 500
      for execution ids listed in result_fail.
    """

    def __init__(
        self, *, metrics=None, result_fail=(), queue_full_once=False, summary_fail_first=0
    ):
        self.metrics = metrics or {}
        self.result_fail = set(result_fail)
        self.queue_full_once = queue_full_once
        self.summary_fail_first = summary_fail_first
        self.execute_attempts = 0
        self.summary_attempts = 0
        self._failed_once = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/execute":
            self.execute_attempts += 1
            if self.queue_full_once and not self._failed_once:
                self._failed_once = True
                return httpx.Response(500, json={"detail": "Execution queue full"})
            return httpx.Response(
                200, json={"executionId": f"exec_{self.execute_attempts}", "message": "started"}
            )
        if path.startswith("/api/executions/") and path.endswith("/summary"):
            self.summary_attempts += 1
            if self.summary_attempts <= self.summary_fail_first:
                return httpx.Response(500, json={"detail": "summary unavailable"})
            exec_id = path.split("/")[3]
            return httpx.Response(
                200, json={"summary": {"executionId": exec_id, "status": "completed"}}
            )
        if path.startswith("/api/result/"):
            exec_id = path.split("/")[3]
            if exec_id in self.result_fail:
                return httpx.Response(500, json={"detail": "result unavailable"})
            return httpx.Response(200, json={"result": {"metrics": self.metrics.get(exec_id, {})}})
        raise AssertionError(path)


def make_server(backend: FakeBackend) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=None),  # type: ignore[arg-type]
        transport=httpx.MockTransport(backend.handler),
    )
    server = FastMCP("t")
    campaign.register(server, client)
    return server


def expired_token_client(handler) -> NimbusClient:
    """Token-mode client whose hosted JWT expired 2 days ago (the day-count
    decode in token_rejected_guidance is local, so an unsigned fake works)."""

    def b64(obj: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    token = "nimb_" + f"{b64({'alg': 'none'})}.{b64({'exp': int(time.time()) - 2 * 86400})}.sig"
    return NimbusClient(
        McpConfig(api_url="http://t", mcp_key="", export_dir=None, nimbus_token=token),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )


def make_token_server(handler) -> FastMCP:
    server = FastMCP("t")
    campaign.register(server, expired_token_client(handler))
    return server


async def wait_for_terminal(c: Client, experiment_id: str, timeout: float = 10.0):
    """Poll experiment.get until the experiment leaves 'running' (<= timeout)."""
    deadline = asyncio.get_running_loop().time() + timeout
    snapshot = None
    while asyncio.get_running_loop().time() < deadline:
        snapshot = await c.call_tool("experiment.get", {"experiment_id": experiment_id})
        if snapshot.data["status"] != "running":
            return snapshot
        await asyncio.sleep(0.05)
    return snapshot


async def test_run_experiment_single_run_completes_with_aggregates(monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.5, "accuracyPct": 70.0}})
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        assert result.data["experimentId"]
        assert result.data["runs"][0]["name"] == "r1"

        snapshot = await wait_for_terminal(c, result.data["experimentId"])
        assert snapshot.data["status"] == "completed"
        run = snapshot.data["runs"][0]
        assert run["executionId"] == "exec_1"
        assert run["status"] == "completed"
        assert run["metrics"]["kappa"] == 0.5
        kappa = snapshot.data["aggregates"]["kappa"]
        assert kappa["mean"] == pytest.approx(0.5)
        assert kappa["std"] is None  # single value: std needs >= 2
        assert kappa["best"] == {"name": "r1", "value": 0.5}


async def test_run_experiment_retries_queue_full(monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    monkeypatch.setattr(campaign, "QUEUE_FULL_BACKOFF_SEC", 0.02)
    backend = FakeBackend(queue_full_once=True, metrics={"exec_2": {"kappa": 0.5}})
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert snapshot.data["status"] == "completed"
    assert backend.execute_attempts == 2  # one queue-full rejection, one success


async def test_run_experiment_retries_transient_summary_500s(monkeypatch):
    """A summary poll that 500s twice then succeeds must not fail the run:
    the retry helper absorbs the transient errors (3 attempts total) and the
    experiment still completes with the run's metrics."""
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    monkeypatch.setattr(campaign, "POLL_RETRY_BACKOFF_SEC", 0.01)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.55}}, summary_fail_first=2)
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert snapshot.data["status"] == "completed"
    run = snapshot.data["runs"][0]
    assert run["status"] == "completed"
    assert run["metrics"]["kappa"] == 0.55
    assert backend.summary_attempts == 3  # two 500s absorbed, third attempt succeeded


async def test_run_experiment_validates_run_count():
    async with Client(make_server(FakeBackend())) as c:
        with pytest.raises(Exception, match="at least 1"):
            await c.call_tool("experiment.run", {"runs": []})
        with pytest.raises(Exception, match="at most 25"):
            await c.call_tool(
                "experiment.run",
                {"runs": [{"name": f"r{i}", "train_graph": TRAIN_GRAPH} for i in range(26)]},
            )


async def test_run_experiment_aggregates_exclude_failed_runs(monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(
        metrics={"exec_1": {"kappa": 0.4}, "exec_2": {"kappa": 0.6}},
        result_fail={"exec_3"},
    )
    runs = [
        {"name": "r_a", "train_graph": TRAIN_GRAPH},
        {"name": "r_b", "train_graph": TRAIN_GRAPH},
        {"name": "r_c", "train_graph": TRAIN_GRAPH},
    ]
    async with Client(make_server(backend)) as c:
        result = await c.call_tool("experiment.run", {"runs": runs})
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert snapshot.data["status"] == "completed"  # not all runs failed
    by_name = {row["name"]: row for row in snapshot.data["runs"]}
    assert by_name["r_a"]["status"] == "completed"
    assert by_name["r_b"]["status"] == "completed"
    assert by_name["r_c"]["status"] == "failed"
    assert by_name["r_c"]["error"]
    assert "metrics" not in by_name["r_c"]
    kappa = snapshot.data["aggregates"]["kappa"]
    assert kappa["mean"] == pytest.approx(0.5)
    assert kappa["std"] == pytest.approx(0.1)  # population std of [0.4, 0.6]
    assert kappa["best"] == {"name": "r_b", "value": 0.6}


async def test_get_experiment_unknown_id():
    async with Client(make_server(FakeBackend())) as c:
        with pytest.raises(Exception, match="Unknown experiment"):
            await c.call_tool("experiment.get", {"experiment_id": "nope"})


async def test_expired_token_on_summary_poll_is_not_retried(monkeypatch):
    """A 401 on the summary poll (hosted token expired mid-experiment) is
    SetupRequired, not a transient 5xx: _poll_summary must re-raise it
    immediately — no pointless 3-attempt retry cycle — and the run error
    carries the token-expiry / re-login story."""
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    seen = {"summary": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/execute":
            return httpx.Response(200, json={"executionId": "exec_1", "message": "started"})
        if path.endswith("/summary"):
            seen["summary"] += 1
            return httpx.Response(401, json={"detail": "no"})
        raise AssertionError(path)

    async with Client(make_token_server(handler)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert seen["summary"] == 1  # re-raised at once: retries would have hit 3
    run = snapshot.data["runs"][0]
    assert run["status"] == "failed"
    assert "expired 2 day" in run["error"]
    assert "nimbus-mcp login" in run["error"]


async def test_expired_token_on_result_fetch_is_not_result_fetch_failed(monkeypatch):
    """Same guard on _fetch_metrics: a 401 while fetching results must surface
    the token-expiry story, not the misleading 'run completed but result fetch
    failed' (a setup problem is not a broken result)."""
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/execute":
            return httpx.Response(200, json={"executionId": "exec_1", "message": "started"})
        if path.endswith("/summary"):
            return httpx.Response(
                200, json={"summary": {"executionId": "exec_1", "status": "completed"}}
            )
        if path.startswith("/api/result/"):
            return httpx.Response(401, json={"detail": "no"})
        raise AssertionError(path)

    async with Client(make_token_server(handler)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    run = snapshot.data["runs"][0]
    assert run["status"] == "failed"
    assert "expired 2 day" in run["error"]
    assert "nimbus-mcp login" in run["error"]
    assert "result fetch failed" not in run["error"]
