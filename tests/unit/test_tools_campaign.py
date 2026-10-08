import asyncio
import base64
import json
import threading
import time

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import HeaderCredentialClient
from nimbus_mcp.tools import campaign

TRAIN_GRAPH = {
    "nodes": [
        {"id": "d", "type": "public_data", "config": {"dataset": "BNCI2014_001", "subject": "S01"}},
        {"id": "m", "type": "rxlda_sdk", "config": {}},
    ],
    "connections": [{"from": "d", "to": "m"}],
}


class FakeBackend:
    """Simulates the three endpoints the campaign orchestrator drives, on the
    REAL backend wire contract (pinned against nimbus_backend):

    - POST /api/execute ALWAYS answers 200 {executionId} immediately — the
      concurrency gate runs later, inside the execution task.
    - GET /api/executions/{id}/summary: 500 for the first summary_fail_first
      attempts across all executions, then completed — except ids in
      queue_full_ids, which answer failed + errorMessage "Execution queue
      full" once (executor.py's queue-gate write), matching
      queue_full_times=1. For queue_full_times > 1 the first N executions'
      summaries fail queue-full in a row.
    - POST may omit executionId when drop_exec_id_first is set (first call
      only): a malformed-200 the submit path must treat as a single failed
      run, not a whole-experiment abort.
    - GET /api/result/{id}: {result: {metrics}} from the metrics map, or 500
      for execution ids listed in result_fail.
    """

    def __init__(
        self,
        *,
        metrics=None,
        result_fail=(),
        queue_full_times=0,
        summary_fail_first=0,
        drop_exec_id_first=False,
    ):
        self.metrics = metrics or {}
        self.result_fail = set(result_fail)
        self.queue_full_times = queue_full_times
        self.summary_fail_first = summary_fail_first
        self.drop_exec_id_first = drop_exec_id_first
        self.execute_attempts = 0
        self.summary_attempts = 0
        self.queue_full_ids: set[str] = set()

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/execute":
            self.execute_attempts += 1
            exec_id = f"exec_{self.execute_attempts}"
            if self.execute_attempts <= self.queue_full_times:
                self.queue_full_ids.add(exec_id)
            if self.drop_exec_id_first and self.execute_attempts == 1:
                return httpx.Response(200, json={"message": "started"})
            return httpx.Response(200, json={"executionId": exec_id, "message": "started"})
        if path.startswith("/api/executions/") and path.endswith("/summary"):
            self.summary_attempts += 1
            if self.summary_attempts <= self.summary_fail_first:
                return httpx.Response(500, json={"detail": "summary unavailable"})
            exec_id = path.split("/")[3]
            if exec_id in self.queue_full_ids:
                self.queue_full_ids.discard(exec_id)  # one queue-full poll per id
                return httpx.Response(
                    200,
                    json={
                        "summary": {
                            "executionId": exec_id,
                            "status": "failed",
                            "errorMessage": "Execution queue full",
                        }
                    },
                )
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
    """Queue-full surfaces via the polled summary (status failed +
    errorMessage), NEVER as an HTTP error — POST /api/execute always answers
    200 immediately. The worker must resubmit the run on a queue-full
    summary and succeed."""
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    monkeypatch.setattr(campaign, "QUEUE_FULL_BACKOFF_SEC", 0.02)
    backend = FakeBackend(queue_full_times=1, metrics={"exec_2": {"kappa": 0.5}})
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert snapshot.data["status"] == "completed"
    run = snapshot.data["runs"][0]
    assert run["executionId"] == "exec_2"  # exec_1 died queue-full; resubmitted
    assert run["metrics"]["kappa"] == 0.5
    assert backend.execute_attempts == 2


async def test_run_experiment_fails_run_after_queue_full_cap(monkeypatch):
    """Queue-full forever: the run fails with the queue-full error after
    MAX_SUBMIT_ATTEMPTS submissions (1 initial + retries), and the experiment
    finalizes failed — not stuck running."""
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    monkeypatch.setattr(campaign, "QUEUE_FULL_BACKOFF_SEC", 0.02)
    backend = FakeBackend(queue_full_times=campaign.MAX_SUBMIT_ATTEMPTS)
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert snapshot.data["status"] == "failed"
    run = snapshot.data["runs"][0]
    assert run["status"] == "failed"
    assert campaign._QUEUE_FULL_MARKER in (run["error"] or "")
    assert backend.execute_attempts == campaign.MAX_SUBMIT_ATTEMPTS


async def test_run_experiment_missing_execution_id_fails_run_only(monkeypatch):
    """A 200 response without an executionId must fail THAT run (a malformed
    backend answer), not poison the experiment with a "None" execution id
    that 404s every poll and aborts the remaining runs."""
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(drop_exec_id_first=True, metrics={"exec_2": {"kappa": 0.6}})
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run",
            {
                "runs": [
                    {"name": "bad", "train_graph": TRAIN_GRAPH},
                    {"name": "good", "train_graph": TRAIN_GRAPH},
                ]
            },
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    runs = {run["name"]: run for run in snapshot.data["runs"]}
    assert runs["bad"]["status"] == "failed"
    assert runs["bad"]["executionId"] is None
    assert "executionId" in (runs["bad"]["error"] or "")
    assert runs["good"]["status"] == "completed"  # sibling run unaffected
    assert runs["good"]["metrics"]["kappa"] == 0.6


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


# ── hosted gateway: the worker thread has no request context ───────────────
# fastmcp's get_http_headers() answers {} outside a request, so a
# HeaderCredentialClient cannot resolve its per-request token inside the
# worker. experiment.run must bind a token-owned client (bound_client) before
# spawning; otherwise every run fails with the gateway setup guidance even
# though the caller sent the header.


def make_header_server(backend: FakeBackend) -> FastMCP:
    """campaign server behind a HeaderCredentialClient whose header getter
    answers everywhere EXCEPT the experiment worker thread (where fastmcp
    reports no request)."""

    def getter() -> dict[str, str]:
        if threading.current_thread().name.startswith("nimbus-experiment"):
            return {}
        return {"x-nimbus-token": "nimb_campaign"}

    def factory(cfg):
        return NimbusClient(cfg, transport=httpx.MockTransport(backend.handler))

    proxy = HeaderCredentialClient(
        McpConfig(api_url="http://t", mcp_key="", export_dir=None),  # type: ignore[arg-type]
        client_factory=factory,
        headers_getter=getter,
    )
    server = FastMCP("t")
    campaign.register(server, proxy)
    return server


async def test_experiment_run_survives_worker_without_request_context(monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.5}})
    async with Client(make_header_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    run = snapshot.data["runs"][0]
    assert run["status"] == "completed", run
    assert run["metrics"]["kappa"] == 0.5


# ── per-credential scoping: the registry must not leak across users ────────
# EXPERIMENTS is process-global, and a shared gateway authenticates many
# users per process: an experiment id registered under token A must be
# invisible to token B, or any authenticated gateway user could poll another
# user's sweep by guessing/seeing the id.


def make_two_tenant_server(backend: FakeBackend) -> tuple[FastMCP, dict]:
    """campaign server behind a HeaderCredentialClient whose token is chosen
    per call via ``state["token"]`` (the worker thread sees no headers, like
    on the real gateway), so one test client can act as two different users."""

    state = {"token": "nimb_alice"}

    def getter() -> dict[str, str]:
        if threading.current_thread().name.startswith("nimbus-experiment"):
            return {}
        return {"x-nimbus-token": state["token"]}

    def factory(cfg):
        return NimbusClient(cfg, transport=httpx.MockTransport(backend.handler))

    proxy = HeaderCredentialClient(
        McpConfig(api_url="http://t", mcp_key="", export_dir=None),  # type: ignore[arg-type]
        client_factory=factory,
        headers_getter=getter,
    )
    server = FastMCP("t")
    campaign.register(server, proxy)
    return server, state


async def test_experiment_invisible_across_credentials(monkeypatch):
    """Cross-tenant: register under token A, experiment.get under token B →
    "Unknown experiment" (the same process-local error as a foreign process);
    the owner still sees the experiment and it completes normally."""
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.5}})
    server, state = make_two_tenant_server(backend)
    async with Client(server) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        experiment_id = result.data["experimentId"]  # returned plain, unchanged

        state["token"] = "nimb_bob"  # same gateway process, different user
        with pytest.raises(Exception, match="Unknown experiment"):
            await c.call_tool("experiment.get", {"experiment_id": experiment_id})

        state["token"] = "nimb_alice"  # the owner keeps full visibility
        snapshot = await wait_for_terminal(c, experiment_id)
        assert snapshot.data["status"] == "completed"
        assert snapshot.data["runs"][0]["metrics"]["kappa"] == 0.5


def test_experiments_registry_keys_by_credential_fingerprint():
    """The registry stores/looks up by ``"<fingerprint>:<experiment_id>"`` —
    the same experiment_id under two fingerprints is two entries, and an
    unknown id under a known fingerprint is a miss."""
    reg = campaign.Experiments()
    state = campaign.ExperimentState("exp1", [], max_in_flight=1)
    reg.add(state, fingerprint="fp_alice")
    assert reg.snapshot("fp_alice", "exp1") is not None
    assert reg.snapshot("fp_alice", "exp1")["experimentId"] == "exp1"
    assert reg.snapshot("fp_bob", "exp1") is None  # other tenant: miss
    assert reg.snapshot("fp_alice", "exp_other") is None
