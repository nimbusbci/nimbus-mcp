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


@pytest.fixture(autouse=True)
def _campaign_disk_under_tmp(tmp_path, monkeypatch):
    """Keep experiment persistence out of the real ``~/.nimbus`` (sandbox /
    CI permission noise, and cross-test leakage)."""
    root = tmp_path / "nimbus-experiments"
    monkeypatch.setattr(
        campaign,
        "_experiment_disk_dir",
        lambda fingerprint: root / fingerprint[:8],
    )


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
        # #518 server-side experiment records: id -> (owner key, PUT body).
        self.experiments: dict[str, tuple[str, dict]] = {}
        self.experiments_down = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/me/profile":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "userId": "u1",
                    "capabilities": {},
                    "computeOrigin": {
                        "mode": "hosted",
                        "nimbusVersion": "1.0.0",
                        "desktopVersion": None,
                        "mcpLocalBridge": False,
                    },
                },
            )
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
        if path.startswith("/api/experiments/"):
            if self.experiments_down:
                return httpx.Response(500, json={"detail": "experiments store down"})
            exp_id = path.split("/")[3]
            owner = req.headers.get("authorization") or req.headers.get("x-mcp-key") or "anon"
            if req.method == "PUT":
                body = json.loads(req.content)
                existing = self.experiments.get(exp_id)
                if existing is not None and existing[0] != owner:
                    return httpx.Response(
                        404,
                        json={"code": "nimbus.experiment.not_found",
                              "detail": "Experiment not found"},
                    )
                self.experiments[exp_id] = (owner, body)
                return httpx.Response(
                    200,
                    json={"ok": True, "record": {
                        "experimentId": exp_id,
                        "status": body.get("status"),
                        "executionIds": body.get("executionIds") or [],
                        "payload": body.get("payload"),
                    }},
                )
            if req.method == "GET":
                found = self.experiments.get(exp_id)
                if found is None or found[0] != owner:
                    return httpx.Response(
                        404,
                        json={"code": "nimbus.experiment.not_found",
                              "detail": "Experiment not found"},
                    )
                body = found[1]
                return httpx.Response(
                    200,
                    json={"ok": True, "record": {
                        "experimentId": exp_id,
                        "status": body.get("status"),
                        "executionIds": body.get("executionIds") or [],
                        "payload": body.get("payload"),
                    }},
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


async def test_experiment_payload_includes_compute_origin_from_profile(monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.5}})
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert snapshot.data["computeOrigin"]["mode"] == "hosted"
    assert snapshot.data["computeOrigin"]["mcpLocalBridge"] is False


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


def test_compute_aggregates_kind_breakdown_mixed_per_subject_and_loso():
    """Mixed campaign: aggregates label per-subject vs LOSO pooling (#470)."""
    per_subject_graph = {
        "nodes": [
            {"id": "d", "type": "public_data", "config": {"subject": "S01"}},
            {
                "id": "ep",
                "type": "evaluation_plan",
                "config": {"method": "within_session"},
            },
        ],
        "connections": [],
    }
    loso_graph = {
        "nodes": [
            {
                "id": "d",
                "type": "public_data",
                "config": {"subject": "S01,S02,S03,S04,S05"},
            },
            {"id": "ep", "type": "evaluation_plan", "config": {"method": "loso"}},
        ],
        "connections": [],
    }
    runs = [
        campaign.RunRow(
            name="S01",
            train_graph=per_subject_graph,
            execution_id="e1",
            status="completed",
            metrics={"accuracyPct": 69.58, "kappa": 0.4},
        ),
        campaign.RunRow(
            name="S02",
            train_graph=per_subject_graph,
            execution_id="e2",
            status="completed",
            metrics={"accuracyPct": 59.56, "kappa": 0.2},
        ),
        campaign.RunRow(
            name="S03",
            train_graph=per_subject_graph,
            execution_id="e3",
            status="completed",
            metrics={"accuracyPct": 55.56, "kappa": 0.1},
        ),
        campaign.RunRow(
            name="LOSO",
            train_graph=loso_graph,
            execution_id="531b846b",
            status="completed",
            metrics={"accuracyPct": 58.44, "kappa": 0.3, "nGroups": 3},
        ),
    ]
    aggregates = campaign._compute_aggregates(runs)
    assert aggregates["pooledExecutionIds"] == ["e1", "e2", "e3", "531b846b"]
    # Pooled headline mean unchanged (still all completed runs, incl. LOSO).
    assert aggregates["accuracyPct"]["n"] == 4
    assert aggregates["accuracyPct"]["mean"] == pytest.approx(
        (69.58 + 59.56 + 55.56 + 58.44) / 4.0
    )
    breakdown = {row["kind"]: row for row in aggregates["kindBreakdown"]}
    assert breakdown["per_subject"]["n"] == 3
    assert breakdown["per_subject"]["mean"] == pytest.approx(
        (69.58 + 59.56 + 55.56) / 3.0
    )
    assert breakdown["per_subject"]["executionIds"] == ["e1", "e2", "e3"]
    assert breakdown["loso"]["n"] == 1
    assert breakdown["loso"]["mean"] == pytest.approx(58.44)
    assert breakdown["loso"]["std"] is None
    assert breakdown["loso"]["executionIds"] == ["531b846b"]
    assert breakdown["loso"]["runNames"] == ["LOSO"]
    assert campaign._infer_evaluation_kind(runs[0]) == "per_subject"
    assert campaign._infer_evaluation_kind(runs[3]) == "loso"


def test_compute_aggregates_kind_breakdown_single_kind_still_present():
    """Single-kind campaigns still get kindBreakdown (always present when aggregates)."""
    graph = {
        "nodes": [
            {"id": "d", "type": "public_data", "config": {"subject": "S01"}},
            {
                "id": "ep",
                "type": "evaluation_plan",
                "config": {"method": "within_session"},
            },
        ],
        "connections": [],
    }
    runs = [
        campaign.RunRow(
            name="S01",
            train_graph=graph,
            execution_id="only-1",
            status="completed",
            metrics={"accuracyPct": 70.0},
        ),
        campaign.RunRow(
            name="S02",
            train_graph=graph,
            execution_id="only-2",
            status="completed",
            metrics={"accuracyPct": 60.0},
        ),
    ]
    aggregates = campaign._compute_aggregates(runs)
    assert "kindBreakdown" in aggregates
    assert len(aggregates["kindBreakdown"]) == 1
    row = aggregates["kindBreakdown"][0]
    assert row["kind"] == "per_subject"
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(65.0)


def _acb2c103_legacy_aggregates() -> dict:
    return {
        "accuracyPct": {
            "mean": (69.58 + 59.56 + 55.56 + 58.44) / 4.0,
            "std": 5.0,
            "best": {"name": "S01", "value": 69.58},
            "n": 4,
        },
        "pooledExecutionIds": ["c73ca095", "b6e89401", "539bdd4c", "531b846b"],
        "pooledRunNames": ["S01", "S02", "S03", "LOSO"],
        # 0.14.7 live fail: bare nGroups→loso collapsed everything into one bucket.
        "kindBreakdown": [
            {
                "kind": "loso",
                "n": 4,
                "executionIds": ["c73ca095", "b6e89401", "539bdd4c", "531b846b"],
                "mean": (69.58 + 59.56 + 55.56 + 58.44) / 4.0,
                "std": 5.0,
            }
        ],
    }


def test_ensure_kind_breakdown_acb2c103_nGroups_on_per_subject_not_all_loso():
    """Live #470 fail shape: unstamped per-subject runs also carry nGroups=5.

    Bare nGroups must not imply loso. Backend-style method / cvGrouping signals
    (as stamped after /api/result enrichment, or present on the row) split
    per_subject n=3 vs loso n=1 with the live means.
    """
    payload = {
        "experimentId": "acb2c103b0d645dfbecbb647fba0f50b",
        "status": "completed",
        "runs": [
            {
                "name": "S01",
                "executionId": "c73ca095",
                "status": "completed",
                "metrics": {"accuracyPct": 69.58, "nGroups": 5},
                "evaluationMethod": "within_session",
                "cvGrouping": "session",
            },
            {
                "name": "S02",
                "executionId": "b6e89401",
                "status": "completed",
                "metrics": {"accuracyPct": 59.56, "nGroups": 5},
                "evaluationMethod": "within_session",
                "cvGrouping": "session",
            },
            {
                "name": "S03",
                "executionId": "539bdd4c",
                "status": "completed",
                "metrics": {"accuracyPct": 55.56, "nGroups": 5},
                "evaluationMethod": "within_session",
                "cvGrouping": "session",
            },
            {
                "name": "LOSO",
                "executionId": "531b846b",
                "status": "completed",
                "metrics": {"accuracyPct": 58.44, "nGroups": 3},
                "evaluationMethod": "loso",
                "groupKind": "subject",
            },
        ],
        "aggregates": _acb2c103_legacy_aggregates(),
    }
    fixed = campaign._ensure_kind_breakdown(payload)
    assert fixed["aggregates"]["accuracyPct"]["mean"] == pytest.approx(
        (69.58 + 59.56 + 55.56 + 58.44) / 4.0
    )
    breakdown = {row["kind"]: row for row in fixed["aggregates"]["kindBreakdown"]}
    assert set(breakdown) == {"per_subject", "loso"}
    assert breakdown["per_subject"]["n"] == 3
    assert breakdown["per_subject"]["mean"] == pytest.approx(
        (69.58 + 59.56 + 55.56) / 3.0
    )
    assert breakdown["loso"]["n"] == 1
    assert breakdown["loso"]["mean"] == pytest.approx(58.44)
    # Prior all-loso bucket replaced; input not mutated.
    assert payload["aggregates"]["kindBreakdown"][0]["n"] == 4


def test_infer_kind_bare_nGroups_alone_is_not_loso():
    """nGroups without method/groupKind must not classify as loso (#470 live)."""
    run = {
        "status": "completed",
        "executionId": "c73ca095",
        "metrics": {"accuracyPct": 69.58, "nGroups": 5},
    }
    assert campaign._infer_kind_from_payload_run(run) == "per_subject"


def test_infer_kind_stamped_evaluation_kind_wins_over_nGroups():
    """Stamped evaluationKind takes priority over conflicting nGroups."""
    run = {
        "status": "completed",
        "executionId": "c73ca095",
        "evaluationKind": "per_subject",
        "metrics": {"accuracyPct": 69.58, "nGroups": 5},
        "evaluationMethod": "loso",  # would be loso if stamp ignored
        "groupKind": "subject",
    }
    assert campaign._infer_kind_from_payload_run(run) == "per_subject"
    loso = {
        "status": "completed",
        "executionId": "531b846b",
        "evaluationKind": "loso",
        "metrics": {"accuracyPct": 58.44},
        "evaluationMethod": "within_session",
        "cvGrouping": "session",
    }
    assert campaign._infer_kind_from_payload_run(loso) == "loso"


def test_ensure_kind_breakdown_enriches_from_result_api():
    """Unstamped nGroups=5 rows: fetch evaluationSummary.method like stats.grouped."""

    class _ResultClient:
        def get(self, path: str) -> dict:
            exec_id = path.rsplit("/", 1)[-1]
            if exec_id == "531b846b":
                return {
                    "result": {
                        "metrics": {"accuracyPct": 58.44, "nGroups": 3},
                        "results": {
                            "evaluationSummary": {
                                "method": "loso",
                                "cvGrouping": "subject",
                            },
                            "foldResults": {
                                "grouped_accuracy_stats": {
                                    "nGroups": 3,
                                    "groupKind": "subject",
                                }
                            },
                        },
                    }
                }
            return {
                "result": {
                    "metrics": {"accuracyPct": 69.58, "nGroups": 5},
                    "results": {
                        "evaluationSummary": {
                            "method": "within_session",
                            "cvGrouping": "session",
                        },
                        "foldResults": {
                            "grouped_accuracy_stats": {
                                "nGroups": 5,
                                "groupKind": "session",
                            }
                        },
                    },
                }
            }

    payload = {
        "experimentId": "acb2c103b0d645dfbecbb647fba0f50b",
        "status": "completed",
        "runs": [
            {
                "name": "S01",
                "executionId": "c73ca095",
                "status": "completed",
                "metrics": {"accuracyPct": 69.58, "nGroups": 5},
            },
            {
                "name": "S02",
                "executionId": "b6e89401",
                "status": "completed",
                "metrics": {"accuracyPct": 59.56, "nGroups": 5},
            },
            {
                "name": "S03",
                "executionId": "539bdd4c",
                "status": "completed",
                "metrics": {"accuracyPct": 55.56, "nGroups": 5},
            },
            {
                "name": "LOSO",
                "executionId": "531b846b",
                "status": "completed",
                "metrics": {"accuracyPct": 58.44, "nGroups": 3},
            },
        ],
        "aggregates": _acb2c103_legacy_aggregates(),
    }
    fixed = campaign._ensure_kind_breakdown(payload, client=_ResultClient())
    breakdown = {row["kind"]: row for row in fixed["aggregates"]["kindBreakdown"]}
    assert breakdown["per_subject"]["n"] == 3
    assert breakdown["per_subject"]["mean"] == pytest.approx(
        (69.58 + 59.56 + 55.56) / 3.0
    )
    assert breakdown["loso"]["n"] == 1
    assert breakdown["loso"]["mean"] == pytest.approx(58.44)
    kinds = {r["executionId"]: r["evaluationKind"] for r in fixed["runs"]}
    assert kinds["c73ca095"] == "per_subject"
    assert kinds["531b846b"] == "loso"


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
        if path == "/api/me/profile":
            return httpx.Response(401, json={"detail": "no"})
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
        if path == "/api/me/profile":
            return httpx.Response(401, json={"detail": "no"})
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


def test_experiments_registry_keys_by_credential_fingerprint(tmp_path, monkeypatch):
    """The registry stores/looks up by ``"<fingerprint>:<experiment_id>"`` —
    the same experiment_id under two fingerprints is two entries, and an
    unknown id under a known fingerprint is a miss."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Path.home() reads the real home on some platforms — pin the store root.
    monkeypatch.setattr(
        campaign,
        "_experiment_disk_dir",
        lambda fingerprint: tmp_path / "exp" / fingerprint[:8],
    )
    reg = campaign.Experiments()
    state = campaign.ExperimentState("exp1", [], max_in_flight=1)
    reg.add(state, fingerprint="fp_alice")
    assert reg.snapshot("fp_alice", "exp1") is not None
    assert reg.snapshot("fp_alice", "exp1")["experimentId"] == "exp1"
    assert reg.snapshot("fp_bob", "exp1") is None  # other tenant: miss
    assert reg.snapshot("fp_alice", "exp_other") is None


def test_experiments_snapshot_survives_process_local_clear(tmp_path, monkeypatch):
    """Disk fallback answers experiment.get after the in-memory entry is gone (#476)."""
    monkeypatch.setattr(
        campaign,
        "_experiment_disk_dir",
        lambda fingerprint: tmp_path / "exp" / fingerprint[:8],
    )
    reg = campaign.Experiments()
    state = campaign.ExperimentState("exp-persist", [], max_in_flight=1)
    state.status = "completed"
    state.aggregates = {"kappa": {"mean": 0.4, "std": None, "best": {"name": "r", "value": 0.4}}}
    reg.add(state, fingerprint="fp_alice")
    # Simulate a fresh process: empty memory, same credential fingerprint.
    reg2 = campaign.Experiments()
    restored = reg2.snapshot("fp_alice", "exp-persist")
    assert restored is not None
    assert restored["experimentId"] == "exp-persist"
    assert restored["status"] == "completed"
    assert restored["aggregates"]["kappa"]["mean"] == 0.4
    assert reg2.snapshot("fp_bob", "exp-persist") is None


def test_experiments_registry_evicts_oldest_terminal_over_cap():
    """The registry is bounded: past MAX_EXPERIMENTS the oldest TERMINAL
    experiments are evicted (long-lived hosted gateway, states holding full
    train_graphs), while a still-running experiment is never evicted."""
    reg = campaign.Experiments()

    def mk(exp_id: str, status: str) -> campaign.ExperimentState:
        state = campaign.ExperimentState(exp_id, [], max_in_flight=1)
        state.status = status
        return state

    for i in range(campaign.MAX_EXPERIMENTS):
        reg.add(mk(f"e{i}", "completed"), fingerprint="fp")
    assert len(reg._states) == campaign.MAX_EXPERIMENTS

    running = mk("run-1", "running")
    reg.add(running, fingerprint="fp")
    assert len(reg._states) == campaign.MAX_EXPERIMENTS  # cap enforced
    assert "fp:e0" not in reg._states  # oldest terminal evicted
    assert reg._states["fp:run-1"] is running  # running survives

    reg.add(mk("e-new", "completed"), fingerprint="fp")
    assert len(reg._states) == campaign.MAX_EXPERIMENTS
    assert "fp:e1" not in reg._states  # next-oldest terminal evicted next
    assert "fp:e2" in reg._states


def test_experiments_registry_never_evicts_when_all_running():
    """A registry full of still-running experiments grows past the cap
    rather than dropping a pollable one — the workers finalize them to a
    terminal state, and the next add() evicts then."""
    reg = campaign.Experiments()
    for i in range(campaign.MAX_EXPERIMENTS + 4):
        reg.add(campaign.ExperimentState(f"r{i}", [], max_in_flight=1), fingerprint="fp")
    assert len(reg._states) == campaign.MAX_EXPERIMENTS + 4
    assert reg.snapshot("fp", "r0") is not None


# ── #518: server-side experiment records (write-through / read-fallback) ──


def _token_server(backend: FakeBackend, token: str) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="", export_dir=None, nimbus_token=token),  # type: ignore[arg-type]
        transport=httpx.MockTransport(backend.handler),
    )
    server = FastMCP("t")
    campaign.register(server, client)
    return server


def _fresh_local_state(monkeypatch, tmp_path) -> None:
    """Simulate a different process/machine: empty registry, empty disk."""
    monkeypatch.setattr(campaign, "EXPERIMENTS", campaign.Experiments())
    monkeypatch.setattr(
        campaign, "_experiment_disk_dir", lambda fingerprint: tmp_path / "elsewhere"
    )


async def test_run_experiment_writes_through_to_server(monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.5}})
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        exp_id = result.data["experimentId"]
        await wait_for_terminal(c, exp_id)
    assert exp_id in backend.experiments
    _owner, body = backend.experiments[exp_id]
    assert body["status"] == "completed"
    assert body["executionIds"] == ["exec_1"]
    assert body["payload"]["aggregates"]["kappa"]["n"] == 1


async def test_get_experiment_falls_back_to_server_record(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"accuracyPct": 61.5}})
    server = _token_server(backend, "tok-a")
    async with Client(server) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        exp_id = result.data["experimentId"]
        await wait_for_terminal(c, exp_id)
        _fresh_local_state(monkeypatch, tmp_path)  # another process/machine
        snapshot = await c.call_tool("experiment.get", {"experiment_id": exp_id})
    assert snapshot.data["status"] == "completed"
    assert snapshot.data["aggregates"]["accuracyPct"]["mean"] == 61.5


async def test_get_experiment_refreshes_unfinished_server_record(monkeypatch, tmp_path):
    # Origin process died after submitting but before finalizing: the stored
    # payload still says running; the executions themselves are the truth.
    backend = FakeBackend(metrics={"exec_1": {"accuracyPct": 55.0}})
    backend.experiments["expseeded"] = (
        "Bearer tok-a",
        {
            "status": "running",
            "executionIds": ["exec_1"],
            "payload": {
                "experimentId": "expseeded",
                "status": "running",
                "runs": [{"name": "r1", "executionId": "exec_1", "status": "running"}],
            },
        },
    )
    _fresh_local_state(monkeypatch, tmp_path)
    async with Client(_token_server(backend, "tok-a")) as c:
        snapshot = await c.call_tool("experiment.get", {"experiment_id": "expseeded"})
    assert snapshot.data["status"] == "completed"
    assert snapshot.data["runs"][0]["status"] == "completed"
    assert snapshot.data["aggregates"]["accuracyPct"]["mean"] == 55.0


async def test_get_experiment_backend_down_is_honest_unknown(monkeypatch, tmp_path):
    backend = FakeBackend()
    backend.experiments_down = True
    _fresh_local_state(monkeypatch, tmp_path)
    async with Client(_token_server(backend, "tok-a")) as c:
        with pytest.raises(Exception, match="Unknown experiment"):
            await c.call_tool("experiment.get", {"experiment_id": "nope"})


async def test_run_experiment_survives_backend_down(monkeypatch):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.25}})
    backend.experiments_down = True
    async with Client(make_server(backend)) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        snapshot = await wait_for_terminal(c, result.data["experimentId"])
    assert snapshot.data["status"] == "completed"  # local path unaffected
    assert backend.experiments == {}  # nothing could be written through


async def test_get_experiment_cross_owner_server_record_not_visible(monkeypatch, tmp_path):
    monkeypatch.setattr(campaign, "POLL_INTERVAL_SEC", 0.05)
    backend = FakeBackend(metrics={"exec_1": {"kappa": 0.5}})
    server_a = _token_server(backend, "tok-a")
    async with Client(server_a) as c:
        result = await c.call_tool(
            "experiment.run", {"runs": [{"name": "r1", "train_graph": TRAIN_GRAPH}]}
        )
        exp_id = result.data["experimentId"]
        await wait_for_terminal(c, exp_id)
    _fresh_local_state(monkeypatch, tmp_path)
    async with Client(_token_server(backend, "tok-b")) as c:
        with pytest.raises(Exception, match="Unknown experiment"):
            await c.call_tool("experiment.get", {"experiment_id": exp_id})
    async with Client(server_a) as c:
        snapshot = await c.call_tool("experiment.get", {"experiment_id": exp_id})
    assert snapshot.data["status"] == "completed"
