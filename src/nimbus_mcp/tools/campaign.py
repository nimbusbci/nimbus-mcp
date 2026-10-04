"""Campaign tools: run a batch of pipelines as one paced experiment.

``run_experiment`` validates, spawns a background worker thread and returns
immediately — the tool call never blocks on the worker. The worker submits at
most ``min(max_concurrent, 2)`` runs in flight, submits the next run as each
one reaches a terminal state (polling execution summaries), backs off and
retries on "Execution queue full" (max 3 attempts per run), then aggregates
trimmed metrics over the completed runs.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ..setup_mode import SetupRequired
from ._guards import safe_segment
from .run import _METRIC_KEYS, synth_layout

POLL_INTERVAL_SEC = 5.0
QUEUE_FULL_BACKOFF_SEC = 5.0
MAX_SUBMIT_ATTEMPTS = 3
MAX_POLL_ATTEMPTS = 3
POLL_RETRY_BACKOFF_SEC = 2.0
MAX_RUNS = 25
MAX_IN_FLIGHT = 2

_AGGREGATE_METRICS = ("accuracyPct", "evalAccuracyPct", "kappa", "itr")
_QUEUE_FULL_MARKER = "Execution queue full"
_TERMINAL = ("completed", "failed", "cancelled")


@dataclass
class RunRow:
    name: str
    train_graph: dict[str, Any]
    execution_id: str | None = None
    status: str = "pending"  # pending | running | completed | failed
    error: str | None = None
    metrics: dict[str, Any] | None = None


class ExperimentState:
    def __init__(
        self, experiment_id: str, runs: list[RunRow], max_in_flight: int
    ) -> None:
        self.experiment_id = experiment_id
        self.runs = runs
        self.max_in_flight = max_in_flight
        self.status = "running"
        self.aggregates: dict[str, Any] | None = None

    def payload(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            row: dict[str, Any] = {
                "name": run.name,
                "executionId": run.execution_id,
                "status": run.status,
            }
            if run.error is not None:
                row["error"] = run.error
            if run.metrics is not None:
                row["metrics"] = run.metrics
            rows.append(row)
        out: dict[str, Any] = {
            "experimentId": self.experiment_id,
            "status": self.status,
            "runs": rows,
        }
        if self.aggregates is not None:
            out["aggregates"] = self.aggregates
        return out


class Experiments:
    """Thread-safe process-local registry of experiment states."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._states: dict[str, ExperimentState] = {}

    def add(self, state: ExperimentState) -> None:
        with self.lock:
            self._states[state.experiment_id] = state

    def get(self, experiment_id: str) -> ExperimentState | None:
        with self.lock:
            return self._states.get(experiment_id)

    def snapshot(self, experiment_id: str) -> dict[str, Any] | None:
        with self.lock:
            state = self._states.get(experiment_id)
            return None if state is None else state.payload()


EXPERIMENTS = Experiments()


def _submit_run(run: RunRow, client: NimbusClient) -> None:
    """POST /api/execute with queue-full retry/backoff; marks the run failed
    on a non-retryable error or after MAX_SUBMIT_ATTEMPTS queue-full hits."""
    body: dict[str, Any] = {
        "train": run.train_graph,
        "layout": synth_layout(run.train_graph),
        "name": run.name,
    }
    attempt = 0
    while True:
        attempt += 1
        try:
            payload = client.post("/api/execute", json=body)
        except McpToolError as err:
            last_error = str(err)
            if _QUEUE_FULL_MARKER in last_error and attempt < MAX_SUBMIT_ATTEMPTS:
                time.sleep(QUEUE_FULL_BACKOFF_SEC)
                continue
            run.status = "failed"
            run.error = last_error
            return
        run.execution_id = str(payload.get("executionId"))
        run.status = "running"
        return


def _fetch_metrics(exec_id: str, client: NimbusClient) -> dict[str, Any] | None:
    """Trimmed metrics for a completed run, or None when the result fetch
    fails (the run is then treated as failed, not completed-without-metrics)."""
    try:
        payload = client.get(f"/api/result/{exec_id}")
    except SetupRequired:
        # A hosted token that died mid-experiment is a setup problem, not a
        # bad result: let the guidance propagate instead of mislabeling the
        # run "completed but result fetch failed".
        raise
    except McpToolError:
        return None
    metrics = (payload.get("result") or {}).get("metrics") or {}
    return {k: metrics[k] for k in _METRIC_KEYS if metrics.get(k) is not None}


def _poll_summary(exec_id: str, client: NimbusClient) -> dict[str, Any]:
    """GET the execution summary, retrying transient failures (a 5xx or a
    momentary connection blip, both surface as McpToolError) up to
    MAX_POLL_ATTEMPTS total attempts with POLL_RETRY_BACKOFF_SEC between them;
    after that the error propagates to the worker's catch-all. Non-McpToolError
    exceptions are never retried, and SetupRequired (token expired mid-session)
    propagates immediately — retrying login guidance is pointless."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.get(f"/api/executions/{exec_id}/summary")
        except SetupRequired:
            raise
        except McpToolError:
            if attempt >= MAX_POLL_ATTEMPTS:
                raise
            time.sleep(POLL_RETRY_BACKOFF_SEC)


def _drive(state: ExperimentState, client: NimbusClient) -> None:
    """Paced submission loop: keep <= max_in_flight executions running, poll
    summaries every POLL_INTERVAL_SEC, promote terminal runs, submit the next."""
    pending = list(state.runs)
    in_flight: dict[str, RunRow] = {}
    while pending or in_flight:
        while pending and len(in_flight) < state.max_in_flight:
            run = pending.pop(0)
            _submit_run(run, client)
            if run.execution_id is not None:
                in_flight[run.execution_id] = run
        if not in_flight:
            break  # every submission failed; nothing to poll
        time.sleep(POLL_INTERVAL_SEC)
        for exec_id in list(in_flight):
            summary = _poll_summary(exec_id, client).get("summary") or {}
            status = summary.get("status")
            run = in_flight[exec_id]
            if status == "completed":
                metrics = _fetch_metrics(exec_id, client)
                if metrics is None:
                    run.status = "failed"
                    run.error = f"run completed but result fetch failed for {exec_id}"
                else:
                    run.metrics = metrics
                    run.status = "completed"
                del in_flight[exec_id]
            elif status in _TERMINAL:  # failed | cancelled
                run.status = "failed"
                run.error = summary.get("errorMessage") or f"terminal status: {status}"
                del in_flight[exec_id]


def _compute_aggregates(runs: list[RunRow]) -> dict[str, Any]:
    """{metric: {mean, std, best: {name, value}}} over completed runs only.
    std is the population standard deviation; None below 2 values."""
    completed = [run for run in runs if run.status == "completed" and run.metrics]
    aggregates: dict[str, Any] = {}
    for key in _AGGREGATE_METRICS:
        pairs: list[tuple[str, Any]] = []
        for run in completed:
            value = run.metrics.get(key) if run.metrics else None
            if value is not None:
                pairs.append((run.name, value))
        if not pairs:
            continue
        values = [value for _, value in pairs]
        best_name, best_value = max(pairs, key=lambda pair: pair[1])
        aggregates[key] = {
            "mean": fmean(values),
            "std": pstdev(values) if len(values) >= 2 else None,
            "best": {"name": best_name, "value": best_value},
        }
    return aggregates


def _experiment_worker(state: ExperimentState, client: NimbusClient) -> None:
    """Background thread body: drive the experiment, then finalize status and
    aggregates. Any unexpected error (e.g. backend gone mid-run) fails the
    still-open runs instead of leaving the experiment 'running' forever."""
    try:
        _drive(state, client)
    except Exception as err:
        with EXPERIMENTS.lock:
            for run in state.runs:
                if run.status not in ("completed", "failed"):
                    run.status = "failed"
                    run.error = f"experiment aborted: {err}"
    finally:
        with EXPERIMENTS.lock:
            state.aggregates = _compute_aggregates(state.runs)
            state.status = (
                "failed" if all(run.status == "failed" for run in state.runs) else "completed"
            )


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool
    def run_experiment(runs: list[dict[str, Any]], max_concurrent: int = 2) -> dict[str, Any]:
        """Run 1-25 pipelines as ONE paced experiment (NON-BLOCKING). Returns an
        experimentId immediately; a background thread submits at most 2 runs at
        a time (min(max_concurrent, 2)), retries queue-full up to 3 times per
        run, and polls each execution to completion. Poll get_experiment() for
        per-run status and, once finished, aggregated metrics."""
        # Preflight BEFORE registering state or spawning the worker thread:
        # in setup mode the worker's catch-all would swallow SetupRequired
        # into per-run errors and the tool would answer a misleading
        # {status: "running"} instead of the setup guidance dict.
        client.ensure_ready()
        if not runs:
            raise McpToolError("run_experiment needs at least 1 run (got 0).")
        if len(runs) > MAX_RUNS:
            raise McpToolError(
                f"run_experiment accepts at most {MAX_RUNS} runs (got {len(runs)}). "
                "Split the sweep into several experiments."
            )
        prepared: list[RunRow] = []
        for index, item in enumerate(runs):
            if not isinstance(item, dict) or not isinstance(item.get("train_graph"), dict):
                raise McpToolError(
                    f"Run {index} is invalid: every run needs a 'train_graph' dict "
                    "(same shape as run_pipeline's train_graph)."
                )
            prepared.append(
                RunRow(
                    name=str(item.get("name") or f"run-{index + 1}"),
                    train_graph=item["train_graph"],
                )
            )
        experiment_id = uuid.uuid4().hex
        state = ExperimentState(
            experiment_id=experiment_id,
            runs=prepared,
            max_in_flight=max(1, min(max_concurrent, MAX_IN_FLIGHT)),
        )
        EXPERIMENTS.add(state)
        threading.Thread(
            target=_experiment_worker,
            args=(state, client),
            name=f"nimbus-experiment-{experiment_id[:8]}",
            daemon=True,
        ).start()
        return {
            "experimentId": experiment_id,
            "status": "running",
            "runs": [{"name": run.name, "executionId": None} for run in prepared],
            "pollHint": "Runs submit in the background. Call get_experiment(experiment_id) "
            "until status is completed or failed.",
        }

    @mcp.tool
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        """Experiment snapshot: status (running/completed/failed), per-run rows
        ({name, executionId, status, error?, metrics?}) and, once finished,
        aggregates {metric: {mean, std, best: {name, value}}} over completed
        runs only (std = population; None below 2 values)."""
        # Symmetry with run_experiment: in setup mode no experiment can exist
        # (its preflight refuses), so answer with guidance rather than the
        # confusing "Unknown experiment".
        client.ensure_ready()
        exp_id = safe_segment(experiment_id, label="experiment id")
        snapshot = EXPERIMENTS.snapshot(exp_id)
        if snapshot is None:
            raise McpToolError(
                f"Unknown experiment '{exp_id}'. It was run in another MCP server "
                "process, or the id is wrong — experiment state is process-local."
            )
        return snapshot
