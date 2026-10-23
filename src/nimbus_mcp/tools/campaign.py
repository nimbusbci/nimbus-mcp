"""Campaign tools: run a batch of pipelines as one paced experiment.

``experiment.run`` validates, spawns a background worker thread and returns
immediately — the tool call never blocks on the worker. The worker submits at
most ``min(max_concurrent, 2)`` runs in flight, submits the next run as each
one reaches a terminal state (polling execution summaries), backs off and
retries on "Execution queue full" (max 3 attempts per run), then aggregates
trimmed metrics over the completed runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from fastmcp import FastMCP

from ..access import (
    HOSTED_FREE_MAX_CAMPAIGN_RUNS,
    MODE_HOSTED,
    MODE_LOCAL,
    TIER_PRO,
    account_tier,
    client_mode,
    require_token_scope,
    tier_gate,
    verify_hosted_credential,
)
from ..client import McpToolError, NimbusClient
from ..errors import local_tool_error
from ..setup_mode import SetupRequired
from ._annotations import DESTRUCTIVE, MUTATING, READ_ONLY
from ._guards import safe_segment
from .run import _METRIC_KEYS, synth_layout

logger = logging.getLogger(__name__)

POLL_INTERVAL_SEC = 5.0
QUEUE_FULL_BACKOFF_SEC = 5.0
MAX_SUBMIT_ATTEMPTS = 3
MAX_POLL_ATTEMPTS = 3
POLL_RETRY_BACKOFF_SEC = 2.0
MAX_RUNS = 25
MAX_IN_FLIGHT = 2
# Registry cap: each state holds up to 25 run rows with the full posted
# train_graph (tens of KB each), so on a long-lived hosted gateway the map
# must evict — oldest TERMINAL experiments go first; a still-running
# experiment is never evicted while it can still be polled to a terminal
# state by its own worker.
MAX_EXPERIMENTS = 64

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
    submit_attempts: int = 0  # submissions used (queue-full retries included)


class ExperimentState:
    def __init__(
        self,
        experiment_id: str,
        runs: list[RunRow],
        max_in_flight: int,
        *,
        compute_origin: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.runs = runs
        self.max_in_flight = max_in_flight
        self.status = "running"
        self.aggregates: dict[str, Any] | None = None
        self.compute_origin = compute_origin

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
                row["evaluationKind"] = _infer_evaluation_kind(run)
            rows.append(row)
        out: dict[str, Any] = {
            "experimentId": self.experiment_id,
            "status": self.status,
            "runs": rows,
        }
        if self.aggregates is not None:
            out["aggregates"] = self.aggregates
        if self.compute_origin is not None:
            out["computeOrigin"] = self.compute_origin
        return out


def _experiment_disk_dir(fingerprint: str) -> Path:
    """Per-credential durable store under ``~/.nimbus/experiments/<hash>/``.

    Does not create the directory — callers that write must mkdir inside their
    own OSError-tolerant path so a read-only / sandboxed home cannot abort
    experiment.run (#476).
    """
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:24]
    return Path.home() / ".nimbus" / "experiments" / digest


def _persist_experiment_payload(fingerprint: str, payload: dict[str, Any]) -> None:
    """Write a terminal (or in-progress) snapshot so experiment.get survives
    process restart on the same machine (#476)."""
    exp_id = payload.get("experimentId")
    if not isinstance(exp_id, str) or not exp_id:
        return
    directory = _experiment_disk_dir(fingerprint)
    path = directory / f"{exp_id}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as err:
        logger.warning("Could not persist experiment %s: %s", exp_id, err)


def _load_experiment_payload(fingerprint: str, experiment_id: str) -> dict[str, Any] | None:
    path = _experiment_disk_dir(fingerprint) / f"{experiment_id}.json"
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class Experiments:
    """Thread-safe registry of experiment states (memory + on-disk fallback).

    Entries are keyed by ``"<credential fingerprint>:<experiment_id>"``: the
    registry lives in one process, and a hosted gateway authenticates many
    users per process — keying by the bare id would let any authenticated
    gateway user read another user's experiment snapshot by id. The scoping
    is internal: experiment.run still returns the plain experiment_id, and
    experiment.get recomputes the caller's key from its own credential.

    Terminal snapshots are also written under ``~/.nimbus/experiments/`` so a
    fresh MCP process on the same host can still answer experiment.get (#476).
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._states: dict[str, ExperimentState] = {}
        self._fingerprints: dict[str, str] = {}

    @staticmethod
    def key(fingerprint: str, experiment_id: str) -> str:
        return f"{fingerprint}:{experiment_id}"

    def add(self, state: ExperimentState, fingerprint: str) -> None:
        with self.lock:
            key = self.key(fingerprint, state.experiment_id)
            self._states[key] = state
            self._fingerprints[key] = fingerprint
            self._evict_locked()
        _persist_experiment_payload(fingerprint, state.payload())

    def persist(self, fingerprint: str, state: ExperimentState) -> None:
        """Flush the current payload to disk (call after aggregates finalize)."""
        _persist_experiment_payload(fingerprint, state.payload())

    def _evict_locked(self) -> None:
        """Drop the oldest terminal experiments while over MAX_EXPERIMENTS
        (insertion order = registration order). A terminal state can only be
        re-read via experiment.get — pollers hold the id and learn the result
        before their experiment leaves the running workers' hands; the cap is
        generous (64), so eviction only bites a long-lived gateway process.
        Called with the lock held. Disk snapshots are kept so a later get can
        still recover the payload."""
        if len(self._states) <= MAX_EXPERIMENTS:
            return
        for key, state in list(self._states.items()):
            if len(self._states) <= MAX_EXPERIMENTS:
                break
            if state.status in _TERMINAL:
                del self._states[key]
                self._fingerprints.pop(key, None)

    def remove(self, fingerprint: str, experiment_id: str) -> str:
        """Drop an experiment from memory + disk: "removed" | "running" | "missing"."""
        key = self.key(fingerprint, experiment_id)
        with self.lock:
            state = self._states.get(key)
            if state is not None and state.status not in _TERMINAL:
                return "running"
            found = self._states.pop(key, None) is not None
            self._fingerprints.pop(key, None)
        path = _experiment_disk_dir(fingerprint) / f"{experiment_id}.json"
        try:
            path.unlink()
            found = True
        except FileNotFoundError:
            pass
        except OSError as err:
            logger.warning("Could not remove experiment %s: %s", experiment_id, err)
        return "removed" if found else "missing"

    def snapshot(self, fingerprint: str, experiment_id: str) -> dict[str, Any] | None:
        """Return the raw payload; ``experiment.get`` runs kind backfill with client."""
        with self.lock:
            state = self._states.get(self.key(fingerprint, experiment_id))
            if state is not None:
                return state.payload()
        return _load_experiment_payload(fingerprint, experiment_id)


EXPERIMENTS = Experiments()


def _persist_experiment_to_server(
    client: NimbusClient, payload: dict[str, Any], *, fatal_if_unreachable: bool = False
) -> None:
    """Best-effort write-through of the snapshot to the backend record
    (#518). NEVER raises: the local registry/disk stay the fast path, and a
    backend outage must not fail or stall the experiment — the record is
    simply absent server-side until the next successful write-through
    (completion writes again, so a start-time outage self-heals)."""
    exp_id = payload.get("experimentId")
    if not isinstance(exp_id, str) or not exp_id:
        return
    execution_ids = [
        row.get("executionId")
        for row in payload.get("runs") or []
        if isinstance(row, dict) and row.get("executionId")
    ]
    try:
        client.put(
            f"/api/experiments/{exp_id}",
            json={
                "status": payload.get("status") or "running",
                "executionIds": execution_ids,
                "payload": payload,
            },
        )
    except McpToolError as err:
        # A transport failure (no HTTP status) means the backend is down. In
        # local/desktop-key mode that backend also runs the pipelines, so the
        # caller asks for this to be fatal BEFORE an experiment id is handed
        # out (N3): an honest error instead of a "running" id that can only fail.
        if fatal_if_unreachable and getattr(err, "status_code", None) is None:
            raise
        logger.warning("Could not persist experiment %s to the server: %s", exp_id, err)
    except Exception as err:  # noqa: BLE001 — best-effort by design (see docstring)
        logger.warning("Could not persist experiment %s to the server: %s", exp_id, err)


def _refresh_server_snapshot(snapshot: dict[str, Any], client: NimbusClient) -> dict[str, Any]:
    """Bring a server-stored snapshot up to date from its executions.

    The stored payload is refreshed when its origin process finalizes; if
    that process died mid-run (or has not finalized yet), the executions
    themselves are server-side, so their live summaries/results are the
    truth. Rows without an executionId were never submitted by the origin
    and cannot be driven from here (the payload carries no train graphs) —
    they stay non-terminal and the experiment stays "running".
    """
    rows = snapshot.get("runs")
    if not isinstance(rows, list) or snapshot.get("status") in _TERMINAL:
        return snapshot
    changed = False
    for row in rows:
        if not isinstance(row, dict) or row.get("status") in _TERMINAL:
            continue
        exec_id = row.get("executionId")
        if not exec_id:
            continue
        try:
            summary = _poll_summary(exec_id, client).get("summary") or {}
        except McpToolError:
            continue
        status = summary.get("status")
        if status == "completed":
            metrics = _fetch_metrics(exec_id, client)
            if metrics is None:
                row["status"] = "failed"
                row["error"] = f"run completed but result fetch failed for {exec_id}"
            else:
                row["status"] = "completed"
                row["metrics"] = metrics
            changed = True
        elif status in _TERMINAL:  # failed | cancelled
            row["status"] = "failed"
            row["error"] = summary.get("errorMessage") or f"terminal status: {status}"
            changed = True
    if changed and all(isinstance(row, dict) and row.get("status") in _TERMINAL for row in rows):
        run_rows = [
            RunRow(
                name=str(row.get("name") or ""),
                train_graph={},
                execution_id=row.get("executionId"),
                status=str(row.get("status") or "pending"),
                error=row.get("error"),
                metrics=row.get("metrics"),
            )
            for row in rows
        ]
        snapshot["aggregates"] = _compute_aggregates(run_rows)
        snapshot["status"] = (
            "failed" if all(row.status == "failed" for row in run_rows) else "completed"
        )
        _persist_experiment_to_server(client, snapshot)
    return snapshot


def _fetch_server_experiment(client: NimbusClient, exp_id: str) -> dict[str, Any] | None:
    """Read-fallback: the backend experiment record's payload, refreshed.

    None on any miss — unknown id, another owner's id (the backend 404s
    both identically), or the backend being unreachable. SetupRequired
    propagates: in setup mode no experiment can exist and the caller wants
    the guidance, not an "unknown" error.
    """
    try:
        body = client.get(f"/api/experiments/{exp_id}")
    except SetupRequired:
        raise
    except Exception:  # noqa: BLE001 — a failed lookup is a miss, handled by the caller
        return None
    record = body.get("record") if isinstance(body, dict) else None
    payload = record.get("payload") if isinstance(record, dict) else None
    if not isinstance(payload, dict) or payload.get("experimentId") != exp_id:
        return None
    return _refresh_server_snapshot(payload, client)


def _submit_run(run: RunRow, client: NimbusClient) -> None:
    """POST /api/execute once; marks the run failed on an error or on a
    malformed 200 (no executionId — failing only THIS run, not the
    experiment). Queue-full is never an HTTP error: POST answers 200
    immediately and the backend's concurrency gate reports "Execution queue
    full" later via the polled summary — _drive resubmits there."""
    body: dict[str, Any] = {
        "train": run.train_graph,
        "layout": synth_layout(run.train_graph),
        "name": run.name,
    }
    run.submit_attempts += 1
    try:
        payload = client.post("/api/execute", json=body)
    except McpToolError as err:
        run.status = "failed"
        run.error = str(err)
        return
    exec_id = payload.get("executionId")
    if not isinstance(exec_id, str) or not exec_id:
        run.status = "failed"
        run.error = "backend accepted the run but returned no executionId"
        return
    run.execution_id = exec_id
    run.status = "running"


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
                error = summary.get("errorMessage") or f"terminal status: {status}"
                if _QUEUE_FULL_MARKER in error and run.submit_attempts < MAX_SUBMIT_ATTEMPTS:
                    # The concurrency gate rejected the run via the summary
                    # (status failed, errorMessage queue-full) — back off and
                    # resubmit the same run as a new execution.
                    del in_flight[exec_id]
                    time.sleep(QUEUE_FULL_BACKOFF_SEC)
                    _submit_run(run, client)
                    if run.execution_id is not None and run.execution_id != exec_id:
                        in_flight[run.execution_id] = run
                    continue
                run.status = "failed"
                run.error = error
                del in_flight[exec_id]


def _evaluation_method_from_graph(train_graph: dict[str, Any]) -> str | None:
    """Return the evaluation_plan ``method`` from a run graph, if present."""
    nodes = train_graph.get("nodes")
    if not isinstance(nodes, list):
        return None
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("type") or "") != "evaluation_plan":
            continue
        cfg = node.get("config")
        if not isinstance(cfg, dict):
            return None
        method = cfg.get("method")
        return str(method).strip().lower() if method not in (None, "") else None
    return None


def _public_data_subject_count(train_graph: dict[str, Any]) -> int | None:
    """Count of subjects listed on the first public_data node, if any."""
    nodes = train_graph.get("nodes")
    if not isinstance(nodes, list):
        return None
    for node in nodes:
        if not isinstance(node, dict) or str(node.get("type") or "") != "public_data":
            continue
        cfg = node.get("config")
        if not isinstance(cfg, dict):
            return None
        raw = str(cfg.get("subject") or "")
        parts = [s.strip() for s in raw.split(",") if s.strip()]
        return len(parts) if parts else None
    return None


_KIND_ORDER = ("per_subject", "within_session", "loso", "losco", "pretrained_few_trial")


def _infer_evaluation_kind(run: RunRow) -> str:
    """Classify a completed run for aggregate kind labelling (#470).

    Prefers the evaluation_plan method on the posted graph. Single-subject
    within-session / k-fold runs are labelled ``per_subject`` so a mixed
    campaign's ``kindBreakdown`` separates them from LOSO. Falls back to
    metrics: ``nGroups`` marks a multi-subject / LOSO-style grouped run.
    """
    method = _evaluation_method_from_graph(run.train_graph)
    n_subjects = _public_data_subject_count(run.train_graph)
    if method == "loso":
        return "loso"
    if method == "losco":
        return "losco"
    if method == "pretrained_few_trial":
        return "pretrained_few_trial"
    if method == "within_session":
        if n_subjects is not None and n_subjects >= 2:
            return "within_session"
        return "per_subject"
    if method in {"kfold", "stratified_kfold"}:
        metrics = run.metrics or {}
        if metrics.get("nGroups") is not None or (n_subjects is not None and n_subjects >= 2):
            return "grouped"
        return "per_subject"
    metrics = run.metrics or {}
    if metrics.get("nGroups") is not None:
        return "loso"
    return "per_subject"


def _headline_accuracy_pct(metrics: dict[str, Any] | None) -> float | None:
    """Prefer ``accuracyPct``, then ``evalAccuracyPct``, for kind-mean labelling."""
    if not isinstance(metrics, dict):
        return None
    for key in ("accuracyPct", "evalAccuracyPct"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _coerce_n_groups(raw: Any) -> int | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    return None


def _group_kind_from_cv_grouping(cv: str) -> str | None:
    """Mirror backend ``plots._resolve_group_kind`` cvGrouping fallback."""
    if cv.startswith("subject-session") or cv == "session" or cv.startswith("session"):
        return "session"
    if cv.startswith("subject"):
        return "subject"
    return None


def _infer_kind_from_signals(
    method: str | None,
    n_groups: int | None,
    group_kind: str | None,
) -> str:
    """Classify using the same priority as backend ``_infer_execution_evaluation_kind``.

    Method wins (within_session → per_subject unless subject-grouped;
    loso → loso). Bare ``nGroups`` alone must NOT imply loso — per-subject
    within-session / k-fold runs also stamp nGroups (live #470 fail: all four
    runs classified loso because each carried nGroups=5).
    """
    meth = (method or "").strip().lower() or None
    if meth == "loso":
        return "loso"
    if meth == "losco":
        return "losco"
    if meth == "pretrained_few_trial":
        return "pretrained_few_trial"
    if meth == "within_session":
        if n_groups is not None and n_groups >= 2 and group_kind == "subject":
            return "within_session"
        return "per_subject"
    if meth in {"kfold", "stratified_kfold"}:
        if n_groups is not None and n_groups >= 2 and group_kind == "subject":
            return "grouped"
        return "per_subject"
    # No evaluation-plan method: use groupKind / cvGrouping-derived kind.
    # Session-level groups → per_subject. Subject-level groups with nGroups →
    # loso (multi-subject / LOSO-style). Never treat nGroups alone as loso.
    if group_kind == "session":
        return "per_subject"
    if group_kind == "subject" and n_groups is not None and n_groups >= 2:
        return "loso"
    return "per_subject"


def _kind_signals_from_payload_run(
    run: dict[str, Any],
) -> tuple[str | None, int | None, str | None]:
    """Extract (method, n_groups, group_kind) from a serialized run row."""
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    method_raw = (
        run.get("evaluationMethod") or metrics.get("evaluationMethod") or metrics.get("method")
    )
    method = str(method_raw).strip().lower() if method_raw not in (None, "") else None
    n_groups = _coerce_n_groups(metrics.get("nGroups") if metrics else None)
    group_kind_raw = run.get("groupKind") or metrics.get("groupKind")
    group_kind = (
        str(group_kind_raw).strip().lower()
        if isinstance(group_kind_raw, str) and group_kind_raw.strip()
        else None
    )
    if group_kind not in ("subject", "session"):
        group_kind = None
    if group_kind is None:
        cv_raw = run.get("cvGrouping") or metrics.get("cvGrouping") or ""
        group_kind = _group_kind_from_cv_grouping(str(cv_raw))
    return method, n_groups, group_kind


def _infer_kind_from_payload_run(run: dict[str, Any]) -> str:
    """Classify a serialized experiment.run row (disk / payload; no train_graph).

    Prefers stamped ``evaluationKind``; otherwise mirrors backend stats.grouped
    signals (evaluation method, then groupKind/cvGrouping, never bare nGroups
    alone as loso — #470 / 0.14.7 live fail).
    """
    stamped = run.get("evaluationKind")
    if isinstance(stamped, str) and stamped.strip():
        return stamped.strip()
    method, n_groups, group_kind = _kind_signals_from_payload_run(run)
    return _infer_kind_from_signals(method, n_groups, group_kind)


def _kind_signals_from_execution_result(
    result: dict[str, Any],
) -> tuple[str | None, int | None, str | None]:
    """Pull method / nGroups / groupKind off a ``GET /api/result/{id}`` body.

    Same sources backend ``stats.grouped`` uses (evaluationSummary.method +
    grouped_accuracy_stats.groupKind / cvGrouping).
    """
    raw_metrics = result.get("metrics")
    metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
    raw_results = result.get("results")
    results: dict[str, Any] = raw_results if isinstance(raw_results, dict) else {}
    summary = results.get("evaluationSummary")
    method: str | None = None
    if isinstance(summary, dict):
        raw_method = summary.get("method")
        if raw_method not in (None, ""):
            method = str(raw_method).strip().lower()
    blob: dict[str, Any] | None = None
    for container in ("foldResults", "evaluationSummary"):
        block = results.get(container)
        if isinstance(block, dict):
            grouped = block.get("grouped_accuracy_stats")
            if isinstance(grouped, dict):
                blob = grouped
                break
    group_kind: str | None = None
    if isinstance(blob, dict) and blob.get("groupKind") in ("subject", "session"):
        group_kind = str(blob.get("groupKind"))
    if group_kind is None and isinstance(summary, dict):
        group_kind = _group_kind_from_cv_grouping(str(summary.get("cvGrouping") or ""))
    n_groups = None
    if isinstance(blob, dict):
        n_groups = _coerce_n_groups(blob.get("nGroups"))
    if n_groups is None:
        n_groups = _coerce_n_groups(metrics.get("nGroups"))
    return method, n_groups, group_kind


def _enrich_payload_runs_kind_signals(
    runs: list[dict[str, Any]],
    client: NimbusClient,
) -> None:
    """Best-effort: fill evaluationMethod / groupKind from backend results.

    Used when legacy disk rows lack ``evaluationKind`` so classification can
    match ``stats.grouped`` on the same execution ids. Mutates ``runs`` in place;
    fetch failures leave the row unchanged (offline signals still apply).
    """
    for run in runs:
        if not isinstance(run, dict):
            continue
        # Only completed runs with metrics enter kindBreakdown; skip the rest
        # so a failed mid-poll experiment.get does not poke /api/result.
        if run.get("status") != "completed":
            continue
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            continue
        stamped = run.get("evaluationKind")
        if isinstance(stamped, str) and stamped.strip():
            continue
        method, _n, group_kind = _kind_signals_from_payload_run(run)
        if method is not None and (group_kind is not None or method in {"loso", "losco"}):
            # Already have enough offline signal to classify like the backend.
            continue
        if method is not None and method == "within_session":
            continue
        exec_id = run.get("executionId")
        if not isinstance(exec_id, str) or not exec_id:
            continue
        try:
            payload = client.get(f"/api/result/{exec_id}")
        except (McpToolError, SetupRequired):
            # Best-effort only: an expired token mid-poll must not turn a
            # readable disk snapshot into setup guidance (#470 backfill).
            continue
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            continue
        fetched_method, _fetched_n, fetched_kind = _kind_signals_from_execution_result(result)
        if fetched_method and not run.get("evaluationMethod"):
            run["evaluationMethod"] = fetched_method
        if fetched_kind and not run.get("groupKind"):
            run["groupKind"] = fetched_kind


def _order_kind_breakdown(by_kind: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable kind order: per_subject first, then loso-family, then remaining alpha."""
    ordered = [by_kind[k] for k in _KIND_ORDER if k in by_kind]
    ordered.extend(by_kind[k] for k in sorted(by_kind) if k not in _KIND_ORDER)
    return ordered


def _finalize_kind_bucket(bucket: dict[str, Any], accuracies: list[float]) -> None:
    """Attach headline accuracy mean/std; std is population, None below 2 values."""
    bucket["mean"] = fmean(accuracies) if accuracies else None
    bucket["std"] = pstdev(accuracies) if len(accuracies) >= 2 else None


def _kind_breakdown_from_run_rows(completed: list[RunRow]) -> list[dict[str, Any]]:
    """Build ``kindBreakdown`` rows (kind, n, mean, std, executionIds, runNames)."""
    by_kind: dict[str, dict[str, Any]] = {}
    acc_by_kind: dict[str, list[float]] = {}
    for run in completed:
        kind = _infer_evaluation_kind(run)
        bucket = by_kind.setdefault(
            kind,
            {"kind": kind, "n": 0, "executionIds": [], "runNames": []},
        )
        bucket["n"] += 1
        if run.execution_id:
            bucket["executionIds"].append(run.execution_id)
        bucket["runNames"].append(run.name)
        acc = _headline_accuracy_pct(run.metrics)
        if acc is not None:
            acc_by_kind.setdefault(kind, []).append(acc)
    for kind, bucket in by_kind.items():
        _finalize_kind_bucket(bucket, acc_by_kind.get(kind, []))
    return _order_kind_breakdown(by_kind)


def _kind_breakdown_from_payload_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild kindBreakdown from a serialized experiment payload (#470 follow-up)."""
    by_kind: dict[str, dict[str, Any]] = {}
    acc_by_kind: dict[str, list[float]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        if run.get("status") != "completed":
            continue
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            continue
        kind = _infer_kind_from_payload_run(run)
        bucket = by_kind.setdefault(
            kind,
            {"kind": kind, "n": 0, "executionIds": [], "runNames": []},
        )
        bucket["n"] += 1
        exec_id = run.get("executionId")
        if isinstance(exec_id, str) and exec_id:
            bucket["executionIds"].append(exec_id)
        name = run.get("name")
        bucket["runNames"].append(str(name) if name is not None else "")
        acc = _headline_accuracy_pct(metrics)
        if acc is not None:
            acc_by_kind.setdefault(kind, []).append(acc)
    for kind, bucket in by_kind.items():
        _finalize_kind_bucket(bucket, acc_by_kind.get(kind, []))
    return _order_kind_breakdown(by_kind)


def _ensure_kind_breakdown(
    payload: dict[str, Any],
    client: NimbusClient | None = None,
) -> dict[str, Any]:
    """Recompute ``aggregates.kindBreakdown`` on read (legacy / wrong labels).

    Always rebuilds from run signals so a prior all-loso misclassification
    (0.14.7 bare-nGroups fallback) is corrected. Optional ``client`` enriches
    unstamped rows from ``GET /api/result/{id}`` so method/groupKind match
    ``stats.grouped``. Does not change pooled metric means; does not rewrite
    the on-disk snapshot.
    """
    aggregates = payload.get("aggregates")
    if not isinstance(aggregates, dict):
        return payload
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        return payload
    runs: list[Any] = []
    dict_runs: list[dict[str, Any]] = []
    for raw in raw_runs:
        if isinstance(raw, dict):
            copied = dict(raw)
            runs.append(copied)
            dict_runs.append(copied)
        else:
            runs.append(raw)
    if client is not None:
        _enrich_payload_runs_kind_signals(dict_runs, client)
    for run in dict_runs:
        if run.get("status") != "completed":
            continue
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            continue
        run["evaluationKind"] = _infer_kind_from_payload_run(run)
    refreshed = _kind_breakdown_from_payload_runs(dict_runs)
    if not refreshed:
        return payload
    out = dict(payload)
    out["runs"] = runs
    out_agg = dict(aggregates)
    out_agg["kindBreakdown"] = refreshed
    out["aggregates"] = out_agg
    return out


def _compute_aggregates(runs: list[RunRow]) -> dict[str, Any]:
    """{metric: {mean, std, best: {name, value}}} over completed runs only.
    std is the population standard deviation; None below 2 values.

    Also labels what was pooled (#470): ``pooledExecutionIds`` /
    ``pooledRunNames`` / ``kindBreakdown`` so a LOSO run mixed into a
    per-subject campaign is visible (ids + evaluation kind + per-kind mean)
    in the aggregate. The pooled headline mean is unchanged (still over all
    completed runs).
    """
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
            "n": len(values),
        }
    if completed:
        aggregates["pooledExecutionIds"] = [
            run.execution_id for run in completed if run.execution_id
        ]
        aggregates["pooledRunNames"] = [run.name for run in completed]
        aggregates["kindBreakdown"] = _kind_breakdown_from_run_rows(completed)
    return aggregates


def _experiment_worker(
    state: ExperimentState,
    binding: AbstractContextManager[NimbusClient],
    fingerprint: str,
) -> None:
    """Background thread body: drive the experiment, then finalize status and
    aggregates. The binding owns the worker's client lifetime — the hosted
    wrapper yields a per-token client (closed here on exit; the worker has no
    request context to resolve headers in), a plain client yields the shared
    singleton (never closed). Any unexpected error (e.g. backend gone
    mid-run) fails the still-open runs instead of leaving the experiment
    'running' forever."""
    with binding as client:
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
            EXPERIMENTS.persist(fingerprint, state)
            _persist_experiment_to_server(client, state.payload())


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="experiment.run", annotations=MUTATING)
    def run_experiment(runs: list[dict[str, Any]], max_concurrent: int = 2) -> dict[str, Any]:
        """Run 1-25 pipelines as ONE paced experiment (NON-BLOCKING). Returns an
        experimentId immediately; a background thread submits at most 2 runs at
        a time (min(max_concurrent, 2)), retries queue-full up to 3 times per
        run, and polls each execution to completion. Poll experiment.get() for
        per-run status and, once finished, aggregated metrics.

        PORTABILITY: the experiment record is written through to the
        Nimbus server under your account (at start and at completion), so
        experiment.get works from ANY entry point — local stdio, hosted
        gateway, another machine — signed in with the same account/token.
        If the server is unreachable when the experiment starts it still
        runs and polls locally, but other entry points cannot read it
        until a write-through succeeds (#518).

        Args:
            runs: 1-25 entries, each {name: str, train_graph: {nodes, connections}}
                (same graph shape as execution.run's train_graph).
            max_concurrent: Parallel submissions cap, clamped to 1-2 (default 2).
        """
        # Preflight BEFORE registering state or spawning the worker thread:
        # in setup mode the worker's catch-all would swallow SetupRequired
        # into per-run errors and the tool would answer a misleading
        # {status: "running"} instead of the setup guidance dict.
        client.ensure_ready()
        # Hosted: a rejected token answers re-login guidance here, not a
        # "running" experiment whose worker dies on 401 (B5). Cached per
        # credential; experiment.get deliberately does NOT re-check.
        hosted_profile = verify_hosted_credential(client)
        # A valid but scoped token (e.g. a read-only CI key) would otherwise
        # get {status: "running"} and fail in the worker on the backend's 403
        # (N4). The worker PUTs /api/experiments/{id} and POSTs /api/execute,
        # both of which need the 'run' scope.
        require_token_scope(client, "run", "experiment.run (POST /api/execute)")
        if not runs:
            raise local_tool_error(
                "experiment.run needs at least 1 run (got 0).",
                code="nimbus.experiment.invalid_request",
            )
        if len(runs) > MAX_RUNS:
            raise McpToolError(
                f"experiment.run accepts at most {MAX_RUNS} runs (got {len(runs)}). "
                "Split the sweep into several experiments."
            )
        if (
            len(runs) > HOSTED_FREE_MAX_CAMPAIGN_RUNS
            and client_mode(client) == MODE_HOSTED
            and account_tier(client) != TIER_PRO
        ):
            # X10 "big campaigns" are Pro on hosted. The plan gives no
            # number; the conservative documented threshold is
            # HOSTED_FREE_MAX_CAMPAIGN_RUNS (5) runs per experiment for
            # free accounts (Pro keeps the full 25). Local backends: ungated.
            return tier_gate(
                "experiment.run",
                current_tier=account_tier(client),
                detail=(
                    f"Hosted campaigns of more than {HOSTED_FREE_MAX_CAMPAIGN_RUNS} "
                    f"runs are Pro (got {len(runs)}). Free accounts can run up to "
                    f"{HOSTED_FREE_MAX_CAMPAIGN_RUNS} runs per experiment."
                ),
                local_alternative=(
                    "Split the sweep into experiments of at most "
                    f"{HOSTED_FREE_MAX_CAMPAIGN_RUNS} runs, or run it against the "
                    "desktop app backend (local, ungated)."
                ),
            )
        prepared: list[RunRow] = []
        for index, item in enumerate(runs):
            if not isinstance(item, dict) or not isinstance(item.get("train_graph"), dict):
                raise McpToolError(
                    f"Run {index} is invalid: every run needs a 'train_graph' dict "
                    "(same shape as execution.run's train_graph)."
                )
            prepared.append(
                RunRow(
                    name=str(item.get("name") or f"run-{index + 1}"),
                    train_graph=item["train_graph"],
                )
            )
        experiment_id = uuid.uuid4().hex
        compute_origin: dict[str, Any] | None = None
        if hosted_profile is not None:
            # Hosted: reuse the profile the credential check just fetched (or
            # cached) — one /api/me/profile per run start, zero on a cache hit.
            origin = hosted_profile.get("computeOrigin")
            if isinstance(origin, dict):
                compute_origin = origin
        else:
            try:
                profile = client.get("/api/me/profile")
                origin = profile.get("computeOrigin") if isinstance(profile, dict) else None
                if isinstance(origin, dict):
                    compute_origin = origin
            except (McpToolError, SetupRequired):
                # Best-effort provenance stamp; never block experiment.start on profile blips.
                pass
        state = ExperimentState(
            experiment_id=experiment_id,
            runs=prepared,
            max_in_flight=max(1, min(max_concurrent, MAX_IN_FLIGHT)),
            compute_origin=compute_origin,
        )
        # Scope the registry entry to the calling credential BEFORE
        # registering/spawning: the fingerprint must be captured here, inside
        # the request (the worker thread has no MCP request context to resolve
        # a per-request token in — it only mutates the state object, never the
        # registry key). The id returned to the caller stays plain.
        fingerprint = client.credential_fingerprint()
        # Write-through (#518): the record lands server-side NOW (status
        # running, no executions yet) so another entry point can already
        # resolve the id; the worker writes the terminal snapshot at
        # finalize. Hosted: best-effort — a gateway blip never fails the run.
        # Local/desktop key: the same backend executes the runs, so an
        # unreachable backend is fatal here, before the id is registered or
        # returned (N3). Local Free already fails in ensure_ready().
        _persist_experiment_to_server(
            client,
            state.payload(),
            fatal_if_unreachable=client_mode(client) == MODE_LOCAL,
        )
        EXPERIMENTS.add(state, fingerprint)
        # Bind BEFORE spawning: the worker thread has no MCP request context
        # (fastmcp's header lookup answers {} there), so a request-scoped
        # client — the hosted gateway — cannot resolve its per-request token
        # mid-experiment. bound_client() runs here, inside the request.
        worker_binding = client.bound_client()
        threading.Thread(
            target=_experiment_worker,
            args=(state, worker_binding, fingerprint),
            name=f"nimbus-experiment-{experiment_id[:8]}",
            daemon=True,
        ).start()
        return {
            "experimentId": experiment_id,
            "status": "running",
            "runs": [{"name": run.name, "executionId": None} for run in prepared],
            "pollHint": "Runs submit in the background. Call experiment.get(experiment_id) "
            "until status is completed or failed — from any entry point signed in "
            "with the same account; the record is stored on the Nimbus server.",
        }

    @mcp.tool(name="experiment.delete", annotations=DESTRUCTIVE)
    def delete_experiment(experiment_id: str, confirm: bool = False) -> dict[str, Any]:
        """Delete one of your experiment records (server + this machine's local
        cache). Its individual runs are kept — delete them with execution.delete.
        A still-running experiment cannot be deleted.

        Args:
            experiment_id: The experiment to delete (from experiment.run).
            confirm: MUST be true to delete — the explicit user go-ahead.
        """
        client.ensure_ready()
        exp_id = safe_segment(experiment_id, label="experiment id")
        if not confirm:
            return {
                "deleted": False,
                "message": "Refusing to delete an experiment: pass confirm=true to proceed.",
            }
        local = EXPERIMENTS.remove(client.credential_fingerprint(), exp_id)
        if local == "running":
            raise local_tool_error(
                f"Experiment '{exp_id}' is still running; wait for it to finish "
                "(experiment.get), then delete it.",
                code="nimbus.experiment.running",
            )
        server = True
        try:
            client.delete(f"/api/experiments/{exp_id}")
        except McpToolError as err:
            if err.status_code != 404:
                raise
            server = False
        if not server and local == "missing":
            raise local_tool_error(
                f"Unknown experiment '{exp_id}' for this account.",
                code="nimbus.experiment.not_found",
            )
        return {"deleted": True, "experimentId": exp_id}

    @mcp.tool(name="experiment.get", annotations=READ_ONLY)
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        """Experiment snapshot: status (running/completed/failed), per-run rows
        ({name, executionId, status, error?, metrics?, evaluationKind?}) and,
        once finished, aggregates over completed runs only: per-metric
        {mean, std, best, n}, plus ``pooledExecutionIds`` / ``pooledRunNames``
        / ``kindBreakdown`` (per-subject vs LOSO etc. — each row has kind, n,
        mean, std, executionIds, runNames; #470). Pooled headline means are
        unchanged; kindBreakdown labels what was pooled. std is population;
        None below 2 values. kindBreakdown is always present when aggregates
        exist (including single-kind experiments).

        PORTABILITY: experiment records are stored on the Nimbus server
        under your account, so any entry point (local stdio, hosted
        gateway, another machine) signed in with the same account/token
        can read them; the serving process and the local
        ~/.nimbus/experiments store remain as fast-path caches.
        Experiments run before server-side storage landed exist only in
        the origin machine's local store and read as unknown elsewhere
        (#518).

        Args:
            experiment_id: The experiment to inspect (from experiment.run).
        """
        # Symmetry with experiment.run: in setup mode no experiment can exist
        # (its preflight refuses), so answer with guidance rather than the
        # confusing "Unknown experiment".
        client.ensure_ready()
        exp_id = safe_segment(experiment_id, label="experiment id")
        # The caller's own key only: another credential's experiment with the
        # same id is indistinguishable from a foreign-process one.
        snapshot = EXPERIMENTS.snapshot(client.credential_fingerprint(), exp_id)
        if snapshot is None:
            # Cross-entry-point read (#518): the backend record is the
            # shared store; local memory/disk are only this process's cache.
            snapshot = _fetch_server_experiment(client, exp_id)
        if snapshot is None:
            raise local_tool_error(
                f"Unknown experiment '{exp_id}'. No record for this "
                "account/token was found in this process, in this machine's "
                "local experiment store, or on the Nimbus server. The id may "
                "be wrong, may belong to a different account/token (foreign "
                "ids read exactly like missing ones), or the experiment may "
                "predate server-side experiment storage — those exist only "
                "in the origin machine's local store. If it was just "
                "started elsewhere, retry in a moment; otherwise re-run it "
                "from an entry point signed in with the same account.",
                code="nimbus.experiment.not_found",
            )
        # Recompute kindBreakdown (and enrich unstamped legacy rows from
        # /api/result) so classification matches stats.grouped on the same ids.
        return _ensure_kind_breakdown(snapshot, client=client)
