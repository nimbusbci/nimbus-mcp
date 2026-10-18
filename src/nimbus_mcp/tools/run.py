"""Run tools: execute pipelines, track executions, read results."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import DESTRUCTIVE, MUTATING, READ_ONLY
from ._guards import safe_segment

_METRIC_KEYS = (
    "accuracyPct",
    "trainAccuracyPct",
    "evalAccuracyPct",
    "balancedAccuracyPct",
    "kappa",
    "evalKappa",
    "itr",
    "pValue",
    "auc",
    "latencyMs",
    "rejectionRate",
    "generalizationGapPct",
    "nGroups",
)


def _compute_origin_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Passthrough server-stamped provenance when present on summary or nested metadata."""
    origin = row.get("computeOrigin")
    if isinstance(origin, dict):
        return origin
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("computeOrigin")
        if isinstance(nested, dict):
            return nested
    return None


def _trim_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "executionId",
        "status",
        "startedAt",
        "completedAt",
        "pipelineName",
        "errorMessage",
        "hasArtifacts",
        "dataset",
        "subject",
        "paradigm",
        "nTrain",
        "nEval",
        "nClasses",
        "chancePct",
    )
    trimmed = {k: row.get(k) for k in keys if row.get(k) is not None}
    if row.get("metrics"):
        trimmed["metrics"] = {
            k: row["metrics"][k] for k in _METRIC_KEYS if row["metrics"].get(k) is not None
        }
    origin = _compute_origin_from_row(row)
    if origin is not None:
        trimmed["computeOrigin"] = origin
    return trimmed


def synth_layout(train_graph: dict[str, Any]) -> dict[str, Any]:
    """Deterministic canvas layout for API-composed graphs (no canvas involved).

    The backend's POST /api/execute requires layout.nodes with x/y positions for
    every node; agents calling this server have no canvas, so positions are
    assigned on a fixed grid. Shared with tools/projects.py (project.save) so
    every agent-built graph lands on the canvas the same way.
    """
    nodes: dict[str, Any] = {}
    index = 0
    for node in train_graph.get("nodes") or []:
        if isinstance(node, dict) and node.get("id"):
            nodes[str(node["id"])] = {
                "x": float((index % 4) * 220),
                "y": float((index // 4) * 160),
            }
            index += 1
    return {"nodes": nodes}


# ── Duplicate in-flight run warning (non-blocking, MCP-side v1) ──────────────
# An agent that retries execution.run after a timeout (or re-submits an
# unchanged graph) burns compute and quota twice. Before POSTing, compare the
# new graph's fingerprint against this principal's RUNNING executions; a match
# adds a ``warning`` to the success payload — the run is never blocked.
_DUP_LIST_LIMIT = 20  # running executions inspected per preflight
_DUP_SNAPSHOT_FETCH_CAP = 5  # backend snapshot reads for runs this process didn't start
_REGISTRY_CAP = 200  # remembered runs per credential

_registry_lock = threading.Lock()
# credential fingerprint -> {execution_id: (graph fingerprint, subject)}
_run_registry: dict[str, dict[str, tuple[str, str | None]]] = {}


def graph_fingerprint(train_graph: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of the train nodes + connections.

    Order-insensitive (nodes sorted by id, connections sorted by their JSON) and
    layout-insensitive — only what changes the computation counts.
    """

    def _canon(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

    nodes = [
        {
            "id": n.get("id"),
            "type": n.get("type"),
            "config": n.get("config") or {},
        }
        for n in (train_graph.get("nodes") or [])
        if isinstance(n, dict)
    ]
    nodes.sort(key=lambda n: str(n["id"]))
    connections = sorted(
        _canon(c) for c in (train_graph.get("connections") or []) if isinstance(c, dict)
    )
    blob = _canon({"nodes": nodes, "connections": connections})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _same_subject(a: str | None, b: str | None) -> bool:
    return not a or not b or a.strip().upper() == b.strip().upper()


def _remember_run(scope: str, execution_id: str, fp: str, subject: str | None) -> None:
    with _registry_lock:
        runs = _run_registry.setdefault(scope, {})
        runs[execution_id] = (fp, subject)
        while len(runs) > _REGISTRY_CAP:
            runs.pop(next(iter(runs)))


def _find_duplicate_runs(
    client: NimbusClient, scope: str, fp: str, subject: str | None
) -> list[str]:
    """Running execution ids of this principal whose graph matches ``fp``.

    Best effort: ANY backend hiccup yields ``[]`` — a failed preflight must
    never stop a run (the warning is advisory).
    """
    try:
        payload = client.get(
            "/api/executions", params={"limit": _DUP_LIST_LIMIT, "status": "running"}
        )
        rows = payload.get("executions") if isinstance(payload, dict) else None
        running = [str(r["executionId"]) for r in rows or [] if r.get("executionId")]
        if not running:
            return []
        with _registry_lock:
            known = dict(_run_registry.get(scope, {}))
        dupes: list[str] = []
        fetched = 0
        for exec_id in running:
            if exec_id in known:
                other_fp, other_subject = known[exec_id]
                if other_fp == fp and _same_subject(subject, other_subject):
                    dupes.append(exec_id)
                continue
            if fetched >= _DUP_SNAPSHOT_FETCH_CAP:
                continue
            fetched += 1
            snap = client.get(f"/api/executions/{exec_id}/pipeline")
            pipeline = snap.get("pipeline") if isinstance(snap, dict) else None
            if not isinstance(pipeline, dict):
                continue
            graph = pipeline.get("train") or pipeline.get("execGraph")
            if isinstance(graph, dict) and graph_fingerprint(graph) == fp:
                if _same_subject(subject, pipeline.get("subject")):
                    dupes.append(exec_id)
        return dupes
    except (McpToolError, AttributeError, KeyError, TypeError, ValueError):
        return []


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="execution.run", annotations=MUTATING)
    def run_pipeline(
        train_graph: dict[str, Any],
        name: str | None = None,
        description: str | None = None,
        subject: str | None = None,
        layout: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a pipeline run (NON-BLOCKING). Returns executionId — poll with
        execution.get() until status is completed/failed, then execution.results().
        If an identical graph is already running for you, the run still starts but
        the response carries a ``warning`` + ``duplicateOf`` ids (cancel one with
        execution.cancel to avoid paying twice).

        Args:
            train_graph: Pipeline graph {nodes: [{id, type, config}], connections:
                [{from, to}]} as built by catalog.template/pipeline.validate.
            name: Display name for the run (shown in the Runs list).
            description: Optional longer description of the experiment.
            subject: Optional dataset subject code (e.g. "S01") recorded with the run.
            layout: Optional canvas positions {nodes: {id: {x, y}}}; a deterministic
                grid is synthesized when omitted.
        """
        body: dict[str, Any] = {
            "train": train_graph,
            "layout": layout if layout is not None else synth_layout(train_graph),
        }
        for key, value in (("name", name), ("description", description), ("subject", subject)):
            if value is not None:
                body[key] = value
        scope = client.credential_fingerprint()
        fp = graph_fingerprint(train_graph)
        duplicates = _find_duplicate_runs(client, scope, fp, subject)
        payload = client.post("/api/execute", json=body)
        execution_id = payload.get("executionId")
        if execution_id:
            _remember_run(scope, str(execution_id), fp, subject)
        result: dict[str, Any] = {
            "executionId": execution_id,
            "message": payload.get("message"),
            "pollHint": "Call execution.get(execution_id) to track progress; "
            "execution.results(execution_id) once completed.",
        }
        if duplicates:
            result["duplicateOf"] = duplicates
            result["warning"] = (
                f"An identical pipeline is already running ({', '.join(duplicates)}). "
                "This run was started anyway — if it was a retry, cancel one with "
                "execution.cancel(execution_id) to avoid duplicate compute and quota use."
            )
        return result

    @mcp.tool(name="execution.cancel", annotations=DESTRUCTIVE)
    def cancel_execution(execution_id: str) -> dict[str, Any]:
        """Cancel a running execution.

        Args:
            execution_id: The run to terminate (from execution.run/execution.list).
        """
        exec_id = safe_segment(execution_id)
        return client.post("/api/cancel", json={"executionId": exec_id})

    @mcp.tool(name="execution.get", annotations=READ_ONLY)
    def get_execution(execution_id: str) -> dict[str, Any]:
        """Execution status summary (status: running/completed/failed/cancelled).

        Args:
            execution_id: The run to check (from execution.run/execution.list).
        """
        exec_id = safe_segment(execution_id)
        payload = client.get(f"/api/executions/{exec_id}/summary")
        return _trim_summary(payload.get("summary", {}))

    @mcp.tool(name="execution.list", annotations=READ_ONLY)
    def list_executions(limit: int = 20, status: str | None = None) -> dict[str, Any]:
        """Recent executions. Optional status filter (running/completed/failed/cancelled).

        Args:
            limit: Maximum number of runs to return (default 20).
            status: Filter by run status: running/completed/failed/cancelled.
        """
        params: dict[str, Any] = {"limit": limit}
        if status is not None:
            params["status"] = status
        payload = client.get("/api/executions", params=params)
        rows = [_trim_summary(row) for row in payload.get("executions", [])]
        return {"total": payload.get("total", len(rows)), "executions": rows}

    @mcp.tool(name="execution.results", annotations=READ_ONLY)
    def get_results(execution_id: str, full: bool = False) -> dict[str, Any]:
        """Metrics for a completed run. Trimmed by default (accuracy, kappa, ITR,
        confusion matrix, per-class); full=True returns the complete result object.

        Args:
            execution_id: The completed run to fetch metrics for.
            full: Return the backend's complete result object (all fields).
        """
        exec_id = safe_segment(execution_id)
        payload = client.get(f"/api/result/{exec_id}")
        result = payload.get("result", {})
        if full:
            return result
        metrics = result.get("metrics", {})
        trimmed: dict[str, Any] = {
            "executionId": exec_id,
            "metrics": {k: metrics[k] for k in _METRIC_KEYS if metrics.get(k) is not None},
            "confusionMatrix": result.get("confusionMatrix"),
            "confusionMatrixLabels": result.get("confusionMatrixLabels"),
            "evalConfusionMatrix": result.get("evalConfusionMatrix"),
            "evalConfusionMatrixLabels": result.get("evalConfusionMatrixLabels"),
            "perClassAccuracy": result.get("perClassAccuracy"),
            "artifactFiles": [
                {"name": f.get("name"), "size": f.get("size")}
                for f in result.get("artifactFiles", [])
            ],
        }
        origin = _compute_origin_from_row(result)
        if origin is not None:
            trimmed["computeOrigin"] = origin
        return trimmed
