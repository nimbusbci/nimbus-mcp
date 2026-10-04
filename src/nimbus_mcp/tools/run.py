"""Run tools: execute pipelines, track executions, read results."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
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


def _trim_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "executionId", "status", "startedAt", "completedAt", "pipelineName",
        "errorMessage", "hasArtifacts", "dataset", "subject", "paradigm",
        "nTrain", "nEval", "nClasses", "chancePct",
    )
    trimmed = {k: row.get(k) for k in keys if row.get(k) is not None}
    if row.get("metrics"):
        trimmed["metrics"] = {
            k: row["metrics"][k] for k in _METRIC_KEYS if row["metrics"].get(k) is not None
        }
    return trimmed


def synth_layout(train_graph: dict[str, Any]) -> dict[str, Any]:
    """Deterministic canvas layout for API-composed graphs (no canvas involved).

    The backend's POST /api/execute requires layout.nodes with x/y positions for
    every node; agents calling this server have no canvas, so positions are
    assigned on a fixed grid. Shared with tools/projects.py (save_pipeline) so
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


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(annotations=MUTATING)
    def run_pipeline(
        train_graph: dict[str, Any],
        name: str | None = None,
        description: str | None = None,
        subject: str | None = None,
        layout: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a pipeline run (NON-BLOCKING). Returns executionId — poll with
        get_execution() until status is completed/failed, then get_results().

        Args:
            train_graph: Pipeline graph {nodes: [{id, type, config}], connections:
                [{from, to}]} as built by get_template/validate_pipeline.
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
        payload = client.post("/api/execute", json=body)
        return {
            "executionId": payload.get("executionId"),
            "message": payload.get("message"),
            "pollHint": "Call get_execution(execution_id) to track progress; "
            "get_results(execution_id) once completed.",
        }

    @mcp.tool(annotations=DESTRUCTIVE)
    def cancel_execution(execution_id: str) -> dict[str, Any]:
        """Cancel a running execution.

        Args:
            execution_id: The run to terminate (from run_pipeline/list_executions).
        """
        exec_id = safe_segment(execution_id)
        return client.post("/api/cancel", json={"executionId": exec_id})

    @mcp.tool(annotations=READ_ONLY)
    def get_execution(execution_id: str) -> dict[str, Any]:
        """Execution status summary (status: running/completed/failed/cancelled).

        Args:
            execution_id: The run to check (from run_pipeline/list_executions).
        """
        exec_id = safe_segment(execution_id)
        payload = client.get(f"/api/executions/{exec_id}/summary")
        return _trim_summary(payload.get("summary", {}))

    @mcp.tool(annotations=READ_ONLY)
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

    @mcp.tool(annotations=READ_ONLY)
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
        return {
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
