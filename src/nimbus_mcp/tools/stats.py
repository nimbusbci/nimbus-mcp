"""Statistics tools: review-grade inference over stored results (MCP v0.12)."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import READ_ONLY
from ._guards import safe_segment


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="stats.grouped", annotations=READ_ONLY)
    def grouped(
        execution_ids: list[str],
        chance: float | None = None,
    ) -> dict[str, Any]:
        """Cluster-aware group statistics across executions: pools the
        per-subject (per-group) accuracies each run's evaluation plan stored,
        then reports the between-group summary — mean accuracy with a t-interval
        and a one-sample t-test vs chance. Use this instead of eyeballing
        pooled accuracy: trials within a subject are not independent, so this
        is the MOABB-style number that survives review.

        Args:
            execution_ids: 1–25 executions (from execution.run) to pool. Runs
                without stored group stats (fewer than 3 groups) are reported
                with nulls but do not fail the request.
            chance: Chance level in [0, 1]; default 1/nClasses per execution.
        """
        if not execution_ids:
            raise McpToolError("execution_ids must contain at least one id.")
        ids = [safe_segment(x, label="execution id") for x in execution_ids][:25]
        body: dict[str, Any] = {"executionIds": ids}
        if chance is not None:
            body["chance"] = chance
        return client.post("/api/stats/grouped", json=body)
