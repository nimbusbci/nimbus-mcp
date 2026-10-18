"""Statistics tools: review-grade inference over stored results (MCP v0.12)."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import READ_ONLY
from ._guards import safe_segment

MAX_EXECUTIONS = 25  # mirrored from the backend's grouped-stats request model


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="stats.grouped", annotations=READ_ONLY)
    def grouped(
        execution_ids: list[str],
        chance: float | None = None,
    ) -> dict[str, Any]:
        """Cluster-aware group statistics across executions.

        REQUIREMENT (read first): each execution must have been run with an
        evaluation plan that STORED per-group (per-subject) accuracies for at
        least 3 groups — e.g. leave-one-subject-out (LOSO) or grouped CV over
        >= 3 subjects. A plain single-subject CV run (trials split within one
        subject, one group) stores no group stats and CANNOT be analysed here.
        If none of the given executions qualifies, the backend fails with
        ``nimbus.stats.insufficient_groups`` — that error means exactly this:
        re-run with a multi-subject / group-wise evaluation plan (>= 3 groups),
        or pass runs from such a sweep.

        What it does: pools the per-subject (per-group) accuracies each run's
        evaluation plan stored, then reports the between-group summary — mean
        accuracy with a t-interval and a one-sample t-test vs chance. Use this
        instead of eyeballing pooled accuracy: trials within a subject are not
        independent, so this is the MOABB-style number that survives review.

        Args:
            execution_ids: 1–25 executions (from execution.run) to pool. Runs
                without stored group stats (fewer than 3 groups) are reported
                with nulls but do not fail the request — unless NONE of them
                has group stats (then ``nimbus.stats.insufficient_groups``).
            chance: Chance level in [0, 1]; default 1/nClasses per execution.
        """
        if not execution_ids:
            raise McpToolError("execution_ids must contain at least one id.")
        if len(execution_ids) > MAX_EXECUTIONS:
            # Refuse loudly instead of silently trimming: a pooled mean/t-test
            # over an arbitrary subset reported as the full result is exactly
            # the confidently-wrong number this tool exists to prevent.
            raise McpToolError(
                f"stats.grouped accepts at most {MAX_EXECUTIONS} execution ids "
                f"(got {len(execution_ids)}). Split the comparison into batches."
            )
        ids = [safe_segment(x, label="execution id") for x in execution_ids]
        body: dict[str, Any] = {"executionIds": ids}
        if chance is not None:
            body["chance"] = chance
        return client.post("/api/stats/grouped", json=body)
