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
        evaluation plan that STORED per-SUBJECT accuracies for at least 3
        subjects — e.g. leave-one-subject-out (LOSO) or grouped CV over
        >= 3 subjects. Session groups within one subject (multi-session
        single-subject runs) are refused with
        ``nimbus.stats.non_subject_groups`` — those are not independent
        subjects. A plain single-subject CV run with no subject/session
        grouping stores no group stats and fails with
        ``nimbus.stats.insufficient_groups``.

        What it does: pools the per-subject accuracies each run's evaluation
        plan stored, then reports the between-group summary — mean accuracy
        with a t-interval and a one-sample t-test vs chance. Responses label
        ``groupKind``, real ``groupKeys`` (subject ids like S01 when stamped
        or recoverable), ``perGroup`` (key + accuracy pairs),
        ``overall.pooledExecutionIds``, and ``kindBreakdown`` / 
        ``overall.kindBreakdown`` (per_subject vs loso etc. with n + mean —
        #470; between-group pooled mean is unchanged).

        Args:
            execution_ids: 1–25 executions (from execution.run) to pool. Runs
                without subject group stats are reported with nulls but do not
                fail the request — unless NONE qualifies
                (``nimbus.stats.insufficient_groups``) or only session groups
                are present (``nimbus.stats.non_subject_groups``).
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
