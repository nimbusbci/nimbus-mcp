"""Leaderboard tools: public benchmark rankings + BYO submissions."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._annotations import MUTATING, READ_ONLY


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="catalog.leaderboard", annotations=READ_ONLY)
    def get_leaderboard() -> dict[str, Any]:
        """Public benchmark leaderboard: pipeline rankings per dataset.

        Two tracks: the top-level ``within_session`` block (curated suite
        pipelines) and ``crossSubject`` (leave-one-subject-out — curated
        pipelines plus BYO ``python_model`` submissions; BYO rows carry
        ``kind: "byo"`` and an ``owner`` name). Within each dataset, ``rows``
        sort desc by ``meanAccuracyPct`` (95% CI in ``ciLoPct``/``ciHiPct``).
        ``packFingerprint`` identifies the exact dataset pack behind the
        scores; ``updated`` marks each dataset's most recent run.
        """
        return client.get("/api/leaderboard")

    @mcp.tool(name="leaderboard.submit", annotations=MUTATING)
    def submit_leaderboard_model(
        dataset_id: str,
        name: str,
        code: str,
        class_name: str | None = None,
        requirements: list[str] | None = None,
    ) -> dict[str, Any]:
        """Submit a BYO python_model to the cross-subject (LOSO) leaderboard.

        Nimbus scores it server-side under the fixed protocol (seed 42,
        leave-one-subject-out over the dataset's subjects) — you submit code,
        never results. Poll ``leaderboard.status`` with the returned
        ``submissionId``; scoring takes minutes (classical) to tens of minutes
        (deep). Requirements must be plain PyPI names/specs (max 20, no URLs).
        Identical code resubmission is idempotent (``deduplicated: true``).

        Args:
            dataset_id: Open dataset id (see catalog.leaderboard crossSubject
                datasets, e.g. "BNCI2014_001").
            name: Display name on the board (<=100 chars).
            code: Python source defining a class with fit(X, y, info) and
                predict(X) (<=256KB; validate first with python.validate).
            class_name: Contract class to instantiate; omit to auto-detect.
            requirements: Optional PyPI requirements (e.g. ["sklearn>=1.4"]).
        """
        body: dict[str, Any] = {"datasetId": dataset_id, "name": name, "code": code}
        if class_name:
            body["className"] = class_name
        if requirements:
            body["requirements"] = requirements
        return client.post("/api/leaderboard/submissions", json=body)

    @mcp.tool(name="leaderboard.status", annotations=READ_ONLY)
    def get_leaderboard_submission(submission_id: str) -> dict[str, Any]:
        """Fetch one of your leaderboard submissions (owner-only).

        Returns status (queued | scoring | done | failed), metrics when done
        (meanAccuracyPct + CI, same shape as leaderboard rows), or the error
        that failed it.

        Args:
            submission_id: The submissionId returned by leaderboard.submit.
        """
        return client.get(f"/api/leaderboard/submissions/{submission_id}")

    @mcp.tool(name="leaderboard.mine", annotations=READ_ONLY)
    def list_my_leaderboard_submissions() -> dict[str, Any]:
        """List your leaderboard submissions (newest first), all datasets."""
        return client.get("/api/leaderboard/submissions")
