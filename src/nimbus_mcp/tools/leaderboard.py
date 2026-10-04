"""Leaderboard tools: public benchmark rankings per dataset."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._annotations import READ_ONLY


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(annotations=READ_ONLY)
    def get_leaderboard() -> dict[str, Any]:
        """Public benchmark leaderboard: pipeline rankings per dataset.

        Rankings are per-dataset under the canonical ``within_session`` protocol
        (see ``protocol``). Within each dataset, ``rows`` are sorted desc by
        ``meanAccuracyPct`` (95% CI in ``ciLoPct``/``ciHiPct``). Use ``pipelineId``
        as the template id hint for ``get_template`` when building a pipeline.
        ``updated`` marks each dataset's most recent run; ``packFingerprint``
        identifies the exact dataset pack the scores came from.
        """
        return client.get("/api/leaderboard")
