"""Foundation-model tools (Nimbus MCP v0.13.1).

The foundation lane's agent answer to "which pretrained backbone should I
use?": run every available one on the user's own data in a single call and
look at the ranked chart, instead of guessing from leaderboard rows
measured on other people's data.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.base import ToolResult
from fastmcp.utilities.types import Image
from mcp.types import TextContent

from ..access import MODE_HOSTED, TIER_PRO, account_tier, client_mode
from ..client import McpToolError, NimbusClient
from ._annotations import FIGURE_OUTPUT_SCHEMA, READ_ONLY
from ._figures import decode_png, figure_info
from ._guards import DATA_SOURCES, PACK_MODES, choice, require_choice
from ._hosted_files import LOCAL_SOURCE_HINT, refuse_local_path_on_gateway

_SOURCES = DATA_SOURCES
SourceArg = choice(DATA_SOURCES)
ModeArg = choice(PACK_MODES)


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(
        name="foundation.compare",
        title="Compare Foundation Backbones",
        annotations=READ_ONLY,
        output_schema=FIGURE_OUTPUT_SCHEMA,
    )
    def compare(
        source: SourceArg,
        dataset: str | None = None,
        subject: str | None = None,
        mode: ModeArg = "all",
        path: str | None = None,
    ) -> ToolResult:
        """Run every available pretrained backbone on YOUR data and see which wins —
        a ranked bar chart with CI whiskers, params, and wall time per backbone.
        Backbones that can't process your data (window size, channel count) are
        noted, not silently skipped. Each backbone embeds your
        trials frozen (the exact foundation_features preprocessing), a LDA is
        cross-validated on the embeddings. Returns a PNG image PLUS a JSON text
        block (and structuredContent.meta.results) with per-backbone accuracy,
        CI, embed dim, params, and time — pick the foundation_features backbone
        by measurement, not vibes.

        Args:
            source: "dataset" (MOABB pack), "upload" (path from data.upload), or
                "local" (absolute path, local backend only).
            dataset: MOABB dataset id, e.g. "BNCI2014_001" (source=dataset).
            subject: Subject id, e.g. "S01" (source=dataset).
            mode: Pack split to load: training | evaluation | all (source=dataset).
            path: Upload path (source=upload) or absolute local path (source=local).
        """
        require_choice(source, _SOURCES, label="source")
        require_choice(mode, PACK_MODES, label="mode")
        if source == "local":
            refuse_local_path_on_gateway(
                client,
                tool="foundation.compare",
                hint=LOCAL_SOURCE_HINT,
            )
        if client_mode(client) == MODE_HOSTED and account_tier(client) != TIER_PRO:
            # X10: hosted foundation.compare is a Pro feature. Local/desktop
            # backends run it on the user's own machine and are NOT gated.
            # (Raised, not returned: this tool's output schema is a figure.)
            raise McpToolError(
                "foundation.compare on the hosted backend is a Pro feature "
                "(tier model X10) and this account is on the free tier. "
                "Alternatives: run the same comparison against the Nimbus "
                "Studio desktop app backend (local, ungated), pick a backbone "
                "from catalog.leaderboard, or upgrade in Nimbus Studio -> "
                "Account."
            )
        payload = client.post(
            "/api/foundation/compare",
            json={
                "source": source,
                "dataset": dataset,
                "subject": subject,
                "mode": mode,
                "path": path,
            },
        )
        png, meta = decode_png(payload)
        info: dict[str, Any] = figure_info(payload, meta)
        # Image + machine-readable text (clients that ignore structuredContent
        # still get meta.results as JSON — #471) + structuredContent sidecar.
        results_text = json.dumps(
            {
                "title": info.get("title"),
                "results": meta.get("results"),
                "best": meta.get("best"),
                "skipped": meta.get("skipped"),
            },
            default=str,
        )
        # Text first so clients that only surface the first content block still
        # get machine-readable numbers alongside the chart (#471).
        return ToolResult(
            content=[
                TextContent(type="text", text=results_text),
                Image(data=png, format="png"),
            ],
            structured_content=info,
        )
