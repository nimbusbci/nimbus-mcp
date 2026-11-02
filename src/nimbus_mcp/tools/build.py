"""Build tools: graph and node-config validation."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ..errors import local_tool_error
from ._annotations import READ_ONLY
from .discovery import NODE_TYPE_NOT_FOUND


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="pipeline.validate", title="Validate Pipeline", annotations=READ_ONLY)
    def validate_pipeline(train_graph: dict[str, Any]) -> dict[str, Any]:
        """Validate a pipeline graph before running.
        ExecGraphSnapshot: {nodes: [{id, type, config}], connections: [{from, to}]}.
        Build it from catalog.template(id).train or from scratch using catalog.nodes().

        Args:
            train_graph: The graph to validate ({nodes, connections}).
        """
        return client.post("/api/validate-pipeline", json={"train": train_graph})

    @mcp.tool(name="pipeline.validate_node", title="Validate Node Config", annotations=READ_ONLY)
    def validate_node_config(
        node_type: str,
        config: dict[str, Any],
        dataset: str | None = None,
    ) -> dict[str, Any]:
        """Validate one node's config (schema + semantic checks).

        Catches the same value errors as pipeline.validate for a single node
        (inverted bandpass, CSP components vs channel count when dataset is set).

        Args:
            node_type: Node type id from catalog.nodes (e.g. "csp", "bandpass_filter").
            config: The node's config object to check.
            dataset: Optional MOABB pack id (e.g. "BNCI2014_001") so CSP/Nyquist
                checks can use channel count and sampling rate. Omitted → those
                context checks are skipped; inverted-band checks still run.
        """
        body: dict[str, Any] = {"nodeType": node_type, "config": config}
        if dataset is not None:
            body["dataset"] = dataset
        result = client.post("/api/validate-node-config", json=body)
        if isinstance(result, dict) and result.get("valid") is False:
            errors = result.get("errors") or []
            if any(isinstance(e, str) and e.startswith("Unknown node type") for e in errors):
                # Same tool error as catalog.node_schema for an unknown id
                # (#644); a normal result with the code in the body was
                # inconsistent with every other not-found path (E2E r8 B5).
                raise local_tool_error(
                    f"Unknown node type {node_type!r}. Use catalog.nodes to list valid types.",
                    code=NODE_TYPE_NOT_FOUND,
                    status_code=404,
                )
        return result
