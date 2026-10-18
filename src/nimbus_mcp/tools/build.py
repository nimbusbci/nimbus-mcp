"""Build tools: graph and node-config validation."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._annotations import READ_ONLY


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="pipeline.validate", annotations=READ_ONLY)
    def validate_pipeline(train_graph: dict[str, Any]) -> dict[str, Any]:
        """Validate a pipeline graph before running.
        ExecGraphSnapshot: {nodes: [{id, type, config}], connections: [{from, to}]}.
        Build it from catalog.template(id).train or from scratch using catalog.nodes().

        Args:
            train_graph: The graph to validate ({nodes, connections}).
        """
        return client.post("/api/validate-pipeline", json={"train": train_graph})

    @mcp.tool(name="pipeline.validate_node", annotations=READ_ONLY)
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
        return client.post("/api/validate-node-config", json=body)
