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
    def validate_node_config(node_type: str, config: dict[str, Any]) -> dict[str, Any]:
        """Validate one node's config object against its schema (catalog.node_schema).

        Args:
            node_type: Node type id from catalog.nodes (e.g. "csp", "nimbus_lda").
            config: The node's config object to check.
        """
        return client.post(
            "/api/validate-node-config", json={"nodeType": node_type, "config": config}
        )
