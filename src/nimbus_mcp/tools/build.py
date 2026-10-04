"""Build tools: graph and node-config validation."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient

GRAPH_HELP = (
    "ExecGraphSnapshot: {nodes: [{id, type, config}], connections: [{from, to}]}. "
    "Build it from get_template(id).train or from scratch using list_nodes()."
)


def register(mcp: FastMCP, client: NimbusClient) -> None:
    # NOTE: description is passed to @mcp.tool explicitly because
    # `"""doc""" + GRAPH_HELP` is not a string literal and would not become __doc__.
    @mcp.tool(description="Validate a pipeline graph before running. " + GRAPH_HELP)
    def validate_pipeline(train_graph: dict[str, Any]) -> dict[str, Any]:
        return client.post("/api/validate-pipeline", json={"train": train_graph})

    @mcp.tool
    def validate_node_config(node_type: str, config: dict[str, Any]) -> dict[str, Any]:
        """Validate one node's config object against its schema (get_node_schema)."""
        return client.post(
            "/api/validate-node-config", json={"nodeType": node_type, "config": config}
        )
