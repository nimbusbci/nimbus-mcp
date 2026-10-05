"""Discovery tools: node catalog, templates, datasets."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import READ_ONLY
from ._guards import safe_segment


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="catalog.nodes", annotations=READ_ONLY)
    def list_nodes(category: str | None = None) -> dict[str, Any]:
        """List Nimbus pipeline node types (data, preprocessing, features, models...).

        Use catalog.node_schema(node_type) for one node's full config schema and ports.

        Args:
            category: Optional filter — e.g. "data", "preprocessing", "features",
                "models" (exact category ids from the unfiltered list).
        """
        payload = client.get("/api/node-types")
        nodes = [
            {
                "id": n.get("id"),
                "name": n.get("name"),
                "category": n.get("category"),
                "description": n.get("description", ""),
                "inputs": [
                    {"name": p.get("name"), "type": p.get("type")} for p in n.get("inputs", [])
                ],
                "outputs": [
                    {"name": p.get("name"), "type": p.get("type")} for p in n.get("outputs", [])
                ],
            }
            for n in payload.get("nodeTypes", [])
            if category is None or n.get("category") == category
        ]
        return {"count": len(nodes), "nodes": nodes}

    @mcp.tool(name="catalog.node_schema", annotations=READ_ONLY)
    def get_node_schema(node_type: str) -> dict[str, Any]:
        """Full config JSON schema + input/output ports for one node type.

        Args:
            node_type: Node id from catalog.nodes (e.g. "csp", "nimbus_lda").
        """
        payload = client.get("/api/node-types")
        for node in payload.get("nodeTypes", []):
            if node.get("id") == node_type:
                return {
                    "id": node_type,
                    "name": node.get("name"),
                    "configSchema": node.get("configSchema", {}),
                    "ports": {"inputs": node.get("inputs", []), "outputs": node.get("outputs", [])},
                }
        raise McpToolError(f"Unknown node type '{node_type}'. Call catalog.nodes() for valid ids.")

    @mcp.tool(name="catalog.templates", annotations=READ_ONLY)
    def list_templates() -> dict[str, Any]:
        """List built-in starter pipelines (MI/P300/SSVEP...).
        catalog.template(id) returns the graph."""
        payload = client.get("/api/templates")
        templates = [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "description": t.get("description", ""),
                "category": t.get("category"),
                "expectedAccuracy": t.get("expectedAccuracy"),
            }
            for t in payload.get("templates", [])
        ]
        return {"count": len(templates), "templates": templates}

    @mcp.tool(name="catalog.template", annotations=READ_ONLY)
    def get_template(template_id: str) -> dict[str, Any]:
        """Full template incl. the 'train' execGraph needed by execution.run/pipeline.validate.

        Args:
            template_id: Template id from catalog.templates (e.g. "mi_csp_lda").
        """
        tid = safe_segment(template_id, label="template id")
        payload = client.get(f"/api/templates/{tid}")
        template = payload.get("template", {})
        return {
            "id": template.get("id"),
            "name": template.get("name"),
            "description": template.get("description", ""),
            "train": template.get("train"),
        }

    @mcp.tool(name="catalog.datasets", annotations=READ_ONLY)
    def list_datasets(only_on_disk: bool = True) -> dict[str, Any]:
        """Curated public EEG datasets (MOABB packs) available to pipelines.

        Args:
            only_on_disk: Only return datasets whose data packs are present on this
                backend (True by default; False also lists known-but-missing sets).
        """
        payload = client.get("/api/public-datasets/index")
        datasets = []
        for dataset_id, entry in payload.get("datasets", {}).items():
            if only_on_disk and not entry.get("onDisk"):
                continue
            subjects = entry.get("subjects") or []
            details = entry.get("details") or {}
            datasets.append(
                {
                    "id": dataset_id,
                    "label": entry.get("label"),
                    "paradigm": entry.get("paradigm"),
                    "onDisk": bool(entry.get("onDisk")),
                    "defaultSubject": entry.get("defaultSubject"),
                    "subjectsCount": len(subjects),
                    "channels": details.get("channels"),
                    "samplingRate": details.get("samplingRate"),
                }
            )
        return {"count": len(datasets), "datasets": datasets}
