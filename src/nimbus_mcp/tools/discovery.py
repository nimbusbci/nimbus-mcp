"""Discovery tools: node catalog, templates, datasets."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ..errors import local_tool_error
from ._annotations import READ_ONLY
from ._guards import safe_segment

NODE_TYPE_NOT_FOUND = "nimbus.catalog.node_type_not_found"
UNKNOWN_CATEGORY = "nimbus.catalog.unknown_category"

#: Plural / common spellings agents try (round 7: the doc itself said "models").
_CATEGORY_ALIASES = {
    "models": "model",
    "features": "feature",
    "outputs": "output",
    "inputs": "data",
    "sources": "data",
}


def _resolve_category(category: str, valid: list[str]) -> str:
    wanted = category.strip().lower()
    if wanted in valid:
        return wanted
    alias = _CATEGORY_ALIASES.get(wanted)
    if alias in valid:
        return alias
    if wanted.endswith("s") and wanted[:-1] in valid:
        return wanted[:-1]
    raise local_tool_error(
        f"Unknown node category {category!r}. Valid categories: {', '.join(valid)}.",
        code=UNKNOWN_CATEGORY,
        status_code=400,
    )


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="catalog.nodes", title="List Pipeline Nodes", annotations=READ_ONLY)
    def list_nodes(category: str | None = None) -> dict[str, Any]:
        """List Nimbus pipeline node types (data, preprocessing, features, models...).

        Use catalog.node_schema(node_type) for one node's full config schema and ports.

        Args:
            category: Optional filter by category id — e.g. "data",
                "preprocessing", "feature", "model" (singular ids; plurals such
                as "models"/"features" are accepted). An unknown category is an
                error that lists the valid ids.
        """
        payload = client.get("/api/node-types")
        all_nodes = payload.get("nodeTypes", [])
        if category is not None:
            valid = sorted({str(n.get("category")) for n in all_nodes if n.get("category")})
            category = _resolve_category(category, valid)
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
            for n in all_nodes
            if category is None or n.get("category") == category
        ]
        return {"count": len(nodes), "nodes": nodes}

    @mcp.tool(name="catalog.node_schema", title="Node Config Schema", annotations=READ_ONLY)
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
        raise local_tool_error(
            f"Unknown node type '{node_type}'. Call catalog.nodes() for valid ids.",
            code=NODE_TYPE_NOT_FOUND,
            status_code=404,
        )

    @mcp.tool(name="catalog.templates", title="List Pipeline Templates", annotations=READ_ONLY)
    def list_templates(dataset: str | None = None, paradigm: str | None = None) -> dict[str, Any]:
        """List built-in starter pipelines (MI/P300/SSVEP...).
        catalog.template(id) returns the graph. Each row carries ``datasetId``
        (catalog id, e.g. "BNCI2014_001" — the id catalog.leaderboard and
        catalog.datasets use) and ``paradigm`` when the template loads a public dataset.

        Args:
            dataset: Only templates on this dataset id (e.g. "BNCI2014_001"); case-insensitive.
            paradigm: Only templates with this paradigm (e.g. "motor_imagery", "p300", "ssvep").
        """
        payload = client.get("/api/templates")
        templates = [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "description": t.get("description", ""),
                "category": t.get("category"),
                "expectedAccuracy": t.get("expectedAccuracy"),
                "datasetId": t.get("datasetId"),
                "paradigm": t.get("paradigm"),
            }
            for t in payload.get("templates", [])
        ]

        def _norm(v: Any) -> str:
            return str(v or "").strip().lower().replace("-", "_").replace(" ", "_")

        if dataset:
            templates = [t for t in templates if _norm(t["datasetId"]) == _norm(dataset)]
        if paradigm:
            templates = [t for t in templates if _norm(t["paradigm"]) == _norm(paradigm)]
        return {"count": len(templates), "templates": templates}

    @mcp.tool(name="catalog.template", title="Get Pipeline Template", annotations=READ_ONLY)
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

    @mcp.tool(name="catalog.datasets", title="List Datasets", annotations=READ_ONLY)
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
