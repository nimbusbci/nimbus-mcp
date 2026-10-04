"""Project tools: persist agent-built pipelines into the studio workspace."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import MUTATING, READ_ONLY
from ._guards import safe_segment
from .run import synth_layout


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(annotations=MUTATING)
    def create_project(name: str, description: str | None = None) -> dict[str, Any]:
        """Create a project (container for one pipeline document). Returns projectId.

        Args:
            name: Project display name (e.g. "MI CSP-LDA sweep").
            description: Optional longer description shown in the studio.
        """
        body: dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        payload = client.post("/api/projects", json=body)
        project = payload.get("project", {})
        return {"ok": True, "projectId": project.get("id"), "name": project.get("name")}

    @mcp.tool(annotations=READ_ONLY)
    def list_projects() -> dict[str, Any]:
        """List projects owned by the current principal (agent work included)."""
        payload = client.get("/api/projects")
        return {
            "count": payload.get("count"),
            "projects": [
                {"id": p.get("id"), "name": p.get("name"),
                 "description": p.get("description"), "updatedAt": p.get("updatedAt")}
                for p in payload.get("projects", [])
            ],
        }

    @mcp.tool(annotations=MUTATING)
    def save_pipeline(
        project_id: str,
        train_graph: dict[str, Any],
        name: str | None = None,
        subject: str | None = None,
    ) -> dict[str, Any]:
        """Save a pipeline graph into a project (visible on the studio canvas).
        Handles revision conflicts automatically (one retry).

        Args:
            project_id: Target project (from create_project/list_projects).
            train_graph: Pipeline graph {nodes, connections} to persist.
            name: Optional pipeline name stored on the document.
            subject: Optional dataset subject code (e.g. "S01") for this pipeline.
        """
        pid = safe_segment(project_id, label="project id")
        current = client.get(f"/api/projects/{pid}/doc")
        body: dict[str, Any] = {
            "train": {**train_graph, "layout": train_graph.get("layout")
                      or synth_layout(train_graph)},
            "expectedRevision": current.get("revision", 0),
        }
        if name is not None:
            body["name"] = name
        if subject is not None:
            body["subject"] = subject
        payload = client.put(f"/api/projects/{pid}/doc", json=body)
        if payload.get("status") == 409:
            # Lost the revision race once — re-read the current revision and
            # retry exactly one more time before surfacing the conflict.
            fresh = client.get(f"/api/projects/{pid}/doc")
            body["expectedRevision"] = fresh.get("revision", body["expectedRevision"])
            payload = client.put(f"/api/projects/{pid}/doc", json=body)
            if payload.get("status") == 409:
                raise McpToolError(
                    f"Pipeline save conflict persists after one retry "
                    f"({payload.get('code', 'conflict')}). The project document was "
                    f"modified concurrently — call load_pipeline('{pid}') to fetch the "
                    "latest revision, merge your changes on top, and save again."
                )
        return {"ok": payload.get("ok", True), "projectId": payload.get("projectId", pid),
                "revision": payload.get("revision"), "message": payload.get("message")}

    @mcp.tool(annotations=READ_ONLY)
    def load_pipeline(project_id: str) -> dict[str, Any]:
        """Load a project's saved pipeline (train graph + meta) for editing/re-running.

        Args:
            project_id: Project whose pipeline document to load.
        """
        pid = safe_segment(project_id, label="project id")
        payload = client.get(f"/api/projects/{pid}/doc")
        return {
            "projectId": payload.get("projectId", pid),
            "name": payload.get("name"),
            "subject": payload.get("subject"),
            "revision": payload.get("revision"),
            "train": payload.get("train"),
        }
