"""Project tools: persist agent-built pipelines into the studio workspace."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import DESTRUCTIVE, MUTATING, READ_ONLY
from ._guards import safe_segment
from .run import synth_layout


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="project.create", annotations=MUTATING)
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

    @mcp.tool(name="project.list", annotations=READ_ONLY)
    def list_projects() -> dict[str, Any]:
        """List projects owned by the current principal (agent work included)."""
        payload = client.get("/api/projects")
        return {
            "count": payload.get("count"),
            "projects": [
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "updatedAt": p.get("updatedAt"),
                }
                for p in payload.get("projects", [])
            ],
        }

    @mcp.tool(name="project.save", annotations=MUTATING)
    def save_pipeline(
        project_id: str,
        train_graph: dict[str, Any],
        expected_revision: int,
        name: str | None = None,
        subject: str | None = None,
    ) -> dict[str, Any]:
        """Save a pipeline graph into a project (visible on the studio canvas).

        Always pass expected_revision from the last project.load (or a prior
        save/create response). When it does not match the current server
        revision, the save fails with a conflict — it does NOT silently
        overwrite concurrent edits. Omitting it is rejected so a stale
        single-writer convenience path cannot wipe another client's work.

        Args:
            project_id: Target project (from project.create/project.list).
            train_graph: Pipeline graph {nodes, connections} to persist.
            name: Optional pipeline name stored on the document.
            subject: Optional dataset subject code (e.g. "S01") for this pipeline.
            expected_revision: Optimistic-concurrency base from project.load /
                a prior save. Required.
        """
        pid = safe_segment(project_id, label="project id")
        revision = int(expected_revision)
        body: dict[str, Any] = {
            "train": {
                **train_graph,
                "layout": train_graph.get("layout") or synth_layout(train_graph),
            },
            "expectedRevision": revision,
        }
        if name is not None:
            body["name"] = name
        if subject is not None:
            body["subject"] = subject
        payload = client.put(f"/api/projects/{pid}/doc", json=body)
        if payload.get("status") == 409:
            raise McpToolError(
                f"Pipeline save conflict ({payload.get('code', 'conflict')}): "
                f"expected revision {revision} is stale. Call project.load('{pid}') "
                "to fetch the latest revision, merge your changes on top, then "
                "project.save again with that revision as expected_revision."
            )
        return {
            "ok": payload.get("ok", True),
            "projectId": payload.get("projectId", pid),
            "revision": payload.get("revision"),
            "message": payload.get("message"),
        }

    @mcp.tool(name="project.load", annotations=READ_ONLY)
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

    @mcp.tool(name="project.rename", annotations=MUTATING)
    def rename_project(project_id: str, name: str) -> dict[str, Any]:
        """Rename a project you own (max 200 characters).

        Args:
            project_id: Project to rename (from project.create/project.list).
            name: New display name.
        """
        pid = safe_segment(project_id, label="project id")
        payload = client.patch(f"/api/projects/{pid}", json={"name": name})
        project = payload.get("project") or {}
        return {"projectId": project.get("id", pid), "name": project.get("name", name)}

    @mcp.tool(name="project.delete", annotations=DESTRUCTIVE)
    def delete_project(project_id: str, confirm: bool = False) -> dict[str, Any]:
        """Delete a project (and its saved pipeline document) from the studio workspace.

        Requires confirm=True: this removes the project from the user's project
        list. Executions already run from it are not affected. Find ids with
        project.list.

        Args:
            project_id: Project to delete (from project.create/project.list).
            confirm: MUST be true to delete — the explicit user go-ahead.
        """
        if not confirm:
            return {
                "deleted": False,
                "message": "Refusing to delete a project: pass confirm=true to proceed. "
                "Call project.list to check the project id first.",
            }
        pid = safe_segment(project_id, label="project id")
        payload = client.delete(f"/api/projects/{pid}")
        return {
            "deleted": bool(payload.get("ok", True)),
            "projectId": payload.get("id", pid),
            "message": payload.get("message"),
        }
