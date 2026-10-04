"""Artifact + export tools: list/download trained artifacts, export standalone Python."""

from __future__ import annotations

import time
from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._annotations import READ_ONLY
from ._guards import safe_segment


def _safe_name(name: str) -> str:
    # Same rejection rules as a path segment, labelled for artifact names.
    return safe_segment(name, label="artifact name")


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(annotations=READ_ONLY)
    def list_artifacts(execution_id: str) -> dict[str, Any]:
        """Trained artifacts (models/filters, e.g. *.pkl) saved by an execution.

        Args:
            execution_id: Run whose artifacts to list.
        """
        exec_id = safe_segment(execution_id)
        payload = client.get(f"/api/executions/{exec_id}/artifacts")
        return {
            "executionId": execution_id,
            "artifacts": [
                {"name": a.get("name"), "size": a.get("size"), "subject": a.get("subject")}
                for a in payload.get("artifacts", [])
            ],
        }

    @mcp.tool(annotations=READ_ONLY)
    def download_artifact(execution_id: str, artifact_name: str) -> dict[str, Any]:
        """Download one artifact file to NIMBUS_EXPORT_DIR/executions/<id>/ and return its path.

        Args:
            execution_id: Run that produced the artifact.
            artifact_name: File name from list_artifacts (e.g. "nimbus_lda.pkl").
        """
        exec_id = safe_segment(execution_id)
        artifact = _safe_name(artifact_name)
        data = client.get_bytes(f"/api/executions/{exec_id}/artifacts/{artifact}")
        dest_dir = client.config.export_dir / "executions" / exec_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / artifact
        dest.write_bytes(data)
        return {"path": str(dest), "size": len(data)}

    @mcp.tool(annotations=READ_ONLY)
    def export_python(train_graph: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        """Export the pipeline as a standalone runnable Python bundle (zip saved locally).

        Args:
            train_graph: Pipeline graph {nodes, connections} to export.
            name: Optional name recorded inside the bundle.
        """
        body: dict[str, Any] = {"train": train_graph}
        if name is not None:
            body["name"] = name
        data = client.post_bytes("/api/export/python", json=body)
        out_dir = client.config.export_dir / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = out_dir / f"{stamp}-nimbus-export.zip"
        suffix = 0
        while dest.exists():
            suffix += 1
            dest = out_dir / f"{stamp}-nimbus-export-{suffix}.zip"
        dest.write_bytes(data)
        return {"path": str(dest), "size": len(data)}
