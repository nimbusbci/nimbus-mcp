"""Artifact + export tools: list/download trained artifacts, export standalone Python."""

from __future__ import annotations

import io
import time
import zipfile
from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._annotations import LOCAL_WRITE, READ_ONLY
from ._guards import safe_segment
from ._hosted_files import inline_file
from ._hosted_files import is_hosted_gateway as _is_hosted_gateway


def _safe_name(name: str) -> str:
    # Same rejection rules as a path segment, labelled for artifact names.
    return safe_segment(name, label="artifact name")


# Inline export cap on the hosted gateway (base64 grows ~4/3): beyond this the
# zip is not returned inline and the user is pointed at Studio's Export.
# Cap on the *encoded* base64 text returned inline (~768 KB of zip). Tool results
# go into the model's context; multi-MB blobs blow client response/context limits.
EXPORT_NOTES_FILE = "EXPORT_NOTES.md"  # written by the backend generator


def _export_notes(data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if EXPORT_NOTES_FILE in zf.namelist():
                return zf.read(EXPORT_NOTES_FILE).decode("utf-8", "replace").strip()
    except zipfile.BadZipFile:
        return None
    return None


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="execution.artifacts", annotations=READ_ONLY)
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

    @mcp.tool(name="execution.download_artifact", annotations=LOCAL_WRITE)
    def download_artifact(execution_id: str, artifact_name: str) -> dict[str, Any]:
        """Download one artifact file.

        Local/stdio: saved to NIMBUS_EXPORT_DIR/executions/<id>/ and its path returned.
        Hosted gateway: returned inline as ``contentBase64`` (up to 1 MB of base64);
        larger files point to Nimbus Studio. Nothing is written on the gateway.

        Args:
            execution_id: Run that produced the artifact.
            artifact_name: File name from execution.artifacts (e.g. "nimbus_lda.pkl").
        """
        exec_id = safe_segment(execution_id)
        artifact = _safe_name(artifact_name)
        data = client.get_bytes(f"/api/executions/{exec_id}/artifacts/{artifact}")
        if _is_hosted_gateway(client):
            return inline_file(
                data, artifact, "download it from Nimbus Studio (run results -> Artifacts)."
            )
        dest_dir = client.config.export_dir / "executions" / exec_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / artifact
        dest.write_bytes(data)
        return {"path": str(dest), "size": len(data)}

    @mcp.tool(name="pipeline.export", annotations=LOCAL_WRITE)
    def export_python(train_graph: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        """Export the pipeline as a standalone runnable Python bundle (zip).

        Local/stdio: the zip is saved under NIMBUS_EXPORT_DIR/exports and its path
        returned. Hosted gateway: the zip is returned inline as ``contentBase64``
        (up to 1 MB of base64). UI-only nodes (interactive_application) are left out and
        listed in ``exportNotes``.

        Args:
            train_graph: Pipeline graph {nodes, connections} to export.
            name: Optional name recorded inside the bundle.
        """
        body: dict[str, Any] = {"train": train_graph}
        if name is not None:
            body["name"] = name
        data = client.post_bytes("/api/export/python", json=body)
        notes = _export_notes(data)
        if _is_hosted_gateway(client):
            # The gateway's disk is not the caller's: a path there is useless.
            out = inline_file(
                data, "nimbus-export.zip", "export it from Nimbus Studio (Pipeline -> Export)."
            )
            if notes:
                out["exportNotes"] = notes
            return out
        out_dir = client.config.export_dir / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = out_dir / f"{stamp}-nimbus-export.zip"
        suffix = 0
        while dest.exists():
            suffix += 1
            dest = out_dir / f"{stamp}-nimbus-export-{suffix}.zip"
        dest.write_bytes(data)
        result: dict[str, Any] = {"path": str(dest), "size": len(data)}
        if notes:
            result["exportNotes"] = notes
        return result
