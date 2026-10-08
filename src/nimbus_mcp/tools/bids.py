"""BIDS export tools: datasets and execution results as BIDS-layout zips."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import LOCAL_WRITE
from ._guards import safe_segment


def _read_manifest(zip_path: Path) -> dict[str, Any]:
    """Pull nimbus_export_manifest.json out of the saved zip (summary fields)."""
    with zipfile.ZipFile(zip_path) as zf:
        member = next(
            (n for n in zf.namelist() if n.endswith("nimbus_export_manifest.json")), None
        )
        if member is None:
            raise McpToolError(
                f"Export zip {zip_path.name} has no nimbus_export_manifest.json member — "
                "the backend response was not a Nimbus BIDS export."
            )
        return json.loads(zf.read(member))


def _save_zip(data: bytes, safe_stem: str, export_dir: Path) -> dict[str, Any]:
    out_dir = export_dir / "bids"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{safe_stem}-bids.zip"
    suffix = 0
    while dest.exists():
        suffix += 1
        dest = out_dir / f"{safe_stem}-bids-{suffix}.zip"
    dest.write_bytes(data)
    manifest = _read_manifest(dest)
    return {
        "path": str(dest),
        "size": len(data),
        "tier": manifest.get("tier"),
        "subjects": manifest.get("subjects", []),
        "files": len(manifest.get("dataFiles", [])),
        "bytes": sum(f.get("bytes", 0) for f in manifest.get("dataFiles", [])),
    }


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="bids.export_dataset", annotations=LOCAL_WRITE)
    def export_dataset(
        dataset: str,
        subjects: list[str] | None = None,
        kind: str | None = None,
    ) -> dict[str, Any]:
        """Export a Nimbus dataset as a BIDS-layout zip saved to NIMBUS_EXPORT_DIR/bids/.

        Continuous sources (EDF/BDF uploads, stream recordings) export as BIDS-raw
        (EDF byte-passthrough; recordings converted HDF5→EDF with events.tsv).
        Epoched sources (MOABB packs, trial-table uploads) export as BIDS-derivatives
        (data passthrough + descriptive trial_index/label/session/run events — no
        fabricated onsets). The zip embeds nimbus_export_manifest.json with sha256s.

        Args:
            dataset: Dataset id — a MOABB pack id (e.g. "BNCI2014_001"), an upload
                path (as returned by data.upload), or a recording name. Auto-detected.
            subjects: Pack exports only — subject filter (e.g. ["S01", "S03"]) to
                bound zip size; default exports all installed subjects.
            kind: Optional explicit source kind: pack|upload|recording (default auto).
        """
        if not dataset:
            raise McpToolError("Invalid dataset id ''.")
        # Dataset ids may be slash-containing upload paths (user_<sha>/<file>,
        # as returned by data.upload) — the id is a JSON body value, and the
        # backend enforces containment + ownership on it, so it passes through
        # verbatim. Only the LOCAL save filename stem is sanitized.
        body: dict[str, Any] = {"source": {"kind": kind or "auto", "id": dataset}}
        if subjects:
            body["subjects"] = subjects
        data = client.post_bytes("/api/bids/export", json=body)
        safe_stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in dataset)[:80]
        return _save_zip(data, safe_stem, client.config.export_dir)

    @mcp.tool(name="bids.export_execution", annotations=LOCAL_WRITE)
    def export_execution(execution_id: str) -> dict[str, Any]:
        """Export an execution's results as a BIDS-derivative zip (metrics.json,
        per-subject participants.tsv, protocol.json, full pipeline_snapshot.json,
        predictions.tsv when stored, provenance/execution.json) saved to
        NIMBUS_EXPORT_DIR/bids/.

        Args:
            execution_id: Run whose results to export (from execution.run).
        """
        exec_id = safe_segment(execution_id)
        body: dict[str, Any] = {"source": {"kind": "execution", "id": execution_id}}
        data = client.post_bytes("/api/bids/export", json=body)
        return _save_zip(data, exec_id, client.config.export_dir)
