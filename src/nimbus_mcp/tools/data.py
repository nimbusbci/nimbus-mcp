"""Data tools: bring your own EEG files into Nimbus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ..errors import local_tool_error
from ._annotations import DESTRUCTIVE, MUTATING
from ._guards import (
    SAMPLING_RATE_REQUIRED_CODE,
    confirm_refusal,
    safe_upload_path,
    validate_sampling_rate,
)
from ._hosted_files import refuse_local_path_on_gateway

_USAGE = (
    "Use this path as a custom_data node's filePath in execution.run/pipeline.validate "
    '(config: {"filePath": "<path>", "format": "<format>", ...}).'
)
_HEADERLESS_EXTS = {".csv", ".tsv", ".txt"}
#: data.upload was given a local path that is not an existing file (E2E r9 N4).
FILE_NOT_FOUND = "nimbus.mcp.file_not_found"
_GATEWAY_UPLOAD_HINT = (
    "Upload the file from Nimbus Studio (web app), or run nimbus-mcp locally "
    "(stdio / desktop app), where data.upload reads files from your own disk."
)


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="data.upload", title="Upload Data File", annotations=MUTATING)
    def upload_data(
        file_path: str,
        dataset_name: str | None = None,
        sampling_rate: float | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        """Upload an EEG file (.edf/.bdf/.mat/.csv/.txt/.tsv/.h5/.hdf5, <=500MB) to the
        backend and get the registered path for a custom_data node.

        sampling_rate (Hz, e.g. 250.0) is REQUIRED for plain CSV/TSV/TXT files
        without embedded metadata — uploads without it are rejected (no silent
        250 Hz assumption). format overrides extension-based detection
        (auto, mat, csv, tsv, txt, edf, bdf, h5, hdf5).

        Args:
            file_path: Local file to upload (.edf/.bdf/.mat/.csv/.txt/.tsv/.h5/.hdf5, <=500MB).
            dataset_name: Optional label for the uploaded dataset.
            sampling_rate: Hz for headerless CSV/TSV/TXT (REQUIRED there, e.g. 250.0).
            format: Override extension-based detection (auto|mat|csv|tsv|txt|edf|bdf|h5|hdf5).
        """
        # Hosted gateway: never resolve the caller's path on the server's own
        # disk (file-existence oracle / server file exfiltration, E2E r9 N1).
        refuse_local_path_on_gateway(client, tool="data.upload", hint=_GATEWAY_UPLOAD_HINT)
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise local_tool_error(
                f"File not found: {file_path}", code=FILE_NOT_FOUND, status_code=404
            )
        fields: dict[str, str] = {}
        if dataset_name:
            fields["dataset_name"] = dataset_name
        # Backend (uploads.py) validates this dict against the custom_data node
        # schema, which rejects unknown keys — so the camelCase schema names
        # samplingRate/format are mandatory, not snake_case.
        config: dict[str, Any] = {}
        if sampling_rate is not None:
            validate_sampling_rate(sampling_rate)
            config["samplingRate"] = sampling_rate
        if format is not None:
            config["format"] = format
        fmt = (format or path.suffix.lstrip(".")).lower() or path.suffix.lower()
        if fmt in {"csv", "tsv", "txt"} or path.suffix.lower() in _HEADERLESS_EXTS:
            if sampling_rate is None:
                raise local_tool_error(
                    "sampling_rate is required for CSV/TSV/TXT uploads (Hz, e.g. 250.0). "
                    "The backend no longer assumes 250 Hz.",
                    code=SAMPLING_RATE_REQUIRED_CODE,
                    status_code=400,
                )
        if config:
            fields["config"] = json.dumps(config)
        payload = client.post_file("/api/upload", path, fields)
        return {
            "ok": payload.get("ok", True),
            "file": payload.get("file"),
            "metadata": payload.get("metadata"),
            "usage": _USAGE,
        }

    @mcp.tool(name="data.delete_upload", title="Delete Upload", annotations=DESTRUCTIVE)
    def delete_upload(path: str, confirm: bool = False) -> dict[str, Any]:
        """Delete an uploaded file from your account (disk + registry).

        Requires confirm=True. Pass the path returned by data.upload
        (file.path). Only your own uploads can be deleted; missing paths
        return not-found.

        Args:
            path: Upload path from data.upload (e.g. user_…/file.csv).
            confirm: MUST be true to delete — the explicit user go-ahead.
        """
        if not confirm:
            return confirm_refusal(
                "deleted",
                "Refusing to delete an upload: pass confirm=true to proceed. "
                "Use the path returned by data.upload.",
            )
        upload_path = safe_upload_path(path)
        payload = client.delete("/api/uploads", params={"path": upload_path})
        return {
            "deleted": bool(payload.get("ok", True)),
            "path": payload.get("path", upload_path),
            "message": payload.get("message"),
        }
