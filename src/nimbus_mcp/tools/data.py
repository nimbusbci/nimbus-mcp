"""Data tools: bring your own EEG files into Nimbus."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import MUTATING

_USAGE = (
    "Use this path as a custom_data node's filePath in run_pipeline/validate_pipeline "
    "(config: {\"filePath\": \"<path>\", \"format\": \"<format>\", ...})."
)


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(annotations=MUTATING)
    def upload_data(
        file_path: str,
        dataset_name: str | None = None,
        sampling_rate: float | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        """Upload an EEG file (.edf/.bdf/.mat/.csv/.txt/.tsv/.h5/.hdf5, <=500MB) to the
        backend and get the registered path for a custom_data node.

        sampling_rate (Hz, e.g. 250.0) is REQUIRED for plain CSV/TSV/TXT files
        without embedded metadata — the backend silently assumes 250 Hz otherwise,
        which mis-times epochs, filters and spectral features. format overrides
        extension-based detection (auto, mat, csv, tsv, txt, edf, bdf, h5, hdf5).

        Args:
            file_path: Local file to upload (.edf/.bdf/.mat/.csv/.txt/.tsv/.h5/.hdf5, <=500MB).
            dataset_name: Optional label for the uploaded dataset.
            sampling_rate: Hz for headerless CSV/TSV/TXT (REQUIRED there, e.g. 250.0).
            format: Override extension-based detection (auto|mat|csv|tsv|txt|edf|bdf|h5|hdf5).
        """
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise McpToolError(f"File not found: {file_path}")
        fields: dict[str, str] = {}
        if dataset_name:
            fields["dataset_name"] = dataset_name
        # Backend (uploads.py) validates this dict against the custom_data node
        # schema, which rejects unknown keys — so the camelCase schema names
        # samplingRate/format are mandatory, not snake_case.
        config: dict[str, Any] = {}
        if sampling_rate is not None:
            if not math.isfinite(sampling_rate) or sampling_rate <= 0:
                raise McpToolError(
                    f"Invalid sampling_rate '{sampling_rate}': provide a positive "
                    "frequency in Hz (e.g. 250.0)."
                )
            config["samplingRate"] = sampling_rate
        if format is not None:
            config["format"] = format
        if config:
            fields["config"] = json.dumps(config)
        payload = client.post_file("/api/upload", path, fields)
        return {
            "ok": payload.get("ok", True),
            "file": payload.get("file"),
            "metadata": payload.get("metadata"),
            "usage": _USAGE,
        }
