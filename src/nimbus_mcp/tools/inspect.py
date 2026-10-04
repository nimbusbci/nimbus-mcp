"""Inspect tools: look at EEG data BEFORE building pipelines (v0.6 data plane).

Both wrap ``GET /api/data/describe`` and pass the payload through verbatim:
channels (with flatline flags), samplingRate, durationSec, trials/classCounts,
epochWindow, per-channel µV stats, canonical band powers (dB), and a
time-budgeted PSD summary with ``computedFrom.sampleStrategy`` telling you
what the numbers were computed from.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import READ_ONLY

# Backend error code for "local paths are not trusted on this deployment"
# (api_codes.DATA_DESCRIBE_LOCAL_FORBIDDEN): flat RFC7807 body, top-level code.
_LOCAL_FORBIDDEN_CODE = "nimbus.data.describe_local_forbidden"

_HOSTED_LOCAL_MESSAGE = (
    "This Nimbus backend is hosted, so it does not open local filesystem "
    "paths: inspecting the file directly is only possible against a local "
    "backend (desktop app / MCP local mode). An uploaded file still works "
    "anywhere — upload it and pass the returned relative path back to "
    "inspect_file. Pick one of the options below and retry."
)

_HOSTED_LOCAL_OPTIONS: list[dict[str, str]] = [
    {
        "action": "upload the file first (upload_data)",
        "detail": "Call upload_data(file_path=...) and pass the relative path "
        "it returns back to inspect_file(path=...) — that works on any "
        "backend, and the registered upload path can also be used in "
        "pipelines (custom_data node).",
    },
    {
        "action": "run against a local backend",
        "detail": "Point this MCP server at a local Nimbus backend (the "
        "desktop app, or a dev backend with NIMBUS_MCP_KEY = MCP_LOCAL_KEY); "
        "local absolute paths are trusted there.",
    },
]


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(annotations=READ_ONLY)
    def inspect_dataset(
        dataset: str,
        subject: str | None = None,
        mode: str = "all",
    ) -> dict[str, Any]:
        """Exploratory summary of a public EEG dataset (MOABB pack): channels,
        sampling rate, trial/class balance, per-channel µV stats, band powers
        and a PSD overview.

        Look at the data BEFORE building pipelines: class balance drives
        stratification choices (imbalanced classes skew accuracy), and
        flatlined channels mean a montage/reference problem worth fixing
        first. subject is REQUIRED (the backend 400s without it) — get the
        subject list via list_datasets, e.g. "S01"; a comma-list like
        "S01,S03" loads a cohort. mode: training | evaluation | all.
        Units note: values are ASSUMED volts by the loader — a µV-native file
        reads 1e6x too large; set unitsScale in a pipeline's custom_data
        config when needed.

        Args:
            dataset: Dataset id from list_datasets (e.g. "BNCI2014_001").
            subject: REQUIRED subject code ("S01") or comma-list cohort ("S01,S03").
            mode: Which split to summarize — training | evaluation | all.
        """
        params: dict[str, Any] = {"source": "dataset", "dataset": dataset, "mode": mode}
        if subject is not None:
            params["subject"] = subject
        return client.get("/api/data/describe", params=params)

    @mcp.tool(annotations=READ_ONLY)
    def inspect_file(path: str) -> dict[str, Any]:
        """Exploratory summary of an EEG file (.edf/.bdf/.mat/.csv/.tsv/.txt/.h5):
        channels, sampling rate, trial/class balance, per-channel µV stats,
        band powers and a PSD overview.

        The path shape picks the source: ABSOLUTE path → read the file from
        disk (only on a LOCAL backend: desktop app / MCP local mode — no
        upload needed); RELATIVE path (the one upload_data returns) → describe
        the uploaded file, which works on ANY backend (hosted or local).

        Look at the data BEFORE building pipelines: class balance drives
        stratification choices, and flatlined channels mean a
        montage/reference problem worth fixing first. Units note: values are
        ASSUMED volts by the loader — a µV-native CSV reads 1e6x too large;
        set unitsScale in a pipeline's custom_data config when needed. On a
        hosted backend absolute paths are refused and this returns guidance
        (upload the file first or switch to a local backend).

        Args:
            path: ABSOLUTE filesystem path (local backends only) or the RELATIVE
                upload path returned by upload_data (works on any backend).
        """
        source = "local" if os.path.isabs(os.path.expanduser(path)) else "upload"
        try:
            return client.get("/api/data/describe", params={"source": source, "path": path})
        except McpToolError as err:
            if source == "local" and err.status_code == 403 and err.code == _LOCAL_FORBIDDEN_CODE:
                # Hosted backend: setup-style guidance instead of a dead end.
                return {
                    "ok": False,
                    "message": _HOSTED_LOCAL_MESSAGE,
                    "options": [dict(option) for option in _HOSTED_LOCAL_OPTIONS],
                }
            raise
