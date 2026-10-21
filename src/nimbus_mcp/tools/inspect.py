"""Inspect tools: look at EEG data BEFORE building pipelines (v0.6 data plane).

Both wrap ``GET /api/data/describe`` and pass the payload through verbatim:
channels (with flatline flags), samplingRate, durationSec, trials/classCounts,
epochWindow, per-channel µV stats, canonical band powers (dB), and a
time-budgeted PSD summary with ``computedFrom.sampleStrategy`` telling you
what the numbers were computed from.
"""

from __future__ import annotations

import math
import os
from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ..errors import local_tool_error
from ._annotations import READ_ONLY

# Backend error code for "local paths are not trusted on this deployment"
# (api_codes.DATA_DESCRIBE_LOCAL_FORBIDDEN): flat RFC7807 body, top-level code.
_LOCAL_FORBIDDEN_CODE = "nimbus.data.describe_local_forbidden"

# Plain-text formats carry no sampling rate; the backend refuses to guess.
_RATELESS_EXTENSIONS = (".csv", ".tsv", ".txt")
SAMPLING_RATE_REQUIRED_CODE = "nimbus.mcp.sampling_rate_required"
SAMPLING_RATE_INVALID_CODE = "nimbus.mcp.sampling_rate_invalid"

_HOSTED_LOCAL_MESSAGE = (
    "This Nimbus backend is hosted, so it does not open local filesystem "
    "paths: inspecting the file directly is only possible against a local "
    "backend (desktop app / MCP local mode). An uploaded file still works "
    "anywhere — upload it and pass the returned relative path back to "
    "data.inspect_file. Pick one of the options below and retry."
)

_HOSTED_LOCAL_OPTIONS: list[dict[str, str]] = [
    {
        "action": "upload the file first (data.upload)",
        "detail": "Call data.upload(file_path=...) and pass the relative path "
        "it returns back to data.inspect_file(path=...) — that works on any "
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
    @mcp.tool(name="data.inspect_dataset", annotations=READ_ONLY)
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
        subject list via catalog.datasets, e.g. "S01"; a comma-list like
        "S01,S03" loads a cohort. mode: training | evaluation | all.
        Units note: values are ASSUMED volts by the loader — a µV-native file
        reads 1e6x too large; set unitsScale in a pipeline's custom_data
        config when needed.

        Args:
            dataset: Dataset id from catalog.datasets (e.g. "BNCI2014_001").
            subject: REQUIRED subject code ("S01") or comma-list cohort ("S01,S03").
            mode: Which split to summarize — training | evaluation | all.
        """
        params: dict[str, Any] = {"source": "dataset", "dataset": dataset, "mode": mode}
        if subject is not None:
            params["subject"] = subject
        return client.get("/api/data/describe", params=params)

    @mcp.tool(name="data.inspect_file", annotations=READ_ONLY)
    def inspect_file(path: str, sampling_rate: float | None = None) -> dict[str, Any]:
        """Exploratory summary of an EEG file (.edf/.bdf/.mat/.csv/.tsv/.txt/.h5):
        channels, sampling rate, trial/class balance, per-channel µV stats,
        band powers and a PSD overview.

        The path shape picks the source: ABSOLUTE path → read the file from
        disk (only on a LOCAL backend: desktop app / MCP local mode — no
        upload needed); RELATIVE path (the one data.upload returns) → describe
        the uploaded file, which works on ANY backend (hosted or local).

        Look at the data BEFORE building pipelines: class balance drives
        stratification choices, and flatlined channels mean a
        montage/reference problem worth fixing first. Units note: values are
        ASSUMED volts by the loader — a µV-native CSV reads 1e6x too large;
        set unitsScale in a pipeline's custom_data config when needed. On a
        hosted backend absolute paths are refused and this returns guidance
        (upload the file first or switch to a local backend).

        CSV/TSV/TXT files have no embedded sampling rate: pass sampling_rate
        (Hz) for them — the backend never assumes 250 Hz. Formats that embed
        their rate (EDF/BDF/MAT/H5) ignore it.

        Args:
            path: ABSOLUTE filesystem path (`~` is expanded; local backends
                only) or the RELATIVE upload path returned by data.upload
                (works on any backend).
            sampling_rate: Hz for CSV/TSV/TXT files (REQUIRED there, e.g. 250.0).
        """
        # One expansion used for BOTH the classification and the wire: the
        # backend's local resolver demands an absolute path, and a raw "~/…"
        # input classified as local but sent unexpanded would be a 400 for a
        # file that exists (same contract as data.upload).
        resolved = os.path.expanduser(path)
        source = "local" if os.path.isabs(resolved) else "upload"
        params: dict[str, Any] = {"source": source, "path": resolved}
        if sampling_rate is not None:
            if not math.isfinite(sampling_rate) or sampling_rate <= 0:
                raise local_tool_error(
                    f"Invalid sampling_rate '{sampling_rate}': provide a positive "
                    "number of Hz (e.g. 250.0).",
                    code=SAMPLING_RATE_INVALID_CODE,
                    status_code=400,
                )
            params["samplingRate"] = sampling_rate
        elif resolved.strip().lower().endswith(_RATELESS_EXTENSIONS):
            raise local_tool_error(
                "sampling_rate is required for CSV/TSV/TXT files: they carry no "
                "sampling rate and Nimbus never assumes one. Retry "
                "data.inspect_file with sampling_rate=<Hz> (e.g. 250.0).",
                code=SAMPLING_RATE_REQUIRED_CODE,
                status_code=400,
            )
        try:
            return client.get("/api/data/describe", params=params)
        except McpToolError as err:
            if source == "local" and err.status_code == 403 and err.code == _LOCAL_FORBIDDEN_CODE:
                # Hosted backend: setup-style guidance instead of a dead end.
                return {
                    "ok": False,
                    "message": _HOSTED_LOCAL_MESSAGE,
                    "options": [dict(option) for option in _HOSTED_LOCAL_OPTIONS],
                }
            raise
