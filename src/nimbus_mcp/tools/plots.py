"""Plot tools: agent-visible PNG figures (Nimbus MCP v0.12).

Every tool renders a figure server-side and returns it as an MCP image
content block the assistant can actually LOOK at — plus a JSON sidecar with
the numbers behind the figure. Optional ``save`` writes the PNG into
NIMBUS_EXPORT_DIR/plots/ (LOCAL_WRITE) for use in reports.

This is the answer to "mne-mcp has figures, nimbus has JSON": figures ride
Nimbus data (persisted execution results, the same describe loaders, the
leaderboard), not raw MNE file plumbing.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from ..client import McpToolError, NimbusClient
from ._annotations import LOCAL_WRITE
from ._guards import safe_segment

_PLOT_KINDS = ("psd", "topomap", "bandpower", "spectrogram")
_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
_TRACKS = ("within_session", "crossSubject")


def _decode_png(payload: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """(png_bytes, meta) off the backend's PlotImagePayload."""
    image_b64 = payload.get("image")
    if not isinstance(image_b64, str) or not image_b64:
        raise McpToolError("Backend returned no figure image (unexpected payload).")
    try:
        return base64.b64decode(image_b64, validate=True), dict(payload.get("meta") or {})
    except (ValueError, TypeError) as e:
        raise McpToolError(f"Backend figure payload was not valid base64 PNG: {e}") from e


def _save_png(data: bytes, safe_stem: str, export_dir: Path) -> dict[str, Any]:
    out_dir = export_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{safe_stem}.png"
    suffix = 0
    while dest.exists():
        suffix += 1
        dest = out_dir / f"{safe_stem}-{suffix}.png"
    dest.write_bytes(data)
    return {"savedPath": str(dest), "sizeBytes": len(data)}


def _safe_stem(*parts: str) -> str:
    raw = "-".join(p for p in parts if p)
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)[:80]


def register(mcp: FastMCP, client: NimbusClient) -> None:
    def _figure_response(
        payload: dict[str, Any], stem: str, save: bool
    ) -> list[Any]:
        png, meta = _decode_png(payload)
        info: dict[str, Any] = {
            "ok": True,
            "title": payload.get("title"),
            "mimeType": payload.get("mimeType", "image/png"),
            "meta": meta,
        }
        if save:
            info.update(_save_png(png, stem, client.config.export_dir))
        return [Image(data=png, format="png"), info]

    @mcp.tool(name="plots.confusion", annotations=LOCAL_WRITE)
    def confusion(
        execution_id: str,
        matrix: str = "eval",
        save: bool = False,
    ) -> list[Any]:
        """Confusion-matrix figure for one execution: heatmap with trial
        counts + per-class accuracy bars, rendered as a PNG the assistant can
        see. Use it right after execution.results to eyeball WHICH classes are
        confused, not just the accuracy number.

        Args:
            execution_id: Run to plot (from execution.run / execution.results).
            matrix: "eval" (held-out, default) or "train" (fit-set sanity check).
            save: Also write the PNG to NIMBUS_EXPORT_DIR/plots/ and return its path.
        """
        exec_id = safe_segment(execution_id)
        if matrix not in ("eval", "train"):
            raise McpToolError(f"matrix must be 'eval' or 'train'; got {matrix!r}.")
        payload = client.post(
            "/api/plots/confusion", json={"executionId": exec_id, "matrix": matrix}
        )
        return _figure_response(payload, f"{exec_id}-{matrix}-confusion", save)

    @mcp.tool(name="plots.dataset", annotations=LOCAL_WRITE)
    def dataset(
        kind: str,
        source: str,
        dataset: str | None = None,
        subject: str | None = None,
        mode: str = "all",
        path: str | None = None,
        band: str = "alpha",
        channel_idx: int = 0,
        save: bool = False,
    ) -> list[Any]:
        """Figure over an EEG data source — the visual companion to
        data.inspect_dataset. kind=psd (Welch, channel mean bold),
        kind=bandpower (channel × band dB heatmap), kind=topomap (band-power
        scalp map), kind=spectrogram (time-frequency for one channel). Sources
        are the same ones inspect uses: MOABB packs, uploads, local files.

        Args:
            kind: Figure type: psd | topomap | bandpower | spectrogram.
            source: "dataset" (MOABB pack), "upload" (path from data.upload), or
                "local" (absolute path, local backend only).
            dataset: MOABB dataset id, e.g. "BNCI2014_001" (source=dataset).
            subject: Subject id, e.g. "S01" (source=dataset).
            mode: Pack split to load: training | evaluation | all (source=dataset).
            path: Upload path (source=upload) or absolute local path (source=local).
            band: Band for kind=topomap: delta|theta|alpha|beta|gamma (default alpha).
            channel_idx: 0-based channel for kind=spectrogram (default 0).
            save: Also write the PNG to NIMBUS_EXPORT_DIR/plots/ and return its path.
        """
        if kind not in _PLOT_KINDS:
            raise McpToolError(f"kind must be one of {_PLOT_KINDS}; got {kind!r}.")
        if band not in _BANDS:
            raise McpToolError(f"band must be one of {_BANDS}; got {band!r}.")
        stem = _safe_stem("dataset", dataset or path or source, subject, kind)
        payload = client.post(
            "/api/plots/dataset",
            json={
                "kind": kind,
                "source": source,
                "dataset": dataset,
                "subject": subject,
                "mode": mode,
                "path": path,
                "band": band,
                "channelIdx": channel_idx,
            },
        )
        return _figure_response(payload, stem, save)

    @mcp.tool(name="plots.erp", annotations=LOCAL_WRITE)
    def erp(
        source: str,
        dataset: str | None = None,
        subject: str | None = None,
        mode: str = "all",
        path: str | None = None,
        max_channels: int = 16,
        save: bool = False,
    ) -> list[Any]:
        """ERP butterfly figure from epoched data: one subplot per class, all
        channel means (thin) + across-channel mean (bold), time-locked to the
        trial event. Shows whether class-conditioned evoked structure exists
        BEFORE training anything.

        Args:
            source: "dataset" (MOABB pack), "upload" (path from data.upload), or
                "local" (absolute path, local backend only).
            dataset: MOABB dataset id, e.g. "BNCI2014_001" (source=dataset).
            subject: Subject id, e.g. "S01" (source=dataset).
            mode: Pack split to load: training | evaluation | all (source=dataset).
            path: Upload path (source=upload) or absolute local path (source=local).
            max_channels: Channels drawn per class subplot (default 16, max 64).
            save: Also write the PNG to NIMBUS_EXPORT_DIR/plots/ and return its path.
        """
        stem = _safe_stem("erp", dataset or path or source, subject)
        payload = client.post(
            "/api/plots/erp",
            json={
                "source": source,
                "dataset": dataset,
                "subject": subject,
                "mode": mode,
                "path": path,
                "maxChannels": max_channels,
            },
        )
        return _figure_response(payload, stem, save)

    @mcp.tool(name="plots.leaderboard", annotations=LOCAL_WRITE)
    def leaderboard(
        dataset_id: str,
        track: str = "within_session",
        save: bool = False,
    ) -> list[Any]:
        """Leaderboard bar chart for one dataset: accuracy with 95% CI
        whiskers, best first; BYO python_model submissions highlighted in a
        distinct color. The visual companion to catalog.leaderboard.

        Args:
            dataset_id: Dataset to chart, e.g. "BNCI2014_001" (see catalog.leaderboard).
            track: "within_session" (curated 5-fold) or "crossSubject" (LOSO,
                curated + BYO submissions).
            save: Also write the PNG to NIMBUS_EXPORT_DIR/plots/ and return its path.
        """
        ds_id = safe_segment(dataset_id, label="dataset id")
        if track not in _TRACKS:
            raise McpToolError(f"track must be one of {_TRACKS}; got {track!r}.")
        payload = client.post(
            "/api/plots/leaderboard", json={"datasetId": ds_id, "track": track}
        )
        return _figure_response(payload, f"leaderboard-{ds_id}-{track}", save)
