"""Leaderboard tools: public benchmark rankings + BYO submissions."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import McpToolError, NimbusClient
from ._annotations import MUTATING, READ_ONLY

_TRACK_KEYS = ("crossSubject", "pretrainedFewTrial")


def _known_dataset_ids(payload: dict[str, Any]) -> list[str]:
    seen: list[str] = []

    def add(rows: object) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            dataset_id = row.get("datasetId")
            if dataset_id and str(dataset_id) not in seen:
                seen.append(str(dataset_id))

    add(payload.get("datasets"))
    for key in _TRACK_KEYS:
        block = payload.get(key)
        if isinstance(block, dict):
            add(block.get("datasets"))
    return seen


def _filter_leaderboard(payload: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    """Keep one dataset's rows on every track. Unknown ids name the board."""
    known = _known_dataset_ids(payload)
    matched = False

    def keep(rows: object) -> list[Any]:
        nonlocal matched
        if not isinstance(rows, list):
            return []
        kept = [
            row
            for row in rows
            if isinstance(row, dict) and str(row.get("datasetId")) == dataset_id
        ]
        if kept:
            matched = True
        return kept

    out = dict(payload)
    if isinstance(out.get("datasets"), list):
        out["datasets"] = keep(out["datasets"])
    for key in _TRACK_KEYS:
        block = out.get(key)
        if isinstance(block, dict):
            block = dict(block)
            if isinstance(block.get("datasets"), list):
                block["datasets"] = keep(block["datasets"])
            out[key] = block
    if matched:
        return out
    close = next((name for name in known if name.lower() == dataset_id.lower()), None)
    hint = f" Did you mean {close}?" if close else ""
    listed = ", ".join(known) if known else "(none on this board)"
    raise McpToolError(
        f"No leaderboard rows for dataset {dataset_id!r}.{hint} Known dataset ids: {listed}."
    )


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="catalog.leaderboard", annotations=READ_ONLY)
    def get_leaderboard(dataset_id: str | None = None) -> dict[str, Any]:
        """Public benchmark leaderboard: pipeline rankings per dataset.

        Two tracks: the top-level ``within_session`` block (curated suite
        pipelines) and ``crossSubject`` (leave-one-subject-out — curated
        pipelines plus BYO ``python_model`` submissions; BYO rows carry
        ``kind: "byo"`` and an ``owner`` name). Within each dataset, ``rows``
        sort desc by ``meanAccuracyPct`` (95% CI in ``ciLoPct``/``ciHiPct``).
        ``packFingerprint`` identifies the exact dataset pack behind the
        scores; ``updated`` marks each dataset's most recent run. The
        ``pretrainedFewTrial`` efficiency track means: accuracy at a
        protocol-fixed 100-trials-per-subject budget — the frozen-backbone /
        few-parameter rows that matter when calibration data is scarce
        (``paramsTrained`` shown per row so cost is readable next to
        accuracy). ``submittableDatasets`` is the set open for
        ``leaderboard.submit``; a dataset can appear under ``crossSubject``
        and still be closed for new submissions.

        Args:
            dataset_id: Optional dataset id. When set, every track is filtered
                to that dataset. Omit it for the full board.
        """
        payload = client.get("/api/leaderboard")
        if not isinstance(payload, dict):
            return payload
        wanted = (dataset_id or "").strip()
        if not wanted:
            return payload
        return _filter_leaderboard(payload, wanted)

    @mcp.tool(name="leaderboard.submit", annotations=MUTATING)
    def submit_leaderboard_model(
        dataset_id: str,
        name: str,
        code: str | None = None,
        class_name: str | None = None,
        requirements: list[str] | None = None,
        track: str = "cross_subject",
        backbone_kind: str | None = None,
        backbone_repo: str | None = None,
        backbone_revision: str | None = None,
        backbone_license: str | None = None,
    ) -> dict[str, Any]:
        """Submit a BYO python_model to a leaderboard track for server-side scoring.

        Nimbus scores it server-side under the track's fixed protocol — you
        submit code, never results. Poll ``leaderboard.status`` with the
        returned ``submissionId``; scoring takes minutes (classical) to tens
        of minutes (deep). Requirements must be plain PyPI names/specs (max
        20, no URLs). Identical code resubmission is idempotent per track
        (``deduplicated: true``).

        Args:
            dataset_id: A dataset id from catalog.leaderboard
                ``submittableDatasets`` (e.g. "BNCI2014_001"). A dataset can
                appear under crossSubject and still be closed for submission
                (BNCI2014_008 is scored there and is not in the open set).
            name: Display name on the board (<=100 chars).
            code: Python source defining a class with fit(X, y, info) and
                predict(X) (<=256KB; validate first with python.validate).
                Required unless backbone_repo is set.
            class_name: Contract class to instantiate; omit to auto-detect.
            requirements: Optional PyPI requirements (e.g. ["sklearn>=1.4"]).
            track: "cross_subject" (default — full-data leave-one-subject-out,
                seed 42) or "pretrained_few_trial" (efficiency track: LOSO with
                a protocol-fixed 100-trials-per-subject cap; code that imports
                torch must declare its trained parameter count as a class
                attribute — PARAMS_TRAINED = <int> — which becomes the row's
                paramsTrained and is cross-checked when computable).
            backbone_kind: Checkpoint family for the BYO backbone lane — one
                of the zoo's supported kinds (e.g. "cbramod", "labram").
            backbone_repo: HF 'org/name' repo id for the BYO backbone lane.
            backbone_revision: Full 40-char commit SHA pinning the backbone.
            backbone_license: License declared for the backbone checkpoint.
            backbone_kind / backbone_repo / backbone_revision / backbone_license:
                BYO pretrained-checkpoint lane (efficiency track only, track
                must be "pretrained_few_trial"): declare an HF 'org/name' repo
                pinned to a full 40-char commit SHA with a license declaration;
                Nimbus loads it through the same pinned adapter as its zoo rows
                and scores frozen-embeddings → linear probe server-side. When
                backbone_repo is set, omit code.
        """
        has_code = bool(code and code.strip())
        has_backbone = bool(backbone_repo and str(backbone_repo).strip())
        if not has_code and not has_backbone:
            raise McpToolError(
                "leaderboard.submit needs model code, or a backbone declaration "
                "(backbone_repo, backbone_kind, backbone_revision, and "
                "backbone_license on the pretrained_few_trial track). "
                "dataset_id and name alone are not a submission. Only datasets "
                "listed in catalog.leaderboard submittableDatasets can be "
                "submitted; a dataset can be scored on crossSubject and still "
                "be closed for new submissions."
            )
        body: dict[str, Any] = {
            "datasetId": dataset_id,
            "name": name,
            "track": track,
        }
        if code is not None:
            body["code"] = code
        if class_name:
            body["className"] = class_name
        if requirements:
            body["requirements"] = requirements
        if backbone_repo is not None:
            body["backbone"] = {
                "kind": backbone_kind,
                "repo": backbone_repo,
                "revision": backbone_revision,
                "license": backbone_license,
            }
        return client.post("/api/leaderboard/submissions", json=body)

    @mcp.tool(name="leaderboard.status", annotations=READ_ONLY)
    def get_leaderboard_submission(submission_id: str) -> dict[str, Any]:
        """Fetch one of your leaderboard submissions (owner-only).

        Returns status (queued | scoring | done | failed), metrics when done
        (meanAccuracyPct + CI, same shape as leaderboard rows), or the error
        that failed it.

        Args:
            submission_id: The submissionId returned by leaderboard.submit.
        """
        return client.get(f"/api/leaderboard/submissions/{submission_id}")

    @mcp.tool(name="leaderboard.mine", annotations=READ_ONLY)
    def list_my_leaderboard_submissions() -> dict[str, Any]:
        """List your leaderboard submissions (newest first), all datasets."""
        return client.get("/api/leaderboard/submissions")
