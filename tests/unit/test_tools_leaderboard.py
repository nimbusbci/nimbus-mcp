from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import leaderboard

LEADERBOARD = {
    "ok": True,
    "generatedAt": "2026-09-01T12:00:00Z",
    "protocol": {"method": "within_session", "nFolds": 5, "stratify": True, "randomSeed": 42},
    "datasets": [
        {
            "datasetId": "BNCI2014_001",
            "paradigm": "MI",
            "packFingerprint": "sha256:abc123",
            "updated": "2026-09-01T12:00:00Z",
            "rows": [
                {
                    "pipelineId": "csp_lda",
                    "name": "CSP + LDA",
                    "meanAccuracyPct": 75.1,
                    "ciLoPct": 70.2,
                    "ciHiPct": 80.0,
                    "kappa": 0.61,
                    "nSubjects": 9,
                    "executedAt": "2026-09-01T12:00:00Z",
                },
                {
                    "pipelineId": "mi_eegnet",
                    "name": "EEGNet",
                    "meanAccuracyPct": 71.3,
                    "ciLoPct": 66.0,
                    "ciHiPct": 76.4,
                    "kappa": 0.55,
                    "nSubjects": 9,
                    "executedAt": "2026-08-30T09:00:00Z",
                },
            ],
        },
        {
            "datasetId": "BNCI2015_004",
            "paradigm": "P300",
            "packFingerprint": "sha256:def456",
            "updated": "2026-08-28T08:00:00Z",
            "rows": [
                {
                    "pipelineId": "p300_lda",
                    "name": "xDAW + LDA",
                    "meanAccuracyPct": 92.0,
                    "ciLoPct": 90.0,
                    "ciHiPct": 94.0,
                    "kappa": 0.84,
                    "nSubjects": 10,
                    "executedAt": "2026-08-28T08:00:00Z",
                }
            ],
        },
    ],
}


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    leaderboard.register(server, client)
    return server


async def test_get_leaderboard_passes_payload_through_in_order():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["auth"] = req.headers.get("x-mcp-key")
        return httpx.Response(200, json=LEADERBOARD)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("catalog.leaderboard", {})

    assert seen["method"] == "GET"
    assert seen["path"] == "/api/leaderboard"
    assert seen["auth"] == "k"  # same client/auth headers, no public-no-auth mode
    data = result.data
    # Payload passes through as-is, ranking order preserved (desc meanAccuracyPct).
    assert data == LEADERBOARD
    rows = data["datasets"][0]["rows"]
    assert [r["pipelineId"] for r in rows] == ["csp_lda", "mi_eegnet"]
    assert [d["datasetId"] for d in data["datasets"]] == ["BNCI2014_001", "BNCI2015_004"]
    assert data["protocol"]["method"] == "within_session"


async def test_submit_posts_body_and_returns_submission_id():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["method"] = req.method
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.read())
        return httpx.Response(
            202,
            json={"ok": True, "submissionId": "sub-1", "status": "queued", "deduplicated": False},
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "My CSP", "code": "class M:\n ..."},
        )
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/leaderboard/submissions"
    assert seen["body"]["datasetId"] == "BNCI2014_001"
    assert result.data["submissionId"] == "sub-1"


async def test_submit_maps_optional_args_to_camel_case():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(req.read())
        return httpx.Response(
            202,
            json={"ok": True, "submissionId": "sub-2", "status": "queued", "deduplicated": False},
        )

    async with Client(make_server(handler)) as c:
        await c.call_tool(
            "leaderboard.submit",
            {
                "dataset_id": "BNCI2014_001",
                "name": "My CSP",
                "code": "class M:\n ...",
                "class_name": "M",
                "requirements": ["sklearn>=1.4"],
            },
        )
    assert seen["body"]["className"] == "M"
    assert seen["body"]["requirements"] == ["sklearn>=1.4"]


async def test_submit_forwards_track_and_defaults_to_cross_subject():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(req.read()))
        return httpx.Response(
            202,
            json={"ok": True, "submissionId": "sub-t", "status": "queued", "deduplicated": False},
        )

    async with Client(make_server(handler)) as c:
        await c.call_tool("leaderboard.submit", {"dataset_id": "D", "name": "n", "code": "c"})
        await c.call_tool(
            "leaderboard.submit",
            {
                "dataset_id": "D",
                "name": "n",
                "code": "c",
                "track": "pretrained_few_trial",
            },
        )
    # Default preserves today's cross-subject LOSO call; the track rides the
    # body verbatim for the efficiency track.
    assert seen[0]["track"] == "cross_subject"
    assert seen[1]["track"] == "pretrained_few_trial"


async def test_submit_backbone_declaration_posts_backbone_without_code():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(req.read())
        return httpx.Response(
            202,
            json={"ok": True, "submissionId": "sub-bb", "status": "queued", "deduplicated": False},
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "leaderboard.submit",
            {
                "dataset_id": "BNCI2014_001",
                "name": "My CBRAMOD",
                "backbone_kind": "cbramod",
                "backbone_repo": "u/c",
                "backbone_revision": "a" * 40,
                "backbone_license": "MIT",
                "track": "pretrained_few_trial",
            },
        )
    # BYO backbone lane: the declaration rides a `backbone` object and no
    # code key is sent (the REST route treats code and backbone as mutually
    # exclusive; an absent code must not become an empty string).
    assert "code" not in seen["body"]
    assert "className" not in seen["body"]
    assert "requirements" not in seen["body"]
    assert seen["body"]["backbone"] == {
        "kind": "cbramod",
        "repo": "u/c",
        "revision": "a" * 40,
        "license": "MIT",
    }
    assert seen["body"]["track"] == "pretrained_few_trial"
    assert seen["body"]["datasetId"] == "BNCI2014_001"
    assert seen["body"]["name"] == "My CBRAMOD"
    assert result.data["submissionId"] == "sub-bb"


async def test_submit_code_only_body_has_no_backbone_key():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(req.read())
        return httpx.Response(
            202,
            json={"ok": True, "submissionId": "sub-c", "status": "queued", "deduplicated": False},
        )

    async with Client(make_server(handler)) as c:
        await c.call_tool(
            "leaderboard.submit",
            {"dataset_id": "BNCI2014_001", "name": "My CSP", "code": "class M:\n ..."},
        )
    # Code-only submissions keep today's body exactly — no backbone key is
    # injected when the declaration params are all omitted.
    assert "backbone" not in seen["body"]
    assert seen["body"]["code"] == "class M:\n ..."
    assert seen["body"]["track"] == "cross_subject"


async def test_status_gets_one_submission():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        return httpx.Response(
            200,
            json={
                # Real route shape: the FLATTENED LeaderboardSubmissionView
                # (no ok/submission envelope — that wrapping belongs to the
                # POST and list endpoints).
                "submissionId": "sub-1",
                "datasetId": "BNCI2014_001",
                "displayName": "My CSP",
                "status": "done",
                "error": None,
                "metrics": {"meanAccuracyPct": 71.2, "ciLoPct": 66.0, "ciHiPct": 76.0},
                "deduplicated": False,
                "className": None,
                "requirements": [],
                "codeHash": "ab" * 32,
                "submittedAt": "2026-10-09T10:00:00",
                "scoringFinishedAt": "2026-10-09T10:05:00",
            },
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("leaderboard.status", {"submission_id": "sub-1"})

    assert seen["method"] == "GET"
    assert seen["path"] == "/api/leaderboard/submissions/sub-1"
    # Payload passes through as-is — no client-side result computation.
    assert result.data["status"] == "done"
    assert result.data["submissionId"] == "sub-1"
    assert result.data["displayName"] == "My CSP"
    assert result.data["metrics"]["meanAccuracyPct"] == 71.2


async def test_mine_lists_submissions():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        return httpx.Response(
            200,
            json={
                "ok": True,
                "submissions": [
                    {
                        "submissionId": "sub-2",
                        "datasetId": "BNCI2014_001",
                        "status": "scoring",
                    },
                    {
                        "submissionId": "sub-1",
                        "datasetId": "BNCI2014_001",
                        "status": "done",
                    },
                ],
            },
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("leaderboard.mine", {})

    assert seen["method"] == "GET"
    assert seen["path"] == "/api/leaderboard/submissions"
    # List order passes through untouched (newest first, server-side).
    assert [s["submissionId"] for s in result.data["submissions"]] == ["sub-2", "sub-1"]


def _board_with_tracks() -> dict:
    import copy

    board = copy.deepcopy(LEADERBOARD)
    board["crossSubject"] = {
        "datasets": [
            {"datasetId": "BNCI2014_001", "rows": [{"pipelineId": "loso"}]},
            {"datasetId": "BNCI2014_008", "rows": [{"pipelineId": "p300"}]},
        ]
    }
    board["pretrainedFewTrial"] = {
        "datasets": [{"datasetId": "BNCI2014_001", "rows": [{"pipelineId": "reve"}]}]
    }
    board["submittableDatasets"] = ["BNCI2014_001", "BNCI2015_004"]
    return board


async def test_get_leaderboard_filters_every_track():
    board = _board_with_tracks()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=board)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("catalog.leaderboard", {"dataset_id": "BNCI2014_001"})

    data = result.data
    assert [d["datasetId"] for d in data["datasets"]] == ["BNCI2014_001"]
    assert [d["datasetId"] for d in data["crossSubject"]["datasets"]] == ["BNCI2014_001"]
    assert [d["datasetId"] for d in data["pretrainedFewTrial"]["datasets"]] == ["BNCI2014_001"]
    assert data["submittableDatasets"] == ["BNCI2014_001", "BNCI2015_004"]


async def test_get_leaderboard_unknown_dataset_names_the_board():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=LEADERBOARD)

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="BNCI2014_001"):
            await c.call_tool("catalog.leaderboard", {"dataset_id": "BNCI2014_999"})


async def test_submit_without_code_or_backbone_does_not_post():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("empty submit must not reach the API")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="submittableDatasets"):
            await c.call_tool(
                "leaderboard.submit",
                {"dataset_id": "BNCI2014_008", "name": "empty"},
            )
