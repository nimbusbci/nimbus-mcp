from pathlib import Path

import httpx
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
