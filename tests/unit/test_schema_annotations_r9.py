"""E2E round 9 N2/N3/N4: enums, annotations, titles, specific error codes."""

import json
from pathlib import Path

import httpx
import pytest
from fastmcp import Client

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.server import build_server


def _server(handler=None):
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler or (lambda r: httpx.Response(500))),
    )
    return build_server(client=client)


async def _tools(server=None):
    async with Client(server or _server()) as c:
        return {t.name: t for t in await c.list_tools()}


def _enum(tool, param):
    return tool.input_schema["properties"][param].get("enum")


async def test_fixed_choice_params_publish_enums():
    t = await _tools()
    assert _enum(t["plots.dataset"], "kind") == ["psd", "topomap", "bandpower", "spectrogram"]
    assert _enum(t["plots.dataset"], "band") == ["delta", "theta", "alpha", "beta", "gamma"]
    for name in ("plots.dataset", "plots.erp", "foundation.compare"):
        assert _enum(t[name], "source") == ["dataset", "upload", "local"], name
        assert _enum(t[name], "mode") == ["training", "evaluation", "all"], name
    assert _enum(t["data.inspect_dataset"], "mode") == ["training", "evaluation", "all"]
    assert _enum(t["plots.leaderboard"], "track") == [
        "within_session",
        "crossSubject",
        "pretrained_few_trial",
    ]
    assert _enum(t["leaderboard.submit"], "track") == ["cross_subject", "pretrained_few_trial"]
    assert _enum(t["leaderboard.submit"], "verification") == ["verified", "unverified_local"]
    assert set(_enum(t["execution.list"], "status")) >= {"running", "completed", "failed"}
    assert "pack" in _enum(t["bids.export_dataset"], "kind")


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("foundation.compare", {"source": "bogus"}),
        ("foundation.compare", {"source": "dataset", "mode": "nope"}),
        ("plots.erp", {"source": "bogus"}),
        ("plots.dataset", {"kind": "psd", "source": "bogus"}),
        ("execution.list", {"status": "bogus"}),
        ("leaderboard.submit", {"dataset_id": "d", "name": "n", "track": "bogus"}),
        ("data.inspect_dataset", {"dataset": "d", "mode": "bogus"}),
        ("bids.export_dataset", {"dataset": "d", "kind": "bogus"}),
    ],
)
async def test_bad_choice_is_invalid_argument_before_http(tool, args):
    def handler(req):
        raise AssertionError("no HTTP for an invalid choice")

    async with Client(_server(handler)) as c:
        result = await c.call_tool(tool, args, raise_on_error=False)
    assert result.is_error
    text = result.content[0].text
    assert "nimbus.mcp.invalid_argument" in text
    assert "nimbus.mcp.tool_error" not in text


async def test_annotation_policy_r9():
    t = await _tools()
    assert t["protocol.stop"].annotations.destructive_hint is True
    assert t["stream.stop"].annotations.destructive_hint is True
    assert t["project.save"].annotations.destructive_hint is True
    assert t["project.rename"].annotations.idempotent_hint is True
    for name in ("python.validate", "pipeline.validate", "protocol.validate"):
        # They POST to the backend validators, so openWorldHint stays true (r9 review).
        assert t[name].annotations.open_world_hint is True, name
        assert t[name].annotations.read_only_hint is True, name


async def test_every_tool_has_a_human_title():
    for name, tool in (await _tools()).items():
        title = tool.title
        assert title, name
        assert "." not in title and "_" not in title, (name, title)
    t = await _tools()
    assert t["catalog.node_schema"].title == "Node Config Schema"
    assert t["bids.export_dataset"].title == "Export Dataset as BIDS"


async def test_project_save_conflict_has_doc_conflict_code():
    def handler(req):
        return httpx.Response(409, json={"code": "nimbus.project.doc_conflict", "detail": "x"})

    async with Client(_server(handler)) as c:
        result = await c.call_tool(
            "project.save",
            {
                "project_id": "p1",
                "train_graph": {"nodes": [], "connections": []},
                "expected_revision": 1,
            },
            raise_on_error=False,
        )
    assert result.is_error
    text = result.content[0].text
    body = json.loads(text) if text.startswith("{") else None
    assert "nimbus.project.doc_conflict" in text
    assert "nimbus.mcp.tool_error" not in text
    if body is not None:
        assert body.get("code") == "nimbus.project.doc_conflict"
