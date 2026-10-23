"""project.rename / execution.delete / experiment.delete (E2E round 2 gaps)."""

import json
from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import campaign, projects, run


@pytest.fixture(autouse=True)
def _disk(tmp_path, monkeypatch):
    root = tmp_path / "exp"
    monkeypatch.setattr(campaign, "_experiment_disk_dir", lambda fp: root / fp[:8])
    return root


def _server(handler, mod) -> tuple[FastMCP, NimbusClient]:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    mod.register(server, client)
    return server, client


async def test_project_rename_patches_name():
    seen = {}

    def handler(req):
        seen["m"], seen["p"], seen["b"] = req.method, req.url.path, json.loads(req.read())
        return httpx.Response(200, json={"project": {"id": "p1", "name": "New"}})

    server, _ = _server(handler, projects)
    async with Client(server) as c:
        r = await c.call_tool("project.rename", {"project_id": "p1", "name": "New"})
    assert (seen["m"], seen["p"], seen["b"]) == ("PATCH", "/api/projects/p1", {"name": "New"})
    assert r.data == {"projectId": "p1", "name": "New"}


async def test_execution_delete_requires_confirm():
    def handler(req):
        raise AssertionError("no API call without confirm")

    server, _ = _server(handler, run)
    async with Client(server) as c:
        r = await c.call_tool("execution.delete", {"execution_id": "e1"})
    assert r.data["deleted"] is False


async def test_execution_delete_calls_delete_and_surfaces_409():
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path))
        if req.url.path.endswith("/e2"):
            return httpx.Response(
                409,
                json={"code": "nimbus.execution.delete_not_terminal", "detail": "running"},
            )
        return httpx.Response(200, json={"deleted": True})

    server, _ = _server(handler, run)
    async with Client(server) as c:
        r = await c.call_tool("execution.delete", {"execution_id": "e1", "confirm": True})
        assert r.data == {"deleted": True, "executionId": "e1"}
        bad = await c.call_tool(
            "execution.delete", {"execution_id": "e2", "confirm": True}, raise_on_error=False
        )
    assert calls[0] == ("DELETE", "/api/executions/e1")
    assert bad.is_error and "delete_not_terminal" in bad.content[0].text


async def test_experiment_delete_removes_local_and_server(_disk):
    calls = []

    def handler(req):
        calls.append((req.method, req.url.path))
        return httpx.Response(404, json={"code": "nimbus.experiment.not_found"})

    server, client = _server(handler, campaign)
    fp = client.credential_fingerprint()
    d = _disk / fp[:8]
    d.mkdir(parents=True)
    (d / "x1.json").write_text("{}")
    async with Client(server) as c:
        r = await c.call_tool("experiment.delete", {"experiment_id": "x1", "confirm": True})
        assert r.data == {"deleted": True, "experimentId": "x1"}
        assert not (d / "x1.json").exists()
        gone = await c.call_tool(
            "experiment.delete", {"experiment_id": "x1", "confirm": True}, raise_on_error=False
        )
    assert ("DELETE", "/api/experiments/x1") in calls
    assert gone.is_error and "nimbus.experiment.not_found" in gone.content[0].text


async def test_experiment_delete_refuses_running():
    def handler(req):
        raise AssertionError("must not delete a running experiment")

    server, client = _server(handler, campaign)
    fp = client.credential_fingerprint()
    state = campaign.ExperimentState.__new__(campaign.ExperimentState)
    state.status = "running"
    key = campaign.EXPERIMENTS.key(fp, "x2")
    campaign.EXPERIMENTS._states[key] = state
    try:
        async with Client(server) as c:
            r = await c.call_tool(
                "experiment.delete", {"experiment_id": "x2", "confirm": True}, raise_on_error=False
            )
        assert r.is_error and "still running" in r.content[0].text
    finally:
        campaign.EXPERIMENTS._states.pop(key, None)
