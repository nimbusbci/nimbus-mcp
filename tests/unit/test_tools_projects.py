import json
from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import projects

TRAIN = {"nodes": [{"id": "d", "type": "public_data", "config": {}}], "connections": []}


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    projects.register(server, client)
    return server


async def test_create_project_posts_name():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"] = req.url.path, req.read()
        return httpx.Response(200, json={"ok": True, "project": {"id": "p1", "name": "agent"}})

    async with Client(make_server(handler)) as c:
        r = await c.call_tool("project.create", {"name": "agent", "description": "d"})
    assert seen["path"] == "/api/projects" and b'"name"' in seen["body"]
    assert r.data["projectId"] == "p1"


async def test_save_pipeline_reads_revision_then_puts_with_layout():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        if req.method == "GET" and req.url.path.endswith("/doc"):
            return httpx.Response(
                200,
                json={"ok": True, "projectId": "p1", "train": TRAIN, "revision": 7, "name": "old"},
            )
        if req.method == "PUT" and req.url.path.endswith("/doc"):
            body = req.read()
            assert b'"expectedRevision"' in body and b'"layout"' in body
            return httpx.Response(
                200, json={"ok": True, "projectId": "p1", "revision": 8, "message": "saved"}
            )
        raise AssertionError((req.method, req.url.path))

    async with Client(make_server(handler)) as c:
        r = await c.call_tool(
            "project.save", {"project_id": "p1", "train_graph": TRAIN, "name": "mi"}
        )
    assert r.data["revision"] == 8
    assert ("GET", "/api/projects/p1/doc") in calls
    assert ("PUT", "/api/projects/p1/doc") in calls


async def test_save_pipeline_retries_once_on_409():
    puts = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/doc"):
            return httpx.Response(
                200, json={"ok": True, "projectId": "p1", "train": TRAIN, "revision": 1}
            )
        if req.method == "PUT" and req.url.path.endswith("/doc"):
            puts.append(1)
            if len(puts) == 1:
                return httpx.Response(
                    409, json={"code": "nimbus.project.doc_conflict", "detail": "stale"}
                )
            return httpx.Response(200, json={"ok": True, "projectId": "p1", "revision": 2})
        raise AssertionError(req.url.path)

    async with Client(make_server(handler)) as c:
        r = await c.call_tool("project.save", {"project_id": "p1", "train_graph": TRAIN})
    assert len(puts) == 2 and r.data["revision"] == 2


async def test_save_pipeline_second_409_surfaces_conflict_error():
    """A 409 that survives the single retry must surface as a tool error that
    names the backend conflict code, not silently return a failed payload."""
    puts, gets, put_bodies = [], [], []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path.endswith("/doc"):
            gets.append(1)
            return httpx.Response(
                200, json={"ok": True, "projectId": "p1", "train": TRAIN, "revision": len(gets)}
            )
        if req.method == "PUT" and req.url.path.endswith("/doc"):
            puts.append(1)
            put_bodies.append(json.loads(req.read()))
            return httpx.Response(
                409,
                json={
                    "code": "nimbus.project.doc_conflict",
                    "title": "Conflict",
                    "detail": "This canvas was modified elsewhere.",
                },
            )
        raise AssertionError((req.method, req.url.path))

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="nimbus.project.doc_conflict"):
            await c.call_tool("project.save", {"project_id": "p1", "train_graph": TRAIN})
    # one initial read + one re-read for the retry; two PUTs, the second
    # carrying the freshly re-read revision.
    assert len(gets) == 2 and len(puts) == 2
    assert put_bodies[1]["expectedRevision"] == 2


async def test_load_pipeline_returns_train_graph():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/projects/p1/doc"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "projectId": "p1",
                "name": "mi",
                "subject": "S01",
                "train": TRAIN,
                "revision": 3,
            },
        )

    async with Client(make_server(handler)) as c:
        r = await c.call_tool("project.load", {"project_id": "p1"})
    assert r.data["train"]["nodes"][0]["type"] == "public_data"
    assert r.data["subject"] == "S01"
