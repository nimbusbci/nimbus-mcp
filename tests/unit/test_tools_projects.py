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


async def test_save_pipeline_requires_expected_revision():
    """Omitting expected_revision must fail closed — never blind-overwrite (#468)."""

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call the API without expected_revision")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="expected_revision"):
            await c.call_tool(
                "project.save", {"project_id": "p1", "train_graph": TRAIN, "name": "mi"}
            )


async def test_save_pipeline_409_surfaces_conflict_without_overwrite_retry():
    """A 409 must surface immediately — never re-GET and force-overwrite (#468)."""
    puts, put_bodies = [], []

    def handler(req: httpx.Request) -> httpx.Response:
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
            await c.call_tool(
                "project.save",
                {"project_id": "p1", "train_graph": TRAIN, "expected_revision": 1},
            )
    assert len(puts) == 1
    assert put_bodies[0]["expectedRevision"] == 1


async def test_save_pipeline_uses_caller_expected_revision_without_get():
    put_bodies = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET":
            raise AssertionError("must not GET when expected_revision is provided")
        if req.method == "PUT" and req.url.path.endswith("/doc"):
            put_bodies.append(json.loads(req.read()))
            return httpx.Response(
                200, json={"ok": True, "projectId": "p1", "revision": 4, "message": "saved"}
            )
        raise AssertionError((req.method, req.url.path))

    async with Client(make_server(handler)) as c:
        r = await c.call_tool(
            "project.save",
            {
                "project_id": "p1",
                "train_graph": TRAIN,
                "expected_revision": 3,
                "subject": "S03",
            },
        )
    assert r.data["revision"] == 4
    assert put_bodies[0]["expectedRevision"] == 3
    assert put_bodies[0]["subject"] == "S03"


async def test_save_pipeline_stale_expected_revision_conflicts():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "PUT" and req.url.path.endswith("/doc"):
            body = json.loads(req.read())
            assert body["expectedRevision"] == 2
            return httpx.Response(
                409,
                json={"code": "nimbus.project.doc_conflict", "detail": "stale"},
            )
        raise AssertionError((req.method, req.url.path))

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="expected revision 2 is stale"):
            await c.call_tool(
                "project.save",
                {"project_id": "p1", "train_graph": TRAIN, "expected_revision": 2},
            )


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


async def test_delete_project_refuses_without_confirm():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("must not touch HTTP without confirm")

    async with Client(make_server(handler)) as c:
        r = await c.call_tool("project.delete", {"project_id": "p1"})
    assert r.data["deleted"] is False
    assert "confirm=true" in r.data["message"]


async def test_delete_project_sends_delete_with_confirm():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"], seen["path"] = req.method, req.url.path
        return httpx.Response(
            200, json={"ok": True, "id": "p1", "message": "Project removed successfully"}
        )

    async with Client(make_server(handler)) as c:
        r = await c.call_tool("project.delete", {"project_id": "p1", "confirm": True})
    assert seen == {"method": "DELETE", "path": "/api/projects/p1"}
    assert r.data["deleted"] is True and r.data["projectId"] == "p1"


async def test_delete_project_rejects_traversal_id():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("must not touch HTTP for an unsafe id")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="project id"):
            await c.call_tool("project.delete", {"project_id": "../evil", "confirm": True})


async def test_delete_project_404_surfaces_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "code": "nimbus.project.not_found",
                "title": "NotFound",
                "detail": "Project not found",
            },
        )

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="Project not found"):
            await c.call_tool("project.delete", {"project_id": "p1", "confirm": True})
