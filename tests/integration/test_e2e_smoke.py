"""True end-to-end smoke against a RUNNING local backend with MCP_LOCAL_KEY set.

Gate: NIMBUS_MCP_E2E=1 + NIMBUS_MCP_KEY + a backend on NIMBUS_API_URL whose
MCP_LOCAL_KEY matches, DEBUG=1, and the BNCI2014_001 pack on disk.
Skipped otherwise (CI runs the unit suite; this is the local demo gate).
"""

import os
import time

import pytest
from fastmcp import Client

pytestmark = pytest.mark.skipif(os.environ.get("NIMBUS_MCP_E2E") != "1", reason="NIMBUS_MCP_E2E!=1")


def _server():
    from nimbus_mcp.server import build_server

    return build_server()


async def test_full_offline_flow():
    async with Client(_server()) as c:
        nodes = await c.call_tool("catalog.nodes", {})
        assert nodes.data["count"] > 0
        templates = await c.call_tool("catalog.templates", {})
        assert templates.data["count"] > 0
        template_id = templates.data["templates"][0]["id"]
        detail = await c.call_tool("catalog.template", {"template_id": template_id})
        train = detail.data["train"]
        validated = await c.call_tool("pipeline.validate", {"train_graph": train})
        assert validated.data["valid"] is True
        started = await c.call_tool(
            "execution.run", {"train_graph": train, "name": "mcp e2e smoke"}
        )
        execution_id = started.data["executionId"]
        deadline = time.time() + 900
        status = "running"
        while time.time() < deadline:
            summary = await c.call_tool("execution.get", {"execution_id": execution_id})
            status = summary.data["status"]
            if status in ("completed", "failed", "cancelled"):
                break
            time.sleep(5)
        assert status == "completed", f"execution ended as {status}"
        results = await c.call_tool("execution.results", {"execution_id": execution_id})
        assert "kappa" in results.data["metrics"]


async def test_projects_and_experiment_flow():
    async with Client(_server()) as c:
        proj = await c.call_tool("project.create", {"name": "mcp v0.2 e2e"})
        pid = proj.data["projectId"]
        templates = await c.call_tool("catalog.templates", {})
        train = (
            await c.call_tool(
                "catalog.template", {"template_id": templates.data["templates"][0]["id"]}
            )
        ).data["train"]
        saved = await c.call_tool("project.save", {"project_id": pid, "train_graph": train})
        assert saved.data["revision"] >= 1
        loaded = await c.call_tool("project.load", {"project_id": pid})
        assert loaded.data["train"]["nodes"]
        exp = await c.call_tool(
            "experiment.run", {"runs": [{"name": "smoke", "train_graph": train}]}
        )
        eid = exp.data["experimentId"]
        deadline = time.time() + 900
        status = "running"
        while time.time() < deadline:
            s = await c.call_tool("experiment.get", {"experiment_id": eid})
            status = s.data["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(5)
        assert status == "completed", f"experiment ended as {status}"
        s = await c.call_tool("experiment.get", {"experiment_id": eid})
        assert s.data["aggregates"]["kappa"]["mean"] is not None
