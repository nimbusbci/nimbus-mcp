"""Drift guard: every REST path mapped by tools must exist in the backend OpenAPI."""

import os

import httpx
import pytest

MAPPED_PATHS = [
    "/api/node-types",
    "/api/templates",
    "/api/leaderboard",
    "/api/public-datasets/index",
    "/api/validate-pipeline",
    "/api/validate-node-config",
    "/api/execute",
    "/api/cancel",
    "/api/executions",
    "/api/result/{execution_id}",
    "/api/executions/{execution_id}/summary",
    "/api/executions/{execution_id}/artifacts",
    "/api/executions/{execution_id}/artifacts/{artifact_name}",
    "/api/export/python",
    "/api/templates/{template_id}",
    "/api/deployment/devices",
    "/api/deployment/test-device",
    "/api/hardware/connect",
    "/api/hardware/disconnect",
    "/api/hardware/start-stream",
    "/api/hardware/stop-stream",
    "/api/hardware/stream-status",
    "/api/hardware/session-live",
    "/api/projects",
    "/api/projects/{project_id}/doc",
    "/api/upload",
    "/api/data/describe",
]

pytestmark = pytest.mark.skipif(
    os.environ.get("NIMBUS_MCP_E2E") != "1", reason="NIMBUS_MCP_E2E!=1"
)


def test_mapped_paths_exist_in_openapi():
    base = os.environ.get("NIMBUS_API_URL", "http://127.0.0.1:8080")
    spec = httpx.get(f"{base}/openapi.json", timeout=10).json()
    paths = set(spec.get("paths", {}))
    missing = [p for p in MAPPED_PATHS if p not in paths]
    assert not missing, f"REST paths referenced by MCP tools no longer exist: {missing}"
