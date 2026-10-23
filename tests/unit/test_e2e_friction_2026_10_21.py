"""E2E friction fixes (Nimbus MCP Test rounds 1+2, 2026-10-21)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from nimbus_mcp.client import McpToolError, NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import StripUnboundSessionIdMiddleware
from nimbus_mcp.server import build_server
from nimbus_mcp.setup_mode import local_mode_guidance
from nimbus_mcp.tools._args import friendly_validation_message, normalize_arguments
from nimbus_mcp.tools._guards import safe_segment
from nimbus_mcp.tools.run import _trim_summary
from nimbus_mcp.tools.whoami import _real_email

_INDEX = {"datasets": {"BNCI2014_001": {"onDisk": True, "defaultSubject": "S01"}}}


def _server(tmp_path: Path, seen: list[httpx.Request]):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.path == "/api/public-datasets/index":
            return httpx.Response(200, json=_INDEX)
        return httpx.Response(200, json={"ok": True})

    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=tmp_path),
        transport=httpx.MockTransport(handler),
    )
    return build_server(client)


async def test_dataset_id_alias_and_default_subject(tmp_path):
    seen: list[httpx.Request] = []
    async with Client(_server(tmp_path, seen)) as c:
        await c.call_tool("data.inspect_dataset", {"dataset_id": "BNCI2014_001"})
    describe = [r for r in seen if r.url.path == "/api/data/describe"][0]
    assert describe.url.params["dataset"] == "BNCI2014_001"
    assert describe.url.params["subject"] == "S01"


async def test_unknown_param_is_one_friendly_line(tmp_path):
    async with Client(_server(tmp_path, [])) as c:
        result = await c.call_tool(
            "data.inspect_dataset", {"bogus": 1, "dataset": "X"}, raise_on_error=False
        )
    text = result.content[0].text
    assert result.is_error
    assert "unknown parameter(s) 'bogus'" in text
    assert "Valid parameters: dataset (required)" in text
    assert "pydantic" not in text


def test_graph_alias_maps_to_train_graph():
    schema = {"properties": {"train_graph": {}, "name": {}}, "required": ["train_graph"]}
    fixed, renamed = normalize_arguments({"graph": {"nodes": []}}, schema)
    assert fixed == {"train_graph": {"nodes": []}} and renamed == {"graph": "train_graph"}


def test_pydantic_dump_rewritten():
    raw = (
        "1 validation error for call[run]\ntrain_graph\n  Input should be a valid "
        "dictionary [type=dict_type, input_value='x', input_type=str]\n"
        "    For further information visit https://errors.pydantic.dev/2/v/dict_type"
    )
    msg = friendly_validation_message("execution.run", raw, {"properties": {"train_graph": {}}})
    assert msg is not None
    assert "train_graph: Input should be a valid dictionary" in msg
    assert "errors.pydantic.dev" not in msg


def test_whoami_drops_clerk_placeholder():
    assert _real_email("user_2abc@clerk.local") is None
    assert _real_email("a@b.com") == "a@b.com"


def test_summary_exposes_error_code_and_correlation_id():
    row = {
        "executionId": "888744af",
        "status": "failed",
        "errorMessage": "Hosted compute is currently unavailable (code: "
        "nimbus.compute.unavailable, correlationId: 3fa9c1d2e4b54a01)",
    }
    out = _trim_summary(row)
    assert out["errorCode"] == "nimbus.compute.unavailable"
    assert out["correlationId"] == "3fa9c1d2e4b54a01"


def test_traversal_rejection_has_code_and_correlation_id():
    with pytest.raises(McpToolError) as info:
        safe_segment("../../etc/passwd", label="artifact name")
    assert info.value.code == "nimbus.mcp.invalid_argument"
    assert info.value.correlation_id
    assert "correlationId:" in str(info.value)


def test_unreachable_guidance_mentions_linux():
    assert "Linux" in local_mode_guidance(None, "http://127.0.0.1:8080")["message"]


async def test_setup_guidance_is_flagged_as_error(tmp_path, monkeypatch):
    for k in ("NIMBUS_MCP_KEY", "NIMBUS_MCP_KEY_FILE", "NIMBUS_TOKEN"):
        monkeypatch.delenv(k, raising=False)

    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    client = NimbusClient(
        McpConfig(api_url="http://127.0.0.1:9", mcp_key="", export_dir=tmp_path),
        transport=httpx.MockTransport(down),
    )
    async with Client(build_server(client)) as c:
        result = await c.call_tool("catalog.templates", {}, raise_on_error=False)
    assert result.is_error
    assert result.structured_content["setupRequired"] is True


def _session_app(status: int):
    async def ep(request):
        return JSONResponse({"e": 1}, status_code=status, headers={"mcp-session-id": "abc"})

    app = Starlette(routes=[Route("/mcp", ep, methods=["POST"])])
    return StripUnboundSessionIdMiddleware(app)


def test_error_without_session_does_not_mint_session_id():
    with TestClient(_session_app(400)) as c:
        assert "mcp-session-id" not in c.post("/mcp").headers
        # A client that already holds a session keeps seeing it.
        assert c.post("/mcp", headers={"mcp-session-id": "abc"}).headers["mcp-session-id"]
    with TestClient(_session_app(200)) as c:
        assert c.post("/mcp").headers["mcp-session-id"] == "abc"


async def test_experiment_errors_carry_code(tmp_path):
    async with Client(_server(tmp_path, [])) as c:
        empty = await c.call_tool("experiment.run", {"runs": []}, raise_on_error=False)
    assert empty.is_error
    assert "code: nimbus.experiment.invalid_request" in empty.content[0].text
    assert "correlationId:" in empty.content[0].text


def test_summary_exposes_catalog_dataset_id():
    out = _trim_summary({"executionId": "e", "dataset": "BCIIV2a", "datasetId": "BNCI2014_001"})
    # dataset = display label, datasetId = catalog id (contract kept after #583 review)
    assert out["dataset"] == "BCIIV2a" and out["datasetId"] == "BNCI2014_001"
    assert "datasetLabel" not in out
