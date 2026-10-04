"""Quality-score guard: every tool annotated, every parameter documented.

Smithery (and MCP clients generally) score/consume tool annotations
(readOnlyHint / destructiveHint / idempotentHint) and per-parameter
descriptions; this suite pins both so a new tool cannot silently ship
without them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nimbus_mcp.client import NullClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.server import build_server


def _make_server():
    config = McpConfig(
        api_url="https://example.test",
        mcp_key="",
        export_dir=Path("/tmp/exports"),
    )
    return build_server(NullClient(config))


def _all_tools():
    return asyncio.get_event_loop().run_until_complete(_make_server().list_tools())


def test_every_tool_has_annotations():
    for tool in _all_tools():
        ann = tool.annotations
        assert ann is not None, f"{tool.name}: missing ToolAnnotations"
        assert ann.read_only_hint is not None, f"{tool.name}: read_only_hint unset"
        assert ann.destructive_hint is not None, f"{tool.name}: destructive_hint unset"
        assert ann.idempotent_hint is not None, f"{tool.name}: idempotent_hint unset"


def test_every_parameter_is_documented():
    for tool in _all_tools():
        for pname, spec in (tool.parameters or {}).get("properties", {}).items():
            desc = (spec.get("description") or "").strip()
            assert desc, f"{tool.name}.{pname}: no parameter description"


def test_annotation_policy_invariants():
    tools = {t.name: t for t in _all_tools()}
    destructive = {n for n, t in tools.items() if t.annotations.destructive_hint}
    assert destructive == {"cancel_execution", "stop_stream"}
    # The two rail-gated tools must not claim read-only.
    assert tools["start_stream"].annotations.read_only_hint is False
    assert tools["run_pipeline"].annotations.read_only_hint is False
    # Reads are idempotent (safe for clients to retry).
    for name in ("whoami", "list_nodes", "get_results", "inspect_dataset"):
        assert tools[name].annotations.read_only_hint is True
        assert tools[name].annotations.idempotent_hint is True


def test_tool_count_is_tracked():
    # A new tool must update this number AND pass the suites above — the count
    # is asserted so additions are conscious.
    assert len(_all_tools()) == 32
