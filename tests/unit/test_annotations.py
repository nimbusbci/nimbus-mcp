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
    # asyncio.run (not get_event_loop): the deprecated loop getter breaks when
    # this file runs after async test files that close their loop.
    return asyncio.run(_make_server().list_tools())


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
    assert destructive == {"execution.cancel", "stream.stop"}
    # The two rail-gated tools must not claim read-only.
    assert tools["stream.start"].annotations.read_only_hint is False
    assert tools["execution.run"].annotations.read_only_hint is False
    # Calibration lifecycle: start/train/pause/resume mutate, status is a read.
    cal_mutating = (
        "calibration.start",
        "calibration.train",
        "calibration.pause",
        "calibration.resume",
    )
    for name in cal_mutating:
        assert tools[name].annotations.read_only_hint is False
        assert tools[name].annotations.destructive_hint is False
    assert tools["calibration.status"].annotations.read_only_hint is True
    assert tools["calibration.status"].annotations.idempotent_hint is True
    # Reads are idempotent (safe for clients to retry).
    for name in ("account.whoami", "catalog.nodes", "execution.results", "data.inspect_dataset"):
        assert tools[name].annotations.read_only_hint is True
        assert tools[name].annotations.idempotent_hint is True
    # Local-write tools (downloads/exports into NIMBUS_EXPORT_DIR) change no
    # backend state but DO write the local environment, so they must not claim
    # readOnlyHint (m8ven annotation audit 2026-10-08); retry-safe.
    for name in (
        "execution.download_artifact",
        "pipeline.export",
        "bids.export_dataset",
        "bids.export_execution",
    ):
        assert tools[name].annotations.read_only_hint is False
        assert tools[name].annotations.destructive_hint is False
        assert tools[name].annotations.idempotent_hint is True


def test_tool_count_is_tracked():
    # A new tool must update this number AND pass the suites above — the count
    # is asserted so additions are conscious.
    assert len(_all_tools()) == 39
