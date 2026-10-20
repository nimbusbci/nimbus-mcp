"""Gateway/local parity: one build_server assembly must serve both.

The hosted gateway (``nimbus-mcp serve``) has no tool definitions of its
own — it runs the same ``build_server`` as local stdio. These tests pin
that, plus the user-facing contract that drifted when the deployed gateway
lagged four releases behind (#517/#518/#519): project.save's mandatory
expected_revision, data.delete_upload's existence, honest sampling-rate
and experiment-portability wording, and every hardcoded tool count /
version string in the package's public surfaces.

If a future change registers tools differently per transport, or re-adds
a hardcoded count that no longer matches reality, this file fails.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from nimbus_mcp.client import NullClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import HeaderCredentialClient
from nimbus_mcp.server import build_server

MCP_ROOT = Path(__file__).resolve().parents[2]


def _config() -> McpConfig:
    return McpConfig(
        api_url="https://example.test",
        mcp_key="",
        export_dir=Path("/tmp/exports"),
    )


def _tools_by_name(client: Any) -> dict[str, Any]:
    # asyncio.run (not get_event_loop): see test_annotations.py.
    tools = asyncio.run(build_server(client).list_tools())
    return {tool.name: tool for tool in tools}


def _stdio_tools() -> dict[str, Any]:
    return _tools_by_name(NullClient(_config()))


def _gateway_tools() -> dict[str, Any]:
    return _tools_by_name(HeaderCredentialClient(_config()))


def _surface(tool: Any) -> str:
    return json.dumps(
        {"description": tool.description, "parameters": tool.parameters},
        sort_keys=True,
    )


def test_gateway_and_stdio_serve_identical_tools() -> None:
    stdio = _stdio_tools()
    gateway = _gateway_tools()
    assert set(gateway) == set(stdio)
    for name in sorted(stdio):
        assert _surface(gateway[name]) == _surface(stdio[name]), (
            f"tool surface diverged between gateway and stdio: {name}"
        )


def test_project_save_requires_expected_revision() -> None:
    # #517: the silent-overwrite protection must be in the served schema.
    for tools in (_stdio_tools(), _gateway_tools()):
        tool = tools["project.save"]
        assert set(tool.parameters["required"]) == {
            "project_id",
            "train_graph",
            "expected_revision",
        }
        assert "expected_revision" in tool.description
        assert "one retry" not in tool.description


def test_delete_upload_is_served() -> None:
    # #519: added in 0.14.5; the lagging gateway lacked it entirely.
    assert "data.delete_upload" in _gateway_tools()


def test_data_upload_description_has_no_silent_250hz_claim() -> None:
    # #519: the fixed behaviour is "rejected without sampling_rate".
    desc = _gateway_tools()["data.upload"].description
    assert "silently assumes 250" not in desc
    assert "rejected" in desc


def test_experiment_tools_warn_about_portability() -> None:
    # #518: until experiment records are stored server-side, the tools must
    # say plainly that ids only resolve via the entry point that ran them.
    tools = _gateway_tools()
    assert "kindBreakdown" in tools["experiment.get"].description
    assert "PORTABILITY" in tools["experiment.get"].description
    assert "PORTABILITY" in tools["experiment.run"].description


def test_readme_tool_counts_match_reality() -> None:
    actual = len(_stdio_tools())
    readme = (MCP_ROOT / "README.md").read_text(encoding="utf-8")
    counts = [int(n) for n in re.findall(r"(\d+) tools", readme)]
    assert counts, "README should state the tool count at least once"
    assert all(n == actual for n in counts), (
        f"README tool counts {counts} != actual {actual} — update the docs "
        "or drop the hardcoded number"
    )


def test_hosted_docstring_has_no_hardcoded_tool_count() -> None:
    hosted = (MCP_ROOT / "src/nimbus_mcp/hosted.py").read_text(encoding="utf-8")
    assert not re.search(r"same \d+ tools", hosted), (
        "hosted.py must not hardcode a tool count — it drifts (was '48 tools' "
        "while the server served 60)"
    )


def test_server_json_version_matches_package() -> None:
    pyproject = (MCP_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    server_json = json.loads((MCP_ROOT / "server.json").read_text(encoding="utf-8"))
    assert server_json["version"] == package_version
    assert server_json["packages"][0]["version"] == package_version
