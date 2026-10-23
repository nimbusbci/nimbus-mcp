"""E2E round 3 (2026-10-22) partial fixes."""

from __future__ import annotations

import httpx
import pytest
from fastmcp import Client

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.server import build_server
from nimbus_mcp.tools.run import _trim_summary, metric_warning


def _server(tmp_path):
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=tmp_path),
        transport=httpx.MockTransport(lambda _req: httpx.Response(200, json={"ok": True})),
    )
    return build_server(client)


@pytest.mark.parametrize(
    ("tool", "args", "field"),
    [
        ("execution.list", {"limit": "abc"}, "limit"),
        ("execution.get", {"execution_id": 12}, "execution_id"),
        ("pipeline.validate", {"train_graph": "notadict"}, "train_graph"),
    ],
)
async def test_wrong_type_argument_is_coded_and_readable(tmp_path, tool, args, field):
    async with Client(_server(tmp_path)) as c:
        result = await c.call_tool(tool, args, raise_on_error=False)
    text = result.content[0].text
    assert result.is_error
    assert f"Invalid arguments for {tool}: {field}:" in text
    assert "code: nimbus.mcp.invalid_argument" in text and "correlationId:" in text
    assert "pydantic" not in text and "validation error for call" not in text


def test_near_majority_accuracy_warns_even_with_balanced_headline():
    metrics = {
        "headlineMetric": {"key": "balancedAccuracy", "valuePct": 58.1, "isInSample": False},
        "classBalance": {"0": 3500, "1": 700},
        "chanceLevelPct": 83.33,
        "evalAccuracyPct": 83.57,
        "accuracyPct": 83.57,
    }
    warning = metric_warning(metrics)
    assert warning is not None and "majority-class" in warning


def test_clear_win_has_no_warning():
    metrics = {
        "headlineMetric": {"key": "accuracy", "valuePct": 80.0},
        "chanceLevelPct": 50.0,
        "accuracyPct": 80.0,
        "evalAccuracyPct": 80.0,
    }
    assert metric_warning(metrics) is None


def test_summary_has_title():
    out = _trim_summary({"executionId": "e", "pipelineName": "r3 CSP6 keep-name"})
    assert out["title"] == "r3 CSP6 keep-name"


def test_inner_validation_error_is_not_relabelled_as_bad_arguments():
    from nimbus_mcp.tools._args import friendly_validation_message

    inner = (
        "1 validation error for ExecutionSummary\nstatus\n  Input should be a valid string "
        "[type=string_type, input_value=1, input_type=int]"
    )
    assert friendly_validation_message("execution.get", inner, {"properties": {}}) is None


def test_balanced_near_chance_warns_once():
    metrics = {
        "headlineMetric": {"key": "accuracy", "valuePct": 52.0},
        "chanceLevelPct": 50.0,
        "accuracyPct": 52.0,
        "evalAccuracyPct": 52.0,
    }
    warning = metric_warning(metrics)
    assert warning is not None and warning.count("chance") == 1 and "majority" not in warning
