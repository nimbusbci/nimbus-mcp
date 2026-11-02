"""Event enrichment (mcp_version/transport/clientInfo/os/install id) and the
operator-opt-in gateway telemetry with internal-account tagging."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from nimbus_mcp import __version__
from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.product_telemetry import (
    GatewayTelemetry,
    Telemetry,
    hash_subject,
    reset_call_context,
    set_call_context,
)
from nimbus_mcp.server import build_server


class Rec:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, payload: dict) -> None:
        self.payloads.append(payload)

    def by_event(self, name: str) -> list[dict]:
        return [p for p in self.payloads if p["event"] == name]


def _local(tmp_path: Path, rec: Rec, **env: str) -> Telemetry:
    return Telemetry(
        state_path=tmp_path / "s.json",
        sender=rec,
        environ={"NIMBUS_MCP_POSTHOG_KEY": "phc_test", "NIMBUS_MCP_TELEMETRY": "on", **env},
    )


def test_local_event_enriched_and_ip_discarded(tmp_path: Path) -> None:
    rec = Rec()
    t = _local(tmp_path, rec)
    token = set_call_context(
        {"mcp_client_name": "claude-ai", "mcp_client_version": "1.2", "session_id": "s1"}
    )
    try:
        t.capture("login_completed", {"trigger": "x"}, sync=True)
    finally:
        reset_call_context(token)
    p = rec.by_event("mcp_login_completed")[0]
    props = p["properties"]
    assert props["mcp_version"] == __version__
    assert props["transport"] == "local"
    assert props["mcp_client_name"] == "claude-ai"
    assert props["mcp_client_version"] == "1.2"
    assert props["os_family"] and "os" not in props and props["install_id"] == p["distinct_id"]
    assert props["is_internal"] is False
    assert props["$ip"] == "" and props["$geoip_disable"] is True


def test_local_internal_flag(tmp_path: Path) -> None:
    rec = Rec()
    _local(tmp_path, rec, NIMBUS_MCP_INTERNAL="1").capture("login_completed", {}, sync=True)
    assert rec.by_event("mcp_login_completed")[0]["properties"]["is_internal"] is True


def test_local_opt_out_still_sends_nothing(tmp_path: Path) -> None:
    rec = Rec()
    t = Telemetry(
        state_path=tmp_path / "s.json", sender=rec, environ={"NIMBUS_MCP_TELEMETRY": "off"}
    )
    t.capture("tool_completed", {}, sync=True)
    assert rec.payloads == []


def test_gateway_off_by_default() -> None:
    rec = Rec()
    t = GatewayTelemetry(sender=rec, environ={})
    t.capture("tool_completed", {}, sync=True)
    assert rec.payloads == [] and not t.enabled() and t.pending_notice() is None


def test_gateway_kill_switch_wins() -> None:
    rec = Rec()
    t = GatewayTelemetry(
        sender=rec, environ={"NIMBUS_MCP_GATEWAY_TELEMETRY": "on", "NIMBUS_MCP_TELEMETRY": "off"}
    )
    t.capture("tool_completed", {}, sync=True)
    assert rec.payloads == []


@pytest.mark.parametrize(
    ("internal", "expected"),
    [
        ("", False),
        ("user_abc", True),
        ("dev@nimbusbci.com", True),
        (hash_subject("user_abc", "k1"), True),
        ("other", False),
    ],
)
def test_gateway_identity_and_internal(internal: str, expected: bool) -> None:
    rec = Rec()
    t = GatewayTelemetry(
        sender=rec,
        environ={
            "NIMBUS_MCP_GATEWAY_TELEMETRY": "on",
            "NIMBUS_MCP_INTERNAL_SUBJECTS": internal,
            "NIMBUS_MCP_SUBJECT_HASH_KEY": "k1",
        },
    )
    token = set_call_context(
        {"_subject": "user_abc", "_email": "Dev@NimbusBCI.com", "session_id": "s9"}
    )
    try:
        t.capture("tool_completed", {"tool": "x"}, sync=True)
        t.capture("first_run", {}, sync=True)  # install lifecycle: never on gateway
    finally:
        reset_call_context(token)
    assert [p["event"] for p in rec.payloads] == ["mcp_tool_completed"]
    p = rec.payloads[0]
    assert p["distinct_id"] == hash_subject("user_abc", "k1")
    props = p["properties"]
    assert props["transport"] == "gateway" and props["session_id"] == "s9"
    assert props["is_internal"] is expected
    blob = repr(p)
    assert (
        "user_abc" not in blob.replace(hash_subject("user_abc", "k1") or "", "")
        and "Dev@" not in blob
    )


async def test_server_attaches_client_info_and_session_started(tmp_path: Path) -> None:
    rec = Rec()
    t = _local(tmp_path, rec)
    server = build_server(
        NimbusClient(McpConfig(api_url="http://127.0.0.1:9", mcp_key=None, export_dir=tmp_path)),
        product_telemetry=t,
    )
    async with Client(server) as c:
        await c.call_tool("account.whoami", {})
        await c.call_tool("account.whoami", {})
    import time

    deadline = time.time() + 3
    while time.time() < deadline and len(rec.by_event("mcp_tool_completed")) < 2:
        time.sleep(0.02)
    started = rec.by_event("mcp_session_started")
    tool = rec.by_event("mcp_tool_completed")
    assert len(started) == 1 and tool, [p["event"] for p in rec.payloads]
    for p in (started[0], tool[0]):
        assert p["properties"]["mcp_client_name"]
        assert p["properties"]["transport"] == "local"


def test_hash_subject_is_keyed() -> None:
    assert hash_subject("user_abc", None) is None
    a, b = hash_subject("user_abc", "k1"), hash_subject("user_abc", "k2")
    assert a and b and a != b and a.startswith("sub_")


def test_gateway_without_hash_key_disables_subject_ids() -> None:
    rec = Rec()
    t = GatewayTelemetry(
        sender=rec,
        environ={"NIMBUS_MCP_GATEWAY_TELEMETRY": "on", "NIMBUS_MCP_INTERNAL_SUBJECTS": "user_abc"},
    )
    token = set_call_context({"_subject": "user_abc", "session_id": "s1"})
    try:
        t.capture("tool_completed", {}, sync=True)
    finally:
        reset_call_context(token)
    p = rec.payloads[0]
    assert p["distinct_id"] == "gateway_anonymous"
    assert p["properties"]["is_internal"] is True  # raw-subject match still works
    assert "user_abc" not in repr(p)


def test_suite_runs_as_internal() -> None:
    import os

    assert os.environ.get("NIMBUS_MCP_INTERNAL") == "1"
