"""X11/X12 acceptance: opt-in PostHog telemetry + version drift check.

Hard requirements under test: telemetry OFF (or undecided) = zero network
calls; payloads carry only allowlisted, content-free properties; the
consent notice appears exactly once; the version check warns when behind
and stays silent when current or offline.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastmcp import Client

from nimbus_mcp import __version__, version_check
from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.product_telemetry import Telemetry, duration_bucket
from nimbus_mcp.server import build_server

FORBIDDEN_KEY_PARTS = (
    "path",
    "file",
    "dataset",
    "message",
    "content",
    "subject",
    "token",
    "key",
)


class Recorder:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.event = threading.Event()

    def __call__(self, payload: dict) -> None:
        self.payloads.append(payload)
        self.event.set()

    def wait(self, n: int, timeout: float = 3.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.payloads) >= n:
                return
            time.sleep(0.02)
        raise AssertionError(f"expected {n} payloads, got {len(self.payloads)}")

    def names(self) -> list[str]:
        return [p["event"] for p in self.payloads]


def _telemetry(tmp_path: Path, recorder: Recorder, **kw) -> Telemetry:
    return Telemetry(
        state_path=tmp_path / "telemetry-state.json",
        sender=recorder,
        environ={"NIMBUS_MCP_POSTHOG_KEY": "phc_test"},
        **kw,
    )


def _client(tmp_path: Path) -> NimbusClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/templates"
        return httpx.Response(200, json={"templates": [{"id": "t1"}]})

    cfg = McpConfig(
        api_url="http://test", mcp_key="k", export_dir=tmp_path, credential_meta={}
    )
    return NimbusClient(cfg, transport=httpx.MockTransport(handler))


# ── consent / zero-network guarantees ────────────────────────────────────────


async def test_telemetry_off_makes_zero_network_calls(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.append((a, k)))
    recorder = Recorder()
    telemetry = _telemetry(tmp_path, recorder)
    telemetry.set_consent(False)
    server = build_server(_client(tmp_path), product_telemetry=telemetry)
    async with Client(server) as c:
        await c.call_tool("catalog.templates", {})
        await c.call_tool("catalog.templates", {})
    time.sleep(0.2)
    assert recorder.payloads == []
    assert calls == []  # not even the default transport was touched


async def test_undecided_consent_sends_nothing_and_notices_once(tmp_path):
    recorder = Recorder()
    telemetry = _telemetry(tmp_path, recorder)
    server = build_server(_client(tmp_path), product_telemetry=telemetry)
    async with Client(server) as c:
        first = await c.call_tool("catalog.templates", {})
        second = await c.call_tool("catalog.templates", {})
    assert first.data["telemetryNotice"]["telemetry"] == "off-until-consent"
    assert "telemetryNotice" not in second.data
    time.sleep(0.2)
    assert recorder.payloads == []


async def test_env_off_is_a_hard_off(tmp_path):
    recorder = Recorder()
    telemetry = Telemetry(
        state_path=tmp_path / "s.json",
        sender=recorder,
        environ={"NIMBUS_MCP_POSTHOG_KEY": "phc_test", "NIMBUS_MCP_TELEMETRY": "off"},
    )
    telemetry._load()["consent"] = "granted"  # stored grant must not beat env off
    telemetry.capture("tool_completed", {"tool": "x"})
    time.sleep(0.2)
    assert recorder.payloads == []
    assert telemetry.pending_notice() is None


# ── event payloads: allowlisted + content-free ───────────────────────────────


async def test_tool_completed_payload_is_allowlisted(tmp_path):
    recorder = Recorder()
    telemetry = _telemetry(tmp_path, recorder)
    telemetry.set_consent(True)
    recorder.wait(1)
    assert recorder.names() == ["mcp_first_run"]  # first_run precedes everything

    server = build_server(_client(tmp_path), product_telemetry=telemetry)
    async with Client(server) as c:
        await c.call_tool("catalog.templates", {})
    recorder.wait(3)
    names = recorder.names()
    assert "mcp_first_local_result" in names
    assert "mcp_tool_completed" in names
    payload = next(p for p in recorder.payloads if p["event"] == "mcp_tool_completed")
    props = payload["properties"]
    assert payload["api_key"] == "phc_test"
    assert payload["distinct_id"]  # install uuid, not an account id
    assert props["$ip"] == "" and props["$geoip_disable"] is True
    assert props["$process_person_profile"] is False
    assert props["tool"] == "catalog.templates"
    assert props["success"] is True and props["error_code"] is None
    assert props["duration_bucket"] in ("<1s", "1-5s", "5-30s", "30s+")
    for key in props:
        tokens = set(key.lower().lstrip("$").split("_"))
        assert not (tokens & set(FORBIDDEN_KEY_PARTS)), (
            f"forbidden property key: {key}"
        )


def test_first_run_sent_once_across_instances(tmp_path):
    recorder = Recorder()
    t1 = _telemetry(tmp_path, recorder)
    t1.set_consent(True)
    recorder.wait(1)
    t2 = _telemetry(tmp_path, recorder)
    t2.capture("tool_completed", {"tool": "x"})
    recorder.wait(2)
    assert recorder.names().count("mcp_first_run") == 1


def test_no_key_configured_means_no_capture_even_when_consented(tmp_path, monkeypatch):
    from nimbus_mcp import product_telemetry

    monkeypatch.setattr(product_telemetry, "DEFAULT_POSTHOG_KEY", "")
    recorder = Recorder()
    telemetry = Telemetry(
        state_path=tmp_path / "s.json", sender=recorder, environ={}
    )
    telemetry.set_consent(True)
    time.sleep(0.2)
    assert recorder.payloads == []


def test_duration_buckets():
    assert duration_bucket(0.2) == "<1s"
    assert duration_bucket(2) == "1-5s"
    assert duration_bucket(10) == "5-30s"
    assert duration_bucket(60) == "30s+"


# ── version check (X11) ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_version_check():
    version_check.reset_for_tests()
    yield
    version_check.reset_for_tests()


def _run_check(tmp_path, fetcher):
    version_check.start_background_check(
        fetcher=fetcher, cache_path=tmp_path / "vc.json"
    )
    deadline = time.time() + 3
    while time.time() < deadline:
        # Peek without consuming: pending_notice() is once-per-process.
        if version_check._notice is not None:
            return dict(version_check._notice)
        time.sleep(0.02)
    return None


def test_version_check_warns_when_behind(tmp_path):
    notice = _run_check(tmp_path, lambda: "99.0.0")
    assert notice is not None
    assert notice["current"] == __version__ and notice["latest"] == "99.0.0"
    first = version_check.pending_notice()
    assert first is not None and first["latest"] == "99.0.0"
    assert version_check.pending_notice() is None  # once per process


def test_version_check_silent_when_current(tmp_path):
    assert _run_check(tmp_path, lambda: __version__) is None


def test_version_check_silent_offline(tmp_path):
    def boom():
        raise RuntimeError("offline")

    assert _run_check(tmp_path, boom) is None


def test_version_check_uses_fresh_cache_without_fetching(tmp_path):
    cache = tmp_path / "vc.json"
    cache.write_text(json.dumps({"checked_at": time.time(), "latest": "99.0.0"}))

    def boom():
        raise AssertionError("must not fetch with a fresh cache")

    notice = _run_check(tmp_path, boom)
    assert notice is not None and notice["latest"] == "99.0.0"


async def test_version_notice_attaches_once_and_fires_event(tmp_path):
    recorder = Recorder()
    telemetry = _telemetry(tmp_path, recorder)
    telemetry.set_consent(True)
    recorder.wait(1)
    _run_check(tmp_path, lambda: "99.0.0")

    server = build_server(_client(tmp_path), product_telemetry=telemetry)
    async with Client(server) as c:
        first = await c.call_tool("catalog.templates", {})
        second = await c.call_tool("catalog.templates", {})
    assert first.data["versionNotice"]["latest"] == "99.0.0"
    assert "versionNotice" not in second.data
    recorder.wait(4)
    assert "mcp_version_drift_warned" in recorder.names()


# ── fix pass ──────────────────────────────────────────────────────────────────


def test_version_check_env_off_makes_no_request(monkeypatch, tmp_path):
    monkeypatch.setenv("NIMBUS_MCP_VERSION_CHECK", "off")

    def boom():
        raise AssertionError("must not fetch when disabled")

    version_check.start_background_check(fetcher=boom, cache_path=tmp_path / "v.json")
    time.sleep(0.2)
    assert version_check._started is False
    assert version_check.pending_notice() is None


def test_install_id_is_race_safe(tmp_path):
    path = tmp_path / "state.json"
    ids: list[str] = []
    barrier = threading.Barrier(8)

    def worker():
        t = Telemetry(state_path=path, sender=lambda p: None, environ={})
        barrier.wait()
        ids.append(t._load()["install_id"])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(set(ids)) == 1
    assert json.loads(path.read_text())["install_id"] == ids[0]


def test_cli_on_reports_env_override(monkeypatch, capsys, tmp_path):
    from nimbus_mcp import cli, product_telemetry

    monkeypatch.setenv("NIMBUS_MCP_TELEMETRY", "off")
    product_telemetry.reset_telemetry_for_tests()
    assert cli.main(["telemetry", "on"]) == 0
    out = capsys.readouterr().out
    assert "overrides" in out and "effectively OFF" in out
    assert "Telemetry ON" not in out


def test_suite_never_touches_real_home_state():
    import os

    assert "NIMBUS_MCP_TELEMETRY_FILE" in os.environ
    from nimbus_mcp.product_telemetry import get_telemetry

    assert str(get_telemetry()._state_path) == os.environ["NIMBUS_MCP_TELEMETRY_FILE"]


def test_shipped_public_key_and_us_host():
    from nimbus_mcp import product_telemetry

    assert product_telemetry.DEFAULT_POSTHOG_KEY.startswith("phc_")
    assert product_telemetry.DEFAULT_POSTHOG_HOST == "https://us.i.posthog.com"


def test_default_transport_is_blocked_in_tests():
    t = Telemetry()
    with pytest.raises(AssertionError, match="real PostHog"):
        t._default_sender({})
