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
    telemetry._update(consent="granted")  # stored grant must not beat env off
    telemetry.capture("tool_completed", {"tool": "x"})
    time.sleep(0.2)
    assert recorder.payloads == []
    assert telemetry.pending_notice() is None


# ── event payloads: allowlisted + content-free ───────────────────────────────


async def test_tool_completed_payload_is_allowlisted(tmp_path):
    recorder = Recorder()
    telemetry = _telemetry(tmp_path, recorder)
    telemetry.set_consent(True)
    # set_consent sends first_run synchronously (B1): no wait needed.
    assert recorder.names() == ["mcp_first_run"]  # first_run precedes everything

    server = build_server(_client(tmp_path), product_telemetry=telemetry)
    async with Client(server) as c:
        await c.call_tool("catalog.templates", {})
    recorder.wait(2)
    time.sleep(0.1)
    names = recorder.names()
    # discovery is not a result (B3): no first_local_result for it
    assert "mcp_first_local_result" not in names
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


def test_cli_consent_delivers_first_run_before_returning(tmp_path):
    """B1: the CLI process exits right after set_consent — the send must
    already have happened (no daemon thread left to die)."""
    recorder = Recorder()
    telemetry = _telemetry(tmp_path, recorder)
    telemetry.set_consent(True)
    assert recorder.names() == ["mcp_first_run"]
    state = json.loads((tmp_path / "telemetry-state.json").read_text())
    assert state["first_run_sent"] is True


def test_failed_first_run_is_not_marked_sent_and_retries(tmp_path):
    attempts: list[dict] = []

    def failing(payload: dict) -> None:
        attempts.append(payload)
        raise OSError("offline")

    t1 = Telemetry(
        state_path=tmp_path / "telemetry-state.json",
        sender=failing,
        environ={"NIMBUS_MCP_POSTHOG_KEY": "phc_test"},
    )
    t1.set_consent(True)
    assert [p["event"] for p in attempts] == ["mcp_first_run"]
    state = json.loads((tmp_path / "telemetry-state.json").read_text())
    assert not state.get("first_run_sent")
    # Same process: one attempt only, no retry storm on every tool call.
    t1.capture("tool_completed", {"tool": "x"}, sync=True)
    t1.capture("tool_completed", {"tool": "y"}, sync=True)
    assert [p["event"] for p in attempts].count("mcp_first_run") == 1

    recorder = Recorder()
    t2 = _telemetry(tmp_path, recorder)  # next session, network back
    t2.capture("tool_completed", {"tool": "x"})
    recorder.wait(2)
    assert recorder.names().count("mcp_first_run") == 1
    deadline = time.time() + 2
    while time.time() < deadline:
        if json.loads((tmp_path / "telemetry-state.json").read_text()).get("first_run_sent"):
            break
        time.sleep(0.02)
    assert json.loads((tmp_path / "telemetry-state.json").read_text())["first_run_sent"] is True


def test_first_local_result_only_for_real_results(tmp_path):
    """B3: whoami / discovery successes are not a 'first local result'."""
    recorder = Recorder()
    telemetry = _telemetry(tmp_path, recorder)
    telemetry.set_consent(True)
    for tool in ("account.whoami", "catalog.templates", "project.list", "python.validate"):
        telemetry.tool_completed(
            tool=tool, mode="anonymous", success=True, error_code=None, duration_s=0.1
        )
    recorder.wait(5)
    time.sleep(0.1)
    assert "mcp_first_local_result" not in recorder.names()
    telemetry.tool_completed(
        tool="stats.grouped", mode="anonymous", success=True, error_code=None, duration_s=0.1
    )
    recorder.wait(7)
    first = [p for p in recorder.payloads if p["event"] == "mcp_first_local_result"]
    assert len(first) == 1 and first[0]["properties"]["mode"] == "anonymous"


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


# ── N1: delivery means HTTP 2xx, not "no exception" ─────────────────────────


def _capture_server(status: int):
    """Real local capture endpoint answering ``status`` to every POST."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    hits: list[str] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("content-length") or 0)
            hits.append(json.loads(self.rfile.read(length))["event"])
            self.send_response(status)
            self.send_header("content-length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):  # silence
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


# The real httpx sender, captured before conftest swaps in its no-network
# guard; these tests only ever point it at a 127.0.0.1 server.
_REAL_SENDER = Telemetry._default_sender


def _real_sender_telemetry(tmp_path: Path, port: int) -> Telemetry:
    t = Telemetry(
        state_path=tmp_path / "telemetry-state.json",
        environ={
            "NIMBUS_MCP_POSTHOG_KEY": "phc_test",
            "NIMBUS_MCP_POSTHOG_HOST": f"http://127.0.0.1:{port}",
        },
    )
    t._sender = lambda payload: _REAL_SENDER(t, payload)
    return t


def _state(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "telemetry-state.json").read_text())


def test_first_run_500_is_not_marked_sent_and_retries_next_process(tmp_path):
    srv, hits = _capture_server(500)
    try:
        t1 = _real_sender_telemetry(tmp_path, srv.server_address[1])
        t1.set_consent(True)  # sync first_run → 500
        assert hits == ["mcp_first_run"]
        assert not _state(tmp_path).get("first_run_sent")
        t1.capture("tool_completed", {"tool": "x"}, sync=True)
        assert hits.count("mcp_first_run") == 1  # once per process
    finally:
        srv.shutdown()

    srv, hits = _capture_server(200)
    try:
        t2 = _real_sender_telemetry(tmp_path, srv.server_address[1])
        t2.capture("tool_completed", {"tool": "x"}, sync=True)
        assert hits.count("mcp_first_run") == 1
        deadline = time.time() + 3
        while time.time() < deadline and not _state(tmp_path).get("first_run_sent"):
            time.sleep(0.02)
        assert _state(tmp_path)["first_run_sent"] is True
    finally:
        srv.shutdown()


@pytest.mark.parametrize("status", [400, 429, 500, 503])
def test_default_sender_treats_non_2xx_as_failure(tmp_path, status):
    srv, _ = _capture_server(status)
    try:
        t = _real_sender_telemetry(tmp_path, srv.server_address[1])
        results: list[bool] = []
        t._safe_send({"event": "mcp_tool_completed"}, results.append)
        assert results == [False]
    finally:
        srv.shutdown()


def test_default_sender_2xx_is_delivered(tmp_path):
    srv, _ = _capture_server(200)
    try:
        t = _real_sender_telemetry(tmp_path, srv.server_address[1])
        results: list[bool] = []
        t._safe_send({"event": "mcp_tool_completed"}, results.append)
        assert results == [True]
    finally:
        srv.shutdown()


def test_failed_first_local_result_is_not_marked_sent_and_retries(tmp_path):
    attempts: list[str] = []

    def failing(payload: dict) -> None:
        attempts.append(payload["event"])
        raise OSError("HTTP 500")

    t1 = Telemetry(
        state_path=tmp_path / "telemetry-state.json",
        sender=failing,
        environ={"NIMBUS_MCP_POSTHOG_KEY": "phc_test"},
    )
    t1.set_consent(True)
    t1.capture("first_local_result", {"mode": "local"}, sync=True)
    t1.capture("first_local_result", {"mode": "local"}, sync=True)
    assert attempts.count("mcp_first_local_result") == 1  # once per process
    assert not _state(tmp_path).get("first_local_result_sent")

    recorder = Recorder()
    t2 = _telemetry(tmp_path, recorder)
    t2.capture("first_local_result", {"mode": "local"}, sync=True)
    assert "mcp_first_local_result" in recorder.names()
    assert _state(tmp_path)["first_local_result_sent"] is True
    t2.capture("first_local_result", {"mode": "local"}, sync=True)
    assert recorder.names().count("mcp_first_local_result") == 1


def test_default_sender_treats_3xx_as_failure(tmp_path):
    srv, _ = _capture_server(302)
    try:
        t = _real_sender_telemetry(tmp_path, srv.server_address[1])
        results: list[bool] = []
        t._safe_send({"event": "mcp_tool_completed"}, results.append)
        assert results == [False]
    finally:
        srv.shutdown()


# ── H1: the state file is shared across processes ───────────────────────────


def test_cli_off_stops_running_server_and_file_stays_denied(tmp_path):
    server_sent: list[str] = []
    path = tmp_path / "telemetry-state.json"
    env = {"NIMBUS_MCP_POSTHOG_KEY": "phc_test"}
    server = Telemetry(
        state_path=path, sender=lambda p: server_sent.append(p["event"]), environ=env
    )
    server.set_consent(True)  # server process has cached 'granted'
    server.capture("tool_completed", {"tool": "x"}, sync=True)
    before = len(server_sent)

    cli = Telemetry(state_path=path, sender=lambda p: None, environ=env)
    cli.set_consent(False)  # `nimbus-mcp telemetry off` in another process

    server.capture("tool_completed", {"tool": "y"}, sync=True)
    server.capture("first_local_result", {"mode": "local"}, sync=True)
    assert len(server_sent) == before  # no further sends
    assert not server.enabled()
    assert _state(tmp_path)["consent"] == "denied"


def test_inflight_delivery_callback_never_rewrites_consent(tmp_path):
    """A send that completes AFTER the CLI turned telemetry off must only add
    its *_sent flag — never write the server's stale 'granted' back."""
    path = tmp_path / "telemetry-state.json"
    env = {"NIMBUS_MCP_POSTHOG_KEY": "phc_test"}
    server = Telemetry(state_path=path, sender=lambda p: None, environ=env)
    server.set_consent(True)
    Telemetry(state_path=path, sender=lambda p: None, environ=env).set_consent(False)
    server._first_local_result_done(True)
    state = _state(tmp_path)
    assert state["consent"] == "denied"
    assert state["first_local_result_sent"] is True


def test_sent_flags_from_two_processes_are_both_preserved(tmp_path):
    path = tmp_path / "telemetry-state.json"
    env = {"NIMBUS_MCP_POSTHOG_KEY": "phc_test"}
    a = Telemetry(state_path=path, sender=lambda p: None, environ=env)
    a.set_consent(True)  # a: first_run_sent
    b = Telemetry(state_path=path, sender=lambda p: None, environ=env)
    b._load()  # b caches state now
    a._first_local_result_done(True)  # a writes after b loaded
    b._update(notice_shown=True)  # b's later write must not drop a's flag
    state = _state(tmp_path)
    assert state["first_run_sent"] is True
    assert state["first_local_result_sent"] is True
    assert state["notice_shown"] is True
    assert a._load()["install_id"] == b._load()["install_id"] == state["install_id"]
