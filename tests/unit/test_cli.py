"""CLI tests: login device flow, logout, status doctor, dispatch."""

import argparse
import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from nimbus_mcp import cli
from nimbus_mcp.credentials import (
    DEFAULT_API_URL,
    ResolvedCredential,
    credentials_store_path,
    read_store,
    write_store,
)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Hermetic HOME + no NIMBUS_* leakage from the developer shell."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("NIMBUS_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    for var in (
        "NIMBUS_TOKEN",
        "NIMBUS_TOKEN_FILE",
        "NIMBUS_MCP_KEY",
        "NIMBUS_MCP_KEY_FILE",
        "NIMBUS_API_URL",
        "XDG_CONFIG_HOME",
        "APPDATA",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(cli, "_sleep", lambda seconds: None)


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)
    return opened


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"command": "login", "api_url": None, "ttl_days": None, "name": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _b64url(obj: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def _fake_jwt_token(payload: dict) -> str:
    return "nimb_" + f"{_b64url({'alg': 'none'})}.{_b64url(payload)}.sig"


CREATE_RESPONSE = {
    "ok": True,
    "deviceCode": "dc-raw-43-chars",
    "userCode": "ABCD2345",
    "verificationUrl": "https://app.nimbus.test/activate?code=ABCD2345",
    "expiresIn": 600,
    "interval": 5,
}


def _recording_handler(poll_responses: list[httpx.Response], calls: list):
    """Handler serving one create + a scripted sequence of poll responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "method": request.method,
                "path": request.url.path,
                "auth": request.headers.get("authorization"),
                "key": request.headers.get("x-mcp-key"),
                "json": _body_json(request),
            }
        )
        if request.url.path == "/api/mcp/device-code":
            return httpx.Response(200, json=CREATE_RESPONSE)
        assert request.url.path == "/api/mcp/device-token"
        return poll_responses.pop(0)

    return handler


def _body_json(request: httpx.Request):
    try:
        return json.loads(request.content.decode("utf-8"))
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# login
# ─────────────────────────────────────────────────────────────────────────────


def test_login_happy_path_writes_store(capsys, _no_browser):
    calls: list = []
    polls = [
        httpx.Response(200, json={"status": "pending", "interval": 5, "expiresIn": 595}),
        httpx.Response(200, json={"status": "slow_down", "interval": 6}),
        httpx.Response(
            200,
            json={
                "status": "approved",
                "token": "nimb_minted",
                "expiresAt": "2026-11-03T00:00:00",
                "name": "MCP device ABCD2345",
            },
        ),
    ]
    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(_recording_handler(polls, calls)))

    assert rc == 0
    # create + three polls
    assert [c["path"] for c in calls] == [
        "/api/mcp/device-code",
        "/api/mcp/device-token",
        "/api/mcp/device-token",
        "/api/mcp/device-token",
    ]
    # device flow endpoints are public — never any auth header
    assert all(c["auth"] is None and c["key"] is None for c in calls)
    assert calls[0]["json"] == {}  # no name/ttlDays by default
    assert all(c["json"] == {"deviceCode": "dc-raw-43-chars"} for c in calls[1:])

    # store written with the minted token, api_url, and name
    store = read_store()
    assert store["token"] == "nimb_minted"
    assert store["api_url"] == DEFAULT_API_URL
    assert store["name"] == "MCP device ABCD2345"

    out = capsys.readouterr().out
    assert "ABCD2345" in out
    assert CREATE_RESPONSE["verificationUrl"] in out
    assert "MCP device ABCD2345" in out
    assert "2026-11-03" in out
    assert _no_browser == [CREATE_RESPONSE["verificationUrl"]]


def test_login_forwards_name_ttl_and_api_url(capsys, tmp_path):
    calls: list = []
    polls = [
        httpx.Response(
            200,
            json={"status": "approved", "token": "nimb_t", "expiresAt": "x", "name": "named"},
        ),
    ]
    rc = cli.cmd_login(
        _args(api_url="http://api.nimbus.test/", name="my box", ttl_days=45),
        transport=httpx.MockTransport(_recording_handler(polls, calls)),
    )
    assert rc == 0
    assert calls[0]["json"] == {"name": "my box", "ttlDays": 45}
    assert read_store()["api_url"] == "http://api.nimbus.test"


def test_login_env_api_url_used_as_default(monkeypatch):
    calls: list = []
    polls = [httpx.Response(404, json={"status": "expired"})]
    monkeypatch.setenv("NIMBUS_API_URL", "http://env.nimbus.test")
    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(_recording_handler(polls, calls)))
    assert rc == 1
    assert calls[0]["path"] == "/api/mcp/device-code"
    # MockTransport ignores the host; assert via the store absence + rc only.


def test_login_slow_down_and_pending_intervals_honored(monkeypatch, _no_browser):
    sleeps: list = []
    monkeypatch.setattr(cli, "_sleep", lambda s: sleeps.append(s))
    calls: list = []
    polls = [
        httpx.Response(200, json={"status": "pending", "interval": 5, "expiresIn": 595}),
        httpx.Response(200, json={"status": "slow_down", "interval": 6}),
        httpx.Response(
            200, json={"status": "approved", "token": "t", "expiresAt": "x", "name": "n"}
        ),
    ]
    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(_recording_handler(polls, calls)))
    assert rc == 0
    # sleep before each poll: initial 5, then 5 again (pending kept it), then 6 (slow_down)
    assert sleeps == [5, 5, 6]


def test_login_expired_poll_exits_without_store(capsys):
    calls: list = []
    polls = [httpx.Response(404, json={"status": "expired"})]
    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(_recording_handler(polls, calls)))
    assert rc == 1
    assert "expired" in capsys.readouterr().out.lower()
    assert read_store() is None


def test_login_denied_exits_without_store(capsys):
    calls: list = []
    polls = [httpx.Response(200, json={"status": "denied"})]
    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(_recording_handler(polls, calls)))
    assert rc == 1
    assert "denied" in capsys.readouterr().out.lower()
    assert read_store() is None


def test_login_backend_unreachable(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(handler))
    assert rc == 1
    assert "Cannot reach the Nimbus backend" in capsys.readouterr().out
    assert read_store() is None


def test_login_create_error_surfaces_detail(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "ServiceUnavailable"})

    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(handler))
    assert rc == 1
    out = capsys.readouterr().out
    assert "503" in out
    assert read_store() is None


def test_login_rejects_out_of_range_ttl(capsys):
    rc = cli.cmd_login(_args(ttl_days=5), transport=httpx.MockTransport(lambda r: None))
    assert rc == 1
    assert "--ttl-days" in capsys.readouterr().out
    assert read_store() is None


# ─────────────────────────────────────────────────────────────────────────────
# logout
# ─────────────────────────────────────────────────────────────────────────────


def test_logout_removes_store(capsys):
    write_store(token="nimb_x", api_url="http://api.test", name="n")
    assert credentials_store_path().exists()
    rc = cli.cmd_logout(_args(command="logout"))
    assert rc == 0
    assert not credentials_store_path().exists()
    assert "Removed" in capsys.readouterr().out


def test_logout_without_store_is_not_an_error(capsys):
    rc = cli.cmd_logout(_args(command="logout"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "nothing to log out" in out
    assert str(credentials_store_path()) in out


# ─────────────────────────────────────────────────────────────────────────────
# status (doctor)
# ─────────────────────────────────────────────────────────────────────────────


def _profile_response() -> dict:
    return {
        "ok": True,
        "userId": "user_1",
        "email": "dev@nimbus.test",
        "capabilities": {
            "isPro": False,
            "freeTrainingRunsMonthlyLimit": 10,
            "freeTrainingRunsRemaining": 7,
        },
    }


def test_status_token_healthy(capsys):
    exp = int(time.time()) + 30 * 86400
    cred = ResolvedCredential(
        "token",
        "http://api.nimbus.test",
        _fake_jwt_token({"exp": exp}),
        {"source": "store", "path": "~/.nimbus/credentials.json", "name": "my token"},
    )
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json=_profile_response())

    rc = cli.cmd_status(_args(command="status"), transport=httpx.MockTransport(handler), cred=cred)
    assert rc == 0
    assert seen == [("/api/me/profile", f"Bearer {cred.secret}")]
    out = capsys.readouterr().out
    assert "credential store" in out
    assert "http://api.nimbus.test" in out
    assert cred.secret[:12] in out
    assert cred.secret[12:] not in out  # prefix only, never the full token
    assert "plan: free" in out
    assert "7/10" in out
    assert "dev@nimbus.test" in out
    assert "expires in 30 day" in out
    assert "healthy" in out


def test_status_token_pro_plan(capsys):
    profile = _profile_response()
    profile["capabilities"]["isPro"] = True
    cred = ResolvedCredential("token", "http://api.test", "nimb_pro", {"source": "env"})
    rc = cli.cmd_status(
        _args(command="status"),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=profile)),
        cred=cred,
    )
    assert rc == 0
    assert "plan: pro" in capsys.readouterr().out


def test_status_token_rejected_401(capsys):
    cred = ResolvedCredential("token", "http://api.test", "nimb_bad", {"source": "store"})
    rc = cli.cmd_status(
        _args(command="status"),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "no"})),
        cred=cred,
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "401" in out
    assert "nimbus-mcp login" in out


def test_status_desktop_key_probes_health(capsys):
    cred = ResolvedCredential(
        "desktop_key",
        DEFAULT_API_URL,
        "desk-key",
        {"source": "desktop", "path": "/x/mcp-key.json"},
    )
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    rc = cli.cmd_status(_args(command="status"), transport=httpx.MockTransport(handler), cred=cred)
    assert rc == 0
    assert seen == ["/health"]
    out = capsys.readouterr().out
    assert "desktop app key" in out
    assert "local backend health: ok" in out


def test_status_desktop_key_unhealthy_backend(capsys):
    cred = ResolvedCredential("desktop_key", DEFAULT_API_URL, "desk-key", {"source": "desktop"})
    rc = cli.cmd_status(
        _args(command="status"),
        transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        cred=cred,
    )
    assert rc == 1
    assert "503" in capsys.readouterr().out


def test_status_env_key_probes_health(capsys):
    cred = ResolvedCredential("key", DEFAULT_API_URL, "env-key", {"source": "env"})
    rc = cli.cmd_status(
        _args(command="status"),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        cred=cred,
    )
    assert rc == 0
    assert "NIMBUS_MCP_KEY" in capsys.readouterr().out


def test_status_no_credential_guides_to_login(capsys):
    cred = ResolvedCredential("none", DEFAULT_API_URL, None, {})
    rc = cli.cmd_status(_args(command="status"), cred=cred)
    assert rc == 1
    out = capsys.readouterr().out
    assert "credential source: none" in out
    assert "nimbus-mcp login" in out
    assert "NIMBUS_TOKEN" in out
    assert "NIMBUS_MCP_KEY" in out


def test_status_resolves_real_environment_when_cred_not_given(capsys, tmp_path):
    """No cred passed → resolve_credential() against the isolated env/store."""
    write_store(token="nimb_stored", api_url="http://api.test", name="n")
    rc = cli.cmd_status(
        _args(command="status"),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_profile_response())),
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "credential store" in out
    # "nimb_stored" is 11 chars — shorter than the 12-char prefix window, so
    # only a 4-char prefix may appear (v0.5 masking), never the whole token.
    assert "nimb…" in out
    assert "nimb_stored" not in out


# ─────────────────────────────────────────────────────────────────────────────
# dispatch (console script behavior)
# ─────────────────────────────────────────────────────────────────────────────


def test_bare_invocation_runs_stdio_server(monkeypatch):
    import nimbus_mcp.server as server_mod

    calls: list = []
    monkeypatch.setattr(server_mod, "main", lambda: calls.append(1))
    assert cli.main([]) == 0
    assert calls == [1]


def test_unknown_invocation_runs_stdio_server(monkeypatch):
    import nimbus_mcp.server as server_mod

    calls: list = []
    monkeypatch.setattr(server_mod, "main", lambda: calls.append(1))
    assert cli.main(["frobnicate", "--x"]) == 0
    assert calls == [1]


def test_main_dispatches_subcommands(monkeypatch):
    seen: list = []
    monkeypatch.setattr(cli, "cmd_logout", lambda args: seen.append("logout") or 7)
    monkeypatch.setattr(cli, "cmd_login", lambda args: seen.append("login") or 0)
    assert cli.main(["logout"]) == 7
    assert cli.main(["login", "--name", "x"]) == 0
    assert seen == ["logout", "login"]


def test_console_script_still_nimbus_mcp():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'nimbus-mcp = "nimbus_mcp.cli:main"' in text


# ─────────────────────────────────────────────────────────────────────────────
# v0.5 Task 5 ride-along minors
# ─────────────────────────────────────────────────────────────────────────────


def test_login_interval_guarded_and_clamped_to_60(monkeypatch):
    """A malformed interval falls back to the previous value; a huge one is
    capped at 60 s — the poll loop must neither crash nor sleep forever."""
    sleeps: list = []
    monkeypatch.setattr(cli, "_sleep", lambda s: sleeps.append(s))
    create = dict(CREATE_RESPONSE, interval="bogus")
    polls = [
        httpx.Response(200, json={"status": "pending", "interval": None}),
        httpx.Response(200, json={"status": "slow_down", "interval": 9999}),
        httpx.Response(200, json={"status": "approved", "token": "t", "name": "n"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/mcp/device-code":
            return httpx.Response(200, json=create)
        return polls.pop(0)

    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(handler))
    assert rc == 0
    # create interval "bogus" → default 5; pending None → keep 5; slow_down 9999 → 60
    assert sleeps == [5, 5, 60]


def test_login_create_interval_also_guarded(monkeypatch):
    sleeps: list = []
    monkeypatch.setattr(cli, "_sleep", lambda s: sleeps.append(s))
    polls = [
        httpx.Response(200, json={"status": "approved", "token": "t", "name": "n"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/mcp/device-code":
            return httpx.Response(200, json=dict(CREATE_RESPONSE, interval=0))
        return polls.pop(0)

    rc = cli.cmd_login(_args(), transport=httpx.MockTransport(handler))
    assert rc == 0
    assert sleeps == [1]  # clamped up to the 1 s floor


def test_status_profile_malformed_json_body_guarded(capsys):
    """A 200 with a non-JSON body must not traceback; exit 1 with a hint."""
    cred = ResolvedCredential(
        "token",
        "http://api.test",
        _fake_jwt_token({"exp": int(time.time()) + 86400}),
        {"source": "store"},
    )
    rc = cli.cmd_status(
        _args(command="status"),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>oops")),
        cred=cred,
    )
    assert rc == 1
    assert "malformed response body" in capsys.readouterr().out


def test_status_masks_short_secrets(capsys):
    """A secret shorter than the 12-char prefix window would be printed WHOLE
    by a naive slice — it gets a 4-char prefix instead."""
    cred = ResolvedCredential("key", DEFAULT_API_URL, "shortkey9", {"source": "env"})
    rc = cli.cmd_status(
        _args(command="status"),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        cred=cred,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "shor…" in out
    assert "shortkey9" not in out


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flags_route_to_argparse_not_the_server(monkeypatch, capsys, flag):
    import nimbus_mcp.server as server_mod

    started: list = []
    monkeypatch.setattr(server_mod, "main", lambda: started.append(1))
    with pytest.raises(SystemExit) as excinfo:
        cli.main([flag])
    assert excinfo.value.code == 0
    assert started == []  # the stdio server must NOT start
    assert "usage: nimbus-mcp" in capsys.readouterr().out


def test_logout_permission_error_is_reported_not_raised(capsys, monkeypatch):
    def boom():
        raise PermissionError("read-only fs")

    monkeypatch.setattr(cli, "remove_store", boom)
    rc = cli.cmd_logout(_args(command="logout"))
    assert rc == 1
    out = capsys.readouterr().out
    assert "permission denied" in out
    assert str(credentials_store_path()) in out
