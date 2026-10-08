"""Hosted gateway: HeaderCredentialClient per-request credential resolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nimbus_mcp import hosted
from nimbus_mcp.config import McpConfig
from nimbus_mcp.hosted import HeaderCredentialClient, hosted_setup_guidance
from nimbus_mcp.setup_mode import SetupRequired, token_rejected_guidance


class FakeClient:
    """Records calls; optionally raises per-call."""

    def __init__(self, config: McpConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.raise_on_get: Exception | None = None

    def get(self, path, params=None):
        self.calls.append(("get", (path, params)))
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return {"ok": True, "token": self.config.nimbus_token}

    def close(self):
        self.calls.append(("close", ()))


def _config(token: str = "") -> McpConfig:
    return McpConfig(
        api_url="https://example.test",
        mcp_key="",
        export_dir=Path("/tmp/exports"),
        nimbus_token=token,
    )


def _client(
    headers: dict[str, str], config: McpConfig | None = None
) -> tuple[HeaderCredentialClient, list[FakeClient]]:
    made: list[FakeClient] = []

    def factory(cfg: McpConfig) -> FakeClient:
        client = FakeClient(cfg)
        made.append(client)
        return client

    proxy = HeaderCredentialClient(
        config if config is not None else _config(),
        client_factory=factory,  # type: ignore[arg-type]
        headers_getter=lambda: headers,
    )
    return proxy, made


def test_headerless_call_returns_hosted_guidance() -> None:
    proxy, made = _client({})
    with pytest.raises(SetupRequired) as err:
        proxy.get("/api/anything")
    assert made == []  # nothing built without a credential
    guidance = err.value.guidance
    assert guidance["setupRequired"] is True
    assert "X-Nimbus-Token" in guidance["message"]
    assert any("uvx nimbus-mcp" in o["action"] for o in guidance["options"])


def test_x_nimbus_token_header_builds_token_client() -> None:
    proxy, made = _client({"x-nimbus-token": "nimb_test_token"})
    result = proxy.get("/api/me/profile")
    assert result == {"ok": True, "token": "nimb_test_token"}
    assert len(made) == 1
    # The per-request client is pure Bearer mode: no local key, header source.
    assert made[0].config.nimbus_token == "nimb_test_token"
    assert made[0].config.mcp_key == ""
    assert made[0].config.credential_meta == {"source": "header"}
    # Base config survives for tool-side reads (export_dir etc.).
    assert proxy.config is not None


def test_authorization_bearer_form_accepted() -> None:
    proxy, made = _client({"authorization": "Bearer nimb_bearer_form"})
    assert proxy.get("/x") == {"ok": True, "token": "nimb_bearer_form"}
    assert len(made) == 1


def test_token_clients_are_cached_per_token() -> None:
    headers: dict[str, str] = {}
    made: list[FakeClient] = []

    def factory(cfg: McpConfig) -> FakeClient:
        client = FakeClient(cfg)
        made.append(client)
        return client

    proxy = HeaderCredentialClient(
        _config(), client_factory=factory, headers_getter=lambda: headers
    )
    headers["x-nimbus-token"] = "nimb_a"
    proxy.get("/1")
    proxy.get("/2")
    headers["x-nimbus-token"] = "nimb_b"
    proxy.get("/3")
    headers["x-nimbus-token"] = "nimb_a"
    proxy.get("/4")
    assert len(made) == 2  # one client per distinct token


def test_startup_env_token_is_fallback_for_headerless_calls() -> None:
    proxy, made = _client({}, config=_config(token="nimb_env_fallback"))
    assert proxy.get("/x") == {"ok": True, "token": "nimb_env_fallback"}
    # An explicit header still wins over the process credential.
    headers = {"x-nimbus-token": "nimb_header_wins"}
    made[0].config = replace(made[0].config, nimbus_token="nimb_header_wins")
    proxy2, _ = _client(headers, config=_config(token="nimb_env_fallback"))
    # (proxy2 builds its own; just assert resolution picks the header)
    assert proxy2._current_token() == "nimb_header_wins"


def test_rejected_token_translates_to_hosted_guidance() -> None:
    made: list[FakeClient] = []

    def factory(cfg: McpConfig) -> FakeClient:
        client = FakeClient(cfg)
        client.raise_on_get = SetupRequired(token_rejected_guidance("nimb_expired", 401))
        made.append(client)
        return client

    proxy = HeaderCredentialClient(
        _config(),
        client_factory=factory,  # type: ignore[arg-type]
        headers_getter=lambda: {"x-nimbus-token": "nimb_expired"},
    )
    with pytest.raises(SetupRequired) as err:
        proxy.get("/api/me/profile")
    guidance = err.value.guidance
    # Hosted flavor: the primary fix is the header (first option), and the
    # message frames the gateway context (login only appears as the LOCAL
    # alternative, which is fine — it just cannot be the only guidance).
    assert "X-Nimbus-Token" in guidance["message"]
    assert "gateway" in guidance["message"]
    assert "X-Nimbus-Token" in guidance["options"][0]["action"]
    # Situation extras survive the translation (rejected-with-days-left case
    # carries daysLeft; the undecodable case carries neither — both fine).
    assert guidance.get("daysLeft") is None or isinstance(guidance["daysLeft"], int)


def test_ensure_ready_enforces_header() -> None:
    proxy, _ = _client({})
    with pytest.raises(SetupRequired):
        proxy.ensure_ready()
    proxy2, _ = _client({"x-nimbus-token": "nimb_ok"})
    proxy2.ensure_ready()  # no raise


def test_client_cache_is_bounded() -> None:
    headers: dict[str, str] = {}
    proxy, made = _client(headers)
    proxy._max_clients = 3
    for i in range(5):
        headers["x-nimbus-token"] = f"nimb_{i}"
        proxy.get(f"/{i}")
    # cap 3 → the 4th build clears the map first: at most 3 alive at a time,
    # 5 tokens total → 5 builds but never more than 3 retained.
    assert len(made) == 5
    assert len(proxy._clients) <= 3


def test_hosted_guidance_shape_matches_spec() -> None:
    guidance = hosted_setup_guidance(daysExpired=2)
    assert guidance["ok"] is False
    assert guidance["setupRequired"] is True
    assert guidance["daysExpired"] == 2
    assert isinstance(guidance["options"], list) and guidance["options"]


# ── bound_client: workers outlive the request context ──────────────────────
# A background worker thread (experiment.run) has no MCP request context:
# get_http_headers() returns {} there, so per-request token resolution cannot
# run. bound_client() yields a token-bound client the worker owns for its
# lifetime — the header getter is never consulted inside the worker.

def test_bound_client_works_after_request_context_gone() -> None:
    """The bound client keeps working once the header getter goes empty (the
    worker-thread situation) and is closed when the context exits."""
    headers: dict[str, str] = {"x-nimbus-token": "nimb_bound"}
    proxy, made = _client(headers)

    with proxy.bound_client() as bound:
        headers.clear()  # request context gone — proxy resolution now fails
        with pytest.raises(SetupRequired):
            proxy.get("/api/noworker")  # sanity: the proxy itself is stranded
        assert bound.get("/api/worker") == {"ok": True, "token": "nimb_bound"}
        assert made, "bound client was built"
    # Worker-owned: closed on context exit (FakeClient records the call).
    assert any(call == ("close", ()) for client in made for call in client.calls)


def test_bound_client_headerless_raises_guidance() -> None:
    proxy, made = _client({})
    with pytest.raises(SetupRequired) as err:
        proxy.bound_client()
    assert made == []
    assert err.value.guidance["setupRequired"] is True


# ── eviction must not close clients other threads are using ────────────────

def test_eviction_retires_before_closing() -> None:
    """At the client cap, eviction drops cache references but closes only the
    set retired by the PREVIOUS eviction — a mid-request client from the
    current set survives; the retired set has a full cycle to drain."""
    headers: dict[str, str] = {}
    made: list[FakeClient] = []

    def factory(cfg: McpConfig) -> FakeClient:
        client = FakeClient(cfg)
        made.append(client)
        return client

    proxy = HeaderCredentialClient(
        _config(), client_factory=factory, headers_getter=lambda: headers, max_clients=2
    )
    headers["x-nimbus-token"] = "nimb_a"
    proxy.get("/1")
    headers["x-nimbus-token"] = "nimb_b"
    proxy.get("/2")
    a, b = made
    assert not a.calls or a.calls[-1] != ("close", ())

    headers["x-nimbus-token"] = "nimb_c"  # cap (2) reached → evict a, b
    proxy.get("/3")
    assert ("close", ()) not in a.calls and ("close", ()) not in b.calls  # retired, alive

    headers["x-nimbus-token"] = "nimb_d"  # cache {c} only — no eviction yet
    proxy.get("/4")
    assert ("close", ()) not in a.calls and ("close", ()) not in b.calls

    headers["x-nimbus-token"] = "nimb_e"  # cap reached again → close the retired a, b
    proxy.get("/5")
    assert ("close", ()) in a.calls and ("close", ()) in b.calls
    # Freshly cached clients from the previous eviction stay open.
    c = made[2]
    assert ("close", ()) not in c.calls


# ── the default header getter must expose `authorization` ──────────────────
# fastmcp's get_http_headers strips credential headers (authorization, cookie)
# by default; a plain default getter makes the Bearer fallback below
# unreachable on the real gateway. Pin the default against real fastmcp.

def test_default_headers_getter_includes_authorization() -> None:
    from fastmcp.server import dependencies as dep

    var = getattr(dep, "_current_http_request", None)
    if var is None or not hasattr(var, "set"):
        pytest.skip("fastmcp request ContextVar moved — re-pin this test")
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"authorization", b"Bearer nimb_live")],
        "query_string": b"",
        "scheme": "http",
    }
    token = var.set(Request(scope))
    try:
        # fastmcp's documented default: credential headers stripped …
        assert dep.get_http_headers().get("authorization") is None
        # … so the gateway's default getter must request them explicitly.
        headers = hosted._default_headers_getter()
    finally:
        var.reset(token)
    assert headers.get("authorization") == "Bearer nimb_live"


def test_default_headers_getter_falls_back_when_include_missing(monkeypatch) -> None:
    """fastmcp <2.10 has no `include` kwarg: the getter degrades to the plain
    call (X-Nimbus-Token resolution unaffected) instead of crashing."""
    calls: list[dict[str, object]] = []

    def fake(**kwargs: object) -> dict[str, str]:
        calls.append(kwargs)
        if "include" in kwargs:
            raise TypeError("unexpected keyword argument 'include'")
        return {"x-nimbus-token": "nimb_ok"}

    monkeypatch.setattr(hosted, "get_http_headers", fake)
    assert hosted._default_headers_getter() == {"x-nimbus-token": "nimb_ok"}
    assert calls == [{"include": {"authorization"}}, {}]  # retried plain
