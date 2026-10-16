"""Gateway Sentry init: deployment opt-in, stdio never instruments.

``sentry-sdk`` is optional (the ``sentry`` extra) and the tests do not
install it — the DSN-present case injects a fake ``sentry_sdk`` module into
``sys.modules`` so the wiring is verifiable without the dependency.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

from nimbus_mcp.observability import init_gateway_sentry


def _install_fake_sentry_sdk() -> types.ModuleType:
    module = types.ModuleType("sentry_sdk")
    module.init = mock.Mock(name="sentry_sdk.init")
    sys.modules["sentry_sdk"] = module
    return module


def test_no_dsn_is_a_silent_no_op(monkeypatch) -> None:
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert init_gateway_sentry() is False


def test_missing_sdk_is_a_silent_no_op(monkeypatch) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://key@o0.ingest.sentry.io/1")
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)  # ImportError on import
    assert init_gateway_sentry() is False


def test_dsn_initializes_sdk_with_gateway_defaults(monkeypatch) -> None:
    fake = _install_fake_sentry_sdk()
    monkeypatch.setenv("SENTRY_DSN", "https://key@o0.ingest.sentry.io/1")
    monkeypatch.delenv("SENTRY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("SENTRY_TRACES_SAMPLE_RATE", raising=False)
    try:
        assert init_gateway_sentry() is True
    finally:
        del sys.modules["sentry_sdk"]
    fake.init.assert_called_once()
    kwargs = fake.init.call_args.kwargs
    assert kwargs["dsn"] == "https://key@o0.ingest.sentry.io/1"
    assert kwargs["environment"] == "production"
    assert kwargs["release"].startswith("nimbus-mcp@")
    assert kwargs["send_default_pii"] is False
    assert kwargs["traces_sample_rate"] == 0.0


def test_expected_gateway_conditions_are_ignored(monkeypatch) -> None:
    """Fly auto-stop SIGINTs the gateway on every idle cycle (the escaping
    KeyboardInterrupt is a normal shutdown), and tokenless hosted calls raise
    SetupRequired guidance — neither is a Sentry error."""
    fake = _install_fake_sentry_sdk()
    monkeypatch.setenv("SENTRY_DSN", "https://key@o0.ingest.sentry.io/1")
    try:
        assert init_gateway_sentry() is True
    finally:
        del sys.modules["sentry_sdk"]
    kwargs = fake.init.call_args.kwargs
    assert KeyboardInterrupt in kwargs["ignore_errors"]

    from nimbus_mcp.setup_mode import SetupRequired

    assert SetupRequired in kwargs["ignore_errors"]
