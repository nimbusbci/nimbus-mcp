"""Suite-wide isolation from the developer's real ~/.nimbus state.

Telemetry consent / install id and the version-check cache would otherwise
be read from (and written to) the real home directory by any test that
builds the server without injecting its own Telemetry.
"""

import pytest

from nimbus_mcp import product_telemetry, version_check


@pytest.fixture(autouse=True)
def _isolate_nimbus_state(tmp_path_factory, monkeypatch):
    state_dir = tmp_path_factory.mktemp("nimbus-state")
    monkeypatch.setenv("NIMBUS_MCP_TELEMETRY_FILE", str(state_dir / "telemetry-state.json"))
    monkeypatch.setenv("NIMBUS_MCP_VERSION_CHECK_FILE", str(state_dir / "version-check.json"))
    monkeypatch.delenv("NIMBUS_MCP_TELEMETRY", raising=False)
    monkeypatch.delenv("NIMBUS_MCP_VERSION_CHECK", raising=False)
    # The real public key now ships: no test may ever reach PostHog or
    # PyPI. Default transports are replaced with hard failures (tests that
    # exercise capture inject their own recording sender/fetcher).
    def _no_network_sender(self, payload):
        raise AssertionError("test attempted a real PostHog capture")

    def _no_network_fetch():
        raise AssertionError("test attempted a real PyPI version check")

    monkeypatch.setattr(product_telemetry.Telemetry, "_default_sender", _no_network_sender)
    monkeypatch.setattr(version_check, "_default_fetcher", _no_network_fetch)
    product_telemetry.reset_telemetry_for_tests()
    version_check.reset_for_tests()
    yield
    product_telemetry.reset_telemetry_for_tests()
    version_check.reset_for_tests()
