"""Optional Sentry init for the hosted gateway (``nimbus-mcp serve``).

The stdio/PyPI surface never initializes telemetry — this module is called
only by the gateway command, and only when the deployment provides
``SENTRY_DSN``. ``sentry-sdk`` rides the ``sentry`` extra, so a plain
``pip install nimbus-mcp`` has no SDK and the gateway simply runs
uninstrumented (returns False; never raises).
"""

from __future__ import annotations

import os


def init_gateway_sentry() -> bool:
    """Initialize Sentry for the gateway when configured. True when enabled."""
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        return False

    from . import __version__
    from .errors import McpToolError
    from .setup_mode import SetupRequired

    try:
        traces = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0") or 0)
    except ValueError:
        traces = 0.0

    sentry_sdk.init(
        dsn=dsn,
        environment=(
            os.environ.get("SENTRY_ENVIRONMENT", "").strip() or "production"
        ),
        release=f"nimbus-mcp@{__version__}",
        send_default_pii=False,
        traces_sample_rate=traces,
        # The Fly deployment runs with auto-stop: every idle stop SIGINTs the
        # process and the KeyboardInterrupt escaping asyncio.run() is a normal
        # shutdown, not an error. SetupRequired is the hosted gateway's
        # guidance path for tokenless client calls — expected, not an error.
        # McpToolError is the agent-facing tool result (bad arguments, a
        # rejected backend call). FastMCP logs it at error, which the logging
        # integration would otherwise open as an issue; the agent already
        # received the message. Unexpected exceptions (TypeError and friends)
        # stay.
        ignore_errors=[KeyboardInterrupt, SetupRequired, McpToolError],
    )
    return True
