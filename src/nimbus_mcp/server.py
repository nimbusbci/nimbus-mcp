"""FastMCP server assembly: one client, fifteen tool modules, stdio transport."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from . import __version__
from .client import NimbusClient, NullClient
from .config import DEFAULT_SERVE_HOST, load_config
from .setup_mode import SetupRequired
from .tools import (
    artifacts,
    bids,
    build,
    calibration,
    campaign,
    data,
    discovery,
    inspect,
    leaderboard,
    live,
    projects,
    python,
    run,
    telemetry,
    whoami,
)

INSTRUCTIONS = """Nimbus Studio BCI tools. Typical flows:
0) Auth: account.whoami() shows the account, plan, quota, and (token mode) token expiry;
   if a tool returns {ok: false, setupRequired: true}, walk the user through the
   options in that payload (nimbus-mcp login / desktop app / NIMBUS_MCP_KEY).
1) Explore: catalog.nodes / catalog.node_schema / catalog.templates /
   catalog.template / catalog.datasets;
   catalog.leaderboard ranks benchmark pipelines per dataset (meanAccuracyPct desc) — pick
   templates by ranking there.
1b) INSPECT BEFORE BUILDING: data.inspect_dataset(dataset, subject) or data.inspect_file(path)
   show channels, class balance, flatlined channels, band powers — class balance
   drives stratification; units are ASSUMED volts (a µV-native CSV reads 1e6x
   large; set unitsScale in custom_data config when needed).
2) Build: compose a train graph ({nodes: [{id, type, config}], connections: [{from, to}]}),
   pipeline.validate it, then execution.run (non-blocking) and poll execution.get.
3) Sweeps: experiment.run (1-25 paced runs, <=2 concurrent) then poll experiment.get for
   per-run status and aggregated metrics (mean/std/best).
4) Results: execution.results (kappa, ITR, confusion matrix),
   execution.artifacts / execution.download_artifact,
   pipeline.export for a standalone zip.
4b) Share/audit: bids.export_dataset / bids.export_execution produce BIDS-layout
   zips with sha256 manifests.
5) Live (use with care): device.list, device.test, then stream.start(confirm=true) only with
   the user's explicit go-ahead — it connects an EEG device to a human session. An idle
   watchdog stops abandoned sessions; polling stream.status keeps them alive.
6) Personal calibration: calibration.start(confirm=true) while the subject wears the device —
   the Studio app shows cues; calibration.status polls; calibration.train turns the recording
   into the subject's own classifier.
7) BYO model code: python.validate(script) statically checks a python_model class
   (fit/predict contract, no execution), then run it via execution.run with the code
   inline in the node config (runtime auto|local|cloud — desktop app: isolated local
   worker; hosted web app: ephemeral cloud sandbox with optional per-run pip
   requirements); leaderboard.submit scores a model class server-side (LOSO).
Note: expect filter/ASR warm-up periods and confidence to start low; signal quality matters."""


class SetupGuidanceMiddleware(Middleware):
    """Converts :class:`SetupRequired` into the structured guidance dict.

    Raised by ``NullClient`` (no credential resolved at startup — setup mode)
    and by ``NimbusClient`` when a hosted token is rejected mid-session (401
    expired/revoked). FastMCP wraps tool exceptions in ``ToolError``, so the
    original rides along as ``__cause__``; both shapes are handled here and no
    tool module needs to know setup mode exists.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext,  # noqa: ARG002 (required by the hook signature)
        call_next: CallNext,
    ) -> ToolResult:
        try:
            return await call_next(context)
        except SetupRequired as err:  # direct, in case fastmcp ever stops wrapping
            return ToolResult(structured_content=err.guidance)
        except ToolError as err:
            cause = err.__cause__ or err.__context__
            if isinstance(cause, SetupRequired):
                return ToolResult(structured_content=cause.guidance)
            raise


def build_server(client: NimbusClient | NullClient | None = None) -> FastMCP:
    if client is None:
        config = load_config()
        client = (
            NimbusClient(config)
            if (config.mcp_key or config.nimbus_token)
            else NullClient(config)  # setup mode: every tool returns guidance
        )
    # version= so the initialize handshake (serverInfo.version) reports THIS
    # package's version — FastMCP defaults to advertising its own.
    mcp = FastMCP("nimbus", version=__version__, instructions=INSTRUCTIONS)
    mcp.add_middleware(SetupGuidanceMiddleware())
    for module in (
        whoami,
        discovery,
        leaderboard,
        build,
        run,
        calibration,
        campaign,
        artifacts,
        bids,
        live,
        telemetry,
        data,
        inspect,
        projects,
        python,
    ):
        module.register(mcp, client)
    return mcp


def main() -> None:
    build_server().run()  # stdio transport


def serve(host: str = DEFAULT_SERVE_HOST, port: int = 8080, path: str = "/mcp") -> None:
    """Hosted gateway: streamable HTTP with per-request credentials.

    ``serve`` never resolves a process credential — the
    :class:`~nimbus_mcp.hosted.HeaderCredentialClient` reads each request's
    ``X-Nimbus-Token`` header, so one shared deployment serves many users as
    themselves (setup guidance for headerless calls). Binds loopback by
    default; passing ``0.0.0.0`` to expose the gateway is explicit opt-in and
    must be combined with those per-request tokens.
    """
    from .hosted import HeaderCredentialClient

    client = HeaderCredentialClient(load_config())
    build_server(client).run(transport="http", host=host, port=port, path=path)
