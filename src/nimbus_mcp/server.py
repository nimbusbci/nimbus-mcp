"""FastMCP server assembly: one client, twelve tool modules, stdio transport."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from .client import NimbusClient, NullClient
from .config import load_config
from .setup_mode import SetupRequired
from .tools import (
    artifacts,
    build,
    campaign,
    data,
    discovery,
    inspect,
    leaderboard,
    live,
    projects,
    run,
    telemetry,
    whoami,
)

INSTRUCTIONS = """Nimbus Studio BCI tools. Typical flows:
0) Auth: whoami() shows the account, plan, quota, and (token mode) token expiry;
   if a tool returns {ok: false, setupRequired: true}, walk the user through the
   options in that payload (nimbus-mcp login / desktop app / NIMBUS_MCP_KEY).
1) Explore: list_nodes / get_node_schema / list_templates / get_template / list_datasets;
   get_leaderboard ranks benchmark pipelines per dataset (meanAccuracyPct desc) — pick
   templates by ranking there.
1b) INSPECT BEFORE BUILDING: inspect_dataset(dataset, subject) or inspect_file(path)
   show channels, class balance, flatlined channels, band powers — class balance
   drives stratification; units are ASSUMED volts (a µV-native CSV reads 1e6x
   large; set unitsScale in custom_data config when needed).
2) Build: compose a train graph ({nodes: [{id, type, config}], connections: [{from, to}]}),
   validate_pipeline it, then run_pipeline (non-blocking) and poll get_execution.
3) Sweeps: run_experiment (1-25 paced runs, <=2 concurrent) then poll get_experiment for
   per-run status and aggregated metrics (mean/std/best).
4) Results: get_results (kappa, ITR, confusion matrix), list_artifacts / download_artifact,
   export_python for a standalone zip.
5) Live (use with care): list_devices, test_device, then start_stream(confirm=true) only with
   the user's explicit go-ahead — it connects an EEG device to a human session. An idle
   watchdog stops abandoned sessions; polling stream_status keeps them alive.
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
    mcp = FastMCP("nimbus", instructions=INSTRUCTIONS)
    mcp.add_middleware(SetupGuidanceMiddleware())
    for module in (
        whoami,
        discovery,
        leaderboard,
        build,
        run,
        campaign,
        artifacts,
        live,
        telemetry,
        data,
        inspect,
        projects,
    ):
        module.register(mcp, client)
    return mcp


def main() -> None:
    build_server().run()  # stdio transport
