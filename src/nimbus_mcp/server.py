"""FastMCP server assembly: one client, nineteen tool modules, stdio transport."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from . import __version__, version_check
from .client import NimbusClient, NullClient
from .config import DEFAULT_SERVE_HOST, load_config
from .errors import McpToolError
from .product_telemetry import Telemetry, get_telemetry
from .setup_mode import SetupRequired
from .tools import (
    artifacts,
    bids,
    build,
    calibration,
    campaign,
    data,
    discovery,
    foundation,
    inspect,
    leaderboard,
    live,
    plots,
    projects,
    protocol,
    python,
    run,
    stats,
    telemetry,
    whoami,
)

INSTRUCTIONS = """Nimbus Studio BCI tools. Typical flows:
0) Auth: Local Free needs NO Nimbus Studio login — templates, own data, local
   execution/plots/stats, BIDS export and local storage work anonymously against
   the local backend (desktop app / dev backend). account.whoami() shows the mode,
   account, plan, quota, and (token mode) token expiry. Login is requested ONLY at
   first cloud value: a tool returning {ok: false, loginRequired: true, trigger}
   names why and how to authenticate (nimbus-mcp login / NIMBUS_TOKEN) — surface
   that, do not pre-emptively ask for credentials. {ok: false, tierGate: true}
   means a hosted Pro feature on a free account. {ok: false, setupRequired: true}
   now only appears when a credential was rejected or no local identity exists —
   walk the user through the options in that payload.
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
4c) SEE the results: plots.confusion / plots.dataset / plots.erp / plots.leaderboard
   render PNG figures you can look at (not just JSON numbers); stats.grouped pools
   per-subject accuracies across runs into cluster-aware stats (t-interval +
   t-test vs chance) - the review-grade companion to execution.results.
5) Live (use with care): device.list, device.test, then stream.start(confirm=true) only with
   the user's explicit go-ahead — it connects an EEG device to a human session. An idle
   watchdog stops abandoned sessions; polling stream.status keeps them alive.
   stream.telemetry works with OR without a deployed model: modelless streams
   report signal quality + focus/relaxation band-power indicators
   (latestPrediction null).
6) Personal calibration: calibration.start(confirm=true) while the subject wears the device —
   the Studio app shows cues; calibration.status polls; calibration.train turns the recording
   into the subject's own classifier.
6b) Neurofeedback protocols: protocol.validate iterates the document (verdict data, no throw),
   protocol.save(confirm=true) versions it, protocol.start_assignment(confirm=true) enrolls a
   subject and prepares the deploy session (yoked sham arms need a donor recording upload);
   simulated-source sessions are free, live hardware requires Pro. Poll protocol.session_status,
   pause/resume/stop drive the session, protocol.export_stats gives paired pre/post stats +
   figure + CSV (inferential from 3 completed sessions).
7) BYO model code: python.validate(script) statically checks a python_model class
   (fit/predict contract, no execution), then run it via execution.run with the code
   inline in the node config (runtime auto|local|cloud — desktop app: isolated local
   worker; hosted web app: ephemeral cloud sandbox with optional per-run pip
   requirements); leaderboard.submit scores a model class server-side (LOSO).
8) Foundation-model lane: foundation_features(backbone=reve|labram|eegpt|cbramod|
   biot|steegformer|codebrain) — 7 pretrained EEG backbones; frozen embeddings
   feed any head. foundation.compare(source/dataset/subject) runs EVERY backbone on
   YOUR data once (frozen embed + LDA 5-fold CV) and returns a ranked chart with
   CIs, params, and wall time — pick the backbone by measurement, not vibes. See
   catalog.leaderboard(track=pretrained_few_trial) for which currently wins at the
   100-trial budget.
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


# fastmcp loggers that print "Error calling tool …" with a full (rich)
# traceback for every tool exception.
_TOOL_ERROR_LOGGERS = ("fastmcp.server.server", "fastmcp.tools.base", "fastmcp.tools.function_tool")


def _expected_condition(exc: BaseException | None) -> BaseException | None:
    """The actionable McpToolError/SetupRequired being logged, if that is what
    it is: the logged exception itself, or its explicit ``__cause__`` (a
    ``raise X from err`` wrapper). ``__context__`` is never followed — a real
    bug raised *while handling* an expected error must keep its traceback."""
    for cur in (exc, getattr(exc, "__cause__", None)):
        if isinstance(cur, (SetupRequired, McpToolError)):
            return cur
    return None


class ExpectedConditionLogFilter(logging.Filter):
    """Expected setup/actionable conditions log ONE stderr line, no traceback.

    A missing backend, rejected token or tier check is an answer the agent
    already receives as structured guidance; dumping a ~6 KB rich traceback
    per call to stderr only floods clients that do not drain it. Genuine
    bugs (any other exception) keep their full traceback.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        expected = _expected_condition(exc)
        if expected is None:
            return True
        if isinstance(expected, SetupRequired):
            detail = str(expected.guidance.get("message") or "setup required")
            kind = "setup required"
        else:
            detail = str(expected)
            kind = getattr(expected, "code", None) or "tool error"
        first = detail.strip().splitlines()[0] if detail.strip() else ""
        record.msg = f"{record.getMessage()} ({kind}): {first[:200]}"
        record.args = None
        record.exc_info = None
        record.exc_text = None
        if record.levelno > logging.WARNING:
            record.levelno, record.levelname = logging.WARNING, "WARNING"
        return True


_LOG_FILTER = ExpectedConditionLogFilter()


def install_expected_condition_log_filter() -> None:
    """Idempotently attach the filter to fastmcp's tool-error loggers."""
    for name in _TOOL_ERROR_LOGGERS:
        logger = logging.getLogger(name)
        if _LOG_FILTER not in logger.filters:
            logger.addFilter(_LOG_FILTER)


def _access_mode_of(client: object) -> str:
    """anonymous | local | hosted for telemetry props (duck-typed so this
    works with the stdio client, NullClient, and the gateway wrapper)."""
    mode = getattr(client, "access_mode", None)
    if mode in ("anonymous", "local", "hosted"):
        return str(mode)
    config = getattr(client, "config", None)
    if config is not None:
        if getattr(config, "nimbus_token", None):
            return "hosted"
        if getattr(config, "mcp_key", None):
            return "local"
    return "anonymous"


def _error_code_of(err: BaseException) -> str:
    """Stable error CODE for telemetry — never the message text (X12)."""
    seen: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        code = getattr(cur, "code", None)
        if isinstance(code, str) and code:
            return code
        if isinstance(cur, SetupRequired):
            return "setup_required"
        cur = cur.__cause__ or cur.__context__
    return "tool_error"


class ProductTelemetryMiddleware(Middleware):
    """X11/X12: per-call telemetry events + one-time consent/version notices.

    Events are consent-gated inside :class:`Telemetry` (off = zero network
    calls). Notices attach to the first eligible dict-shaped tool result:
    the telemetry consent notice once per fresh install, and the version
    drift notice once per process when the client is behind.
    """

    def __init__(self, telemetry: Telemetry, client: object) -> None:
        self._telemetry = telemetry
        self._mode = _access_mode_of(client)

    async def on_call_tool(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> ToolResult:
        tool = str(getattr(getattr(context, "message", None), "name", "") or "unknown")
        started = time.monotonic()
        try:
            result = await call_next(context)
        except Exception as err:
            self._telemetry.tool_completed(
                tool=tool,
                mode=self._mode,
                success=False,
                error_code=_error_code_of(err),
                duration_s=time.monotonic() - started,
            )
            raise
        structured = result.structured_content
        success = True
        error_code: str | None = None
        if isinstance(structured, dict):
            if structured.get("loginRequired"):
                success, error_code = False, "login_required"
                self._telemetry.login_prompt_shown(
                    trigger=str(structured.get("trigger") or "unknown"),
                    mode=self._mode,
                )
            elif structured.get("tierGate"):
                success, error_code = False, "tier_gate"
            elif structured.get("setupRequired"):
                success, error_code = False, "setup_required"
            elif structured.get("ok") is False:
                success = False
                code = structured.get("code")
                error_code = str(code) if isinstance(code, str) and code else "ok_false"
        self._telemetry.tool_completed(
            tool=tool,
            mode=self._mode,
            success=success,
            error_code=error_code,
            duration_s=time.monotonic() - started,
        )
        if isinstance(structured, dict):
            extra: dict = {}
            notice = self._telemetry.pending_notice()
            if notice is not None:
                extra["telemetryNotice"] = notice
            drift = version_check.pending_notice()
            if drift is not None:
                extra["versionNotice"] = drift
                self._telemetry.version_drift_warned(
                    current=str(drift.get("current")),
                    latest=str(drift.get("latest")),
                    mode=self._mode,
                )
            if extra:
                return ToolResult(
                    content=result.content,
                    structured_content={**structured, **extra},
                )
        return result


def build_server(
    client: NimbusClient | NullClient | None = None,
    *,
    product_telemetry: Telemetry | None = None,
    auth: Any = None,
) -> FastMCP:
    if client is None:
        config = load_config()
        # Local Free tier (plan X9): no credential -> an anonymous client
        # against the local backend, NOT a setup-mode stand-in. Tools run;
        # cloud-only features answer with a structured login prompt at the
        # moment they need an account (access.py).
        client = NimbusClient(config)
    # version= so the initialize handshake (serverInfo.version) reports THIS
    # package's version — FastMCP defaults to advertising its own.
    mcp = FastMCP("nimbus", version=__version__, instructions=INSTRUCTIONS, auth=auth)
    install_expected_condition_log_filter()
    mcp.add_middleware(SetupGuidanceMiddleware())
    if product_telemetry is None:
        # The hosted gateway has no install identity and must not attach
        # per-install consent notices to other people's sessions.
        product_telemetry = (
            Telemetry.disabled()
            if type(client).__name__ == "HeaderCredentialClient"
            else get_telemetry()
        )
    mcp.add_middleware(ProductTelemetryMiddleware(product_telemetry, client))
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
        plots,
        telemetry,
        data,
        inspect,
        projects,
        protocol,
        python,
        stats,
        foundation,
    ):
        module.register(mcp, client)
    return mcp


def main() -> None:
    version_check.start_background_check()  # non-blocking; silent offline
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
    from .observability import init_gateway_sentry

    # Deployment opt-in only (Fly sets SENTRY_DSN); stdio installs never
    # reach this code path and the SDK rides the `sentry` extra.
    init_gateway_sentry()

    from .oauth import oauth_enabled, provider_from_env

    client = HeaderCredentialClient(load_config())
    # PR-O3: MCP-spec OAuth only when NIMBUS_MCP_OAUTH=1; otherwise the gateway
    # behaves exactly as before (header tokens + setupRequired guidance).
    auth = provider_from_env() if oauth_enabled() else None
    if auth is not None and client.config.nimbus_token:
        # With OAuth on, headerless requests get the spec 401 before any tool
        # runs, so the single-tenant NIMBUS_TOKEN fallback never applies.

        logging.getLogger("nimbus_mcp.server").warning(
            "NIMBUS_MCP_OAUTH=1: the startup NIMBUS_TOKEN fallback is ignored "
            "(headerless requests must sign in)"
        )
    build_server(client, product_telemetry=Telemetry.disabled(), auth=auth).run(
        transport="http", host=host, port=port, path=path
    )
