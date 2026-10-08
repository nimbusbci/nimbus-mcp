"""data.inspect_dataset / data.inspect_file — passthrough, params, source pick, 403 guidance."""

from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import inspect as inspect_tools

_DESCRIBE_PAYLOAD = {
    "ok": True,
    "source": "dataset",
    "description": "BNCI2014_001 S01",
    "channels": {"count": 22, "names": ["C1", "C2"], "flatlined": []},
    "samplingRate": 250.0,
    "durationSec": 1152.0,
    "trials": {
        "count": 288,
        "classLabels": ["left", "right"],
        "classCounts": {"left": 144, "right": 144},
    },
    "epochWindow": {"tmin": 2.0, "tmax": 6.0},
    "channelStats": [{"name": "C1", "meanUv": -1.2, "stdUv": 8.4, "variance": 70.6}],
    "bandPowers": {"alpha": {"C1": -12.5}},
    "psd": {"freqs": [1.0, 2.0], "perChannelMean": {"C1": [-20.0, -21.0]}},
    "computedFrom": {"nSamples": 12000, "sampleStrategy": "stratified_sample_seed42_4_of_20"},
}


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    inspect_tools.register(server, client)
    return server


# ── data.inspect_dataset ──────────────────────────────────────────────────────────


async def test_inspect_dataset_passes_payload_through_and_encodes_params():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json=_DESCRIBE_PAYLOAD)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "data.inspect_dataset",
            {"dataset": "BNCI2014_001", "subject": "S01", "mode": "training"},
        )
    assert seen["path"] == "/api/data/describe"
    assert seen["params"] == {
        "source": "dataset",
        "dataset": "BNCI2014_001",
        "subject": "S01",
        "mode": "training",
    }
    # Verbatim passthrough: the backend's describe shape reaches the agent intact.
    assert result.data == _DESCRIBE_PAYLOAD
    assert result.data["trials"]["classCounts"] == {"left": 144, "right": 144}


async def test_inspect_dataset_defaults_mode_all_and_omits_missing_subject():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        # The backend 400s a missing subject — pass that detail through.
        return httpx.Response(
            400,
            json={
                "code": "nimbus.data.describe_invalid_request",
                "title": "BadRequest",
                "detail": "source=dataset requires both dataset and subject.",
            },
        )

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="requires both dataset and subject"):
            await c.call_tool("data.inspect_dataset", {"dataset": "BNCI2014_001"})
    assert seen["params"] == {"source": "dataset", "dataset": "BNCI2014_001", "mode": "all"}


async def test_inspect_dataset_accepts_comma_cohort_subject():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json={**_DESCRIBE_PAYLOAD, "source": "dataset"})

    async with Client(make_server(handler)) as c:
        await c.call_tool("data.inspect_dataset", {"dataset": "BNCI2014_001", "subject": "S01,S03"})
    assert seen["params"]["subject"] == "S01,S03"


# ── data.inspect_file ─────────────────────────────────────────────────────────────


async def test_inspect_file_absolute_path_uses_local_source():
    """Absolute paths read the file from the backend machine — source=local
    (trusted on local-mode backends; hosted refuses it with guidance)."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json={**_DESCRIBE_PAYLOAD, "source": "local"})

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "data.inspect_file", {"path": "/Users/you/recordings/session-01.edf"}
        )
    assert seen["path"] == "/api/data/describe"
    assert seen["params"] == {
        "source": "local",
        "path": "/Users/you/recordings/session-01.edf",
    }
    assert result.data == {**_DESCRIBE_PAYLOAD, "source": "local"}


async def test_inspect_file_relative_path_uses_upload_source():
    """Relative paths (as data.upload returns them) resolve via the ownership
    table on ANY backend — the tool must send source=upload, not local."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json={**_DESCRIBE_PAYLOAD, "source": "upload"})

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("data.inspect_file", {"path": "uploads/usr1/session-01.edf"})
    assert seen["params"] == {
        "source": "upload",
        "path": "uploads/usr1/session-01.edf",
    }
    assert result.data["source"] == "upload"


async def test_inspect_file_hosted_403_local_forbidden_returns_guidance():
    """Hosted backends refuse local paths with DATA_DESCRIBE_LOCAL_FORBIDDEN —
    the tool converts that 403 into setup-style guidance, not a traceback."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "type": "urn:nimbus:error:Forbidden",
                "title": "Forbidden",
                "status": 403,
                "detail": (
                    "Local file inspection is only available on a local backend "
                    "(desktop / MCP local mode). Upload the file first (e.g. via "
                    "data.upload), or run against a local backend."
                ),
                "code": "nimbus.data.describe_local_forbidden",
            },
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("data.inspect_file", {"path": "/abs/eeg.csv"})
    data = result.data
    assert data["ok"] is False
    assert "local backend" in data["message"]
    actions = [opt["action"] for opt in data["options"]]
    assert len(actions) == 2
    assert "upload" in actions[0]
    assert "local backend" in actions[1]
    # Guidance shape mirrors setup mode: message + actionable options.
    assert all("detail" in opt for opt in data["options"])


async def test_inspect_file_relative_path_local_forbidden_still_raises():
    """Guidance fires ONLY for absolute paths: a relative path goes to the
    upload source, so a stray DATA_DESCRIBE_LOCAL_FORBIDDEN there is a real
    error, not convertible guidance."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "code": "nimbus.data.describe_local_forbidden",
                "detail": "Local file inspection is only available on a local backend.",
            },
        )

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="local backend"):
            await c.call_tool("data.inspect_file", {"path": "uploads/usr1/eeg.csv"})


async def test_inspect_file_other_403_still_raises():
    """A 403 with a different code is a real denial (e.g. quota) — normal error."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "code": "nimbus.freemium.monthly_quota_exceeded",
                "detail": "Monthly free training quota exceeded.",
            },
        )

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="quota exceeded"):
            await c.call_tool("data.inspect_file", {"path": "/abs/eeg.csv"})


async def test_inspect_file_404_raises_normally():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "code": "nimbus.data.describe_source_not_found",
                "detail": "Local file not found: /abs/eeg.csv",
            },
        )

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="not found"):
            await c.call_tool("data.inspect_file", {"path": "/abs/eeg.csv"})
