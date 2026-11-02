"""E2E round 8 (2026-11-01): ids are validated before any request is built."""

from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.errors import McpToolError
from nimbus_mcp.tools import discovery, leaderboard, projects, run
from nimbus_mcp.tools._guards import INVALID_ARGUMENT, MAX_ID_LENGTH, safe_segment


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "abc\u0000",
        "a\nb",
        "a\tb",
        "a\x7fb",
        " abc",
        "abc ",
        "x" * (MAX_ID_LENGTH + 1),
        "../x",
        "a/b",
        ".",
    ],
)
def test_safe_segment_rejects(value):
    with pytest.raises(McpToolError) as exc:
        safe_segment(value)
    assert exc.value.code == INVALID_ARGUMENT
    assert "\u0000" not in str(exc.value)
    assert len(str(exc.value)) < 400


@pytest.mark.parametrize(
    "value",
    [
        "mi_csp_lda",
        "3f2b8c1e-0d4a-4c6e-9a7b-1234567890ab",
        "x" * MAX_ID_LENGTH,
        "my model v2.pkl",  # inner spaces (artifact file names)
        "Se\u00f1al_\u30c7\u30fc\u30bf",  # non-ASCII letters
        "rec\u00a0001",  # NBSP inside
        "team\u200d\U0001f9e0.pkl",  # ZWJ in an emoji sequence
    ],
)
def test_safe_segment_accepts(value):
    assert safe_segment(value) == value


def _server(seen: list) -> FastMCP:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, text="")

    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    mcp = FastMCP("t")
    for mod in (discovery, leaderboard, projects, run):
        mod.register(mcp, client)
    return mcp


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("execution.results", {"execution_id": "abc\u0000"}),  # B1
        ("project.load", {"project_id": "abc\u0000"}),  # B1
        ("catalog.template", {"template_id": "t" * 200_000}),  # B2
        ("leaderboard.status", {"submission_id": ""}),  # B3
    ],
)
async def test_tools_reject_bad_ids_with_code(tool, args):
    seen: list = []
    async with Client(_server(seen)) as c:
        result = await c.call_tool(tool, args, raise_on_error=False)
    assert result.is_error
    text = result.content[0].text
    assert INVALID_ARGUMENT in text
    assert "correlationId" in text
    assert "non-printable ASCII character in URL" not in text
    assert "URL too long" not in text
    assert seen == []  # never reached the HTTP layer
