"""whoami tool: profile passthrough + local JWT decode for token info."""

import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import whoami


def _b64url(obj: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def _fake_jwt_token(payload: dict) -> str:
    # decode_token_claims strips nimb_ and decodes without verification, so an
    # unsigned fake is enough — same helper shape as tests/unit/test_cli.py.
    return "nimb_" + f"{_b64url({'alg': 'none'})}.{_b64url(payload)}.sig"


def _profile() -> dict:
    return {
        "userId": "user_77",
        "email": "dev@nimbus.test",
        "capabilities": {
            "isPro": False,
            "hasPioneerAccess": True,
            "freeTrainingRunsMonthlyLimit": 10,
            "freeTrainingRunsRemaining": 7,
        },
    }


def make_server(handler, *, token: str = "", meta: dict | None = None) -> FastMCP:
    cfg = McpConfig(
        api_url="http://t",
        mcp_key="" if token else "k",
        export_dir=Path("/tmp/nx"),
        nimbus_token=token,
        credential_meta=meta if meta is not None else {},
    )
    server = FastMCP("t")
    whoami.register(server, NimbusClient(cfg, transport=httpx.MockTransport(handler)))
    return server


async def test_whoami_token_mode_full_payload():
    """Mocked profile + a locally-minted fake nimb_ JWT: the payload carries the
    account, plan/quota, token name/expiry/daysLeft, and the credential source."""
    exp = int(time.time()) + 30 * 86400
    seen: list = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, req.headers.get("authorization")))
        return httpx.Response(200, json=_profile())

    token = _fake_jwt_token({"exp": exp, "sub": "user_77"})
    server = make_server(handler, token=token, meta={"source": "store", "name": "dev token"})

    async with Client(server) as c:
        result = await c.call_tool("whoami", {})

    assert seen == [("/api/me/profile", f"Bearer {token}")]
    data = result.data
    assert data["userId"] == "user_77"
    assert data["email"] == "dev@nimbus.test"
    assert data["plan"] == {"isPro": False, "pioneerAccess": True}
    assert data["freeRuns"] == {"monthlyLimit": 10, "remaining": 7}
    assert data["source"] == "store"
    token_info = data["token"]
    assert token_info["name"] == "dev token"
    assert token_info["daysLeft"] == 30
    expected_iso = datetime.fromtimestamp(exp, timezone.utc).isoformat()
    assert token_info["expiresAt"] == expected_iso
    # never the raw secret, not even a prefix
    assert token not in json.dumps(data)
    assert token[:12] not in json.dumps(data)


async def test_whoami_pro_plan_reported():
    profile = _profile()
    profile["capabilities"]["isPro"] = True
    profile["capabilities"]["freeTrainingRunsMonthlyLimit"] = None
    profile["capabilities"]["freeTrainingRunsRemaining"] = None
    server = make_server(
        lambda r: httpx.Response(200, json=profile),
        token=_fake_jwt_token({"exp": int(time.time()) + 86400}),
        meta={"source": "env"},
    )
    async with Client(server) as c:
        data = (await c.call_tool("whoami", {})).data
    assert data["plan"]["isPro"] is True
    assert data["freeRuns"] == {"monthlyLimit": None, "remaining": None}
    assert "name" not in data["token"]  # no store metadata in env mode
    assert data["source"] == "env"


async def test_whoami_local_key_mode_has_no_token_info():
    """X-MCP-Key modes carry no JWT — token is null; source reflects the meta."""
    server = make_server(
        lambda r: httpx.Response(200, json=_profile()), meta={"source": "desktop"}
    )
    async with Client(server) as c:
        data = (await c.call_tool("whoami", {})).data
    assert data["token"] is None
    assert data["source"] == "desktop"


async def test_whoami_token_without_exp_claims_omits_expiry():
    server = make_server(
        lambda r: httpx.Response(200, json=_profile()),
        token=_fake_jwt_token({"sub": "user_77"}),  # no exp claim, no name
        meta={"source": "token_file"},
    )
    async with Client(server) as c:
        data = (await c.call_tool("whoami", {})).data
    # nothing displayable is known about the token → null rather than a stub
    assert data["token"] is None
