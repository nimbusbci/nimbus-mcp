from pathlib import Path

import httpx
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import python

GOOD_CODE = """class MyModel:
    def fit(self, X, y, info):
        return self
    def predict(self, X):
        return [0] * len(X)
"""


def make_server(handler) -> FastMCP:
    client = NimbusClient(
        McpConfig(api_url="http://t", mcp_key="k", export_dir=Path("/tmp/nx")),
        transport=httpx.MockTransport(handler),
    )
    server = FastMCP("t")
    python.register(server, client)
    return server


async def test_validate_posts_code_and_class_name():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        import json

        seen["body"] = json.loads(req.read())
        return httpx.Response(
            200,
            json={
                "ok": True,
                "className": "MyModel",
                "error": None,
                "warnings": [
                    "No predict_proba() — confidence and uncertainty panels will be unavailable."
                ],
                "codeHash": "abc123",
            },
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool(
            "python.validate", {"code": GOOD_CODE, "class_name": "MyModel"}
        )
    assert seen["path"] == "/api/python-model/validate"
    assert seen["body"] == {"code": GOOD_CODE, "className": "MyModel"}
    assert result.data["ok"] is True
    assert result.data["className"] == "MyModel"
    assert result.data["codeHash"] == "abc123"
    assert any("predict_proba" in w for w in result.data["warnings"])
    assert "python_model" in result.data["usage"]


async def test_validate_omits_class_name_when_absent():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(req.read())
        return httpx.Response(
            200,
            json={"ok": False, "className": None, "error": "Syntax error (line 1)", "warnings": []},
        )

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("python.validate", {"code": "class Broken("})
    assert seen["body"] == {"code": "class Broken("}
    assert result.data["ok"] is False
    assert "Syntax error" in result.data["error"]
    assert result.data["usage"]


async def test_validate_empty_code_short_circuits_without_http():
    def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be hit
        raise AssertionError("empty code must not reach the backend")

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("python.validate", {"code": "   "})
    assert result.data["ok"] is False
    assert "empty" in result.data["error"]


async def test_validate_oversized_code_short_circuits():
    def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be hit
        raise AssertionError("oversized code must not reach the backend")

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("python.validate", {"code": "x" * (256 * 1024 + 1)})
    assert result.data["ok"] is False
    assert "exceeds" in result.data["error"]
