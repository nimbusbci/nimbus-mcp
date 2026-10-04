from pathlib import Path

import httpx
import pytest
from fastmcp import Client, FastMCP

from nimbus_mcp.client import NimbusClient
from nimbus_mcp.config import McpConfig
from nimbus_mcp.tools import discovery

NODE_TYPES = {
    "ok": True,
    "count": 2,
    "nodeTypes": [
        {
            "id": "rxlda_sdk",
            "name": "Nimbus LDA",
            "category": "model",
            "description": "Bayesian LDA",
            "inputs": [{"name": "features", "type": "2d_features", "required": True}],
            "outputs": [{"name": "predictions", "type": "predictions", "required": True}],
            "configSchema": {"type": "object", "properties": {"iterations": {"type": "number"}}},
        },
        {
            "id": "csp",
            "name": "CSP",
            "category": "feature",
            "description": "Spatial filters",
            "inputs": [],
            "outputs": [],
            "configSchema": {"type": "object"},
        },
    ],
}

TEMPLATES = {
    "ok": True,
    "templates": [
        {
            "id": "mi_bciiv2a_csp_lda",
            "name": "Motor Imagery (BCI IV 2a) - CSP -> Nimbus LDA",
            "description": "Classic MI pipeline",
            "category": "motor_imagery",
            "expectedAccuracy": 0.7,
        }
    ],
}

TEMPLATE_DETAIL = {
    "ok": True,
    "template": {
        "id": "mi_bciiv2a_csp_lda",
        "name": "MI CSP LDA",
        "version": "1",
        "train": {
            "nodes": [{"id": "data", "type": "public_data", "config": {}}],
            "connections": [],
        },
    },
}

DATASETS = {
    "ok": True,
    "datasets": {
        "BNCI2014_001": {
            "id": "BNCI2014_001",
            "source": "moabb_preprocessed",
            "paradigm": "MI",
            "onDisk": True,
            "defaultSubject": "S01",
            "subjects": ["S01", "S02"],
            "details": {"channels": 22, "samplingRate": 250.0},
        },
        "MISSING_ONE": {
            "id": "MISSING_ONE",
            "source": "moabb",
            "paradigm": "P300",
            "onDisk": False,
            "subjects": None,
            "details": None,
        },
    },
}


def make_server(handler) -> FastMCP:
    cfg = McpConfig(api_url="http://test", mcp_key="k", export_dir=Path("/tmp/nx"))
    client = NimbusClient(cfg, transport=httpx.MockTransport(handler))
    server = FastMCP("nimbus-test")
    discovery.register(server, client)
    return server


async def test_list_nodes_filters_category():
    server = make_server(lambda req: httpx.Response(200, json=NODE_TYPES))
    async with Client(server) as c:
        result = await c.call_tool("list_nodes", {"category": "model"})
    ids = [n["id"] for n in result.data["nodes"]]
    assert ids == ["rxlda_sdk"]


async def test_get_node_schema_returns_schema_and_ports():
    server = make_server(lambda req: httpx.Response(200, json=NODE_TYPES))
    async with Client(server) as c:
        result = await c.call_tool("get_node_schema", {"node_type": "rxlda_sdk"})
    assert "iterations" in result.data["configSchema"]["properties"]
    assert result.data["ports"]["inputs"][0]["type"] == "2d_features"


async def test_get_node_schema_unknown_type_errors():
    server = make_server(lambda req: httpx.Response(200, json=NODE_TYPES))
    async with Client(server) as c:
        with pytest.raises(Exception, match="Unknown node type"):
            await c.call_tool("get_node_schema", {"node_type": "nope"})


async def test_list_templates_compact():
    server = make_server(lambda req: httpx.Response(200, json=TEMPLATES))
    async with Client(server) as c:
        result = await c.call_tool("list_templates", {})
    assert result.data["templates"][0]["id"] == "mi_bciiv2a_csp_lda"


async def test_get_template_returns_train_graph():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/templates/mi_bciiv2a_csp_lda"
        return httpx.Response(200, json=TEMPLATE_DETAIL)

    async with Client(make_server(handler)) as c:
        result = await c.call_tool("get_template", {"template_id": "mi_bciiv2a_csp_lda"})
    assert result.data["train"]["nodes"][0]["type"] == "public_data"


async def test_get_template_rejects_bad_template_id():
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("validation must precede the HTTP request")

    async with Client(make_server(handler)) as c:
        with pytest.raises(Exception, match="Invalid template id"):
            await c.call_tool("get_template", {"template_id": "../../api/admin"})


async def test_list_datasets_on_disk_filter():
    server = make_server(lambda req: httpx.Response(200, json=DATASETS))
    async with Client(server) as c:
        result = await c.call_tool("list_datasets", {})
    ids = [d["id"] for d in result.data["datasets"]]
    assert ids == ["BNCI2014_001"]
    entry = result.data["datasets"][0]
    assert entry["channels"] == 22 and entry["subjectsCount"] == 2
