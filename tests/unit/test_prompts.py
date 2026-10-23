from __future__ import annotations

import asyncio

from nimbus_mcp.prompts import EXAMPLE_PROMPTS
from nimbus_mcp.server import build_server


def test_example_prompts_registered_and_reference_real_tools():
    mcp = build_server()
    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    prompts = {p.name: p for p in asyncio.run(mcp.list_prompts())}
    assert set(prompts) == set(EXAMPLE_PROMPTS)
    for name, (_title, _desc, text, used) in EXAMPLE_PROMPTS.items():
        assert prompts[name].description
        for tool in used:
            assert tool in tools, f"{name} references unknown tool {tool}"
            assert tool in text, f"{name} lists {tool} but its text does not mention it"


def test_prompt_renders_text():
    mcp = build_server()
    result = asyncio.run(mcp.render_prompt("mi_csp_lda_bci_iv_2a"))
    rendered = str(result)
    assert "mi_bciiv2a_csp_lda" in rendered and "plots.confusion" in rendered
