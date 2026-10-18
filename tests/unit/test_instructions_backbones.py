"""Pin the foundation-backbone list in the MCP instructions string."""

from __future__ import annotations

import re

from nimbus_mcp.server import INSTRUCTIONS

_EXPECTED = (
    "reve",
    "labram",
    "eegpt",
    "cbramod",
    "biot",
    "steegformer",
    "codebrain",
)


def test_instructions_pin_seven_foundation_backbones():
    match = re.search(r"backbone=([a-z|\s]+)", INSTRUCTIONS)
    assert match is not None
    names = [part.strip() for part in match.group(1).split("|") if part.strip()]
    assert names == list(_EXPECTED)
    assert "stub" not in names
