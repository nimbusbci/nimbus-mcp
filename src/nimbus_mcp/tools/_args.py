"""Tool-argument normalization: common aliases and friendly errors.

E2E round 1/2 (2026-10-21): ``data.inspect_dataset {dataset_id}`` and
``pipeline.export {graph}`` returned raw pydantic dumps. Known aliases are
rewritten to the real parameter; anything still unknown or missing gets one
readable line listing the tool's valid parameters.
"""

from __future__ import annotations

import re
from typing import Any

#: Explicit cross-name aliases (alias -> canonical), applied only when the
#: canonical name is a parameter of the tool and was not passed itself.
EXPLICIT_ALIASES: dict[str, str] = {
    "graph": "train_graph",
    "exec_graph": "train_graph",
    "execGraph": "train_graph",
    "pipeline": "train_graph",
    "train": "train_graph",
    "dataset_name": "dataset",
}

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def _candidates(key: str) -> list[str]:
    snake = _CAMEL.sub("_", key).lower()
    out = [EXPLICIT_ALIASES.get(key, ""), snake]
    for k in (key, snake):
        if k.endswith("_id"):
            out.append(k[: -len("_id")])
        else:
            out.append(f"{k}_id")
    return [c for c in out if c and c != key]


def normalize_arguments(
    arguments: dict[str, Any], schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    """``(arguments with aliases resolved, {alias: canonical})``."""
    props = schema.get("properties") or {}
    if not isinstance(props, dict) or not arguments:
        return arguments, {}
    out = dict(arguments)
    renamed: dict[str, str] = {}
    for key in list(arguments):
        if key in props:
            continue
        for cand in _candidates(key):
            if cand in props and cand not in out:
                out[cand] = out.pop(key)
                renamed[key] = cand
                break
    return out, renamed


def describe_params(schema: dict[str, Any]) -> str:
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    parts = [f"{name} (required)" if name in required else name for name in props]
    return ", ".join(parts) if parts else "none"


def argument_problem(tool: str, arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Friendly message for unknown/missing parameters, else ``None``."""
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return None
    strict = schema.get("additionalProperties", True) is False
    unknown = [k for k in arguments if k not in props] if strict else []
    missing = [k for k in (schema.get("required") or []) if k not in arguments]
    if not unknown and not missing:
        return None
    bits = []
    if unknown:
        bits.append("unknown parameter(s) " + ", ".join(repr(k) for k in unknown))
    if missing:
        bits.append("missing required parameter(s) " + ", ".join(repr(k) for k in missing))
    return (
        f"Invalid arguments for {tool}: {'; '.join(bits)}. "
        f"Valid parameters: {describe_params(schema)}."
    )


# Only FastMCP's own argument validation (model title ``call[<fn>]``): a
# ValidationError raised *inside* a tool (e.g. parsing a backend response) is
# a server-side failure and must not be relabelled "Invalid arguments".
_PYDANTIC_HEAD = re.compile(r"^\d+ validation errors? for call\[")
_PYDANTIC_ITEM = re.compile(r"^(\S[^\n]*)\n\s+(.+?)\s*\[type=([^,\]]+)", re.M)


def friendly_validation_message(tool: str, text: str, schema: dict[str, Any]) -> str | None:
    """Rewrite a raw pydantic dump into ``Invalid arguments for tool: field: msg``."""
    if not _PYDANTIC_HEAD.match(text or ""):
        return None
    items = [
        f"{field.strip()}: {msg.rstrip('.')}" for field, msg, _t in _PYDANTIC_ITEM.findall(text)
    ]
    detail = "; ".join(items) if items else "arguments did not match the tool's schema"
    return f"Invalid arguments for {tool}: {detail}. Valid parameters: {describe_params(schema)}."
