"""Shared backend-figure decode/response scaffolding.

``plots.*`` and ``foundation.compare`` both consume the backend's
PlotImagePayload (base64 PNG + meta) and must agree on the decode rules and
the ToolResult sidecar shape — one divergence here means a payload change
gets fixed in one tool and not the other.
"""

from __future__ import annotations

import base64
from typing import Any

from ..errors import McpToolError


def decode_png(payload: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """(png_bytes, meta) off the backend's PlotImagePayload."""
    image_b64 = payload.get("image")
    if not isinstance(image_b64, str) or not image_b64:
        raise McpToolError("Backend returned no figure image (unexpected payload).")
    try:
        return base64.b64decode(image_b64, validate=True), dict(payload.get("meta") or {})
    except (ValueError, TypeError) as e:
        raise McpToolError(f"Backend figure payload was not valid base64 PNG: {e}") from e


def figure_info(payload: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """The structuredContent sidecar shared by every figure tool."""
    return {
        "ok": True,
        "title": payload.get("title"),
        "mimeType": payload.get("mimeType", "image/png"),
        "meta": meta,
    }
