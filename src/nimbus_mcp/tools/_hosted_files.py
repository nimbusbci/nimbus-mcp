"""Hosted-gateway file returns: inline base64 instead of a gateway-local path.

On the multi-user hosted gateway the gateway's disk is not the caller's, so a
saved path (/tmp/nimbus-exports/...) is useless and leaks server layout. File
tools return the bytes inline (capped on the *encoded* size so tool results
stay within client response/context limits) or point to Nimbus Studio.
"""

from __future__ import annotations

import base64
from typing import Any

#: Cap on the encoded base64 text (~768 KB of raw bytes).
HOSTED_INLINE_MAX_BYTES = 1024 * 1024


def is_hosted_gateway(client: Any) -> bool:
    from ..hosted import HeaderCredentialClient

    return isinstance(client, HeaderCredentialClient)


def inline_file(data: bytes, filename: str, studio_hint: str) -> dict[str, Any]:
    """``{filename, size, contentBase64, encoding}`` or ``{filename, size, note}``."""
    out: dict[str, Any] = {"filename": filename, "size": len(data)}
    encoded_len = 4 * ((len(data) + 2) // 3)
    if encoded_len <= HOSTED_INLINE_MAX_BYTES:
        out["contentBase64"] = base64.b64encode(data).decode("ascii")
        out["encoding"] = "base64"
    else:
        out["note"] = (
            f"File is {encoded_len} bytes as base64, over the {HOSTED_INLINE_MAX_BYTES} "
            f"byte inline limit; {studio_hint}"
        )
    return out
