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


#: A local filesystem path was passed to the multi-user hosted gateway (E2E r9 N1).
LOCAL_PATH_UNSUPPORTED = "nimbus.mcp.local_path_unsupported"

#: Hint for figure tools refusing source="local" on the gateway.
LOCAL_SOURCE_HINT = (
    'Use source="upload" (data.upload with contentBase64) or source="dataset" instead.'
)


def refuse_local_path_on_gateway(client: Any, *, tool: str, hint: str) -> None:
    """Refuse a caller-supplied LOCAL path when running as the hosted gateway.

    The gateway's disk is not the caller's: resolving the path there would
    read server files into the caller's account and, via differing errors,
    act as a file-existence oracle. Raise BEFORE touching the filesystem, with
    one fixed message whatever the path is (never echo it, never stat it).
    """
    if not is_hosted_gateway(client):
        return
    from ..errors import local_tool_error

    raise local_tool_error(
        f"{tool} cannot read local file paths on the hosted Nimbus gateway: the "
        f"gateway runs on a remote server, not on your computer. {hint}",
        code=LOCAL_PATH_UNSUPPORTED,
        status_code=400,
    )


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
