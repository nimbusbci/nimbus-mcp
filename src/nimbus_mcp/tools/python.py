"""Python-model tools: pre-flight validation of BYO user model code."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..client import NimbusClient
from ._annotations import READ_ONLY

_MAX_CODE_CHARS = 256 * 1024

_USAGE = (
    "Fix the reported issues, then embed the code in a python_model node config "
    '(config: {"code": "<script>", "inputKind": "epochs"|"features", "className": ...}) '
    "and run it with execution.run. python_model runtime is auto|local|cloud: the "
    "desktop app runs the isolated local worker; the hosted web app runs it in an "
    "ephemeral cloud sandbox (Modal) with optional per-run pip requirements."
)


def register(mcp: FastMCP, client: NimbusClient) -> None:
    @mcp.tool(name="python.validate", annotations=READ_ONLY)
    def validate_python_model(
        code: str,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        """Statically check a python_model script (BYO model class) without running it.

        Compiles the source and verifies the contract — a class with
        fit(X, y, info) and predict(X); optional predict_proba(X) unlocks
        confidence panels. Never executes the code, so it is safe against the
        hosted backend. Use it before execution.run to catch syntax errors,
        missing methods, or the wrong class name cheaply.

        Args:
            code: Python source of the model class (<=256KB).
            class_name: Contract class to instantiate; omit to auto-detect the
                first class defining fit+predict.
        """
        if not isinstance(code, str) or not code.strip():
            return {
                "ok": False,
                "error": "code is empty — provide the Python source of a model class "
                "with fit(X, y, info) and predict(X).",
                "warnings": [],
                "usage": _USAGE,
            }
        if len(code) > _MAX_CODE_CHARS:
            return {
                "ok": False,
                "error": f"code exceeds {_MAX_CODE_CHARS // 1024} KB.",
                "warnings": [],
                "usage": _USAGE,
            }
        body: dict[str, Any] = {"code": code}
        if class_name:
            body["className"] = class_name
        payload = client.post("/api/python-model/validate", json=body)
        result: dict[str, Any] = {
            "ok": bool(payload.get("ok")),
            "className": payload.get("className"),
            "error": payload.get("error"),
            "warnings": payload.get("warnings") or [],
        }
        if payload.get("codeHash"):
            result["codeHash"] = payload.get("codeHash")
        result["usage"] = _USAGE
        return result
