"""nimbus-mcp: MCP tools for Nimbus BCI pipelines (local backend or hosted API token)."""

try:  # single source of truth: the wheel version from pyproject
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("nimbus-mcp")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0.dev0"
