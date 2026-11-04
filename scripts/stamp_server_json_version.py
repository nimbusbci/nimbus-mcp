#!/usr/bin/env python3
"""Stamp mcp/server.json's version fields from pyproject.toml (the SSOT).

Release chore used to be a 3-edit manual bump (pyproject + two server.json
spots); test_gateway_parity.py catches drift after the fact — this script
removes the drift window. Run from mcp/ after bumping pyproject.toml:

    python scripts/stamp_server_json_version.py [--check]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tomllib

MCP_ROOT = Path(__file__).resolve().parents[1]


def current_version() -> str:
    with (MCP_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify server.json is in sync instead of writing it",
    )
    args = parser.parse_args()

    version = current_version()
    server_json_path = MCP_ROOT / "server.json"
    data = json.loads(server_json_path.read_text(encoding="utf-8"))

    spots: list[str] = []
    if data.get("version") != version:
        spots.append("$.version")
        data["version"] = version
    packages = data.get("packages")
    if isinstance(packages, list):
        for i, package in enumerate(packages):
            if isinstance(package, dict) and package.get("version") != version:
                spots.append(f"$.packages[{i}].version")
                package["version"] = version

    if args.check:
        if spots:
            print(f"❌ server.json version drift at {', '.join(spots)}: expected {version}")
            print("   Fix: python scripts/stamp_server_json_version.py")
            return 1
        print(f"✅ server.json version matches pyproject.toml ({version}).")
        return 0

    if not spots:
        print(f"server.json already at {version}; nothing to do.")
        return 0

    server_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Stamped {', '.join(spots)} -> {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
