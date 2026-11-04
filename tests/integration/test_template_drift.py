"""Drift guard: the shipped backend calibration templates must survive
``_graph_section`` canonicalization.

The execute request's ``GraphConnection`` is ``extra="forbid"`` with only
``{from, to}``, but canvas-style templates (sart, target_hit) carry
``sourceHandle``/``targetHandle`` on their connections. If a new paradigm
template ships with handles (or a handle-free template regresses), this test
fails before calibration.start 422s in production. File-based — no live
backend needed; skipped when the backend tree is absent (sdist/standalone).
"""

from pathlib import Path

import pytest

# importorskip (not a plain import): sdist-based installs may lack pyyaml, and
# a collection-time ImportError would fail the whole suite instead of skipping.
yaml = pytest.importorskip("yaml")

from nimbus_mcp.tools.calibration import (  # noqa: E402
    MI_DEFAULT_TRAIN_TEMPLATE,
    P300_DEFAULT_TRAIN_TEMPLATE,
    PARADIGM_TEMPLATES,
    _graph_section,
)
from nimbus_mcp.tools.protocol import PROTOCOL_TEMPLATE  # noqa: E402

BACKEND_TEMPLATES = (
    Path(__file__).resolve().parents[3]
    / "backend-py"
    / "nimbus_backend"
    / "storage"
    / "templates"
    / "calibration"
)
BACKEND_TEMPLATES_ROOT = BACKEND_TEMPLATES.parent

pytestmark = pytest.mark.skipif(
    not BACKEND_TEMPLATES.is_dir(), reason="backend template tree not present"
)


@pytest.mark.parametrize("paradigm", sorted(PARADIGM_TEMPLATES))
def test_shipped_template_connections_reduce_to_from_to(paradigm: str) -> None:
    template_id = PARADIGM_TEMPLATES[paradigm]
    path = BACKEND_TEMPLATES / f"{template_id}.yaml"
    raw = yaml.safe_load(path.read_text())
    section = _graph_section(raw, "calibrate", template_id)
    assert section["connections"], f"{template_id}: no connections in calibrate section"
    for conn in section["connections"]:
        assert set(conn) == {"from", "to"}, (
            f"{template_id}: connection {conn} carries keys beyond from/to — "
            "the execute request's GraphConnection is extra=forbid"
        )


def _backend_template_ids() -> set[str]:
    """Template ids the backend catalog would serve (YAML ``id`` else file stem)."""
    ids: set[str] = set()
    for path in sorted(BACKEND_TEMPLATES_ROOT.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        template_id = data.get("id") if isinstance(data, dict) else None
        ids.add(template_id.strip() if isinstance(template_id, str) else path.stem)
    return ids


def test_pinned_template_ids_exist_in_backend_catalog() -> None:
    """Every template id the MCP tools hardcode must resolve at runtime.

    nimbus-mcp ships to PyPI on its own cadence — a backend template rename
    otherwise turns calibration.train / protocol.start_assignment into a
    runtime "Template not found" with no test failing.
    """
    pinned = {
        *PARADIGM_TEMPLATES.values(),
        MI_DEFAULT_TRAIN_TEMPLATE,
        P300_DEFAULT_TRAIN_TEMPLATE,
        PROTOCOL_TEMPLATE,
    }
    known = _backend_template_ids()
    missing = pinned - known
    assert not missing, (
        f"MCP pins template ids absent from {BACKEND_TEMPLATES_ROOT}: {sorted(missing)}"
    )
