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

from nimbus_mcp.tools.calibration import PARADIGM_TEMPLATES, _graph_section  # noqa: E402

BACKEND_TEMPLATES = (
    Path(__file__).resolve().parents[3]
    / "backend-py"
    / "nimbus_backend"
    / "storage"
    / "templates"
    / "calibration"
)

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
