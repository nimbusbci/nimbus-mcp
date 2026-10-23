"""execution.results warns on in-sample / near-chance headlines (E2E round 2)."""

from __future__ import annotations

from nimbus_mcp.tools.run import metric_warning


def test_p300_repro_warns_in_sample():
    w = metric_warning(
        {
            "headlineMetric": {"key": "balancedAccuracy", "valuePct": 61.3, "isInSample": True},
            "chanceLevelPct": 83.3,
            "classBalance": {"NonTarget": 3500, "Target": 700},
        }
    )
    assert w and "IN-SAMPLE" in w and "chance" not in w  # 61.3 vs 50 (balanced chance)


def test_near_chance_accuracy_warns():
    w = metric_warning(
        {
            "headlineMetric": {"key": "accuracy", "valuePct": 52.0, "isInSample": False},
            "chanceLevelPct": 50.0,
        }
    )
    assert w and "within 5 points of chance" in w


def test_healthy_or_legacy_has_no_warning():
    assert metric_warning({"headlineMetric": {"key": "accuracy", "valuePct": 80.0}}) is None
    assert metric_warning({"accuracyPct": 85.6}) is None
