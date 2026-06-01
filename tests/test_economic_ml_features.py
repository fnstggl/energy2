"""Feature-contract tests for Economic ML Alpha v1.

Proves: the leakage blocker works; value_quality provenance is preserved;
no production module is imported/modified by the feature pipeline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.forecasting import economic_ml_features as F  # noqa: E402,N812


def _row(**kw):
    base = {
        "source_trace_id": "t", "source_dataset_id": "ds", "gpu_type": "A100",
        "model_id": "m", "gpu_count": 1, "prompt_tokens": 100,
        "output_tokens": 50, "ttft_s": 0.4, "tpot_s": 0.02,
        "e2e_latency_s": 1.4, "cache_reuse_pct": 0.3, "kv_utilization": 0.5,
        "gpu_price_usd_per_hour": 2.0, "estimated_gpu_cost_usd": 0.01,
        "sla_safe_goodput_per_dollar": 5000.0,
        "value_quality_by_field": {"ttft_s": "measured",
                                    "cache_reuse_pct": "measured"},
    }
    base.update(kw)
    return base


# ───────────────────────── leakage blocker ─────────────────────────


def test_target_itself_is_blocked():
    spec = F.resolve_feature_spec("ttft_s")
    assert "ttft_s" in spec.blocked
    assert "ttft_s" not in spec.numeric


def test_post_decision_actuals_always_blocked():
    for tgt in ("ttft_s", "cache_reuse_pct", "e2e_latency_s"):
        spec = F.resolve_feature_spec(tgt)
        for blocked in ("estimated_gpu_cost_usd", "sla_safe_goodput_per_dollar",
                        "estimated_cache_value_usd", "sla_met"):
            assert blocked in spec.blocked, (tgt, blocked)


def test_output_tokens_blocked_for_latency_targets():
    for tgt in ("e2e_latency_s", "tpot_s", "ttft_s"):
        spec = F.resolve_feature_spec(tgt)
        assert "output_tokens" in spec.blocked
        assert "output_tokens" not in spec.numeric


def test_output_tokens_allowed_when_explicitly_unblocked():
    spec = F.resolve_feature_spec("cache_reuse_pct", allow_output_tokens=True)
    assert "output_tokens" not in spec.blocked


def test_build_matrix_raises_on_blocked_feature():
    bad = F.FeatureSpec(target="ttft_s", numeric=["e2e_latency_s"],
                        categorical=[], blocked=frozenset({"e2e_latency_s"}))
    with pytest.raises(F.LeakageError):
        F.build_matrix([_row()], bad)


def test_cross_latency_features_blocked():
    """Predicting ttft must not see tpot/e2e/throughput as features."""
    spec = F.resolve_feature_spec("ttft_s")
    for f in ("tpot_s", "e2e_latency_s", "throughput_tok_s"):
        assert f in spec.blocked


# ───────────────────────── matrix correctness ─────────────────────────


def test_build_matrix_drops_none_target_rows():
    rows = [_row(ttft_s=0.4), _row(ttft_s=None), _row(ttft_s=0.6)]
    spec = F.resolve_feature_spec("ttft_s")
    X, y, names, _ = F.build_matrix(rows, spec)
    assert len(y) == 2
    assert X.shape[0] == 2


def test_build_matrix_feature_names_match_columns():
    spec = F.resolve_feature_spec("ttft_s")
    X, y, names, _ = F.build_matrix([_row(), _row()], spec)
    assert X.shape[1] == len(names)
    assert "gpu_type" in names
    assert "ttft_s" not in names


def test_categorical_encoding_is_deterministic():
    rows = [_row(gpu_type="A100"), _row(gpu_type="H100"), _row(gpu_type="A100")]
    spec = F.resolve_feature_spec("ttft_s")
    X1, _, names, _ = F.build_matrix(rows, spec)
    X2, _, _, _ = F.build_matrix(rows, spec)
    assert np.array_equal(X1, X2)


# ───────────────────────── provenance preserved ─────────────────────────


def test_value_quality_distribution_preserved():
    rows = [_row(value_quality_by_field={"ttft_s": "measured"}),
            _row(value_quality_by_field={"ttft_s": "prior"}),
            _row(value_quality_by_field={})]
    dist = F.value_quality_distribution(rows, "ttft_s")
    assert dist["measured"] == 1
    assert dist["prior"] == 1
    assert dist["missing"] == 1


def test_feature_module_does_not_import_production_scheduler():
    """The feature module must not pull in production scheduler / scorer /
    residency / frontier modules."""
    src = (REPO_ROOT / "aurelius" / "forecasting"
           / "economic_ml_features.py").read_text()
    for forbidden in ("optimization.scheduler", "constraint_shadow_scorer",
                      "residency.decision", "frontier.controller"):
        assert forbidden not in src
