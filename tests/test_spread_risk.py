"""Tests for the DA->RT SpreadRiskModel."""

import pandas as pd

from aurelius.forecasting.spread_risk import SpreadRiskModel


def _maps(days=20):
    """Aligned DA/RT maps where us-east hour 18 *intermittently* spikes in RT.

    The spike hits ~1 in 3 days, so the median spread at hour 18 is ~0 (debias
    small) but the upside spread is large (lambda-weighted penalty large). Hour 3
    and all of us-west match DA exactly (no spread).
    """
    plan = {"us-east": {}, "us-west": {}}
    settle = {"us-east": {}, "us-west": {}}
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    for d in range(days):
        for h in range(24):
            ts = (base + pd.Timedelta(days=d, hours=h)).to_pydatetime()
            plan["us-east"][ts] = 50.0
            spike = (h == 18) and (d % 3 == 0)
            settle["us-east"][ts] = 250.0 if spike else 50.0
            plan["us-west"][ts] = 30.0
            settle["us-west"][ts] = 30.0  # stable: no spread
    return plan, settle


def test_spike_hour_gets_large_adjustment():
    plan, settle = _maps()
    m = SpreadRiskModel(risk_lambda=1.0, min_samples=5).fit(plan, settle)
    ts18 = pd.Timestamp("2025-02-01T18:00:00Z").to_pydatetime()
    ts03 = pd.Timestamp("2025-02-01T03:00:00Z").to_pydatetime()
    assert m.adjustment("us-east", ts18) > 100.0   # spike hour penalized
    assert m.adjustment("us-east", ts03) < 5.0      # calm hour barely touched


def test_stable_region_no_adjustment():
    plan, settle = _maps()
    m = SpreadRiskModel(risk_lambda=1.0, min_samples=5).fit(plan, settle)
    ts = pd.Timestamp("2025-02-01T12:00:00Z").to_pydatetime()
    assert abs(m.adjustment("us-west", ts)) < 1e-6


def test_lambda_zero_is_debias_only():
    plan, settle = _maps()
    m0 = SpreadRiskModel(risk_lambda=0.0, min_samples=5).fit(plan, settle)
    ts18 = pd.Timestamp("2025-02-01T18:00:00Z").to_pydatetime()
    # With lambda=0, only the median spread (0 for a single-spike hour) applies,
    # so the spike hour's adjustment is far smaller than with lambda=1.
    m1 = SpreadRiskModel(risk_lambda=1.0, min_samples=5).fit(plan, settle)
    assert m0.adjustment("us-east", ts18) < m1.adjustment("us-east", ts18)


def test_unfitted_model_is_identity():
    m = SpreadRiskModel()
    pmap = {"us-east": {pd.Timestamp("2025-01-01T00:00:00Z").to_pydatetime(): 50.0}}
    assert m.adjust_price_map(pmap) == pmap


def test_adjust_price_map_raises_spike_hour():
    plan, settle = _maps()
    m = SpreadRiskModel(risk_lambda=1.0, min_samples=5).fit(plan, settle)
    ts18 = pd.Timestamp("2025-03-01T18:00:00Z").to_pydatetime()
    pmap = {"us-east": {ts18: 50.0}}
    adj = m.adjust_price_map(pmap)
    assert adj["us-east"][ts18] > 150.0  # 50 DA + ~200 upside penalty
