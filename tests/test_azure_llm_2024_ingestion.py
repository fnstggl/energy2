"""Tests for the Azure LLM 2024 week-long ingestion + forecast-robustness harness.

CANONICAL_TRACE_BACKTEST_AZURE_LLM_2024_WEEK_V1. Unit tests run on the committed
sample fixture (no full download required). A full-trace integration test skips
if the raw files are absent. Covers schema validation, timestamp ordering, token
mapping, missing-latency honesty, forecast-mode leakage rules, alpha_survival,
determinism, and the no-production-savings / no-TTFT docs gates.
"""

from __future__ import annotations

import os

import pytest

from aurelius.traces import azure_llm as az
from aurelius.traces import azure_llm_forecast as fc
from aurelius.traces.schema import TraceSchemaError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv")
RAW_DIR = os.path.join(REPO_ROOT, "data", "external", "azure_llm_2024", "raw")
DOC = os.path.join(REPO_ROOT, "docs", "AZURE_LLM_2024_BACKTEST_RESULTS.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


def _agg(tick_seconds=60.0):
    return az.stream_week_aggregate({"conv": FIXTURE}, tick_seconds=tick_seconds)


# 1 ---------------------------------------------------------------------------
def test_sample_fixture_parses():
    agg = _agg()
    s = agg["summary"]
    assert s["row_count"] > 0
    assert agg["arrival_ticks"]
    assert s["dataset"] == az.DATASET_NAME_2024


# 2 ---------------------------------------------------------------------------
def test_schema_validation_catches_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("TIMESTAMP,ContextTokens\n2024-05-10 00:00:00.0+00:00,5\n")
    with pytest.raises(TraceSchemaError):
        list(az._fast_iter_rows(str(bad)))
    bad2 = tmp_path / "bad2.csv"
    bad2.write_text("when,ctx,gen\n2024-05-10 00:00:00.0+00:00,5,3\n")
    with pytest.raises(TraceSchemaError):
        list(az._fast_iter_rows(str(bad2)))


# 3 ---------------------------------------------------------------------------
def test_timestamp_ordering_preserved():
    ticks = _agg()["arrival_ticks"]
    starts = [t.start_s for t in ticks]
    assert starts == sorted(starts)
    assert all(t.tick_index == i for i, t in enumerate(ticks))


# 4 ---------------------------------------------------------------------------
def test_token_fields_map_correctly():
    # ContextTokens -> prompt, GeneratedTokens -> output; total derived
    rows = list(az._fast_iter_rows(FIXTURE))
    assert rows and all(len(r) == 3 for r in rows)
    s = _agg()["summary"]
    assert s["prompt_tokens"]["p50"] is not None
    assert s["output_tokens"]["p50"] is not None
    # total = prompt + output (percentile sanity: total p50 >= each component is
    # not guaranteed, but totals must be positive)
    assert s["total_tokens"]["max"] >= s["prompt_tokens"]["max"]


def test_2024_timestamp_with_tz_offset_parses():
    # the 2024 form carries a +00:00 offset + 6 fractional digits
    a = az.parse_timestamp_s("2024-05-10 00:00:00.009930+00:00")
    b = az.parse_timestamp_s("2024-05-10 00:00:01.009930+00:00")
    assert round(b - a, 4) == 1.0
    # 'Z' suffix and the 2023 7-digit no-tz form both still parse
    assert az.parse_timestamp_s("2024-05-10 00:00:00.5Z") > 0
    assert az.parse_timestamp_s("2023-11-16 18:15:46.6805900") > 0


# 5 ---------------------------------------------------------------------------
def test_missing_latency_not_treated_as_ttft_or_e2e():
    # Azure has no latency column → the normalized request elapsed_s is None
    reqs = az.load_csv(FIXTURE, variant="conv", sample_size=20)
    assert reqs and all(r.elapsed_s is None for r in reqs)
    # and no session/cache key (no cache-affinity claim)
    assert all(r.cache_affinity_key is None for r in reqs)
    # arrival ticks carry zero reuse (no invented cache benefit)
    assert all(t.reuse_fraction == 0.0 for t in _agg()["arrival_ticks"])


# 6 + 12 ----------------------------------------------------------------------
def test_backtest_and_forecast_deterministic():
    ticks = _agg()["arrival_ticks"]
    e1 = fc.run_forecast_experiment(ticks, tick_seconds=60.0, seed=7)
    e2 = fc.run_forecast_experiment(ticks, tick_seconds=60.0, seed=7)
    assert e1.to_dict() == e2.to_dict()
    # moving_average / ewma are deterministic series
    ma1 = fc.forecast_series(ticks, "moving_average", tick_seconds=60.0)
    ma2 = fc.forecast_series(ticks, "moving_average", tick_seconds=60.0)
    ew1 = fc.forecast_series(ticks, "ewma", tick_seconds=60.0)
    ew2 = fc.forecast_series(ticks, "ewma", tick_seconds=60.0)
    assert ma1 == ma2 and ew1 == ew2


# 7 (implicit: all the above use the fixture, not a download) -----------------
def test_unit_tests_do_not_require_full_download():
    # the fixture is small and committed; no network / no raw needed
    assert os.path.exists(FIXTURE)
    assert os.path.getsize(FIXTURE) < 2_000_000


# 8 ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.path.exists(os.path.join(RAW_DIR, "AzureLLMInferenceTrace_code_1week.csv")),
    reason="full Azure 2024 raw trace not present (downloaded-only)")
def test_full_trace_integration_if_raw_present():
    paths = {v: os.path.join(RAW_DIR, f) for v, f in
             (("code", "AzureLLMInferenceTrace_code_1week.csv"),
              ("conv", "AzureLLMInferenceTrace_conv_1week.csv"))
             if os.path.exists(os.path.join(RAW_DIR, f))}
    agg = az.stream_week_aggregate(paths, tick_seconds=60.0)
    s = agg["summary"]
    assert s["row_count"] > 1_000_000
    assert s["duration_days"] >= 6.5  # week-long


# 9 ---------------------------------------------------------------------------
def test_forecast_modes_do_not_leak_future_except_oracle():
    ticks = _agg()["arrival_ticks"]
    actual = [(t.arrival_rate_rps, max(1.0, t.output_tokens_mean) if t.request_count
               else 0.0) for t in ticks]
    # oracle == actual (analysis-only)
    oracle = fc.forecast_series(ticks, "oracle_future", tick_seconds=60.0)
    assert oracle == actual
    # reactive[i] uses actual[i-1] (no leakage); never equals actual[i] when the
    # series actually changes
    react = fc.forecast_series(ticks, "no_forecast_reactive", tick_seconds=60.0)
    assert react[0] == actual[0]
    assert all(react[i] == actual[i - 1] for i in range(1, len(actual)))
    # moving_average[i] depends only on ticks < i
    ma = fc.forecast_series(ticks, "moving_average", tick_seconds=60.0, window=5)
    for i in range(2, len(actual)):
        lo = max(0, i - 5)
        expect = sum(r for r, _ in actual[lo:i]) / len(actual[lo:i])
        assert abs(ma[i][0] - expect) < 1e-9


# 10 + 11 ---------------------------------------------------------------------
def test_alpha_survival_and_oracle_labelled_analysis_only():
    ticks = _agg()["arrival_ticks"]
    exp = fc.run_forecast_experiment(ticks, tick_seconds=60.0)
    assert exp.modes["oracle_future"].analysis_only is True
    assert exp.modes["no_forecast_reactive"].analysis_only is False
    asv = exp.alpha_survival()
    assert "oracle_alpha" in asv and "per_mode" in asv
    # alpha_survival is None (not applicable) when oracle alpha <= 0
    if not asv["oracle_alpha_positive"]:
        assert all(d["alpha_survival_ratio"] is None for d in asv["per_mode"].values())
    else:
        # ratios are computable numbers when oracle alpha > 0
        assert any(d["alpha_survival_ratio"] is not None
                   for d in asv["per_mode"].values())


def test_alpha_survival_math_on_synthetic():
    # construct a tiny experiment-like object to verify the ratio formula
    exp = fc.ForecastExperiment(tick_seconds=60.0)

    class _Stub:
        def __init__(self, kpi):
            self.result = type("R", (), {"kpi": type("K", (), {
                "sla_safe_goodput_per_infra_dollar": kpi})()})()
    exp.modes = {"no_forecast_reactive": _Stub(100.0), "oracle_future": _Stub(200.0),
                 "moving_average": _Stub(150.0)}
    asv = exp.alpha_survival()
    assert asv["oracle_alpha"] == 100.0
    assert asv["per_mode"]["moving_average"]["alpha_survival_ratio"] == 0.5


# 13 + 14 ---------------------------------------------------------------------
def test_docs_state_missing_fields_no_ttft_no_production_claims():
    assert os.path.exists(DOC), "run scripts/run_azure_llm_2024_backtest.py"
    text = open(DOC, encoding="utf-8").read()
    low = text.lower()
    # states missing fields + no TTFT claim
    assert "no ttft" in low or "no_ttft" in low or "no ttft is claimed" in low
    assert "latency / ttft" in low or "latency/ttft" in low
    assert "model / service id" in low or "model/service id" in low
    # banned production-savings claims only in hedged/negated context
    flat = " ".join(low.split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = flat.find(phrase, i)
            if pos == -1:
                break
            pre = flat[max(0, pos - 30):pos]
            assert any(n in pre for n in ("not ", "no ", "never ", "n't ", "without ")), \
                f"unhedged '{phrase}' in {os.path.basename(DOC)}"
            i = pos + len(phrase)


# 15 -- existing trace suites still pass is covered by running the full suite;
# here we assert the shared schema contract the other datasets rely on is intact.
def test_shared_schema_contract_intact():
    from aurelius.traces.schema import NormalizedLLMRequest
    r = az.load_csv(FIXTURE, variant="conv", sample_size=1)[0]
    assert isinstance(r, NormalizedLLMRequest)
    assert r.model == az.DEFAULT_MODEL and r.log_type == "conv"
    assert r.total_tokens == r.prompt_tokens + r.output_tokens
