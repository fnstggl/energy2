"""Tests for the GenAI 2026 ablation / affinity-audit harness.

Measurement-only harness: it must compose the EXISTING genai_backtest
mechanisms (affinity flag + sizing strategies) and never depend on the full
dataset. Uses the committed fixture.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces import alibaba_genai as ag  # noqa: E402
from aurelius.traces import genai_ablation as abl  # noqa: E402

FIX = os.path.join(REPO_ROOT, "tests", "fixtures", "alibaba_genai_sample")


def _load():
    layers = ag.load_all_layers(FIX, request_kwargs=dict(include_failures=False))
    by_stage = {}
    for e in layers["pipeline"]:
        by_stage.setdefault(e.stage, []).append(e)
    return layers["requests"], ag.calibrate_cold_start(by_stage)


def test_all_ablation_configs_run():
    reqs, cold = _load()
    res = abl.run_ablation(reqs, tick_seconds=3600.0, cold_start_s=cold)
    expected = {n for n, _, _ in abl.ABLATION_CONFIGS}
    assert set(res) == expected
    # requested ablations are all present
    for name in ("fifo", "fifo_plus_affinity", "sla_aware", "sla_aware_plus_affinity",
                 "constraint_aware", "constraint_aware_no_affinity",
                 "queue_aware", "utilization_aware"):
        assert name in res
        assert res[name].goodput_per_dollar is not None


def test_affinity_reduces_cold_start():
    reqs, cold = _load()
    res = abl.run_ablation(reqs, tick_seconds=3600.0, cold_start_s=cold)
    # affinity ON must not increase modelled cold-start vs the OFF counterpart
    pairs = [("fifo", "fifo_plus_affinity"),
             ("sla_aware", "sla_aware_plus_affinity"),
             ("constraint_aware_no_affinity", "constraint_aware")]
    for off, on in pairs:
        assert res[on].mean_cold_start_s <= res[off].mean_cold_start_s + 1e-9
        assert res[on].affinity is True
        assert res[off].affinity is False


def test_prewarm_equals_affinity_documented():
    reqs, cold = _load()
    res = abl.run_ablation(reqs, tick_seconds=3600.0, cold_start_s=cold)
    attr = abl.attribute(res)
    assert attr["prewarm_equals_affinity"] is True
    sf = attr["single_factor_lift_vs_fifo_pct"]
    # prewarm-alone is defined identically to affinity-alone
    assert sf["prewarm_alone"] == sf["model_affinity_alone"]


def test_attribution_shares_sum_to_100():
    reqs, cold = _load()
    res = abl.run_ablation(reqs, tick_seconds=3600.0, cold_start_s=cold)
    attr = abl.attribute(res)
    sh = attr["shapley_attribution_of_ca_vs_sla_gain"]
    total = (sh["affinity_share_pct"] + sh["sizing_share_pct"]
             + sh["interaction_share_pct"])
    # the tiny fixture may be under-loaded (no gain to attribute -> all 0 shares);
    # otherwise the Shapley shares must partition the gain (sum ~ 100%).
    if abs(attr["constraint_aware_vs_sla_aware_gain_pct"]) > 1.0:
        assert abs(total - 100.0) < 0.5
    else:
        assert abs(total) < 0.5 or abs(total - 100.0) < 0.5



def test_deterministic():
    reqs, cold = _load()
    a = abl.run_ablation(reqs, tick_seconds=3600.0, cold_start_s=cold)
    b = abl.run_ablation(reqs, tick_seconds=3600.0, cold_start_s=cold)
    assert {n: r.summary() for n, r in a.items()} == {n: r.summary() for n, r in b.items()}


def test_no_production_logic_imported_only():
    # the harness must only reuse genai_backtest primitives (no new policies)
    import inspect

    from aurelius.traces import genai_backtest as gb
    src = inspect.getsource(abl)
    # it calls the existing sizing/eval primitives
    for prim in ("_size_for_sla", "_size_for_target", "_eval_tick", "_aggregate_ticks"):
        assert hasattr(gb, prim) and prim in src


def test_docs_exist_and_honest():
    md = os.path.join(REPO_ROOT, "docs", "ALIBABA_GENAI_ABLATION_RESULTS.md")
    assert os.path.exists(md)
    text = open(md).read().lower()
    assert "not production" in text
    assert "prewarm" in text and "affinity" in text
    # must NOT contain unhedged production-savings claims
    for phrase in ("production savings", "production-proven"):
        idx = 0
        while True:
            pos = text.find(phrase, idx)
            if pos == -1:
                break
            pre = text[max(0, pos - 24):pos]
            assert any(n in pre for n in ("not ", "no ", "never", "n't"))
            idx = pos + len(phrase)
