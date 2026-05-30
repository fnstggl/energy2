"""Tests for the Azure 2024 safe-utilization frontier audit tool.

Measurement / attribution artifact. Unit tests run the tool in-process on the
committed sample fixture (no full download needed); value tests validate the
committed full-trace summary JSON + doc. They prove the frontier sweep covers
the required rho targets, that constraint_aware matches the committed Azure 2024
benchmark within tolerance, that safe/unsafe points are identified, that the
forecasting contribution is reported SEPARATELY from the utilization
contribution, that the tool does NOT mutate optimizer behavior/defaults, and
that the docs make no production-savings claims.
"""

from __future__ import annotations

import json
import os

from aurelius.traces import azure_llm as az
from aurelius.traces import backtest as bt
from scripts import run_azure_2024_safe_utilization_frontier as fr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv")
FRONTIER_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")
BACKTEST_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_llm_2024_backtest_summary.json")
DOC = os.path.join(REPO_ROOT, "docs", "AZURE_2024_SAFE_UTILIZATION_FRONTIER.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")

EXPECTED_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


def _fixture_ticks():
    return az.stream_week_aggregate({"conv": FIXTURE}, tick_seconds=60.0)["arrival_ticks"]


# 1 — the script runs on fixture (and cached) data -------------------------
def test_tool_runs_on_fixture():
    payload = fr.build(_fixture_ticks(), scale=200.0, tick_seconds=60.0)
    assert payload["frontier_reactive"] and payload["frontier_anticipatory"]
    assert payload["named_policies"] and payload["decomposition"]


# 2 — frontier sweep includes the required rho targets ---------------------
def test_frontier_sweep_covers_required_rhos():
    assert fr.RHOS == EXPECTED_RHOS
    payload = fr.build(_fixture_ticks(), scale=200.0, tick_seconds=60.0)
    for R in EXPECTED_RHOS:
        assert any(f"@{R}" in m["policy"] for m in payload["frontier_reactive"])
        assert any(f"@{R}" in m["policy"] for m in payload["frontier_anticipatory"])
    # committed full-trace artifact also covers them
    d = json.load(open(FRONTIER_JSON))
    for R in EXPECTED_RHOS:
        assert any(f"@{R}" in m["policy"] for m in d["frontier_reactive"])


# 3 — constraint_aware matches the committed Azure 2024 benchmark ----------
def test_constraint_aware_matches_committed_backtest():
    fr_d = json.load(open(FRONTIER_JSON))
    ca_frontier = fr_d["named_policies"]["constraint_aware"]["goodput_per_dollar"]
    bt_d = json.load(open(BACKTEST_JSON))
    ca_bench = bt_d["base_backtest_primary"]["policies"]["constraint_aware"][
        "sla_safe_goodput_per_infra_dollar"]
    # same harness/scale → should match within a tight tolerance
    assert abs(ca_frontier - ca_bench) / ca_bench < 0.01
    # decomposition's CA equals the named CA
    assert abs(fr_d["decomposition"]["constraint_aware_gpd"] - ca_frontier) < 1.0


# 4 — best safe rho is identified ------------------------------------------
def test_best_safe_rho_identified():
    d = json.load(open(FRONTIER_JSON))
    fs = d["frontier_summary"]
    assert fs["reactive"]["best_safe"] is not None
    assert fs["anticipatory"]["best_safe"] is not None
    # the anticipatory frontier's best safe point is at a higher rho than CA's
    # default (the headline "near but inside" finding)
    assert "@0.75" in fs["anticipatory"]["best_safe"]
    assert fs["constraint_aware_is"].startswith("inside")


# 5 — unsafe rho points are marked unsafe ----------------------------------
def test_unsafe_points_marked():
    d = json.load(open(FRONTIER_JSON))
    react = {m["policy"]: m for m in d["frontier_reactive"]}
    # high rho saturates → unsafe (timeout/queue breach)
    assert react["reactive@0.95"]["safe"] is False
    assert any(not m["safe"] for m in d["frontier_reactive"])
    # low rho is safe
    assert react["reactive@0.45"]["safe"] is True


# 6 — forecasting reported separately from utilization ---------------------
def test_forecasting_separate_from_utilization():
    d = json.load(open(FRONTIER_JSON))
    dc = d["decomposition"]
    # utilization lever (rho step) and forecasting lever are distinct fields
    assert "step_raise_rho_0.50_to_0.65" in dc
    assert "forecast_contribution_pct_of_kpi" in dc
    assert "forecast_alpha" in d
    # utilization step dominates; forecasting is a tiny % of KPI
    assert dc["step_raise_rho_0.50_to_0.65"] > 0
    assert abs(dc["forecast_contribution_pct_of_kpi"]) < 5.0  # << utilization


# 7 — docs contain no unhedged production-savings claims -------------------
def test_docs_no_production_savings_claims():
    assert os.path.exists(DOC), "run scripts/run_azure_2024_safe_utilization_frontier.py"
    text = open(DOC, encoding="utf-8").read()
    low = " ".join(text.lower().split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 30):pos]
            assert any(n in pre for n in ("not ", "no ", "never ", "n't ", "without ")), \
                f"unhedged '{phrase}' in {os.path.basename(DOC)}"
            i = pos + len(phrase)


def test_docs_state_required_caveats():
    low = " ".join(open(DOC, encoding="utf-8").read().lower().split())
    assert "simulator" in low and "public-trace" in low
    assert "inside the safe frontier" in low or "inside (conservative)" in low
    assert "do not change the production default rho" in low or "blindly set rho" in low
    assert "pilot" in low and "calibrate" in low and "safe rho" in low


# 8 — no optimizer behavior / defaults changed -----------------------------
def test_no_optimizer_behavior_or_defaults_changed():
    # the engine's documented rho defaults the tool MIRRORS (it never sets them):
    # sla_aware=0.50, constraint_aware=0.65. The tool only reads bt internals.
    assert fr.SLA_AWARE_RHO == 0.50 and fr.CA_RHO == 0.65
    # running the tool must NOT mutate bt's optimizer behavior: a constraint_aware
    # run via the UNCHANGED bt path is identical before and after build().
    ticks = _fixture_ticks()
    th = 60.0 / 3600.0
    before = bt._run_policy("constraint_aware", ticks, tick_hours=th)
    fr.build(ticks, scale=50.0, tick_seconds=60.0)
    after = bt._run_policy("constraint_aware", ticks, tick_hours=th)
    assert (before.kpi.sla_safe_goodput_per_infra_dollar
            == after.kpi.sla_safe_goodput_per_infra_dollar)
    assert bt.MIN_REPLICAS == 1  # engine constant untouched


# 9 — committed JSON/doc are internally consistent -------------------------
def test_committed_artifacts_consistent():
    d = json.load(open(FRONTIER_JSON))
    assert d["decomposition"]["total_win_pct"] > 0
    # safe thresholds are documented
    assert d["safe_thresholds"]["timeout_pct"] == 10.0
    assert d["safe_thresholds"]["queue_p99_ms"] == 2000.0
