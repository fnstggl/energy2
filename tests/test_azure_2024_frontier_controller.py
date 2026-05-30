"""Tests for the Azure LLM 2024 — Safe Utilization Frontier Controller v1 driver.

Proves:

- the script runs and writes both the JSON summary and the markdown report;
- frontier sweep covers every required rho;
- constraint_aware baseline is preserved within tolerance vs the committed
  Azure 2024 benchmark JSON;
- the controller's safe-rho choice is on the anticipatory safe frontier;
- the recommended rho beats constraint_aware on goodput/$ in the
  simulator (this is a simulator/shadow finding, NOT a production claim);
- the controller default execution mode is shadow and the summary
  declares real_execution_disabled_by_default=True;
- docs contain no unhedged production-savings claims;
- existing Azure 2024 audit results are untouched (read-only ingestion).
"""

from __future__ import annotations

import json
import os

import pytest

from scripts import run_azure_2024_frontier_controller as fc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")
BACKTEST_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_llm_2024_backtest_summary.json")
FC_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_frontier_controller_summary.json")
FC_MD = os.path.join(REPO_ROOT, "docs",
                     "AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")

EXPECTED_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


# 1 — driver writes both artifacts -----------------------------------------
def test_driver_writes_summary_and_md(tmp_path):
    out_json = str(tmp_path / "summary.json")
    out_md = str(tmp_path / "report.md")
    rc = fc.main(["--audit-json", AUDIT_JSON,
                  "--out-json", out_json, "--out-md", out_md])
    assert rc == 0
    assert os.path.exists(out_json) and os.path.exists(out_md)
    d = json.load(open(out_json))
    assert d["controller_version"] == "frontier_controller_v1"
    assert d["execution_mode_default"] == "shadow"
    assert d["real_execution_disabled_by_default"] is True


# 2 — frontier covers every required rho -----------------------------------
def test_summary_covers_required_rhos():
    d = json.load(open(FC_JSON))
    rhos_antic = {round(p["rho_target"], 2) for p in d["frontier_points_anticipatory"]}
    rhos_reactive = {round(p["rho_target"], 2) for p in d["frontier_points_reactive"]}
    for R in EXPECTED_RHOS:
        assert R in rhos_antic
        assert R in rhos_reactive
    assert d["candidate_rhos"] == list(EXPECTED_RHOS)


# 3 — constraint_aware baseline preserved within tolerance -----------------
def test_constraint_aware_baseline_preserved_within_tolerance():
    fc_d = json.load(open(FC_JSON))
    bt_d = json.load(open(BACKTEST_JSON))
    ca_fc = fc_d["baseline_preserved"]["constraint_aware_goodput_per_dollar"]
    ca_bt = bt_d["base_backtest_primary"]["policies"]["constraint_aware"][
        "sla_safe_goodput_per_infra_dollar"]
    tol = fc_d["baseline_preserved"]["tolerance_pct"] / 100.0
    assert abs(ca_fc - ca_bt) / ca_bt < tol


def test_committed_audit_is_unchanged_by_driver(tmp_path):
    """The driver is read-only on the committed audit JSON."""
    before = json.load(open(AUDIT_JSON))
    fc.main(["--audit-json", AUDIT_JSON,
             "--out-json", str(tmp_path / "x.json"),
             "--out-md", str(tmp_path / "x.md")])
    after = json.load(open(AUDIT_JSON))
    assert before == after


# 4 — selected safe rho is on the anticipatory safe frontier ---------------
def test_selected_rho_is_on_safe_frontier():
    d = json.load(open(FC_JSON))
    decision = d["decision"]
    pts = d["frontier_points_anticipatory"]
    # selected rho must be SAFE in the audit frontier
    safe_rhos = {round(p["rho_target"], 2) for p in pts
                 if p["safety_status"] == "SAFE"}
    assert round(decision["selected_rho"], 2) in safe_rhos
    # the highest UNSAFE rho is NOT selected
    assert round(decision["selected_rho"], 2) != 0.95


def test_safe_peak_is_0_75():
    """The Azure 2024 audit-doc reports anticipatory@0.75 as the safe peak."""
    d = json.load(open(FC_JSON))
    pts = {round(p["rho_target"], 2): p for p in d["frontier_points_anticipatory"]}
    assert pts[0.75]["safety_status"] == "SAFE"
    assert pts[0.85]["safety_status"] != "SAFE"
    # the controller picks 0.75 — the safe peak.
    assert round(d["decision"]["selected_rho"], 2) == 0.75


# 5 — controller beats constraint_aware in simulator -----------------------
def test_controller_beats_constraint_aware_in_simulator():
    d = json.load(open(FC_JSON))
    dlt = d["deltas"]
    # simulator finding only — NOT a production claim
    assert dlt["frontier_vs_constraint_aware_pct"] > 0.0
    assert dlt["frontier_vs_sla_aware_pct"] > 0.0


# 6 — shadow + simulator effects are honest ---------------------------------
def test_shadow_log_summary_has_recommendation_only_invariants():
    d = json.load(open(FC_JSON))
    sl = d["shadow_log_summary"]
    # shadow records are recommendation-only by construction.
    assert sl["n_executed"] == 0
    assert "shadow" in sl["execution_modes"]


def test_simulator_effect_mutates_only_simulated_state():
    d = json.load(open(FC_JSON))
    se = d["simulator_effect"]
    # the simulator effect should reflect mutation of the *simulated* state
    # — not a real cluster (which is verified separately by the unit tests).
    assert se["mode"] == "simulator"
    assert se["mutated"] is True
    assert se["selected_rho"] is not None
    assert "simulator" in " ".join(se["notes"]).lower()


# 7 — docs contain no unhedged production-savings claims --------------------
def test_md_no_unhedged_production_savings_claims():
    assert os.path.exists(FC_MD)
    text = open(FC_MD, encoding="utf-8").read()
    low = " ".join(text.lower().split())
    for phrase in BANNED:
        i = 0
        while True:
            pos = low.find(phrase, i)
            if pos == -1:
                break
            pre = low[max(0, pos - 30):pos]
            assert any(n in pre for n in
                       ("not ", "no ", "never ", "n't ", "without ")), \
                f"unhedged '{phrase}' in {os.path.basename(FC_MD)}"
            i = pos + len(phrase)


def test_md_states_required_caveats():
    low = " ".join(open(FC_MD, encoding="utf-8").read().lower().split())
    assert "simulator" in low and "shadow" in low
    assert "disabled by default" in low
    assert "pilot telemetry" in low


# 8 — controller defaults match spec ----------------------------------------
def test_controller_default_modes_match_spec():
    d = json.load(open(FC_JSON))
    assert d["execution_mode_default"] == "shadow"
    assert d["real_execution_disabled_by_default"] is True
    # The recommendation is *not* executable in real clusters.
    assert d["decision"]["executable_in_real_cluster"] is False
    assert d["decision"]["execution_mode"] in ("shadow", "real_disabled")


# 9 — frontier audit JSON's constraint_aware value still reproduces -------
def test_audit_constraint_aware_matches_committed_backtest():
    audit = json.load(open(AUDIT_JSON))
    bt = json.load(open(BACKTEST_JSON))
    ca_audit = audit["named_policies"]["constraint_aware"]["goodput_per_dollar"]
    ca_bt = bt["base_backtest_primary"]["policies"]["constraint_aware"][
        "sla_safe_goodput_per_infra_dollar"]
    # the existing audit-vs-backtest test already enforces 1% — we double-
    # check here so the frontier controller's "baseline preserved" claim
    # is anchored to the same source.
    assert abs(ca_audit - ca_bt) / ca_bt < 0.01
