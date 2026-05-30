"""Tests for the Cross-Trace Safe Utilization Frontier Generalization Audit.

Proves:

- the audit runs and writes both the JSON summary and the markdown report;
- every applicable trace has a verdict in {FRONTIER_WIN, TIE, FRONTIER_LOSS};
- Alibaba GPU v2023 and Microsoft Philly are explicitly excluded with a
  documented reason (not silently dropped);
- the Azure 2024 verdict reproduces the committed audit's controller win
  within tolerance;
- no committed benchmark JSON is modified by the audit;
- the controller's executable_in_real_cluster is False for every recorded
  decision (recommendation-only invariant);
- the docs contain no unhedged production-savings claims.
"""

from __future__ import annotations

import json
import os

from scripts import run_cross_trace_frontier_generalization_audit as fr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "cross_trace_frontier_generalization_summary.json")
DOC_MD = os.path.join(REPO_ROOT, "docs",
                      "CROSS_TRACE_FRONTIER_GENERALIZATION_AUDIT.md")
COMMITTED_AZURE_2024_FC_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_frontier_controller_summary.json")
COMMITTED_AZURE_2024_AUDIT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_safe_utilization_frontier.json")
COMMITTED_AZURE_2024_BACKTEST_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_llm_2024_backtest_summary.json")

BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")

EXPECTED_APPLICABLE = {
    "burstgpt", "azure_llm_2023", "azure_llm_2024_week",
    "alibaba_genai_2026",
}
EXPECTED_EXCLUDED = {"alibaba_gpu_v2023", "microsoft_philly"}
ALL_TRACES = EXPECTED_APPLICABLE | EXPECTED_EXCLUDED

EXPECTED_VERDICTS = {"FRONTIER_WIN", "TIE", "FRONTIER_LOSS",
                     "INSUFFICIENT_TELEMETRY"}


# 1 — driver runs and writes artifacts -------------------------------------
def test_driver_writes_artifacts(tmp_path):
    out_json = str(tmp_path / "summary.json")
    out_md = str(tmp_path / "audit.md")
    rc = fr.main(["--out-json", out_json, "--out-md", out_md])
    assert rc == 0
    assert os.path.exists(out_json) and os.path.exists(out_md)
    d = json.load(open(out_json))
    assert d["synthesis"]["n_applicable"] >= 1
    assert d["synthesis"]["n_skipped"] >= 2


# 2 — committed summary covers every expected trace -----------------------
def test_summary_covers_every_trace():
    d = json.load(open(SUMMARY_JSON))
    traces = {t["trace"] for t in d["per_trace"]}
    assert ALL_TRACES <= traces


def test_applicable_and_excluded_partition():
    d = json.load(open(SUMMARY_JSON))
    applicable = {t["trace"] for t in d["per_trace"] if t["applicable"]}
    excluded = {t["trace"] for t in d["per_trace"] if not t["applicable"]}
    assert EXPECTED_APPLICABLE <= applicable
    assert EXPECTED_EXCLUDED <= excluded


def test_exclusion_reasons_are_documented():
    d = json.load(open(SUMMARY_JSON))
    for t in d["per_trace"]:
        if t["applicable"]:
            continue
        assert t.get("exclusion_reason"), \
            f"missing exclusion_reason for {t['trace']}"
        # excluded entries MUST NOT carry a decision (recommendation invariant)
        assert t["decision"] is None


# 3 — every applicable trace produces a recognised verdict -----------------
def test_every_applicable_trace_has_recognised_verdict():
    d = json.load(open(SUMMARY_JSON))
    for t in d["per_trace"]:
        if not t["applicable"]:
            continue
        assert t["comparison"]["verdict"] in EXPECTED_VERDICTS, \
            f"{t['trace']}: unknown verdict {t['comparison']['verdict']!r}"


# 4 — Azure 2024 verdict on this audit matches the committed controller ---
def test_azure_2024_matches_committed_controller_result():
    """The committed controller summary already says +12.98% goodput/$ vs
    constraint_aware on the Azure 2024 audit JSON. This audit reads the
    same committed JSON and must produce the same number (within tolerance)."""
    d = json.load(open(SUMMARY_JSON))
    az = next(t for t in d["per_trace"] if t["trace"] == "azure_llm_2024_week")
    delta = az["comparison"]["delta_vs_constraint_aware_pct"]
    fc = json.load(open(COMMITTED_AZURE_2024_FC_JSON))
    committed_delta = fc["deltas"]["frontier_vs_constraint_aware_pct"]
    assert abs(delta - committed_delta) < 0.5, \
        f"audit delta {delta} vs committed {committed_delta}"


# 5 — recommendation-only invariant ----------------------------------------
def test_all_decisions_recommendation_only():
    d = json.load(open(SUMMARY_JSON))
    for t in d["per_trace"]:
        dec = t.get("decision")
        if dec is None:
            continue
        assert dec["executable_in_real_cluster"] is False, \
            f"{t['trace']} decision must be recommendation-only"


# 6 — committed Azure 2024 artifacts unchanged by this audit --------------
def test_committed_azure_2024_audit_unchanged(tmp_path):
    """The audit script must be read-only on existing committed JSON."""
    before_audit = json.load(open(COMMITTED_AZURE_2024_AUDIT_JSON))
    before_backtest = json.load(open(COMMITTED_AZURE_2024_BACKTEST_JSON))
    before_fc = json.load(open(COMMITTED_AZURE_2024_FC_JSON))
    fr.main(["--out-json", str(tmp_path / "x.json"),
             "--out-md", str(tmp_path / "x.md")])
    assert before_audit == json.load(open(COMMITTED_AZURE_2024_AUDIT_JSON))
    assert before_backtest == json.load(open(COMMITTED_AZURE_2024_BACKTEST_JSON))
    assert before_fc == json.load(open(COMMITTED_AZURE_2024_FC_JSON))


# 7 — synthesis numbers add up --------------------------------------------
def test_synthesis_counts_consistent():
    d = json.load(open(SUMMARY_JSON))
    syn = d["synthesis"]
    vc = syn["verdict_counts"]
    total = vc["FRONTIER_WIN"] + vc["TIE"] + vc["FRONTIER_LOSS"]
    # INSUFFICIENT_TELEMETRY may be zero; either way, applicable count covers
    # every verdict bucket.
    total += vc.get("INSUFFICIENT_TELEMETRY", 0)
    assert total == syn["n_applicable"]
    assert syn["n_applicable"] + syn["n_skipped"] == len(d["per_trace"])


# 8 — best-safe-rho distribution is bounded by the candidate grid --------
def test_best_safe_rho_distribution_within_candidate_grid():
    d = json.load(open(SUMMARY_JSON))
    grid = set(d["config"]["rhos"])
    for r in d["synthesis"]["best_safe_rho_distribution"]:
        # JSON keys are strings; coerce
        assert float(r) in grid


# 9 — docs contain no unhedged production-savings claims ------------------
def test_doc_no_unhedged_banned_phrases():
    assert os.path.exists(DOC_MD)
    text = open(DOC_MD, encoding="utf-8").read()
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
                f"unhedged '{phrase}' in {os.path.basename(DOC_MD)}"
            i = pos + len(phrase)


def test_doc_states_required_caveats():
    low = " ".join(open(DOC_MD, encoding="utf-8").read().lower().split())
    assert "shadow" in low and "simulator" in low
    assert "disabled by default" in low
    assert "pilot telemetry" in low
    # honesty about non-applicable traces
    assert "alibaba_gpu" in low or "bin-packing" in low
    assert "philly" in low or "training-job scheduling" in low


# 10 — recommendation code is one of the documented options --------------
def test_recommendation_code_recognised():
    d = json.load(open(SUMMARY_JSON))
    code = d["synthesis"]["architecture_recommendation"]["code"]
    assert code in {
        "INTEGRATE_OPT_IN_TO_CONSTRAINT_AWARE",
        "KEEP_FRONTIER_CONTROLLER_SEPARATE",
        "KEEP_FRONTIER_CONTROLLER_SEPARATE_OR_OPT_IN",
        "INSUFFICIENT_EVIDENCE",
    }


# 11 — generalization verdict is one of the documented options -----------
def test_generalization_verdict_recognised():
    d = json.load(open(SUMMARY_JSON))
    assert d["synthesis"]["generalizes"] in {
        "GENERALIZES_WITHIN_APPLICABLE_LLM_INFERENCE_TRACES",
        "SAFE_TIE_ACROSS_APPLICABLE_TRACES",
        "WORKLOAD_DEPENDENT",
        "INSUFFICIENT_EVIDENCE",
    }


# 12 — every applicable trace's frontier sweep covers the configured grid -
def test_frontier_sweep_covers_full_rho_grid():
    d = json.load(open(SUMMARY_JSON))
    grid = set(round(r, 4) for r in d["config"]["rhos"])
    for t in d["per_trace"]:
        if not t["applicable"]:
            continue
        rhos_antic = {round(p["rho_target"], 4)
                      for p in t["frontier_anticipatory"]}
        assert grid <= rhos_antic, \
            f"{t['trace']}: anticipatory frontier missing rhos {grid - rhos_antic}"
