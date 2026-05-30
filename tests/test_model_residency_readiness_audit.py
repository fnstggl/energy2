"""Validation tests for the Model-Residency readiness audit (docs + summary).

Docs-only / measurement-only validation: assert the audit doc has the required
sections (inventory, benchmark table, conformance matrix, dataset coverage,
verdict, next tasks), that its verdict matches the generated summary JSON, that
the conformance matrix covers every required spec requirement, and that there
are no unhedged production-savings claims.
"""

from __future__ import annotations

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(REPO_ROOT, "docs", "MODEL_RESIDENCY_READINESS_AUDIT.md")
SUMMARY = os.path.join(REPO_ROOT, "data", "external", "alibaba_genai", "processed",
                       "model_residency_audit_summary.json")
SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_model_residency_audit.py")

VALID_VERDICTS = ("SPEC_ONLY", "SIMULATOR_APPROXIMATION",
                  "TRACE_BACKTESTED_APPROXIMATION",
                  "SHADOW_PILOT_READY_READ_ONLY", "PRODUCTION_READY")
BANNED = ("production savings", "guaranteed savings",
          "enterprise-ready autonomous optimization",
          "hyperscaler-validated economics", "production-proven")


def _read(p):
    assert os.path.exists(p), f"missing: {p}"
    return open(p).read()


def test_audit_doc_exists_with_required_sections():
    text = _read(AUDIT).lower()
    for section in ("implementation inventory", "benchmark", "conformance matrix",
                    "dataset coverage", "readiness verdict",
                    "next engineering tasks"):
        assert section in text, f"audit doc missing section: {section}"


def test_summary_json_exists_and_valid_verdict():
    assert os.path.exists(SUMMARY), "run scripts/run_model_residency_audit.py"
    d = json.load(open(SUMMARY))
    assert d["readiness_verdict"] in VALID_VERDICTS
    # the audit must be conservative — not claiming pilot/production readiness
    assert d["readiness_verdict"] == "TRACE_BACKTESTED_APPROXIMATION"
    assert d["directional_only_not_production_savings"] is True


def test_doc_and_summary_verdict_agree():
    d = json.load(open(SUMMARY))
    assert d["readiness_verdict"] in _read(AUDIT)


def test_conformance_matrix_covers_required_requirements():
    text = _read(AUDIT)
    for req in ("model_id", "adapter_id", "model_loaded_before_request",
                "adapter_loaded_before_request", "model load start/end",
                "adapter load start/end", "residency hit rate", "cold-start rate",
                "cold-start p95/p99", "warm-pool cost", "no-substitution",
                "preserve-affinity decision", "prewarm recommendation",
                "shadow-mode logging"):
        assert req in text, f"conformance matrix missing requirement: {req}"


def test_inventory_classifies_real_vs_simulated():
    text = _read(AUDIT)
    # must distinguish the one real signal (prefix-cache hit rate) from the
    # missing model-load signal — no overstating
    assert "prefix_cache_hit_rate" in text
    assert "model_loaded_before_request" in text
    assert "MISSING" in text or "missing" in text
    # must name the real recommendation-only actions
    assert "PRESERVE_AFFINITY" in text and "PREWARM_REPLICA" in text


def test_affinity_dependence_quantified():
    d = json.load(open(SUMMARY))["genai_2026_affinity_dependence"]
    assert d["affinity_prewarm_share_pct"] is not None
    ww = d["with_vs_without"]
    # with-affinity goodput/$ must exceed without-affinity (the dependence)
    assert (ww["with_affinity_prewarm"]["goodput_per_dollar"]
            > ww["without_affinity_prewarm"]["goodput_per_dollar"])
    assert d["cold_start_calibration_s"].get("basemodel_load", 0) > 0


def test_dataset_coverage_marks_azure_not_applicable():
    d = json.load(open(SUMMARY))
    cov = {row["dataset"]: row for row in d["dataset_coverage"]}
    assert "NOT APPLICABLE" in cov["azure_llm"]["can_measure_vs_simulate"].upper()
    # the live connector row must state model residency is NOT measured
    assert cov["live_vllm_connector"]["per_request_residency_hit"].upper().startswith("NO")


def test_biggest_missing_pieces_listed():
    d = json.load(open(SUMMARY))
    miss = " ".join(d["biggest_missing_pieces"]).lower()
    assert "model_loaded_before_request" in miss or "model-load" in miss
    assert "adapter" in miss or "lora" in miss


def test_summary_is_reproducible():
    # the generator reads committed summaries only (no optimizer logic / dataset)
    import runpy
    import sys
    out = SUMMARY + ".regen"
    argv = sys.argv
    try:
        sys.argv = ["run_model_residency_audit.py", "--out", out]
        runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    a = json.load(open(SUMMARY))
    b = json.load(open(out))
    os.remove(out)
    assert a["readiness_verdict"] == b["readiness_verdict"]
    assert (a["genai_2026_affinity_dependence"]["affinity_prewarm_share_pct"]
            == b["genai_2026_affinity_dependence"]["affinity_prewarm_share_pct"])


def test_no_unhedged_production_savings_claims():
    for path in (AUDIT,):
        text = _read(path)
        low = " ".join(text.lower().split())
        for phrase in BANNED:
            i = 0
            while True:
                pos = low.find(phrase, i)
                if pos == -1:
                    break
                pre = low[max(0, pos - 30):pos]
                assert any(n in pre for n in ("not ", "no ", "never ", "n't ",
                                              "without ")), \
                    f"unhedged '{phrase}' in {os.path.basename(path)}"
                i = pos + len(phrase)
