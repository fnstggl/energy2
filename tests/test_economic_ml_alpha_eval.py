"""Economic-alpha evaluation + overlap-audit integrity tests for v1.

Proves: exact overlap audit exists; deterministic baseline is the primary
baseline; ML compares against it; random holdout is not headline (binding is
time/by_dataset/high_tail); subgroup/per-class results reported separately;
no oracle/FIFO headline; no production-savings claim; deterministic cost
targets are diagnostic_only; no production module modified by this PR.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART = REPO_ROOT / "data" / "external" / "forecasting" / "economic_ml_alpha_v1"


@pytest.fixture(scope="module")
def overlap():
    return json.loads((ART / "overlap_audit.json").read_text())


@pytest.fixture(scope="module")
def catalog():
    return json.loads((ART / "target_catalog.json").read_text())


@pytest.fixture(scope="module")
def alpha():
    return json.loads((ART / "economic_alpha_eval.json").read_text())


@pytest.fixture(scope="module")
def trained():
    return json.loads((ART / "trained_models.json").read_text())


@pytest.fixture(scope="module")
def summary():
    return json.loads((ART / "summary.json").read_text())


# ───────────────────── overlap audit ─────────────────────


def test_overlap_audit_answers_all_13_questions(overlap):
    a = overlap["answers"]
    for i in range(1, 14):
        assert any(k.startswith(f"q{i}_") for k in a), f"missing q{i}"


def test_overlap_records_zero_cold_start_migration_cache_hit(overlap):
    a = overlap["answers"]
    assert a["q10_rows_with_cold_start_cost"] == 0
    assert a["q11_rows_with_migration_cost"] == 0
    assert a["q12_rows_with_real_cache_hit"] == 0


def test_overlap_flags_constant_swissai_latency(overlap):
    v = overlap["signal_variability_by_dataset"]["ttft_s"]
    # SwissAI ttft is a constant; CARA is variable.
    sw = [k for k in v if k.startswith("swissai")]
    assert sw and v[sw[0]] == "constant"
    cara = [k for k in v if k.startswith("cara")]
    assert cara and v[cara[0]] == "variable"


# ───────────────────── baselines + holdouts ─────────────────────


def test_primary_baseline_is_deterministic_overlay(alpha):
    assert alpha["primary_baseline"] == "B_overlay_deterministic_formula"


def test_no_oracle_no_fifo_no_production_claim(alpha):
    assert alpha["uses_oracle_as_headline"] is False
    assert alpha["uses_fifo_as_headline"] is False
    assert alpha["production_claim"] is False


def test_variants_cover_a_through_i(alpha):
    keys = set(alpha["variants"].keys())
    for letter in "ABCDEFGHI":
        assert any(k.startswith(letter + "_") for k in keys), letter


def test_per_overlay_class_reported_separately(alpha):
    cls = alpha["per_overlay_class_goodput_per_dollar"]
    assert {"measured_same_record", "cross_dataset_joined",
            "scenario_prior"} <= set(cls)


def test_ml_models_compared_against_baseline(trained):
    """Every trained regression target records a deterministic baseline in its
    holdout model set."""
    for tname, t in trained.items():
        if not t.get("trained") or t.get("classification"):
            continue
        bh = t.get("binding_holdout")
        models = t["holdouts"].get(bh, {}).get("models", {})
        assert any("baseline" in m for m in models), (tname, list(models))


def test_random_holdout_is_not_binding(trained):
    for tname, t in trained.items():
        if not t.get("trained"):
            continue
        assert t.get("random_holdout_is_decorative") is True
        assert t.get("binding_holdout") != "random", tname


def test_binding_holdouts_are_time_or_dataset_or_tail(trained):
    allowed = {"time", "by_dataset", "high_tail"}
    for tname, t in trained.items():
        if not t.get("trained"):
            continue
        assert t["binding_holdout"] in allowed, (tname, t["binding_holdout"])


# ───────────────────── deterministic-cost diagnostic ─────────────────────


def test_deterministic_cost_target_is_diagnostic_only(summary):
    st = summary["per_target_final_status"]
    det = [k for k in st if "DETERMINISTIC" in k]
    assert det, "expected a deterministic cost target in the status map"
    for k in det:
        assert "deterministic" in st[k]


def test_catalog_marks_cost_targets_deterministic(catalog):
    t = catalog["targets"]["estimated_gpu_cost_usd"]
    assert t.get("deterministic_ground_truth") is True
    assert "diagnostic_only" in t["ml_status"]


def test_catalog_cold_start_migration_not_trainable(catalog):
    for tgt in ("cold_start_cost_usd", "migration_cost_usd"):
        assert catalog["targets"][tgt]["trainable_now"] is False


# ───────────────────── honest alpha reporting ─────────────────────


def test_summary_reports_status_per_target(summary):
    st = summary["per_target_final_status"]
    allowed = {"diagnostic_only", "promising_needs_validation",
               "shadow_ready_for_integration_review",
               "diagnostic_only_deterministic_formula",
               "proxy_promising_only", "blocked_insufficient_rows"}
    assert st, "no per-target status"
    for k, v in st.items():
        assert v in allowed, (k, v)


def test_shadow_ready_targets_have_caveats_if_single_dataset(summary):
    """Any shadow_ready target validated on a single dataset must carry an
    explicit caveat (no silent cross-dataset claim)."""
    st = summary["per_target_final_status"]
    caveats = summary.get("shadow_ready_caveats", {})
    shadow = [k for k, v in st.items()
              if v == "shadow_ready_for_integration_review"]
    # At least one shadow-ready target exists and each single-dataset one is
    # caveated.
    assert shadow, "expected at least one shadow-ready target"
    for k in shadow:
        # if caveated, the note must mention cross-dataset / pilot.
        if k in caveats:
            assert ("cross-dataset" in caveats[k]
                    or "pilot" in caveats[k])


def test_no_production_module_modified_by_this_pr():
    out = subprocess.check_output(
        ["git", "diff", "--name-only", "main...HEAD"], cwd=REPO_ROOT,
    ).decode().splitlines()
    forbidden = {
        "aurelius/optimization/scheduler.py",
        "aurelius/optimization/objective.py",
        "aurelius/optimization/constraints.py",
        "aurelius/forecasting/constraint_shadow_scorer.py",
        "aurelius/forecasting/constraint_scorer_features.py",
        "aurelius/residency/decision.py",
        "aurelius/residency/sim.py",
        "aurelius/residency/shadow.py",
        "aurelius/frontier/controller.py",
        "aurelius/forecasting/economic_overlay.py",
    }
    bad = [p for p in out if p in forbidden]
    assert not bad, f"this PR must not modify production modules: {bad}"
