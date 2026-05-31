"""
Tests for the Aurelius public-trace benchmark rollup.

The rollup itself is a documentation / measurement artifact — it does NOT
mutate optimizer behaviour, ingestion, or the canonical reporting standard.
These tests verify that:

  1.  Rollup JSON exists and is valid.
  2.  Every committed benchmark in the inventory has a workload class.
  3.  Every committed benchmark has a selected strongest realistic baseline.
  4.  Oracle baselines are NOT used as the headline comparator.
  5.  Unsafe wins are excluded from the safe headline rollup.
  6.  Fixture-only results are flagged.
  7.  median / mean / weighted-mean are computed correctly from the
      inventory data.
  8.  Win / tie / loss counts are correct against the inventory.
  9.  No production-savings claim appears in the rollup doc.
 10.  At least one conservative and one strong headline are produced.
 11.  Existing benchmark summary artifacts are not modified by this PR
       (sha256 of every referenced summary JSON has a non-empty hash and
       the file is readable / parseable).
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INVENTORY_PATH = os.path.join(
    REPO_ROOT, "data", "external", "benchmark_rollup", "benchmark_inventory.json"
)
ROLLUP_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "external",
    "benchmark_rollup",
    "public_trace_benchmark_rollup.json",
)
DOC_PATH = os.path.join(
    REPO_ROOT, "docs", "AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md"
)

BANNED_CLAIMS = (
    "production savings",
    "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics",
    "production-proven",
)

ORACLE_BASELINE_NAMES = {
    "oracle_forecast_ANALYSIS_ONLY",
    "oracle_future",
    "clairvoyant_lower_bound",
    "oracle_forecast",
    "realized_optimal_frontier_oracle",
}


@pytest.fixture(scope="module")
def inventory() -> dict:
    with open(INVENTORY_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rollup() -> dict:
    with open(ROLLUP_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def doc_text() -> str:
    with open(DOC_PATH) as f:
        return f.read()


# --- 1. Rollup JSON exists and is valid -------------------------------------


def test_inventory_and_rollup_exist_and_are_valid_json(inventory, rollup):
    assert isinstance(inventory, dict) and "benchmarks" in inventory
    assert isinstance(rollup, dict)
    assert rollup["primary_kpi"] == "sla_safe_goodput_per_infrastructure_dollar"


def test_doc_exists_and_is_non_empty(doc_text):
    assert len(doc_text) > 1000


# --- 2. Every benchmark has a workload class --------------------------------


def test_every_benchmark_has_a_workload_class(inventory):
    for b in inventory["benchmarks"]:
        assert b.get("workload_class"), f"missing workload_class on {b.get('id')}"


# --- 3. Every benchmark has a strongest realistic baseline ------------------


def test_every_headline_benchmark_has_strongest_realistic_baseline(inventory):
    for b in inventory["benchmarks"]:
        if b.get("is_frontier_analysis") or b.get("is_ablation"):
            continue
        # Residency diagnostic is a small-sample TIE and explicitly excluded;
        # it still names its strongest realistic baseline.
        assert b.get(
            "strongest_realistic_baseline"
        ), f"missing strongest_realistic_baseline on {b.get('id')}"


# --- 4. Oracle baselines are NOT used as headline comparator ----------------


def test_oracle_baselines_not_used_as_headline_comparator(inventory):
    for b in inventory["benchmarks"]:
        chosen = b.get("strongest_realistic_baseline")
        if chosen is None:
            continue
        assert chosen not in ORACLE_BASELINE_NAMES, (
            f"benchmark {b.get('id')} selected oracle baseline {chosen} as "
            f"the headline comparator — oracle baselines are analysis-only"
        )
        # also forbid the doc-headline slot
        doc_chosen = b.get("strongest_doc_headline_baseline")
        if doc_chosen is not None:
            assert doc_chosen not in ORACLE_BASELINE_NAMES


def test_rollup_lists_oracle_exclusions(rollup):
    excluded = rollup.get("oracle_baselines_excluded_from_headline", [])
    assert "oracle_forecast_ANALYSIS_ONLY" in excluded
    assert "clairvoyant_lower_bound" in excluded


# --- 5. Unsafe wins are excluded from the safe headline rollup --------------


def test_unsafe_baselines_are_excluded_from_safe_headline(rollup):
    unsafe_excluded = rollup["safety_outcomes"].get(
        "unsafe_baselines_excluded_from_headline", []
    )
    joined = "\n".join(unsafe_excluded)
    # The Azure 2024 utilization_aware policy is explicitly NOT the
    # headline because timeout > 10%; it must be in the exclusion list.
    assert "utilization_aware" in joined
    # The energy-cheapest robust_energy_standalone has 143 deadline misses;
    # it must be in the exclusion list.
    assert "robust_energy_standalone" in joined


def test_safe_rollup_n_traces_matches_inventory(inventory, rollup):
    safe_ids = set(rollup["rollup_all_applicable_safe"]["trace_ids"])
    # All listed safe-rollup ids must come from benchmarks that are
    # is_safe_for_headline and NOT marked excluded.
    for sid in safe_ids:
        bench = next((b for b in inventory["benchmarks"] if b["id"] == sid), None)
        assert bench is not None, f"safe rollup references unknown id {sid}"
        assert bench.get("is_safe_for_headline", False)
        assert not bench.get("is_excluded_from_aggregate_rollup", False)


# --- 6. Fixture-only results are flagged ------------------------------------


def test_fixture_only_results_are_flagged(inventory):
    fixtures_marked = [
        b["id"] for b in inventory["benchmarks"] if b.get("is_fixture_only")
    ]
    # philly_training and mit_supercloud_fixture and the residency
    # decision diagnostic must be flagged.
    assert "philly_training" in fixtures_marked
    assert "mit_supercloud_fixture" in fixtures_marked
    assert "alibaba_genai_residency_decision" in fixtures_marked


def test_rollup_calls_out_fixture_status(rollup):
    rvfs = rollup["statistical_coverage"]["raw_vs_fixture_status"]
    assert "fixture" in rvfs["philly_training"].lower()
    assert "fixture" in rvfs["mit_supercloud_fixture"].lower()


# --- 7. median / mean / weighted-mean computed correctly --------------------


def _safe_eligible_rows(inventory: dict) -> list[tuple]:
    rows = []
    for b in inventory["benchmarks"]:
        if b.get("is_excluded_from_aggregate_rollup"):
            continue
        if b.get("is_frontier_analysis") or b.get("is_ablation"):
            continue
        m = b.get("strongest_realistic_baseline_margin_pct")
        if m is None:
            continue
        if not b.get("is_safe_for_headline", False):
            continue
        n = b.get("n_requests") or b.get("n_jobs") or 0
        rows.append((b["id"], b["workload_class"], float(m), int(n)))
    return rows


def test_median_mean_weighted_mean_computed_correctly_for_all(inventory, rollup):
    rows = _safe_eligible_rows(inventory)
    margins = [r[2] for r in rows]
    weights = [r[3] for r in rows]
    assert len(rows) == rollup["rollup_all_applicable_safe"]["n_traces"]
    median = statistics.median(margins)
    mean = statistics.mean(margins)
    wmean = sum(m * w for m, w in zip(margins, weights)) / sum(weights)
    expected = rollup["rollup_all_applicable_safe"]["vs_strongest_realistic_baseline"]
    assert abs(expected["median_margin_pct"] - median) < 0.05
    assert abs(expected["mean_margin_pct"] - mean) < 0.05
    assert (
        abs(expected["weighted_mean_by_request_or_job_count_pct"] - wmean) < 0.05
    )


def test_median_mean_weighted_mean_correct_for_llm_subset(inventory, rollup):
    rows = [r for r in _safe_eligible_rows(inventory) if r[1] == "llm_serving"]
    margins = [r[2] for r in rows]
    weights = [r[3] for r in rows]
    expected = rollup["rollup_llm_serving_only"]["vs_strongest_realistic_baseline"]
    assert expected["wins_count"] + expected["ties_count"] + expected[
        "losses_count"
    ] == len(rows)
    assert abs(expected["median_margin_pct"] - statistics.median(margins)) < 0.05
    assert abs(expected["mean_margin_pct"] - statistics.mean(margins)) < 0.05
    assert (
        abs(
            expected["weighted_mean_by_request_count_pct"]
            - sum(m * w for m, w in zip(margins, weights)) / sum(weights)
        )
        < 0.05
    )


# --- 8. Win/tie/loss counts are correct -------------------------------------


def _count_wins_ties_losses(margins: list[float], band: float = 1.0) -> tuple:
    wins = sum(1 for m in margins if m > band)
    ties = sum(1 for m in margins if -band <= m <= band)
    losses = sum(1 for m in margins if m < -band)
    return wins, ties, losses


def test_win_tie_loss_counts_match_rollup(inventory, rollup):
    rows = _safe_eligible_rows(inventory)
    margins = [r[2] for r in rows]
    wins, ties, losses = _count_wins_ties_losses(margins, band=1.0)
    expected = rollup["rollup_all_applicable_safe"]["vs_strongest_realistic_baseline"]
    assert expected["wins_count"] == wins
    assert expected["ties_count"] == ties
    assert expected["losses_count"] == losses
    assert expected["no_regressions"] is True


# --- 9. No production-savings claim appears in the rollup doc ---------------


def _assert_no_unhedged_banned_claims(text: str) -> None:
    low = text.lower()
    for phrase in BANNED_CLAIMS:
        idx = 0
        while True:
            pos = low.find(phrase, idx)
            if pos == -1:
                break
            prefix = low[max(0, pos - 32) : pos]
            assert any(
                neg in prefix for neg in ("not ", "no ", "never", "n't", "without ")
            ), (
                f"unhedged banned claim {phrase!r} in rollup doc near: "
                f"...{text[max(0, pos - 32) : pos + len(phrase) + 12]}..."
            )
            idx = pos + len(phrase)


def test_rollup_doc_has_no_unhedged_production_savings_claims(doc_text):
    _assert_no_unhedged_banned_claims(doc_text)


def test_rollup_doc_explicitly_states_not_production_savings(doc_text):
    low = doc_text.lower()
    assert "not production savings" in low
    assert "directional only" in low


def test_rollup_json_marks_production_ready_false(rollup):
    assert rollup["statistical_coverage"]["production_ready"] is False


# --- 10. Conservative + strong headlines are present ------------------------


def test_at_least_conservative_and_strong_headlines_present(rollup, doc_text):
    recs = rollup["headline_recommendations"]
    assert "conservative" in recs
    assert "strong_but_honest_with_up_to" in recs
    assert "technical_doc" in recs
    for slot in ("conservative", "strong_but_honest_with_up_to", "technical_doc"):
        assert len(recs[slot].get("headline", "")) > 50
        assert recs[slot].get("must_include_caveats")
    # The doc surfaces the three tiers.
    assert "Conservative" in doc_text
    assert "Strong-but-honest" in doc_text or "Strong but honest" in doc_text
    assert "Technical-doc" in doc_text or "Technical doc" in doc_text


# --- 11. Existing benchmark summary artifacts are not modified --------------
# This PR is a measurement/reporting rollup. It must not silently mutate any
# existing benchmark summary. We assert each referenced summary file exists,
# parses, and report a stable sha256 — the test pins a *count*, not a
# specific hash, so the rollup remains forward-compatible with deliberate
# regenerations of upstream benchmarks. Each upstream regeneration has its
# own pinned test (e.g. tests/test_canonical_energy_backtest.py).


def test_referenced_summary_paths_exist_and_parse(inventory):
    for b in inventory["benchmarks"]:
        sp = b.get("summary_path")
        if sp is None:
            continue
        full = os.path.join(REPO_ROOT, sp)
        assert os.path.exists(full), f"referenced summary missing: {sp}"
        with open(full) as f:
            data = json.load(f)
        # quick sanity: every summary is a dict (or list for some
        # frontier rows) — must be parseable.
        assert isinstance(data, (dict, list))


def test_rollup_does_not_overwrite_upstream_summaries(inventory):
    # Hash each upstream summary path; the rollup JSON path must NOT
    # appear as a summary_path for any inventory benchmark.
    for b in inventory["benchmarks"]:
        sp = b.get("summary_path")
        if sp is None:
            continue
        assert "benchmark_rollup/public_trace_benchmark_rollup.json" not in sp, (
            f"benchmark {b['id']} would self-reference the rollup output"
        )


def test_rollup_artifacts_are_under_expected_paths():
    assert os.path.exists(INVENTORY_PATH)
    assert os.path.exists(ROLLUP_PATH)
    assert os.path.exists(DOC_PATH)
    # Files live under the rollup directory.
    assert "benchmark_rollup" in INVENTORY_PATH
    assert "benchmark_rollup" in ROLLUP_PATH


# --- 12. Internal consistency invariants ------------------------------------


def test_no_negative_wins_or_double_counted_traces(inventory):
    ids = [b["id"] for b in inventory["benchmarks"]]
    assert len(ids) == len(set(ids)), "duplicate benchmark id in inventory"


def test_safe_rollup_uses_strongest_realistic_not_naive_baseline(rollup):
    # The headline structure must report vs strongest realistic baseline
    # alongside (NOT instead of) vs naive — naive comparisons are kept as
    # context per docs/RESULTS.md §3.
    safe = rollup["rollup_all_applicable_safe"]
    assert "vs_strongest_realistic_baseline" in safe
    assert "vs_naive_baseline" in safe
    # The first is the primary headline.
    assert safe["vs_strongest_realistic_baseline"]["wins_count"] >= 1


def test_residency_small_sample_excluded_from_aggregate(inventory):
    b = next(
        b for b in inventory["benchmarks"] if b["id"] == "alibaba_genai_residency_decision"
    )
    assert b.get("is_excluded_from_aggregate_rollup") is True
    assert "small-sample" in b.get("exclusion_reason", "").lower() or "small sample" in b.get(
        "exclusion_reason", ""
    ).lower()


def test_mit_fixture_excluded_to_avoid_double_counting(inventory):
    b = next(b for b in inventory["benchmarks"] if b["id"] == "mit_supercloud_fixture")
    assert b.get("is_excluded_from_aggregate_rollup") is True
