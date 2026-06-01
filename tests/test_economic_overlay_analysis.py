"""Analysis-tier Economic Overlay tests.

Proves on the committed analysis summary + eval + market manifest:
  - no secrets printed/committed anywhere under the overlay dir or scripts;
  - no invented constants in the analysis build/eval scripts;
  - all scenario fields are explicitly labelled scenario_prior;
  - live PJM/ERCOT/CAISO/WattTime success/failure is recorded in the manifest;
  - analysis-tier row counts are sufficient (>= 50k total; >= 50k for the
    CARA + SwissAI scaling targets combined);
  - economic-target coverage is non-zero and subgroup counts are reported;
  - the A-H eval ran all 8 variants and recorded a known promotion state;
  - carbon cost stays operator-policy-only (0 rows under default policy);
  - raw downloads + full corpus remain gitignored; committed samples bounded.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OVERLAY = REPO_ROOT / "data" / "external" / "economic_overlay"
SUMMARY = OVERLAY / "economic_overlay_analysis_summary.json"
EVAL = OVERLAY / "economic_overlay_analysis_eval.json"
MANIFEST = OVERLAY / "market_fetch_manifest.json"
DOC = REPO_ROOT / "docs" / "ECONOMIC_OVERLAY_LAYER_V1_ANALYSIS.md"
SCRIPTS = [
    REPO_ROOT / "scripts" / "build_economic_overlay_analysis.py",
    REPO_ROOT / "scripts" / "run_economic_overlay_analysis_eval.py",
    REPO_ROOT / "scripts" / "fetch_economic_overlay_analysis_sources.py",
    REPO_ROOT / "scripts" / "fetch_economic_overlay_market_data.py",
]

# Credential VALUES are read from the environment at test time so no literal
# is ever stored in this repo. When an env var is set, its exact value must
# not appear in any committed file. Pattern checks run regardless.
_SECRET_ENV_VARS = [
    "PJM_API_KEY", "ERCOT_API_KEY", "ERCOT_PASSWORD", "ERCOT_USERNAME",
    "WATTTIME_PASSWORD", "WATTTIME_USERNAME", "HF_TOKEN",
]


def _secret_values() -> list[str]:
    vals = []
    for k in _SECRET_ENV_VARS:
        v = os.environ.get(k)
        # Skip trivially-short or absent values to avoid false positives.
        if v and len(v) >= 5:
            vals.append(v)
    return vals


SECRET_PATTERNS = [
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b"),
]


@pytest.fixture(scope="module")
def summary() -> dict:
    assert SUMMARY.exists(), f"missing analysis summary: {SUMMARY}"
    return json.loads(SUMMARY.read_text())


@pytest.fixture(scope="module")
def eval_data() -> dict:
    assert EVAL.exists(), f"missing analysis eval: {EVAL}"
    return json.loads(EVAL.read_text())


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST.exists(), f"missing market manifest: {MANIFEST}"
    return json.loads(MANIFEST.read_text())


# ───────────────────────── 1. No secrets committed ─────────────────────────


def _iter_committed_text_files():
    out = subprocess.check_output(
        ["git", "ls-files",
         "data/external/economic_overlay", "scripts", "docs", "aurelius",
         "tests"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    for rel in out:
        p = REPO_ROOT / rel
        if p.is_file() and p.suffix in (".json", ".jsonl", ".py", ".md", ".txt"):
            yield p


def test_no_secret_values_committed():
    secrets = _secret_values()
    leaks = []
    for p in _iter_committed_text_files():
        try:
            body = p.read_text(errors="ignore")
        except OSError:
            continue
        for v in secrets:
            if v in body:
                leaks.append((str(p.relative_to(REPO_ROOT)), "env-credential"))
        for pat in SECRET_PATTERNS:
            if pat.search(body):
                leaks.append((str(p.relative_to(REPO_ROOT)), pat.pattern))
    assert not leaks, f"secret material in committed files: {leaks}"


def test_scripts_read_credentials_from_env_only():
    """No script embeds a credential value; the fetch scripts (the only ones
    that touch credentials) read them via os.environ."""
    secrets = _secret_values()
    fetch_scripts = {
        "fetch_economic_overlay_analysis_sources.py",
        "fetch_economic_overlay_market_data.py",
    }
    for p in SCRIPTS:
        body = p.read_text()
        for v in secrets:
            assert v not in body, f"{p.name} embeds a credential value"
        if p.name in fetch_scripts:
            assert "environ" in body, (
                f"{p.name} (a fetch script) must read creds from os.environ")


# ───────────────────────── 2. No invented constants ─────────────────────────


def test_no_invented_constants_in_scripts():
    forbidden = [
        r"GPU_HOUR_PRICE\s*=", r"DEFAULT_GPU_HOUR\s*=",
        r"ELECTRICITY_PRICE\s*=\s*\d", r"CARBON_PRICE\s*=\s*\d",
        r"CACHE_WEIGHT\s*=", r"MIGRATION_PENALTY\s*=", r"UTILITY_SCORE\s*=",
        r"COMPOSITE_WEIGHT\s*=",
    ]
    for p in (REPO_ROOT / "scripts" / "build_economic_overlay_analysis.py",
              REPO_ROOT / "scripts" / "run_economic_overlay_analysis_eval.py"):
        body = p.read_text()
        for pat in forbidden:
            assert not re.search(pat, body), f"{p.name}: forbidden {pat}"


# ───────────────────────── 3. Scenario fields labelled ─────────────────────


def test_energy_and_carbon_are_scenario_prior(summary):
    """Region-less traces -> energy price is scenario_prior; WattTime live
    failed -> carbon intensity is scenario_prior. Both must be labelled."""
    fq = summary["global_field_quality"]
    assert set(fq["electricity_price_usd_per_kwh"]) <= {"scenario_prior",
                                                        "measured", "missing"}
    # Every non-missing electricity price in the analysis corpus is scenario.
    assert fq["electricity_price_usd_per_kwh"].get("measured", 0) == 0, (
        "region-less traces must not carry a 'measured' energy price")
    assert "scenario_prior" in fq["carbon_intensity_g_per_kwh"]
    assert fq["carbon_intensity_g_per_kwh"].get("measured", 0) == 0


def test_overlay_summary_records_scenario_region_assignment(summary):
    assert summary["energy_overlay"]["value_quality"] == "scenario_prior"
    assert "scenario" in summary["energy_overlay"]["region_assignment"].lower()
    assert summary["carbon_overlay"]["value_quality"] == "scenario_prior"


# ───────────────────────── 4. Live provider success/failure recorded ───────


def test_manifest_records_all_four_market_providers(manifest):
    provs = manifest["providers"]
    for key in ("pjm_da_energy_price", "ercot_da_energy_price",
                "caiso_da_energy_price", "watttime_carbon_intensity"):
        assert key in provs, f"manifest missing provider {key}"
        assert "status" in provs[key], f"{key} has no status field"


def test_manifest_status_values_known(manifest):
    allowed = {"success", "failed", "skipped_403"}
    for key, rec in manifest["providers"].items():
        assert rec["status"] in allowed, f"{key}: bad status {rec['status']}"


def test_at_least_two_energy_markets_measured(manifest):
    """PJM + CAISO + ERCOT should be live-measured this run."""
    measured = [k for k, v in manifest["providers"].items()
                if v.get("value_quality") == "measured"]
    assert len(measured) >= 2, f"expected >=2 measured markets; got {measured}"


# ───────────────────────── 5. Analysis-tier scale ─────────────────────────


def test_total_rows_analysis_tier(summary):
    assert summary["total_rows"] >= 50000, (
        f"analysis tier needs >=50k rows; got {summary['total_rows']}")


def test_cara_and_swissai_scaled(summary):
    ps = summary["per_source"]
    cara = ps["cara_train_flat"]["rows"]
    swissai = ps["swissai_bucket_reuse"]["rows"]
    assert cara >= 50000, f"CARA only {cara} rows (<50k)"
    assert swissai >= 50000, f"SwissAI only {swissai} rows (<50k)"


def test_subgroup_counts_reported(summary):
    sg = summary["global_subgroups"]
    assert sg["by_gpu_type"], "no GPU subgroup counts"
    assert sg["by_overlay_class"], "no overlay-class subgroup counts"
    assert sg["by_source_dataset"], "no dataset subgroup counts"
    # >= 4 distinct GPU types covered.
    assert len(sg["by_gpu_type"]) >= 4


# ───────────────────────── 6. Coverage non-zero ─────────────────────────


def test_economic_coverage_nonzero(summary):
    cov = summary["global_coverage"]
    assert cov["estimated_gpu_cost_usd"]["rows_with_value"] > 0
    assert cov["sla_safe_goodput_per_dollar"]["rows_with_value"] > 0
    assert cov["estimated_energy_cost_usd"]["rows_with_value"] > 0
    assert cov["estimated_cache_value_usd"]["rows_with_value"] > 0


def test_carbon_cost_is_operator_policy_only(summary):
    cov = summary["global_coverage"]
    assert cov["estimated_carbon_cost_usd"]["rows_with_value"] == 0, (
        "carbon cost must be 0 under default (no operator carbon price)")
    fq = summary["global_field_quality"]
    assert fq["estimated_carbon_cost_usd"].get("missing", 0) == \
        summary["total_rows"]


def test_carbon_kg_is_physical_quantity_present(summary):
    """carbon_kg (physical) should be present wherever energy is, even though
    carbon COST is operator-policy-only."""
    cov = summary["global_coverage"]
    assert cov["estimated_carbon_kg"]["rows_with_value"] > 0


# ───────────────────────── 7. A-H eval integrity ─────────────────────────


def test_eval_runs_all_eight_variants(eval_data):
    expected = {"A_existing_scorer_baseline", "B_existing_plus_gpu_price",
                "C_existing_plus_energy_carbon", "D_existing_plus_cache_value",
                "E_existing_plus_full_overlay", "F_full_plus_ttft_prior",
                "G_full_plus_cache_prefix_prior", "H_full_plus_both_priors"}
    assert set(eval_data["variants"].keys()) == expected


def test_eval_baseline_has_zero_goodput(eval_data):
    a = eval_data["variants"]["A_existing_scorer_baseline"]["metrics"]
    assert a["by_field_stats"]["sla_safe_goodput_per_dollar"]["n"] == 0


def test_eval_full_overlay_large_coverage(eval_data):
    e = eval_data["variants"]["E_existing_plus_full_overlay"]["metrics"]
    n = e["by_field_stats"]["sla_safe_goodput_per_dollar"]["n"]
    assert n >= 50000, f"full overlay goodput/$ only {n} rows"


def test_eval_promotion_known_state(eval_data):
    allowed = {"diagnostic_only", "economic_overlay_ready",
               "shadow_ready_for_integration_review", "blocked_by_pilot_telemetry"}
    promo = eval_data["promotion"]["final_status"]
    assert promo in allowed
    assert eval_data["promotion"]["reason"]
    assert eval_data["promotion"][
        "carbon_cost_requires_operator_carbon_price_per_kg_usd"] is True


def test_eval_no_oracle_no_fifo_headline(eval_data):
    assert eval_data["uses_oracle_as_headline"] is False
    assert eval_data["uses_fifo_as_headline"] is False
    assert eval_data["production_claim"] is False


def test_eval_reports_three_classes_separately(eval_data):
    for name, v in eval_data["variants"].items():
        ch = v["metrics"]["headline_sla_safe_goodput_per_dollar_per_class"]
        assert set(ch.keys()) >= {"measured_same_record",
                                  "cross_dataset_joined", "scenario_prior"}, name


def test_eval_classification_not_all_scenario(eval_data):
    """The refined classifier must NOT collapse everything to scenario_prior:
    rows whose cost is driven only by the public GPU price prior are
    cross_dataset_joined, not scenario."""
    e = eval_data["variants"]["E_existing_plus_full_overlay"]["metrics"]
    by_class = e["by_overlay_class_count"]
    assert by_class.get("cross_dataset_joined", 0) > 0, (
        "expected a non-empty cross_dataset_joined class")


# ───────────────────────── 8. Raw + corpus gitignored, samples bounded ─────


def test_raw_and_corpus_not_tracked():
    """The full analysis corpus and any economic-overlay raw download must
    stay gitignored. (.gitkeep placeholders elsewhere are allowed.)"""
    out = subprocess.check_output(
        ["git", "ls-files", "data/external/economic_overlay",
         "data/external/hf"], cwd=REPO_ROOT,
    ).decode().splitlines()
    bad = [p for p in out
           if "/analysis_corpus/" in p
           or ("/economic_overlay/raw/" in p and not p.endswith(".gitkeep"))
           or ("/hf/" in p and "/raw/" in p and not p.endswith(".gitkeep"))]
    assert not bad, f"raw / full corpus committed (gitignore broken): {bad[:8]}"


def test_committed_samples_bounded():
    samples = OVERLAY / "economic_overlay_samples"
    total = 0
    for p in samples.glob("*.jsonl"):
        sz = p.stat().st_size
        assert sz <= 100 * 1024 * 1024, f"{p.name} > 100 MB"
        total += sz
    assert total <= 300 * 1024 * 1024, f"committed samples total {total} > 300 MB"


def test_doc_exists_and_covers_required_sections():
    assert DOC.exists(), f"missing analysis doc: {DOC}"
    body = DOC.read_text().lower()
    for marker in ("rows joined", "api coverage", "field coverage",
                   "a-h", "operator-policy", "pilot", "scenario_prior"):
        assert marker in body, f"doc missing section marker: {marker!r}"
