"""Frontier Discovery Audit v1 — artifact-integrity tests.

Discovery/audit-only: tests read committed JSON; they do NOT hit the HF API.
Prove: artifacts exist + schema; no HF_TOKEN committed; no synthetic dataset
promoted; honest negative findings recorded (autoscaling empty, cold-start
RL-noise not promoted); no production code modified; economic-relevance and
forecastability are bounded + present per candidate.
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

ART = REPO_ROOT / "data" / "external" / "hf_discovery" / "frontier_v1"
REG = ART / "frontier_dataset_registry.json"
MATRIX = ART / "frontier_field_matrix.json"
RANK = ART / "economic_frontier_priority_ranking.json"
DOC_H = REPO_ROOT / "docs" / "FRONTIER_SIGNAL_HYPOTHESES.md"
DOC_A = REPO_ROOT / "docs" / "FRONTIER_DISCOVERY_AUDIT_V1.md"
SCRIPT = REPO_ROOT / "scripts" / "discover_frontier_signals.py"

HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")


@pytest.fixture(scope="module")
def registry():
    assert REG.exists(), f"missing {REG}"
    return json.loads(REG.read_text())


@pytest.fixture(scope="module")
def ranking():
    assert RANK.exists(), f"missing {RANK}"
    return json.loads(RANK.read_text())


@pytest.fixture(scope="module")
def matrix():
    assert MATRIX.exists(), f"missing {MATRIX}"
    return json.loads(MATRIX.read_text())


# ───────────────── existence + schema ─────────────────


def test_all_required_artifacts_exist():
    for p in (REG, MATRIX, RANK, DOC_H, DOC_A):
        assert p.exists(), f"missing required output: {p}"


def test_registry_safety_flags(registry):
    assert registry["production_claim"] is False
    assert registry["candidates"], "no candidates inspected"


def test_ranking_safety_flags(ranking):
    assert ranking["production_claim"] is False
    assert ranking["no_training"] is True
    assert ranking["no_ingestion"] is True


def test_every_candidate_has_required_fields(registry):
    for c in registry["candidates"]:
        for k in ("dataset_id", "signals", "economic_relevance",
                  "forecastability_score", "extract", "url",
                  "previously_evaluated"):
            assert k in c, (c.get("dataset_id"), k)


def test_economic_relevance_values_known(registry):
    allowed = {"Very High", "High", "Medium", "Low", "Reject"}
    for c in registry["candidates"]:
        assert c["economic_relevance"] in allowed, c["dataset_id"]


def test_forecastability_bounded(registry):
    for c in registry["candidates"]:
        assert 0 <= c["forecastability_score"] <= 100, c["dataset_id"]


def test_field_matrix_covers_six_frontier_categories(matrix):
    cats = set(matrix["signal_categories"])
    for fc in ("cold_start", "migration", "queueing", "memory_pressure",
               "serving_stability", "autoscaling"):
        assert fc in cats, fc


# ───────────────── no token / no production change ─────────────────


def test_no_hf_token_in_artifacts_or_script():
    for p in list(ART.glob("*.json")) + [SCRIPT, DOC_H, DOC_A]:
        body = p.read_text(errors="ignore")
        assert not HF_TOKEN_RE.search(body), f"HF_TOKEN leaked into {p.name}"


def test_no_production_module_modified():
    out = subprocess.check_output(
        ["git", "diff", "--name-only", "main...HEAD"], cwd=REPO_ROOT,
    ).decode().splitlines()
    forbidden = {
        "aurelius/optimization/scheduler.py",
        "aurelius/optimization/objective.py",
        "aurelius/optimization/constraints.py",
        "aurelius/forecasting/constraint_shadow_scorer.py",
        "aurelius/forecasting/economic_overlay.py",
        "aurelius/forecasting/economic_ml_forecaster.py",
        "aurelius/forecasting/economic_ml_features.py",
        "aurelius/residency/decision.py",
        "aurelius/frontier/controller.py",
    }
    bad = [p for p in out if p in forbidden]
    assert not bad, f"frontier discovery must not modify production: {bad}"


def test_discovery_is_metadata_only_no_data_committed():
    """No raw dataset payloads under the frontier_v1 dir — only JSON audits."""
    for p in ART.rglob("*"):
        if p.is_file():
            assert p.suffix == ".json", f"unexpected non-audit file: {p}"


# ───────────────── honest findings recorded ─────────────────


def test_autoscaling_negative_result_recorded(ranking):
    """No public HF dataset has autoscaling telemetry — must be empty + honest."""
    assert ranking["datasets_with_autoscaling"] == []


def test_no_synthetic_or_rl_coldstart_promoted_high(registry):
    """RL-finetuning 'cold-start' datasets must NOT be rated High/Very High
    economic relevance (they are not serving telemetry)."""
    for c in registry["candidates"]:
        ds = c["dataset_id"].lower()
        looks_rl = any(t in ds for t in ("coldstart", "cold-start", "cold_start",
                                         "sft", "grpo", "reasoning", "math",
                                         "msmarco", "multimodal"))
        if looks_rl:
            assert c["economic_relevance"] in ("Low", "Reject"), (
                f"RL cold-start dataset wrongly promoted: {c['dataset_id']} "
                f"-> {c['economic_relevance']}")


def test_high_relevance_datasets_are_real_serving_telemetry(registry):
    """Anything rated High must carry a real serving/ops signal, never be a
    bare NLP/eval set."""
    ops = {"ttft", "tpot_itl", "e2e_latency", "throughput", "gpu_telemetry",
           "energy", "memory_pressure", "serving_stability", "queueing"}
    for c in registry["candidates"]:
        if c["economic_relevance"] in ("High", "Very High"):
            assert ops & set(c["signals"]), (
                f"High-rated dataset lacks a real ops signal: {c['dataset_id']}")


def test_ranking_reports_top_signals_per_frontier_category(ranking):
    for cat in ("cold_start", "migration", "queueing", "memory_pressure",
                "serving_stability", "autoscaling"):
        assert f"datasets_with_{cat}" in ranking


def test_doc_records_negative_findings():
    body = DOC_A.read_text().lower()
    assert "autoscaling: 0" in body or "autoscaling**: 0" in body \
        or "0 datasets" in body
    assert "blocked_by_pilot_telemetry" in body
    assert "rl fine-tuning" in body or "rl-finetuning" in body \
        or "rl warm-up" in body


def test_hypotheses_doc_covers_all_six_categories():
    body = DOC_H.read_text()
    for marker in ("## A. Cold starts", "## B. Migration", "## C. Queueing",
                   "## D. Memory pressure", "## E. Serving stability",
                   "## F. Autoscaling"):
        assert marker in body, marker
