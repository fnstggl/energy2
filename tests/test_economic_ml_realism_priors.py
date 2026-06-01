"""Realism-prior + cold-start/migration safety tests for Economic ML Alpha v1.

Proves: cold-start/migration missing labels are NOT silently zeroed;
simulator_prior-only targets cannot become headline; every realism-prior
parameter carries a source + source_type + confidence; the sensitivity file
is labelled simulator_prior_only; pilot-telemetry needs are listed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART = REPO_ROOT / "data" / "external" / "forecasting" / "economic_ml_alpha_v1"


@pytest.fixture(scope="module")
def realism():
    p = ART / "realism_prior_audit.json"
    assert p.exists(), f"missing {p}"
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def sensitivity():
    p = ART / "cold_start_migration_sensitivity.json"
    assert p.exists(), f"missing {p}"
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def summary():
    p = ART / "summary.json"
    assert p.exists(), f"missing {p}"
    return json.loads(p.read_text())


# ───────────────────── realism-prior schema ─────────────────────


def test_every_realism_term_has_sourced_provenance(realism):
    allowed = {"measured", "derived", "prior", "scenario_prior",
               "simulator_prior", "missing", "operator_policy", "proxy"}
    for group in ("cold_start_terms", "migration_terms"):
        for name, term in realism[group].items():
            assert "source" in term and "source_type" in term, (group, name)
            assert term["source_type"] in allowed, (name, term["source_type"])
            assert "confidence" in term
            assert term.get("production_ready") is False


def test_cold_start_and_migration_verdict_blocked(realism):
    v = realism["verdict"]
    assert "blocked_by_missing_labels" in v["cold_start"]
    assert "blocked_by_missing_labels" in v["migration"]


def test_no_realism_term_is_silently_zero(realism):
    """Missing terms must be labelled missing/simulator_prior, never value=0."""
    for group in ("cold_start_terms", "migration_terms"):
        for name, term in realism[group].items():
            if term["source_type"] in ("missing", "simulator_prior"):
                assert term.get("value") in (None, ), (
                    f"{name} is {term['source_type']} but has a value "
                    f"{term.get('value')!r} (must not be silently set)")


# ───────────────────── sensitivity safety ─────────────────────


def test_sensitivity_is_simulator_prior_only(sensitivity):
    assert "simulator_prior_only" in sensitivity["status"].lower()
    assert "never headline" in sensitivity["status"].lower()


def test_sensitivity_parameters_are_sourced(sensitivity):
    for name, p in sensitivity["parameters"].items():
        assert "source" in p and "source_type" in p, name
        assert "confidence" in p
        assert p.get("production_ready") is False


def test_sensitivity_sweeps_nonempty(sensitivity):
    assert len(sensitivity["migration_cost_sweep"]) > 0
    assert len(sensitivity["cold_start_cost_sweep"]) > 0


def test_sensitivity_lists_pilot_telemetry_needs(sensitivity):
    needs = sensitivity["parameters_needing_pilot_telemetry"]
    assert any("model_load_duration_s" in n for n in needs)


# ───────────────────── headline safety ─────────────────────


def test_simulator_prior_targets_not_in_trainable(summary):
    trainable = summary["targets_trainable_now"]
    assert "cold_start_cost_usd" not in trainable
    assert "migration_cost_usd" not in trainable


def test_cold_start_migration_in_blocked_list(summary):
    blocked = summary["targets_blocked_by_missing_labels"]
    assert "cold_start_cost_usd" in blocked
    assert "migration_cost_usd" in blocked


def test_pilot_telemetry_explicitly_listed(summary):
    pilot = summary["pilot_telemetry_needed"]
    assert any("model_load" in p for p in pilot)
    assert any("cache_hit" in p for p in pilot)
