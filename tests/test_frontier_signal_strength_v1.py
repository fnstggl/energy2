"""Frontier Signal Strength v1 — Phase 3 integrity tests.

Proves: signal_strength.json exists with all three sources; measured/proxy/
simulated breakdown present; suitable_for flags are honest (no GPU cold-start
ML from FaaS; autoscaling/queue are proxy-only; Mooncake cross-dataset
validation is limited).
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

SS = REPO_ROOT / "data" / "external" / "frontier_ingest_v1" / "signal_strength.json"
SOURCES = ["mooncake", "huawei_faas_2025", "alibaba_gpu_v2025"]


@pytest.fixture(scope="module")
def ss():
    return json.loads(SS.read_text())


def test_exists_with_all_sources(ss):
    assert set(SOURCES) <= set(ss["sources"].keys())
    assert ss["public_data_is_not_pilot_telemetry"] is True
    assert ss["production_claim"] is False


@pytest.mark.parametrize("src", SOURCES)
def test_breakdown_and_flags_present(src, ss):
    s = ss["sources"][src]
    assert s.get("rows_normalized", 0) > 0
    assert "measured_proxy_simulated_breakdown" in s
    assert "suitable_for" in s
    assert "target_labels_available" in s


def test_huawei_no_gpu_coldstart_ml(ss):
    sf = ss["sources"]["huawei_faas_2025"]["suitable_for"]
    assert sf["cold_start_ml_training"].startswith("no")
    assert sf["cold_start_simulator_prior"].lower().startswith("yes")
    assert "calibrat" in sf["cold_start_simulator_prior"].lower()
    # GPU model-load is explicitly absent from the label set
    labels = ss["sources"]["huawei_faas_2025"]["target_labels_available"]
    assert "ABSENT" in labels["gpu_model_load_s"]


def test_alibaba_proxy_only(ss):
    sf = ss["sources"]["alibaba_gpu_v2025"]["suitable_for"]
    assert "proxy_only" in sf["autoscaling_proxy_training"]
    assert "proxy_only" in sf["queue_risk_training"]
    labels = ss["sources"]["alibaba_gpu_v2025"]["target_labels_available"]
    assert labels["queue_wait_s"] == "ABSENT"
    assert "ABSENT" in labels["autoscaling_event"]


def test_mooncake_cross_dataset_validation_limited(ss):
    sf = ss["sources"]["mooncake"]["suitable_for"]
    assert "limited" in sf["cache_reuse_cross_dataset_validation"].lower()
    # reuse label is proxy, cache_hit not measured
    labels = ss["sources"]["mooncake"]["target_labels_available"]
    assert labels["cache_reuse_pct"] == "derived_proxy"
    assert labels["cache_hit"] == "not_measured"


def test_measured_vs_proxy_summary_honest(ss):
    m = ss["measured_vs_proxy_summary"]
    assert "derived proxy" in m["mooncake"]
    assert "prior/proxy" in m["huawei_faas_2025"] or "prior" in m["huawei_faas_2025"]
    assert "proxy" in m["alibaba_gpu_v2025"]
