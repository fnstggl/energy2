"""Frontier Signal Bounded Ingestion v1 — ingest-integrity tests.

Proves: no raw large files committed; raw paths gitignored; normalized samples
respect size caps; no secrets/tokens; no raw prompt/completion/message text;
every source has schema_profile/schema_mapping/summary/rollups; field_quality /
provenance / limitations exist; Mooncake is not pilot telemetry; Huawei FaaS is
not GPU model-load truth; Alibaba autoscaling labels are measured/proxy
classified explicitly; no production module modified by this ingest.
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

SIG = REPO_ROOT / "data" / "external" / "frontier_signals"
INGEST = REPO_ROOT / "data" / "external" / "frontier_ingest_v1"
SOURCES = ["mooncake", "huawei_faas_2025", "alibaba_gpu_v2025"]
SAMPLE_BYTES_CAP = 8 * 1024 * 1024
SAMPLE_ROWS_CAP = 5000


def _git_tracked(path_glob):
    out = subprocess.run(["git", "ls-files", path_glob], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def test_source_audit_exists_and_classifies():
    audit = json.loads((INGEST / "source_audit.json").read_text())
    names = {s["name"] for s in audit["sources"]}
    assert set(SOURCES) <= names
    for s in audit["sources"]:
        assert s["classification"]
        assert "bounded_ingest_safe" in s


def test_no_raw_files_committed():
    # nothing under any raw/ dir is tracked
    tracked = _git_tracked("data/external/frontier_signals/raw")
    tracked += _git_tracked("data/external/frontier_signals/*/raw")
    assert tracked == [], f"raw files must not be committed: {tracked}"


def test_raw_paths_gitignored():
    gi = (REPO_ROOT / ".gitignore").read_text()
    assert "data/external/frontier_signals/raw/" in gi
    assert "analysis_sample.jsonl" in gi


def test_analysis_sample_not_committed():
    tracked = _git_tracked("data/external/frontier_signals/*/processed/analysis_sample.jsonl")
    assert tracked == [], f"full analysis_sample must be gitignored: {tracked}"


@pytest.mark.parametrize("src", SOURCES)
def test_processed_files_exist(src):
    pdir = SIG / src / "processed"
    for name in ("schema_profile.json", "schema_mapping.json", "summary.json",
                 "statistical_rollups.json", "normalized_sample.jsonl"):
        assert (pdir / name).exists(), f"{src} missing {name}"


@pytest.mark.parametrize("src", SOURCES)
def test_normalized_sample_within_caps(src):
    p = SIG / src / "processed" / "normalized_sample.jsonl"
    assert p.stat().st_size <= SAMPLE_BYTES_CAP, f"{src} sample exceeds 8 MiB"
    rows = [l for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) <= SAMPLE_ROWS_CAP, f"{src} sample exceeds {SAMPLE_ROWS_CAP} rows"
    # each row is valid JSON
    for l in rows[:50]:
        json.loads(l)


@pytest.mark.parametrize("src", SOURCES)
def test_field_quality_and_limitations_present(src):
    rows = [json.loads(l) for l in
            (SIG / src / "processed" / "normalized_sample.jsonl").read_text().splitlines() if l.strip()]
    assert rows
    r = rows[0]
    assert "field_quality" in r and isinstance(r["field_quality"], dict)
    assert any(k in r for k in ("limitation", "limitations")), "provenance/limitations required"


@pytest.mark.parametrize("src", SOURCES)
def test_no_raw_text_or_secrets_committed(src):
    blob = (SIG / src / "processed" / "normalized_sample.jsonl").read_text()
    forbidden_fields = ('"prompt"', '"completion"', '"message"', '"messages"',
                        '"text"', '"content"', '"response"')
    for f in forbidden_fields:
        assert f not in blob, f"{src} committed raw text field {f}"
    secret_pat = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{20,})")
    assert not secret_pat.search(blob), f"{src} sample contains a secret-like token"


def test_huawei_request_id_hash_not_committed():
    blob = (SIG / "huawei_faas_2025" / "processed" / "normalized_sample.jsonl").read_text()
    assert "requestID" not in blob and "request_id" not in blob, \
        "Huawei requestID hash must be dropped from committed sample"


def test_mooncake_is_not_pilot_telemetry():
    audit = json.loads((INGEST / "source_audit.json").read_text())
    mc = next(s for s in audit["sources"] if s["name"] == "mooncake")
    assert "workload_only" in mc["classification"]
    assert "derived" in mc["measured_vs_derived"].lower()
    assert audit["public_data_is_not_pilot_telemetry"] is True
    # reuse label is derived, NOT measured cache_hit
    summ = json.loads((SIG / "mooncake" / "processed" / "summary.json").read_text())
    assert summ["measured_vs_proxy"]["reuse_label"] == "derived_proxy"


def test_huawei_is_not_gpu_model_load_truth():
    summ = json.loads((SIG / "huawei_faas_2025" / "processed" / "summary.json").read_text())
    assert summ["is_gpu_model_load"] is False
    assert summ["calibration_only"] is True
    assert summ["measured_vs_proxy"]["gpu_llm_cold_start"] == "prior_proxy_only"


def test_alibaba_autoscaling_classified_proxy():
    summ = json.loads((SIG / "alibaba_gpu_v2025" / "processed" / "summary.json").read_text())
    assert summ["has_measured_serving_autoscaling"] is False
    assert summ["has_per_request_queue_wait"] is False
    assert summ["measured_vs_proxy"]["queue_autoscaling"] == "proxy_inferred"


def test_no_production_module_modified_by_ingest_scripts():
    # the ingest scripts must not import production scheduler/scorer/residency/frontier
    for script in ("ingest_frontier_signals_v1.py", "audit_frontier_signal_strength_v1.py"):
        txt = (REPO_ROOT / "scripts" / script).read_text()
        for prod in ("aurelius.scheduler", "aurelius.residency", "aurelius.frontier",
                     "aurelius.benchmarks.economics"):
            assert prod not in txt, f"{script} must not touch production module {prod}"
