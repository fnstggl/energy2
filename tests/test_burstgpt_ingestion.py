"""Tests for the BurstGPT public-trace ingestion + replay backtest.

Unit tests use ONLY ``tests/fixtures/burstgpt_sample.csv`` and never touch the
network or the full CSV. The full-trace backtest is an integration test that is
skipped when the raw file is absent.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces import burstgpt  # noqa: E402
from aurelius.traces.backtest import ALL_POLICIES, run_backtest  # noqa: E402
from aurelius.traces.replay import requests_to_arrival_ticks  # noqa: E402
from aurelius.traces.schema import (  # noqa: E402
    TraceSchemaError,
    validate_columns,
)

FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv")
RAW_FULL = os.path.join(REPO_ROOT, "data", "external", "burstgpt", "raw",
                        "BurstGPT_1.csv")
RESULTS_MD = os.path.join(REPO_ROOT, "docs", "BURSTGPT_BACKTEST_RESULTS.md")
PUBLIC_DOC = os.path.join(REPO_ROOT, "docs", "PUBLIC_TRACE_BACKTESTS.md")

BANNED_CLAIMS = (
    "production savings",
    "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics",
    "production-proven",
)


# --- 1. Sample fixture parses ------------------------------------------------

def test_sample_fixture_parses():
    reqs = burstgpt.load_csv(FIXTURE, include_failures=True)
    assert len(reqs) > 0
    assert all(r.model in ("ChatGPT", "GPT-4") for r in reqs)
    assert all(r.total_tokens >= 0 for r in reqs)


# --- 2. Schema validation catches missing columns ---------------------------

def test_schema_validation_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Timestamp,Model,Total tokens,Log Type\n5,ChatGPT,490,API log\n")
    with pytest.raises(TraceSchemaError):
        burstgpt.load_csv(str(bad))


def test_validate_columns_helper():
    with pytest.raises(TraceSchemaError):
        validate_columns(["Timestamp", "Model"], burstgpt.REQUIRED_COLUMNS, "burstgpt")
    # full header passes
    validate_columns(list(burstgpt.REQUIRED_COLUMNS), burstgpt.REQUIRED_COLUMNS,
                     "burstgpt")


# --- 3. response_tokens == 0 marks failure ----------------------------------

def test_zero_response_tokens_is_failure():
    reqs = burstgpt.load_csv(FIXTURE, include_failures=True)
    failures = [r for r in reqs if r.is_failure]
    assert failures, "fixture must contain at least one failure row"
    assert all(r.output_tokens == 0 for r in failures)
    # and excluding failures drops them
    kept = burstgpt.load_csv(FIXTURE, include_failures=False)
    assert all(not r.is_failure for r in kept)
    assert len(kept) < len(reqs)


# --- 4. Timestamp ordering preserved ----------------------------------------

def test_timestamp_ordering_preserved():
    reqs = burstgpt.load_csv(FIXTURE, include_failures=True)
    ts = [r.timestamp_s for r in reqs]
    assert ts == sorted(ts)


# --- 5. Token fields map correctly ------------------------------------------

def test_token_fields_map_correctly():
    # First data row: 5,ChatGPT,472,18,490,Conversation log
    reqs = burstgpt.load_csv(FIXTURE, include_failures=True)
    first = min(reqs, key=lambda r: r.timestamp_s)
    assert first.timestamp_s == 5.0
    assert first.model == "ChatGPT"
    assert first.prompt_tokens == 472
    assert first.output_tokens == 18
    assert first.total_tokens == 490
    assert first.log_type == "Conversation log"


# --- 6. Session ID maps to cache_affinity_key -------------------------------

def test_session_id_maps_to_cache_affinity_key(tmp_path):
    # Build an extended-schema CSV (with Session ID + Elapsed time) in memory.
    p = tmp_path / "ext.csv"
    rows = [
        ["Timestamp", "Session ID", "Elapsed time", "Model", "Request tokens",
         "Response tokens", "Total tokens", "Log Type"],
        ["10", "sess-A", "1.5", "ChatGPT", "100", "50", "150", "Conversation log"],
        ["12", "sess-B", "2.0", "GPT-4", "200", "0", "200", "API log"],  # failure
        ["20", "", "", "ChatGPT", "300", "40", "340", "API log"],  # no session
    ]
    with open(p, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    reqs = burstgpt.load_csv(str(p), include_failures=True)
    by_ts = {r.timestamp_s: r for r in reqs}
    assert by_ts[10.0].session_id == "sess-A"
    assert by_ts[10.0].cache_affinity_key == "sess-A"
    assert by_ts[10.0].elapsed_s == 1.5
    # zero response tokens still a failure even with valid elapsed
    assert by_ts[12.0].is_failure
    # absent session id -> model-level proxy key, not a session id
    assert by_ts[20.0].session_id is None
    assert by_ts[20.0].cache_affinity_key == "model:ChatGPT"


def test_burstgpt_1_has_no_session_or_elapsed():
    # The published BurstGPT_1.csv schema (mirrored by the fixture) has neither.
    reqs = burstgpt.load_csv(FIXTURE, include_failures=True)
    assert all(r.session_id is None for r in reqs)
    assert all(r.elapsed_s is None for r in reqs)
    # cache key degrades to the documented model-level proxy
    assert all(r.cache_affinity_key.startswith("model:") for r in reqs)


# --- 7. Normalized trace generates simulator arrivals -----------------------

def test_normalized_trace_generates_arrivals():
    reqs = burstgpt.load_csv(FIXTURE, include_failures=True)
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    assert ticks, "should produce arrival ticks"
    # arrivals preserve token + timing content
    assert any(t.request_count > 0 for t in ticks)
    assert all(t.arrival_rate_rps >= 0 for t in ticks)
    total_out = sum(t.total_output_tokens for t in ticks)
    assert total_out == sum(r.output_tokens for r in reqs)
    # time-ordered, contiguous tick windows
    starts = [t.start_s for t in ticks]
    assert starts == sorted(starts)


# --- 8. Backtest deterministic under fixed seed -----------------------------

def test_backtest_deterministic_fixed_seed():
    r1 = burstgpt.load_csv(FIXTURE, sample_size=40, seed=123, include_failures=True)
    r2 = burstgpt.load_csv(FIXTURE, sample_size=40, seed=123, include_failures=True)
    assert [r.to_dict() for r in r1] == [r.to_dict() for r in r2]
    b1 = run_backtest(r1, tick_seconds=60.0, policies=ALL_POLICIES)
    b2 = run_backtest(r2, tick_seconds=60.0, policies=ALL_POLICIES)
    assert b1.to_summary_dict() == b2.to_summary_dict()


def test_backtest_runs_all_policies():
    reqs = burstgpt.load_csv(FIXTURE, include_failures=False)
    result = run_backtest(reqs, tick_seconds=60.0, policies=ALL_POLICIES)
    assert set(result.policy_results) == set(ALL_POLICIES)
    for r in result.policy_results.values():
        # KPI is defined (positive cost basis) for every policy
        assert r.kpi.total_infrastructure_cost > 0


# --- 9. Unit tests do not download the full CSV -----------------------------

def test_no_network_download_for_fixture(monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("unit tests must not hit the network")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    # The whole fixture flow must work with the network poisoned.
    reqs = burstgpt.load_csv(FIXTURE, include_failures=True)
    result = run_backtest(reqs, tick_seconds=60.0, policies=ALL_POLICIES)
    assert result.n_requests == len(reqs)


# --- 10. Full-trace backtest is integration-only / skipped if raw missing ---

@pytest.mark.skipif(not os.path.exists(RAW_FULL),
                    reason="raw BurstGPT_1.csv not present (integration only)")
def test_full_trace_backtest_integration():
    reqs = burstgpt.load_csv(
        RAW_FULL, start_s=0, duration_s=600000, scale_rps=300,
    )
    assert len(reqs) > 1000
    result = run_backtest(reqs, tick_seconds=60.0, policies=ALL_POLICIES)
    ca = result.policy_results["constraint_aware"]
    assert ca.kpi.sla_safe_goodput_per_infra_dollar is not None
    # constraint_aware must never regress SLA vs FIFO (docs/RESULTS.md §6).
    fifo = result.policy_results["fifo"]
    assert ca.timeout_rate_pct_mean <= fifo.timeout_rate_pct_mean + 1e-6


# --- 11 & 12. Docs: elapsed!=TTFT, and no production-savings claims ----------

def _assert_no_unhedged_banned_claims(text: str):
    low = text.lower()
    for phrase in BANNED_CLAIMS:
        idx = 0
        while True:
            pos = low.find(phrase, idx)
            if pos == -1:
                break
            # allow only when negated ("not", "no", "never") within 24 chars before
            prefix = low[max(0, pos - 24):pos]
            assert any(neg in prefix for neg in ("not ", "no ", "never", "n't")), (
                f"unhedged banned claim '{phrase}' in docs near: "
                f"...{text[max(0, pos-24):pos+len(phrase)+10]}..."
            )
            idx = pos + len(phrase)


def test_docs_state_elapsed_is_not_ttft():
    for path in (RESULTS_MD, PUBLIC_DOC):
        text = open(path).read()
        assert "NOT TTFT" in text or "not TTFT" in text, f"{path} must state elapsed != TTFT"


def test_docs_no_production_savings_claims():
    for path in (RESULTS_MD, PUBLIC_DOC):
        _assert_no_unhedged_banned_claims(open(path).read())


def test_generated_report_is_honest(tmp_path):
    # End-to-end: run the script on the fixture, scan the generated markdown.
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "scripts", "run_burstgpt_backtest.py"),
        "--csv", FIXTURE, "--results-md", str(out_md),
        "--summary-json", str(out_json), "--no-sweep",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    md = out_md.read_text()
    assert "NOT TTFT" in md
    assert "directional only" in md.lower()
    _assert_no_unhedged_banned_claims(md)
    payload = json.loads(out_json.read_text())
    assert payload["backtest"]["primary_kpi"] == \
        "sla_safe_goodput_per_infrastructure_dollar"
