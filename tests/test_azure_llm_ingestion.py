"""Tests for the Azure LLM public-trace ingestion + replay backtest.

Unit tests use ONLY ``tests/fixtures/azure_llm_sample.csv`` and never touch the
network or the full dataset. The full-trace backtest is an integration test
that is skipped when the raw file is absent.
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

from aurelius.traces import azure_llm  # noqa: E402
from aurelius.traces.backtest import run_backtest  # noqa: E402
from aurelius.traces.replay import requests_to_arrival_ticks  # noqa: E402
from aurelius.traces.schema import TraceSchemaError, validate_columns  # noqa: E402

FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures", "azure_llm_sample.csv")
RAW_CONV = os.path.join(REPO_ROOT, "data", "external", "azure_llm", "raw",
                        "AzureLLMInferenceTrace_conv.csv")
RESULTS_MD = os.path.join(REPO_ROOT, "docs", "AZURE_LLM_BACKTEST_RESULTS.md")
PUBLIC_DOC = os.path.join(REPO_ROOT, "docs", "PUBLIC_TRACE_BACKTESTS.md")

AZURE_POLICIES = ("fifo", "sla_aware", "constraint_aware", "queue_aware")

BANNED_CLAIMS = (
    "production savings",
    "guaranteed savings",
    "enterprise-ready autonomous optimization",
    "hyperscaler-validated economics",
    "production-proven",
)


# --- 1. Azure sample fixture parses -----------------------------------------

def test_sample_fixture_parses():
    reqs = azure_llm.load_csv(FIXTURE, variant="conv", include_failures=True)
    assert len(reqs) > 0
    assert all(r.model == "azure-llm" for r in reqs)
    assert all(r.log_type == "conv" for r in reqs)
    assert all(r.total_tokens == r.prompt_tokens + r.output_tokens for r in reqs)


# --- 2. Schema validation catches missing required token fields -------------

def test_schema_validation_missing_token_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    # Missing ContextTokens + GeneratedTokens
    bad.write_text("TIMESTAMP\n2023-11-16 18:15:46.6805900\n")
    with pytest.raises(TraceSchemaError):
        azure_llm.load_csv(str(bad))


def test_validate_columns_helper():
    with pytest.raises(TraceSchemaError):
        validate_columns(["TIMESTAMP", "ContextTokens"],
                         azure_llm.REQUIRED_COLUMNS, "azure_llm")
    validate_columns(list(azure_llm.REQUIRED_COLUMNS),
                     azure_llm.REQUIRED_COLUMNS, "azure_llm")


# --- 3. Token fields map correctly to prompt/output/total -------------------

def test_token_fields_map_correctly():
    # First fixture row: 2023-11-16 18:15:46.6805900,374,44
    reqs = azure_llm.load_csv(FIXTURE, variant="conv", include_failures=True)
    first = min(reqs, key=lambda r: r.timestamp_s)
    assert first.prompt_tokens == 374      # ContextTokens -> prompt
    assert first.output_tokens == 44       # GeneratedTokens -> output
    assert first.total_tokens == 374 + 44  # derived
    assert first.log_type == "conv"


def test_timestamp_parsed_subsecond():
    # 7-fractional-digit .NET timestamps must parse (strptime alone cannot).
    ts = azure_llm.parse_timestamp_s("2023-11-16 18:15:46.6805900")
    ts2 = azure_llm.parse_timestamp_s("2023-11-16 18:15:46.7805900")
    assert abs((ts2 - ts) - 0.1) < 1e-6


# --- 4. Missing timestamp / value handling ----------------------------------

def test_missing_timestamp_raises():
    with pytest.raises(ValueError):
        azure_llm.parse_timestamp_s("")


def test_zero_generated_is_failure(tmp_path):
    p = tmp_path / "z.csv"
    rows = [
        ["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
        ["2023-11-16 18:15:46.6805900", "374", "44"],
        ["2023-11-16 18:15:47.0000000", "500", "0"],  # failure (zero output)
    ]
    with open(p, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    kept = azure_llm.load_csv(str(p), variant="conv", include_failures=False)
    allrows = azure_llm.load_csv(str(p), variant="conv", include_failures=True)
    assert len(allrows) == 2
    assert len(kept) == 1
    failures = [r for r in allrows if r.is_failure]
    assert len(failures) == 1 and failures[0].output_tokens == 0


# --- 5. Missing session / cache info handled honestly -----------------------

def test_missing_session_and_cache_handled_honestly():
    reqs = azure_llm.load_csv(FIXTURE, variant="conv", include_failures=True)
    assert all(r.session_id is None for r in reqs)
    assert all(r.cache_affinity_key is None for r in reqs)
    assert all(r.elapsed_s is None for r in reqs)  # token-demand, not latency
    s = azure_llm.summarize(reqs)
    assert s.has_session_ids is False
    assert s.has_cache_affinity is False
    assert s.has_elapsed is False
    assert s.distinct_cache_keys == 0
    assert s.cache_key_reuse_rate_pct == 0.0


def test_none_cache_key_yields_zero_reuse():
    reqs = azure_llm.load_csv(FIXTURE, variant="conv", include_failures=True)
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    # No affinity key anywhere -> no invented reuse.
    assert all(t.reuse_fraction == 0.0 for t in ticks)
    assert all(t.distinct_cache_keys == 0 for t in ticks)


# --- 6. Normalized trace generates simulator arrivals -----------------------

def test_normalized_trace_generates_arrivals():
    reqs = azure_llm.load_csv(FIXTURE, variant="conv", include_failures=True)
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    assert ticks
    assert any(t.request_count > 0 for t in ticks)
    total_out = sum(t.total_output_tokens for t in ticks)
    assert total_out == sum(r.output_tokens for r in reqs)


# --- 7. Backtest deterministic under fixed seed -----------------------------

def test_backtest_deterministic_fixed_seed():
    r1 = azure_llm.load_csv(FIXTURE, variant="conv", sample_size=30, seed=7,
                            include_failures=True)
    r2 = azure_llm.load_csv(FIXTURE, variant="conv", sample_size=30, seed=7,
                            include_failures=True)
    assert [r.to_dict() for r in r1] == [r.to_dict() for r in r2]
    b1 = run_backtest(r1, tick_seconds=60.0, policies=AZURE_POLICIES)
    b2 = run_backtest(r2, tick_seconds=60.0, policies=AZURE_POLICIES)
    assert b1.to_summary_dict() == b2.to_summary_dict()


def test_backtest_omits_cache_affinity_and_zero_cache_benefit():
    reqs = azure_llm.load_csv(FIXTURE, variant="conv", include_failures=False)
    result = run_backtest(reqs, tick_seconds=60.0, policies=AZURE_POLICIES)
    assert "cache_affinity_baseline" not in result.policy_results
    # constraint_aware receives no cache benefit when there is no affinity key.
    assert result.policy_results["constraint_aware"].mean_reuse_fraction == 0.0


# --- 8. Unit tests do not require full dataset download ----------------------

def test_no_network_download_for_fixture(monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("unit tests must not hit the network")

    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    reqs = azure_llm.load_csv(FIXTURE, variant="conv", include_failures=True)
    result = run_backtest(reqs, tick_seconds=60.0, policies=AZURE_POLICIES)
    assert result.n_requests == len(reqs)


# --- 9. Full-trace backtest is integration-only / skipped if raw missing ----

@pytest.mark.skipif(not os.path.exists(RAW_CONV),
                    reason="raw Azure conv CSV not present (integration only)")
def test_full_trace_backtest_integration():
    reqs = azure_llm.load_csv(RAW_CONV, variant="conv", scale_rps=12)
    assert len(reqs) > 1000
    result = run_backtest(reqs, tick_seconds=15.0, policies=AZURE_POLICIES)
    ca = result.policy_results["constraint_aware"]
    assert ca.kpi.sla_safe_goodput_per_infra_dollar is not None
    # constraint_aware must never regress SLA vs FIFO (docs/RESULTS.md §6).
    fifo = result.policy_results["fifo"]
    assert ca.timeout_rate_pct_mean <= fifo.timeout_rate_pct_mean + 1e-6


# --- 10 & 11. Docs honesty: fields stated, no production-savings claims ------

def _assert_no_unhedged_banned_claims(text: str):
    low = text.lower()
    for phrase in BANNED_CLAIMS:
        idx = 0
        while True:
            pos = low.find(phrase, idx)
            if pos == -1:
                break
            prefix = low[max(0, pos - 24):pos]
            assert any(neg in prefix for neg in ("not ", "no ", "never", "n't")), (
                f"unhedged banned claim '{phrase}' near: "
                f"...{text[max(0, pos-24):pos+len(phrase)+10]}..."
            )
            idx = pos + len(phrase)


def test_docs_state_available_and_missing_fields():
    text = open(RESULTS_MD).read()
    # explicitly documents what Azure does and does NOT provide
    assert "token-demand and arrival replay, NOT a measured-latency replay" in text
    assert "TIMESTAMP,ContextTokens,GeneratedTokens" in text
    for missing in ("model / service id", "request / session id",
                    "latency / TTFT / elapsed", "cache / prefix info"):
        assert missing in text
    assert "cache_affinity_baseline" in text  # stated omitted / n/a


def test_docs_no_production_savings_claims():
    for path in (RESULTS_MD, PUBLIC_DOC):
        _assert_no_unhedged_banned_claims(open(path).read())


def test_generated_report_is_honest(tmp_path):
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "scripts", "run_azure_llm_backtest.py"),
        "--csv", FIXTURE, "--workload", "conv", "--results-md", str(out_md),
        "--summary-json", str(out_json), "--no-sweep",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    md = out_md.read_text()
    assert "NOT a measured-latency replay" in md
    assert "directional only" in md.lower()
    _assert_no_unhedged_banned_claims(md)
    payload = json.loads(out_json.read_text())
    assert payload["backtest"]["primary_kpi"] == \
        "sla_safe_goodput_per_infrastructure_dollar"
    assert payload["cache_affinity_baseline"].startswith("omitted")


# --- 12. Existing BurstGPT tests still pass (cross-check import/compat) ------

def test_burstgpt_still_importable_and_cache_key_str():
    from aurelius.traces import burstgpt
    reqs = burstgpt.load_csv(
        os.path.join(REPO_ROOT, "tests", "fixtures", "burstgpt_sample.csv"),
        include_failures=True,
    )
    # BurstGPT still populates a string model-level cache key (unchanged).
    assert all(isinstance(r.cache_affinity_key, str)
               and r.cache_affinity_key.startswith("model:") for r in reqs)
