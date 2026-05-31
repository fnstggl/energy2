"""Tests for the bounded HF ingestion pipeline.

The tests use small JSON fixture rows — never network, never large files.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from aurelius.traces.hf_corpus import ingestion  # noqa: E402
from aurelius.traces.hf_corpus.schemas import (  # noqa: E402
    BenchmarkLatencyRecord,
    HFCorpusSchemaError,
)


# A small handful of synthetic latency-benchmark rows that match the
# AgentPerfBench trace_replay shape (no real proprietary data).
_LATENCY_ROWS = [
    {
        "run_id": "run-0001", "model": "Llama-3-8B", "model_family": "llama3",
        "hardware": "H100", "engine": "vllm", "profile": "p99-tail",
        "tensor_parallelism": 1, "concurrency": 1, "num_requests": 100,
        "duration_s": 12.5, "request_throughput": 8.0,
        "input_token_throughput": 100.0, "output_token_throughput": 50.0,
        "total_token_throughput": 150.0,
        "mean_ttft_ms": 80.0, "median_ttft_ms": 75.0,
        "p90_ttft_ms": 120.0, "p99_ttft_ms": 250.0,
        "mean_tpot_ms": 25.0, "median_tpot_ms": 24.0,
        "p90_tpot_ms": 30.0, "p99_tpot_ms": 50.0,
        "mean_itl_ms": 24.0, "median_itl_ms": 23.0,
        "p90_itl_ms": 28.0, "p99_itl_ms": 48.0,
        "mean_e2el_ms": 500.0, "median_e2el_ms": 480.0,
        "p90_e2el_ms": 720.0, "p99_e2el_ms": 1100.0,
    },
    {
        "run_id": "run-0002", "model": "Llama-3-8B", "model_family": "llama3",
        "hardware": "H100", "engine": "vllm", "profile": "p99-tail",
        "tensor_parallelism": 1, "concurrency": 4, "num_requests": 200,
        "duration_s": 14.0, "request_throughput": 14.3,
        "input_token_throughput": 180.0, "output_token_throughput": 90.0,
        "total_token_throughput": 270.0,
        "mean_ttft_ms": 110.0, "median_ttft_ms": 100.0,
        "p90_ttft_ms": 160.0, "p99_ttft_ms": 320.0,
        "mean_tpot_ms": 32.0, "median_tpot_ms": 31.0,
        "p90_tpot_ms": 40.0, "p99_tpot_ms": 65.0,
        "mean_itl_ms": 31.0, "median_itl_ms": 30.0,
        "p90_itl_ms": 38.0, "p99_itl_ms": 62.0,
        "mean_e2el_ms": 700.0, "median_e2el_ms": 680.0,
        "p90_e2el_ms": 900.0, "p99_e2el_ms": 1500.0,
    },
]


# --- 1. Normalisation enforces the schema ---------------------------------


def test_normalize_rows_renames_known_columns():
    normalized, unknown, fq = ingestion.normalize_rows(
        _LATENCY_ROWS, "latency_benchmark_trace",
        allow_unknown_columns=False, source_dataset_id="test/data",
    )
    assert not unknown
    assert all("p50_ttft_ms" in r for r in normalized)
    # median_ttft_ms got renamed to p50_ttft_ms.
    assert all("median_ttft_ms" not in r for r in normalized)
    # field_quality covers every normalized column we wrote.
    for r in normalized:
        for k in r.keys():
            assert k in fq


def test_normalize_rows_rejects_unknown_columns_strict_mode():
    rows = [dict(_LATENCY_ROWS[0], mystery_metric=42)]
    with pytest.raises(ingestion.IngestionUnknownColumns):
        ingestion.normalize_rows(
            rows, "latency_benchmark_trace",
            allow_unknown_columns=False, source_dataset_id="test/data",
        )


def test_normalize_rows_records_unknown_columns_in_loose_mode():
    rows = [dict(_LATENCY_ROWS[0], mystery_metric=42, second_unknown="x")]
    _, unknown, _ = ingestion.normalize_rows(
        rows, "latency_benchmark_trace",
        allow_unknown_columns=True, source_dataset_id="test/data",
    )
    assert sorted(unknown) == ["mystery_metric", "second_unknown"]


def test_normalize_rows_rejects_mixed_or_unknown_trace_type():
    with pytest.raises(ValueError):
        ingestion.normalize_rows(_LATENCY_ROWS, "mixed_or_unknown_trace")


def test_normalize_rows_rejects_invalid_trace_type():
    with pytest.raises(ValueError):
        ingestion.normalize_rows(_LATENCY_ROWS, "not_a_real_type")


# --- 2. Bounded sample writing -------------------------------------------


def test_ingest_from_records_writes_bounded_sample(tmp_path):
    result = ingestion.ingest_from_records(
        repo_root=str(tmp_path),
        dataset_id="test/lat-bench",
        source_url="https://huggingface.co/datasets/test/lat-bench",
        license_str="apache-2.0", gated=False,
        raw_records=_LATENCY_ROWS,
        trace_type="latency_benchmark_trace",
        provenance="test/lat-bench@unit-test#v1",
        available_signals_list=["ttft", "tpot", "e2e_latency"],
        missing_signals_list=["queue_wait"],
        limitations=["unit-test fixture; not a real benchmark"],
        max_rows=10, max_bytes=64 * 1024,
        write_fixture=False,
    )
    assert result.sample_rows == 2
    assert result.sample_bytes > 0
    assert result.sha256 != ""
    assert os.path.exists(result.sample_path)
    assert os.path.exists(result.summary_path)
    # JSONL deterministic (sorted keys, no trailing whitespace).
    with open(result.sample_path, "rb") as fh:
        contents = fh.read()
    assert hashlib.sha256(contents).hexdigest() == result.sha256


def test_max_rows_caps_committed_sample(tmp_path):
    rows = _LATENCY_ROWS * 100  # 200 rows
    result = ingestion.ingest_from_records(
        repo_root=str(tmp_path),
        dataset_id="test/lat-bench-big",
        source_url="https://huggingface.co/datasets/test/lat-bench-big",
        license_str="apache-2.0", gated=False,
        raw_records=rows,
        trace_type="latency_benchmark_trace",
        provenance="test/big",
        available_signals_list=["ttft"],
        missing_signals_list=[],
        limitations=["bounded"],
        max_rows=7,  # cap
        max_bytes=1024 * 1024,
        write_fixture=False,
    )
    assert result.sample_rows == 7


def test_max_bytes_cap_triggers_truncation(tmp_path):
    # Force tiny byte cap so the writer must halve.
    result = ingestion.ingest_from_records(
        repo_root=str(tmp_path),
        dataset_id="test/lat-bench-tinybytes",
        source_url="https://huggingface.co/datasets/test/lat-bench-tinybytes",
        license_str="apache-2.0", gated=False,
        raw_records=_LATENCY_ROWS * 10,
        trace_type="latency_benchmark_trace",
        provenance="test/tinybytes",
        available_signals_list=["ttft"],
        missing_signals_list=[],
        limitations=["bounded"],
        max_rows=20,
        max_bytes=2048,
        write_fixture=False,
    )
    assert result.sample_bytes <= 2048


def test_invalid_bounds_raise(tmp_path):
    with pytest.raises(ValueError):
        ingestion.ingest_from_records(
            repo_root=str(tmp_path), dataset_id="x/y",
            source_url="u", license_str="apache-2.0", gated=False,
            raw_records=_LATENCY_ROWS, trace_type="latency_benchmark_trace",
            provenance="p", available_signals_list=[], missing_signals_list=[],
            limitations=[], max_rows=0, max_bytes=10,
        )
    with pytest.raises(ValueError):
        ingestion.ingest_from_records(
            repo_root=str(tmp_path), dataset_id="x/y",
            source_url="u", license_str="apache-2.0", gated=False,
            raw_records=_LATENCY_ROWS, trace_type="latency_benchmark_trace",
            provenance="p", available_signals_list=[], missing_signals_list=[],
            limitations=[], max_rows=10, max_bytes=0,
        )


# --- 3. Summary JSON shape -----------------------------------------------


def test_summary_json_has_required_fields(tmp_path):
    result = ingestion.ingest_from_records(
        repo_root=str(tmp_path),
        dataset_id="test/full-summary",
        source_url="https://huggingface.co/datasets/test/full-summary",
        license_str="apache-2.0", gated=False,
        raw_records=_LATENCY_ROWS,
        trace_type="latency_benchmark_trace",
        provenance="test/full",
        available_signals_list=["ttft", "tpot"],
        missing_signals_list=["queue_wait"],
        limitations=["unit-test fixture"],
        derived_fields=["throughput_ratio_derived"],
        proxy_fields=[],
        synthetic_fields=[],
        max_rows=10, max_bytes=64 * 1024,
        git_sha="deadbeef",
        write_fixture=False,
    )
    with open(result.summary_path) as fh:
        s = json.load(fh)
    for key in [
        "dataset_id", "source_url", "license", "gated", "canonical_trace_type",
        "trust_tier", "committed_sample_rows", "committed_sample_bytes",
        "sample_sha256", "raw_schema", "normalized_schema", "unknown_columns",
        "field_quality", "available_signals", "missing_signals",
        "derived_fields", "proxy_fields", "synthetic_fields", "limitations",
        "provenance", "ingestion_timestamp_s", "git_sha",
    ]:
        assert key in s, f"summary missing {key}"
    assert s["git_sha"] == "deadbeef"
    assert s["trust_tier"] == "tier_4_latency_benchmark_traces"


def test_fixture_file_written_when_requested(tmp_path):
    result = ingestion.ingest_from_records(
        repo_root=str(tmp_path),
        dataset_id="test/with-fixture",
        source_url="u",
        license_str="apache-2.0", gated=False,
        raw_records=_LATENCY_ROWS,
        trace_type="latency_benchmark_trace",
        provenance="p", available_signals_list=["ttft"],
        missing_signals_list=[],
        limitations=["test"], max_rows=10, max_bytes=64 * 1024,
        write_fixture=True,
    )
    paths = ingestion.safe_sample_paths(str(tmp_path), "test/with-fixture")
    assert os.path.exists(paths["fixture_path"])


# --- 4. Canonical record validation enforced ------------------------------


def test_canonical_record_rejects_wrong_trace_type():
    with pytest.raises(HFCorpusSchemaError):
        BenchmarkLatencyRecord(
            source_dataset_id="x", trace_type="kernel_profile_trace",
            provenance="p", field_quality={"model": "real"},
            limitations=(),
        )


def test_canonical_record_rejects_unknown_field_quality_key():
    with pytest.raises(HFCorpusSchemaError):
        BenchmarkLatencyRecord(
            source_dataset_id="x", trace_type="latency_benchmark_trace",
            provenance="p", field_quality={"not_a_field": "real"},
            limitations=(),
        )


def test_canonical_record_rejects_unknown_field_quality_value():
    with pytest.raises(HFCorpusSchemaError):
        BenchmarkLatencyRecord(
            source_dataset_id="x", trace_type="latency_benchmark_trace",
            provenance="p", field_quality={"model": "totally_made_up"},
            limitations=(),
        )


# --- 5. Determinism --------------------------------------------------------


def test_deterministic_jsonl_writing(tmp_path):
    out1 = tmp_path / "1" / "x.jsonl"
    out2 = tmp_path / "2" / "x.jsonl"
    n1, sha1 = ingestion.write_jsonl_sample(_LATENCY_ROWS, str(out1))
    n2, sha2 = ingestion.write_jsonl_sample(_LATENCY_ROWS, str(out2))
    assert n1 == n2
    assert sha1 == sha2


# --- 6. Loaders -----------------------------------------------------------


def test_try_load_json_rows_jsonl(tmp_path):
    p = tmp_path / "in.jsonl"
    with open(p, "w") as fh:
        for r in _LATENCY_ROWS:
            fh.write(json.dumps(r) + "\n")
    rows = ingestion.try_load_json_rows(str(p), max_rows=10)
    assert len(rows) == 2
    assert rows[0]["model"] == "Llama-3-8B"


def test_try_load_json_rows_json_array(tmp_path):
    p = tmp_path / "in.json"
    with open(p, "w") as fh:
        json.dump(_LATENCY_ROWS, fh)
    rows = ingestion.try_load_json_rows(str(p), max_rows=10)
    assert len(rows) == 2
