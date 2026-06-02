#!/usr/bin/env python3
"""Broadened HF discovery — bounded ingest of optimum-benchmark/llm-perf-leaderboard.

This script extends the federated benchmark corpus with HuggingFace's own
``optimum-benchmark`` performance leaderboard data. The leaderboard contains
real measured prefill (TTFT) and decode (TPOT) latency at p50/p90/p95/p99,
GPU/CPU/RAM energy per request (kWh), and peak VRAM/RAM memory across a
matrix of (model, hardware, backend, quantization) configurations.

This fills a major gap in Aurelius' priors:

- **Quantization-aware placement priors** — measured latency × memory ×
  energy delta for AWQ / GPTQ / BNB / TorchAO vs unquantized across A100 /
  A10 / T4. The constraint-aware placement engine currently has no
  cross-quantization performance surface; this is the strongest public one.
- **Energy-aware scheduling priors** — real GPU/CPU/RAM energy per request
  (kWh), measured by ``codecarbon`` inside optimum-benchmark. Directly
  feeds the energy / carbon cost terms in the Aurelius objective.
- **Cross-hardware decode throughput priors** — tokens/s decode + prefill
  throughput across A100 / A10 / T4 / Sapphire-Rapids vCPU. Lets the
  routing/residency engine reason about which GPU class a model fits.
- **OOM / memory-pressure priors** — peak ``max_global_vram`` and
  ``max_allocated`` per (model, quantization, hardware). Drops in for
  the constraint-aware placement engine's memory headroom check.

This is a Tier-4 ``latency_benchmark_trace`` dataset — NOT pilot telemetry.
Treat as performance / energy priors only. Pilot telemetry remains the
only Tier 1 calibration source.

The HF dataset card has NO declared license (frontmatter is empty, no
LICENSE file at the root). The upstream library (huggingface/optimum-
benchmark) is Apache-2.0, but the dataset itself does not carry an
explicit license, so we record ``license=None`` and DO NOT commit a
normalised redistributable sample — only the bounded fixture (≤ 16 KiB)
plus the summary / schema / rollups (which are metadata about the source
schema, not redistributed source data). The raw CSVs are gitignored.

Redistribution gate wiring
--------------------------

Seventh consumer of the canonical
:func:`aurelius.ingestion.redistribution_gate.decide_redistribution`
gate (after ``scripts/audit_hf_redistribution_gate.py``,
``scripts/commit_hf_gap_normalized_samples.py``,
``scripts/ingest_hf_agent_llm_traces.py``,
``scripts/ingest_hf_h200_quantization.py``,
``scripts/ingest_hf_llm_energy_consumption.py``, and
``scripts/ingest_hf_latency_benchmarks.py``).

The pre-wiring shape carried a hard-coded ``LICENSE = None`` constant
and an inline skip-reason string
``"license_unspecified_no_redistribution_promise"`` baked into
``_finalize_config``. The refactor lifts the raw HF license tag and
its human-curated provenance to module-level constants
(``LICENSE_TAG`` / ``LICENSE_SOURCE``) and routes the commit decision
through the canonical gate; the skip-reason string is preserved
verbatim on the committed
``committed_normalized_sample_reason_skipped`` field for downstream
back-compat (pinned by ``test_license_recorded_as_none`` in
``tests/test_hf_optimum_benchmark_ingest.py``).

Under the default ``deny_all`` / zero-grants ledger and the dataset's
declared-None license, the gate returns
``permitted=False``, ``license_status="unspecified_no_committed_sample"``,
``reason_code="no_grant_recorded"`` — identical to the v1 commit
behaviour (no committed normalised sample), with the gate verdict
additively recorded in the new ``redistribution_gate_*`` fields. The
committed federated corpus is unchanged on disk.

Schema observed (149 raw columns per CSV; only ~20 are used by Aurelius):

    config.backend.model          → model
    config.backend.name           → backend_name (pytorch | openvino | onnxruntime)
    config.backend.torch_dtype    → dtype
    config.backend.quantization_scheme → quantization_scheme (unquantized | awq | bnb | gptq | torchao)
    config.environment.gpu        → gpu (list, take first)
    config.scenario.input_shapes.batch_size       → batch_size
    config.scenario.input_shapes.sequence_length  → sequence_length
    config.scenario.generate_kwargs.max_new_tokens → new_tokens
    report.prefill.latency.{mean,p50,p90,p95,p99,count,total} → *_ttft_s + count
    report.prefill.throughput.value (tokens/s)   → prefill_throughput_tok_s
    report.prefill.memory.{max_global_vram,max_allocated,max_reserved} (MB) → prefill_*_mb
    report.prefill.energy.{cpu,ram,gpu,total} (kWh) → prefill_energy_*_kwh
    report.decode.latency.{mean,p50,p90,p95,p99,count}        → *_tpot_s + count
    report.decode.throughput.value (tokens/s)    → decode_throughput_tok_s
    report.decode.memory.{...}                   → decode_*_mb
    report.decode.energy.{cpu,ram,gpu,total}     → decode_energy_*_kwh
    error_type, error_message                    → failure_label

NO production claims. NO scheduler / controller / robust-energy-engine changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.ingestion.operator_redistribution_policy import (  # noqa: E402
    OperatorPolicyLedger,
)
from aurelius.ingestion.redistribution_gate import (  # noqa: E402
    RedistributionGateDecision,
    decide_redistribution,
)
from aurelius.traces.hf_corpus import promotion  # noqa: E402

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"
POLICY_PATH = DISC_DIR / "operator_redistribution_policy.json"

DATASET_ID = "optimum-benchmark/llm-perf-leaderboard"
SAFE_NAME = "optimum-benchmark__llm-perf-leaderboard"

# Raw HF license tag + human-curated provenance for the canonical
# redistribution gate. The gate (not this script) classifies the tag
# into a ``permissive_*`` / ``unspecified_no_committed_sample`` /
# ``declared_non_permissive`` status code. Keeping the raw tag separate
# from the canonical status here means a future HF tag change (e.g.
# the dataset owner declaring ``license: apache-2.0``) is a one-line
# edit; the gate handles the rest.
LICENSE_TAG: Optional[str] = None  # no `license:` field on the HF card
LICENSE_SOURCE = (
    "HF card frontmatter has no `license:` field; recorded as unspecified"
)
GATE_SCOPE = "committed_normalized_sample"

# Pre-existing script-level reason string for skipping the committed
# normalised sample under ``license = None``. Kept VERBATIM —
# ``test_license_recorded_as_none`` in
# ``tests/test_hf_optimum_benchmark_ingest.py`` pins this exact string
# on every config's summary.json. The canonical gate verdict is
# recorded additively in the new ``redistribution_gate_*`` fields.
COMMITTED_NORMALIZED_SAMPLE_SKIP_REASON = (
    "license_unspecified_no_redistribution_promise"
)

# Backwards-compatibility alias — some external callers (and the older
# audit summary writer below) still reference ``LICENSE``. The single
# source of truth is ``LICENSE_TAG`` above.
LICENSE: Optional[str] = LICENSE_TAG

FIXTURE_MAX_BYTES = 16 * 1024
PER_CSV_MAX_BYTES = 25 * 1024 * 1024  # 25 MiB per file cap
PER_DATASET_TIMEOUT_S = 30 * 60

# CSV reader needs a higher cell-size limit for some configs (torchao CSV
# stores per-iteration latency arrays >131 KiB inside ``report.prefill.
# latency.values``). We never normalise those columns; bumping the limit
# just lets DictReader iterate without raising.
csv.field_size_limit(64 * 1024 * 1024)

logger = logging.getLogger("aurelius.hf_optimum_benchmark_ingest")


# ---------------------------------------------------------------------------
# Config matrix — 10 representative CSVs covering A100 / A10 / T4 / C7i vCPU
# × pytorch-cuda (unquantized, awq, bnb, gptq, torchao) and CPU backends
# (pytorch, openvino, onnxruntime).
# ---------------------------------------------------------------------------


CONFIGS: list[dict] = [
    {
        "config": "pytorch_cuda_unquantized_1xA100",
        "remote": "data/perf-df-pytorch-cuda-unquantized-1xA100.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "NVIDIA A100-SXM4-80GB", "quantization": "unquantized",
    },
    {
        "config": "pytorch_cuda_unquantized_1xA10",
        "remote": "data/perf-df-pytorch-cuda-unquantized-1xA10.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "NVIDIA A10G", "quantization": "unquantized",
    },
    {
        "config": "pytorch_cuda_unquantized_1xT4",
        "remote": "data/perf-df-pytorch-cuda-unquantized-1xT4.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "Tesla T4", "quantization": "unquantized",
    },
    {
        "config": "pytorch_cuda_bnb_1xA100",
        "remote": "data/perf-df-pytorch-cuda-bnb-1xA100.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "NVIDIA A100-SXM4-80GB", "quantization": "bnb",
    },
    {
        "config": "pytorch_cuda_gptq_1xA100",
        "remote": "data/perf-df-pytorch-cuda-gptq-1xA100.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "NVIDIA A100-SXM4-80GB", "quantization": "gptq",
    },
    {
        "config": "pytorch_cuda_awq_1xA10",
        "remote": "data/perf-df-pytorch-cuda-awq-1xA10.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "NVIDIA A10G", "quantization": "awq",
    },
    {
        "config": "pytorch_cuda_bnb_1xT4",
        "remote": "data/perf-df-pytorch-cuda-bnb-1xT4.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "Tesla T4", "quantization": "bnb",
    },
    {
        "config": "pytorch_cuda_torchao_1xA10",
        "remote": "data/perf-df-pytorch-cuda-torchao-1xA10.csv",
        "backend": "pytorch", "device_class": "cuda",
        "gpu": "NVIDIA A10G", "quantization": "torchao",
    },
    {
        "config": "pytorch_cpu_unquantized_32vCPU_C7i",
        "remote": "data/perf-df-pytorch-cpu-unquantized-32vCPU-C7i.csv",
        "backend": "pytorch", "device_class": "cpu",
        "gpu": "32vCPU Sapphire-Rapids (AWS C7i)",
        "quantization": "unquantized",
    },
    # NOTE: ``openvino_cpu_unquantized_32vCPU_C7i`` is intentionally omitted.
    # Every row in that CSV is an isolated-process crash with
    # ``report.traceback`` populated but ZERO ``report.prefill.latency.*``
    # / ``report.decode.latency.*`` columns present. Including it would
    # write an all-null latency_benchmark_trace, which is misleading.
    # Recorded as a discovery-only failure case in the audit summary so
    # the next-run agent doesn't re-probe.
]


# ---------------------------------------------------------------------------
# HTTP / hashing helpers
# ---------------------------------------------------------------------------


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN")


def _hf_url(dataset_id: str, path: str) -> str:
    return (f"https://huggingface.co/datasets/{dataset_id}/resolve/main/"
            f"{urllib.parse.quote(path)}")


def _download(url: str, dest: Path, *, max_bytes: int) -> dict:
    headers = {}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return {
            "url": url, "dest": str(dest),
            "downloaded_bytes": dest.stat().st_size,
            "cached": True, "truncated": None, "error": None,
        }
    req = urllib.request.Request(url, headers=headers)
    truncated = False
    written = 0
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            with open(dest, "wb") as fh:
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    if written + len(chunk) > max_bytes:
                        chunk = chunk[: max_bytes - written]
                        fh.write(chunk)
                        written += len(chunk)
                        truncated = True
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    if (time.time() - t0) > 60.0 and written > 0:
                        logger.info(
                            "    downloading %s: %.1f MiB so far",
                            url.rsplit("/", 1)[-1], written / 1024 / 1024,
                        )
                        t0 = time.time()
    except urllib.error.HTTPError as e:
        return {
            "url": url, "dest": str(dest), "downloaded_bytes": 0,
            "cached": False, "truncated": None, "error": f"HTTP {e.code}",
        }
    return {
        "url": url, "dest": str(dest), "downloaded_bytes": written,
        "cached": False, "truncated": truncated, "error": None,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(64 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Redistribution gate — wire the canonical gate, do not classify here
# ---------------------------------------------------------------------------


def _load_ledger(policy_path: Path = POLICY_PATH) -> OperatorPolicyLedger:
    """Load the operator policy ledger from disk, or fall back to empty.

    The committed default file ships zero grants under
    ``policy_default=deny_all``; an absent file is identical in
    behaviour. We use ``empty()`` as the fallback so the script
    remains self-sufficient in a fresh checkout that may not yet have
    the committed JSON pulled — the gate still produces correct
    decisions (deny for ``license = None``) instead of crashing.
    """

    if policy_path.exists():
        return OperatorPolicyLedger.load(policy_path)
    return OperatorPolicyLedger.empty()


def evaluate_redistribution(
    *,
    ledger: OperatorPolicyLedger,
    license_tag: Optional[str] = LICENSE_TAG,
    dataset_id: str = DATASET_ID,
    scope: str = GATE_SCOPE,
    now_iso: Optional[str] = None,
) -> RedistributionGateDecision:
    """Ask the canonical gate whether the committed normalised sample
    of this dataset may be redistributed under the supplied license tag.

    Pure function — no I/O. Exposed so tests can drive the gate path
    without invoking the CSV download / normalisation pipeline. The
    defaults reflect the module-level constants this script ships;
    tests override them to verify the wiring (e.g. swap ``license_tag``
    to ``"apache-2.0"`` and check that the gate permits).

    Under the default-empty ledger and the dataset's declared-None
    license tag, this returns
    ``permitted=False``,
    ``license_status="unspecified_no_committed_sample"``,
    ``reason_code="no_grant_recorded"`` — identical to the v1 commit
    behaviour (skip the committed normalised sample).
    """

    return decide_redistribution(
        dataset_id=dataset_id,
        license_str=license_tag,
        scope=scope,
        ledger=ledger,
        now_iso=now_iso,
    )


# ---------------------------------------------------------------------------
# CSV row → normalized BenchmarkLatencyRecord-shaped row
# ---------------------------------------------------------------------------


def _maybe_float(v: Any) -> Optional[float]:
    if v is None or v == "" or v == "None" or v == "nan":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _maybe_int(v: Any) -> Optional[int]:
    if v is None or v == "" or v == "None" or v == "nan":
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _maybe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "None", "nan", "[]"):
        return None
    return s


def _model_family(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    # ``namespace/name`` → family from the name half. Common HF naming.
    name = model.split("/", 1)[-1].lower()
    for prefix, family in (
        ("yi-", "Yi"), ("decilm", "DeciLM"), ("decicoder", "DeciCoder"),
        ("gpt-neo", "GPT-Neo"), ("gpt-j", "GPT-J"), ("pythia", "Pythia"),
        ("polyglot", "Polyglot"), ("gpt-neox", "GPT-NeoX"),
        ("qwen", "Qwen"), ("llama", "Llama"), ("codellama", "CodeLlama"),
        ("mistral", "Mistral"), ("mixtral", "Mixtral"),
        ("falcon", "Falcon"), ("opt-", "OPT"), ("bloom", "Bloom"),
        ("gemma", "Gemma"), ("phi-", "Phi"), ("stablelm", "StableLM"),
        ("codegen", "CodeGen"), ("redpajama", "RedPajama"),
        ("openchat", "OpenChat"), ("vicuna", "Vicuna"),
        ("xglm", "XGLM"), ("xlm", "XLM"), ("baichuan", "Baichuan"),
        ("internlm", "InternLM"), ("rwkv", "RWKV"),
    ):
        if name.startswith(prefix):
            return family
    # Fallback: first token before "-".
    return name.split("-", 1)[0]


def _normalize_row(r: dict, meta: dict, dataset_id: str, git_sha: str) -> dict:
    """Normalize one CSV row into a BenchmarkLatencyRecord-shaped dict.

    Latency values in the CSV are seconds; we convert to milliseconds for
    consistency with the canonical ``mean_ttft_ms`` etc. naming.
    Energy values are kWh; we keep ``kWh`` units explicit in the field
    name so downstream consumers don't accidentally treat them as joules.
    """

    model = _maybe_str(r.get("config.backend.model"))
    dtype = _maybe_str(r.get("config.backend.torch_dtype"))
    quant = _maybe_str(r.get("config.backend.quantization_scheme")) or meta["quantization"]
    backend_name = _maybe_str(r.get("config.backend.name")) or meta["backend"]
    bs = _maybe_int(r.get("config.scenario.input_shapes.batch_size"))
    sl = _maybe_int(r.get("config.scenario.input_shapes.sequence_length"))
    new_tokens = _maybe_int(
        r.get("config.scenario.generate_kwargs.max_new_tokens")
    )

    def _to_ms(field: str) -> Optional[float]:
        v = _maybe_float(r.get(field))
        return v * 1000.0 if v is not None else None

    rec = {
        "source_dataset_id": dataset_id,
        "trace_type": "latency_benchmark_trace",
        "provenance": (
            f"{dataset_id}@{meta['config']}#{meta['remote']}"
            f"#git={git_sha[:7]}"
        ),
        # Identity / config
        "model": model,
        "model_family": _model_family(model),
        "gpu": meta["gpu"],
        "device_class": meta["device_class"],
        "backend_name": backend_name,
        "engine": backend_name,
        "quantization_scheme": quant if quant != "" else "unquantized",
        "dtype": dtype,
        "batch_size": bs,
        "sequence_length": sl,
        "new_tokens": new_tokens,
        # Prefill (= TTFT for the first new token) — seconds → ms.
        "mean_ttft_ms": _to_ms("report.prefill.latency.mean"),
        "p50_ttft_ms": _to_ms("report.prefill.latency.p50"),
        "p90_ttft_ms": _to_ms("report.prefill.latency.p90"),
        "p95_ttft_ms": _to_ms("report.prefill.latency.p95"),
        "p99_ttft_ms": _to_ms("report.prefill.latency.p99"),
        "ttft_count": _maybe_int(r.get("report.prefill.latency.count")),
        "prefill_throughput_tok_s": _maybe_float(
            r.get("report.prefill.throughput.value")
        ),
        "prefill_max_vram_mb": _maybe_float(
            r.get("report.prefill.memory.max_global_vram")
        ),
        "prefill_max_allocated_mb": _maybe_float(
            r.get("report.prefill.memory.max_allocated")
        ),
        "prefill_max_reserved_mb": _maybe_float(
            r.get("report.prefill.memory.max_reserved")
        ),
        "prefill_max_ram_mb": _maybe_float(
            r.get("report.prefill.memory.max_ram")
        ),
        "prefill_energy_total_kwh": _maybe_float(
            r.get("report.prefill.energy.total")
        ),
        "prefill_energy_gpu_kwh": _maybe_float(
            r.get("report.prefill.energy.gpu")
        ),
        "prefill_energy_cpu_kwh": _maybe_float(
            r.get("report.prefill.energy.cpu")
        ),
        "prefill_energy_ram_kwh": _maybe_float(
            r.get("report.prefill.energy.ram")
        ),
        # Decode (= TPOT inter-token-latency) — seconds → ms.
        "mean_tpot_ms": _to_ms("report.decode.latency.mean"),
        "p50_tpot_ms": _to_ms("report.decode.latency.p50"),
        "p90_tpot_ms": _to_ms("report.decode.latency.p90"),
        "p95_tpot_ms": _to_ms("report.decode.latency.p95"),
        "p99_tpot_ms": _to_ms("report.decode.latency.p99"),
        "tpot_count": _maybe_int(r.get("report.decode.latency.count")),
        "decode_throughput_tok_s": _maybe_float(
            r.get("report.decode.throughput.value")
        ),
        "decode_max_vram_mb": _maybe_float(
            r.get("report.decode.memory.max_global_vram")
        ),
        "decode_max_allocated_mb": _maybe_float(
            r.get("report.decode.memory.max_allocated")
        ),
        "decode_energy_total_kwh": _maybe_float(
            r.get("report.decode.energy.total")
        ),
        "decode_energy_gpu_kwh": _maybe_float(
            r.get("report.decode.energy.gpu")
        ),
        # Failure
        "error_type": _maybe_str(r.get("error_type")),
        "error_message": _maybe_str(r.get("error_message")),
    }
    return rec


# ---------------------------------------------------------------------------
# Statistical helpers (shared layout w/ existing latency-benchmark script)
# ---------------------------------------------------------------------------


def _percentiles(values: list[float]) -> dict:
    vs = sorted(v for v in values if v is not None)
    if not vs:
        return {}
    n = len(vs)

    def _q(p: float) -> float:
        idx = max(0, min(n - 1, int(round((n - 1) * p))))
        return vs[idx]

    return {
        "count": n, "min": vs[0],
        "p50": _q(0.50), "p90": _q(0.90),
        "p95": _q(0.95), "p99": _q(0.99),
        "max": vs[-1], "mean": sum(vs) / n,
    }


_ROLLUP_KEYS = [
    "mean_ttft_ms", "p50_ttft_ms", "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p50_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "prefill_throughput_tok_s", "decode_throughput_tok_s",
    "prefill_energy_total_kwh", "prefill_energy_gpu_kwh",
    "decode_energy_total_kwh", "decode_energy_gpu_kwh",
    "prefill_max_vram_mb", "decode_max_vram_mb",
    "prefill_max_allocated_mb", "decode_max_allocated_mb",
]


def _statistical_rollups(rows: list[dict],
                         stratification_keys: list[str]) -> dict:
    overall: dict[str, dict] = {}
    for k in _ROLLUP_KEYS:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        if vals:
            overall[k] = _percentiles(vals)
    by_strata: dict[str, dict] = {}
    subgroup_counts: dict[str, int] = {}
    if stratification_keys:
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            key = tuple(r.get(k) for k in stratification_keys)
            groups.setdefault(key, []).append(r)
        for grp_key, grp_rows in groups.items():
            label = "|".join(
                f"{stratification_keys[i]}={grp_key[i]}"
                for i in range(len(grp_key))
            )
            subgroup_counts[label] = len(grp_rows)
            if len(grp_rows) >= 5:
                grp_stats: dict[str, dict] = {}
                for k in [
                    "mean_ttft_ms", "p95_ttft_ms",
                    "mean_tpot_ms", "p95_tpot_ms",
                    "prefill_throughput_tok_s", "decode_throughput_tok_s",
                    "prefill_energy_total_kwh", "decode_energy_total_kwh",
                    "prefill_max_vram_mb", "decode_max_vram_mb",
                ]:
                    vals = [r.get(k) for r in grp_rows if r.get(k) is not None]
                    if vals:
                        grp_stats[k] = _percentiles(vals)
                if grp_stats:
                    by_strata[label] = grp_stats
    return {
        "overall": overall, "by_strata": by_strata,
        "subgroup_counts": subgroup_counts,
    }


def _sample_strength(rows: int, has_strata_coverage: bool) -> str:
    if rows == 0:
        return "fixture_only"
    if rows < 25:
        return "weak" if not has_strata_coverage else "moderate"
    if rows < 200:
        return "moderate"
    return "strong"


# ---------------------------------------------------------------------------
# Schema profile + mapping for the 149-column raw CSV
# ---------------------------------------------------------------------------


def _normalized_field_for(raw_col: str) -> tuple[str, str, str, list[str]]:
    """Map a raw column → (normalized_field, field_quality, signal_category, usable_for).

    Returns ``(normalized_field, field_quality, aurelius_signal_category,
    usable_for)``. Columns with no Aurelius mapping fall through to
    ``metadata_only``.
    """
    mapping = {
        # Identity / scenario config — all "real" measurements of the run.
        "config.backend.model":
            ("model", "real", "metadata_only", ["latency_prior"]),
        "config.backend.name":
            ("backend_name", "real", "metadata_only", ["latency_prior"]),
        "config.backend.torch_dtype":
            ("dtype", "real", "metadata_only", ["latency_prior"]),
        "config.backend.quantization_scheme":
            ("quantization_scheme", "real", "metadata_only",
             ["latency_prior", "constraint_aware_backtest"]),
        "config.environment.gpu":
            ("gpu", "real", "gpu_resource",
             ["latency_prior", "constraint_aware_backtest"]),
        "config.environment.gpu_count":
            ("gpu_count", "real", "gpu_resource", ["latency_prior"]),
        "config.environment.gpu_vram_mb":
            ("gpu_vram_mb", "real", "gpu_resource", ["latency_prior"]),
        "config.scenario.input_shapes.batch_size":
            ("batch_size", "real", "request_arrival",
             ["latency_prior", "constraint_aware_backtest"]),
        "config.scenario.input_shapes.sequence_length":
            ("sequence_length", "real", "tokens", ["latency_prior"]),
        "config.scenario.generate_kwargs.max_new_tokens":
            ("new_tokens", "real", "tokens", ["latency_prior"]),
        # Prefill (TTFT) — measured latency p50/p90/p95/p99.
        "report.prefill.latency.mean":
            ("mean_ttft_ms", "real", "latency", ["latency_prior"]),
        "report.prefill.latency.p50":
            ("p50_ttft_ms", "real", "latency", ["latency_prior"]),
        "report.prefill.latency.p90":
            ("p90_ttft_ms", "real", "latency", ["latency_prior"]),
        "report.prefill.latency.p95":
            ("p95_ttft_ms", "real", "latency", ["latency_prior"]),
        "report.prefill.latency.p99":
            ("p99_ttft_ms", "real", "latency", ["latency_prior"]),
        "report.prefill.latency.count":
            ("ttft_count", "real", "metadata_only", ["latency_prior"]),
        "report.prefill.latency.total":
            ("ttft_total_s", "real", "latency", ["latency_prior"]),
        "report.prefill.latency.values":
            ("ttft_values_list_skipped", "real", "irrelevant", []),
        "report.prefill.throughput.value":
            ("prefill_throughput_tok_s", "real", "throughput",
             ["throughput_prior"]),
        "report.prefill.memory.max_global_vram":
            ("prefill_max_vram_mb", "real", "memory",
             ["latency_prior", "constraint_aware_backtest"]),
        "report.prefill.memory.max_allocated":
            ("prefill_max_allocated_mb", "real", "memory",
             ["latency_prior"]),
        "report.prefill.memory.max_reserved":
            ("prefill_max_reserved_mb", "real", "memory",
             ["latency_prior"]),
        "report.prefill.memory.max_ram":
            ("prefill_max_ram_mb", "real", "memory", ["latency_prior"]),
        "report.prefill.memory.max_process_vram":
            ("prefill_max_process_vram_mb", "real", "memory",
             ["latency_prior"]),
        "report.prefill.energy.total":
            ("prefill_energy_total_kwh", "real", "cost_energy_carbon",
             ["latency_prior", "constraint_aware_backtest"]),
        "report.prefill.energy.gpu":
            ("prefill_energy_gpu_kwh", "real", "cost_energy_carbon",
             ["latency_prior", "constraint_aware_backtest"]),
        "report.prefill.energy.cpu":
            ("prefill_energy_cpu_kwh", "real", "cost_energy_carbon",
             ["latency_prior"]),
        "report.prefill.energy.ram":
            ("prefill_energy_ram_kwh", "real", "cost_energy_carbon",
             ["latency_prior"]),
        # Decode (TPOT)
        "report.decode.latency.mean":
            ("mean_tpot_ms", "real", "latency", ["latency_prior"]),
        "report.decode.latency.p50":
            ("p50_tpot_ms", "real", "latency", ["latency_prior"]),
        "report.decode.latency.p90":
            ("p90_tpot_ms", "real", "latency", ["latency_prior"]),
        "report.decode.latency.p95":
            ("p95_tpot_ms", "real", "latency", ["latency_prior"]),
        "report.decode.latency.p99":
            ("p99_tpot_ms", "real", "latency", ["latency_prior"]),
        "report.decode.latency.count":
            ("tpot_count", "real", "metadata_only", ["latency_prior"]),
        "report.decode.throughput.value":
            ("decode_throughput_tok_s", "real", "throughput",
             ["throughput_prior"]),
        "report.decode.memory.max_global_vram":
            ("decode_max_vram_mb", "real", "memory",
             ["latency_prior", "constraint_aware_backtest"]),
        "report.decode.memory.max_allocated":
            ("decode_max_allocated_mb", "real", "memory",
             ["latency_prior"]),
        "report.decode.energy.total":
            ("decode_energy_total_kwh", "real", "cost_energy_carbon",
             ["latency_prior", "constraint_aware_backtest"]),
        "report.decode.energy.gpu":
            ("decode_energy_gpu_kwh", "real", "cost_energy_carbon",
             ["latency_prior", "constraint_aware_backtest"]),
        # Failure / error labels
        "error_type":
            ("error_type", "real", "failure_timeout", ["latency_prior"]),
        "error_message":
            ("error_message", "real", "failure_timeout", []),
    }
    if raw_col in mapping:
        return mapping[raw_col]
    # All remaining 100+ optimum-benchmark hydra config / environment
    # columns are real metadata about the run (Python version, transformers
    # version, scenario flags, ...). Recorded but not routed.
    return (raw_col, "real", "metadata_only", [])


def _build_columns_map(raw_columns: list[str],
                       normalized_rows: list[dict],
                       raw_rows: list[dict]) -> list[dict]:
    cols = []
    n_probe = min(100, len(raw_rows)) or 1
    for col in raw_columns:
        norm, fq, sig, usable = _normalized_field_for(col)
        miss = sum(
            1 for r in raw_rows[:n_probe]
            if r.get(col) in (None, "", "None", "nan")
        )
        cols.append({
            "raw_column_name": col,
            "normalized_field": norm,
            "dtypes": ["str"],  # CSV reads everything as str; coercion is downstream
            "field_quality": fq,
            "aurelius_signal_category": sig,
            "usable_for": usable,
            "presence_rate": 1.0 - miss / n_probe,
            "missing_rate": miss / n_probe,
            "notes": f"raw '{col}' → '{norm}'",
        })
    return cols


# ---------------------------------------------------------------------------
# Finalize one config — write all artefacts, run promotion gates
# ---------------------------------------------------------------------------


AURELIUS_AVAILABLE_SIGNALS = [
    "ttft",
    "tpot",
    "throughput",
    "input_tokens",
    "output_tokens",
    "batch_size",
    "gpu_type",
    "model_id",
    "engine",
    "memory_pressure",
    "energy_per_request",
    "failure_label",
]


AURELIUS_MISSING_SIGNALS = [
    "concurrency",  # all runs are single-stream, batch_size=1
    "queue_state",
    "queue_wait",
    "request_arrival",
    "cache_hit",
    "prefix_reuse",
    "model_residency",
    "autoscaling",
    "replica_count",
    "kv_block_hashes",
    "sla_label",
    "timeout_label",
    "cold_start",
    "routing",
    "cost_per_request",  # we have energy but not $ cost
    "carbon_intensity",  # energy is in kWh; carbon intensity is location-dependent
]


def _finalize_config(*, meta: dict, raw_local_path: Path,
                     raw_columns: list[str], raw_rows: list[dict],
                     normalized_rows: list[dict],
                     git_sha: str,
                     gate_decision: RedistributionGateDecision) -> dict:
    config = meta["config"]
    safe_name = SAFE_NAME
    proc_dir = HF_DIR / safe_name / config / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Analysis sample JSONL (gitignored — large)
    analysis_path = proc_dir / "analysis_sample.jsonl"
    analysis_bytes = io.BytesIO()
    for row in normalized_rows:
        analysis_bytes.write(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        )
    analysis_path.write_bytes(analysis_bytes.getvalue())
    analysis_sha = _sha256_file(analysis_path)
    analysis_size = analysis_path.stat().st_size

    # 2. Fixture sample (≤16 KiB, ≤5 rows, deterministic, committed).
    # Prefer rows with measured TTFT or TPOT — the CSV is alphabetical by
    # model name, so the first 5 rows are often OOM failures on large
    # models (e.g. databricks/dbrx-base, meta-llama/Llama-2-70b-hf).
    # Falling back to the head if no measured rows exist.
    fixture_path = FIXTURES_DIR / f"{safe_name}__{config}_sample.jsonl"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    measured_rows = [
        r for r in normalized_rows
        if r.get("mean_ttft_ms") is not None
        or r.get("mean_tpot_ms") is not None
    ]
    fixture_source = measured_rows if measured_rows else normalized_rows
    fixture_rows: list[dict] = []
    fixture_bytes = io.BytesIO()
    for row in fixture_source:
        line = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        if fixture_bytes.tell() + len(line) > FIXTURE_MAX_BYTES:
            break
        if len(fixture_rows) >= 5:
            break
        fixture_bytes.write(line)
        fixture_rows.append(row)
    fixture_path.write_bytes(fixture_bytes.getvalue())
    fixture_sha = _sha256_file(fixture_path)
    fixture_size = fixture_path.stat().st_size

    # 3. Committed normalised sample — the canonical redistribution
    # gate (not this script) decides whether the sample may be
    # redistributed. Under the default ``deny_all`` / zero-grants
    # ledger and the dataset's declared-None license, the gate returns
    # ``permitted=False`` so no committed normalised sample is written
    # — identical to the v1 behaviour. The pre-existing
    # ``"license_unspecified_no_redistribution_promise"`` reason string
    # is preserved verbatim on ``committed_normalized_sample_reason_skipped``
    # so downstream tests stay green (pinned by
    # ``test_license_recorded_as_none`` in
    # ``tests/test_hf_optimum_benchmark_ingest.py``); the canonical
    # gate verdict is recorded additively in the new
    # ``redistribution_gate_*`` fields written into the summary below.
    committed_norm_path: Optional[str] = None
    committed_norm_rows = 0
    committed_norm_bytes = 0
    committed_norm_sha: Optional[str] = None
    committed_norm_reason_skipped: Optional[str] = None
    if gate_decision.permitted:
        committed_path = proc_dir / "committed_normalized_sample.jsonl"
        buf = io.BytesIO()
        for row in normalized_rows:
            buf.write((json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        committed_path.write_bytes(buf.getvalue())
        committed_norm_rows = len(normalized_rows)
        committed_norm_path = str(committed_path.relative_to(REPO_ROOT))
        committed_norm_bytes = committed_path.stat().st_size
        committed_norm_sha = _sha256_file(committed_path)
    else:
        committed_norm_reason_skipped = COMMITTED_NORMALIZED_SAMPLE_SKIP_REASON

    # 4. Schema profile — every raw column gets a dtype + up to 3 examples.
    dtypes: dict[str, list[str]] = {}
    example_values: dict[str, list[str]] = {}
    missing_rates: dict[str, float] = {}
    n_probe = min(100, len(raw_rows)) or 1
    for col in raw_columns:
        examples: list[str] = []
        missing = 0
        for r in raw_rows[:n_probe]:
            v = r.get(col)
            if v in (None, "", "None", "nan"):
                missing += 1
                continue
            if len(examples) < 3:
                examples.append(repr(v)[:120])
        dtypes[col] = ["str"]
        example_values[col] = examples
        missing_rates[col] = missing / n_probe

    schema_profile = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_file": meta["remote"],
        "inspected_row_count": len(raw_rows),
        "raw_columns": raw_columns,
        "nested_keys": [],
        "dtypes": dtypes,
        "example_values": example_values,
        "missing_rates": missing_rates,
        "unknown_columns": [],
        "rejected_columns": [],
        "accepted_columns": raw_columns,
        "list_length_summaries": {},
        "file_size_bytes": (
            raw_local_path.stat().st_size if raw_local_path.exists() else 0
        ),
    }
    (proc_dir / "schema_profile.json").write_text(
        json.dumps(schema_profile, indent=2, sort_keys=True)
    )

    # 5. Schema mapping
    columns_map = _build_columns_map(raw_columns, normalized_rows, raw_rows)
    schema_mapping = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "accepted_columns": raw_columns,
        "rejected_columns": [],
        "unknown_columns": [],
        "columns": columns_map,
    }
    (proc_dir / "schema_mapping.json").write_text(
        json.dumps(schema_mapping, indent=2, sort_keys=True)
    )

    # 6. Statistical rollups — stratify by (model_family, dtype, quantization).
    strat_keys = ["model_family", "dtype", "quantization_scheme"]
    rollups = _statistical_rollups(normalized_rows, strat_keys)
    rollups_path = proc_dir / "statistical_rollups.json"
    rollups_path.write_text(json.dumps(rollups, indent=2, sort_keys=True))
    has_strata = bool(rollups["by_strata"])
    strength = _sample_strength(len(normalized_rows), has_strata)

    normalized_schema = sorted({
        k for r in normalized_rows for k in r.keys()
        if k not in ("source_dataset_id", "trace_type", "provenance")
    })

    # 7. Summary
    field_quality_map = {}
    for c in columns_map:
        nf = c["normalized_field"]
        if nf and nf not in field_quality_map:
            field_quality_map[nf] = c["field_quality"]

    limitations = [
        "HuggingFace optimum-benchmark official leaderboard (Apache-2.0 "
        "library; the HF dataset card itself has NO declared license — "
        "recorded as license=None). Bounded normalised sample is NOT "
        "committed for this dataset (license_redistribution_status=unspecified).",
        f"Single hardware/backend/quantization combo per CSV: "
        f"{meta['gpu']} × {meta['backend']} × {meta['quantization']}. "
        f"Cross-config queries must explicitly select the matching CSV.",
        "All measurements are single-stream (batch_size=1, sequence_length=256, "
        "new_tokens=64). NO concurrency / queue / arrival-process signal. "
        "Treat as performance/energy priors, not as scheduler arrival trace.",
        "Latency values converted from seconds (raw) to milliseconds "
        "(normalised). Energy values kept in kWh (raw unit). The "
        "kWh→Joules conversion factor is 3.6e6 — left explicit for "
        "downstream consumers to choose.",
        "Energy is per-request kWh measured by codecarbon. CPU/RAM/GPU "
        "energies are reported separately. NOT carbon-intensity-aware — "
        "consumers must combine with regional CO2 g/kWh to derive carbon.",
        "Memory metrics (max_global_vram, max_allocated, max_reserved) are "
        "real measurements from the GPU. CPU runs report max_ram only "
        "(VRAM is None for CPU rows).",
        "TTFT here = prefill latency (time to generate the first new "
        "token). TPOT here = decode latency (per-token decode time). "
        "Both are real measurements with p50/p90/p95/p99 from "
        "report.prefill.latency.* and report.decode.latency.* respectively.",
        "Benchmark-only — Tier 4. Not pilot telemetry. Treat as cross-"
        "hardware × quantization performance / energy priors only.",
    ]

    summary = {
        "dataset_id": DATASET_ID,
        "config_name": config,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}",
        "license": LICENSE_TAG,
        "license_redistribution_status": gate_decision.license_status,
        "license_redistribution_source": LICENSE_SOURCE,
        "redistribution_gate_reason_code": gate_decision.reason_code,
        "redistribution_gate_reason_detail": gate_decision.reason_detail,
        "redistribution_gate_permitted": gate_decision.permitted,
        "redistribution_gate_operator_grant_dataset_id": (
            gate_decision.operator_grant_dataset_id
        ),
        "redistribution_gate_scope": GATE_SCOPE,
        "gated": False,
        "canonical_trace_type": "latency_benchmark_trace",
        "raw_committed": False,
        "raw_file_size_committed": False,
        "raw_columns": raw_columns,
        "raw_schema": raw_columns,
        "normalized_schema": normalized_schema,
        "available_signals": AURELIUS_AVAILABLE_SIGNALS,
        "missing_signals": AURELIUS_MISSING_SIGNALS,
        "derived_fields": [],
        "proxy_fields": [],
        "synthetic_fields": [],
        "real_fields": list(field_quality_map.keys()),
        "field_quality": field_quality_map,
        "limitations": limitations,
        "stratification_keys": strat_keys,
        "sampling_method": "full_bounded",
        "fixture_sample_rows": len(fixture_rows),
        "fixture_sample_bytes": fixture_size,
        "fixture_sample_path": str(fixture_path.relative_to(REPO_ROOT)),
        "analysis_sample_rows": len(normalized_rows),
        "analysis_sample_bytes": analysis_size,
        "analysis_sample_sha256": analysis_sha,
        "analysis_sample_path": str(analysis_path.relative_to(REPO_ROOT)),
        "committed_normalized_sample_rows": committed_norm_rows,
        "committed_normalized_sample_bytes": committed_norm_bytes,
        "committed_normalized_sample_path": committed_norm_path,
        "committed_normalized_sample_sha256": committed_norm_sha,
        "committed_normalized_sample_reason_skipped":
            committed_norm_reason_skipped,
        "committed_sample_rows": len(fixture_rows),
        "committed_sample_bytes": fixture_size,
        "sample_sha256": fixture_sha,
        "subgroup_counts": rollups["subgroup_counts"],
        "statistical_sample_strength": strength,
        "schema_profile_path": str(
            (proc_dir / "schema_profile.json").relative_to(REPO_ROOT)
        ),
        "schema_mapping_path": str(
            (proc_dir / "schema_mapping.json").relative_to(REPO_ROOT)
        ),
        "statistical_rollups_path": str(rollups_path.relative_to(REPO_ROOT)),
        "summary_path_relative": str(
            (proc_dir / "summary.json").relative_to(REPO_ROOT)
        ),
        "provenance": (
            f"{DATASET_ID}@{config}#{meta['remote']}#git={git_sha[:7]}"
        ),
        "git_sha": git_sha,
        "ingestion_timestamp_s": time.time(),
        "raw_download_manifest": {
            "url": _hf_url(DATASET_ID, meta["remote"]),
            "dest": str(raw_local_path),
            "downloaded_bytes": (
                raw_local_path.stat().st_size
                if raw_local_path.exists() else 0
            ),
        },
        "config_metadata": {
            "gpu": meta["gpu"],
            "device_class": meta["device_class"],
            "backend": meta["backend"],
            "quantization": meta["quantization"],
        },
    }
    (proc_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    decision = promotion.evaluate_promotion(summary)
    entry = promotion.build_registry_entry(summary, decision)
    return {"summary": summary, "decision": decision, "registry_entry": entry}


# ---------------------------------------------------------------------------
# Per-config ingest pipeline
# ---------------------------------------------------------------------------


def _ingest_one(
    meta: dict,
    *,
    ledger: Optional[OperatorPolicyLedger] = None,
) -> Optional[dict]:
    cfg = meta["config"]
    if ledger is None:
        ledger = _load_ledger()
    raw_dir = HF_DIR / SAFE_NAME / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_local = raw_dir / meta["remote"].replace("/", "__")

    logger.info("optimum: downloading %s → %s",
                meta["remote"], raw_local.name)
    manifest = _download(
        _hf_url(DATASET_ID, meta["remote"]),
        raw_local,
        max_bytes=PER_CSV_MAX_BYTES,
    )
    if manifest["error"]:
        logger.warning("optimum: %s — %s", meta["remote"], manifest["error"])
        return None
    if not raw_local.exists() or raw_local.stat().st_size == 0:
        logger.warning("optimum: %s — empty file", meta["remote"])
        return None

    with open(raw_local, newline="") as fh:
        reader = csv.DictReader(fh)
        raw_columns = list(reader.fieldnames or [])
        try:
            raw_rows = list(reader)
        except csv.Error as e:
            logger.warning(
                "optimum: %s — CSV parse error: %s; trying to recover",
                meta["remote"], e,
            )
            return None

    if not raw_rows:
        logger.warning("optimum: %s — empty CSV body", meta["remote"])
        return None

    git_sha = _git_sha()
    normalized_rows = [
        _normalize_row(r, meta, DATASET_ID, git_sha) for r in raw_rows
    ]

    logger.info(
        "optimum: %s → %d raw rows, %d normalized rows",
        cfg, len(raw_rows), len(normalized_rows),
    )

    gate_decision = evaluate_redistribution(ledger=ledger)

    return _finalize_config(
        meta=meta,
        raw_local_path=raw_local,
        raw_columns=raw_columns,
        raw_rows=raw_rows,
        normalized_rows=normalized_rows,
        git_sha=git_sha,
        gate_decision=gate_decision,
    )


# ---------------------------------------------------------------------------
# Audit summary
# ---------------------------------------------------------------------------


# Datasets that came up in the broadened discovery search but were
# REJECTED / deferred. Recorded so the next-run agent does not re-probe.
DISCOVERY_ONLY_RECORDS_ROUND2 = [
    {
        "dataset_id": "Exgentic/agent-llm-traces",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": "cdla-permissive-2.0",
        "gated": False,
        "kind": "defer_high_value_large_size",
        "reason": (
            "1,781 OpenTelemetry agent traces across 6 benchmarks × 5 "
            "frameworks × 6 models (Claude Opus 4.5, GPT-4.1/5.2, "
            "Gemini 3 Pro, DeepSeek-V3.2, Kimi-K2.5). Has span "
            "start_time / end_time + gen_ai.usage.input_tokens / "
            "output_tokens + status.code. Total 2.77 GB across 39 "
            "parquet files. License is cdla-permissive-2.0 (redistribution-"
            "friendly). HIGH VALUE for agent workload-shape + per-model "
            "duration priors, but the timing is closed-API end-to-end "
            "latency (API + network + serving), NOT GPU-serving "
            "telemetry. Deferred to next-run for a targeted single-"
            "parquet bounded ingest (1 of 39 files ≈ 70 MiB) once the "
            "request-shape ingester contract is generalised to handle "
            "OpenTelemetry span lists. Documented as the exact next "
            "task in docs/HF_DATASET_REGISTRY.md §10."
        ),
    },
    {
        "dataset_id": "wseaton/prefix-cache-bench",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_low_information_density",
        "reason": (
            "Single ``text`` column with 500 prompt strings. Despite the "
            "'prefix-cache-bench' name, the dataset contains NO measured "
            "cache hit / miss / latency / queue / GPU signal — it is a "
            "workload-shape fixture only. ShareGPT-style. Duplicates the "
            "existing ``sharegpt_aiperf`` request-shape role at lower "
            "density. Rejected to preserve information density."
        ),
    },
    {
        "dataset_id": "aintech/vdf_prefix-cache",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_misleading_name",
        "reason": (
            "Despite 'prefix-cache' in the name, this is a vector-DB "
            "VDF (vector-io) export — embedding vectors, not LLM prefix-"
            "cache telemetry. No latency / cache hit / model residency "
            "signal. Tier 6 — rejected."
        ),
    },
    {
        "dataset_id": "kshitijthakkar/moe-inference-benchmark",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "defer_pending_schema_inspection",
        "reason": (
            "Apache-2.0 MoE inference benchmark — small (single parquet "
            "file, n<1K rows). README returned HTTP 403 during discovery "
            "and the datasets-server info endpoint returned 404 (no "
            "auto-converted parquet view yet). Deferred until the HF "
            "auto-conversion completes OR a manual schema probe is done."
        ),
    },
    {
        "dataset_id": "kshitijthakkar/large-moe-inference-benchmark",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "kind": "defer_pending_schema_inspection",
        "reason": (
            "Companion 'large' MoE benchmark to "
            "``kshitijthakkar/moe-inference-benchmark``. No declared "
            "license; schema not yet accessible via datasets-server. "
            "Deferred to next-run paired with the small MoE benchmark."
        ),
    },
    {
        "dataset_id": "JohnGavin/llmtelemetry-metrics",
        "candidate_trace_type": "mixed_or_unknown_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_no_infrastructure_signal",
        "reason": (
            "costs.parquet + sessions.parquet with columns "
            "(cost_id, project, date, source, daily_cost_usd, "
            "n_sessions, duration_min, valid_from). This is daily "
            "billing roll-up, not infrastructure telemetry — no "
            "request-level latency, queue, GPU, cache. Despite the "
            "'llmtelemetry' name, the schema is project-level cost "
            "accounting. Rejected to preserve information density."
        ),
    },
    {
        "dataset_id": "abdallah1008/semantic-router-benchmark-data",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_classification_labels_only",
        "reason": (
            "Single JSONL with prompt + route-label pairs for training a "
            "semantic-router classifier. NO measured routing latency, "
            "throughput, model residency, or cache hit signal. The "
            "routing-quality term in the Aurelius objective needs "
            "measured-routing telemetry — this dataset is router "
            "training labels only. Rejected."
        ),
    },
    {
        "dataset_id": "Nathan-Maine/dgx-spark-kv-cache-benchmark",
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "Same KV cache benchmark CSV as ``memoriant/dgx-spark-kv-"
            "cache-benchmark`` (which is already ingested as Tier-4 "
            "config v3_corrected). Apache-2.0. The two repos appear to "
            "be near-duplicates of the same upstream Nathan-Maine work. "
            "Rejected as duplicate-content."
        ),
    },
    {
        "dataset_id": "fabric/inference-benchmarker",
        "candidate_trace_type": "request_shape_trace",
        "license_observed": "apache-2.0",
        "gated": False,
        "kind": "duplicate_existing",
        "reason": (
            "ShareGPT-derived prompt fixtures used to drive the upstream "
            "huggingface/inference-benchmarker tool — identical role to "
            "``hlarcher/inference-benchmarker`` (already rejected as "
            "duplicate) and to the existing aurelius/traces/"
            "sharegpt_aiperf.py request-shape ingester. Rejected as "
            "duplicate-shape."
        ),
    },
    {
        # Sub-config of optimum-benchmark/llm-perf-leaderboard recorded
        # as a discovery-only failure case so the next-run agent doesn't
        # re-probe.
        "dataset_id": (
            "optimum-benchmark/llm-perf-leaderboard"
            "@openvino_cpu_unquantized_32vCPU_C7i"
        ),
        "candidate_trace_type": "latency_benchmark_trace",
        "license_observed": None,
        "gated": False,
        "kind": "reject_failure_only_no_measurements",
        "reason": (
            "Every row in ``data/perf-df-openvino-cpu-unquantized-32vCPU-"
            "C7i.csv`` is an isolated-process crash "
            "(``report.traceback = 'RuntimeError: Isolated process "
            "exited with non-zero code -6'``) — ZERO measured "
            "``report.prefill.latency.*`` / ``report.decode.latency.*`` "
            "columns are present in the CSV header. Ingesting this "
            "config would write an all-null latency_benchmark_trace, "
            "which is misleading. The 9 working configs already cover "
            "the pytorch-cpu C7i baseline for cross-backend comparison. "
            "If a future openvino sub-run produces actual latency data, "
            "the config can be re-added to the matrix in scripts/"
            "ingest_hf_optimum_benchmark.py."
        ),
    },
]


def _write_audit_summary(ingested: list[dict],
                         existing_path: Optional[Path] = None,
                         *,
                         ledger: Optional[OperatorPolicyLedger] = None) -> Path:
    """Write/merge the broadened-discovery audit summary.

    If ``existing_path`` exists, we MERGE — the previous round's
    ``ingested`` + ``discovery_only_records`` entries are kept, and this
    round's entries are appended. This keeps the audit summary a
    running log of all broadened-discovery work to date.

    This is the seventh gate-consumer rewrite of this function. The
    payload's ``doc_version`` stays at ``broadened_discovery_audit_summary_v2``
    (introduced by ``scripts/ingest_hf_latency_benchmarks.py`` when it
    was wired through the gate in PR #158); this script now also
    writes ``redistribution_gate_*`` fields onto every optimum-benchmark
    ingested row and refreshes the top-level
    ``redistribution_gate_policy_default`` /
    ``redistribution_gate_policy_grant_count`` /
    ``redistribution_gate_scope`` fields the v2 schema introduced.
    Pre-existing entries from other scripts (odyn, memoriant,
    intellistream, llm_energy_consumption, h200_quantization, etc.)
    are preserved byte-for-byte.
    """

    if ledger is None:
        ledger = _load_ledger()

    out = DISC_DIR / "broadened_discovery_audit_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    prev_payload = None
    if existing_path and existing_path.exists():
        try:
            prev_payload = json.loads(existing_path.read_text())
        except Exception:
            prev_payload = None

    prev_ingested = (prev_payload or {}).get("ingested") or []
    prev_disc_only = (prev_payload or {}).get("discovery_only_records") or []
    prev_failed = (prev_payload or {}).get("failed") or []

    new_ingested = [
        {
            "dataset_id": x["summary"]["dataset_id"],
            "config_name": x["summary"]["config_name"],
            "canonical_trace_type": x["summary"]["canonical_trace_type"],
            "license": x["summary"]["license"],
            "license_redistribution_status": x["summary"][
                "license_redistribution_status"],
            "redistribution_gate_reason_code": x["summary"][
                "redistribution_gate_reason_code"],
            "redistribution_gate_permitted": x["summary"][
                "redistribution_gate_permitted"],
            "redistribution_gate_operator_grant_dataset_id": x["summary"][
                "redistribution_gate_operator_grant_dataset_id"],
            "gated": x["summary"]["gated"],
            "analysis_sample_rows": x["summary"]["analysis_sample_rows"],
            "fixture_sample_rows": x["summary"]["fixture_sample_rows"],
            "committed_normalized_sample_rows": x["summary"][
                "committed_normalized_sample_rows"],
            "committed_normalized_sample_bytes": x["summary"][
                "committed_normalized_sample_bytes"],
            "available_signals": x["summary"]["available_signals"],
            "missing_signals": x["summary"]["missing_signals"],
            "limitations": x["summary"]["limitations"],
            "statistical_sample_strength": x["summary"][
                "statistical_sample_strength"],
            "promotion_state": x["decision"]["state"],
            "promotion_tags": x["decision"]["promotion_tags"],
            "promotion_reasons": x["decision"]["reasons"],
        }
        for x in ingested
    ]

    # Merge previous + new, deduplicating on (dataset_id, config_name).
    merged_ingested_keys = {
        (e["dataset_id"], e["config_name"]) for e in new_ingested
    }
    merged_ingested = [
        e for e in prev_ingested
        if (e.get("dataset_id"), e.get("config_name")) not in merged_ingested_keys
    ] + new_ingested

    merged_disc_keys = {r["dataset_id"] for r in DISCOVERY_ONLY_RECORDS_ROUND2}
    merged_disc = [
        r for r in prev_disc_only if r.get("dataset_id") not in merged_disc_keys
    ] + DISCOVERY_ONLY_RECORDS_ROUND2

    # Preserve any top-level v2 metadata fields the upstream
    # (latency-benchmarks) writer attached. ``doc_version`` /
    # ``uses_oracle_as_headline`` / the gate-scope/default/grant-count
    # triple are owned by the v2 schema and re-emitted here from the
    # in-memory ledger so the file always reflects the current ledger
    # state at write time (not whatever the previous writer stamped).
    prev_scope = (prev_payload or {}).get("scope")
    new_scope = (
        "Round-2 broadened HF discovery — bounded ingest of "
        "``optimum-benchmark/llm-perf-leaderboard`` (10 configs "
        "across A100 / A10 / T4 / 32vCPU-C7i × pytorch / openvino × "
        "unquantized / awq / bnb / gptq / torchao) plus 9 new "
        "rejection / deferral records. Merged with round-1 audit "
        "(odyn-network / memoriant / intellistream). v2 wires this "
        "script through the canonical redistribution gate (seventh "
        "consumer)."
    )
    payload = {
        "doc_version": "broadened_discovery_audit_summary_v2",
        "scope": new_scope if not prev_scope or "v2" in (
            (prev_payload or {}).get("doc_version", "")
        ) else prev_scope,
        "production_claim": False,
        "modifies_robust_energy_engine": False,
        "modifies_controllers_or_defaults": False,
        "uses_oracle_as_headline": False,
        "git_sha": _git_sha(),
        "audited_at_s": time.time(),
        "redistribution_gate_scope": GATE_SCOPE,
        "redistribution_gate_policy_default": ledger.policy_default,
        "redistribution_gate_policy_grant_count": len(ledger.grants),
        "ingested": merged_ingested,
        "discovery_only_records": merged_disc,
        "failed": prev_failed,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


# ---------------------------------------------------------------------------
# Registry merge
# ---------------------------------------------------------------------------


def _merge_into_registry(new_entries: list[dict]) -> Path:
    registry_path = DISC_DIR / "canonical_corpus_registry.json"
    existing = promotion.load_canonical_registry(str(registry_path))
    entries = list((existing or {}).get("entries", []))

    new_keys = {(e["dataset_id"], e["config_name"]) for e in new_entries}
    entries = [
        e for e in entries
        if (e["dataset_id"], e["config_name"]) not in new_keys
    ]
    entries.extend(new_entries)
    entries.sort(key=lambda e: (e["dataset_id"], e["config_name"] or ""))
    promotion.write_canonical_registry(entries, str(registry_path))
    return registry_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", default="all",
        help="Comma-separated config names to ingest (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    if args.only == "all":
        targets = list(CONFIGS)
    else:
        wanted = {x.strip() for x in args.only.split(",")}
        targets = [c for c in CONFIGS if c["config"] in wanted]

    # Load the operator policy ledger once; thread it through every
    # ingest call and the audit summary writer so a single source of
    # truth records ``redistribution_gate_policy_default`` and
    # ``redistribution_gate_policy_grant_count``.
    ledger = _load_ledger()

    ingested: list[dict] = []
    for meta in targets:
        t0 = time.time()
        try:
            result = _ingest_one(meta, ledger=ledger)
            if result is not None:
                ingested.append(result)
        except Exception as e:
            logger.exception("optimum: %s failed: %s", meta["config"], e)
        elapsed = time.time() - t0
        logger.info("optimum: %s done in %.1fs", meta["config"], elapsed)
        if elapsed > PER_DATASET_TIMEOUT_S:
            logger.warning("optimum: %s exceeded timeout — stopping",
                           meta["config"])
            break

    logger.info("Ingested %d configs out of %d", len(ingested), len(targets))

    if args.dry_run:
        return 0

    new_entries = [x["registry_entry"] for x in ingested]
    if new_entries:
        path = _merge_into_registry(new_entries)
        logger.info("Updated registry at %s", path)

    existing_audit = DISC_DIR / "broadened_discovery_audit_summary.json"
    audit_path = _write_audit_summary(ingested, existing_audit, ledger=ledger)
    logger.info("Wrote audit summary %s", audit_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
