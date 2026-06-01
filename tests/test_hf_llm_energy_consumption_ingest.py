"""Tests for the Round-4 HF discovery ingest of
``ejhusom/llm-inference-energy-consumption``.

Audit-only: tests read committed artefacts; they do NOT hit the HF API.

Covered:

* four new ``latency_benchmark_trace`` configs (cross-hardware-tier ×
  workload × model-size matrix from the SINTEF Digital
  arxiv:2407.16893 study)
* corpus invariants: no raw / analysis sample committed; license =
  cc-by-sa-4.0; fixture is tiny + deterministic; committed normalised
  sample is bounded per config
* Round-4 discovery-only / negative-result records
* mandatory caveats pinned in ``limitations`` so the Aurelius
  constraint-aware engine does not treat Ollama ``prompt_duration`` as a
  measured first-token-wall-clock TTFT, or aggregate across (model,
  hardware_tier) cells without matching keys
* the cross-hardware-tier mission: at least one laptop tier AND at
  least one workstation tier are present so the placement engine has
  cross-tier evidence for the first time
* energy signals: at least one config exposes a populated GPU energy
  column (``has_gpu_energy_signal``)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.traces.hf_corpus import promotion  # noqa: E402

HF_DIR = REPO_ROOT / "data" / "external" / "hf"
DISC_DIR = REPO_ROOT / "data" / "external" / "hf_discovery"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "hf"

DATASET_ID = "ejhusom/llm-inference-energy-consumption"
SAFE_NAME = "ejhusom__llm-inference-energy-consumption"

CONFIGS = [
    "alpaca_gemma_7b_laptop2",
    "alpaca_gemma_7b_workstation",
    "codefeedback_codellama_7b_workstation",
    "codefeedback_codellama_70b_workstation",
]

AUDIT_PATH = (
    DISC_DIR / "round4_broadened_discovery_audit_summary.json"
)

ROUND4_DISCOVERY_IDS = [
    "ohdoking/energy_consumption_by_model_and_gpu",
    "adityaupasani/llm-inference-energy-consumption",
    "Nayan10767/llm-inference-energy-consumption",
    "vgyhj/llm-inference-energy-consumption",
    "nishant-k/speculative-decoding-benchmark-results",
    "inference-optimization/speculators_benchmarks_tool_call",
    "kshitijthakkar/large-moe-inference-benchmark",
]


def _proc_dir(config: str) -> Path:
    return HF_DIR / SAFE_NAME / config / "processed"


def _summary(config: str) -> dict:
    with open(_proc_dir(config) / "summary.json") as fh:
        return json.load(fh)


def _audit() -> dict:
    with open(AUDIT_PATH) as fh:
        return json.load(fh)


# ───────────────────────── 1. No raw / analysis files committed ─────────


def test_no_raw_files_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    raw_committed = [p for p in out if "/raw/" in p]
    assert raw_committed == [], (
        f"Raw downloads committed for {SAFE_NAME}: {raw_committed}"
    )


def test_no_analysis_sample_tracked_by_git():
    out = subprocess.check_output(
        ["git", "ls-files", f"data/external/hf/{SAFE_NAME}"],
        cwd=REPO_ROOT,
    ).decode().splitlines()
    analysis_committed = [
        p for p in out if p.endswith("/analysis_sample.jsonl")
    ]
    assert analysis_committed == [], (
        f"analysis_sample.jsonl committed for {SAFE_NAME}: "
        f"{analysis_committed}"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_fixture_is_committed_and_tiny(config):
    f = FIXTURES_DIR / f"{SAFE_NAME}__{config}_sample.jsonl"
    assert f.exists(), f"missing fixture: {f}"
    sz = f.stat().st_size
    assert 0 < sz <= 16 * 1024, (
        f"fixture {f} size {sz}B outside (0, 16 KiB]"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_committed_normalized_sample_under_100kib(config):
    s = _summary(config)
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    assert path.exists(), f"missing committed sample {path}"
    sz = path.stat().st_size
    assert 0 < sz <= 100 * 1024 * 1024, (
        f"committed sample {path} > 100 MB cap"
    )
    # Per-config local bound (script enforces 100 KiB)
    assert sz <= 110 * 1024, (
        f"committed sample {path} unexpectedly large ({sz}B); "
        "tighten COMMITTED_NORMALIZED_MAX_BYTES_PER_CONFIG if intentional."
    )


def test_total_committed_normalised_under_pr_budget():
    """All four configs combined must stay well under the 300 MB PR cap."""
    total = 0
    for c in CONFIGS:
        s = _summary(c)
        path = REPO_ROOT / s["committed_normalized_sample_path"]
        total += path.stat().st_size
    assert total <= 600 * 1024, (
        f"sum of committed normalised samples {total}B > 600 KiB; "
        "this dataset's commit budget is very generous already"
    )
    assert total <= 300 * 1024 * 1024, (
        f"sum of committed normalised samples {total}B > 300 MB PR cap"
    )


# ───────────────────────── 2. Processed artefacts present ───────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_processed_artifacts_present(config):
    for name in (
        "schema_profile.json",
        "schema_mapping.json",
        "statistical_rollups.json",
        "summary.json",
        "committed_normalized_sample.jsonl",
    ):
        path = _proc_dir(config) / name
        assert path.exists(), f"missing {path}"


# ───────────────────────── 3. Summary invariants ────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_license_and_gating(config):
    s = _summary(config)
    assert s["dataset_id"] == DATASET_ID
    assert s["license"] == "cc-by-sa-4.0"
    assert s["gated"] is False


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_canonical_trace_type(config):
    s = _summary(config)
    assert s["canonical_trace_type"] == "latency_benchmark_trace"


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_engine_is_ollama(config):
    s = _summary(config)
    assert s["engine"] == "ollama"


def test_at_least_one_laptop_tier_config():
    tiers = {_summary(c)["hardware_tier"] for c in CONFIGS}
    assert "laptop" in tiers, (
        "Round-4 mission requires at least one laptop-tier config so the "
        "Aurelius placement engine has its first cross-tier evidence."
    )


def test_at_least_one_workstation_tier_config():
    tiers = {_summary(c)["hardware_tier"] for c in CONFIGS}
    assert "workstation" in tiers


def test_both_workloads_present():
    """alpaca (instruction-following) + codefeedback (code) must both
    appear so the workload-shape × energy cross-evaluation is feasible."""
    workloads = {_summary(c)["prompt_dataset"] for c in CONFIGS}
    assert "alpaca" in workloads and "codefeedback" in workloads


def test_cross_model_size_present():
    sizes = {_summary(c)["model_size_b"] for c in CONFIGS}
    # 7B + 70B both present so codellama:7b vs codellama:70b comparison
    # works on identical workstation hardware.
    assert 7.0 in sizes
    assert 70.0 in sizes


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_has_energy_signal_label(config):
    s = _summary(config)
    assert isinstance(s.get("has_gpu_energy_signal"), bool)
    # All 4 of our chosen configs have GPU energy; if a future config has
    # has_gpu_energy_signal=False the limitations must explain it.
    if not s["has_gpu_energy_signal"]:
        lim_text = " ".join(s["limitations"]).lower()
        assert "cpu-only" in lim_text or "cpu only" in lim_text


def test_at_least_one_config_has_gpu_energy():
    """The mission is to give the Aurelius energy term cross-hardware
    evidence. At least one config must expose GPU energy."""
    assert any(_summary(c)["has_gpu_energy_signal"] for c in CONFIGS)


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_provenance_includes_dataset_and_git(config):
    s = _summary(config)
    prov = s.get("provenance")
    assert prov
    assert s["dataset_id"] in prov
    assert "git=" in prov


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_fixture_and_analysis_split(config):
    s = _summary(config)
    assert isinstance(s.get("fixture_sample_rows"), int)
    assert s["fixture_sample_rows"] >= 1
    assert isinstance(s.get("analysis_sample_rows"), int)
    assert s["analysis_sample_rows"] >= s["fixture_sample_rows"]


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_committed_sample_hash_present(config):
    s = _summary(config)
    sha = s.get("committed_normalized_sample_sha256")
    assert isinstance(sha, str) and len(sha) == 64


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_committed_sample_hash_matches_file(config):
    s = _summary(config)
    path = REPO_ROOT / s["committed_normalized_sample_path"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    assert h.hexdigest() == s["committed_normalized_sample_sha256"]


@pytest.mark.parametrize("config", CONFIGS)
def test_summary_records_redistribution_attribution(config):
    s = _summary(config)
    lrs = s.get("license_redistribution_status", "").lower()
    assert "cc-by-sa-4.0" in lrs
    assert "attribution" in lrs
    assert "share-alike" in lrs
    # arxiv citation must be preserved
    assert "2407.16893" in lrs


# ───────────────────────── 4. Limitations pinned ────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_ttft_proxy_caveat_pinned(config):
    """Aurelius must NOT treat ttft_proxy_ms as a measured first-token
    wall-clock TTFT — Ollama's prompt_duration is only an APPROXIMATION
    of TTFT under single-stream serving."""
    s = _summary(config)
    lim_text = " ".join(s["limitations"]).lower()
    assert "proxy" in lim_text
    assert "prompt_duration" in lim_text or "prompt duration" in lim_text


@pytest.mark.parametrize("config", CONFIGS)
def test_concurrency_one_caveat_pinned(config):
    """Concurrency = 1 means NO queue / contention / batching signal —
    the queue-risk and batch-frontier modules MUST NOT consume this."""
    s = _summary(config)
    lim_text = " ".join(s["limitations"]).lower()
    assert "concurrency = 1" in lim_text or "concurrency=1" in lim_text
    assert "queue" in lim_text


@pytest.mark.parametrize("config", CONFIGS)
def test_tier4_caveat_pinned(config):
    s = _summary(config)
    lim_text = " ".join(s["limitations"]).lower()
    assert "tier 4" in lim_text
    assert "pilot telemetry" in lim_text


@pytest.mark.parametrize("config", CONFIGS)
def test_ollama_engine_caveat_pinned(config):
    s = _summary(config)
    lim_text = " ".join(s["limitations"]).lower()
    assert "ollama" in lim_text
    # Forbid generalisation to vLLM / SGLang / etc.
    assert "vllm" in lim_text or "sglang" in lim_text or "tgi" in lim_text


# ───────────────────────── 5. Promotion gates ───────────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_promotion_gate_log_clean(config):
    s = _summary(config)
    gates = promotion.gates(s)
    failed = [g for g in gates if not g["passed"]]
    assert failed == [], f"failed gates for {config}: {failed}"


@pytest.mark.parametrize("config", CONFIGS)
def test_promotion_state_eligible(config):
    s = _summary(config)
    decision = promotion.evaluate_promotion(s)
    assert decision["state"] in {
        "promoted_for_performance_priors",
        "promoted_for_training_priors",
        "promoted_for_schema_only",
    }, f"{config} unexpectedly {decision['state']}"


def test_strong_config_promoted_for_performance_priors():
    """The two ≥5K-row configs (laptop2 gemma:7b + workstation gemma:7b)
    must be promoted_for_performance_priors at strong strength."""
    for cfg in ("alpaca_gemma_7b_laptop2", "alpaca_gemma_7b_workstation"):
        s = _summary(cfg)
        assert s["statistical_sample_strength"] == "strong"
        decision = promotion.evaluate_promotion(s)
        assert (
            decision["state"] == "promoted_for_performance_priors"
        ), (cfg, decision)
        assert (
            "promoted_for_performance_priors" in decision["promotion_tags"]
        )
        assert (
            "promoted_for_constraint_aware_evaluation"
            in decision["promotion_tags"]
        )


def test_weak_config_downgraded_to_training_priors_only():
    """The 70B workstation config has only 161 rows → weak strength.
    Must NOT be promoted for performance_priors or
    constraint_aware_evaluation (those need moderate+ strength)."""
    s = _summary("codefeedback_codellama_70b_workstation")
    assert s["statistical_sample_strength"] == "weak"
    decision = promotion.evaluate_promotion(s)
    assert (
        "promoted_for_performance_priors" not in decision["promotion_tags"]
    ), (
        "70B 161-row config must NOT claim performance_priors evidence; "
        "weak strength is insufficient"
    )
    assert (
        "promoted_for_constraint_aware_evaluation"
        not in decision["promotion_tags"]
    )
    assert "promoted_for_training_priors" in decision["promotion_tags"]


# ───────────────────────── 6. Schema profile sanity ─────────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_profile_no_unknown_columns(config):
    with open(_proc_dir(config) / "schema_profile.json") as fh:
        sp = json.load(fh)
    assert sp["unknown_columns"] == [], (
        f"schema profile for {config} has unknown columns: "
        f"{sp['unknown_columns']}"
    )


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_profile_accepts_energy_columns(config):
    with open(_proc_dir(config) / "schema_profile.json") as fh:
        sp = json.load(fh)
    for col in (
        "energy_consumption_llm_cpu",
        "energy_consumption_llm_gpu",
        "energy_consumption_llm_total",
        "energy_consumption_monitoring",
    ):
        assert col in sp["accepted_columns"], (
            f"schema_profile for {config} did not accept {col}"
        )


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_profile_accepts_timing_columns(config):
    with open(_proc_dir(config) / "schema_profile.json") as fh:
        sp = json.load(fh)
    for col in (
        "total_duration",
        "load_duration",
        "prompt_duration",
        "response_duration",
        "prompt_token_length",
        "response_token_length",
    ):
        assert col in sp["accepted_columns"]


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_profile_rejects_text_and_linguistic_columns(config):
    """Verbose text + 60-odd linguistic columns must be dropped to keep
    committed sample small AND focus on Aurelius signals."""
    with open(_proc_dir(config) / "schema_profile.json") as fh:
        sp = json.load(fh)
    rejected_set = set(sp["rejected_columns"])
    for col in (
        "prompt",
        "response",
        "word_count",
        "sentence_count",
        "sentiment_polarity",
        "flesch_reading_ease",
    ):
        assert col in rejected_set, (
            f"schema_profile for {config} should reject {col}"
        )


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_mapping_energy_columns_have_aurelius_category(config):
    with open(_proc_dir(config) / "schema_mapping.json") as fh:
        sm = json.load(fh)
    cols = {c["raw_column_name"]: c for c in sm["columns"]}
    for col in (
        "energy_consumption_llm_cpu",
        "energy_consumption_llm_gpu",
        "energy_consumption_llm_total",
    ):
        assert cols[col]["aurelius_signal_category"] == "cost_energy_carbon"
        assert cols[col]["field_quality"] == "real"


@pytest.mark.parametrize("config", CONFIGS)
def test_schema_mapping_ttft_proxy_marked_derived(config):
    """The ttft_proxy_ms field is derived from prompt_duration — must
    not be marked as 'real' to prevent confusion with measured TTFT."""
    s = _summary(config)
    fq = s["field_quality"]
    assert fq["ttft_proxy_ms"] == "derived", (
        f"{config}: ttft_proxy_ms field_quality must be 'derived' "
        f"not '{fq.get('ttft_proxy_ms')}' (prevents Aurelius engine "
        "from treating proxy as measured TTFT)"
    )
    assert fq["tpot_proxy_ms_per_token"] == "derived"
    assert fq["e2e_latency_ms"] == "derived"


# ───────────────────────── 7. Statistical rollups sanity ────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_statistical_rollups_have_energy_and_latency(config):
    with open(_proc_dir(config) / "statistical_rollups.json") as fh:
        r = json.load(fh)
    o = r["overall"]
    # Latency
    assert "e2e_latency_ms" in o
    for k in ("p50", "p95", "p99", "mean", "count"):
        assert k in o["e2e_latency_ms"]
    # Energy
    assert "energy_kwh_llm_total" in o
    assert "energy_kwh_llm_cpu" in o
    # tokens
    assert "prompt_token_length" in o
    assert "response_token_length" in o


def test_workstation_codellama_70b_e2e_latency_dominates_7b():
    """70B codellama must be much slower than 7B codellama on the
    same workstation under the same codefeedback workload."""
    with open(_proc_dir("codefeedback_codellama_7b_workstation")
              / "statistical_rollups.json") as fh:
        r7 = json.load(fh)
    with open(_proc_dir("codefeedback_codellama_70b_workstation")
              / "statistical_rollups.json") as fh:
        r70 = json.load(fh)
    p50_7 = r7["overall"]["e2e_latency_ms"]["p50"]
    p50_70 = r70["overall"]["e2e_latency_ms"]["p50"]
    assert p50_70 > p50_7 * 5, (
        f"70B p50 latency ({p50_70} ms) must be >= 5× 7B p50 "
        f"({p50_7} ms) on the same workstation — sanity check on "
        "the scale ordering"
    )


# ───────────────────────── 8. Round-4 audit + candidates ────────────────


def test_audit_summary_present():
    assert AUDIT_PATH.exists(), f"missing audit {AUDIT_PATH}"
    a = _audit()
    assert a["production_claim"] is False
    assert a["modifies_robust_energy_engine"] is False
    assert a["modifies_controllers_or_defaults"] is False


def test_audit_includes_all_ingested_configs():
    a = _audit()
    ingested_cfg_keys = {
        (i["dataset_id"], i["config_name"]) for i in a["ingested"]
    }
    for c in CONFIGS:
        assert (DATASET_ID, c) in ingested_cfg_keys


def test_audit_includes_negative_results():
    a = _audit()
    audit_ids = {r["dataset_id"] for r in a["discovery_only_records"]}
    for did in ROUND4_DISCOVERY_IDS:
        assert did in audit_ids, f"missing discovery record for {did}"


def test_candidate_registry_updated():
    cands_path = DISC_DIR / "hf_dataset_candidates.json"
    with open(cands_path) as fh:
        d = json.load(fh)
    cands = {c["dataset_id"]: c for c in d["candidates"]}
    assert DATASET_ID in cands
    c = cands[DATASET_ID]
    assert c["license"] == "cc-by-sa-4.0"
    assert c["gated_status"] == "public"
    assert c["recommended_action"] == "ingest_now_bounded"
    assert c["candidate_trace_type"] == "latency_benchmark_trace"
    assert set(c["configs_ingested"]) == set(CONFIGS)


def test_candidate_registry_includes_negative_results():
    cands_path = DISC_DIR / "hf_dataset_candidates.json"
    with open(cands_path) as fh:
        d = json.load(fh)
    cand_ids = {c["dataset_id"] for c in d["candidates"]}
    for did in ROUND4_DISCOVERY_IDS:
        assert did in cand_ids, (
            f"candidate registry missing Round-4 record for {did}"
        )


def test_canonical_registry_has_all_configs():
    reg_path = DISC_DIR / "canonical_corpus_registry.json"
    with open(reg_path) as fh:
        d = json.load(fh)
    keys = {(e["dataset_id"], e.get("config_name")) for e in d["entries"]}
    for c in CONFIGS:
        assert (DATASET_ID, c) in keys, (
            f"canonical registry missing {DATASET_ID}/{c}"
        )


def test_canonical_registry_promotion_states_recorded():
    reg_path = DISC_DIR / "canonical_corpus_registry.json"
    with open(reg_path) as fh:
        d = json.load(fh)
    for e in d["entries"]:
        if e["dataset_id"] != DATASET_ID:
            continue
        assert e["trust_tier"] == "tier_4_latency_benchmark_traces"
        assert e["license"] == "cc-by-sa-4.0"
        assert e["promotion_state"] in {
            "promoted_for_performance_priors",
            "promoted_for_training_priors",
            "promoted_for_schema_only",
        }


# ───────────────────────── 9. No secrets / tokens leaked ────────────────


@pytest.mark.parametrize("config", CONFIGS)
def test_no_hf_token_in_committed_artifacts(config):
    """Defensive: ensure HF_TOKEN never leaks into any committed file
    (summary / schema / fixture / committed sample)."""
    files = list(_proc_dir(config).glob("*.json"))
    files.append(
        _proc_dir(config) / "committed_normalized_sample.jsonl"
    )
    files.append(
        FIXTURES_DIR / f"{SAFE_NAME}__{config}_sample.jsonl"
    )
    for f in files:
        text = f.read_text()
        assert "hf_" not in text.lower() or all(
            tok not in text for tok in (
                "hf_hGyzo", "Bearer hf_", "HF_TOKEN=hf_",
            )
        ), f"Possible HF token leak in {f}"
