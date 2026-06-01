"""Feature pipeline for the Constraint-Aware Shadow Scorer Upgrade.

Pure / deterministic / stdlib-only. Loads bounded priors from:

- ``optimum-benchmark/llm-perf-leaderboard`` — per-(model_family, gpu,
  quantization, batch_size) measured TTFT/TPOT/prefill_energy/decode_energy/
  VRAM. Tier 4 benchmark — used as a **Level 3 prior** at the Aurelius
  integration layer (the benchmark cluster is NOT the production
  cluster being scored).
- ``Qinghao/AcmeTrace`` — Tier 2 cluster GPU power telemetry (IPMI Watts
  per server group). Used as a **Level 3 prior** to bound the per-GPU
  energy cost when an Optimum cell is missing.
- ``ejhusom/llm-inference-energy-consumption`` — Tier 4 per-request
  energy + TTFT/TPOT proxies. Used as an additional Level 3 cross-
  hardware prior.
- The TTFT p50 shadow prior (``ttft_shadow_prior.py``) — Level 3.
- The cache / prefix-reuse forecaster — Level 3.

Binding signal hierarchy (mission spec):

- LEVEL 1 — MEASURED. Pilot telemetry / hardware measurement / per-
  request observation / explicit operator-configurable policy. Source
  + units must be recorded; no fitted coefficient allowed.
- LEVEL 2 — DERIVED. A transparent formula on Level-1 and Level-3
  inputs. The formula must be shown; no learned utility coefficient.
- LEVEL 3 — PRIOR. Bounded benchmark / public-trace / forecasted
  values. Tagged ``value_quality = "prior"``; never silently treated
  as production truth.
- LEVEL 4 — PROHIBITED. No invented utility coefficients, weights, or
  rewards.

This module imports nothing from the production scheduler, frontier
controllers, executors, or the residency engine itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Canonical model-family / GPU-type parsing
# ---------------------------------------------------------------------------

# Maps a few common HF-style model strings to a normalized family token.
_MODEL_FAMILY_KEYWORDS = (
    ("llama", "llama"), ("qwen", "qwen"), ("mistral", "mistral"),
    ("mixtral", "mixtral"), ("gemma", "gemma"), ("phi", "phi"),
    ("pythia", "pythia"), ("gpt-neo", "gpt_neo"), ("gptj", "gptj"),
    ("gpt-j", "gptj"), ("falcon", "falcon"), ("apertus", "apertus"),
    ("claude", "claude"), ("opt", "opt"), ("starcoder", "starcoder"),
    ("codellama", "codellama"), ("vicuna", "vicuna"),
)


def derive_model_family(model_id: Optional[str]) -> Optional[str]:
    """Return a normalized model-family token (``"llama"``, ``"qwen"``, ...).

    Returns ``None`` when the family cannot be inferred.
    """
    if not model_id:
        return None
    s = str(model_id).lower()
    for needle, family in _MODEL_FAMILY_KEYWORDS:
        if needle in s:
            return family
    return None


def derive_model_size_b(model_id: Optional[str]) -> Optional[float]:
    """Return an approximate size-in-billions token from ``model_id``.

    Handles patterns like ``llama-3-70b``, ``qwen2.5-3b``,
    ``mistral-7b-v0.1``, ``mixtral-8x7b`` (8×7 = 56). Returns ``None``
    when no size token is found.
    """
    if not model_id:
        return None
    s = str(model_id).lower()
    import re
    # Mixture-of-experts pattern first (so ``8x7b`` -> 56, not 7).
    mx = re.search(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b", s)
    if mx:
        try:
            return float(mx.group(1)) * float(mx.group(2))
        except ValueError:
            return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def derive_gpu_type(location_key: Optional[str]) -> Optional[str]:
    """Pull the GPU token out of a ModelLocationState ``location_key`` or
    a raw GPU descriptor string. Returns lowercase like ``"a100"``."""
    if not location_key:
        return None
    s = str(location_key).lower()
    # Try a CARA-style ``instance_type`` substring first ("qwen2.5-3b_a30").
    parts = s.split("/")
    for p in reversed(parts):
        if "_" in p:
            tail = p.rsplit("_", 1)[-1]
            for needle in ("a100", "a10", "h100", "h200", "v100",
                           "t4", "p100", "p40", "l4", "l40"):
                if tail.startswith(needle):
                    return needle
    for needle in ("a100", "a10", "h100", "h200", "v100",
                   "t4", "p100", "p40", "l4", "l40"):
        if needle in s:
            return needle
    return None


def derive_gpu_family_from_optimum(gpu_string: Optional[str]) -> Optional[str]:
    """Map an Optimum-benchmark ``gpu`` field (e.g. ``"NVIDIA A100-SXM4-80GB"``)
    to a normalized GPU token (``"a100"``)."""
    if not gpu_string:
        return None
    s = str(gpu_string).lower()
    for needle in ("a100", "a10g", "a10", "h100", "h200", "v100",
                   "t4", "p100", "p40", "l4", "l40"):
        if needle in s:
            return needle.rstrip("g")
    return None


PROMPT_TOKEN_BINS = [(0, 50), (50, 200), (200, 800), (800, 3200),
                     (3200, 1_000_000)]


def bin_prompt_tokens(n) -> str:
    if n is None:
        return "missing"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "missing"
    for lo, hi in PROMPT_TOKEN_BINS:
        if lo <= n < hi:
            return f"[{lo},{hi})"
    return f">={PROMPT_TOKEN_BINS[-1][1]}"


# ---------------------------------------------------------------------------
# Optimum-benchmark prior loader
# ---------------------------------------------------------------------------


@dataclass
class OptimumPriorTable:
    """Per-(model_family, gpu_type, quantization) latency + energy prior.

    Fitted from fixture rows committed at
    ``tests/fixtures/hf/optimum-benchmark__llm-perf-leaderboard__*.jsonl``
    (per-config samples). Each cell is the median across the matching
    rows. Field-quality: ``real`` for the underlying TTFT/TPOT/energy
    measurements; ``derived`` for the per-cell median.
    """

    by_cell: dict = field(default_factory=dict)
    # (family, gpu) -> dict, derived from `by_cell` aggregating over
    # quantizations.
    by_family_gpu: dict = field(default_factory=dict)
    by_gpu: dict = field(default_factory=dict)
    fit_row_count: int = 0
    fixture_count: int = 0
    value_quality: str = "prior"           # Level 3 — never production truth
    source_dataset_id: str = "optimum-benchmark/llm-perf-leaderboard"
    source_units: str = "ms (TTFT/TPOT), tok/s (throughput), kWh (energy), MB (VRAM)"

    @classmethod
    def from_fixtures(
        cls, fixtures_dir: Optional[Path] = None,
    ) -> "OptimumPriorTable":
        fixtures_dir = fixtures_dir or (REPO_ROOT / "tests" / "fixtures" / "hf")
        cells: dict = {}
        fg: dict = {}
        gpu_only: dict = {}
        n_rows = 0
        n_fixtures = 0
        for path in sorted(fixtures_dir.glob(
                "optimum-benchmark__llm-perf-leaderboard__*_sample.jsonl")):
            n_fixtures += 1
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_rows += 1
                family = derive_model_family(r.get("model_family") or r.get("model"))
                gpu = derive_gpu_family_from_optimum(r.get("gpu"))
                quant = r.get("quantization_scheme") or "unquantized"
                bsz = r.get("batch_size") or 1
                if family is None or gpu is None:
                    continue
                key = (family, gpu, quant, int(bsz))
                cells.setdefault(key, []).append(r)
                fg.setdefault((family, gpu), []).append(r)
                gpu_only.setdefault(gpu, []).append(r)
        out_cells: dict = {}
        for k, rows in cells.items():
            out_cells[k] = _aggregate_optimum_rows(rows)
        out_fg: dict = {k: _aggregate_optimum_rows(v) for k, v in fg.items()}
        out_gpu: dict = {k: _aggregate_optimum_rows(v) for k, v in gpu_only.items()}
        return cls(
            by_cell=out_cells, by_family_gpu=out_fg, by_gpu=out_gpu,
            fit_row_count=n_rows, fixture_count=n_fixtures,
        )

    def lookup(
        self, *, model_family: Optional[str], gpu_type: Optional[str],
        quantization: str = "unquantized", batch_size: int = 1,
    ) -> tuple[Optional[dict], str]:
        """Return ``(prior_dict, fallback_level)``. ``prior_dict`` is
        ``None`` when no prior is available at any granularity.
        ``fallback_level`` is one of ``"cell"``, ``"family_gpu"``,
        ``"gpu_only"``, ``"missing"``."""
        if model_family is None and gpu_type is None:
            return None, "missing"
        k = (model_family, gpu_type, quantization, batch_size)
        if k in self.by_cell:
            return self.by_cell[k], "cell"
        if (model_family, gpu_type) in self.by_family_gpu:
            return self.by_family_gpu[(model_family, gpu_type)], "family_gpu"
        if gpu_type in self.by_gpu:
            return self.by_gpu[gpu_type], "gpu_only"
        return None, "missing"


def _aggregate_optimum_rows(rows: list) -> dict:
    """Median TTFT/TPOT/energy/VRAM across a list of Optimum fixture rows."""
    if not rows:
        return {}

    def _med(key):
        vals = []
        for r in rows:
            v = r.get(key)
            if isinstance(v, (int, float)) and not _is_nan(v):
                vals.append(float(v))
        if not vals:
            return None
        vals.sort()
        return vals[len(vals) // 2]

    return {
        "ttft_ms_p50": _med("p50_ttft_ms"),
        "tpot_ms_p50": _med("p50_tpot_ms"),
        "ttft_ms_mean": _med("mean_ttft_ms"),
        "tpot_ms_mean": _med("mean_tpot_ms"),
        "decode_throughput_tok_s": _med("decode_throughput_tok_s"),
        "prefill_throughput_tok_s": _med("prefill_throughput_tok_s"),
        "prefill_energy_kwh": _med("prefill_energy_total_kwh"),
        "decode_energy_kwh_per_64_tokens": _med("decode_energy_total_kwh"),
        "decode_energy_gpu_kwh_per_64_tokens": _med("decode_energy_gpu_kwh"),
        "prefill_max_vram_mb": _med("prefill_max_vram_mb"),
        "decode_max_vram_mb": _med("decode_max_vram_mb"),
        "row_count": len(rows),
        "value_quality": "prior",
        "source": "optimum-benchmark/llm-perf-leaderboard",
        "aggregation": "median_over_fixture_rows",
    }


def _is_nan(v):
    try:
        return v != v
    except Exception:
        return False


# ---------------------------------------------------------------------------
# AcmeTrace IPMI GPU power prior
# ---------------------------------------------------------------------------


@dataclass
class GPUPowerPrior:
    """Per-GPU mean / p95 power draw in watts.

    AcmeTrace's seren IPMI head sample exposes per-host-group GPU power.
    This is Tier-2 cluster telemetry (the highest trust in the federated
    HF corpus). Used to compute energy_kwh per request when an Optimum
    cell is missing.
    """

    mean_w: Optional[float] = None
    p95_w: Optional[float] = None
    p99_w: Optional[float] = None
    sample_count: int = 0
    # Tier-2 cluster telemetry — REAL at the source cluster, but at the
    # Aurelius integration layer this is still a PRIOR (the production
    # cluster being scored is not the AcmeTrace cluster).
    value_quality: str = "prior"
    source_dataset_id: str = "Qinghao/AcmeTrace/seren_ipmi_gpu_power_head"
    source_units: str = "watts per GPU"

    @classmethod
    def from_fixtures(
        cls, fixtures_dir: Optional[Path] = None,
    ) -> "GPUPowerPrior":
        fixtures_dir = fixtures_dir or (REPO_ROOT / "tests" / "fixtures" / "hf")
        path = (fixtures_dir
                / "Qinghao__AcmeTrace__seren_ipmi_gpu_power_head_sample.jsonl")
        if not path.exists():
            return cls()
        means, p95s, p99s = [], [], []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # ``value_mean`` here is total watts across ``host_count`` GPUs.
            n = r.get("host_count") or 1
            if r.get("value_mean") is not None:
                means.append(float(r["value_mean"]) / max(1, n))
            if r.get("value_p95") is not None:
                p95s.append(float(r["value_p95"]) / max(1, n))
            if r.get("value_p99") is not None:
                p99s.append(float(r["value_p99"]) / max(1, n))
        return cls(
            mean_w=(sum(means) / len(means)) if means else None,
            p95_w=(sum(p95s) / len(p95s)) if p95s else None,
            p99_w=(sum(p99s) / len(p99s)) if p99s else None,
            sample_count=len(means),
        )


# ---------------------------------------------------------------------------
# Operator pricing policy (uncalibrated unless supplied)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorPricingPolicy:
    """All $-denominated coefficients that the shadow scorer needs.

    These are NEVER invented by the scorer. The operator supplies them
    (production pilot, CFO, FinOps).  When a coefficient is ``None``,
    the term it drives is reported as ``not_computable_without_operator_policy``
    and excluded from the headline SLA-safe goodput/$ comparison.

    Calibration sources (when supplied):
    - ``gpu_hour_price_per_type`` — cloud invoice, contract, internal
      chargeback policy.
    - ``energy_price_per_kwh_usd`` — utility bill / spot price feed
      (Aurelius already ingests CAISO/PJM/ERCOT real data via
      ``aurelius.forecasting.price_model``; pass that here).
    - ``carbon_price_per_kg_usd`` — internal carbon shadow price or
      external offset price.
    """

    gpu_hour_price_per_type: dict = field(default_factory=dict)
    energy_price_per_kwh_usd: Optional[float] = None
    carbon_price_per_kg_usd: Optional[float] = None
    source: str = "operator_supplied"

    def lookup_gpu_hour_price(
        self, gpu_type: Optional[str], default: float,
    ) -> tuple[float, str]:
        """Return ``(usd_per_hour, calibration_tag)``.

        Falls back to ``default`` (the caller's global
        ``ctx.gpu_hour_price``) when the policy has no entry for the
        GPU type. The fallback tag makes it visible that the scorer
        used the operator's global default, not a per-GPU calibration.
        """
        if gpu_type is None:
            return default, "operator_global_default"
        v = self.gpu_hour_price_per_type.get(gpu_type)
        if v is None:
            return default, "operator_global_default"
        return float(v), "operator_per_gpu_policy"


# ---------------------------------------------------------------------------
# Term coverage matrix — published with the audit summary
# ---------------------------------------------------------------------------


SCORER_TERM_CATALOG = (
    "ttft_per_gpu_prior",
    "tpot_per_gpu_prior",
    "service_time_s",
    "queue_wait_s",
    "queue_depth",
    "expected_latency_s",
    "prefill_cost_avoided",
    "cache_hit_value",
    "cold_start_penalty",
    "migration_cache_loss_penalty",
    "per_gpu_hour_price",
    "vram_pressure_penalty",
    "energy_kwh_per_request",
    "energy_cost_per_request_usd",
    "carbon_cost_per_request",
    "sla_risk_probability",
    "timeout_risk",
    "incremental_gpu_cost",
)


# ---------------------------------------------------------------------------
# Per-term signal-hierarchy classification
# ---------------------------------------------------------------------------


# Maps a "classification" tag (as emitted by ``term_coverage_for_scorer``)
# to the binding mission-spec signal level.
SIGNAL_LEVEL_BY_CLASSIFICATION = {
    "missing": None,
    "heuristic": "level_4_prohibited_or_uncalibrated",
    "static_global_default": "level_1_operator_policy",
    "operator_global_default": "level_1_operator_policy",
    "operator_per_gpu_policy": "level_1_operator_policy",
    "operator_supplied_via_ModelLoadProfile": "level_1_operator_policy",
    "measured": "level_1_measured",
    "measured_or_proxy": "level_1_measured",
    "static_prior": "level_3_prior",
    "static_prior_public_order_of_magnitude": (
        "level_4_prohibited_or_uncalibrated"),
    "derived": "level_2_derived",
    "derived_binary": "level_2_derived",
    "derived_from_per_gpu_prior": "level_2_derived",
    "derived_from_optimum_benchmark": "level_3_prior",
    "derived_from_optimum_or_acmetrace": "level_3_prior",
    "derived_iff_operator_energy_price_supplied": "level_2_derived",
    "derived_iff_operator_carbon_price_supplied": "level_2_derived",
    "derived_latency_margin_indicator": "level_2_derived",
    "operator_policy_or_operator_global_default": "level_1_operator_policy",
    "forecasted_cara_ttft_p50_shadow_prior": "level_3_prior",
    "forecasted_cache_prefix_reuse_v1_proxy": "level_3_prior",
}


def signal_level_for(classification: str) -> Optional[str]:
    return SIGNAL_LEVEL_BY_CLASSIFICATION.get(classification)


def term_coverage_for_scorer(scorer_kind: str) -> dict:
    """Return whether each catalogued term is expressed by the named
    scorer. ``scorer_kind`` is one of:

    - ``"existing"`` — the production ``score_residency_candidate``,
    - ``"shadow_default_priors"`` — the upgraded shadow scorer with no
      ML forecasts (uses Optimum + AcmeTrace + static cost priors only),
    - ``"shadow_ttft_prior"`` — shadow_default_priors + CARA TTFT p50,
    - ``"shadow_cache_prior"`` — shadow_default_priors + cache reuse,
    - ``"shadow_full"`` — all priors enabled.
    """
    base_existing = {
        "ttft_per_gpu_prior": "missing",
        "tpot_per_gpu_prior": "missing",
        "service_time_s": "heuristic",
        "queue_wait_s": "measured_or_proxy",
        "queue_depth": "measured",
        "expected_latency_s": "derived",
        "prefill_cost_avoided": "missing",
        "cache_hit_value": "missing",
        "cold_start_penalty": "static_prior",
        "migration_cache_loss_penalty": "missing",
        "per_gpu_hour_price": "static_global_default",
        "vram_pressure_penalty": "derived",
        "energy_kwh_per_request": "missing",
        "energy_cost_per_request_usd": "missing",
        "carbon_cost_per_request": "missing",
        "sla_risk_probability": "derived_binary",
        "timeout_risk": "missing",
        "incremental_gpu_cost": "derived",
    }
    if scorer_kind == "existing":
        return base_existing
    upgraded = dict(base_existing)
    upgraded.update({
        "service_time_s": "derived_from_per_gpu_prior",
        "prefill_cost_avoided": "missing",  # filled by cache-prior variant
        "cache_hit_value": "missing",
        "cold_start_penalty": "operator_supplied_via_ModelLoadProfile",
        "migration_cache_loss_penalty": "derived_from_per_gpu_prior",
        # $-denominated; calibrated only if operator policy supplies it,
        # otherwise falls back to ctx.gpu_hour_price (existing scorer
        # behaviour).
        "per_gpu_hour_price": (
            "operator_policy_or_operator_global_default"),
        "vram_pressure_penalty": "derived",
        # Energy *quantity* (kWh) is derived from real benchmark
        # measurements. Energy *cost* ($) is uncalibrated unless the
        # operator supplies energy_price_per_kwh_usd.
        "energy_kwh_per_request": "derived_from_optimum_or_acmetrace",
        "energy_cost_per_request_usd": (
            "derived_iff_operator_energy_price_supplied"),
        "carbon_cost_per_request": (
            "derived_iff_operator_carbon_price_supplied"),
        "sla_risk_probability": "derived_latency_margin_indicator",
    })
    if scorer_kind == "shadow_default_priors":
        upgraded["ttft_per_gpu_prior"] = "derived_from_optimum_benchmark"
        upgraded["tpot_per_gpu_prior"] = "derived_from_optimum_benchmark"
        return upgraded
    if scorer_kind == "shadow_ttft_prior":
        upgraded["ttft_per_gpu_prior"] = "forecasted_cara_ttft_p50_shadow_prior"
        upgraded["tpot_per_gpu_prior"] = "derived_from_optimum_benchmark"
        return upgraded
    if scorer_kind == "shadow_cache_prior":
        upgraded["ttft_per_gpu_prior"] = "derived_from_optimum_benchmark"
        upgraded["tpot_per_gpu_prior"] = "derived_from_optimum_benchmark"
        upgraded["prefill_cost_avoided"] = (
            "forecasted_cache_prefix_reuse_v1_proxy")
        upgraded["cache_hit_value"] = (
            "forecasted_cache_prefix_reuse_v1_proxy")
        return upgraded
    if scorer_kind == "shadow_full":
        upgraded["ttft_per_gpu_prior"] = "forecasted_cara_ttft_p50_shadow_prior"
        upgraded["tpot_per_gpu_prior"] = "derived_from_optimum_benchmark"
        upgraded["prefill_cost_avoided"] = (
            "forecasted_cache_prefix_reuse_v1_proxy")
        upgraded["cache_hit_value"] = (
            "forecasted_cache_prefix_reuse_v1_proxy")
        return upgraded
    raise ValueError(f"unknown scorer_kind {scorer_kind!r}")


# ---------------------------------------------------------------------------
# Composite prior bundle
# ---------------------------------------------------------------------------


@dataclass
class ScorerPriors:
    """Bundle of all priors the shadow scorer can consume.

    Every field may be ``None`` — the scorer must degrade gracefully.
    Any $-denominated term that depends on a missing operator policy is
    reported as ``not_computable_without_operator_policy`` and excluded
    from the headline SLA-safe goodput/$ comparison.
    """

    optimum: Optional[OptimumPriorTable] = None
    gpu_power: Optional[GPUPowerPrior] = None
    ttft_p50_shadow: Optional[object] = None    # TTFTShadowPrior instance
    cache_reuse_predict: Optional[object] = None  # callable(features) -> float
    operator_policy: OperatorPricingPolicy = field(
        default_factory=OperatorPricingPolicy)
    field_quality: dict = field(default_factory=dict)

    @classmethod
    def load_defaults(
        cls, operator_policy: Optional[OperatorPricingPolicy] = None,
    ) -> "ScorerPriors":
        """Load all fixtures-committed priors.

        Returns a bundle where ``optimum`` and ``gpu_power`` are filled
        from the bounded fixtures shipped in the repo. The TTFT shadow
        prior and the cache-reuse predictor are NOT auto-loaded — the
        caller wires them in for the shadow-ttft / shadow-cache variants.
        ``operator_policy`` defaults to an empty policy (no invented
        $-denominated constants); the scorer reports terms it cannot
        compute as ``not_computable_without_operator_policy``.
        """
        return cls(
            optimum=OptimumPriorTable.from_fixtures(),
            gpu_power=GPUPowerPrior.from_fixtures(),
            ttft_p50_shadow=None,
            cache_reuse_predict=None,
            operator_policy=(operator_policy or OperatorPricingPolicy()),
            field_quality={
                "optimum_priors": "derived_median_of_real_benchmark_rows",
                "gpu_power_prior": "real",
                "ttft_p50_shadow": "missing_until_wired",
                "cache_reuse_predict": "missing_until_wired",
                "operator_gpu_hour_price_per_type": (
                    "operator_supplied" if operator_policy and
                    operator_policy.gpu_hour_price_per_type
                    else "not_supplied_uses_global_default"),
                "operator_energy_price_per_kwh_usd": (
                    "operator_supplied" if operator_policy and
                    operator_policy.energy_price_per_kwh_usd is not None
                    else "not_supplied_energy_terms_uncalibrated"),
                "operator_carbon_price_per_kg_usd": (
                    "operator_supplied" if operator_policy and
                    operator_policy.carbon_price_per_kg_usd is not None
                    else "not_supplied_carbon_terms_uncalibrated"),
            },
        )
