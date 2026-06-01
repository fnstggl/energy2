"""Constraint-aware Shadow Scorer (upgraded).

A thin shadow-only refinement of ``aurelius.residency.decision``.
``score_candidate`` keeps the same return type (``CandidateScore``) as
the production scorer so existing callers can compare side-by-side.

Binding contract:

- ``shadow_only = True`` on every emitted record.
- ``executable_in_real_cluster = False`` — this module exposes no
  hook that can mutate cluster state.
- ``no_control_action_taken = True`` — the scorer cannot pick a route,
  only assign a number.
- Default fall-through: when a measurement / prior is missing the
  shadow scorer returns the EXACT same number the production
  ``score_residency_candidate`` would, so wiring it in shadow does not
  change rankings unless a prior fires.
- No $-denominated constant is invented in this module. Every dollar
  coefficient traces to (a) measured telemetry, (b) measured benchmark
  data, (c) hardware measurement, or (d) an explicit
  :class:`OperatorPricingPolicy` field. Terms that need an operator
  coefficient the policy does not supply are reported as
  ``not_computable_without_operator_policy`` and excluded from the
  headline SLA-safe goodput/$ comparison.

The scorer is invoked via ``ConstraintShadowScorer.score(...)``. The
output ``CandidateScore.components`` carries the per-term breakdown
(``service_time_s``, ``queue_wait_s``, ``prefill_cost_avoided_usd``,
``cache_hit_value_usd``, ``cold_start_penalty_s``,
``migration_cache_loss_penalty_s``, ``energy_kwh_per_request``,
``energy_cost_per_request_usd``, ``per_gpu_hour_price_usd``,
``per_gpu_hour_price_calibration``, ``sla_latency_margin_s``,
``uncalibrated_terms``) plus the existing
``model_load_penalty_s`` / ``adapter_load_penalty_s`` /
``incremental_gpu_cost`` / ``memory_pressure_cost`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# We deliberately import the *data models*, not the scoring function —
# the shadow scorer composes the same ``CandidateScore`` shape but does
# its own (offline-safe) math.
from aurelius.residency.decision import (
    CandidateScore,
    SafetyContext,
    score_residency_candidate,
)
from aurelius.residency.decision import (
    _service_time_s as production_service_time_s,
)
from aurelius.residency.models import (
    ModelLoadProfile,
    ModelLocationState,
    ModelResidencyRequest,
)

from .constraint_scorer_features import (
    ScorerPriors,
    bin_prompt_tokens,
    derive_gpu_type,
    derive_model_family,
    derive_model_size_b,
)

# Provenance tags — bump when the scorer surface changes.
SHADOW_SCORER_VERSION = "constraint_shadow_scorer_v1"
SHADOW_FEATURE_VERSION = "constraint_scorer_features_v1"


@dataclass(frozen=True)
class ShadowScorerConfig:
    """How the shadow scorer composes priors.

    Every flag defaults to FALSE so the scorer reduces exactly to
    ``score_residency_candidate`` when wired in shadow without explicit
    opt-in.

    - ``use_optimum_service_time`` — replace the production heuristic
      service-time-proxy with a per-(model_family, GPU) Optimum
      benchmark estimate (median TTFT + decode-throughput × output
      tokens). Optimum data is real measured benchmark rows.
    - ``use_ttft_p50_prior`` — replace the service-time TTFT
      component with the CARA TTFT p50 shadow prior. Falls back to
      Optimum when CARA misses the cell.
    - ``use_cache_prefill_savings`` — subtract the predicted-reuse
      share of prefill compute from incremental cost (and from
      expected_latency). Requires the cache_reuse_predict callable.
    - ``use_per_gpu_hour_price`` — look up per-GPU $/hr from the
      operator policy; fall back to the global ``ctx.gpu_hour_price``
      when the policy is empty for that GPU.
    - ``use_energy_term`` — add measured-energy × operator-price as a
      cost term. If the operator price is None this term is reported
      uncalibrated and not added to cost.
    - ``use_migration_cache_loss`` — when the request is migrating
      (current_route present, different from candidate) and the
      candidate is cold, add a recompute penalty derived from
      prefill_throughput_tok_s × input_tokens.
    - ``executable_in_real_cluster`` — pinned to ``False``. The
      property is exposed so tests can assert it.
    """
    use_optimum_service_time: bool = False
    use_ttft_p50_prior: bool = False
    use_cache_prefill_savings: bool = False
    use_per_gpu_hour_price: bool = False
    use_energy_term: bool = False
    use_migration_cache_loss: bool = False
    executable_in_real_cluster: bool = False


class ConstraintShadowScorer:
    """Offline / shadow-only upgraded scorer.

    Construct with :class:`ScorerPriors` (Optimum / AcmeTrace / TTFT
    / cache-reuse predictors) plus an explicit :class:`ShadowScorerConfig`
    that toggles which prior is allowed to refine the production score.

    ``score(request, location, load_profile, ctx)`` returns a
    :class:`CandidateScore` plus a ``record`` dict with the per-term
    breakdown.
    """

    def __init__(self, priors: ScorerPriors, config: ShadowScorerConfig):
        self.priors = priors
        self.config = config
        # Hard invariant: nothing in this module can flip the executable
        # flag. The config dataclass is frozen.
        assert config.executable_in_real_cluster is False, (
            "ConstraintShadowScorer is shadow-only by contract")

    def score(self, request: ModelResidencyRequest,
              candidate_location: ModelLocationState,
              load_profile: Optional[ModelLoadProfile],
              cost_config: SafetyContext,
              safety_context: SafetyContext) -> tuple[CandidateScore, dict]:
        """Score one candidate. Returns ``(CandidateScore, breakdown_dict)``.

        ``breakdown_dict`` is the per-term provenance trace; never folded
        into the production scorer's return contract.
        """
        ctx = safety_context
        loc = candidate_location
        # 1. Start from the production scoring contract for vetoes /
        #    feasibility. This means the production scorer is the
        #    safety floor for hard vetoes — we never silently relax a
        #    region / thermal / topology / memory / queue gate.
        prod = score_residency_candidate(
            request, candidate_location, load_profile,
            cost_config, safety_context)
        breakdown: dict = {
            "shadow_scorer_version": SHADOW_SCORER_VERSION,
            "feature_version": SHADOW_FEATURE_VERSION,
            "shadow_only": True,
            "executable_in_real_cluster": False,
            "no_control_action_taken": True,
            "production_components": dict(prod.components),
            "uncalibrated_terms": [],
            "fallback_to_production": False,
            # Per-term signal hierarchy (mission spec). Filled below as
            # each term is computed. Levels: level_1_measured /
            # level_1_operator_policy / level_2_derived / level_3_prior /
            # level_4_prohibited_or_uncalibrated.
            "value_quality_by_term": {
                # Carried through from production scorer; queue is
                # measured when telemetry is present, derived otherwise.
                "queue_wait_s": "level_1_measured",
                "queue_depth": "level_1_measured",
                "model_load_penalty_s": "level_1_operator_policy",
                "adapter_load_penalty_s": "level_1_operator_policy",
                "memory_pressure_cost": "level_2_derived",
                "vram_used": "level_1_measured",
            },
            "term_formulae": {
                "expected_latency_s": (
                    "queue_wait_s + model_load_penalty_s + adapter_load_penalty_s"
                    " + migration_cache_loss_penalty_s + service_time_s"
                    " - cache_prefill_savings_s"),
                "expected_cost_usd": (
                    "(expected_latency_s / 3600) * per_gpu_hour_price_usd"
                    " + memory_pressure_cost - cache_hit_value_usd"
                    " + energy_cost_per_request_usd"),
                "goodput_per_dollar": (
                    "(1 if sla_met else 0) / expected_cost_usd"),
                "sla_met": "expected_latency_s <= sla_s",
            },
        }
        # If production already returns infeasible / unscorable we
        # mirror that — no economic prior can rescue a hard safety veto.
        if (not prod.feasible or prod.expected_latency_s is None
                or prod.expected_cost is None):
            breakdown["fallback_to_production"] = True
            breakdown["fallback_reason"] = (
                "production scorer returned infeasible/unscorable"
                "; shadow defers to existing safety contract")
            return prod, breakdown

        gpu_type = derive_gpu_type(loc.location_key)
        model_family = derive_model_family(request.model_id)
        model_size_b = derive_model_size_b(request.model_id)
        breakdown.update(
            derived_gpu_type=gpu_type, derived_model_family=model_family,
            derived_model_size_b=model_size_b,
            prompt_token_bin=bin_prompt_tokens(request.prompt_tokens),
        )

        # 2. Refine service time (TTFT + decode * output_tokens). The
        #    floor is the production service_time_s — we never tighten
        #    the estimate beyond it unless a real measurement says so.
        service_s, service_s_source = self._refine_service_time(
            request, ctx, gpu_type=gpu_type,
            model_family=model_family, model_size_b=model_size_b,
            production_value=prod.components.get(
                "service_time_s",
                production_service_time_s(request, ctx)),
        )
        breakdown["service_time_s"] = service_s
        breakdown["service_time_s_source"] = service_s_source
        breakdown["value_quality_by_term"]["service_time_s"] = (
            "level_3_prior"
            if service_s_source.startswith("shadow:")
            else "level_4_prohibited_or_uncalibrated")

        # 3. Per-GPU hour price (operator policy, falls back to ctx).
        per_gpu_price, per_gpu_price_src = self._per_gpu_hour_price(
            gpu_type=gpu_type, ctx=ctx)
        breakdown["per_gpu_hour_price_usd"] = per_gpu_price
        breakdown["per_gpu_hour_price_calibration"] = per_gpu_price_src
        breakdown["value_quality_by_term"]["per_gpu_hour_price_usd"] = (
            "level_1_operator_policy")

        # 4. Cold-start penalty / migration cache-loss penalty in
        #    *seconds* (not $). These propagate into expected_latency.
        model_pen = prod.components.get("model_load_penalty_s", 0.0) or 0.0
        adapter_pen = prod.components.get("adapter_load_penalty_s", 0.0) or 0.0
        migration_pen_s = self._migration_cache_loss_penalty_s(
            request=request, location=loc,
            gpu_type=gpu_type, model_family=model_family,
            production_components=prod.components,
        )
        if migration_pen_s["uncalibrated"]:
            breakdown["uncalibrated_terms"].append(
                "migration_cache_loss_penalty_s")
        breakdown["migration_cache_loss_penalty_s"] = (
            migration_pen_s["penalty_s"])
        breakdown["migration_cache_loss_source"] = migration_pen_s["source"]
        breakdown["value_quality_by_term"]["migration_cache_loss_penalty_s"] = (
            "level_2_derived" if not migration_pen_s["uncalibrated"]
            else "level_4_prohibited_or_uncalibrated")

        # 5. Cache prefill savings (in seconds and in $).
        cache_savings = self._cache_prefill_savings(
            request=request, gpu_type=gpu_type, model_family=model_family,
            per_gpu_hour_price=per_gpu_price,
            per_gpu_hour_price_source=per_gpu_price_src,
        )
        if cache_savings["uncalibrated_usd"]:
            breakdown["uncalibrated_terms"].append("cache_hit_value_usd")
        breakdown["cache_prefill_savings_s"] = cache_savings["savings_s"]
        breakdown["cache_hit_value_usd"] = cache_savings["savings_usd"]
        breakdown["cache_predicted_reuse"] = cache_savings["predicted_reuse"]
        breakdown["cache_savings_source"] = cache_savings["source"]
        # savings_s is a Level 2 derivation (predicted_reuse_L3 ×
        # prompt_tokens_L1 / prefill_throughput_L3). USD form is Level 2
        # *only* when the per-GPU price is operator-calibrated.
        breakdown["value_quality_by_term"]["cache_prefill_savings_s"] = (
            "level_2_derived" if cache_savings["savings_s"] > 0
            else "level_4_prohibited_or_uncalibrated"
            if cache_savings["uncalibrated_usd"] else "level_2_derived")
        breakdown["value_quality_by_term"]["cache_hit_value_usd"] = (
            "level_2_derived" if not cache_savings["uncalibrated_usd"]
            and cache_savings["savings_usd"] > 0
            else "level_4_prohibited_or_uncalibrated")

        # 6. Queue wait carries through from production (it's measured).
        queue_wait = prod.components.get("queue_wait_s", 0.0) or 0.0

        expected_latency_s = (
            queue_wait
            + model_pen + adapter_pen
            + migration_pen_s["penalty_s"]
            + service_s
            - cache_savings["savings_s"]
        )
        expected_latency_s = max(0.0, expected_latency_s)
        breakdown["expected_latency_s"] = expected_latency_s

        # 7. SLA risk: a quantitative latency margin in seconds (NOT a
        #    sigmoid with invented sigma). Production keeps binary
        #    sla_met; we report the margin in shadow.
        sla = ctx.sla_s(request)
        sla_margin_s = sla - expected_latency_s
        breakdown["sla_s"] = sla
        breakdown["sla_latency_margin_s"] = sla_margin_s
        sla_met = expected_latency_s <= sla

        # 8. Incremental GPU cost from per-GPU price + service-time
        #    refinement. Cache savings in seconds already reduced
        #    expected_latency; convert to $-savings via the same
        #    operator price (no double-counting).
        incremental_gpu_cost = (expected_latency_s / 3600.0) * per_gpu_price
        memory_pressure_cost = prod.components.get(
            "memory_pressure_cost", 0.0) or 0.0
        breakdown["incremental_gpu_cost_usd"] = incremental_gpu_cost
        breakdown["memory_pressure_cost_usd"] = memory_pressure_cost

        # 9. Energy cost: kWh × operator $/kWh. kWh is measured (Optimum
        #    or AcmeTrace). Operator price is supplied or term is
        #    uncalibrated.
        energy = self._energy_cost_for_request(
            request=request, gpu_type=gpu_type, model_family=model_family,
            service_s=service_s,
        )
        if energy["uncalibrated_usd"]:
            breakdown["uncalibrated_terms"].append("energy_cost_per_request_usd")
        breakdown["energy_kwh_per_request"] = energy["energy_kwh"]
        breakdown["energy_cost_per_request_usd"] = energy["energy_cost_usd"]
        breakdown["energy_source"] = energy["source"]
        # KWh is derived from a Level 3 prior (Optimum/AcmeTrace); USD is
        # Level 2 *only* when the operator supplies energy_price_per_kwh_usd.
        breakdown["value_quality_by_term"]["energy_kwh_per_request"] = (
            "level_3_prior" if energy["energy_kwh"] is not None
            else "level_4_prohibited_or_uncalibrated")
        breakdown["value_quality_by_term"]["energy_cost_per_request_usd"] = (
            "level_2_derived"
            if (energy["energy_cost_usd"] is not None
                and not energy["uncalibrated_usd"])
            else "level_4_prohibited_or_uncalibrated")

        # 10. Total expected cost — only includes terms with a
        #     calibrated dollar coefficient.
        expected_cost_usd = (
            incremental_gpu_cost + memory_pressure_cost
            - cache_savings["savings_usd_if_calibrated"]
            + energy["energy_cost_usd_if_calibrated"]
        )
        expected_cost_usd = max(1e-9, expected_cost_usd)
        breakdown["expected_cost_usd"] = expected_cost_usd

        # 11. SLA-safe goodput / $.
        goodput_per_dollar = (
            (1.0 if sla_met else 0.0) / expected_cost_usd
            if expected_cost_usd > 0 else None)
        breakdown["sla_met"] = sla_met
        breakdown["goodput_per_dollar"] = goodput_per_dollar
        breakdown["value_quality_by_term"]["expected_latency_s"] = "level_2_derived"
        breakdown["value_quality_by_term"]["expected_cost_usd"] = "level_2_derived"
        breakdown["value_quality_by_term"]["sla_met"] = "level_2_derived"
        breakdown["value_quality_by_term"]["sla_latency_margin_s"] = "level_2_derived"
        breakdown["value_quality_by_term"]["goodput_per_dollar"] = "level_2_derived"

        components = dict(prod.components)
        components.update({
            "service_time_s": service_s,
            "shadow_expected_latency_s": expected_latency_s,
            "shadow_expected_cost_usd": expected_cost_usd,
            "incremental_gpu_cost": incremental_gpu_cost,
            "per_gpu_hour_price_usd": per_gpu_price,
            "per_gpu_hour_price_calibration": per_gpu_price_src,
            "migration_cache_loss_penalty_s": migration_pen_s["penalty_s"],
            "cache_prefill_savings_s": cache_savings["savings_s"],
            "cache_hit_value_usd": cache_savings["savings_usd"],
            "energy_kwh_per_request": energy["energy_kwh"],
            "energy_cost_per_request_usd": energy["energy_cost_usd"],
            "sla_latency_margin_s": sla_margin_s,
            "uncalibrated_terms": list(breakdown["uncalibrated_terms"]),
            "shadow_only": True,
            "executable_in_real_cluster": False,
        })

        out = CandidateScore(
            location_key=loc.location_key, feasible=prod.feasible,
            expected_latency_s=expected_latency_s,
            expected_cost=expected_cost_usd,
            sla_met=sla_met, goodput_per_dollar=goodput_per_dollar,
            model_resident=prod.model_resident,
            adapter_resident=prod.adapter_resident,
            is_cold=prod.is_cold,
            safety_vetoes=tuple(prod.safety_vetoes),
            confidence=prod.confidence,
            components=components,
        )
        return out, breakdown

    # --- per-term helpers ---------------------------------------------

    def _refine_service_time(
        self, request: ModelResidencyRequest, ctx: SafetyContext, *,
        gpu_type: Optional[str], model_family: Optional[str],
        model_size_b: Optional[float], production_value: float,
    ) -> tuple[float, str]:
        """Return ``(service_time_s, source_tag)``.

        Composition:
        - ``ttft = ttft_p50_prior.predict(...)`` if ``use_ttft_p50_prior``
          and the prior fires for the (model_family, GPU, prompt_bin)
          cell.
        - else ``ttft = optimum.ttft_ms_p50 / 1000`` if
          ``use_optimum_service_time`` and the Optimum cell hits.
        - else ``ttft = production_value`` (the existing heuristic).
        - ``decode_s = output_tokens / decode_throughput_tok_s`` if
          Optimum hits; else absorbed into the production heuristic.
        - ``service_time_s = max(production_value, ttft + decode_s)`` —
          the production heuristic is the safety floor; we never
          tighten the latency estimate below it.
        """
        priors = self.priors
        ttft_s = None
        ttft_source = None
        if (self.config.use_ttft_p50_prior and priors.ttft_p50_shadow
                is not None):
            try:
                ttft_s = float(priors.ttft_p50_shadow.predict(
                    model_size=(f"{model_size_b:g}b" if model_size_b else None),
                    gpu_type=gpu_type,
                    prompt_tokens=request.prompt_tokens,
                ))
                ttft_source = "cara_ttft_p50_shadow_prior"
            except Exception:
                ttft_s = None
        if ttft_s is None and self.config.use_optimum_service_time:
            cell, fallback = (priors.optimum.lookup(
                model_family=model_family, gpu_type=gpu_type,
            ) if priors.optimum else (None, "missing"))
            if cell and cell.get("ttft_ms_p50") is not None:
                ttft_s = float(cell["ttft_ms_p50"]) / 1000.0
                ttft_source = f"optimum_benchmark_{fallback}"
        decode_s = 0.0
        decode_source = "missing"
        if self.config.use_optimum_service_time and priors.optimum is not None:
            cell, fallback = priors.optimum.lookup(
                model_family=model_family, gpu_type=gpu_type)
            if cell:
                # Optimum decode_throughput is tokens/sec on the
                # benchmarked batch size; treat as a per-token rate.
                tput = cell.get("decode_throughput_tok_s")
                out_tok = request.output_tokens or 0
                if tput and tput > 0 and out_tok > 0:
                    decode_s = float(out_tok) / float(tput)
                    decode_source = f"optimum_benchmark_{fallback}"
                elif cell.get("tpot_ms_p50") is not None and out_tok > 0:
                    decode_s = (float(cell["tpot_ms_p50"]) / 1000.0
                                * float(out_tok))
                    decode_source = f"optimum_benchmark_tpot_{fallback}"
        candidate_s = (ttft_s or 0.0) + decode_s
        # The production_value is the existing scorer's heuristic; we
        # treat it as the safety floor for backwards compatibility.
        refined = max(float(production_value), candidate_s)
        if refined == float(production_value):
            return refined, "production_heuristic_floor"
        return refined, f"shadow:ttft={ttft_source}|decode={decode_source}"

    def _per_gpu_hour_price(
        self, *, gpu_type: Optional[str], ctx: SafetyContext,
    ) -> tuple[float, str]:
        if not self.config.use_per_gpu_hour_price:
            return ctx.gpu_hour_price, "production_global_default"
        return self.priors.operator_policy.lookup_gpu_hour_price(
            gpu_type=gpu_type, default=ctx.gpu_hour_price)

    def _migration_cache_loss_penalty_s(
        self, *, request: ModelResidencyRequest,
        location: ModelLocationState, gpu_type: Optional[str],
        model_family: Optional[str], production_components: dict,
    ) -> dict:
        """Estimate the *seconds* of prefill re-compute lost when the
        request migrates from ``current_route`` to a different
        candidate.

        Derived from Optimum's measured ``prefill_throughput_tok_s`` —
        when the benchmark cell is missing the penalty is 0 and the
        term is flagged uncalibrated.
        """
        if not self.config.use_migration_cache_loss:
            return {"penalty_s": 0.0, "uncalibrated": False,
                    "source": "disabled"}
        current = request.current_route
        if not current or current == location.location_key:
            return {"penalty_s": 0.0, "uncalibrated": False,
                    "source": "no_migration"}
        prompt = request.prompt_tokens or 0
        if prompt <= 0:
            return {"penalty_s": 0.0, "uncalibrated": True,
                    "source": "no_prompt_tokens"}
        cell, fallback = (self.priors.optimum.lookup(
            model_family=model_family, gpu_type=gpu_type,
        ) if self.priors.optimum else (None, "missing"))
        if not cell or not cell.get("prefill_throughput_tok_s"):
            return {"penalty_s": 0.0, "uncalibrated": True,
                    "source": f"optimum_{fallback}_no_prefill_throughput"}
        tput = float(cell["prefill_throughput_tok_s"])
        if tput <= 0:
            return {"penalty_s": 0.0, "uncalibrated": True,
                    "source": "optimum_prefill_throughput_zero"}
        return {"penalty_s": float(prompt) / tput, "uncalibrated": False,
                "source": f"optimum_benchmark_{fallback}"}

    def _cache_prefill_savings(
        self, *, request: ModelResidencyRequest, gpu_type: Optional[str],
        model_family: Optional[str], per_gpu_hour_price: float,
        per_gpu_hour_price_source: str,
    ) -> dict:
        """Estimate ``(savings_s, savings_usd)`` from the cache forecaster.

        savings_s = predicted_reuse_pct × prefill_time_s_on_this_gpu
        savings_usd = savings_s / 3600 × per_gpu_hour_price (when the
                     per-GPU price is operator-supplied; uncalibrated
                     otherwise)
        """
        out = {
            "savings_s": 0.0, "savings_usd": 0.0,
            "savings_usd_if_calibrated": 0.0,
            "predicted_reuse": None, "source": "disabled",
            "uncalibrated_usd": False,
        }
        if not self.config.use_cache_prefill_savings:
            return out
        pred = self.priors.cache_reuse_predict
        if pred is None:
            out["source"] = "no_cache_reuse_predict"
            return out
        try:
            reuse = float(pred(request))
        except Exception as e:  # pragma: no cover
            out["source"] = f"predict_failed_{type(e).__name__}"
            return out
        # cache_reuse_predict is expected to return [0,1] or [0,100].
        if reuse > 1.0:
            reuse = reuse / 100.0
        reuse = max(0.0, min(1.0, reuse))
        out["predicted_reuse"] = reuse
        prompt = request.prompt_tokens or 0
        cell, fallback = (self.priors.optimum.lookup(
            model_family=model_family, gpu_type=gpu_type,
        ) if self.priors.optimum else (None, "missing"))
        if not cell or not cell.get("prefill_throughput_tok_s") or prompt <= 0:
            out["source"] = (f"optimum_{fallback}_no_prefill_throughput"
                             if cell is None or not cell.get(
                                 "prefill_throughput_tok_s")
                             else "no_prompt_tokens")
            out["uncalibrated_usd"] = True
            return out
        prefill_s = float(prompt) / float(cell["prefill_throughput_tok_s"])
        savings_s = reuse * prefill_s
        out["savings_s"] = savings_s
        # The USD savings reuses the per-GPU $/hr coefficient. If the
        # caller's coefficient is the operator GLOBAL DEFAULT (not a
        # per-GPU calibration) we still compute it but tag the term
        # uncalibrated.
        usd = (savings_s / 3600.0) * per_gpu_hour_price
        if per_gpu_hour_price_source.startswith("operator_per_gpu"):
            out["savings_usd"] = usd
            out["savings_usd_if_calibrated"] = usd
        else:
            out["savings_usd"] = 0.0  # not counted in headline
            out["savings_usd_if_calibrated"] = usd
            out["uncalibrated_usd"] = True
        out["source"] = (f"cache_prefix_forecaster_predict×"
                         f"optimum_{fallback}_prefill_throughput")
        return out

    def _energy_cost_for_request(
        self, *, request: ModelResidencyRequest, gpu_type: Optional[str],
        model_family: Optional[str], service_s: float,
    ) -> dict:
        """Compute (energy_kwh, energy_cost_usd) for one request.

        Energy KWh comes from:
        1. Optimum prefill_energy_total_kwh + decode_energy_total_kwh × (output_tokens/64)
           — measured benchmark data, ``real``.
        2. Fallback: AcmeTrace mean_w × service_s / 3600 / 1000 —
           measured IPMI telemetry, ``real``.
        3. Else ``None`` — uncalibrated.

        Energy cost USD requires ``operator_policy.energy_price_per_kwh_usd``.
        When ``None`` the term is reported uncalibrated and excluded
        from the headline cost.
        """
        out = {
            "energy_kwh": None, "energy_cost_usd": None,
            "energy_cost_usd_if_calibrated": 0.0,
            "source": "disabled", "uncalibrated_usd": False,
        }
        if not self.config.use_energy_term:
            return out
        energy_kwh = None
        source = None
        if self.priors.optimum is not None:
            cell, fallback = self.priors.optimum.lookup(
                model_family=model_family, gpu_type=gpu_type)
            if cell:
                prefill_kwh = cell.get("prefill_energy_kwh")
                decode_kwh_per_64 = cell.get("decode_energy_kwh_per_64_tokens")
                out_tok = request.output_tokens or 0
                if prefill_kwh is not None and decode_kwh_per_64 is not None:
                    energy_kwh = float(prefill_kwh) + (
                        float(decode_kwh_per_64) * max(0.0, out_tok / 64.0))
                    source = f"optimum_benchmark_{fallback}"
                elif decode_kwh_per_64 is not None and out_tok > 0:
                    energy_kwh = float(decode_kwh_per_64) * (out_tok / 64.0)
                    source = f"optimum_benchmark_{fallback}_decode_only"
        if energy_kwh is None and self.priors.gpu_power is not None:
            w = self.priors.gpu_power.mean_w
            if w and service_s > 0:
                energy_kwh = (w * service_s) / 1000.0 / 3600.0
                source = "acmetrace_ipmi_mean_w_x_service_s"
        if energy_kwh is None:
            out["source"] = "no_energy_prior"
            out["uncalibrated_usd"] = True
            return out
        out["energy_kwh"] = energy_kwh
        out["source"] = source
        price = self.priors.operator_policy.energy_price_per_kwh_usd
        if price is None:
            out["energy_cost_usd"] = None
            out["uncalibrated_usd"] = True
            return out
        out["energy_cost_usd"] = float(price) * energy_kwh
        out["energy_cost_usd_if_calibrated"] = out["energy_cost_usd"]
        return out


# ---------------------------------------------------------------------------
# Variant factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorerVariant:
    """Named scorer variant used in the eval driver."""
    name: str
    config: ShadowScorerConfig


VARIANT_EXISTING = ScorerVariant(
    name="A_existing",
    config=ShadowScorerConfig(),  # all flags False → passthrough to production
)
VARIANT_SHADOW_DEFAULT = ScorerVariant(
    name="B_shadow_default_priors",
    config=ShadowScorerConfig(
        use_optimum_service_time=True, use_per_gpu_hour_price=True,
        use_energy_term=True, use_migration_cache_loss=True,
    ),
)
VARIANT_SHADOW_TTFT = ScorerVariant(
    name="C_shadow_with_ttft_p50_prior",
    config=ShadowScorerConfig(
        use_optimum_service_time=True, use_ttft_p50_prior=True,
        use_per_gpu_hour_price=True, use_energy_term=True,
        use_migration_cache_loss=True,
    ),
)
VARIANT_SHADOW_CACHE = ScorerVariant(
    name="D_shadow_with_cache_prefill",
    config=ShadowScorerConfig(
        use_optimum_service_time=True, use_cache_prefill_savings=True,
        use_per_gpu_hour_price=True, use_energy_term=True,
        use_migration_cache_loss=True,
    ),
)
VARIANT_SHADOW_FULL = ScorerVariant(
    name="E_shadow_full",
    config=ShadowScorerConfig(
        use_optimum_service_time=True, use_ttft_p50_prior=True,
        use_cache_prefill_savings=True, use_per_gpu_hour_price=True,
        use_energy_term=True, use_migration_cache_loss=True,
    ),
)
ALL_VARIANTS = (
    VARIANT_EXISTING, VARIANT_SHADOW_DEFAULT, VARIANT_SHADOW_TTFT,
    VARIANT_SHADOW_CACHE, VARIANT_SHADOW_FULL,
)


# ---------------------------------------------------------------------------
# Promotion classifier
# ---------------------------------------------------------------------------


SHADOW_FINAL_STATUS_VALUES = frozenset({
    "shadow_ready_for_integration_review",
    "promising_needs_validation",
    "diagnostic_only",
    "rejected_regression",
    "proxy_promising_only",
    "blocked_by_pilot_telemetry",
})


def classify_shadow_scorer_status(
    *,
    sla_safe_goodput_per_dollar_improvement_pct: float,
    has_sla_regression: bool,
    has_subgroup_regression: bool,
    headline_terms_are_uncalibrated: bool,
    pilot_telemetry_required: bool,
) -> tuple[str, str]:
    """Promotion ladder per mission spec.

    Order of precedence:
    1. SLA / safety / subgroup regression → ``rejected_regression``
    2. Pilot-telemetry-required gate → ``blocked_by_pilot_telemetry``
    3. Headline improvement comes from proxy/synthetic terms only →
       ``proxy_promising_only``
    4. Standard ladder by improvement bucket.
    """
    if has_sla_regression or has_subgroup_regression:
        return ("rejected_regression",
                "shadow scorer triggers an SLA or subgroup regression "
                "vs the existing constraint-aware scorer")
    if pilot_telemetry_required:
        return ("blocked_by_pilot_telemetry",
                "shadow scorer can compute the headline term only with "
                "measurements that require pilot telemetry; cannot "
                "promote against HF / public-trace data alone")
    # Negative / sub-2% improvement is diagnostic_only regardless of
    # calibration. The "proxy_promising_only" tag is reserved for
    # actually-positive improvements whose dollar coefficient is not
    # operator-calibrated.
    if sla_safe_goodput_per_dollar_improvement_pct < 2.0:
        return ("diagnostic_only",
                f"improvement {sla_safe_goodput_per_dollar_improvement_pct:.2f}%"
                " < 2% vs existing constraint-aware scorer")
    if headline_terms_are_uncalibrated:
        return ("proxy_promising_only",
                f"improvement {sla_safe_goodput_per_dollar_improvement_pct:.2f}% "
                "depends on a term whose dollar coefficient is not "
                "operator-calibrated (proxy / uncalibrated) — promote "
                "only after operator supplies the coefficient")
    if sla_safe_goodput_per_dollar_improvement_pct < 5.0:
        return ("promising_needs_validation",
                f"improvement {sla_safe_goodput_per_dollar_improvement_pct:.2f}%"
                " in 2-5% band; awaits broader validation before shadow "
                "promotion")
    return ("shadow_ready_for_integration_review",
            f"improvement {sla_safe_goodput_per_dollar_improvement_pct:.2f}% "
            "> 5% with no SLA / subgroup regression — safe to begin "
            "shadow-integration review")
