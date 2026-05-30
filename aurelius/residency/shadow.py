"""Recommendation-only shadow logging for model residency / cold-start.

Implements the ``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §5 shadow posture:
every residency decision is a **logged recommendation**, never an action. This
module performs **no cluster mutation**: it never loads/evicts a real model,
never reroutes traffic, and never calls a Kubernetes (or any) write API. There
is no actuation code path here at all.

Binding invariants (asserted in :class:`ResidencyShadowLog` and the tests):

- ``ResidencyShadowDecision.executed`` is ``False`` by default and the logger
  refuses to record a decision with ``executed=True``.
- A decision NEVER substitutes the requested model/adapter
  (``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §4.4 no-substitution rule): the
  recommendation acts on *where/when* a model is warmed, never on *which* model
  is served. ``ResidencyShadowDecision`` has no "substitute_model" field by
  construction.
- When telemetry is insufficient the recommendation is
  ``insufficient_telemetry`` — it does not guess.

No production-savings claim is implied; expected_* fields are directional
estimates, not measured outcomes.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

RECOMMENDED_ACTIONS = (
    "prewarm", "preserve_affinity", "no_op", "evict_candidate",
    "insufficient_telemetry",
)

# This module is observation/recommendation-only. Flipped to True nowhere.
MUTATION_ALLOWED = False


class ResidencyMutationError(RuntimeError):
    """Raised if anything attempts to mark a shadow decision as executed —
    shadow mode performs no cluster mutation."""


def _now_utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@dataclass
class ResidencyShadowDecision:
    """One recommendation-only residency decision (logged, never executed)."""

    timestamp: float
    workload_id: str
    model_id: str
    current_location: Optional[str]
    recommended_action: str
    reason: str
    confidence: str = "unknown"
    request_id: Optional[str] = None
    adapter_id: Optional[str] = None
    expected_cold_start_saved_s: Optional[float] = None
    expected_warm_pool_cost: Optional[float] = None
    expected_sla_risk_delta: Optional[float] = None
    executed: bool = False  # INVARIANT: recommendation-only

    def __post_init__(self):
        if self.recommended_action not in RECOMMENDED_ACTIONS:
            raise ValueError(
                f"unknown recommended_action {self.recommended_action!r}; "
                f"expected one of {RECOMMENDED_ACTIONS}")
        if self.executed:
            raise ResidencyMutationError(
                "shadow decisions are recommendation-only; executed must be False")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "ResidencyShadowDecision":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})

    @classmethod
    def from_json(cls, line: str) -> "ResidencyShadowDecision":
        return cls.from_dict(json.loads(line))


class ResidencyShadowLog:
    """Append-only JSONL log of recommendation-only residency decisions.

    Refuses any decision flagged ``executed`` — there is no mutation path.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.decisions: list = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def record(self, decision: ResidencyShadowDecision) -> ResidencyShadowDecision:
        if decision.executed or MUTATION_ALLOWED:
            raise ResidencyMutationError(
                "shadow log is recommendation-only; refusing executed decision")
        self.decisions.append(decision)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(decision.to_json() + "\n")
        return decision

    def summary(self) -> dict:
        counts = {a: 0 for a in RECOMMENDED_ACTIONS}
        for d in self.decisions:
            counts[d.recommended_action] = counts.get(d.recommended_action, 0) + 1
        saved = [d.expected_cold_start_saved_s for d in self.decisions
                 if d.expected_cold_start_saved_s is not None]
        return {
            "n_decisions": len(self.decisions),
            "action_counts": counts,
            "total_expected_cold_start_saved_s": round(sum(saved), 4) if saved else None,
            "all_recommendation_only": all(not d.executed for d in self.decisions),
        }


@dataclass
class ShadowRecommenderConfig:
    """Transparent thresholds for the recommendation rules. These are *not*
    optimizer constants and feed no KPI — they only gate which advisory label is
    logged. Conservative defaults grounded in the GenTD26 cold-start magnitudes
    (``docs/MODEL_RESIDENCY_COLD_START_SPEC.md`` §0)."""

    min_cold_start_s_to_prewarm: float = 5.0
    slo_s: Optional[float] = None
    gpu_hour_cost: Optional[float] = None
    # GPU-hours a prewarm would hold a replica resident for the lookahead window;
    # used only to estimate expected_warm_pool_cost when a cost basis is given.
    prewarm_hold_hours: float = 1.0


class ShadowRecommender:
    """Maps a residency observation to a recommendation-only decision.

    This is NOT optimizer behavior: it produces an advisory label + directional
    estimate for the shadow log. It never mutates anything and never substitutes
    a model. When residency is unknown it returns ``insufficient_telemetry``.
    """

    def __init__(self, config: Optional[ShadowRecommenderConfig] = None):
        self.config = config or ShadowRecommenderConfig()

    def recommend(self, obs, *, linkage_quality: Optional[str] = None,
                  popularity_decayed: Optional[bool] = None) -> ResidencyShadowDecision:
        cfg = self.config
        location = obs.container_id or obs.gpu_id or obs.node_id
        cold = obs.is_cold_start  # Optional[bool]

        expected_cost = None
        if cfg.gpu_hour_cost is not None:
            expected_cost = round(cfg.gpu_hour_cost * cfg.prewarm_hold_hours, 4)

        # 1. Insufficient telemetry — do not guess (overrides everything).
        if cold is None:
            return ResidencyShadowDecision(
                timestamp=obs.timestamp, request_id=obs.request_id,
                workload_id=obs.workload_id or obs.model_id, model_id=obs.model_id,
                adapter_id=obs.adapter_id, current_location=location,
                recommended_action="insufficient_telemetry",
                reason="model_loaded_before_request unknown; residency not "
                       "classifiable for this request",
                confidence="unknown")

        # 2. Cold start observed → prewarm candidate when the avoided latency is
        #    material. expected_cold_start_saved_s = the latency this load cost.
        if cold is True:
            saved = obs.total_load_latency_s
            if saved is not None and saved >= cfg.min_cold_start_s_to_prewarm:
                sla_delta = None
                if cfg.slo_s is not None and obs.e2e_latency_s is not None:
                    # directional: prewarming removes the load from e2e
                    warm_e2e = obs.e2e_latency_s - saved
                    was_violation = obs.e2e_latency_s > cfg.slo_s
                    would_meet = warm_e2e <= cfg.slo_s
                    sla_delta = -1.0 if (was_violation and would_meet) else 0.0
                return ResidencyShadowDecision(
                    timestamp=obs.timestamp, request_id=obs.request_id,
                    workload_id=obs.workload_id or obs.model_id,
                    model_id=obs.model_id, adapter_id=obs.adapter_id,
                    current_location=location, recommended_action="prewarm",
                    reason=f"cold start paid ~{round(saved, 2)}s load latency "
                           f">= {cfg.min_cold_start_s_to_prewarm}s threshold",
                    expected_cold_start_saved_s=round(saved, 4),
                    expected_warm_pool_cost=expected_cost,
                    expected_sla_risk_delta=sla_delta,
                    confidence=_blend_confidence(obs.confidence, linkage_quality))
            # cold but cheap / unmeasured → no_op (don't recommend churn)
            return ResidencyShadowDecision(
                timestamp=obs.timestamp, request_id=obs.request_id,
                workload_id=obs.workload_id or obs.model_id, model_id=obs.model_id,
                adapter_id=obs.adapter_id, current_location=location,
                recommended_action="no_op",
                reason="cold start below prewarm threshold or load latency "
                       "unmeasured; no prewarm recommended",
                confidence=_blend_confidence(obs.confidence, linkage_quality))

        # 3. Warm hit. If popularity has decayed, flag an evict candidate
        #    (subject to cooldown elsewhere); otherwise preserve affinity.
        if popularity_decayed is True:
            return ResidencyShadowDecision(
                timestamp=obs.timestamp, request_id=obs.request_id,
                workload_id=obs.workload_id or obs.model_id, model_id=obs.model_id,
                adapter_id=obs.adapter_id, current_location=location,
                recommended_action="evict_candidate",
                reason="warm residency hit but popularity decayed below warm-pool "
                       "value; eviction candidate (advisory; respect cooldown)",
                expected_warm_pool_cost=expected_cost,
                confidence=_blend_confidence(obs.confidence, linkage_quality))
        return ResidencyShadowDecision(
            timestamp=obs.timestamp, request_id=obs.request_id,
            workload_id=obs.workload_id or obs.model_id, model_id=obs.model_id,
            adapter_id=obs.adapter_id, current_location=location,
            recommended_action="preserve_affinity",
            reason="warm residency hit; keep request on its already-warm replica "
                   "(avoid a cold reroute)",
            expected_cold_start_saved_s=None,
            confidence=_blend_confidence(obs.confidence, linkage_quality))


def _blend_confidence(obs_confidence: str, linkage_quality: Optional[str]) -> str:
    """Cap recommendation confidence by linkage attribution (contract §4)."""
    if linkage_quality in (None, "no_join", "time_join"):
        # unattributed telemetry → never claim better than low
        return "low" if obs_confidence in ("high", "medium") else obs_confidence
    return obs_confidence


def recommend_all(observations, *, config: Optional[ShadowRecommenderConfig] = None,
                  linkage_quality: Optional[str] = None,
                  log: Optional[ResidencyShadowLog] = None) -> list:
    """Run the recommender over a batch, logging each decision (rec-only)."""
    rec = ShadowRecommender(config)
    out = []
    for obs in observations:
        decision = rec.recommend(obs, linkage_quality=linkage_quality)
        if log is not None:
            log.record(decision)
        out.append(decision)
    return out
