"""Simulator-only execution of residency decisions.

A :class:`ResidencyDecision` from ``decision.py`` is a *recommendation*. This
module applies it to **simulated** :class:`ModelLocationState` for backtests —
and **only** in simulator mode. In real/customer mode every call is a no-op that
returns ``mutated=False``; there is no code path here that writes to a real
cluster, router, Kubernetes API, or serving engine.

Binding invariant (asserted by tests):
``apply_residency_decision(..., mode=REAL_MODE)`` NEVER mutates state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import ModelLoadProfile, ModelLocationState, ResidencyAction, ResidencyDecision

SIMULATOR_MODE = "simulator"
REAL_MODE = "real"


class RealModeMutationError(RuntimeError):
    """Raised if real/customer mode is asked to perform a mutating action."""


@dataclass
class DecisionEffect:
    """What applying a decision did (or, in real mode, would have recommended)."""

    action: str
    mode: str
    placed_at: Optional[str] = None
    paid_cold_start_s: float = 0.0
    paid_adapter_load_s: float = 0.0
    mutated: bool = False
    evicted_model_id: Optional[str] = None
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action, "mode": self.mode, "placed_at": self.placed_at,
            "paid_cold_start_s": round(self.paid_cold_start_s, 4),
            "paid_adapter_load_s": round(self.paid_adapter_load_s, 4),
            "mutated": self.mutated, "evicted_model_id": self.evicted_model_id,
            "notes": list(self.notes),
        }


def _pick_evict_victim(loc: ModelLocationState, protect: Optional[str]) -> Optional[str]:
    """Choose a resident model to evict (least-recently-added proxy: first in
    the list), never the model we are trying to admit."""
    for m in list(loc.loaded_model_ids):
        if m != protect:
            return m
    return None


def apply_residency_decision(decision: ResidencyDecision,
                             locations_by_key: dict,
                             *,
                             mode: str,
                             request=None,
                             load_profile: Optional[ModelLoadProfile] = None,
                             memory_required_gb: Optional[float] = None) -> DecisionEffect:
    """Apply ``decision`` to ``locations_by_key`` (key → ModelLocationState).

    In :data:`REAL_MODE` this is a strict no-op (``mutated=False``) — residency
    actions are recommendation-only. In :data:`SIMULATOR_MODE` it mutates the
    simulated target location (load/evict/place) and returns the effect.
    """
    if mode not in (SIMULATOR_MODE, REAL_MODE):
        raise ValueError(f"unknown mode {mode!r}")

    eff = DecisionEffect(action=decision.action, mode=mode)

    # ---- REAL MODE: never mutate ----
    if mode == REAL_MODE:
        if decision.executable_in_real_cluster:  # defensive; engine never sets true
            raise RealModeMutationError(
                "residency decisions are recommendation-only in real mode")
        eff.mutated = False
        eff.placed_at = decision.target_location or decision.current_location
        eff.notes.append("real mode: recommendation logged, no cluster mutation")
        return eff

    # ---- SIMULATOR MODE: mutate simulated state ----
    target = locations_by_key.get(decision.target_location)
    model_id = getattr(request, "model_id", None)
    adapter_id = getattr(request, "adapter_id", None)
    mem_gb = memory_required_gb
    if mem_gb is None and load_profile is not None:
        mem_gb = load_profile.memory_required_gb

    if decision.action in (ResidencyAction.REJECT_UNSAFE_ROUTE,
                           ResidencyAction.INSUFFICIENT_TELEMETRY):
        eff.notes.append("no placement (unsafe / insufficient telemetry)")
        return eff

    if target is None:
        eff.notes.append("target location not found in simulated cluster")
        return eff

    def _load_model():
        if model_id and not target.has_model(model_id):
            target.loaded_model_ids.append(model_id)
            if mem_gb and target.gpu_memory_used is not None:
                target.gpu_memory_used += mem_gb * 1e9
            p = (load_profile.model_load_penalty_s(
                safety_critical=False) if load_profile else None)
            eff.paid_cold_start_s += (p or 0.0)
            eff.mutated = True

    def _load_adapter():
        if adapter_id and not target.has_adapter(adapter_id):
            target.loaded_adapter_ids.append(adapter_id)
            p = (load_profile.adapter_load_penalty_s(
                safety_critical=False) if load_profile else None)
            eff.paid_adapter_load_s += (p or 0.0)
            eff.mutated = True

    if decision.action == ResidencyAction.EVICT_CANDIDATE:
        victim = _pick_evict_victim(target, protect=model_id)
        if victim is not None:
            target.loaded_model_ids.remove(victim)
            if mem_gb and target.gpu_memory_used is not None:
                target.gpu_memory_used = max(0.0, target.gpu_memory_used - mem_gb * 1e9)
            eff.evicted_model_id = victim
            eff.mutated = True
        # after eviction, admit the requested model
        _load_model()
        _load_adapter()
        eff.placed_at = target.location_key
        return eff

    if decision.action in (ResidencyAction.PREWARM_MODEL,):
        _load_model()
        eff.placed_at = target.location_key
        return eff

    if decision.action in (ResidencyAction.PREWARM_ADAPTER,):
        _load_adapter()
        eff.placed_at = target.location_key
        return eff

    # ROUTE_TO_RESIDENT_MODEL / PRESERVE_AFFINITY / KEEP_CURRENT_ROUTE:
    # place the request; load on miss (a cold route still pays the load).
    _load_model()
    _load_adapter()
    eff.placed_at = target.location_key
    return eff
