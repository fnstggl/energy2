"""Frontier estimator — predict KPI + safety metrics for candidate rho targets.

For v1 this is **deterministic replay** over a telemetry window (a sequence
of aggregated arrival ticks), not ML. The estimator wraps the unchanged
``aurelius/traces/backtest.py`` serving physics — the same harness used by
the Azure 2024 audit. No optimizer constant is tuned and no constant is
modified to force a result.

Two replay modes are supported:

- ``reactive`` — sla_aware-style: provision for the previous tick at rho R.
- ``anticipatory`` — constraint_aware-style: EWMA-anticipatory plan at rho R.
  This is the safer dominant frontier per the Azure 2024 audit.

The estimator may also accept a pre-computed list of point-dicts (from the
Azure 2024 frontier audit JSON, say) via ``estimate_frontier_from_points`` —
useful for tests + offline replays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .models import FrontierPoint, SafetyStatus, WorkloadFrontierProfile
from .safety import SafetyConfig, classify_point_safety

ANTICIPATORY = "anticipatory"
REACTIVE = "reactive"
_MODES = (ANTICIPATORY, REACTIVE)


@dataclass
class FrontierEstimatorConfig:
    """Estimator settings (no optimizer constants).

    ``mode`` selects reactive vs anticipatory replay. ``tick_seconds`` and
    ``prefill_savings`` mirror the canonical backtest defaults; both are
    transparent and adjustable per call.
    """

    mode: str = ANTICIPATORY
    tick_seconds: float = 60.0
    prefill_savings: float = 0.0

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(
                f"unknown estimator mode {self.mode!r}; expected one of {_MODES}")


# Lazy import — the estimator is usable from fixture-input callers without
# pulling the full backtest physics into module-load.
def _bt():  # pragma: no cover - import indirection
    from aurelius.traces import backtest as bt
    return bt


# ---------------------------------------------------------------------------
# Replay sizers (mirror scripts/run_azure_2024_safe_utilization_frontier.py)
# ---------------------------------------------------------------------------

class _Reactive:
    """Provision for the PREVIOUS tick at target rho R (sla_aware-style)."""

    def __init__(self, R: float):
        self.R = R
        self.prev = None

    def size(self, t):
        bt = _bt()
        src = self.prev if self.prev is not None else t
        r = bt._size_for_target(src.arrival_rate_rps,
                                max(1.0, src.output_tokens_mean),
                                bt._tick_throughput_tokps(src), self.R)
        self.prev = t
        return r


class _Anticipatory:
    """EWMA-anticipatory plan at rho R (constraint_aware-style)."""

    def __init__(self, R: float, *, tick_hours: float):
        self.R = R
        self.tick_hours = tick_hours
        self.ewma_r = 0.0
        self.ewma_o = 0.0
        self.prev_replicas = None

    def size(self, t):
        bt = _bt()
        a = 0.5
        if t.request_count > 0:
            self.ewma_r = (a * t.arrival_rate_rps + (1 - a) * self.ewma_r
                           if self.ewma_r else t.arrival_rate_rps)
            self.ewma_o = (a * t.output_tokens_mean + (1 - a) * self.ewma_o
                           if self.ewma_o else t.output_tokens_mean)
        plan_rate = max(t.arrival_rate_rps, self.ewma_r)
        plan_out = (max(t.output_tokens_mean, self.ewma_o) if t.request_count
                    else self.ewma_o)
        base = bt._size_for_target(plan_rate, max(1.0, plan_out),
                                   bt._tick_throughput_tokps(t), self.R)
        r = bt._constraint_trim(t, base, 0.0, self.tick_hours, self.prev_replicas)
        self.prev_replicas = r
        return r


def _evaluate_for_rho(ticks, R: float, *, mode: str, tick_seconds: float,
                      prefill_savings: float) -> dict:
    bt = _bt()
    tick_hours = tick_seconds / 3600.0
    sizer = (_Anticipatory(R, tick_hours=tick_hours) if mode == ANTICIPATORY
             else _Reactive(R))
    evals = []
    prev_r = None
    for t in ticks:
        r = sizer.size(t)
        ev = bt.evaluate_tick(t, r, prefill_savings=prefill_savings,
                              tick_hours=tick_hours)
        if prev_r is not None and ev.replicas != prev_r:
            ev.scale_event = True
        prev_r = ev.replicas
        evals.append(ev)
    res = bt._aggregate(f"{mode}@{R}", evals, cache_aware=False, ticks=ticks)
    active = [(e, t) for e, t in zip(evals, ticks) if t.request_count > 0]
    aw = sum(t.request_count for _, t in active) or 1
    mean_rho = sum(e.rho * t.request_count for e, t in active) / aw
    timeout_w = sum(e.timeout_rate_pct * t.request_count for e, t in active) / aw
    reps = [e.replicas for e in evals]
    churn = sum(abs(reps[i] - reps[i - 1]) for i in range(1, len(reps)))
    scale_events = sum(1 for e in evals if e.scale_event)
    return {
        "rho_target": R,
        "predicted_goodput_per_dollar": float(
            res.kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "predicted_sla_safe_goodput": float(res.kpi.sla_compliant_goodput),
        "predicted_gpu_hours": float(res.kpi.active_gpu_hours),
        "predicted_timeout_pct": float(timeout_w),
        "predicted_queue_p95_ms": float(res.queue_p95_ms),
        "predicted_queue_p99_ms": float(res.queue_p99_ms),
        "predicted_latency_p95_ms": float(getattr(res, "latency_p95_ms", 0.0))
            or None,
        "predicted_latency_p99_ms": float(getattr(res, "latency_p99_ms", 0.0))
            or None,
        "predicted_scale_events": int(scale_events),
        "predicted_churn_score": float(churn),
        "predicted_mean_utilization": float(mean_rho),
    }


# ---------------------------------------------------------------------------
# Public estimator API
# ---------------------------------------------------------------------------

def estimate_frontier(profile: WorkloadFrontierProfile,
                      telemetry_window,
                      *,
                      candidate_rhos: Optional[Iterable[float]] = None,
                      predictor_config: Optional[FrontierEstimatorConfig] = None,
                      safety_config: Optional[SafetyConfig] = None
                      ) -> list[FrontierPoint]:
    """Estimate the frontier for ``profile`` over ``telemetry_window``.

    ``telemetry_window`` is a sequence of aggregated arrival ticks (as used
    by ``aurelius/traces/backtest.py``). When the window is empty the
    estimator returns one INSUFFICIENT_TELEMETRY point per candidate rho —
    it does not invent data.

    ``candidate_rhos`` defaults to ``profile.clamp_candidates()``; any rho
    outside the profile's ``[min_rho, max_rho]`` is silently dropped.
    """
    cfg = predictor_config or FrontierEstimatorConfig()
    safety = safety_config or SafetyConfig()
    rhos = (tuple(candidate_rhos) if candidate_rhos is not None
            else profile.clamp_candidates())
    rhos = tuple(r for r in rhos if profile.min_rho <= r <= profile.max_rho)

    ticks = list(telemetry_window) if telemetry_window is not None else []

    points: list[FrontierPoint] = []
    if not ticks:
        for R in rhos:
            points.append(FrontierPoint(
                rho_target=R,
                safety_status=SafetyStatus.INSUFFICIENT_TELEMETRY,
                safety_vetoes=("empty_telemetry_window",),
                notes=("estimator received empty telemetry window",)))
        return points

    for R in rhos:
        metrics = _evaluate_for_rho(ticks, R, mode=cfg.mode,
                                    tick_seconds=cfg.tick_seconds,
                                    prefill_savings=cfg.prefill_savings)
        # build a provisional point (status TBD) then classify
        provisional = FrontierPoint(safety_status=SafetyStatus.SAFE, **metrics)
        status, vetoes = classify_point_safety(
            provisional, safety,
            telemetry_confidence=profile.telemetry_confidence)
        points.append(FrontierPoint(safety_status=status, safety_vetoes=vetoes,
                                    notes=(cfg.mode,), **metrics))
    return points


def estimate_frontier_from_points(profile: WorkloadFrontierProfile,
                                  raw_points: Iterable[dict],
                                  *,
                                  safety_config: Optional[SafetyConfig] = None
                                  ) -> list[FrontierPoint]:
    """Build :class:`FrontierPoint` objects from pre-computed metric dicts.

    Each ``raw_points`` entry MUST carry ``rho_target`` and may carry any
    of the ``predicted_*`` metric fields used by :class:`FrontierPoint`. Used
    by the Azure 2024 integration script to reuse the existing audit JSON
    without re-running the full week-long simulator.
    """
    safety = safety_config or SafetyConfig()
    out: list[FrontierPoint] = []
    field_names = {f for f in FrontierPoint.__dataclass_fields__
                   if f.startswith("predicted_")} | {"rho_target"}
    for raw in raw_points:
        kwargs = {k: raw.get(k) for k in field_names if k in raw}
        if "rho_target" not in kwargs:
            raise ValueError("frontier point dict missing 'rho_target'")
        prov = FrontierPoint(safety_status=SafetyStatus.SAFE, **kwargs)
        status, vetoes = classify_point_safety(
            prov, safety, telemetry_confidence=profile.telemetry_confidence)
        out.append(FrontierPoint(safety_status=status, safety_vetoes=vetoes,
                                 notes=tuple(raw.get("notes", ())), **kwargs))
    return out
