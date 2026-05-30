"""Forecast-robustness experiment for the Azure LLM 2024 week-long backtest.

Answers Task 4: *how much of Aurelius's autoscaling alpha survives imperfect
demand forecasting?* and feeds the attribution analysis (forecasting vs
autoscaling-timing).

A single forecast-driven autoscaler provisions replicas for the demand a chosen
forecast predicts; the **only** thing that differs across modes is the demand
estimate. Everything else — the unchanged serving physics
(``aurelius/simulation/cluster/serving.py`` via ``backtest.evaluate_tick``), the
sizing function, the target utilization, and the canonical KPI
(``economics.py``) — is identical. No future leakage except ``oracle_future``,
which is explicitly **analysis-only**.

Forecast modes:
  * ``oracle_future``       — perfect knowledge of the tick being served (UPPER
    BOUND, analysis-only).
  * ``no_forecast_reactive``— provision for the *previous* tick (one-tick lag);
    the no-forecasting baseline against which forecast alpha is measured.
  * ``moving_average``      — mean of the last ``window`` ticks (prior only).
  * ``ewma``                — exponentially weighted moving average (prior only).
  * ``seasonal_time_of_day``— average of the same time-of-day phase over PRIOR
    days (prior only).
  * ``noisy_forecast``      — oracle perturbed by seeded multiplicative noise (a
    good-but-imperfect forecaster).

Directional simulator/backtest evidence — not production savings
(``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from . import backtest as bt

FORECAST_MODES = (
    "oracle_future", "seasonal_time_of_day", "moving_average", "ewma",
    "noisy_forecast", "no_forecast_reactive",
)
ANALYSIS_ONLY_MODES = ("oracle_future",)
HEADLINE_FORECAST_BASELINE = "no_forecast_reactive"

# Single fixed sizing target across all modes (only the forecast differs).
FORECAST_TARGET_RHO = 0.60
DEFAULT_WINDOW = 10
DEFAULT_EWMA_ALPHA = 0.3
DEFAULT_NOISE = 0.15


def _actual(ticks):
    return [(t.arrival_rate_rps, max(1.0, t.output_tokens_mean) if t.request_count
             else 0.0) for t in ticks]


def forecast_series(ticks, mode, *, tick_seconds, window=DEFAULT_WINDOW,
                    ewma_alpha=DEFAULT_EWMA_ALPHA, noise=DEFAULT_NOISE,
                    seed=0) -> list:
    """Per-tick ``(forecast_rate, forecast_output_mean)``.

    Leakage rule (binding): every mode except ``oracle_future`` uses ONLY ticks
    strictly before the one being provisioned.
    """
    actual = _actual(ticks)
    n = len(actual)
    out: list = [(0.0, 0.0)] * n
    if n == 0:
        return out

    if mode == "oracle_future":
        return list(actual)  # analysis-only upper bound (uses the served tick)

    if mode == "noisy_forecast":
        rng = random.Random(seed)
        for i in range(n):
            ar, ao = actual[i]
            # multiplicative seeded noise on the oracle (good-but-imperfect)
            er = max(0.0, ar * (1.0 + rng.gauss(0.0, noise)))
            eo = max(0.0, ao * (1.0 + rng.gauss(0.0, noise)))
            out[i] = (er, eo)
        return out

    if mode == "no_forecast_reactive":
        out[0] = actual[0]
        for i in range(1, n):
            out[i] = actual[i - 1]
        return out

    if mode == "moving_average":
        out[0] = actual[0]
        for i in range(1, n):
            lo = max(0, i - window)
            w = actual[lo:i]
            out[i] = (sum(r for r, _ in w) / len(w),
                      sum(o for _, o in w) / len(w))
        return out

    if mode == "ewma":
        out[0] = actual[0]
        er, eo = actual[0]
        for i in range(1, n):
            ar, ao = actual[i - 1]
            er = ewma_alpha * ar + (1 - ewma_alpha) * er
            eo = ewma_alpha * ao + (1 - ewma_alpha) * eo
            out[i] = (er, eo)
        return out

    if mode == "seasonal_time_of_day":
        ticks_per_day = int(round(86400.0 / tick_seconds))
        # accumulate per-phase running sums over PRIOR days only
        phase_sum_r: dict = {}
        phase_sum_o: dict = {}
        phase_cnt: dict = {}
        # moving-average fallback state for phases with no prior observation
        for i in range(n):
            phase = i % ticks_per_day if ticks_per_day > 0 else 0
            cnt = phase_cnt.get(phase, 0)
            if cnt > 0:
                out[i] = (phase_sum_r[phase] / cnt, phase_sum_o[phase] / cnt)
            else:
                # no prior same-phase day yet → reactive fallback
                out[i] = actual[i - 1] if i > 0 else actual[0]
            ar, ao = actual[i]
            phase_sum_r[phase] = phase_sum_r.get(phase, 0.0) + ar
            phase_sum_o[phase] = phase_sum_o.get(phase, 0.0) + ao
            phase_cnt[phase] = cnt + 1
        return out

    raise ValueError(f"unknown forecast mode {mode}")


def _forecast_error(forecast, ticks) -> dict:
    """MAE / MAPE / RMSE of the forecast vs actual over ACTIVE ticks, for RPS
    and for output-token demand."""
    rab = [(f[0], t.arrival_rate_rps) for f, t in zip(forecast, ticks)
           if t.request_count > 0]
    oab = [(f[1], max(1.0, t.output_tokens_mean)) for f, t in zip(forecast, ticks)
           if t.request_count > 0]

    def _metrics(pairs):
        if not pairs:
            return {"mae": None, "mape": None, "rmse": None}
        n = len(pairs)
        mae = sum(abs(f - a) for f, a in pairs) / n
        rmse = math.sqrt(sum((f - a) ** 2 for f, a in pairs) / n)
        mape = sum(abs(f - a) / a for f, a in pairs if a > 0) / n
        return {"mae": round(mae, 4), "mape": round(mape, 4), "rmse": round(rmse, 4)}

    return {"rps": _metrics(rab), "output_tokens": _metrics(oab)}


@dataclass
class ForecastModeResult:
    mode: str
    analysis_only: bool
    result: object  # backtest.PolicyResult
    forecast_error: dict
    scale_events: int

    def summary(self) -> dict:
        s = self.result.summary()
        return {
            "mode": self.mode,
            "analysis_only": self.analysis_only,
            "sla_safe_goodput_per_infra_dollar": s["sla_safe_goodput_per_infra_dollar"],
            "sla_compliant_goodput": s["sla_compliant_goodput"],
            "total_infrastructure_cost": s["total_infrastructure_cost"],
            "active_gpu_hours": s["active_gpu_hours"],
            "latency_p99_ms": s["latency_p99_ms"],
            "queue_p99_ms": s["queue_p99_ms"],
            "timeout_rate_pct_mean": s["timeout_rate_pct_mean"],
            "scale_events": self.scale_events,
            "forecast_error": self.forecast_error,
        }


def _run_forecast_mode(ticks, forecast, *, tick_hours, mode) -> ForecastModeResult:
    evals = []
    prev_replicas = None
    for t, (f_rate, f_out) in zip(ticks, forecast):
        throughput = bt._tick_throughput_tokps(t)
        replicas = bt._size_for_target(max(0.0, f_rate), max(1.0, f_out or 1.0),
                                       throughput, FORECAST_TARGET_RHO)
        ev = bt.evaluate_tick(t, replicas, prefill_savings=0.0, tick_hours=tick_hours)
        if prev_replicas is not None and ev.replicas != prev_replicas:
            ev.scale_event = True
        prev_replicas = ev.replicas
        evals.append(ev)
    res = bt._aggregate(mode, evals, cache_aware=False, ticks=ticks)
    scale_events = sum(1 for e in evals if e.scale_event)
    return ForecastModeResult(
        mode=mode, analysis_only=(mode in ANALYSIS_ONLY_MODES), result=res,
        forecast_error=_forecast_error(forecast, ticks), scale_events=scale_events)


@dataclass
class ForecastExperiment:
    tick_seconds: float
    modes: dict = field(default_factory=dict)         # mode -> ForecastModeResult
    headline_baseline: str = HEADLINE_FORECAST_BASELINE

    def _kpi(self, mode):
        r = self.modes.get(mode)
        return (r.result.kpi.sla_safe_goodput_per_infra_dollar or 0.0) if r else 0.0

    def alpha_survival(self) -> dict:
        """alpha(mode) = KPI(mode) - KPI(no_forecast_reactive);
        alpha_survival = alpha(mode) / alpha(oracle_future)."""
        base = self._kpi(self.headline_baseline)
        oracle_alpha = self._kpi("oracle_future") - base
        out = {}
        for mode in self.modes:
            if mode in (self.headline_baseline, "oracle_future"):
                continue
            alpha = self._kpi(mode) - base
            if oracle_alpha <= 0:
                survival = None  # not applicable (no oracle alpha to survive)
            else:
                survival = round(alpha / oracle_alpha, 4)
            out[mode] = {
                "alpha_vs_no_forecast": round(alpha, 6),
                "alpha_survival_ratio": survival,
            }
        return {
            "headline_baseline": self.headline_baseline,
            "no_forecast_goodput_per_dollar": round(base, 6),
            "oracle_goodput_per_dollar": round(self._kpi("oracle_future"), 6),
            "oracle_alpha": round(oracle_alpha, 6),
            "oracle_alpha_positive": oracle_alpha > 0,
            "per_mode": out,
        }

    def to_dict(self) -> dict:
        return {
            "tick_seconds": self.tick_seconds,
            "target_rho": FORECAST_TARGET_RHO,
            "leakage_note": ("no future leakage except oracle_future "
                             "(analysis-only upper bound)"),
            "modes": {m: r.summary() for m, r in self.modes.items()},
            "alpha_survival": self.alpha_survival(),
        }


def run_forecast_experiment(ticks, *, tick_seconds, window=DEFAULT_WINDOW,
                            ewma_alpha=DEFAULT_EWMA_ALPHA, noise=DEFAULT_NOISE,
                            seed=0, modes=FORECAST_MODES) -> ForecastExperiment:
    tick_hours = tick_seconds / 3600.0
    exp = ForecastExperiment(tick_seconds=tick_seconds)
    for mode in modes:
        fc = forecast_series(ticks, mode, tick_seconds=tick_seconds, window=window,
                             ewma_alpha=ewma_alpha, noise=noise, seed=seed)
        exp.modes[mode] = _run_forecast_mode(ticks, fc, tick_hours=tick_hours,
                                             mode=mode)
    return exp
