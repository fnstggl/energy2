"""Leakage-free walk-forward backtesting for Aurelius."""

from .engine import BacktestEngine, BacktestRound
from .splitter import TemporalSplitter
from .baselines import (
    BaselinePolicy,
    fifo_policy,
    peak_blind_asap_policy,
    latency_first_policy,
    closest_region_policy,
    fixed_primary_region_policy,
    current_price_only_policy,
    round_robin_policy,
    ALL_BASELINES,
)
from .evaluator import evaluate_schedule

__all__ = [
    "BacktestEngine",
    "BacktestRound",
    "TemporalSplitter",
    "BaselinePolicy",
    "fifo_policy",
    "peak_blind_asap_policy",
    "latency_first_policy",
    "closest_region_policy",
    "fixed_primary_region_policy",
    "current_price_only_policy",
    "round_robin_policy",
    "ALL_BASELINES",
    "evaluate_schedule",
]
