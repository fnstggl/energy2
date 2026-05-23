"""Leakage-free walk-forward backtesting for Aurelius."""

from .baselines import (
    ALL_BASELINES,
    BaselinePolicy,
    closest_region_policy,
    current_price_only_policy,
    fifo_policy,
    fixed_primary_region_policy,
    latency_first_policy,
    peak_blind_asap_policy,
    round_robin_policy,
)
from .engine import BacktestEngine, BacktestRound
from .evaluator import evaluate_schedule
from .splitter import TemporalSplitter

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
