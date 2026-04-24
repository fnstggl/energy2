"""Safety gates for Aurelius scheduler decisions.

This module provides safety gates that filter optimizer decisions
based on risk thresholds. Gates do NOT modify decisions - they only
decide whether execution should proceed.

Available gates:
- QuantileSafetyGate: Filters based on quantile forecast uncertainty
"""

from .quantile_gate import QuantileGateConfig, QuantileSafetyGate

__all__ = [
    "QuantileGateConfig",
    "QuantileSafetyGate",
]
