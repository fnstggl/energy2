"""Production shadow mode for Aurelius.

Shadow mode runs the optimizer as if live — making real scheduling decisions
against current price data — without executing any actual workloads.

After 7-14 days, realized RT prices are loaded and compared against the
optimizer's predicted savings to produce an honest pilot-grade report.

Workflow:
    1. shadow run   → DecisionRecords with predicted costs (realized = None)
    2. shadow realize → DecisionRecords with realized RT costs filled in
    3. shadow report  → ShadowReport: predicted vs realized savings comparison

Leakage invariant:
    LiveShadowRunner trains only on data with timestamp < decision_time.
    Realized RT prices are never visible at decision time.
    RealizedSavingsCalculator only runs AFTER the scheduled window has passed.
"""

from .models import DecisionRecord
from .recorder import DecisionRecorder
from .runner import LiveShadowRunner
from .realizer import RealizedSavingsCalculator
from .report import ShadowReport

__all__ = [
    "DecisionRecord",
    "DecisionRecorder",
    "LiveShadowRunner",
    "RealizedSavingsCalculator",
    "ShadowReport",
]
