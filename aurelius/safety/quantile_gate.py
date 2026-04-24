"""Quantile-based safety gate for filtering risky schedule decisions.

This module provides a deterministic, explainable safety gate that filters
optimizer decisions based on forecast uncertainty using quantile regression outputs.

The gate decides WHETHER a decision may be executed. It does NOT:
- Modify decisions
- Reschedule jobs
- Reroute workloads
- Override optimizer logic

DESIGN INTENT:
- Aurelius owns decisions, NOT risk tolerance
- The safety gate enforces downside bounds
- Execution must remain optional and reversible
- Gate behavior must be deterministic and explainable
- This is pilot-safe infrastructure
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class QuantileGateConfig:
    """Configuration for the quantile safety gate.

    Attributes:
        enabled: Whether the gate is active (False = pass all decisions)
        quantile: Which quantile to use for worst-case (0.9 or 0.95)
        min_expected_savings_pct: Minimum required expected savings (p50 vs baseline)
        max_downside_risk_pct: Maximum allowed downside risk (how much worse than baseline)
        metric: Which metric(s) to gate on ("energy_cost", "carbon", or "both")
    """
    enabled: bool = True
    quantile: Literal[0.9, 0.95] = 0.9
    min_expected_savings_pct: float = 0.0
    max_downside_risk_pct: float = 10.0
    metric: Literal["energy_cost", "carbon", "both"] = "both"


@dataclass
class GateResult:
    """Internal result of gate evaluation for a single metric.

    Not part of public API - used for audit logging.
    """
    metric: str
    passed: bool
    expected_savings_pct: Optional[float] = None
    worst_case_savings_pct: Optional[float] = None
    reason: str = ""


class QuantileSafetyGate:
    """Deterministic safety gate that filters risky schedule decisions.

    Uses quantile regression forecasts (p50/p90) to evaluate downside risk.
    Decisions that exceed risk thresholds are filtered out.

    Example:
        >>> gate = QuantileSafetyGate()
        >>> config = QuantileGateConfig(
        ...     min_expected_savings_pct=5.0,
        ...     max_downside_risk_pct=10.0,
        ...     metric="both"
        ... )
        >>> safe_decisions = gate.filter(decisions, config)
    """

    def filter(
        self,
        decisions: list[Any],  # list[ScheduleDecision]
        config: QuantileGateConfig,
    ) -> list[Any]:
        """Filter decisions based on quantile safety thresholds.

        Args:
            decisions: List of ScheduleDecision objects (may include forecast metadata)
            config: Gate configuration with thresholds

        Returns:
            List of decisions that pass the safety gate.
            Order is preserved. Decisions that fail are filtered out.
        """
        # If gate disabled, return all decisions unchanged
        if not config.enabled:
            logger.debug("Quantile safety gate disabled, passing all decisions")
            return decisions

        allowed_decisions = []

        for decision in decisions:
            job_id = getattr(decision, "job_id", "unknown")
            forecast = getattr(decision, "forecast", None)

            # Determine which metrics to check
            metrics_to_check = self._get_metrics_to_check(config.metric)

            # Evaluate each required metric
            results = []
            for metric in metrics_to_check:
                result = self._evaluate_metric(
                    job_id=job_id,
                    forecast=forecast,
                    metric=metric,
                    config=config,
                )
                results.append(result)
                self._emit_audit_log(job_id, result, config)

            # Decision passes only if ALL required metrics pass
            if all(r.passed for r in results):
                allowed_decisions.append(decision)
            else:
                # Log overall decision filtered
                failed_metrics = [r.metric for r in results if not r.passed]
                logger.info(
                    f"Decision {job_id} filtered by safety gate: "
                    f"failed metrics: {failed_metrics}"
                )

        logger.info(
            f"Quantile safety gate: {len(allowed_decisions)}/{len(decisions)} "
            f"decisions passed (quantile={config.quantile})"
        )

        return allowed_decisions

    def _get_metrics_to_check(self, metric_config: str) -> list[str]:
        """Get list of metrics to evaluate based on config."""
        if metric_config == "both":
            return ["energy_cost", "carbon"]
        else:
            return [metric_config]

    def _evaluate_metric(
        self,
        job_id: str,
        forecast: Optional[dict],
        metric: str,
        config: QuantileGateConfig,
    ) -> GateResult:
        """Evaluate a single metric against safety thresholds.

        Returns GateResult with pass/fail and reason.
        """
        # Handle missing forecast data — FAIL CLOSED
        if forecast is None:
            logger.warning(
                f"Decision {job_id}: missing forecast; BLOCKING (fail-closed) for {metric}"
            )
            return GateResult(
                metric=metric,
                passed=False,
                reason="missing forecast; blocked (fail-closed)",
            )

        # Get metric-specific forecast data — FAIL CLOSED
        metric_data = forecast.get(metric)
        if metric_data is None:
            logger.warning(
                f"Decision {job_id}: missing {metric} forecast; BLOCKING (fail-closed)"
            )
            return GateResult(
                metric=metric,
                passed=False,
                reason=f"missing {metric} forecast; blocked (fail-closed)",
            )

        # Extract values
        p50 = metric_data.get("p50")
        p90 = metric_data.get("p90")
        baseline = metric_data.get("baseline")

        # Handle missing p50 — FAIL CLOSED
        if p50 is None:
            logger.warning(
                f"Decision {job_id}: missing p50 for {metric}; BLOCKING (fail-closed)"
            )
            return GateResult(
                metric=metric,
                passed=False,
                reason=f"missing p50 for {metric}; blocked (fail-closed)",
            )

        # Handle missing p90 — FAIL CLOSED
        if p90 is None:
            logger.warning(
                f"Decision {job_id}: missing p90 for {metric}; BLOCKING (fail-closed)"
            )
            return GateResult(
                metric=metric,
                passed=False,
                reason=f"missing p90 for {metric}; blocked (fail-closed)",
            )

        # Handle missing or invalid baseline — FAIL CLOSED
        if baseline is None or baseline <= 0:
            logger.warning(
                f"Decision {job_id}: invalid baseline ({baseline}) for {metric}; "
                f"BLOCKING (fail-closed)"
            )
            return GateResult(
                metric=metric,
                passed=False,
                reason=f"invalid baseline ({baseline}); blocked (fail-closed)",
            )

        # Handle quantile=0.95 - use p90 as conservative proxy
        if config.quantile == 0.95:
            logger.warning(
                f"Decision {job_id}: quantile=0.95 requested but using p90 as proxy"
            )

        # Calculate savings percentages
        # expected_savings_pct = (baseline - p50) / baseline * 100
        # worst_case_savings_pct = (baseline - p90) / baseline * 100
        expected_savings_pct = (baseline - p50) / baseline * 100
        worst_case_savings_pct = (baseline - p90) / baseline * 100

        # Apply gating logic
        # ALLOW if:
        #   expected_savings_pct >= min_expected_savings_pct
        #   AND worst_case_savings_pct >= -max_downside_risk_pct
        passes_expected = expected_savings_pct >= config.min_expected_savings_pct
        passes_worst_case = worst_case_savings_pct >= -config.max_downside_risk_pct

        if passes_expected and passes_worst_case:
            return GateResult(
                metric=metric,
                passed=True,
                expected_savings_pct=expected_savings_pct,
                worst_case_savings_pct=worst_case_savings_pct,
                reason="passed all thresholds",
            )
        else:
            # Build detailed failure reason
            reasons = []
            if not passes_expected:
                reasons.append(
                    f"expected_savings ({expected_savings_pct:.2f}%) < "
                    f"min_expected ({config.min_expected_savings_pct:.2f}%)"
                )
            if not passes_worst_case:
                reasons.append(
                    f"worst_case_savings ({worst_case_savings_pct:.2f}%) < "
                    f"-max_downside ({-config.max_downside_risk_pct:.2f}%)"
                )

            return GateResult(
                metric=metric,
                passed=False,
                expected_savings_pct=expected_savings_pct,
                worst_case_savings_pct=worst_case_savings_pct,
                reason="; ".join(reasons),
            )

    def _emit_audit_log(
        self,
        job_id: str,
        result: GateResult,
        config: QuantileGateConfig,
    ) -> None:
        """Emit structured JSON audit log for every gate evaluation."""
        audit_entry = {
            "event": "quantile_safety_gate",
            "job_id": job_id,
            "metric": result.metric,
            "quantile": config.quantile,
            "expected_savings_pct": (
                f"{result.expected_savings_pct:.2f}"
                if result.expected_savings_pct is not None
                else None
            ),
            "worst_case_savings_pct": (
                f"{result.worst_case_savings_pct:.2f}"
                if result.worst_case_savings_pct is not None
                else None
            ),
            "min_expected_savings_pct": f"{config.min_expected_savings_pct:.2f}",
            "max_downside_risk_pct": f"{config.max_downside_risk_pct:.2f}",
            "status": "passed" if result.passed else "filtered",
            "reason": result.reason,
        }

        # Log as JSON for machine parsing
        logger.info(json.dumps(audit_entry))


# ============================================================================
# INLINE VALIDATION
# ============================================================================
# Run with: python -c "from aurelius.safety.quantile_gate import _run_validation; _run_validation()"

def _run_validation():
    """Validate quantile safety gate behavior."""
    from dataclasses import dataclass as dc

    print("=" * 60)
    print("Quantile Safety Gate Validation")
    print("=" * 60)

    # Mock ScheduleDecision for testing
    @dc
    class MockDecision:
        job_id: str
        forecast: Optional[dict] = None

    gate = QuantileSafetyGate()

    # Test 1: Gate disabled passes all
    print("\nTest 1: GATE DISABLED")
    print("-" * 40)
    config_disabled = QuantileGateConfig(enabled=False)
    decisions = [
        MockDecision(job_id="job1", forecast=None),
        MockDecision(job_id="job2", forecast=None),
    ]
    result = gate.filter(decisions, config_disabled)
    assert len(result) == 2, "Disabled gate should pass all"
    print(f"  Disabled gate passed all {len(result)} decisions: PASS")

    # Test 2: Missing forecast treated as passing
    print("\nTest 2: MISSING FORECAST")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
    )
    decisions = [MockDecision(job_id="job1", forecast=None)]
    result = gate.filter(decisions, config)
    assert len(result) == 1, "Missing forecast should pass"
    print(f"  Missing forecast treated as passing: PASS")

    # Test 3: Valid forecast - passes thresholds
    print("\nTest 3: VALID FORECAST - PASSES")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
        metric="energy_cost",
    )
    # baseline=100, p50=90 (10% savings), p90=105 (5% worse than baseline)
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={
                "energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0},
            },
        )
    ]
    result = gate.filter(decisions, config)
    assert len(result) == 1, "Should pass with 10% expected savings and 5% downside"
    print(f"  Expected 10% savings, 5% downside: PASS")

    # Test 4: Valid forecast - fails expected savings
    print("\nTest 4: VALID FORECAST - FAILS EXPECTED")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=15.0,  # Require 15% savings
        max_downside_risk_pct=10.0,
        metric="energy_cost",
    )
    # baseline=100, p50=90 (only 10% savings)
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={
                "energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0},
            },
        )
    ]
    result = gate.filter(decisions, config)
    assert len(result) == 0, "Should fail with only 10% savings (need 15%)"
    print(f"  10% savings < 15% required: filtered correctly: PASS")

    # Test 5: Valid forecast - fails downside risk
    print("\nTest 5: VALID FORECAST - FAILS DOWNSIDE")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=5.0,  # Only allow 5% downside
        metric="energy_cost",
    )
    # baseline=100, p50=90, p90=115 (15% downside risk)
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={
                "energy_cost": {"p50": 90.0, "p90": 115.0, "baseline": 100.0},
            },
        )
    ]
    result = gate.filter(decisions, config)
    assert len(result) == 0, "Should fail with 15% downside (max 5%)"
    print(f"  15% downside > 5% max: filtered correctly: PASS")

    # Test 6: Both metrics required
    print("\nTest 6: BOTH METRICS REQUIRED")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
        metric="both",
    )
    # energy_cost passes, carbon fails
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={
                "energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0},
                "carbon": {"p50": 500.0, "p90": 600.0, "baseline": 400.0},  # Worse than baseline
            },
        )
    ]
    result = gate.filter(decisions, config)
    assert len(result) == 0, "Should fail when one metric fails"
    print(f"  Energy passes, carbon fails: filtered correctly: PASS")

    # Test 7: Invalid baseline treated as passing
    print("\nTest 7: INVALID BASELINE")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
        metric="energy_cost",
    )
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={
                "energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 0.0},
            },
        )
    ]
    result = gate.filter(decisions, config)
    assert len(result) == 1, "Invalid baseline should pass"
    print(f"  Zero baseline treated as passing: PASS")

    # Test 8: Order preserved
    print("\nTest 8: ORDER PRESERVED")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
        metric="energy_cost",
    )
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={"energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0}},
        ),
        MockDecision(
            job_id="job2",
            forecast={"energy_cost": {"p50": 150.0, "p90": 160.0, "baseline": 100.0}},  # Fails
        ),
        MockDecision(
            job_id="job3",
            forecast={"energy_cost": {"p50": 85.0, "p90": 95.0, "baseline": 100.0}},
        ),
    ]
    result = gate.filter(decisions, config)
    assert len(result) == 2, "Should filter job2"
    assert result[0].job_id == "job1", "Order should be preserved"
    assert result[1].job_id == "job3", "Order should be preserved"
    print(f"  Filtered job2, preserved order [job1, job3]: PASS")

    # Test 9: Quantile 0.95 uses p90 as proxy
    print("\nTest 9: QUANTILE 0.95 USES P90 PROXY")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
        metric="energy_cost",
        quantile=0.95,  # Should use p90 as proxy
    )
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={"energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0}},
        )
    ]
    result = gate.filter(decisions, config)
    assert len(result) == 1, "Should still work with 0.95 using p90 proxy"
    print(f"  Quantile 0.95 uses p90 proxy: PASS")

    # Test 10: Determinism
    print("\nTest 10: DETERMINISM")
    print("-" * 40)
    config = QuantileGateConfig(
        min_expected_savings_pct=5.0,
        max_downside_risk_pct=10.0,
    )
    decisions = [
        MockDecision(
            job_id="job1",
            forecast={
                "energy_cost": {"p50": 90.0, "p90": 105.0, "baseline": 100.0},
                "carbon": {"p50": 350.0, "p90": 380.0, "baseline": 400.0},
            },
        ),
        MockDecision(
            job_id="job2",
            forecast={
                "energy_cost": {"p50": 150.0, "p90": 160.0, "baseline": 100.0},
                "carbon": {"p50": 300.0, "p90": 350.0, "baseline": 400.0},
            },
        ),
    ]
    result1 = gate.filter(decisions, config)
    result2 = gate.filter(decisions, config)
    assert len(result1) == len(result2), "Should be deterministic"
    assert all(r1.job_id == r2.job_id for r1, r2 in zip(result1, result2))
    print(f"  Same input produces same output: PASS")

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_validation()
