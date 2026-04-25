"""Post-execution measurement and learning hooks for Aurelius.

This module provides observability-only post-execution recording:
- Records realized outcomes after execution
- Measures forecast errors
- Labels decision outcomes

This is NOT online learning. This is NOT retraining.
This is measurement only for offline analysis.

IMPORTANT:
- Read-only with respect to decisions
- Deterministic
- Side-effect free (except file writes)
- Safe in dry_run and live modes
- Failure-tolerant (never blocks execution)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
import hashlib
import json
import logging

from ..models import ScheduleDecision
from ..data.persistence import JSONLWriter, RotatingJSONLWriter, get_default_post_execution_path
from .base import ExecutionConfig, ExecutionResult

logger = logging.getLogger(__name__)


# Threshold for "close to zero" savings (in currency units)
# Used for neutral vs good_decision labeling
# Epsilon = 0.01 means savings within +/- $0.01 are considered neutral
SAVINGS_EPSILON = 0.01


@dataclass
class ForecastSnapshot:
    """Snapshot of forecast values at decision time.

    Captures the p50, p90, and baseline forecasts for a decision.
    These values are immutable and used for post-hoc error analysis.

    Attributes:
        energy_cost_p50: Predicted energy cost at p50 quantile
        energy_cost_p90: Predicted energy cost at p90 quantile
        energy_cost_baseline: Baseline energy cost (no optimization)
        carbon_p50: Predicted carbon emissions at p50 quantile
        carbon_p90: Predicted carbon emissions at p90 quantile
        carbon_baseline: Baseline carbon emissions (no optimization)
    """
    energy_cost_p50: Optional[float] = None
    energy_cost_p90: Optional[float] = None
    energy_cost_baseline: Optional[float] = None
    carbon_p50: Optional[float] = None
    carbon_p90: Optional[float] = None
    carbon_baseline: Optional[float] = None


@dataclass
class RealizedOutcome:
    """Realized outcome after job execution.

    Captures what actually happened after the decision was executed.
    Used for measuring forecast accuracy and decision quality.

    Attributes:
        realized_start_time: When the job actually started
        realized_energy_price: Actual energy price ($/kWh) during execution
        realized_carbon_intensity: Actual carbon intensity (gCO2/kWh) during execution
        realized_energy_cost: Actual total energy cost (if computable)
        realized_carbon: Actual total carbon emissions (if computable)
    """
    realized_start_time: Optional[datetime] = None
    realized_energy_price: Optional[float] = None
    realized_carbon_intensity: Optional[float] = None
    realized_energy_cost: Optional[float] = None
    realized_carbon: Optional[float] = None


@dataclass
class PostExecutionRecord:
    """Immutable record of post-execution measurement.

    This record captures everything needed for offline analysis:
    - Decision context (job, region, timing)
    - Forecast values at decision time
    - Realized values after execution
    - Computed errors and labels

    Attributes:
        job_id: The job identifier
        decision_id: Stable identifier (derived from job_id + start_time)
        region: Execution region
        baseline_start_time: When baseline would have started
        optimized_start_time: When optimization scheduled start
        realized_start_time: When job actually started
        realized_energy_price: Actual energy price during execution
        realized_carbon_intensity: Actual carbon intensity during execution
        forecast_energy_cost_p50: Predicted energy cost at p50
        forecast_energy_cost_p90: Predicted energy cost at p90
        forecast_energy_cost_baseline: Baseline energy cost forecast
        forecast_carbon_p50: Predicted carbon at p50
        forecast_carbon_p90: Predicted carbon at p90
        forecast_carbon_baseline: Baseline carbon forecast
        energy_cost_p50_error: realized - p50 (null if unavailable)
        energy_cost_p90_covered: realized <= p90 (null if unavailable)
        carbon_p50_error: realized - p50 (null if unavailable)
        carbon_p90_covered: realized <= p90 (null if unavailable)
        realized_savings: baseline_cost - realized_cost (null if unavailable)
        decision_outcome_label: "good_decision" | "neutral" | "conservative_skip"
        execution_mode: "dry_run" | "live"
        constraint_profile: "batch_optimized" | "latency_safe"
        execution_status: Status from ExecutionResult
        recorded_at: When this record was created
    """
    job_id: str
    decision_id: str
    region: str
    baseline_start_time: Optional[datetime]
    optimized_start_time: datetime
    realized_start_time: Optional[datetime]
    realized_energy_price: Optional[float]
    realized_carbon_intensity: Optional[float]
    forecast_energy_cost_p50: Optional[float]
    forecast_energy_cost_p90: Optional[float]
    forecast_energy_cost_baseline: Optional[float]
    forecast_carbon_p50: Optional[float]
    forecast_carbon_p90: Optional[float]
    forecast_carbon_baseline: Optional[float]
    energy_cost_p50_error: Optional[float]
    energy_cost_p90_covered: Optional[bool]
    carbon_p50_error: Optional[float]
    carbon_p90_covered: Optional[bool]
    realized_savings: Optional[float]
    decision_outcome_label: Literal["good_decision", "neutral", "conservative_skip"]
    execution_mode: Literal["dry_run", "live"]
    constraint_profile: Literal["batch_optimized", "latency_safe"]
    execution_status: Literal["dry_run", "submitted", "skipped", "aborted"]
    recorded_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "decision_id": self.decision_id,
            "region": self.region,
            "baseline_start_time": (
                self.baseline_start_time.isoformat()
                if self.baseline_start_time else None
            ),
            "optimized_start_time": self.optimized_start_time.isoformat(),
            "realized_start_time": (
                self.realized_start_time.isoformat()
                if self.realized_start_time else None
            ),
            "realized_energy_price": self.realized_energy_price,
            "realized_carbon_intensity": self.realized_carbon_intensity,
            "forecast_energy_cost_p50": self.forecast_energy_cost_p50,
            "forecast_energy_cost_p90": self.forecast_energy_cost_p90,
            "forecast_energy_cost_baseline": self.forecast_energy_cost_baseline,
            "forecast_carbon_p50": self.forecast_carbon_p50,
            "forecast_carbon_p90": self.forecast_carbon_p90,
            "forecast_carbon_baseline": self.forecast_carbon_baseline,
            "energy_cost_p50_error": self.energy_cost_p50_error,
            "energy_cost_p90_covered": self.energy_cost_p90_covered,
            "carbon_p50_error": self.carbon_p50_error,
            "carbon_p90_covered": self.carbon_p90_covered,
            "realized_savings": self.realized_savings,
            "decision_outcome_label": self.decision_outcome_label,
            "execution_mode": self.execution_mode,
            "constraint_profile": self.constraint_profile,
            "execution_status": self.execution_status,
            "recorded_at": self.recorded_at.isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


def generate_decision_id(job_id: str, start_time: datetime) -> str:
    """Generate a stable decision ID from job_id and start_time.

    The decision_id is a deterministic hash that uniquely identifies
    a specific scheduling decision for a job.

    Args:
        job_id: The job identifier
        start_time: The scheduled start time

    Returns:
        A stable 16-character hex string
    """
    content = f"{job_id}:{start_time.isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compute_forecast_errors(
    forecast: ForecastSnapshot,
    realized: RealizedOutcome,
) -> dict[str, Any]:
    """Compute forecast errors from realized outcomes.

    Rules:
    - If forecast missing -> skip silently (return None)
    - If realized value missing -> skip silently (return None)
    - p50_error = realized_value - forecast.p50
    - p90_covered = (realized_value <= forecast.p90)

    Args:
        forecast: The forecast snapshot at decision time
        realized: The realized outcome after execution

    Returns:
        Dictionary with error metrics (values may be None)
    """
    result = {
        "energy_cost_p50_error": None,
        "energy_cost_p90_covered": None,
        "carbon_p50_error": None,
        "carbon_p90_covered": None,
    }

    # Energy cost errors
    if (
        realized.realized_energy_cost is not None
        and forecast.energy_cost_p50 is not None
    ):
        result["energy_cost_p50_error"] = (
            realized.realized_energy_cost - forecast.energy_cost_p50
        )

    if (
        realized.realized_energy_cost is not None
        and forecast.energy_cost_p90 is not None
    ):
        result["energy_cost_p90_covered"] = (
            realized.realized_energy_cost <= forecast.energy_cost_p90
        )

    # Carbon errors
    if realized.realized_carbon is not None and forecast.carbon_p50 is not None:
        result["carbon_p50_error"] = realized.realized_carbon - forecast.carbon_p50

    if realized.realized_carbon is not None and forecast.carbon_p90 is not None:
        result["carbon_p90_covered"] = realized.realized_carbon <= forecast.carbon_p90

    return result


def compute_realized_savings(
    forecast: ForecastSnapshot,
    realized: RealizedOutcome,
) -> Optional[float]:
    """Compute realized savings from baseline.

    realized_savings = baseline_energy_cost - realized_energy_cost

    Rules:
    - If baseline_energy_cost is missing -> return None
    - If realized_energy_cost is missing -> return None
    - Only use values we have deterministic access to

    Args:
        forecast: The forecast snapshot (contains baseline)
        realized: The realized outcome (contains realized cost if computable)

    Returns:
        Realized savings in currency units, or None if not computable
    """
    if forecast.energy_cost_baseline is None:
        return None
    if realized.realized_energy_cost is None:
        return None

    return forecast.energy_cost_baseline - realized.realized_energy_cost


def label_decision_outcome(
    execution_status: str,
    realized_savings: Optional[float],
    was_skipped_by_safety_or_latency: bool,
) -> Literal["good_decision", "neutral", "conservative_skip"]:
    """Label the decision outcome deterministically.

    Labeling rules (deterministic):

    good_decision:
    - decision executed (status in ["submitted", "dry_run"])
    - realized_savings is known and >= SAVINGS_EPSILON
    - no latency or resource violations (implied by execution)

    neutral:
    - decision executed
    - realized_savings is unknown OR
    - realized_savings is known but |savings| < SAVINGS_EPSILON

    conservative_skip:
    - decision was skipped due to safety or latency constraints
    - AND realized_savings is known and would have been > SAVINGS_EPSILON
    - If realized_savings is unknown, use "neutral" instead

    Args:
        execution_status: Status from ExecutionResult
        realized_savings: Computed savings (may be None)
        was_skipped_by_safety_or_latency: True if skipped due to constraints

    Returns:
        One of "good_decision", "neutral", "conservative_skip"
    """
    # Check if decision was executed
    executed = execution_status in ["submitted", "dry_run"]

    if executed:
        # Decision was executed
        if realized_savings is None:
            # Unknown savings -> neutral
            return "neutral"
        elif realized_savings >= SAVINGS_EPSILON:
            # Positive savings -> good decision
            return "good_decision"
        else:
            # Zero or negative savings -> neutral
            # (includes cases where abs(savings) < epsilon)
            return "neutral"
    else:
        # Decision was skipped or aborted
        if was_skipped_by_safety_or_latency:
            if realized_savings is not None and realized_savings > SAVINGS_EPSILON:
                # Skipped but would have saved money -> conservative skip
                return "conservative_skip"
            else:
                # Unknown savings or no savings -> neutral
                return "neutral"
        else:
            # Skipped for other reasons (e.g., delay exceeded)
            return "neutral"


def lookup_realized_price(
    region: str,
    start_time: datetime,
    runtime_hours: float,
    market_registry: Optional[Any] = None,
) -> Optional[float]:
    """Look up the average realized energy price for a job execution window.

    Queries the market registry for hourly prices covering the job's runtime
    and returns the simple average $/MWh.  Safe to call with market_registry=None
    — returns None without side effects.

    Args:
        region: Canonical region identifier (e.g. "us-east", "CAISO").
        start_time: When the job started (UTC datetime).
        runtime_hours: Duration of the job in fractional hours.
        market_registry: Optional provider registry.  If None, returns None.

    Returns:
        Average realized price in $/MWh, or None if unavailable.
    """
    if market_registry is None:
        return None
    try:
        from datetime import timedelta as _td
        end_time = start_time + _td(hours=runtime_hours)
        # Market registry may expose fetch_prices(region, start, end) → DataFrame
        price_df = market_registry.fetch_prices(region, start_time, end_time)
        if price_df is None:
            return None
        import pandas as pd
        if not isinstance(price_df, pd.DataFrame) or price_df.empty:
            return None
        if "price_per_mwh" not in price_df.columns:
            return None
        mean_price = float(price_df["price_per_mwh"].mean())
        return mean_price if not (mean_price != mean_price) else None  # NaN guard
    except Exception as exc:
        logger.debug(
            "lookup_realized_price failed for region=%s start=%s: %s",
            region,
            start_time,
            exc,
        )
        return None


class PostExecutionRecorder:
    """Records post-execution measurements for offline analysis.

    This recorder:
    - Creates PostExecutionRecord from decision + forecast + realized data
    - Computes forecast errors
    - Labels decision outcomes
    - Persists to JSONL file
    - Logs audit records

    All operations are failure-tolerant and never affect execution.

    Usage:
        recorder = PostExecutionRecorder()

        # After execution completes
        recorder.record(
            decision=optimized_decision,
            baseline_decision=baseline_decision,
            execution_result=result,
            config=execution_config,
            forecast=ForecastSnapshot(...),
            realized=RealizedOutcome(...),
        )
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        market_registry: Optional[Any] = None,
        max_size_bytes: Optional[int] = RotatingJSONLWriter.DEFAULT_MAX_SIZE_BYTES,
        max_records: Optional[int] = None,
        keep_rotated: int = 10,
    ):
        """Initialize the recorder.

        Args:
            output_path:    Path to JSONL output file (uses default if None).
            market_registry: Optional market registry for live realized-price
                lookups.  When provided, the recorder will attempt to populate
                ``realized_energy_price`` from the registry for decisions that
                don't already have it.
            max_size_bytes: Rotate the JSONL when it exceeds this size
                (default: 100 MB).  Set to None to disable size rotation.
            max_records:    Rotate after this many records since last rotation
                (default: None = disabled).
            keep_rotated:   How many rotated archive files to retain
                (default: 10).
        """
        path = output_path or get_default_post_execution_path()
        self._writer = RotatingJSONLWriter(
            path,
            max_size_bytes=max_size_bytes,
            max_records=max_records,
            keep_rotated=keep_rotated,
        )
        self._market_registry = market_registry

    def record(
        self,
        decision: ScheduleDecision,
        baseline_decision: Optional[ScheduleDecision],
        execution_result: ExecutionResult,
        config: ExecutionConfig,
        forecast: Optional[ForecastSnapshot] = None,
        realized: Optional[RealizedOutcome] = None,
        was_skipped_by_safety_or_latency: bool = False,
    ) -> Optional[PostExecutionRecord]:
        """Record post-execution measurement.

        Args:
            decision: The optimized scheduling decision
            baseline_decision: The baseline decision for comparison
            execution_result: Result from execution
            config: Execution configuration
            forecast: Forecast snapshot at decision time (optional)
            realized: Realized outcome after execution (optional)
            was_skipped_by_safety_or_latency: True if skipped by constraints

        Returns:
            The created PostExecutionRecord, or None if recording failed

        Note:
            This method never raises exceptions. Failures are logged
            at debug level to avoid cluttering production logs.
        """
        try:
            realized_to_use = realized or RealizedOutcome()

            # Populate realized_energy_price from the market registry when missing
            if (
                self._market_registry is not None
                and realized_to_use.realized_energy_price is None
            ):
                runtime = getattr(decision, "actual_runtime_hours", None) or 1.0
                looked_up = lookup_realized_price(
                    region=decision.region,
                    start_time=decision.start_time,
                    runtime_hours=runtime,
                    market_registry=self._market_registry,
                )
                if looked_up is not None:
                    from dataclasses import replace as _dc_replace
                    realized_to_use = _dc_replace(
                        realized_to_use,
                        realized_energy_price=looked_up,
                    )

            record = self._create_record(
                decision=decision,
                baseline_decision=baseline_decision,
                execution_result=execution_result,
                config=config,
                forecast=forecast or ForecastSnapshot(),
                realized=realized_to_use,
                was_skipped_by_safety_or_latency=was_skipped_by_safety_or_latency,
            )

            # Persist to JSONL
            self._writer.append(record.to_dict())

            # Log audit record
            self._log_audit(record)

            return record

        except Exception as e:
            logger.debug(f"Failed to record post-execution data: {e}")
            return None

    def _create_record(
        self,
        decision: ScheduleDecision,
        baseline_decision: Optional[ScheduleDecision],
        execution_result: ExecutionResult,
        config: ExecutionConfig,
        forecast: ForecastSnapshot,
        realized: RealizedOutcome,
        was_skipped_by_safety_or_latency: bool,
    ) -> PostExecutionRecord:
        """Create a PostExecutionRecord from inputs.

        This is the core logic for computing errors and labels.
        """
        # Generate stable decision ID
        decision_id = generate_decision_id(decision.job_id, decision.start_time)

        # Compute forecast errors
        errors = compute_forecast_errors(forecast, realized)

        # Compute realized savings
        realized_savings = compute_realized_savings(forecast, realized)

        # Label decision outcome
        outcome_label = label_decision_outcome(
            execution_status=execution_result.status,
            realized_savings=realized_savings,
            was_skipped_by_safety_or_latency=was_skipped_by_safety_or_latency,
        )

        return PostExecutionRecord(
            job_id=decision.job_id,
            decision_id=decision_id,
            region=decision.region,
            baseline_start_time=(
                baseline_decision.start_time if baseline_decision else None
            ),
            optimized_start_time=decision.start_time,
            realized_start_time=realized.realized_start_time,
            realized_energy_price=realized.realized_energy_price,
            realized_carbon_intensity=realized.realized_carbon_intensity,
            forecast_energy_cost_p50=forecast.energy_cost_p50,
            forecast_energy_cost_p90=forecast.energy_cost_p90,
            forecast_energy_cost_baseline=forecast.energy_cost_baseline,
            forecast_carbon_p50=forecast.carbon_p50,
            forecast_carbon_p90=forecast.carbon_p90,
            forecast_carbon_baseline=forecast.carbon_baseline,
            energy_cost_p50_error=errors["energy_cost_p50_error"],
            energy_cost_p90_covered=errors["energy_cost_p90_covered"],
            carbon_p50_error=errors["carbon_p50_error"],
            carbon_p90_covered=errors["carbon_p90_covered"],
            realized_savings=realized_savings,
            decision_outcome_label=outcome_label,
            execution_mode=config.mode,
            constraint_profile=config.constraint_profile,
            execution_status=execution_result.status,
        )

    def _log_audit(self, record: PostExecutionRecord) -> None:
        """Log structured audit record for post-execution measurement."""
        audit_record = {
            "event": "post_execution_recorded",
            "job_id": record.job_id,
            "decision_id": record.decision_id,
            "decision_outcome": record.decision_outcome_label,
            "realized_savings": record.realized_savings,
            "energy_cost_p50_error": record.energy_cost_p50_error,
            "energy_cost_p90_covered": record.energy_cost_p90_covered,
            "carbon_p50_error": record.carbon_p50_error,
            "carbon_p90_covered": record.carbon_p90_covered,
        }
        logger.info(f"AUDIT: {json.dumps(audit_record)}")


# Inline tests
if __name__ == "__main__":
    from datetime import timedelta
    import tempfile

    print("=" * 60)
    print("PostExecutionRecorder Inline Tests")
    print("=" * 60)

    # Test 1: generate_decision_id is deterministic
    print("\n[Test 1] Decision ID is deterministic")
    now = datetime.utcnow()
    id1 = generate_decision_id("job-001", now)
    id2 = generate_decision_id("job-001", now)
    id3 = generate_decision_id("job-002", now)
    assert id1 == id2, "Same inputs should produce same ID"
    assert id1 != id3, "Different job_id should produce different ID"
    assert len(id1) == 16, "ID should be 16 characters"
    print(f"  PASSED: id1={id1}, id2={id2}")

    # Test 2: compute_forecast_errors with full data
    print("\n[Test 2] Forecast error computation - full data")
    forecast = ForecastSnapshot(
        energy_cost_p50=100.0,
        energy_cost_p90=120.0,
        energy_cost_baseline=150.0,
        carbon_p50=50.0,
        carbon_p90=60.0,
        carbon_baseline=80.0,
    )
    realized = RealizedOutcome(
        realized_energy_cost=110.0,
        realized_carbon=55.0,
    )
    errors = compute_forecast_errors(forecast, realized)
    assert errors["energy_cost_p50_error"] == 10.0  # 110 - 100
    assert errors["energy_cost_p90_covered"] is True  # 110 <= 120
    assert errors["carbon_p50_error"] == 5.0  # 55 - 50
    assert errors["carbon_p90_covered"] is True  # 55 <= 60
    print(f"  PASSED: errors={errors}")

    # Test 3: compute_forecast_errors with missing data
    print("\n[Test 3] Forecast error computation - missing data")
    empty_forecast = ForecastSnapshot()
    empty_realized = RealizedOutcome()
    errors = compute_forecast_errors(empty_forecast, empty_realized)
    assert all(v is None for v in errors.values())
    print(f"  PASSED: All errors are None when data missing")

    # Test 4: compute_realized_savings
    print("\n[Test 4] Realized savings computation")
    savings = compute_realized_savings(forecast, realized)
    assert savings == 40.0  # 150 - 110
    print(f"  PASSED: savings={savings}")

    # Test 5: compute_realized_savings with missing baseline
    print("\n[Test 5] Realized savings - missing baseline")
    no_baseline = ForecastSnapshot(energy_cost_p50=100.0)
    savings = compute_realized_savings(no_baseline, realized)
    assert savings is None
    print(f"  PASSED: savings=None when baseline missing")

    # Test 6: label_decision_outcome - good_decision
    print("\n[Test 6] Decision labeling - good_decision")
    label = label_decision_outcome(
        execution_status="submitted",
        realized_savings=10.0,
        was_skipped_by_safety_or_latency=False,
    )
    assert label == "good_decision"
    print(f"  PASSED: label={label}")

    # Test 7: label_decision_outcome - neutral (unknown savings)
    print("\n[Test 7] Decision labeling - neutral (unknown savings)")
    label = label_decision_outcome(
        execution_status="submitted",
        realized_savings=None,
        was_skipped_by_safety_or_latency=False,
    )
    assert label == "neutral"
    print(f"  PASSED: label={label}")

    # Test 8: label_decision_outcome - neutral (zero savings)
    print("\n[Test 8] Decision labeling - neutral (zero savings)")
    label = label_decision_outcome(
        execution_status="dry_run",
        realized_savings=0.005,  # Below epsilon
        was_skipped_by_safety_or_latency=False,
    )
    assert label == "neutral"
    print(f"  PASSED: label={label}")

    # Test 9: label_decision_outcome - conservative_skip
    print("\n[Test 9] Decision labeling - conservative_skip")
    label = label_decision_outcome(
        execution_status="skipped",
        realized_savings=50.0,
        was_skipped_by_safety_or_latency=True,
    )
    assert label == "conservative_skip"
    print(f"  PASSED: label={label}")

    # Test 10: label_decision_outcome - neutral (skipped, unknown savings)
    print("\n[Test 10] Decision labeling - neutral (skipped, unknown)")
    label = label_decision_outcome(
        execution_status="skipped",
        realized_savings=None,
        was_skipped_by_safety_or_latency=True,
    )
    assert label == "neutral"
    print(f"  PASSED: label={label}")

    # Test 11: Full recording flow
    print("\n[Test 11] Full recording flow")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/test_records.jsonl"
        recorder = PostExecutionRecorder(output_path)

        decision = ScheduleDecision(
            job_id="job-test-001",
            start_time=now + timedelta(hours=1),
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        )
        baseline = ScheduleDecision(
            job_id="job-test-001",
            start_time=now,
            region="us-east",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        )
        result = ExecutionResult(
            job_id="job-test-001",
            submitted=True,
            aws_job_id="aws-123",
            region="us-west",
            submit_time=now,
            status="submitted",
        )
        config = ExecutionConfig(mode="live", constraint_profile="batch_optimized")

        record = recorder.record(
            decision=decision,
            baseline_decision=baseline,
            execution_result=result,
            config=config,
            forecast=forecast,
            realized=realized,
        )

        assert record is not None
        assert record.job_id == "job-test-001"
        assert record.decision_outcome_label == "good_decision"
        assert record.realized_savings == 40.0
        assert record.execution_mode == "live"
        assert record.constraint_profile == "batch_optimized"
        print(f"  PASSED: record created with label={record.decision_outcome_label}")

        # Verify persistence
        from aurelius.data.persistence import JSONLWriter
        writer = JSONLWriter(output_path)
        records = writer.read_all()
        assert len(records) == 1
        assert records[0]["job_id"] == "job-test-001"
        print(f"  PASSED: record persisted to JSONL")

    # Test 12: Recording with minimal data (all optional)
    print("\n[Test 12] Recording with minimal data")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/minimal_records.jsonl"
        recorder = PostExecutionRecorder(output_path)

        decision = ScheduleDecision(
            job_id="job-minimal",
            start_time=now,
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=1.0,
        )
        result = ExecutionResult(
            job_id="job-minimal",
            submitted=False,
            aws_job_id=None,
            region="us-west",
            submit_time=now,
            status="dry_run",
        )
        config = ExecutionConfig()

        record = recorder.record(
            decision=decision,
            baseline_decision=None,
            execution_result=result,
            config=config,
            # No forecast or realized data
        )

        assert record is not None
        assert record.decision_outcome_label == "neutral"
        assert record.realized_savings is None
        print(f"  PASSED: minimal record created")

    # Test 13: p90 coverage edge case (exactly at p90)
    print("\n[Test 13] p90 coverage - exactly at threshold")
    forecast_exact = ForecastSnapshot(
        energy_cost_p50=100.0,
        energy_cost_p90=110.0,
    )
    realized_exact = RealizedOutcome(
        realized_energy_cost=110.0,  # Exactly at p90
    )
    errors = compute_forecast_errors(forecast_exact, realized_exact)
    assert errors["energy_cost_p90_covered"] is True  # 110 <= 110
    print(f"  PASSED: exactly at p90 is covered")

    # Test 14: p90 coverage - exceeded
    print("\n[Test 14] p90 coverage - exceeded")
    realized_exceeded = RealizedOutcome(
        realized_energy_cost=111.0,  # Above p90
    )
    errors = compute_forecast_errors(forecast_exact, realized_exceeded)
    assert errors["energy_cost_p90_covered"] is False
    print(f"  PASSED: above p90 is not covered")

    # Test 15: Negative realized savings (bad decision)
    print("\n[Test 15] Negative savings labeling")
    label = label_decision_outcome(
        execution_status="submitted",
        realized_savings=-20.0,  # Lost money
        was_skipped_by_safety_or_latency=False,
    )
    assert label == "neutral"  # Not good, but not a conservative_skip
    print(f"  PASSED: negative savings -> neutral")

    print("\n" + "=" * 60)
    print("All 15 tests passed!")
    print("=" * 60)
