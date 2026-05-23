"""Base abstractions for Aurelius execution adapters.

Aurelius owns decisions, NOT infrastructure.
Execution is optional, gated, and reversible.
Default mode is DRY RUN.

This module defines the core interfaces that all execution adapters
must implement. The execution layer is intentionally thin - it only
translates Aurelius scheduling decisions into infrastructure actions.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from ..models import ScheduleDecision

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of attempting to execute a single scheduling decision.

    Attributes:
        job_id: The Aurelius job ID from the scheduling decision
        submitted: True ONLY if an actual submission call was made successfully
        aws_job_id: The AWS job ID if submission succeeded, None otherwise
        region: The target region for execution
        submit_time: When the execution was attempted
        status: One of "dry_run", "submitted", "skipped", "aborted"
        reason: Human-readable explanation of the result
    """
    job_id: str
    submitted: bool
    aws_job_id: Optional[str]
    region: str
    submit_time: datetime
    status: Literal["dry_run", "submitted", "skipped", "aborted"]
    reason: Optional[str] = None

    def to_audit_dict(self) -> dict:
        """Convert to dictionary for audit logging."""
        return {
            "job_id": self.job_id,
            "submitted": self.submitted,
            "aws_job_id": self.aws_job_id,
            "region": self.region,
            "submit_time": self.submit_time.isoformat(),
            "status": self.status,
            "reason": self.reason,
        }

    def to_audit_json(self) -> str:
        """Convert to JSON string for audit logging."""
        return json.dumps(self.to_audit_dict())


@dataclass
class ExecutionConfig:
    """Configuration for execution behavior.

    Attributes:
        mode: "dry_run" (default) or "live"
        max_delay_hours: Maximum hours past decision.start_time to still execute
        max_power_reduction_pct: Maximum allowed power reduction (0-100)
        kill_switch_env_var: Environment variable name for kill switch
        constraint_profile: "batch_optimized" (default) or "latency_safe"
        latency_slack_threshold_hours: Max allowed start time deviation for latency_safe (default 0.05 = 3 min)
    """
    mode: Literal["dry_run", "live"] = "dry_run"
    max_delay_hours: float = 2.0
    max_power_reduction_pct: float = 50.0
    kill_switch_env_var: str = "AURELIUS_KILL_SWITCH"
    constraint_profile: Literal["batch_optimized", "latency_safe"] = "batch_optimized"
    latency_slack_threshold_hours: float = 0.05  # ~3 minutes

    def is_kill_switch_active(self) -> bool:
        """Check if the kill switch is enabled."""
        value = os.environ.get(self.kill_switch_env_var, "").lower()
        return value == "true"

    def is_dry_run(self) -> bool:
        """Check if running in dry run mode."""
        return self.mode == "dry_run"

    def is_live(self) -> bool:
        """Check if running in live mode."""
        return self.mode == "live"

    def is_latency_safe(self) -> bool:
        """Check if running in latency_safe constraint profile."""
        return self.constraint_profile == "latency_safe"

    def is_batch_optimized(self) -> bool:
        """Check if running in batch_optimized constraint profile (default)."""
        return self.constraint_profile == "batch_optimized"


class Executor(ABC):
    """Abstract base class for execution adapters.

    All execution adapters must implement the execute method.
    Execution adapters translate Aurelius scheduling decisions
    into infrastructure actions (e.g., AWS Batch job submissions).

    Execution is:
    - Optional: Decisions can be made without execution
    - Gated: Default is dry_run, live requires explicit opt-in
    - Reversible: No permanent infrastructure changes
    - Auditable: Every decision is logged with full context
    """

    @abstractmethod
    def execute(
        self,
        decisions: list[ScheduleDecision],
        config: ExecutionConfig,
    ) -> list[ExecutionResult]:
        """Execute a list of scheduling decisions.

        Args:
            decisions: List of ScheduleDecision objects from the optimizer
            config: ExecutionConfig controlling execution behavior

        Returns:
            List of ExecutionResult objects, one per decision

        Behavior:
            - In dry_run mode: No actual infrastructure calls are made
            - In live mode: Jobs are submitted to the execution backend
            - Kill switch: If active, all executions abort immediately
            - Guardrails: Decisions exceeding limits are skipped
        """
        pass


def log_execution_audit(
    decision: ScheduleDecision,
    result: ExecutionResult,
    config: ExecutionConfig,
) -> None:
    """Log a structured audit record for an execution attempt.

    Every execution attempt (including dry_run, skipped, aborted)
    is logged with full context for auditability.

    Args:
        decision: The original scheduling decision
        result: The execution result
        config: The execution configuration
    """
    audit_record = {
        "event": "execution_attempt",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": config.mode,
        "decision": {
            "job_id": decision.job_id,
            "start_time": decision.start_time.isoformat(),
            "end_time": decision.end_time.isoformat(),
            "region": decision.region,
            "power_fraction": decision.power_fraction,
            "actual_runtime_hours": decision.actual_runtime_hours,
        },
        "result": result.to_audit_dict(),
    }

    # Log as JSON for structured logging / parsing
    logger.info(f"AUDIT: {json.dumps(audit_record)}")
