"""AWS Batch execution adapter for Aurelius.

This module submits jobs to AWS Batch based on Aurelius scheduling decisions.
Aurelius decides WHEN / WHERE / HOW - AWS Batch handles execution.

IMPORTANT:
- Default mode is DRY RUN (no AWS calls)
- Live mode requires explicit opt-in
- Kill switch can abort all execution instantly
- Every execution attempt is logged for audit

Environment Variables:
    AWS_REGION: AWS region for Batch operations
    AWS_BATCH_JOB_QUEUE: Job queue name (can be region-specific)
    AWS_BATCH_JOB_DEFINITION: Job definition ARN or name
    AURELIUS_KILL_SWITCH: Set to "true" to abort all execution
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

from ..models import ScheduleDecision
from .base import (
    Executor,
    ExecutionConfig,
    ExecutionResult,
    log_execution_audit,
)

logger = logging.getLogger(__name__)


# Power to vCPU ratio (same as metrics.py for consistency)
KW_PER_VCPU = 0.00625


def estimate_vcpus_from_power(power_kw: float, power_fraction: float) -> int:
    """Estimate vCPU count from power consumption.

    Args:
        power_kw: Base power in kW
        power_fraction: Power throttle fraction (0-1)

    Returns:
        Estimated vCPU count (minimum 1)
    """
    effective_power = power_kw * power_fraction
    vcpus = int(effective_power / KW_PER_VCPU)
    return max(1, vcpus)


def estimate_memory_from_vcpus(vcpus: int) -> int:
    """Estimate memory from vCPU count.

    Uses a ratio of 2GB per vCPU as a reasonable default.

    Args:
        vcpus: Number of vCPUs

    Returns:
        Memory in MB
    """
    return vcpus * 2048  # 2GB per vCPU


class AWSBatchExecutor(Executor):
    """Executor that submits jobs to AWS Batch.

    This is a thin adapter that:
    - Reads AWS config from environment variables
    - Validates guardrails before submission
    - Submits jobs with appropriate vCPU/memory settings
    - Tags jobs with submitted_by="aurelius"
    - Returns structured ExecutionResult objects

    It does NOT:
    - Modify existing jobs
    - Cancel non-Aurelius jobs
    - Touch autoscaling or compute environments
    - Manage IAM or permissions
    - Implement custom retry logic
    """

    def __init__(
        self,
        job_queue: Optional[str] = None,
        job_definition: Optional[str] = None,
        region: Optional[str] = None,
        region_queue_map: Optional[dict[str, str]] = None,
    ):
        """Initialize the AWS Batch executor.

        Args:
            job_queue: Default job queue name (or from AWS_BATCH_JOB_QUEUE)
            job_definition: Job definition ARN/name (or from AWS_BATCH_JOB_DEFINITION)
            region: AWS region (or from AWS_REGION)
            region_queue_map: Optional mapping of Aurelius regions to AWS job queues
        """
        self.default_job_queue = job_queue or os.environ.get("AWS_BATCH_JOB_QUEUE", "")
        self.job_definition = job_definition or os.environ.get("AWS_BATCH_JOB_DEFINITION", "")
        self.aws_region = region or os.environ.get("AWS_REGION", "us-east-1")

        # Map Aurelius regions to AWS job queues
        # Default: use the same queue for all regions
        self.region_queue_map = region_queue_map or {}

        # Lazy-load boto3 client
        self._batch_client = None

    def _get_batch_client(self):
        """Get or create the boto3 Batch client."""
        if self._batch_client is None:
            try:
                import boto3
                self._batch_client = boto3.client("batch", region_name=self.aws_region)
            except ImportError:
                raise RuntimeError("boto3 is required for AWS Batch execution")
        return self._batch_client

    def _get_job_queue(self, decision_region: str) -> str:
        """Get the job queue for a given Aurelius region.

        Args:
            decision_region: The Aurelius region from the scheduling decision

        Returns:
            AWS Batch job queue name
        """
        return self.region_queue_map.get(decision_region, self.default_job_queue)

    def execute(
        self,
        decisions: list[ScheduleDecision],
        config: ExecutionConfig,
    ) -> list[ExecutionResult]:
        """Execute scheduling decisions by submitting jobs to AWS Batch.

        Execution is sequential (no parallelization).

        For each decision:
        1. Check kill switch
        2. Validate guardrails (max_delay, max_power_reduction)
        3. If dry_run: log and return without AWS calls
        4. If live: wait until start_time, then submit to AWS Batch

        Args:
            decisions: List of ScheduleDecision objects from the optimizer
            config: ExecutionConfig controlling behavior

        Returns:
            List of ExecutionResult objects, one per decision
        """
        results = []

        logger.info(f"Starting execution: {len(decisions)} decisions, mode={config.mode}")

        for decision in decisions:
            result = self._execute_single(decision, config)
            results.append(result)

            # Log audit record for every execution attempt
            log_execution_audit(decision, result, config)

        # Summary logging
        submitted_count = sum(1 for r in results if r.status == "submitted")
        dry_run_count = sum(1 for r in results if r.status == "dry_run")
        skipped_count = sum(1 for r in results if r.status == "skipped")
        aborted_count = sum(1 for r in results if r.status == "aborted")

        logger.info(
            f"Execution complete: {submitted_count} submitted, "
            f"{dry_run_count} dry_run, {skipped_count} skipped, {aborted_count} aborted"
        )

        return results

    def _execute_single(
        self,
        decision: ScheduleDecision,
        config: ExecutionConfig,
    ) -> ExecutionResult:
        """Execute a single scheduling decision.

        Args:
            decision: The scheduling decision to execute
            config: Execution configuration

        Returns:
            ExecutionResult describing what happened
        """
        now = datetime.utcnow()

        # GUARDRAIL 1: Kill switch
        if config.is_kill_switch_active():
            logger.warning(f"KILL SWITCH ACTIVE: Aborting execution for {decision.job_id}")
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=now,
                status="aborted",
                reason=f"Kill switch active ({config.kill_switch_env_var}=true)",
            )

        # GUARDRAIL 2: Max delay check
        if decision.start_time < now:
            delay_hours = (now - decision.start_time).total_seconds() / 3600
            if delay_hours > config.max_delay_hours:
                logger.warning(
                    f"Skipping {decision.job_id}: delay {delay_hours:.2f}h exceeds "
                    f"max_delay_hours={config.max_delay_hours}"
                )
                return ExecutionResult(
                    job_id=decision.job_id,
                    submitted=False,
                    aws_job_id=None,
                    region=decision.region,
                    submit_time=now,
                    status="skipped",
                    reason=f"Delay {delay_hours:.2f}h exceeds max {config.max_delay_hours}h",
                )

        # GUARDRAIL 3: Max power reduction check
        power_reduction_pct = (1.0 - decision.power_fraction) * 100
        if power_reduction_pct > config.max_power_reduction_pct:
            logger.warning(
                f"Skipping {decision.job_id}: power reduction {power_reduction_pct:.1f}% "
                f"exceeds max_power_reduction_pct={config.max_power_reduction_pct}"
            )
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=now,
                status="skipped",
                reason=f"Power reduction {power_reduction_pct:.1f}% exceeds max {config.max_power_reduction_pct}%",
            )

        # DRY RUN MODE: Log what would happen, no AWS calls
        if config.is_dry_run():
            logger.info(
                f"[DRY RUN] Would submit {decision.job_id} to {decision.region} "
                f"at {decision.start_time.isoformat()} with power_fraction={decision.power_fraction}"
            )
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=now,
                status="dry_run",
                reason="Dry run mode - no AWS calls made",
            )

        # LIVE MODE: Wait and submit
        return self._submit_to_aws_batch(decision, config)

    def _submit_to_aws_batch(
        self,
        decision: ScheduleDecision,
        config: ExecutionConfig,
    ) -> ExecutionResult:
        """Submit a job to AWS Batch.

        Waits until decision.start_time before submitting.

        Args:
            decision: The scheduling decision
            config: Execution configuration

        Returns:
            ExecutionResult with AWS job ID if successful
        """
        now = datetime.utcnow()

        # Wait until start_time if it's in the future
        if decision.start_time > now:
            wait_seconds = (decision.start_time - now).total_seconds()
            # Cap wait to avoid blocking forever (re-check delay on next iteration)
            max_wait = config.max_delay_hours * 3600
            if wait_seconds > max_wait:
                logger.warning(
                    f"Wait time {wait_seconds:.0f}s exceeds max, capping to {max_wait:.0f}s"
                )
                wait_seconds = max_wait

            logger.info(f"Waiting {wait_seconds:.1f}s until start_time for {decision.job_id}")
            time.sleep(wait_seconds)

        # Re-check kill switch after waiting
        if config.is_kill_switch_active():
            logger.warning(f"KILL SWITCH ACTIVE after wait: Aborting {decision.job_id}")
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="aborted",
                reason="Kill switch activated during wait",
            )

        # Build job submission parameters
        job_queue = self._get_job_queue(decision.region)
        job_name = f"aurelius-{decision.job_id}"

        # Estimate resources from power/throttling
        # Note: This is a placeholder - real implementation would use job metadata
        vcpus = estimate_vcpus_from_power(100.0, decision.power_fraction)  # Assume 100kW base
        memory = estimate_memory_from_vcpus(vcpus)

        submit_params = {
            "jobName": job_name,
            "jobQueue": job_queue,
            "jobDefinition": self.job_definition,
            "containerOverrides": {
                "vcpus": vcpus,
                "memory": memory,
            },
            "tags": {
                "submitted_by": "aurelius",
                "aurelius_job_id": decision.job_id,
                "aurelius_region": decision.region,
                "aurelius_power_fraction": str(decision.power_fraction),
            },
        }

        try:
            client = self._get_batch_client()

            logger.info(f"Submitting job {decision.job_id} to queue {job_queue}")
            response = client.submit_job(**submit_params)

            aws_job_id = response.get("jobId")
            logger.info(f"Successfully submitted {decision.job_id} as AWS job {aws_job_id}")

            return ExecutionResult(
                job_id=decision.job_id,
                submitted=True,
                aws_job_id=aws_job_id,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="submitted",
                reason=f"Submitted to queue {job_queue}",
            )

        except Exception as e:
            # Log error but don't raise - return result indicating failure
            logger.error(f"Failed to submit {decision.job_id}: {e}")
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="skipped",
                reason=f"AWS submission failed: {str(e)}",
            )
