"""Slurm execution adapter for Aurelius.

This module submits jobs to Slurm via sbatch based on Aurelius scheduling decisions.
Aurelius decides WHEN / WHERE / HOW - Slurm handles execution.

IMPORTANT:
- Default mode is DRY RUN (no sbatch calls)
- Live mode requires explicit opt-in
- Kill switch can abort all execution instantly
- Every execution attempt is logged for audit

Environment Variables:
    SLURM_SCRIPT_PATH: Path to the job script to submit (REQUIRED)
    SLURM_PARTITION_MAP: JSON mapping region -> partition (e.g. '{"us-west": "gpu"}')
    SLURM_CONSTRAINT_MAP: JSON mapping region -> constraint (e.g. '{"us-west": "v100"}')
    SLURM_ACCOUNT: Slurm account for job submission (optional)
    SLURM_QOS: Quality of service level (optional)
    AURELIUS_KILL_SWITCH: Set to "true" to abort all execution
"""

import json
import logging
import os
import re
import subprocess
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


# Resource baselines
BASELINE_CPU = 4
BASELINE_MEMORY_GB = 16

# Resource clamps
MIN_CPU = 1
MAX_CPU = 64
MIN_MEMORY_GB = 1
MAX_MEMORY_GB = 256

# Sleep chunk size for interruptible waits
SLEEP_CHUNK_SECONDS = 30.0


def calculate_resources(power_fraction: float) -> dict:
    """Calculate CPU and memory resources from power fraction.

    Args:
        power_fraction: Power throttle fraction (0-1)

    Returns:
        Dict with cpus (int) and memory_gb (int) values
    """
    cpus = int(BASELINE_CPU * power_fraction)
    memory_gb = int(BASELINE_MEMORY_GB * power_fraction)

    # Clamp values
    cpus = max(MIN_CPU, min(MAX_CPU, cpus))
    memory_gb = max(MIN_MEMORY_GB, min(MAX_MEMORY_GB, memory_gb))

    return {
        "cpus": cpus,
        "memory_gb": memory_gb,
    }


def format_slurm_time(dt: datetime) -> str:
    """Format datetime for Slurm --begin option.

    Slurm accepts: YYYY-MM-DDTHH:MM:SS

    Args:
        dt: Datetime to format

    Returns:
        Slurm-compatible time string
    """
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def parse_job_id(sbatch_output: str) -> Optional[str]:
    """Parse Slurm job ID from sbatch output.

    Expected format: "Submitted batch job 12345678"

    Args:
        sbatch_output: stdout from sbatch command

    Returns:
        Job ID string or None if not found
    """
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if match:
        return match.group(1)
    return None


class SlurmExecutor(Executor):
    """Executor that submits jobs to Slurm via sbatch.

    This is a thin adapter that:
    - Reads Slurm config from environment variables
    - Validates guardrails before submission
    - Submits jobs with --begin for future scheduling
    - Uses partition/constraint for regional placement
    - Returns structured ExecutionResult objects

    It does NOT:
    - Generate or modify job scripts
    - Cancel or modify existing jobs
    - Implement custom retry logic
    - Manage queues or partitions
    """

    def __init__(
        self,
        script_path: Optional[str] = None,
        partition_map: Optional[dict[str, str]] = None,
        constraint_map: Optional[dict[str, str]] = None,
        account: Optional[str] = None,
        qos: Optional[str] = None,
    ):
        """Initialize the Slurm executor.

        Args:
            script_path: Path to job script (or from SLURM_SCRIPT_PATH)
            partition_map: Region to partition mapping (or from SLURM_PARTITION_MAP)
            constraint_map: Region to constraint mapping (or from SLURM_CONSTRAINT_MAP)
            account: Slurm account (or from SLURM_ACCOUNT)
            qos: Quality of service (or from SLURM_QOS)
        """
        self.script_path = script_path or os.environ.get("SLURM_SCRIPT_PATH", "")
        self.account = account or os.environ.get("SLURM_ACCOUNT", "")
        self.qos = qos or os.environ.get("SLURM_QOS", "")

        # Parse partition map from JSON env var
        partition_map_json = os.environ.get("SLURM_PARTITION_MAP", "{}")
        try:
            self.partition_map = partition_map or json.loads(partition_map_json)
        except json.JSONDecodeError:
            logger.warning(f"Invalid SLURM_PARTITION_MAP JSON: {partition_map_json}")
            self.partition_map = {}

        # Parse constraint map from JSON env var
        constraint_map_json = os.environ.get("SLURM_CONSTRAINT_MAP", "{}")
        try:
            self.constraint_map = constraint_map or json.loads(constraint_map_json)
        except json.JSONDecodeError:
            logger.warning(f"Invalid SLURM_CONSTRAINT_MAP JSON: {constraint_map_json}")
            self.constraint_map = {}

    def _get_partition(self, region: str) -> Optional[str]:
        """Get the Slurm partition for a region.

        Args:
            region: Aurelius region from decision

        Returns:
            Partition name or None if not mapped
        """
        return self.partition_map.get(region)

    def _get_constraint(self, region: str) -> Optional[str]:
        """Get the Slurm constraint for a region.

        Args:
            region: Aurelius region from decision

        Returns:
            Constraint string or None if not mapped
        """
        return self.constraint_map.get(region)

    def execute(
        self,
        decisions: list[ScheduleDecision],
        config: ExecutionConfig,
    ) -> list[ExecutionResult]:
        """Execute scheduling decisions by submitting jobs to Slurm.

        Execution is sequential (no parallelization).

        For each decision:
        1. Check kill switch
        2. Validate guardrails (max_delay, max_power_reduction)
        3. If dry_run: log and return without sbatch calls
        4. If live: wait until start_time, then submit via sbatch

        Args:
            decisions: List of ScheduleDecision objects from the optimizer
            config: ExecutionConfig controlling behavior

        Returns:
            List of ExecutionResult objects, one per decision
        """
        results = []

        logger.info(f"Starting execution: {len(decisions)} decisions, mode={config.mode}")

        # Check for missing script upfront
        if not self.script_path:
            logger.error("SLURM_SCRIPT_PATH is missing - all jobs will be skipped")

        for decision in decisions:
            # Check kill switch before each job
            if config.is_kill_switch_active():
                # Abort ALL remaining decisions
                result = self._create_aborted_result(decision, config)
                results.append(result)
                log_execution_audit(decision, result, config)
                self._log_audit_json(decision, result, config)
                continue

            result = self._execute_single(decision, config)
            results.append(result)

            # Log audit record for every execution attempt
            log_execution_audit(decision, result, config)
            self._log_audit_json(decision, result, config)

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

    def _create_aborted_result(
        self,
        decision: ScheduleDecision,
        config: ExecutionConfig,
    ) -> ExecutionResult:
        """Create an aborted result for kill switch."""
        logger.warning(f"KILL SWITCH ACTIVE: Aborting execution for {decision.job_id}")
        return ExecutionResult(
            job_id=decision.job_id,
            submitted=False,
            aws_job_id=None,
            region=decision.region,
            submit_time=datetime.utcnow(),
            status="aborted",
            reason=f"Kill switch active ({config.kill_switch_env_var}=true)",
        )

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

        # Check for missing script
        if not self.script_path:
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=now,
                status="skipped",
                reason="SLURM_SCRIPT_PATH missing",
            )

        # GUARDRAIL: Max delay check
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

        # GUARDRAIL: Max power reduction check
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

        # Build sbatch command for logging/submission
        sbatch_cmd = self._build_sbatch_command(decision)

        # DRY RUN MODE: Log what would happen, no sbatch calls
        if config.is_dry_run():
            logger.info(
                f"[DRY RUN] Would run: {' '.join(sbatch_cmd)}"
            )
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="dry_run",
                reason="Dry run mode - no sbatch calls made",
            )

        # LIVE MODE: Submit via sbatch
        return self._submit_to_slurm(decision, config, sbatch_cmd)

    def _build_sbatch_command(self, decision: ScheduleDecision) -> list[str]:
        """Build the sbatch command line.

        Args:
            decision: The scheduling decision

        Returns:
            List of command arguments for subprocess
        """
        resources = calculate_resources(decision.power_fraction)

        cmd = ["sbatch"]

        # Job name
        job_name = f"aurelius-{decision.job_id}"
        cmd.extend(["--job-name", job_name])

        # Resources
        cmd.extend(["--cpus-per-task", str(resources["cpus"])])
        cmd.extend(["--mem", f"{resources['memory_gb']}G"])

        # Scheduled start time (use --begin for future scheduling)
        if decision.start_time > datetime.utcnow():
            begin_time = format_slurm_time(decision.start_time)
            cmd.extend(["--begin", begin_time])

        # Regional placement via partition
        partition = self._get_partition(decision.region)
        if partition:
            cmd.extend(["--partition", partition])

        # Regional placement via constraint
        constraint = self._get_constraint(decision.region)
        if constraint:
            cmd.extend(["--constraint", constraint])

        # Optional account
        if self.account:
            cmd.extend(["--account", self.account])

        # Optional QoS
        if self.qos:
            cmd.extend(["--qos", self.qos])

        # Environment variables to pass to job
        env_exports = [
            f"AURELIUS_JOB_ID={decision.job_id}",
            f"AURELIUS_REGION={decision.region}",
            f"AURELIUS_POWER_FRACTION={decision.power_fraction}",
            f"AURELIUS_RUNTIME_HOURS={decision.actual_runtime_hours}",
            f"AURELIUS_SCHEDULED_START_TIME={decision.start_time.isoformat()}",
        ]
        cmd.extend(["--export", ",".join(env_exports)])

        # The script to submit (MUST be last)
        cmd.append(self.script_path)

        return cmd

    def _submit_to_slurm(
        self,
        decision: ScheduleDecision,
        config: ExecutionConfig,
        sbatch_cmd: list[str],
    ) -> ExecutionResult:
        """Submit a job to Slurm via sbatch.

        Args:
            decision: The scheduling decision
            config: Execution configuration
            sbatch_cmd: Pre-built sbatch command

        Returns:
            ExecutionResult with Slurm job ID if successful
        """
        now = datetime.utcnow()

        # Wait until start_time if it's in the future (chunked for kill switch checks)
        # Note: We still wait because --begin is relative and we want precise timing
        if decision.start_time > now:
            wait_seconds = (decision.start_time - now).total_seconds()

            # Cap wait to avoid blocking forever
            max_wait = config.max_delay_hours * 3600
            if wait_seconds > max_wait:
                logger.warning(
                    f"Wait time {wait_seconds:.0f}s exceeds max, capping to {max_wait:.0f}s"
                )
                wait_seconds = max_wait

            logger.info(f"Waiting {wait_seconds:.1f}s until start_time for {decision.job_id}")

            # Chunked sleep with kill switch checks
            remaining = wait_seconds
            while remaining > 0:
                sleep_time = min(SLEEP_CHUNK_SECONDS, remaining)
                time.sleep(sleep_time)
                remaining -= sleep_time

                # Re-check kill switch after each chunk
                if config.is_kill_switch_active():
                    logger.warning(f"KILL SWITCH ACTIVE during wait: Aborting {decision.job_id}")
                    return ExecutionResult(
                        job_id=decision.job_id,
                        submitted=False,
                        aws_job_id=None,
                        region=decision.region,
                        submit_time=datetime.utcnow(),
                        status="aborted",
                        reason="Kill switch activated during wait",
                    )

        # Final kill switch check before submission
        if config.is_kill_switch_active():
            logger.warning(f"KILL SWITCH ACTIVE before submit: Aborting {decision.job_id}")
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="aborted",
                reason="Kill switch activated before submission",
            )

        # Submit via sbatch
        try:
            logger.info(f"Submitting job: {' '.join(sbatch_cmd)}")
            result = subprocess.run(
                sbatch_cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout for sbatch
            )

            if result.returncode != 0:
                logger.error(f"sbatch failed: {result.stderr}")
                return ExecutionResult(
                    job_id=decision.job_id,
                    submitted=False,
                    aws_job_id=None,
                    region=decision.region,
                    submit_time=datetime.utcnow(),
                    status="skipped",
                    reason=f"sbatch failed: {result.stderr.strip()}",
                )

            # Parse job ID from output
            slurm_job_id = parse_job_id(result.stdout)
            if not slurm_job_id:
                logger.warning(f"Could not parse job ID from: {result.stdout}")
                slurm_job_id = "unknown"

            logger.info(f"Successfully submitted {decision.job_id} as Slurm job {slurm_job_id}")

            return ExecutionResult(
                job_id=decision.job_id,
                submitted=True,
                aws_job_id=slurm_job_id,  # Reusing field for Slurm job ID
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="submitted",
                reason=f"Submitted as Slurm job {slurm_job_id}",
            )

        except subprocess.TimeoutExpired:
            logger.error(f"sbatch timed out for {decision.job_id}")
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="skipped",
                reason="sbatch command timed out",
            )
        except Exception as e:
            logger.error(f"Failed to submit {decision.job_id}: {e}")
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="skipped",
                reason=f"Slurm submission failed: {str(e)}",
            )

    def _log_audit_json(
        self,
        decision: ScheduleDecision,
        result: ExecutionResult,
        config: ExecutionConfig,
    ) -> None:
        """Log a structured JSON audit record with Slurm-specific fields.

        Args:
            decision: The scheduling decision
            result: The execution result
            config: The execution configuration
        """
        resources = calculate_resources(decision.power_fraction)

        audit_record = {
            "event": "slurm_execution_attempt",
            "timestamp": datetime.utcnow().isoformat(),
            "mode": config.mode,
            "decision": {
                "job_id": decision.job_id,
                "start_time": decision.start_time.isoformat(),
                "region": decision.region,
                "power_fraction": decision.power_fraction,
                "runtime_hours": decision.actual_runtime_hours,
            },
            "slurm": {
                "script_path": self.script_path,
                "partition": self._get_partition(decision.region),
                "constraint": self._get_constraint(decision.region),
                "cpus": resources["cpus"],
                "memory_gb": resources["memory_gb"],
                "account": self.account,
                "qos": self.qos,
            },
            "result": {
                "status": result.status,
                "submitted": result.submitted,
                "slurm_job_id": result.aws_job_id,
                "reason": result.reason,
            },
        }

        logger.info(f"SLURM_AUDIT: {json.dumps(audit_record)}")


# ============================================================================
# INLINE TEST SNIPPET
# ============================================================================
# Run with: python -c "from aurelius.execution.slurm import _run_tests; _run_tests()"

def _run_tests():
    """Inline tests for SlurmExecutor."""
    from datetime import timedelta
    import os

    print("=" * 60)
    print("SlurmExecutor Inline Tests")
    print("=" * 60)

    # Set up test script path
    os.environ["SLURM_SCRIPT_PATH"] = "/path/to/test/script.sh"
    os.environ["SLURM_PARTITION_MAP"] = '{"us-west": "gpu", "us-east": "compute"}'
    os.environ["SLURM_CONSTRAINT_MAP"] = '{"us-west": "v100"}'

    # Create test decisions
    decisions = [
        ScheduleDecision(
            job_id="test-job-1",
            start_time=datetime.utcnow() + timedelta(minutes=5),
            region="us-west",
            power_fraction=1.0,
            actual_runtime_hours=2.0,
        ),
        ScheduleDecision(
            job_id="test-job-delayed",
            start_time=datetime.utcnow() - timedelta(hours=5),
            region="us-east",
            power_fraction=0.75,
            actual_runtime_hours=3.0,
        ),
        ScheduleDecision(
            job_id="test-job-throttled",
            start_time=datetime.utcnow(),
            region="eu-west",
            power_fraction=0.4,  # 60% reduction
            actual_runtime_hours=1.5,
        ),
    ]

    executor = SlurmExecutor()

    # Test 1: Dry run produces no sbatch calls
    print("\nTest 1: DRY RUN mode")
    print("-" * 40)
    config = ExecutionConfig(mode="dry_run", max_delay_hours=2.0, max_power_reduction_pct=50.0)
    results = executor.execute(decisions, config)

    for r in results:
        print(f"  {r.job_id}: status={r.status}, submitted={r.submitted}")
        assert not r.submitted, "Dry run should not submit"

    dry_run_count = sum(1 for r in results if r.status == "dry_run")
    skipped_count = sum(1 for r in results if r.status == "skipped")
    print(f"  Results: {dry_run_count} dry_run, {skipped_count} skipped")
    assert dry_run_count == 1, "Should have 1 dry_run"
    assert skipped_count == 2, "Should have 2 skipped (delay + power)"

    # Test 2: Kill switch aborts
    print("\nTest 2: KILL SWITCH")
    print("-" * 40)
    os.environ["AURELIUS_KILL_SWITCH"] = "true"
    config = ExecutionConfig(mode="live")
    results = executor.execute(decisions[:1], config)
    os.environ.pop("AURELIUS_KILL_SWITCH", None)

    for r in results:
        print(f"  {r.job_id}: status={r.status}")
        assert r.status == "aborted", "Kill switch should abort"
        assert not r.submitted, "Aborted should not submit"

    # Test 3: Max delay skips
    print("\nTest 3: MAX DELAY guardrail")
    print("-" * 40)
    config = ExecutionConfig(mode="dry_run", max_delay_hours=2.0)
    results = executor.execute([decisions[1]], config)

    assert results[0].status == "skipped", "Delayed job should be skipped"
    assert "exceeds max" in results[0].reason, "Should have delay reason"
    print(f"  {results[0].job_id}: {results[0].reason}")

    # Test 4: Max power reduction skips
    print("\nTest 4: MAX POWER REDUCTION guardrail")
    print("-" * 40)
    config = ExecutionConfig(mode="dry_run", max_power_reduction_pct=50.0)
    results = executor.execute([decisions[2]], config)

    assert results[0].status == "skipped", "Over-throttled job should be skipped"
    assert "Power reduction" in results[0].reason, "Should have power reason"
    print(f"  {results[0].job_id}: {results[0].reason}")

    # Test 5: Missing script skips all
    print("\nTest 5: MISSING SCRIPT")
    print("-" * 40)
    os.environ.pop("SLURM_SCRIPT_PATH", None)
    executor_no_script = SlurmExecutor()
    config = ExecutionConfig(mode="dry_run")
    results = executor_no_script.execute([decisions[0]], config)

    assert results[0].status == "skipped", "Missing script should skip"
    assert "SLURM_SCRIPT_PATH missing" in results[0].reason
    print(f"  {results[0].job_id}: {results[0].reason}")

    # Restore script for other tests
    os.environ["SLURM_SCRIPT_PATH"] = "/path/to/test/script.sh"

    # Test 6: Resource calculation
    print("\nTest 6: RESOURCE CALCULATION")
    print("-" * 40)
    resources = calculate_resources(0.5)
    print(f"  power_fraction=0.5 -> cpus={resources['cpus']}, memory={resources['memory_gb']}G")
    assert resources["cpus"] == 2, "CPUs should be 2"
    assert resources["memory_gb"] == 8, "Memory should be 8G"

    resources = calculate_resources(0.1)
    print(f"  power_fraction=0.1 -> cpus={resources['cpus']}, memory={resources['memory_gb']}G")
    assert resources["cpus"] >= MIN_CPU, "CPUs should be clamped to min"
    assert resources["memory_gb"] >= MIN_MEMORY_GB, "Memory should be clamped to min"

    # Test 7: Slurm time formatting
    print("\nTest 7: SLURM TIME FORMATTING")
    print("-" * 40)
    test_time = datetime(2025, 1, 15, 10, 30, 45)
    formatted = format_slurm_time(test_time)
    print(f"  {test_time} -> '{formatted}'")
    assert formatted == "2025-01-15T10:30:45", "Should format correctly for Slurm"

    # Test 8: Job ID parsing
    print("\nTest 8: JOB ID PARSING")
    print("-" * 40)
    test_output = "Submitted batch job 12345678\n"
    job_id = parse_job_id(test_output)
    print(f"  '{test_output.strip()}' -> job_id='{job_id}'")
    assert job_id == "12345678", "Should parse job ID correctly"

    bad_output = "Error: some error message"
    job_id = parse_job_id(bad_output)
    print(f"  '{bad_output}' -> job_id={job_id}")
    assert job_id is None, "Should return None for bad output"

    # Test 9: sbatch command building
    print("\nTest 9: SBATCH COMMAND BUILDING")
    print("-" * 40)
    executor = SlurmExecutor()
    decision = ScheduleDecision(
        job_id="build-test",
        start_time=datetime.utcnow() + timedelta(hours=1),
        region="us-west",
        power_fraction=0.75,
        actual_runtime_hours=2.0,
    )
    cmd = executor._build_sbatch_command(decision)
    cmd_str = " ".join(cmd)
    print(f"  Command: {cmd_str}")

    assert cmd[0] == "sbatch", "Should start with sbatch"
    assert "--job-name" in cmd, "Should have job name"
    assert "--cpus-per-task" in cmd, "Should have CPUs"
    assert "--mem" in cmd, "Should have memory"
    assert "--begin" in cmd, "Should have begin time for future job"
    assert "--partition" in cmd, "Should have partition for us-west"
    assert "--constraint" in cmd, "Should have constraint for us-west"
    assert "--export" in cmd, "Should have env exports"
    assert cmd[-1] == "/path/to/test/script.sh", "Script should be last"

    # Cleanup
    os.environ.pop("SLURM_SCRIPT_PATH", None)
    os.environ.pop("SLURM_PARTITION_MAP", None)
    os.environ.pop("SLURM_CONSTRAINT_MAP", None)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
