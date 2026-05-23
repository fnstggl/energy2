"""Kubernetes Job execution adapter for Aurelius.

This module submits Kubernetes batch/v1 Jobs based on Aurelius scheduling decisions.
Aurelius decides WHEN / WHERE / HOW - Kubernetes handles execution.

IMPORTANT:
- Default mode is DRY RUN (no Kubernetes API calls)
- Live mode requires explicit opt-in
- Kill switch can abort all execution instantly
- Every execution attempt is logged for audit

Environment Variables:
    K8S_JOB_IMAGE: Container image to run (REQUIRED)
    K8S_NAMESPACE: Kubernetes namespace (default: "default")
    K8S_NODE_SELECTOR_KEY: Node selector key (default: "topology.kubernetes.io/region")
    K8S_SERVICE_ACCOUNT: Service account for pods (optional)
    K8S_KUBECONFIG: Path to kubeconfig file (optional)
    K8S_IN_CLUSTER: Set to "true" for in-cluster config (optional)
    K8S_JOB_TTL_SECONDS_AFTER_FINISHED: TTL for completed jobs (default: 3600)
    K8S_REGION_LABEL_PREFIX: Prefix for region label values (optional)
    AURELIUS_KILL_SWITCH: Set to "true" to abort all execution
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

from ..models import ScheduleDecision
from .base import (
    ExecutionConfig,
    ExecutionResult,
    Executor,
    log_execution_audit,
)

logger = logging.getLogger(__name__)


# Resource baselines
BASELINE_VCPU = 4.0
BASELINE_MEMORY_GI = 16.0

# Resource clamps
MIN_CPU = 0.25
MAX_CPU = 16.0
MIN_MEMORY_GI = 1.0
MAX_MEMORY_GI = 64.0

# Sleep chunk size for interruptible waits
SLEEP_CHUNK_SECONDS = 30.0


def sanitize_job_name(job_id: str, start_time: datetime) -> str:
    """Create a valid Kubernetes job name.

    Args:
        job_id: Aurelius job ID
        start_time: Scheduled start time

    Returns:
        Sanitized job name (lowercase alphanum + hyphen, max 63 chars)
    """
    timestamp = start_time.strftime("%Y%m%d%H%M")
    raw_name = f"aurelius-{job_id}-{timestamp}"

    # Lowercase and replace invalid chars with hyphen
    sanitized = re.sub(r"[^a-z0-9-]", "-", raw_name.lower())

    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Strip leading/trailing hyphens
    sanitized = sanitized.strip("-")

    # Truncate to 63 chars
    if len(sanitized) > 63:
        sanitized = sanitized[:63].rstrip("-")

    return sanitized


def calculate_resources(power_fraction: float) -> dict:
    """Calculate CPU and memory resources from power fraction.

    Args:
        power_fraction: Power throttle fraction (0-1)

    Returns:
        Dict with cpu (str) and memory (str) values
    """
    cpu = BASELINE_VCPU * power_fraction
    memory_gi = BASELINE_MEMORY_GI * power_fraction

    # Clamp values
    cpu = max(MIN_CPU, min(MAX_CPU, cpu))
    memory_gi = max(MIN_MEMORY_GI, min(MAX_MEMORY_GI, memory_gi))

    return {
        "cpu": str(cpu),
        "memory": f"{int(memory_gi)}Gi",
    }


class KubernetesJobExecutor(Executor):
    """Executor that submits Kubernetes batch/v1 Jobs.

    This is a thin adapter that:
    - Reads K8s config from environment variables
    - Validates guardrails before submission
    - Submits Jobs with appropriate resources and placement
    - Adds labels/annotations for auditability
    - Returns structured ExecutionResult objects

    It does NOT:
    - Create controllers/operators/CRDs
    - Watch Jobs or implement reconciliation
    - Modify or delete existing jobs
    - Touch autoscaling or RBAC
    - Implement custom retry logic
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        image: Optional[str] = None,
        node_selector_key: Optional[str] = None,
        service_account: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        region_label_prefix: Optional[str] = None,
    ):
        """Initialize the Kubernetes Job executor.

        Args:
            namespace: Kubernetes namespace (or from K8S_NAMESPACE)
            image: Container image (or from K8S_JOB_IMAGE)
            node_selector_key: Node selector key (or from K8S_NODE_SELECTOR_KEY)
            service_account: Service account (or from K8S_SERVICE_ACCOUNT)
            ttl_seconds: TTL after finished (or from K8S_JOB_TTL_SECONDS_AFTER_FINISHED)
            region_label_prefix: Prefix for region values (or from K8S_REGION_LABEL_PREFIX)
        """
        self.namespace = namespace or os.environ.get("K8S_NAMESPACE", "default")
        self.image = image or os.environ.get("K8S_JOB_IMAGE", "")
        self.node_selector_key = node_selector_key or os.environ.get(
            "K8S_NODE_SELECTOR_KEY", "topology.kubernetes.io/region"
        )
        self.service_account = service_account or os.environ.get("K8S_SERVICE_ACCOUNT", "")
        self.ttl_seconds = ttl_seconds or int(
            os.environ.get("K8S_JOB_TTL_SECONDS_AFTER_FINISHED", "3600")
        )
        self.region_label_prefix = region_label_prefix or os.environ.get(
            "K8S_REGION_LABEL_PREFIX", ""
        )

        # Lazy-load Kubernetes client
        self._batch_api = None
        self._client_loaded = False

    def _get_batch_api(self):
        """Get or create the Kubernetes BatchV1Api client."""
        if not self._client_loaded:
            try:
                from kubernetes import client, config

                # Load configuration
                kubeconfig_path = os.environ.get("K8S_KUBECONFIG", "")
                in_cluster = os.environ.get("K8S_IN_CLUSTER", "").lower() == "true"

                if in_cluster:
                    config.load_incluster_config()
                elif kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path)
                else:
                    config.load_kube_config()

                self._batch_api = client.BatchV1Api()
                self._client_loaded = True

            except ImportError:
                raise RuntimeError("kubernetes package is required for Kubernetes execution")
            except Exception as e:
                raise RuntimeError(f"Failed to load Kubernetes config: {e}")

        return self._batch_api

    def _get_node_selector_value(self, region: str) -> str:
        """Get the node selector value for a region.

        Args:
            region: Aurelius region from decision

        Returns:
            Node selector value (with optional prefix)
        """
        if self.region_label_prefix:
            return f"{self.region_label_prefix}{region}"
        return region

    def execute(
        self,
        decisions: list[ScheduleDecision],
        config: ExecutionConfig,
    ) -> list[ExecutionResult]:
        """Execute scheduling decisions by submitting Kubernetes Jobs.

        Execution is sequential (no parallelization).

        For each decision:
        1. Check kill switch
        2. Validate guardrails (max_delay, max_power_reduction)
        3. If dry_run: log and return without K8s API calls
        4. If live: wait until start_time, then submit Job

        Args:
            decisions: List of ScheduleDecision objects from the optimizer
            config: ExecutionConfig controlling behavior

        Returns:
            List of ExecutionResult objects, one per decision
        """
        results = []

        logger.info(f"Starting execution: {len(decisions)} decisions, mode={config.mode}")

        # Check for missing image upfront
        if not self.image:
            logger.error("K8S_JOB_IMAGE is missing - all jobs will be skipped")

        for decision in decisions:
            # Check kill switch before each job
            if config.is_kill_switch_active():
                # Abort ALL remaining decisions
                result = self._create_aborted_result(decision, config)
                results.append(result)
                log_execution_audit(decision, result, config)
                self._log_audit_json(decision, result, config)
                # Continue to abort remaining
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

        # Check for missing image
        if not self.image:
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=now,
                status="skipped",
                reason="K8S_JOB_IMAGE missing",
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

        # Build job spec for logging/submission
        job_spec = self._build_job_spec(decision, config)

        # DRY RUN MODE: Log what would happen, no K8s API calls
        if config.is_dry_run():
            logger.info(
                f"[DRY RUN] Would submit Job {job_spec['name']} to namespace {job_spec['namespace']} "
                f"with image={job_spec['image']}, resources={job_spec['resources']}, "
                f"nodeSelector={job_spec['node_selector']}"
            )
            return ExecutionResult(
                job_id=decision.job_id,
                submitted=False,
                aws_job_id=None,
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="dry_run",
                reason="Dry run mode - no Kubernetes API calls made",
            )

        # LIVE MODE: Wait and submit
        return self._submit_to_kubernetes(decision, config, job_spec)

    def _build_job_spec(
        self,
        decision: ScheduleDecision,
        config: ExecutionConfig,
    ) -> dict:
        """Build the job specification dictionary.

        Args:
            decision: The scheduling decision
            config: Execution configuration

        Returns:
            Dict with job spec fields for logging and submission
        """
        job_name = sanitize_job_name(decision.job_id, decision.start_time)
        resources = calculate_resources(decision.power_fraction)
        node_selector_value = self._get_node_selector_value(decision.region)

        return {
            "name": job_name,
            "namespace": self.namespace,
            "image": self.image,
            "resources": resources,
            "node_selector": {
                self.node_selector_key: node_selector_value,
            },
            "labels": {
                "app": "aurelius",
                "submitted_by": "aurelius",
                "aurelius_job_id": decision.job_id,
                "aurelius_region": decision.region,
                "aurelius_mode": config.mode,
            },
            "annotations": {
                "aurelius/decision_time": datetime.utcnow().isoformat(),
                "aurelius/start_time": decision.start_time.isoformat(),
                "aurelius/power_fraction": str(decision.power_fraction),
                "aurelius/runtime_hours": str(decision.actual_runtime_hours),
            },
            "env": {
                "AURELIUS_JOB_ID": decision.job_id,
                "AURELIUS_REGION": decision.region,
                "AURELIUS_POWER_FRACTION": str(decision.power_fraction),
                "AURELIUS_RUNTIME_HOURS": str(decision.actual_runtime_hours),
                "AURELIUS_SCHEDULED_START_TIME": decision.start_time.isoformat(),
            },
            "service_account": self.service_account,
            "ttl_seconds": self.ttl_seconds,
        }

    def _submit_to_kubernetes(
        self,
        decision: ScheduleDecision,
        config: ExecutionConfig,
        job_spec: dict,
    ) -> ExecutionResult:
        """Submit a Job to Kubernetes.

        Waits until decision.start_time before submitting using chunked sleeps.

        Args:
            decision: The scheduling decision
            config: Execution configuration
            job_spec: Pre-built job specification

        Returns:
            ExecutionResult with K8s job name if successful
        """
        now = datetime.utcnow()

        # Wait until start_time if it's in the future (chunked for kill switch checks)
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

        # Build and submit the Kubernetes Job
        try:
            from kubernetes import client

            batch_api = self._get_batch_api()

            # Build container env vars
            env_vars = [
                client.V1EnvVar(name=k, value=v)
                for k, v in job_spec["env"].items()
            ]

            # Build container
            container = client.V1Container(
                name="job",
                image=job_spec["image"],
                env=env_vars,
                resources=client.V1ResourceRequirements(
                    requests={
                        "cpu": job_spec["resources"]["cpu"],
                        "memory": job_spec["resources"]["memory"],
                    },
                    limits={
                        "cpu": job_spec["resources"]["cpu"],
                        "memory": job_spec["resources"]["memory"],
                    },
                ),
                # DO NOT set command or args - let image ENTRYPOINT/CMD run
            )

            # Build pod spec
            pod_spec_kwargs = {
                "containers": [container],
                "restart_policy": "Never",
                "node_selector": job_spec["node_selector"],
            }
            if job_spec["service_account"]:
                pod_spec_kwargs["service_account_name"] = job_spec["service_account"]

            pod_spec = client.V1PodSpec(**pod_spec_kwargs)

            # Build pod template
            pod_template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels=job_spec["labels"],
                    annotations=job_spec["annotations"],
                ),
                spec=pod_spec,
            )

            # Build job spec
            k8s_job_spec = client.V1JobSpec(
                template=pod_template,
                backoff_limit=0,  # Pilot safety: no silent retries
                ttl_seconds_after_finished=job_spec["ttl_seconds"],
            )

            # Build job
            job = client.V1Job(
                api_version="batch/v1",
                kind="Job",
                metadata=client.V1ObjectMeta(
                    name=job_spec["name"],
                    namespace=job_spec["namespace"],
                    labels=job_spec["labels"],
                    annotations=job_spec["annotations"],
                ),
                spec=k8s_job_spec,
            )

            # Log warning if node selector might be empty
            if not job_spec["node_selector"].get(self.node_selector_key):
                logger.warning(
                    f"Node selector value is empty for {decision.job_id}, "
                    f"job may run on any node"
                )

            # Submit the job
            logger.info(f"Submitting Job {job_spec['name']} to namespace {job_spec['namespace']}")
            response = batch_api.create_namespaced_job(
                namespace=job_spec["namespace"],
                body=job,
            )

            k8s_job_name = response.metadata.name
            logger.info(f"Successfully submitted {decision.job_id} as K8s job {k8s_job_name}")

            return ExecutionResult(
                job_id=decision.job_id,
                submitted=True,
                aws_job_id=k8s_job_name,  # Reusing field for K8s job name
                region=decision.region,
                submit_time=datetime.utcnow(),
                status="submitted",
                reason=f"Submitted to namespace {job_spec['namespace']}",
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
                reason=f"Kubernetes submission failed: {str(e)}",
            )

    def _log_audit_json(
        self,
        decision: ScheduleDecision,
        result: ExecutionResult,
        config: ExecutionConfig,
    ) -> None:
        """Log a structured JSON audit record with K8s-specific fields.

        Args:
            decision: The scheduling decision
            result: The execution result
            config: The execution configuration
        """
        job_spec = self._build_job_spec(decision, config) if self.image else {}

        audit_record = {
            "event": "k8s_execution_attempt",
            "timestamp": datetime.utcnow().isoformat(),
            "mode": config.mode,
            "decision": {
                "job_id": decision.job_id,
                "start_time": decision.start_time.isoformat(),
                "region": decision.region,
                "power_fraction": decision.power_fraction,
                "runtime_hours": decision.actual_runtime_hours,
            },
            "k8s": {
                "job_name": job_spec.get("name", ""),
                "namespace": job_spec.get("namespace", self.namespace),
                "image": job_spec.get("image", self.image),
                "resources": job_spec.get("resources", {}),
                "node_selector": job_spec.get("node_selector", {}),
            },
            "result": {
                "status": result.status,
                "submitted": result.submitted,
                "k8s_job_name": result.aws_job_id,
                "reason": result.reason,
            },
        }

        logger.info(f"K8S_AUDIT: {json.dumps(audit_record)}")


# ============================================================================
# INLINE TEST SNIPPET
# ============================================================================
# Run with: python -c "from aurelius.execution.kubernetes import _run_tests; _run_tests()"

def _run_tests():
    """Inline tests for KubernetesJobExecutor."""
    import os
    from datetime import timedelta

    print("=" * 60)
    print("KubernetesJobExecutor Inline Tests")
    print("=" * 60)

    # Set up test image
    os.environ["K8S_JOB_IMAGE"] = "alpine:latest"

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

    executor = KubernetesJobExecutor()

    # Test 1: Dry run produces no API calls
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

    # Test 5: Missing image skips all
    print("\nTest 5: MISSING IMAGE")
    print("-" * 40)
    os.environ.pop("K8S_JOB_IMAGE", None)
    executor_no_image = KubernetesJobExecutor()
    config = ExecutionConfig(mode="dry_run")
    results = executor_no_image.execute([decisions[0]], config)

    assert results[0].status == "skipped", "Missing image should skip"
    assert "K8S_JOB_IMAGE missing" in results[0].reason
    print(f"  {results[0].job_id}: {results[0].reason}")

    # Restore image for other tests
    os.environ["K8S_JOB_IMAGE"] = "alpine:latest"

    # Test 6: Job name sanitization
    print("\nTest 6: JOB NAME SANITIZATION")
    print("-" * 40)
    test_time = datetime(2025, 1, 15, 10, 30)
    name = sanitize_job_name("My_Test.Job!123", test_time)
    print(f"  Input: 'My_Test.Job!123' -> '{name}'")
    assert name.islower() or "-" in name, "Should be lowercase"
    assert len(name) <= 63, "Should be max 63 chars"
    assert re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", name), "Should be valid K8s name"

    # Test 7: Resource calculation
    print("\nTest 7: RESOURCE CALCULATION")
    print("-" * 40)
    resources = calculate_resources(0.5)
    print(f"  power_fraction=0.5 -> cpu={resources['cpu']}, memory={resources['memory']}")
    assert float(resources["cpu"]) == 2.0, "CPU should be 2.0"
    assert resources["memory"] == "8Gi", "Memory should be 8Gi"

    resources = calculate_resources(0.1)
    print(f"  power_fraction=0.1 -> cpu={resources['cpu']}, memory={resources['memory']}")
    assert float(resources["cpu"]) >= MIN_CPU, "CPU should be clamped to min"
    assert resources["memory"] == f"{int(MIN_MEMORY_GI)}Gi", "Memory should be clamped to min"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
