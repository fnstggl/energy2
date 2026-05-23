"""Execution adapters for Aurelius.

This module provides thin execution adapters that translate Aurelius
scheduling decisions into infrastructure actions.

IMPORTANT:
- Default mode is DRY RUN (no infrastructure calls)
- Live execution requires explicit opt-in
- Kill switch can abort all execution instantly
- Aurelius owns decisions, NOT infrastructure

Usage:
    from aurelius.execution import AWSBatchExecutor, KubernetesJobExecutor, SlurmExecutor, ExecutionConfig

    # Create executor (AWS Batch, Kubernetes, or Slurm)
    executor = AWSBatchExecutor()
    # or
    executor = KubernetesJobExecutor()
    # or
    executor = SlurmExecutor()

    # Configure (default is dry_run)
    config = ExecutionConfig(mode="dry_run")

    # Execute decisions
    results = executor.execute(decisions, config)

Environment Variables (common):
    AURELIUS_KILL_SWITCH: Set to "true" to abort all execution

Environment Variables (AWS Batch):
    AWS_REGION: AWS region for Batch operations
    AWS_BATCH_JOB_QUEUE: Default job queue name
    AWS_BATCH_JOB_DEFINITION: Job definition ARN or name

Environment Variables (Kubernetes):
    K8S_JOB_IMAGE: Container image to run (REQUIRED)
    K8S_NAMESPACE: Kubernetes namespace (default: "default")
    K8S_NODE_SELECTOR_KEY: Node selector key (default: "topology.kubernetes.io/region")
    K8S_SERVICE_ACCOUNT: Service account for pods (optional)
    K8S_KUBECONFIG: Path to kubeconfig file (optional)
    K8S_IN_CLUSTER: Set to "true" for in-cluster config (optional)

Environment Variables (Slurm):
    SLURM_SCRIPT_PATH: Path to the job script to submit (REQUIRED)
    SLURM_PARTITION_MAP: JSON mapping region -> partition (e.g. '{"us-west": "gpu"}')
    SLURM_CONSTRAINT_MAP: JSON mapping region -> constraint (e.g. '{"us-west": "v100"}')
    SLURM_ACCOUNT: Slurm account for job submission (optional)
    SLURM_QOS: Quality of service level (optional)
"""

from .aws_batch import (
    AWSBatchExecutor,
)
from .base import (
    ExecutionConfig,
    ExecutionResult,
    Executor,
    log_execution_audit,
)
from .constraints import (
    ConstraintEvaluation,
    ConstraintEvaluator,
    apply_constraint_filter,
)
from .kubernetes import (
    KubernetesJobExecutor,
)
from .policy import (
    AuthorizationResult,
    PolicyBundle,
    PolicyConfig,
    SignatureInfo,
    authorize_execution,
    canonical_json_bytes,
    get_policy_path,
    load_policy_bundle,
    verify_signature,
)
from .post_execution import (
    SAVINGS_EPSILON,
    ForecastSnapshot,
    PostExecutionRecord,
    PostExecutionRecorder,
    RealizedOutcome,
    compute_forecast_errors,
    compute_realized_savings,
    generate_decision_id,
    label_decision_outcome,
)
from .slurm import (
    SlurmExecutor,
)

__all__ = [
    "Executor",
    "ExecutionConfig",
    "ExecutionResult",
    "AWSBatchExecutor",
    "KubernetesJobExecutor",
    "SlurmExecutor",
    "log_execution_audit",
    "ConstraintEvaluation",
    "ConstraintEvaluator",
    "apply_constraint_filter",
    "ForecastSnapshot",
    "RealizedOutcome",
    "PostExecutionRecord",
    "PostExecutionRecorder",
    "generate_decision_id",
    "compute_forecast_errors",
    "compute_realized_savings",
    "label_decision_outcome",
    "SAVINGS_EPSILON",
    "PolicyConfig",
    "PolicyBundle",
    "SignatureInfo",
    "AuthorizationResult",
    "authorize_execution",
    "load_policy_bundle",
    "verify_signature",
    "canonical_json_bytes",
    "get_policy_path",
]
