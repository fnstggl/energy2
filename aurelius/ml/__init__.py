"""Offline ML estimation layer for Aurelius.

This package provides OFFLINE-ONLY ML pipelines that:
- Read PostExecutionRecord data from local disk
- Train simple, transparent models to improve estimation quality
- Write versioned, immutable artifacts to disk (JSON)
- Provide deterministic artifact loader APIs

CRITICAL ML PHILOSOPHY:
- ML may improve estimates
- ML must NEVER grant permission
- ML outputs are advisory only
- Deterministic control layer remains unchanged
- All artifacts are for offline analysis

This package does NOT:
- Run during job execution
- Affect live runtime behavior
- Modify forecasting/optimization/safety modules
- Override policy or safety gates
"""

from .artifacts import (
    ArtifactWriter,
    get_default_artifact_dir,
    load_artifact,
)
from .dataset import (
    TrainingRecord,
    compute_dataset_hash,
    extract_training_dataset,
    load_post_execution_records,
)
from .forecast_evaluator import (
    EvaluationResult,
    ForecastEvaluator,
    ForecastPoint,
    ModelComparisonResult,
    compare_models,
)
from .model_store import ModelStore
from .trainers import (
    generate_uncertainty_rules,
    train_error_models,
    train_forecast_corrections,
    train_risk_priors,
    train_savings_model,
)

__all__ = [
    # Dataset
    "load_post_execution_records",
    "extract_training_dataset",
    "compute_dataset_hash",
    "TrainingRecord",
    # Artifacts
    "ArtifactWriter",
    "load_artifact",
    "get_default_artifact_dir",
    # Trainers
    "train_forecast_corrections",
    "train_error_models",
    "generate_uncertainty_rules",
    "train_savings_model",
    "train_risk_priors",
    # Forecast evaluation
    "ForecastEvaluator",
    "ForecastPoint",
    "EvaluationResult",
    "ModelComparisonResult",
    "compare_models",
    # Model store
    "ModelStore",
]
