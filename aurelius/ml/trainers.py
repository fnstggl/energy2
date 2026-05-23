"""Offline ML trainers for Aurelius.

Simple, transparent, deterministic training algorithms that produce
versioned artifacts for offline use only.

These trainers:
- Are deterministic given same input + seed
- Produce explainable, audit-friendly outputs
- Never affect execution or grant permissions

CRITICAL: ML outputs are advisory estimates only.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .artifacts import generate_timestamp_utc
from .dataset import TrainingRecord

logger = logging.getLogger(__name__)


# ============================================================================
# STATISTICS UTILITIES (numpy-free for simplicity, but could use numpy)
# ============================================================================

def mean(values: list[float]) -> float:
    """Compute mean of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    """Compute standard deviation of values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def median(values: list[float]) -> float:
    """Compute median of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# ============================================================================
# BUCKET KEY GENERATION
# ============================================================================

@dataclass(frozen=True)
class BucketKey:
    """Immutable bucket key for grouping records."""
    region: str
    hour_utc: Optional[int] = None
    constraint_profile: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {"region": self.region}
        if self.hour_utc is not None:
            result["hour_utc"] = self.hour_utc
        if self.constraint_profile is not None:
            result["constraint_profile"] = self.constraint_profile
        return result


def make_region_hour_key(record: TrainingRecord) -> Optional[BucketKey]:
    """Create bucket key from region + hour_utc."""
    if not record.region:
        return None
    return BucketKey(region=record.region, hour_utc=record.hour_utc)


def make_region_profile_key(record: TrainingRecord) -> Optional[BucketKey]:
    """Create bucket key from region + constraint_profile."""
    if not record.region:
        return None
    return BucketKey(
        region=record.region,
        constraint_profile=record.constraint_profile or "unknown",
    )


# ============================================================================
# 1. FORECAST CORRECTIONS TRAINER
# ============================================================================

def train_forecast_corrections(
    records: list[TrainingRecord],
) -> dict[str, Any]:
    """Train forecast correction model (bias adjustment).

    Learns systematic bias in p50 forecasts:
      bias = mean(realized - forecast_p50)
      adjusted_p50 = forecast_p50 + bias

    Buckets by region + hour_utc.

    Args:
        records: Training records from PostExecutionRecord

    Returns:
        Artifact dictionary matching forecast_corrections_v1.json schema
    """
    # Collect errors by bucket
    energy_errors: dict[BucketKey, list[float]] = defaultdict(list)
    carbon_errors: dict[BucketKey, list[float]] = defaultdict(list)

    for record in records:
        key = make_region_hour_key(record)
        if key is None:
            continue

        # Energy cost error
        if (
            record.energy_cost_p50_error is not None
            and record.forecast_energy_cost_p50 is not None
        ):
            energy_errors[key].append(record.energy_cost_p50_error)

        # Carbon error
        if (
            record.carbon_p50_error is not None
            and record.forecast_carbon_p50 is not None
        ):
            carbon_errors[key].append(record.carbon_p50_error)

    # Build buckets
    all_keys = sorted(
        set(energy_errors.keys()) | set(carbon_errors.keys()),
        key=lambda k: (k.region, k.hour_utc or -1),
    )

    buckets = []
    for key in all_keys:
        bucket = key.to_dict()

        energy_vals = energy_errors.get(key, [])
        carbon_vals = carbon_errors.get(key, [])

        bucket["energy_cost_p50_bias"] = round(mean(energy_vals), 4) if energy_vals else None
        bucket["carbon_p50_bias"] = round(mean(carbon_vals), 4) if carbon_vals else None
        bucket["n"] = max(len(energy_vals), len(carbon_vals))

        buckets.append(bucket)

    return {
        "version": 1,
        "generated_at_utc": generate_timestamp_utc(),
        "method": "bucketed_mean_residual",
        "buckets": buckets,
    }


# ============================================================================
# 2. ERROR MODELS TRAINER
# ============================================================================

def train_error_models(
    records: list[TrainingRecord],
) -> dict[str, Any]:
    """Train error distribution models.

    Computes per-bucket error statistics:
    - mean error, std error
    - p50/p90 absolute error
    - empirical p90 coverage rate
    - recommended p90 scale factor

    Args:
        records: Training records from PostExecutionRecord

    Returns:
        Artifact dictionary matching error_models_v1.json schema
    """
    # Collect data by bucket
    energy_data: dict[BucketKey, dict[str, list]] = defaultdict(
        lambda: {"errors": [], "abs_errors": [], "p90_covered": []}
    )
    carbon_data: dict[BucketKey, dict[str, list]] = defaultdict(
        lambda: {"errors": [], "abs_errors": [], "p90_covered": []}
    )

    for record in records:
        key = make_region_hour_key(record)
        if key is None:
            continue

        # Energy cost
        if record.energy_cost_p50_error is not None:
            energy_data[key]["errors"].append(record.energy_cost_p50_error)
            energy_data[key]["abs_errors"].append(abs(record.energy_cost_p50_error))
        if record.energy_cost_p90_covered is not None:
            energy_data[key]["p90_covered"].append(
                1.0 if record.energy_cost_p90_covered else 0.0
            )

        # Carbon
        if record.carbon_p50_error is not None:
            carbon_data[key]["errors"].append(record.carbon_p50_error)
            carbon_data[key]["abs_errors"].append(abs(record.carbon_p50_error))
        if record.carbon_p90_covered is not None:
            carbon_data[key]["p90_covered"].append(
                1.0 if record.carbon_p90_covered else 0.0
            )

    # Build buckets
    all_keys = sorted(
        set(energy_data.keys()) | set(carbon_data.keys()),
        key=lambda k: (k.region, k.hour_utc or -1),
    )

    buckets = []
    for key in all_keys:
        bucket = key.to_dict()

        # Energy cost stats
        e_data = energy_data.get(key, {"errors": [], "abs_errors": [], "p90_covered": []})
        if e_data["errors"]:
            coverage_rate = mean(e_data["p90_covered"]) if e_data["p90_covered"] else 1.0
            # Recommend multiplier if coverage < 0.90
            recommended_mult = 1.0
            if coverage_rate < 0.90 and e_data["abs_errors"]:
                # Simple heuristic: scale up by coverage shortfall ratio
                recommended_mult = min(1.50, max(1.05, 0.90 / max(coverage_rate, 0.5)))

            bucket["energy_cost"] = {
                "mean_error": round(mean(e_data["errors"]), 4),
                "std_error": round(std(e_data["errors"]), 4) if len(e_data["errors"]) >= 2 else None,
                "p50_abs_error": round(median(e_data["abs_errors"]), 4),
                "p90_abs_error": round(percentile(e_data["abs_errors"], 90), 4),
                "p90_coverage_rate": round(coverage_rate, 4),
                "recommended_p90_multiplier": round(recommended_mult, 2),
                "n": len(e_data["errors"]),
            }

        # Carbon stats
        c_data = carbon_data.get(key, {"errors": [], "abs_errors": [], "p90_covered": []})
        if c_data["errors"]:
            coverage_rate = mean(c_data["p90_covered"]) if c_data["p90_covered"] else 1.0
            recommended_mult = 1.0
            if coverage_rate < 0.90 and c_data["abs_errors"]:
                recommended_mult = min(1.50, max(1.05, 0.90 / max(coverage_rate, 0.5)))

            bucket["carbon"] = {
                "mean_error": round(mean(c_data["errors"]), 4),
                "std_error": round(std(c_data["errors"]), 4) if len(c_data["errors"]) >= 2 else None,
                "p50_abs_error": round(median(c_data["abs_errors"]), 4),
                "p90_abs_error": round(percentile(c_data["abs_errors"], 90), 4),
                "p90_coverage_rate": round(coverage_rate, 4),
                "recommended_p90_multiplier": round(recommended_mult, 2),
                "n": len(c_data["errors"]),
            }

        buckets.append(bucket)

    return {
        "version": 1,
        "generated_at_utc": generate_timestamp_utc(),
        "method": "empirical_error_stats",
        "buckets": buckets,
    }


# ============================================================================
# 3. UNCERTAINTY RULES GENERATOR
# ============================================================================

def generate_uncertainty_rules(
    error_models: dict[str, Any],
) -> dict[str, Any]:
    """Generate deterministic uncertainty shaping rules.

    Rules reference ML-derived statistics from error_models
    but remain deterministic in application.

    Args:
        error_models: Output from train_error_models()

    Returns:
        Artifact dictionary matching uncertainty_rules_v1.json schema
    """
    # Define rules based on common patterns
    rules = [
        {
            "name": "widen_if_energy_p90_coverage_low",
            "condition": {
                "metric": "energy_cost",
                "p90_coverage_below": 0.90,
                "min_n": 50,
            },
            "action": {
                "apply_p90_multiplier_from_error_model": True,
                "min_multiplier": 1.05,
                "max_multiplier": 1.50,
            },
        },
        {
            "name": "widen_if_carbon_p90_coverage_low",
            "condition": {
                "metric": "carbon",
                "p90_coverage_below": 0.90,
                "min_n": 50,
            },
            "action": {
                "apply_p90_multiplier_from_error_model": True,
                "min_multiplier": 1.05,
                "max_multiplier": 1.50,
            },
        },
        {
            "name": "narrow_if_overconservative",
            "condition": {
                "metric": "energy_cost",
                "p90_coverage_above": 0.98,
                "min_n": 100,
            },
            "action": {
                "apply_p90_multiplier": 0.95,
                "min_multiplier": 0.90,
                "max_multiplier": 1.00,
            },
        },
    ]

    return {
        "version": 1,
        "generated_at_utc": generate_timestamp_utc(),
        "rules": rules,
    }


# ============================================================================
# 4. SAVINGS MODEL TRAINER
# ============================================================================

def train_savings_model(
    records: list[TrainingRecord],
) -> dict[str, Any]:
    """Train savings estimation model.

    Computes per-bucket savings distribution statistics
    for counterfactual analysis.

    Args:
        records: Training records from PostExecutionRecord

    Returns:
        Artifact dictionary matching savings_model_v1.json schema
    """
    # Collect savings by bucket
    savings_data: dict[BucketKey, list[float]] = defaultdict(list)

    for record in records:
        key = make_region_profile_key(record)
        if key is None:
            continue

        if record.realized_savings is not None:
            savings_data[key].append(record.realized_savings)

    # Build buckets
    all_keys = sorted(
        savings_data.keys(),
        key=lambda k: (k.region, k.constraint_profile or ""),
    )

    buckets = []
    has_data = False

    for key in all_keys:
        savings = savings_data[key]
        if not savings:
            continue

        has_data = True
        positive_count = sum(1 for s in savings if s > 0)

        bucket = {
            "region": key.region,
            "constraint_profile": key.constraint_profile or "unknown",
            "mean_savings": round(mean(savings), 4),
            "p10_savings": round(percentile(savings, 10), 4),
            "p50_savings": round(median(savings), 4),
            "p90_savings": round(percentile(savings, 90), 4),
            "fraction_positive": round(positive_count / len(savings), 4),
            "n": len(savings),
        }
        buckets.append(bucket)

    notes = None
    if not has_data:
        notes = "No realized_savings data available. Buckets are empty."

    result = {
        "version": 1,
        "generated_at_utc": generate_timestamp_utc(),
        "method": "bucketed_savings_stats",
        "buckets": buckets,
    }

    if notes:
        result["notes"] = notes

    return result


# ============================================================================
# 5. RISK PRIORS TRAINER
# ============================================================================

def train_risk_priors(
    records: list[TrainingRecord],
    error_models: dict[str, Any],
) -> dict[str, Any]:
    """Train risk prior model.

    Computes advisory risk scores based on:
    - Coverage shortfall (how often p90 missed)
    - Tail error magnitude

    Risk score is in [0, 1] where higher = riskier.
    This is an INPUT to deterministic gates, never a decider.

    Args:
        records: Training records from PostExecutionRecord
        error_models: Output from train_error_models()

    Returns:
        Artifact dictionary matching risk_priors_v1.json schema
    """
    # Weights for risk components
    WEIGHT_COVERAGE = 0.6
    WEIGHT_TAIL_ERROR = 0.4

    # Build error model lookup
    error_lookup: dict[BucketKey, dict] = {}
    for bucket in error_models.get("buckets", []):
        key = BucketKey(
            region=bucket.get("region", ""),
            hour_utc=bucket.get("hour_utc"),
        )
        error_lookup[key] = bucket

    # Collect outcome labels by bucket
    outcome_data: dict[BucketKey, dict[str, int]] = defaultdict(
        lambda: {"good_decision": 0, "neutral": 0, "conservative_skip": 0, "total": 0}
    )

    for record in records:
        key = make_region_profile_key(record)
        if key is None:
            continue

        label = record.decision_outcome_label or "neutral"
        outcome_data[key][label] = outcome_data[key].get(label, 0) + 1
        outcome_data[key]["total"] += 1

    # Build buckets
    all_keys = sorted(
        outcome_data.keys(),
        key=lambda k: (k.region, k.constraint_profile or ""),
    )

    buckets = []
    for key in all_keys:
        outcomes = outcome_data[key]
        if outcomes["total"] == 0:
            continue

        # Get error model for this region (aggregate across hours)
        # Find matching error buckets
        matching_error_buckets = [
            b for k, b in error_lookup.items()
            if k.region == key.region
        ]

        # Compute coverage shortfall component
        coverage_shortfall = 0.0
        tail_error_component = 0.0

        if matching_error_buckets:
            # Average coverage shortfall across hours for this region
            coverage_rates = []
            p90_errors = []

            for eb in matching_error_buckets:
                if "energy_cost" in eb:
                    ec = eb["energy_cost"]
                    if ec.get("p90_coverage_rate") is not None:
                        coverage_rates.append(ec["p90_coverage_rate"])
                    if ec.get("p90_abs_error") is not None:
                        p90_errors.append(ec["p90_abs_error"])

            if coverage_rates:
                avg_coverage = mean(coverage_rates)
                # Shortfall = how far below 0.90 target
                coverage_shortfall = max(0.0, 0.90 - avg_coverage)

            if p90_errors:
                # Normalize tail error (assume reasonable max of 10)
                avg_tail_error = mean(p90_errors)
                tail_error_component = min(1.0, avg_tail_error / 10.0)

        # Compute weighted risk prior
        risk_prior = (
            WEIGHT_COVERAGE * coverage_shortfall +
            WEIGHT_TAIL_ERROR * tail_error_component
        )
        risk_prior = min(1.0, max(0.0, risk_prior))

        bucket = {
            "region": key.region,
            "constraint_profile": key.constraint_profile or "unknown",
            "risk_prior": round(risk_prior, 4),
            "components": {
                "coverage_shortfall": round(coverage_shortfall, 4),
                "tail_error": round(tail_error_component, 4),
            },
            "n": outcomes["total"],
        }
        buckets.append(bucket)

    return {
        "version": 1,
        "generated_at_utc": generate_timestamp_utc(),
        "method": "weighted_empirical_risk",
        "weights": {
            "coverage_shortfall": WEIGHT_COVERAGE,
            "tail_error": WEIGHT_TAIL_ERROR,
        },
        "buckets": buckets,
    }


# ============================================================================
# 6. LIGHTGBM FEATURE EXTRACTION
# ============================================================================

def _region_to_int(region: Optional[str], region_map: dict[str, int]) -> int:
    """Encode region string to integer. Unknown regions map to -1."""
    if region is None:
        return -1
    return region_map.get(region, -1)


def _extract_lgbm_features(
    records: list[TrainingRecord],
    region_map: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Extract numeric feature matrix from TrainingRecord list.

    Features (all normalized to floats, missing → 0.0):
        region_enc, hour_utc, forecast_p50_energy, p90_energy_range,
        forecast_p50_carbon, p90_carbon_range, baseline_energy

    Returns:
        (X np.ndarray of shape [N, n_features], feature_names list)
    """
    feature_names = [
        "region_enc",
        "hour_utc",
        "forecast_p50_energy",
        "p90_energy_range",
        "forecast_p50_carbon",
        "p90_carbon_range",
        "baseline_energy",
    ]

    rows = []
    for r in records:
        p50e = r.forecast_energy_cost_p50 or 0.0
        p90e = r.forecast_energy_cost_p90 or 0.0
        p50c = r.forecast_carbon_p50 or 0.0
        p90c = r.forecast_carbon_p90 or 0.0
        baseline_e = r.forecast_energy_cost_baseline or p50e

        rows.append([
            float(_region_to_int(r.region, region_map)),
            float(r.hour_utc) if r.hour_utc is not None else 0.0,
            p50e,
            max(0.0, p90e - p50e),   # uncertainty range
            p50c,
            max(0.0, p90c - p50c),
            baseline_e,
        ])

    return np.array(rows, dtype=float), feature_names


# ============================================================================
# 7. LIGHTGBM SAVINGS MODEL
# ============================================================================

_MIN_LGBM_RECORDS = 50   # Minimum records before LightGBM training is attempted
_LGBM_HOLDOUT_FRAC = 0.2  # Fraction of records reserved for holdout eval


def train_savings_model_lgbm(
    records: list[TrainingRecord],
    seed: int = 42,
    min_records: int = _MIN_LGBM_RECORDS,
) -> dict[str, Any]:
    """Train a LightGBM gradient-boosted savings estimation model.

    Replaces the bucketed-statistics approach with a real gradient-boosted
    regression. Falls back to bucketed stats if LightGBM is unavailable or
    there are fewer than ``min_records`` labelled samples.

    Training uses a time-ordered 80/20 split so the holdout represents
    genuinely unseen future records.  The split is time-ordered (not random)
    to prevent leakage.

    Args:
        records: Training records from PostExecutionRecord.
        seed: Random seed for reproducibility.
        min_records: Minimum labelled records required for LightGBM training.

    Returns:
        Artifact dict with model_string, feature_names, metrics, and metadata.
        Falls back to bucketed stats if LightGBM not available / insufficient data.
    """
    # Extract labelled records (need realized_savings)
    labelled = [r for r in records if r.realized_savings is not None]

    if len(labelled) < min_records:
        logger.warning(
            f"train_savings_model_lgbm: only {len(labelled)} labelled records "
            f"(need {min_records}); falling back to bucketed stats"
        )
        result = train_savings_model(records)
        result["method"] = "bucketed_savings_stats_fallback"
        result["lgbm_skipped_reason"] = f"insufficient_records:{len(labelled)}<{min_records}"
        return result

    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        logger.warning("train_savings_model_lgbm: lightgbm not available; falling back to bucketed stats")
        result = train_savings_model(records)
        result["method"] = "bucketed_savings_stats_fallback"
        result["lgbm_skipped_reason"] = "lightgbm_not_installed"
        return result

    # Time-ordered split (no randomisation — prevents leakage)
    n_holdout = max(1, int(len(labelled) * _LGBM_HOLDOUT_FRAC))
    train_records = labelled[:-n_holdout]
    holdout_records = labelled[-n_holdout:]

    # Build region encoding from training set
    regions_seen = sorted({r.region for r in train_records if r.region})
    region_map = {r: i for i, r in enumerate(regions_seen)}

    X_train, feature_names = _extract_lgbm_features(train_records, region_map)
    y_train = np.array([r.realized_savings for r in train_records], dtype=float)

    X_holdout, _ = _extract_lgbm_features(holdout_records, region_map)
    y_holdout = np.array([r.realized_savings for r in holdout_records], dtype=float)

    # Naive baseline: predict the training mean on holdout
    naive_pred = np.full(len(y_holdout), float(np.mean(y_train)))
    naive_rmse = float(np.sqrt(np.mean((y_holdout - naive_pred) ** 2)))

    # Train LightGBM regressor
    import lightgbm as lgb
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": max(5, len(train_records) // 20),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate on holdout
    y_pred_holdout = model.predict(X_holdout)
    model_rmse = float(np.sqrt(np.mean((y_holdout - y_pred_holdout) ** 2)))
    model_mae = float(np.mean(np.abs(y_holdout - y_pred_holdout)))

    beats_naive = model_rmse < naive_rmse
    logger.info(
        f"train_savings_model_lgbm: "
        f"model_rmse={model_rmse:.4f}, naive_rmse={naive_rmse:.4f}, "
        f"beats_naive={beats_naive}, "
        f"n_train={len(train_records)}, n_holdout={len(holdout_records)}"
    )

    # Serialize model using LightGBM's native text format
    model_string = model.booster_.model_to_string()

    return {
        "version": 2,
        "generated_at_utc": generate_timestamp_utc(),
        "method": "lightgbm_regression",
        "feature_names": feature_names,
        "region_map": region_map,
        "model_string": model_string,
        "metrics": {
            "model_rmse_holdout": round(model_rmse, 4),
            "naive_mean_rmse_holdout": round(naive_rmse, 4),
            "model_mae_holdout": round(model_mae, 4),
            "beats_naive_baseline": beats_naive,
            "n_train": len(train_records),
            "n_holdout": len(holdout_records),
        },
        "params": params,
    }


# ============================================================================
# 8. LIGHTGBM RISK PRIORS MODEL
# ============================================================================

def train_risk_priors_lgbm(
    records: list[TrainingRecord],
    error_models: dict[str, Any],
    seed: int = 42,
    min_records: int = _MIN_LGBM_RECORDS,
) -> dict[str, Any]:
    """Train a LightGBM binary classification model for p90 coverage risk.

    Target: 1 if p90 was NOT covered (risk event), 0 if covered.
    Falls back to weighted empirical risk if insufficient data or LightGBM unavailable.

    Args:
        records: Training records from PostExecutionRecord.
        error_models: Output from train_error_models() (used by fallback).
        seed: Random seed.
        min_records: Minimum labelled records needed for LightGBM training.

    Returns:
        Artifact dict with model_string, metrics, and metadata.
    """
    # Need energy_cost_p90_covered labels
    labelled = [r for r in records if r.energy_cost_p90_covered is not None]

    if len(labelled) < min_records:
        logger.warning(
            f"train_risk_priors_lgbm: only {len(labelled)} labelled records "
            f"(need {min_records}); falling back to empirical risk"
        )
        result = train_risk_priors(records, error_models)
        result["method"] = "weighted_empirical_risk_fallback"
        result["lgbm_skipped_reason"] = f"insufficient_records:{len(labelled)}<{min_records}"
        return result

    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        logger.warning("train_risk_priors_lgbm: lightgbm not available; falling back")
        result = train_risk_priors(records, error_models)
        result["method"] = "weighted_empirical_risk_fallback"
        result["lgbm_skipped_reason"] = "lightgbm_not_installed"
        return result

    # Time-ordered split
    n_holdout = max(1, int(len(labelled) * _LGBM_HOLDOUT_FRAC))
    train_records = labelled[:-n_holdout]
    holdout_records = labelled[-n_holdout:]

    regions_seen = sorted({r.region for r in train_records if r.region})
    region_map = {r: i for i, r in enumerate(regions_seen)}

    X_train, feature_names = _extract_lgbm_features(train_records, region_map)
    # y=1 means risk event (p90 NOT covered)
    y_train = np.array(
        [0.0 if r.energy_cost_p90_covered else 1.0 for r in train_records], dtype=float
    )

    X_holdout, _ = _extract_lgbm_features(holdout_records, region_map)
    y_holdout = np.array(
        [0.0 if r.energy_cost_p90_covered else 1.0 for r in holdout_records], dtype=float
    )

    # Naive baseline: always predict training class rate
    naive_rate = float(np.mean(y_train))
    naive_log_loss = _log_loss_scalar(y_holdout, np.full(len(y_holdout), naive_rate))

    import lightgbm as lgb
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "min_child_samples": max(5, len(train_records) // 20),
        "subsample": 0.8,
        "seed": seed,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_holdout)[:, 1]
    model_log_loss = _log_loss_scalar(y_holdout, y_pred_proba)
    beats_naive = model_log_loss < naive_log_loss

    logger.info(
        f"train_risk_priors_lgbm: "
        f"model_logloss={model_log_loss:.4f}, naive_logloss={naive_log_loss:.4f}, "
        f"beats_naive={beats_naive}, "
        f"n_train={len(train_records)}, n_holdout={len(holdout_records)}"
    )

    model_string = model.booster_.model_to_string()

    return {
        "version": 2,
        "generated_at_utc": generate_timestamp_utc(),
        "method": "lightgbm_binary_classification",
        "feature_names": feature_names,
        "region_map": region_map,
        "model_string": model_string,
        "metrics": {
            "model_logloss_holdout": round(model_log_loss, 4),
            "naive_logloss_holdout": round(naive_log_loss, 4),
            "beats_naive_baseline": beats_naive,
            "risk_event_rate_train": round(naive_rate, 4),
            "n_train": len(train_records),
            "n_holdout": len(holdout_records),
        },
        "params": params,
    }


def _log_loss_scalar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary log-loss, clipping predictions for numerical stability."""
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Trainers Module Inline Tests")
    print("=" * 60)

    # Create test records
    test_records = [
        TrainingRecord(
            job_id="job-1",
            region="us-east",
            hour_utc=10,
            forecast_energy_cost_p50=100.0,
            forecast_energy_cost_p90=120.0,
            energy_cost_p50_error=5.0,  # realized was 105
            energy_cost_p90_covered=True,  # 105 <= 120
            forecast_carbon_p50=50.0,
            carbon_p50_error=-2.0,  # realized was 48
            carbon_p90_covered=True,
            realized_savings=10.0,
            decision_outcome_label="good_decision",
            constraint_profile="batch_optimized",
        ),
        TrainingRecord(
            job_id="job-2",
            region="us-east",
            hour_utc=10,
            forecast_energy_cost_p50=100.0,
            forecast_energy_cost_p90=120.0,
            energy_cost_p50_error=-3.0,  # realized was 97
            energy_cost_p90_covered=True,
            forecast_carbon_p50=50.0,
            carbon_p50_error=1.0,
            carbon_p90_covered=True,
            realized_savings=15.0,
            decision_outcome_label="good_decision",
            constraint_profile="batch_optimized",
        ),
        TrainingRecord(
            job_id="job-3",
            region="us-west",
            hour_utc=14,
            forecast_energy_cost_p50=80.0,
            forecast_energy_cost_p90=100.0,
            energy_cost_p50_error=25.0,  # realized was 105 (exceeded p90!)
            energy_cost_p90_covered=False,
            realized_savings=-5.0,
            decision_outcome_label="neutral",
            constraint_profile="latency_safe",
        ),
    ]

    # Test 1: Forecast corrections
    print("\n[Test 1] Forecast corrections training")
    corrections = train_forecast_corrections(test_records)
    assert corrections["version"] == 1
    assert corrections["method"] == "bucketed_mean_residual"
    assert len(corrections["buckets"]) >= 1
    # us-east, hour 10: mean of [5.0, -3.0] = 1.0
    us_east_bucket = next(
        (b for b in corrections["buckets"] if b["region"] == "us-east" and b.get("hour_utc") == 10),
        None
    )
    assert us_east_bucket is not None
    assert us_east_bucket["energy_cost_p50_bias"] == 1.0  # (5 + -3) / 2
    print(f"  PASSED: {len(corrections['buckets'])} buckets created")

    # Test 2: Error models
    print("\n[Test 2] Error models training")
    error_models = train_error_models(test_records)
    assert error_models["version"] == 1
    assert len(error_models["buckets"]) >= 1
    print(f"  PASSED: {len(error_models['buckets'])} buckets created")

    # Test 3: Uncertainty rules
    print("\n[Test 3] Uncertainty rules generation")
    uncertainty_rules = generate_uncertainty_rules(error_models)
    assert uncertainty_rules["version"] == 1
    assert len(uncertainty_rules["rules"]) >= 1
    assert uncertainty_rules["rules"][0]["name"] == "widen_if_energy_p90_coverage_low"
    print(f"  PASSED: {len(uncertainty_rules['rules'])} rules generated")

    # Test 4: Savings model
    print("\n[Test 4] Savings model training")
    savings_model = train_savings_model(test_records)
    assert savings_model["version"] == 1
    assert len(savings_model["buckets"]) >= 1
    us_east_savings = next(
        (b for b in savings_model["buckets"] if b["region"] == "us-east"),
        None
    )
    assert us_east_savings is not None
    # Mean of [10, 15] = 12.5
    assert us_east_savings["mean_savings"] == 12.5
    print(f"  PASSED: {len(savings_model['buckets'])} buckets created")

    # Test 5: Risk priors
    print("\n[Test 5] Risk priors training")
    risk_priors = train_risk_priors(test_records, error_models)
    assert risk_priors["version"] == 1
    assert "weights" in risk_priors
    assert risk_priors["weights"]["coverage_shortfall"] == 0.6
    print(f"  PASSED: {len(risk_priors['buckets'])} buckets created")

    # Test 6: Determinism
    print("\n[Test 6] Determinism check")
    corrections1 = train_forecast_corrections(test_records)
    corrections2 = train_forecast_corrections(test_records)
    # Compare buckets (ignoring timestamp)
    assert len(corrections1["buckets"]) == len(corrections2["buckets"])
    for b1, b2 in zip(corrections1["buckets"], corrections2["buckets"]):
        assert b1["region"] == b2["region"]
        assert b1["energy_cost_p50_bias"] == b2["energy_cost_p50_bias"]
    print("  PASSED: Same input produces same buckets")

    # Test 7: Empty records handling
    print("\n[Test 7] Empty records handling")
    empty_corrections = train_forecast_corrections([])
    assert empty_corrections["buckets"] == []
    empty_savings = train_savings_model([])
    assert empty_savings["buckets"] == []
    print("  PASSED: Empty input produces empty buckets")

    # Test 8: Null field handling
    print("\n[Test 8] Null field handling")
    null_records = [
        TrainingRecord(job_id="job-null", region="us-east"),
        TrainingRecord(job_id="job-null2"),  # No region
    ]
    corrections = train_forecast_corrections(null_records)
    # Should not crash, may have empty buckets
    print("  PASSED: Null fields handled without crash")

    print("\n" + "=" * 60)
    print("All 8 tests passed!")
    print("=" * 60)
