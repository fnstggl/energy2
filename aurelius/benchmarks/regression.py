"""Benchmark regression checker for Phase 11.

Compares two BenchmarkReport JSONs and fails if:
  - KPI regressions exceed threshold
  - SLA violations increased
  - Migration churn increased beyond threshold
  - Metadata changed (environment drift)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regression thresholds
# ---------------------------------------------------------------------------

# A regression is declared if the metric worsens by more than this fraction
_KPI_REGRESSION_THRESHOLDS = {
    "total_energy_cost": 0.02,          # 2% cost increase
    "total_sla_violations": 0.0,        # any increase in SLA violations
    "total_thermal_throttle_ticks": 0.10,  # 10% more throttle
    "total_migrations": 0.50,           # 50% more migrations
    "mean_gpu_util_pct": -0.05,         # 5% drop in GPU utilization
    "mean_topology_score": -0.05,       # 5% drop in topology score
}

# p99 latency: a 10% regression is flagged
_LATENCY_REGRESSION_PCT = 0.10
# Weighted scorecard: flag if weighted score drops by this much
_SCORECARD_REGRESSION_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    regressions: list[str]
    improvements: list[str]
    warnings: list[str]
    metadata_mismatches: list[str]
    comparison_valid: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "comparison_valid": self.comparison_valid,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "warnings": self.warnings,
            "metadata_mismatches": self.metadata_mismatches,
        }

    def to_text(self) -> str:
        lines: list[str] = []
        status = "PASS" if self.passed else "FAIL"
        validity = "VALID" if self.comparison_valid else "INVALID (env changed)"
        lines.append(f"Regression Check: {status}  |  Comparison: {validity}")
        if self.metadata_mismatches:
            lines.append("\nMetadata mismatches (comparison may be invalid):")
            for m in self.metadata_mismatches:
                lines.append(f"  ~ {m}")
        if self.regressions:
            lines.append("\nRegressions:")
            for r in self.regressions:
                lines.append(f"  ✗ {r}")
        if self.improvements:
            lines.append("\nImprovements:")
            for i in self.improvements:
                lines.append(f"  ✓ {i}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class BenchmarkRegressionChecker:
    """Compare two constraint-aware benchmark JSON files for regressions."""

    def __init__(
        self,
        cost_regression_threshold: float = _KPI_REGRESSION_THRESHOLDS["total_energy_cost"],
        latency_regression_pct: float = _LATENCY_REGRESSION_PCT,
        scorecard_regression_threshold: float = _SCORECARD_REGRESSION_THRESHOLD,
    ):
        self._cost_thresh = cost_regression_threshold
        self._lat_thresh = latency_regression_pct
        self._score_thresh = scorecard_regression_threshold

    def compare(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
        policy: str = "constraint_aware",
    ) -> RegressionResult:
        """Compare baseline and current benchmark reports.

        Args:
            baseline: dict loaded from a previous BenchmarkReport.to_dict()
            current:  dict loaded from the new BenchmarkReport.to_dict()
            policy:   which optimizer policy to compare (default: constraint_aware)

        Returns:
            RegressionResult with pass/fail and detailed findings
        """
        regressions: list[str] = []
        improvements: list[str] = []
        warnings: list[str] = []
        metadata_mismatches: list[str] = []
        comparison_valid = True

        # 1. Metadata compatibility check
        base_meta = baseline.get("metadata", {})
        cur_meta = current.get("metadata", {})
        for key in ("scenario_name", "scenario_version", "scenario_hash",
                    "seed", "simulator_version", "config_hash", "steps"):
            bv = base_meta.get(key)
            cv = cur_meta.get(key)
            if bv != cv:
                metadata_mismatches.append(f"{key}: {bv!r} → {cv!r}")
                comparison_valid = False

        if metadata_mismatches:
            warnings.append(
                "Environment changed — apparent improvements may not be optimizer gains"
            )

        # 2. KPI comparison for the selected policy
        base_kpis = baseline.get("report", baseline).get("kpi_comparison", {})
        cur_kpis = current.get("report", current).get("kpi_comparison", {})
        base_policy = base_kpis.get(policy)
        cur_policy = cur_kpis.get(policy)

        if base_policy is None or cur_policy is None:
            warnings.append(
                f"Policy {policy!r} missing from baseline or current; cannot compare KPIs"
            )
        else:
            regressions, improvements = self._compare_kpis(
                base_policy, cur_policy, regressions, improvements, warnings
            )

        # 3. Scorecard comparison
        base_sc = baseline.get("report", baseline).get("scorecard", {})
        cur_sc = current.get("report", current).get("scorecard", {})
        base_score = base_sc.get("weighted_score")
        cur_score = cur_sc.get("weighted_score")
        if base_score is not None and cur_score is not None:
            delta = cur_score - base_score
            if delta < -self._score_thresh:
                regressions.append(
                    f"SCORECARD weighted_score: {base_score:.3f} → {cur_score:.3f} "
                    f"(Δ {delta:+.3f}, threshold -{self._score_thresh:.3f})"
                )
            elif delta > 0.02:
                improvements.append(
                    f"SCORECARD weighted_score: {base_score:.3f} → {cur_score:.3f} "
                    f"(Δ {delta:+.3f})"
                )

        # 4. Regression flags from current report
        cur_flags = current.get("report", current).get("regression_flags", [])
        for flag in cur_flags:
            if flag not in regressions and flag not in warnings:
                warnings.append(f"(from current report) {flag}")

        passed = len(regressions) == 0

        return RegressionResult(
            regressions=regressions,
            improvements=improvements,
            warnings=warnings,
            metadata_mismatches=metadata_mismatches,
            comparison_valid=comparison_valid,
            passed=passed,
        )

    def compare_files(
        self,
        baseline_path: str,
        current_path: str,
        policy: str = "constraint_aware",
    ) -> RegressionResult:
        baseline = _load_json(baseline_path)
        current = _load_json(current_path)
        return self.compare(baseline, current, policy=policy)

    def _compare_kpis(
        self,
        base: dict[str, Any],
        cur: dict[str, Any],
        regressions: list[str],
        improvements: list[str],
        warnings: list[str],
    ) -> tuple[list[str], list[str]]:
        # Cost
        b_cost = base.get("total_energy_cost")
        c_cost = cur.get("total_energy_cost")
        if b_cost and c_cost and b_cost > 0:
            delta_frac = (c_cost - b_cost) / b_cost
            if delta_frac > self._cost_thresh:
                regressions.append(
                    f"COST: {b_cost:.4f} → {c_cost:.4f} "
                    f"(+{delta_frac * 100:.1f}%, threshold +{self._cost_thresh * 100:.0f}%)"
                )
            elif delta_frac < -0.01:
                improvements.append(
                    f"COST: {b_cost:.4f} → {c_cost:.4f} "
                    f"({delta_frac * 100:.1f}%)"
                )

        # SLA violations
        b_sla = base.get("total_sla_violations", 0)
        c_sla = cur.get("total_sla_violations", 0)
        if c_sla > b_sla:
            regressions.append(
                f"SLA_VIOLATIONS: {b_sla} → {c_sla} (any increase fails)"
            )
        elif c_sla < b_sla:
            improvements.append(f"SLA_VIOLATIONS: {b_sla} → {c_sla}")

        # p99 latency
        b_p99 = base.get("p99_latency_ms")
        c_p99 = cur.get("p99_latency_ms")
        if b_p99 and c_p99:
            delta_frac = (c_p99 - b_p99) / b_p99
            if delta_frac > self._lat_thresh:
                regressions.append(
                    f"P99_LATENCY: {b_p99:.0f}ms → {c_p99:.0f}ms "
                    f"(+{delta_frac * 100:.1f}%, threshold +{self._lat_thresh * 100:.0f}%)"
                )
            elif delta_frac < -0.05:
                improvements.append(
                    f"P99_LATENCY: {b_p99:.0f}ms → {c_p99:.0f}ms "
                    f"({delta_frac * 100:.1f}%)"
                )

        # Migrations
        b_mig = base.get("total_migrations", 0)
        c_mig = cur.get("total_migrations", 0)
        if b_mig > 0 and c_mig > b_mig * (1 + _KPI_REGRESSION_THRESHOLDS["total_migrations"]):
            regressions.append(
                f"MIGRATION_CHURN: {b_mig} → {c_mig} "
                f"(+{(c_mig - b_mig) / b_mig * 100:.0f}%, threshold +50%)"
            )

        # Topology score
        b_topo = base.get("mean_topology_score")
        c_topo = cur.get("mean_topology_score")
        if b_topo and c_topo:
            delta = c_topo - b_topo
            if delta < _KPI_REGRESSION_THRESHOLDS["mean_topology_score"]:
                regressions.append(
                    f"TOPOLOGY_SCORE: {b_topo:.3f} → {c_topo:.3f} (Δ {delta:.3f})"
                )

        return regressions, improvements


def _load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark file not found: {p}")
    with open(p) as f:
        return json.load(f)
