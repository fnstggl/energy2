"""Phase 12 — Aurelius internal observability metrics.

Exports Aurelius operational state as Prometheus-compatible text for scraping.
Thread-safe. No Prometheus client library required.

Collected per ConstraintAwareEngine.run() cycle:
  - constraints_detected_total     (counter, by constraint type)
  - recommendations_generated_total (counter, by action_type)
  - recommendations_blocked_by_sla_total (counter)
  - estimated_net_savings_dollars  (counter, accumulated from net_benefit)
  - confidence_current             (gauge, last engine run)
  - connector_health               (gauge, per connector name)
  - stale_data_count               (gauge, last cycle)

Design rules:
  - Missing/unknown values → metrics absent or 0, never fabricated
  - Secrets are never recorded (no URL, no auth headers)
  - Thread-safe via threading.Lock
  - to_prometheus_text() produces valid Prometheus exposition format
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .engine import EngineResult


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConnectorHealth:
    """Health status for a single connector, as reported by the connector itself."""
    name: str
    is_healthy: bool
    stale_metric_count: int = 0
    last_error: Optional[str] = None


@dataclass
class AureliusMetrics:
    """Point-in-time snapshot of Aurelius internal operational metrics."""
    constraints_detected: dict[str, int]          # constraint_type → count
    recommendations_generated: dict[str, int]     # action_type → count
    recommendations_blocked_by_sla: int
    estimated_net_savings_dollars: float
    confidence_current: Optional[float]           # None before first cycle
    connector_health: dict[str, bool]             # connector_name → is_healthy
    stale_data_count: int
    total_engine_cycles: int


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class AureliusObserver:
    """Thread-safe collector of Aurelius operational metrics.

    Usage::

        observer = AureliusObserver()
        # in engine loop:
        result = engine.run(state)
        observer.record_engine_result(result)
        # from connector health checks:
        observer.record_connector_health("dcgm", is_healthy=True, stale_count=0)
        # export:
        print(observer.to_prometheus_text())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._constraints_detected: dict[str, int] = defaultdict(int)
        self._recs_generated: dict[str, int] = defaultdict(int)
        self._blocked_by_sla: int = 0
        self._net_savings: float = 0.0
        self._confidence_current: Optional[float] = None
        self._connector_health: dict[str, bool] = {}
        self._stale_data_count: int = 0
        self._total_cycles: int = 0

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_engine_result(self, result: "EngineResult") -> None:
        """Extract and accumulate metrics from one ConstraintAwareEngine.run() cycle."""
        with self._lock:
            self._total_cycles += 1

            # Binding constraint from assessment
            bc = result.assessment.binding_constraint
            if bc is not None:
                self._constraints_detected[bc.value] += 1

            # Per-recommendation counts
            for rec in result.recommendations:
                self._recs_generated[rec.action_type] += 1
                if not rec.is_noop and rec.net_benefit is not None and rec.net_benefit > 0:
                    self._net_savings += rec.net_benefit

            # SLA-blocked rejections
            for rej in result.rejected:
                reason = rej.get("reject_reason", "")
                if reason.startswith("sla_gate"):
                    self._blocked_by_sla += 1

            # Confidence gauge (last cycle wins)
            self._confidence_current = result.assessment.confidence

    def record_connector_health(
        self,
        name: str,
        is_healthy: bool,
        stale_count: int = 0,
    ) -> None:
        """Record health status for a named connector.

        Args:
            name:        connector identifier, e.g. "dcgm", "vllm", "kubernetes"
            is_healthy:  True if the connector returned fresh data without error
            stale_count: number of stale metrics detected (0 if healthy)
        """
        with self._lock:
            self._connector_health[name] = is_healthy
            self._stale_data_count = sum(
                stale_count if k == name else 0
                for k in self._connector_health
            )
            # Re-accumulate stale_data_count cleanly from the per-connector map
            # (we track it as a simple gauge over all connectors in this cycle)
            self._stale_data_count = stale_count  # last-write-wins gauge

    def record_stale_data_count(self, count: int) -> None:
        """Record total stale metric sources detected in the last cycle (gauge)."""
        with self._lock:
            self._stale_data_count = count

    def get_metrics(self) -> AureliusMetrics:
        """Return a point-in-time snapshot of all metrics."""
        with self._lock:
            return AureliusMetrics(
                constraints_detected=dict(self._constraints_detected),
                recommendations_generated=dict(self._recs_generated),
                recommendations_blocked_by_sla=self._blocked_by_sla,
                estimated_net_savings_dollars=self._net_savings,
                confidence_current=self._confidence_current,
                connector_health=dict(self._connector_health),
                stale_data_count=self._stale_data_count,
                total_engine_cycles=self._total_cycles,
            )

    def reset(self) -> None:
        """Reset all counters. Useful between benchmark runs / test isolation."""
        with self._lock:
            self._constraints_detected.clear()
            self._recs_generated.clear()
            self._blocked_by_sla = 0
            self._net_savings = 0.0
            self._confidence_current = None
            self._connector_health.clear()
            self._stale_data_count = 0
            self._total_cycles = 0

    # ------------------------------------------------------------------
    # Prometheus exposition
    # ------------------------------------------------------------------

    def to_prometheus_text(self) -> str:
        """Export all metrics in Prometheus text exposition format (version 0.0.4).

        The output can be scraped by any Prometheus-compatible server or pushed
        to a Pushgateway.  No Prometheus client library is required.

        Returns:
            Valid Prometheus text exposition string, newline-terminated.
        """
        m = self.get_metrics()
        lines: list[str] = []

        def _append(help_text: str, metric_type: str, name: str, value: float,
                    labels: Optional[dict[str, str]] = None) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            label_str = ""
            if labels:
                kv = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                label_str = "{" + kv + "}"
            lines.append(f"{name}{label_str} {_fmt(value)}")

        def _fmt(v: float) -> str:
            if v == int(v):
                return str(int(v))
            return f"{v:.6g}"

        # --- constraints_detected_total (counter, labeled by type) ---
        lines.append("# HELP aurelius_constraints_detected_total "
                     "Total constraint detection events by constraint type")
        lines.append("# TYPE aurelius_constraints_detected_total counter")
        if m.constraints_detected:
            for ctype, count in sorted(m.constraints_detected.items()):
                lines.append(
                    f'aurelius_constraints_detected_total{{constraint="{ctype}"}} {count}'
                )
        else:
            lines.append('aurelius_constraints_detected_total{constraint="none"} 0')

        lines.append("")

        # --- recommendations_generated_total (counter, labeled by action_type) ---
        lines.append("# HELP aurelius_recommendations_generated_total "
                     "Total recommendations generated by action type")
        lines.append("# TYPE aurelius_recommendations_generated_total counter")
        if m.recommendations_generated:
            for atype, count in sorted(m.recommendations_generated.items()):
                lines.append(
                    f'aurelius_recommendations_generated_total{{action_type="{atype}"}} {count}'
                )
        else:
            lines.append(
                'aurelius_recommendations_generated_total{action_type="KEEP"} 0'
            )

        lines.append("")

        # --- recommendations_blocked_by_sla_total (counter) ---
        lines.append("# HELP aurelius_recommendations_blocked_by_sla_total "
                     "Total recommendations blocked by SLA gate")
        lines.append("# TYPE aurelius_recommendations_blocked_by_sla_total counter")
        lines.append(f"aurelius_recommendations_blocked_by_sla_total "
                     f"{m.recommendations_blocked_by_sla}")

        lines.append("")

        # --- estimated_net_savings_dollars (counter, accumulated) ---
        lines.append("# HELP aurelius_estimated_net_savings_dollars "
                     "Accumulated estimated net savings from non-noop recommendations (dollars)")
        lines.append("# TYPE aurelius_estimated_net_savings_dollars counter")
        lines.append(f"aurelius_estimated_net_savings_dollars "
                     f"{_fmt(m.estimated_net_savings_dollars)}")

        lines.append("")

        # --- confidence_current (gauge) ---
        lines.append("# HELP aurelius_confidence_current "
                     "Constraint classifier confidence from last engine cycle [0, 1]")
        lines.append("# TYPE aurelius_confidence_current gauge")
        if m.confidence_current is not None:
            lines.append(f"aurelius_confidence_current {_fmt(m.confidence_current)}")
        else:
            lines.append("# no engine cycle recorded yet")

        lines.append("")

        # --- connector_health (gauge, per connector) ---
        lines.append("# HELP aurelius_connector_health "
                     "Connector health status (1=healthy, 0=unhealthy)")
        lines.append("# TYPE aurelius_connector_health gauge")
        if m.connector_health:
            for conn, healthy in sorted(m.connector_health.items()):
                lines.append(
                    f'aurelius_connector_health{{connector="{conn}"}} '
                    f'{"1" if healthy else "0"}'
                )
        else:
            lines.append("# no connector health reported yet")

        lines.append("")

        # --- stale_data_count (gauge) ---
        _append(
            "Number of stale metric sources detected in last engine cycle",
            "gauge",
            "aurelius_stale_data_count",
            float(m.stale_data_count),
        )

        lines.append("")

        # --- total_engine_cycles (counter) ---
        _append(
            "Total ConstraintAwareEngine.run() cycles completed",
            "counter",
            "aurelius_engine_cycles_total",
            float(m.total_engine_cycles),
        )

        lines.append("")  # Prometheus text must end with a newline
        return "\n".join(lines) + "\n"
