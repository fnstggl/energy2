"""Before/after SLA-aware optimization report.

Produces the human-readable report described in the spec, comparing the
unconstrained (savings-only) recommendation against the SLA-aware decision,
listing what was blocked and how much savings were sacrificed for SLA safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .selector import SLADecision


@dataclass
class SLAReport:
    """Aggregates per-workload SLA decisions into a report."""

    decisions: list[SLADecision] = field(default_factory=list)

    def add(self, decision: SLADecision) -> None:
        self.decisions.append(decision)

    def to_dict(self) -> dict:
        total_unconstrained = sum(d.unconstrained_savings_pct for d in self.decisions)
        total_chosen = sum(d.chosen_savings_pct for d in self.decisions)
        total_sacrificed = sum(d.savings_sacrificed_pct for d in self.decisions)
        return {
            "workloads": len(self.decisions),
            "corrected_count": sum(1 for d in self.decisions if d.was_corrected),
            "total_unconstrained_savings_pct": round(total_unconstrained, 4),
            "total_sla_aware_savings_pct": round(total_chosen, 4),
            "total_savings_sacrificed_pct": round(total_sacrificed, 4),
            "decisions": [d.to_dict() for d in self.decisions],
        }

    def render_text(self) -> str:
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append("AURELIUS SLA-AWARE OPTIMIZATION REPORT")
        lines.append("=" * 70)
        for d in self.decisions:
            lines.append("")
            lines.append(f"Workload: {d.workload_id}")
            ua = d.unconstrained_action
            ca = d.chosen_action
            ua_desc = _action_desc(ua)
            ca_desc = _action_desc(ca)
            lines.append(f"Unconstrained action: {ua_desc}")
            lines.append(f"Unconstrained expected savings: {d.unconstrained_savings_pct:.1f}%")
            lines.append(f"SLA-aware action: {ca_desc}")
            lines.append(f"SLA-aware expected savings: {d.chosen_savings_pct:.1f}%")
            if d.blocked_reasons:
                lines.append("Blocked because:")
                for r in d.blocked_reasons:
                    lines.append(f"  - {r}")
            if d.was_corrected:
                lines.append(
                    f"Savings sacrificed for SLA safety: {d.savings_sacrificed_pct:.1f}%"
                )
                if d.risk_avoided:
                    lines.append(f"SLA risk avoided (risk score): {d.risk_avoided:.2f}")
            else:
                lines.append("No hard SLA violations")
        lines.append("")
        lines.append("-" * 70)
        agg = self.to_dict()
        lines.append(
            f"TOTAL: {agg['workloads']} workloads, "
            f"{agg['corrected_count']} corrected, "
            f"{agg['total_savings_sacrificed_pct']:.1f}% savings sacrificed for SLA safety"
        )
        lines.append("=" * 70)
        return "\n".join(lines)


def _action_desc(action) -> str:
    if action.is_noop:
        region = action.target_region or "current"
        return f"keep {region}"
    if action.target_region:
        return f"{action.action_type.value} -> {action.target_region}"
    return action.action_type.value
