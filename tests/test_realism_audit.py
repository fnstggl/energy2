"""Tests for the simulator realism audit (task spec §8)."""

from aurelius.benchmarks.realism_audit import (
    NOT_PRODUCTION_REALISTIC_YET,
    RealismAuditReport,
    run_realism_audit,
)


def _report() -> RealismAuditReport:
    return run_realism_audit(seed=42)


def test_audit_runs_and_is_deterministic():
    r1 = run_realism_audit(seed=42)
    r2 = run_realism_audit(seed=42)
    v1 = {s.subsystem: s.verdict for s in r1.subsystems}
    v2 = {s.subsystem: s.verdict for s in r2.subsystems}
    assert v1 == v2


def test_overall_verdict_is_capped_until_calibrated():
    # No calibration parameter is measured on real hardware, so the audit MUST
    # refuse to certify production realism.
    r = _report()
    assert r.overall_verdict == NOT_PRODUCTION_REALISTIC_YET


def test_all_required_subsystems_present():
    r = _report()
    names = {s.subsystem for s in r.subsystems}
    assert {"serving", "migration", "telemetry", "actions", "energy", "robustness"} <= names


def test_serving_dynamics_are_realistic():
    r = _report()
    serving = next(s for s in r.subsystems if s.subsystem == "serving")
    qs = {c.question: c.realistic for c in serving.checks}
    # Convex saturation, growing tails, and a batching knee must all hold.
    assert qs["Does queue wait increase convexly near saturation?"]
    assert qs["Are tail latencies fixed multipliers of the mean?"]
    assert qs["Is throughput linear in replicas (no batching knee)?"]


def test_migration_is_not_free():
    r = _report()
    migration = next(s for s in r.subsystems if s.subsystem == "migration")
    free_check = next(c for c in migration.checks if "free" in c.question)
    assert free_check.realistic, "migration must carry a cold-start / cache cost"


def test_telemetry_perfection_is_flagged_as_blocker():
    # The canonical ClusterState hardcodes confidence='high'/is_partial=False.
    # The audit must surface this honestly as a blocker, not hide it.
    r = _report()
    telemetry = next(s for s in r.subsystems if s.subsystem == "telemetry")
    perfect = next(c for c in telemetry.checks if "always perfect" in c.question)
    assert perfect.severity == "blocker"
    assert perfect.realistic is False
    assert telemetry.verdict == "NEEDS_REAL_TELEMETRY"


def test_no_safe_action_states_are_reachable():
    r = _report()
    actions = next(s for s in r.subsystems if s.subsystem == "actions")
    noop = next(c for c in actions.checks if "no-safe-action" in c.question)
    assert noop.realistic, "the engine must be able to emit KEEP / no-op"


def test_actions_mutate_state():
    r = _report()
    actions = next(s for s in r.subsystems if s.subsystem == "actions")
    mutate = next(c for c in actions.checks if "mutate" in c.question)
    assert mutate.realistic


def test_constraint_aware_preserves_sla():
    r = _report()
    robustness = next(s for s in r.subsystems if s.subsystem == "robustness")
    sla = next(c for c in robustness.checks if "SLA" in c.question)
    assert sla.realistic, "constraint_aware must not regress hard SLA compliance vs FIFO"


def test_report_serializes():
    r = _report()
    d = r.to_dict()
    assert "overall_verdict" in d
    assert "subsystems" in d
    assert isinstance(r.to_text(), str)
    assert "OVERALL VERDICT" in r.to_text()


def test_headline_includes_uncalibrated_warning():
    r = _report()
    assert any("uncalibrated" in f or "directional" in f for f in r.headline_findings)
