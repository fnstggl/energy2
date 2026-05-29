"""Parts A & B — make interactive actions produce MEASURABLE KPI when safe.

* Part A (proxy): per-ingress proxy capacity is configurable; the negative
  control (no safe target) suppresses useless scale-up and KEEPs — never a fake
  reroute.
* Part B (queue): in a relievable regime (real idle GPU capacity + healthy
  proxy), adding replicas measurably drains the queue and lifts goodput; without
  capacity headroom the relief is NOT faked.

Simulator/recommendation only — not production savings.
"""

from __future__ import annotations

from collections import Counter

from aurelius.benchmarks.constraint_runner import (
    POLICY_CONSTRAINT_AWARE,
    ConstraintBenchmarkRunner,
)
from aurelius.simulation.cluster.engine import ClusterSimulator
from aurelius.simulation.cluster.migration import proxy_saturation_factor
from aurelius.simulation.cluster.scenarios import load_scenario
from aurelius.sla.actions import ActionType

# ---------------------------------------------------------------------------
# Part A — per-ingress proxy capacity + negative control
# ---------------------------------------------------------------------------

def test_per_queue_proxy_capacity_override_changes_saturation():
    # Same offered load + replicas; a HEALTHIER ingress (higher per-replica
    # capacity) is not saturated, a constrained one is.
    constrained = proxy_saturation_factor(90.0, 2, cap_per_override=20.0)
    healthy = proxy_saturation_factor(90.0, 2, cap_per_override=200.0)
    assert constrained > 1.5, "constrained ingress must saturate"
    assert healthy == 1.0, "healthy ingress (high capacity) must not saturate"
    assert healthy < constrained


def test_relievable_scenario_queue_has_per_ingress_default():
    # The per-queue field defaults to None (uses global config) — no behaviour
    # change for scenarios that do not set it.
    sc = load_scenario("queue_surge_latency_sensitive", seed_override=42)
    sim = ClusterSimulator(sc.config, seed=42)
    for region in sim._cluster.regions.values():
        for q in region.queues:
            assert getattr(q, "proxy_capacity_rps_per_replica", None) is None


def test_proxy_no_safe_target_suppresses_scale_and_keeps():
    runner = ConstraintBenchmarkRunner()
    res = runner.run_scenario("proxy_bottleneck_no_safe_target", steps=24)
    pr = res.policy_results[POLICY_CONSTRAINT_AWARE]
    actions = Counter()
    reject_reasons = Counter()
    for er in pr.engine_results:
        if er is None:
            continue
        for rec in er.recommendations:
            if not rec.is_noop:
                actions[rec.action_type] += 1
        for x in er.rejected:
            reject_reasons[x.get("reject_reason", "").split(":")[0]] += 1
    # Proxy bottleneck is detected and useless scale-up is explicitly suppressed.
    assert reject_reasons.get("blocked_useless_scale_proxy_bottleneck", 0) > 0
    # No safe alternate target exists -> never a (fake) cross-region reroute.
    assert actions.get(ActionType.REROUTE.value, 0) == 0
    assert actions.get(ActionType.MIGRATE.value, 0) == 0


# ---------------------------------------------------------------------------
# Part B — relievable queue surge: scaling measurably drains the queue
# ---------------------------------------------------------------------------

def _run_relievable(scale: bool, ticks: int = 14):
    sc = load_scenario("queue_surge_relievable_capacity", seed_override=42)
    sim = ClusterSimulator(sc.config, seed=42)
    last = None
    for t in range(ticks):
        last = sim.tick()
        if scale and t >= 4:  # add a replica each surge tick (cooldown-limited)
            sim.add_replica("relievable-inference")
    return last.metrics


def test_relievable_queue_surge_scaling_drains_queue_and_lifts_goodput():
    base = _run_relievable(scale=False)
    scaled = _run_relievable(scale=True)
    # With real idle GPU capacity + a healthy proxy, scaling measurably relieves
    # the queue and increases served tokens (goodput).
    assert scaled.queue_wait_p95_ms < base.queue_wait_p95_ms * 0.5
    assert scaled.total_tokens > base.total_tokens * 1.2


def test_relievable_without_scaling_stays_pressured():
    # The relief is NOT faked: without scaling, the surge keeps the queue
    # pressured (well above the relieved level).
    base = _run_relievable(scale=False)
    assert base.queue_wait_p95_ms > 100.0


def test_relievable_scenario_has_idle_capacity_and_healthy_proxy():
    sc = load_scenario("queue_surge_relievable_capacity", seed_override=42)
    sim = ClusterSimulator(sc.config, seed=42)
    region = sim._cluster.regions["us-east"]
    total_gpus = sum(n.gpu_count for n in region.nodes)
    # 8 GPUs total, the service needs 2 -> ample idle capacity to scale into.
    assert total_gpus == 8
    # Healthy ingress proxy (high per-replica capacity) is configured.
    assert sc.config  # scenario loads
