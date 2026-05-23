# Pilot Deck Outline

A concise structure for a first pilot conversation with an infrastructure buyer.
Eleven slides. The arc moves from a quantified problem to validated evidence to
a low-risk pilot path. It is built to survive a technical and a procurement
reviewer in the same room: lead with the cost mechanism, substantiate with
leakage-free evidence, and close on a read-only pilot.

Keep claims conservative throughout: savings are workload-dependent and
pilot-validated; the historical benchmark is a prior, not a guarantee.

---

**1 — Title / positioning**
Aurelius: an infrastructure orchestration layer that reduces GPU operating cost
through SLA-aware placement and market-aware scheduling. One line on the
category, one line on the audience.

**2 — The problem**
GPU energy is priced in markets that move hour to hour and region to region.
Most schedulers ignore this. Flexible workloads — training, batch inference,
data processing, maintenance — carry unused scheduling slack. The overspend is
structural and invisible per-invoice.

**3 — The approach**
Forecast near-term prices; place each flexible job in the cheapest region and
hour within its constraints. Same work, better time and place. Aurelius decides;
the customer's scheduler executes.

**4 — Where savings come from**
Three mechanisms: time-shifting (largest), region routing, migration of eligible
jobs (smallest). All scale with flexibility — so savings are highest for batch
and maintenance, lowest for latency-hard inference.

**5 — Validated results**
Mean 25.0% (Q1 2026) and 22.8% (Summer 2025) reduction versus a strong
`current_price_only` baseline, on real CAISO/PJM/ERCOT day-ahead prices,
leakage-free walk-forward. Per-workload table. State plainly: observed in
historical replay, not guaranteed.

**6 — Why the evidence is credible**
Strongest realistic baseline; leakage-free walk-forward; 0% missing hours; real
ISO data only (synthetic data barred from claims); reproducible from committed
data and a fixed seed. One line on the upper-bound diagnostic as the honesty
check.

**7 — Safety and control boundary**
Advisory, reversible decisions. Deterministic fallback to baseline on missing or
low-confidence forecasts (fail-closed safety gate). Aurelius owns the decision;
the platform owns execution. Worst case is a valid-but-suboptimal placement.

**8 — Deployment model**
Read-only by default. Minimum input: a workload trace plus read-only market
access. No custody of workloads, data, or weights. Three modes: offline replay,
shadow, controlled execution (opt-in, signed, kill switch).

**9 — The pilot**
Three evidence-gated phases: offline replay → shadow validation on the
customer's own workload and markets → optional controlled execution. The
decision to proceed rests on the shadow-mode result, not the benchmark.

**10 — What's validated vs. what needs integration**
Validated: Tier 1 region/time on three U.S. markets. Needs customer
integration: Tier 2 queue-aware, Tier 3 GPU/node-level, EU markets, broader
carbon coverage. Stated as roadmap, not as current capability.

**11 — Ask and next step**
Request a 30–90 day workload trace and confirmation of regions and SLA classes.
First deliverable: an offline-replay projection, then a shadow-mode validation
on their footprint. Note SOC 2 is on the roadmap; pilot runs read-only.
