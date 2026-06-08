"""Phase 11.4-11.7 — carbon actually changes (or correctly does not change) placement.

Scenarios:
 4. Constant emissions -> carbon does not change placement (cost decides).
 5. Equal prices, varying MOER -> prefer the lower-emission region.
 6. Cheap-dirty vs expensive-clean -> behaviour matches the configured objective.
 7. Carbon-disabled ablation -> decisions differ only when carbon should matter.
"""

from datetime import datetime, timezone

import pytest

from aurelius.carbon.candidate import CandidatePlacement, evaluate_carbon_candidates
from aurelius.carbon.constraints import CarbonConstraints
from aurelius.carbon.scoring import ScoreWeights

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)


def _placements(regions):
    return [
        CandidatePlacement(
            region=r, start_time=T0, power_kw=100.0, duration_hours=1.0,
            utilization_fraction=1.0, pue=1.0,
        )
        for r in regions
    ]


def _evaluate(prices, moers, *, weights=None, carbon_price=None, constraints=None):
    price_data = {r: {T0: p} for r, p in prices.items()}
    moer_data = {r: {T0: m} for r, m in moers.items()}
    return evaluate_carbon_candidates(
        job_id="j1", scheduler_name="opt",
        placements=_placements(list(prices)),
        price_data=price_data, moer_data=moer_data,
        weights=weights, carbon_price_usd_per_tonne=carbon_price,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# 4. Constant emissions -> carbon does not change placement
# ---------------------------------------------------------------------------

class TestConstantEmissions:
    def test_carbon_does_not_change_placement(self):
        prices = {"us-west": 100.0, "us-east": 50.0}   # us-east cheaper
        moers = {"us-west": 300.0, "us-east": 300.0}   # identical carbon
        no_carbon = _evaluate(prices, moers, weights=ScoreWeights(beta=0.0))
        heavy_carbon = _evaluate(prices, moers, weights=ScoreWeights(beta=10.0))
        # Cheaper region wins in both; carbon weight is irrelevant when MOER is flat.
        assert no_carbon.best.placement.region == "us-east"
        assert heavy_carbon.best.placement.region == "us-east"


# ---------------------------------------------------------------------------
# 5. Equal prices, varying MOER -> prefer lower-emission region
# ---------------------------------------------------------------------------

class TestEqualPriceVaryingMoer:
    def test_prefers_lower_emission_region(self):
        prices = {"us-west": 100.0, "us-east": 100.0}  # identical price
        moers = {"us-west": 200.0, "us-east": 500.0}   # us-west cleaner
        res = _evaluate(prices, moers, weights=ScoreWeights(beta=1.0))
        assert res.best.placement.region == "us-west"
        # And the chosen candidate really does have lower forecast emissions.
        assert res.best.forecast_emissions_kgco2 == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 6. Cheap-dirty vs expensive-clean
# ---------------------------------------------------------------------------

class TestCheapDirtyVsExpensiveClean:
    PRICES = {"cheap_dirty": 50.0, "expensive_clean": 100.0}
    MOERS = {"cheap_dirty": 600.0, "expensive_clean": 100.0}

    def test_cost_only_picks_cheap_dirty(self):
        res = _evaluate(self.PRICES, self.MOERS, weights=ScoreWeights(beta=0.0))
        assert res.best.placement.region == "cheap_dirty"

    def test_heavy_carbon_weight_picks_clean(self):
        res = _evaluate(self.PRICES, self.MOERS, weights=ScoreWeights(alpha=1.0, beta=5.0))
        assert res.best.placement.region == "expensive_clean"

    def test_carbon_price_flips_to_clean(self):
        # Explicit carbon price (Option B). At $1000/tonne the dirty option's
        # carbon cost dominates its energy savings.
        cheap = _evaluate(self.PRICES, self.MOERS, carbon_price=0.0)
        priced = _evaluate(self.PRICES, self.MOERS, carbon_price=1000.0)
        assert cheap.best.placement.region == "cheap_dirty"
        assert priced.best.placement.region == "expensive_clean"

    def test_intensity_constraint_rejects_dirty(self):
        res = _evaluate(
            self.PRICES, self.MOERS, weights=ScoreWeights(beta=0.0),
            constraints=CarbonConstraints(max_emissions_intensity_gco2_per_kwh=400.0),
        )
        # Dirty region exceeds the intensity cap and is rejected outright, so the
        # clean region wins even under a cost-only objective.
        assert res.best.placement.region == "expensive_clean"
        rejected_regions = {c.placement.region for c in res.rejected}
        assert "cheap_dirty" in rejected_regions


# ---------------------------------------------------------------------------
# 7. Carbon-disabled ablation
# ---------------------------------------------------------------------------

class TestCarbonDisabledAblation:
    def test_decisions_differ_when_carbon_matters(self):
        prices = {"cheap_dirty": 50.0, "expensive_clean": 100.0}
        moers = {"cheap_dirty": 600.0, "expensive_clean": 100.0}
        disabled = _evaluate(prices, moers, weights=ScoreWeights(beta=0.0))
        enabled = _evaluate(prices, moers, weights=ScoreWeights(beta=5.0))
        assert disabled.best.placement.region != enabled.best.placement.region

    def test_decisions_same_when_carbon_irrelevant(self):
        # Flat MOER: enabling/disabling carbon must NOT change the placement.
        prices = {"us-west": 100.0, "us-east": 50.0}
        moers = {"us-west": 300.0, "us-east": 300.0}
        disabled = _evaluate(prices, moers, weights=ScoreWeights(beta=0.0))
        enabled = _evaluate(prices, moers, weights=ScoreWeights(beta=5.0))
        assert disabled.best.placement.region == enabled.best.placement.region
