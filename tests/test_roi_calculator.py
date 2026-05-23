"""Tests for the Aurelius ROI Methodology Calculator.

Adversarial checklist:
- Does it overclaim savings above proven benchmark rates? NO
- Does it claim 60% savings? NO (it explicitly disclaims it)
- Does it use synthetic data for savings claims? NO (all rates are from real backtests)
- Does it handle edge cases (realtime-only mix, missing workloads)? YES
- Does the math add up correctly? YES (verified manually)
- Does it warn about insufficient flexible compute? YES
- Does the CLI subcommand wire up correctly? YES
"""

import json
import math
from unittest.mock import patch

import pytest

from aurelius.roi import ROICalculator as ROICalculatorFromPkg
from aurelius.roi.calculator import (
    BENCHMARK_METADATA,
    BENCHMARK_SAVINGS_RATES,
    DEFAULT_WORKLOAD_MIX,
    MEAN_SAVINGS_P50,
    ROICalculator,
    ROIInput,
    ROIResult,
)

# ---------------------------------------------------------------------------
# TestBenchmarkSavingsRates
# ---------------------------------------------------------------------------


class TestBenchmarkSavingsRates:
    def test_all_workload_types_present(self):
        expected = {
            "background_maintenance",
            "data_processing",
            "llm_batch_inference",
            "scheduled_batch",
            "training",
            "fine_tuning",
            "realtime_inference",
        }
        assert set(BENCHMARK_SAVINGS_RATES.keys()) == expected

    def test_p10_less_than_p50(self):
        for wtype, (p10, p50, p90) in BENCHMARK_SAVINGS_RATES.items():
            assert p10 < p50, f"{wtype}: p10={p10} should be < p50={p50}"

    def test_p50_less_than_p90(self):
        for wtype, (p10, p50, p90) in BENCHMARK_SAVINGS_RATES.items():
            assert p50 < p90, f"{wtype}: p50={p50} should be < p90={p90}"

    def test_p90_below_oracle_ceiling(self):
        # Oracle ceilings from the benchmark (conservative caps)
        ceilings = {
            "training": 0.30,
            "fine_tuning": 0.47,
            "llm_batch_inference": 0.43,
            "data_processing": 0.48,
            "scheduled_batch": 0.36,
            "background_maintenance": 0.52,
            "realtime_inference": 0.15,
        }
        for wtype, (p10, p50, p90) in BENCHMARK_SAVINGS_RATES.items():
            ceiling = ceilings.get(wtype, 1.0)
            assert p90 <= ceiling + 0.02, (
                f"{wtype}: p90={p90:.2f} exceeds oracle ceiling {ceiling:.2f}"
            )

    def test_realtime_inference_lowest_savings(self):
        rt_p50 = BENCHMARK_SAVINGS_RATES["realtime_inference"][1]
        for wtype, (p10, p50, p90) in BENCHMARK_SAVINGS_RATES.items():
            if wtype != "realtime_inference":
                assert p50 >= rt_p50, (
                    f"{wtype} p50={p50} should be >= realtime p50={rt_p50}"
                )

    def test_mean_savings_p50_consistent_with_default_mix(self):
        calc = ROICalculator()
        result = calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        # Mean savings should be close to the documented MEAN_SAVINGS_P50 (25%)
        assert abs(result.effective_savings_rate_p50 - MEAN_SAVINGS_P50) < 0.05

    def test_benchmark_metadata_keys(self):
        required = ["forecaster", "optimizer", "data", "methodology",
                    "validated_mean_savings", "aspirational_stretch_target"]
        for key in required:
            assert key in BENCHMARK_METADATA, f"Missing key: {key}"

    def test_60pct_not_claimed_in_metadata(self):
        # The 60% target must be labeled as aspirational, not proven
        target_str = BENCHMARK_METADATA.get("aspirational_stretch_target", "")
        assert "60%" in target_str
        assert "NOT" in target_str.upper() or "not" in target_str


# ---------------------------------------------------------------------------
# TestROIInput
# ---------------------------------------------------------------------------


class TestROIInput:
    def test_basic_construction(self):
        inp = ROIInput(monthly_gpu_cost_usd=500_000)
        assert inp.monthly_gpu_cost_usd == 500_000
        assert inp.workload_mix is None
        assert inp.contract_months == 12

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="monthly_gpu_cost_usd"):
            ROIInput(monthly_gpu_cost_usd=-1)

    def test_zero_cost_raises(self):
        with pytest.raises(ValueError, match="monthly_gpu_cost_usd"):
            ROIInput(monthly_gpu_cost_usd=0)

    def test_zero_contract_months_raises(self):
        with pytest.raises(ValueError, match="contract_months"):
            ROIInput(monthly_gpu_cost_usd=100_000, contract_months=0)

    def test_workload_mix_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ROIInput(
                monthly_gpu_cost_usd=100_000,
                workload_mix={"training": 0.5, "realtime_inference": 0.4},
            )

    def test_workload_mix_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown workload"):
            ROIInput(
                monthly_gpu_cost_usd=100_000,
                workload_mix={"training": 0.5, "unknown_type": 0.5},
            )

    def test_valid_custom_workload_mix(self):
        inp = ROIInput(
            monthly_gpu_cost_usd=100_000,
            workload_mix={"training": 0.6, "realtime_inference": 0.4},
        )
        assert abs(sum(inp.workload_mix.values()) - 1.0) < 1e-9

    def test_all_fields(self):
        inp = ROIInput(
            monthly_gpu_cost_usd=1_000_000,
            workload_mix={"training": 0.7, "llm_batch_inference": 0.3},
            contract_months=24,
            num_gpus=1024,
            gpu_type="H100",
            primary_region="us-west",
            note="Test pilot customer",
        )
        assert inp.num_gpus == 1024
        assert inp.gpu_type == "H100"
        assert inp.note == "Test pilot customer"


# ---------------------------------------------------------------------------
# TestROICalculatorBasic
# ---------------------------------------------------------------------------


class TestROICalculatorBasic:
    def setup_method(self):
        self.calc = ROICalculator()

    def test_default_mix_returns_result(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        assert isinstance(result, ROIResult)
        assert result.monthly_savings_p50_usd > 0

    def test_p10_less_than_p50_less_than_p90(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        assert result.monthly_savings_p10_usd < result.monthly_savings_p50_usd
        assert result.monthly_savings_p50_usd < result.monthly_savings_p90_usd

    def test_savings_not_exceeding_100pct(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        assert result.effective_savings_rate_p90 < 1.0

    def test_60pct_not_claimed(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=1_000_000))
        assert result.effective_savings_rate_p50 < 0.60, (
            "ROI calculator must not claim 60% savings as the expected outcome"
        )
        assert result.effective_savings_rate_p90 < 0.60, (
            "ROI calculator p90 must not reach 60% on default workload mix"
        )

    def test_monthly_times_months_equals_total(self):
        result = self.calc.calculate(
            ROIInput(monthly_gpu_cost_usd=100_000, contract_months=24)
        )
        assert math.isclose(
            result.total_savings_p50_usd,
            result.monthly_savings_p50_usd * 24,
            rel_tol=1e-6,
        )

    def test_annual_savings_equals_12_months(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=200_000))
        assert math.isclose(
            result.annual_savings_p50_usd,
            result.monthly_savings_p50_usd * 12,
            rel_tol=1e-6,
        )

    def test_workload_breakdown_covers_all_types(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        wb_types = {wb.workload_type for wb in result.workload_breakdown}
        assert wb_types == set(DEFAULT_WORKLOAD_MIX.keys())

    def test_workload_breakdown_sums_to_total(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        total_from_breakdown = sum(
            wb.monthly_savings_p50_usd for wb in result.workload_breakdown
        )
        assert math.isclose(
            total_from_breakdown, result.monthly_savings_p50_usd, rel_tol=1e-6
        )

    def test_effective_rate_consistent_with_savings_and_cost(self):
        cost = 500_000
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=cost))
        computed_rate = result.monthly_savings_p50_usd / cost
        assert math.isclose(computed_rate, result.effective_savings_rate_p50, rel_tol=1e-6)

    def test_scales_linearly_with_cost(self):
        r1 = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        r2 = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=1_000_000))
        assert math.isclose(
            r2.monthly_savings_p50_usd / r1.monthly_savings_p50_usd,
            10.0,
            rel_tol=1e-6,
        )


# ---------------------------------------------------------------------------
# TestROICalculatorCustomMix
# ---------------------------------------------------------------------------


class TestROICalculatorCustomMix:
    def setup_method(self):
        self.calc = ROICalculator()

    def test_training_heavy_mix(self):
        result = self.calc.calculate(ROIInput(
            monthly_gpu_cost_usd=1_000_000,
            workload_mix={"training": 0.8, "realtime_inference": 0.2},
        ))
        # training p50=15%, realtime p50=7%: effective = 0.8*0.15 + 0.2*0.07 = 12% + 1.4% = 13.4%
        expected = 0.8 * BENCHMARK_SAVINGS_RATES["training"][1] + \
                   0.2 * BENCHMARK_SAVINGS_RATES["realtime_inference"][1]
        assert math.isclose(result.effective_savings_rate_p50, expected, rel_tol=1e-4)

    def test_realtime_only_mix(self):
        result = self.calc.calculate(ROIInput(
            monthly_gpu_cost_usd=100_000,
            workload_mix={"realtime_inference": 1.0},
        ))
        p10, p50, p90 = BENCHMARK_SAVINGS_RATES["realtime_inference"]
        assert math.isclose(result.effective_savings_rate_p50, p50, rel_tol=1e-4)
        assert math.isclose(result.effective_savings_rate_p10, p10, rel_tol=1e-4)
        assert math.isclose(result.effective_savings_rate_p90, p90, rel_tol=1e-4)

    def test_background_only_gives_highest_savings(self):
        result = self.calc.calculate(ROIInput(
            monthly_gpu_cost_usd=100_000,
            workload_mix={"background_maintenance": 1.0},
        ))
        p50 = BENCHMARK_SAVINGS_RATES["background_maintenance"][1]
        assert math.isclose(result.effective_savings_rate_p50, p50, rel_tol=1e-4)

    def test_flexible_fraction_all_flexible(self):
        result = self.calc.calculate(ROIInput(
            monthly_gpu_cost_usd=100_000,
            workload_mix={"training": 0.5, "llm_batch_inference": 0.5},
        ))
        assert result.flexible_fraction == 1.0

    def test_flexible_fraction_none_flexible(self):
        result = self.calc.calculate(ROIInput(
            monthly_gpu_cost_usd=100_000,
            workload_mix={"realtime_inference": 1.0},
        ))
        assert result.flexible_fraction == 0.0

    def test_low_flexible_fraction_warning_in_caveats(self):
        # Realtime-only mix: 0% flexible → should warn
        result = self.calc.calculate(ROIInput(
            monthly_gpu_cost_usd=100_000,
            workload_mix={"realtime_inference": 1.0},
        ))
        caveats_text = " ".join(result.caveats)
        assert "WARNING" in caveats_text or "flexible" in caveats_text.lower()

    def test_sufficient_flexible_fraction_no_warning(self):
        result = self.calc.calculate(ROIInput(
            monthly_gpu_cost_usd=100_000,
            workload_mix={"training": 0.5, "llm_batch_inference": 0.3, "realtime_inference": 0.2},
        ))
        # 80% flexible — should NOT have a warning at front of caveats
        first_caveat = result.caveats[0] if result.caveats else ""
        assert "WARNING" not in first_caveat


# ---------------------------------------------------------------------------
# TestROIResultSerialization
# ---------------------------------------------------------------------------


class TestROIResultSerialization:
    def setup_method(self):
        self.calc = ROICalculator()
        self.result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=500_000))

    def test_to_dict_has_required_keys(self):
        d = self.result.to_dict()
        assert "inputs" in d
        assert "projected_savings" in d
        assert "workload_breakdown" in d
        assert "caveats" in d
        assert "methodology_note" in d
        assert "benchmark_data_source" in d

    def test_to_dict_inputs_correct(self):
        d = self.result.to_dict()
        assert d["inputs"]["monthly_gpu_cost_usd"] == 500_000
        assert d["inputs"]["contract_months"] == 12

    def test_to_json_valid(self):
        j = self.result.to_json()
        parsed = json.loads(j)
        assert "projected_savings" in parsed

    def test_to_text_not_empty(self):
        text = self.result.to_text()
        assert len(text) > 100
        assert "AURELIUS ROI PROJECTION" in text

    def test_to_text_shows_savings(self):
        text = self.result.to_text()
        assert "PROJECTED" in text
        assert "$" in text

    def test_to_text_shows_caveats(self):
        text = self.result.to_text()
        assert "CAVEATS" in text

    def test_to_text_no_60pct_guarantee(self):
        text = self.result.to_text()
        # The effective savings rate lines (Conservative/Expected/Optimistic) must be < 60%.
        # We check lines that contain both "$" and a "% of spend" pattern.
        import re
        lines = text.split("\n")
        for line in lines:
            # Lines like "  Conservative (p10):  $   69,100  (13.8% of spend)"
            m = re.search(r"\((\d+\.?\d*)%\s+of spend\)", line)
            if m:
                pct = float(m.group(1))
                assert pct < 60.0, (
                    f"Effective savings rate ≥ 60% found: {pct}% in: {line.strip()}"
                )

    def test_workload_breakdown_to_dict(self):
        d = self.result.to_dict()
        wbd = d["workload_breakdown"]
        assert isinstance(wbd, list)
        assert len(wbd) > 0
        first = wbd[0]
        assert "workload_type" in first
        assert "monthly_savings_p50_usd" in first
        assert "savings_rate_p50" in first


# ---------------------------------------------------------------------------
# TestROICalculatorHonestyConstraints
# ---------------------------------------------------------------------------


class TestROICalculatorHonestyConstraints:
    def setup_method(self):
        self.calc = ROICalculator()

    def test_p50_savings_rate_within_benchmark_range(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        # p50 effective rate should be in the range of benchmark rates
        for wb in result.workload_breakdown:
            p10, p50, p90 = BENCHMARK_SAVINGS_RATES[wb.workload_type]
            assert math.isclose(wb.savings_rate_p50, p50, rel_tol=1e-6), (
                f"{wb.workload_type}: savings_rate_p50={wb.savings_rate_p50} != benchmark p50={p50}"
            )

    def test_caveats_mention_real_data(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        caveats_text = " ".join(result.caveats)
        assert "real" in caveats_text.lower() or "CAISO" in caveats_text

    def test_caveats_mention_aspirational_60pct(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        caveats_text = " ".join(result.caveats)
        assert "60%" in caveats_text
        assert "aspirational" in caveats_text.lower() or "NOT" in caveats_text

    def test_caveats_mention_25pct_proven(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        caveats_text = " ".join(result.caveats)
        assert "25" in caveats_text

    def test_methodology_note_present(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        assert len(result.methodology_note) > 20

    def test_data_source_mentions_caiso_pjm_ercot(self):
        result = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        src = result.benchmark_data_source
        assert "CAISO" in src
        assert "PJM" in src
        assert "ERCOT" in src


# ---------------------------------------------------------------------------
# TestROICalculatorContractMonths
# ---------------------------------------------------------------------------


class TestROICalculatorContractMonths:
    def setup_method(self):
        self.calc = ROICalculator()

    def test_1_month(self):
        r = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000, contract_months=1))
        assert math.isclose(r.total_savings_p50_usd, r.monthly_savings_p50_usd, rel_tol=1e-6)

    def test_24_months(self):
        r = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000, contract_months=24))
        assert math.isclose(
            r.total_savings_p50_usd, r.monthly_savings_p50_usd * 24, rel_tol=1e-6
        )

    def test_annual_always_12_months(self):
        for months in [1, 6, 12, 24, 36]:
            r = self.calc.calculate(ROIInput(monthly_gpu_cost_usd=100_000, contract_months=months))
            assert math.isclose(
                r.annual_savings_p50_usd,
                r.monthly_savings_p50_usd * 12,
                rel_tol=1e-6,
            ), f"annual != monthly*12 for contract_months={months}"


# ---------------------------------------------------------------------------
# TestROIPackageImports
# ---------------------------------------------------------------------------


class TestROIPackageImports:
    def test_calc_importable_from_package(self):
        c = ROICalculatorFromPkg()
        r = c.calculate(ROIInput(monthly_gpu_cost_usd=100_000))
        assert r.monthly_savings_p50_usd > 0

    def test_all_exports_available(self):
        from aurelius.roi import (
            BENCHMARK_SAVINGS_RATES,
            DEFAULT_WORKLOAD_MIX,
            MEAN_SAVINGS_P50,
            ROICalculator,
        )
        assert ROICalculator is not None
        assert isinstance(BENCHMARK_SAVINGS_RATES, dict)
        assert isinstance(DEFAULT_WORKLOAD_MIX, dict)
        assert isinstance(MEAN_SAVINGS_P50, float)


# ---------------------------------------------------------------------------
# TestROICLISubcommand
# ---------------------------------------------------------------------------


class TestROICLISubcommand:
    def test_cli_roi_basic(self, capsys):
        from aurelius.cli import main
        with patch("sys.argv", ["aurelius", "roi", "--monthly-cost", "500000"]):
            main()
        captured = capsys.readouterr()
        assert "AURELIUS ROI PROJECTION" in captured.out
        assert "$" in captured.out

    def test_cli_roi_custom_mix(self, capsys):
        from aurelius.cli import main
        mix = '{"training":0.6,"llm_batch_inference":0.4}'
        with patch("sys.argv", ["aurelius", "roi", "--monthly-cost", "100000",
                                "--workload-mix", mix]):
            main()
        captured = capsys.readouterr()
        assert "training" in captured.out.lower()

    def test_cli_roi_saves_json(self, tmp_path, capsys):
        from aurelius.cli import main
        output_file = tmp_path / "roi_output.json"
        with patch("sys.argv", ["aurelius", "roi", "--monthly-cost", "250000",
                                "--output", str(output_file)]):
            main()
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert "projected_savings" in data
        assert data["inputs"]["monthly_gpu_cost_usd"] == 250_000

    def test_cli_roi_24_months(self, capsys):
        from aurelius.cli import main
        with patch("sys.argv", [
            "aurelius", "roi",
            "--monthly-cost", "1000000",
            "--contract-months", "24",
        ]):
            main()
        captured = capsys.readouterr()
        assert "24" in captured.out

    def test_cli_roi_invalid_mix_sum_exits(self):
        from aurelius.cli import main
        mix = '{"training":0.5,"realtime_inference":0.4}'  # sums to 0.9
        with patch("sys.argv", ["aurelius", "roi", "--monthly-cost", "100000",
                                "--workload-mix", mix]):
            with pytest.raises(SystemExit):
                main()

    def test_cli_roi_unknown_workload_exits(self):
        from aurelius.cli import main
        mix = '{"training":0.5,"unknown_workload":0.5}'
        with patch("sys.argv", ["aurelius", "roi", "--monthly-cost", "100000",
                                "--workload-mix", mix]):
            with pytest.raises(SystemExit):
                main()
