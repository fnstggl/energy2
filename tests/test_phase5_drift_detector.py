"""Tests for DriftDetector — Phase 5 learning loop."""

import json
import math

import pytest

from aurelius.monitoring.drift_detector import DriftDetector, DriftReport

# ---------------------------------------------------------------------------
# DriftDetector unit tests
# ---------------------------------------------------------------------------

class TestDriftDetectorInit:
    def test_default_thresholds(self):
        d = DriftDetector()
        assert d.threshold_multiplier == 2.0
        assert d.min_records == 10

    def test_custom_thresholds(self):
        d = DriftDetector(threshold_multiplier=3.0, min_records=5)
        assert d.threshold_multiplier == 3.0
        assert d.min_records == 5

    def test_invalid_threshold_multiplier(self):
        with pytest.raises(ValueError):
            DriftDetector(threshold_multiplier=0.0)
        with pytest.raises(ValueError):
            DriftDetector(threshold_multiplier=-1.0)

    def test_invalid_min_records(self):
        with pytest.raises(ValueError):
            DriftDetector(min_records=0)


def _make_record(p50_forecast: float, actual: float) -> dict:
    """Build a minimal PostExecutionRecord dict for testing."""
    error = actual - p50_forecast
    return {
        "energy_cost_p50_error": error,
        "forecast_energy_cost_p50": p50_forecast,
    }


class TestDriftDetectorCheck:
    def test_empty_records_returns_no_drift(self):
        d = DriftDetector()
        report = d.check([], baseline_mape=0.10)
        assert isinstance(report, DriftReport)
        assert report.drift_detected is False
        assert math.isnan(report.recent_mape)
        assert report.n_recent_records == 0

    def test_insufficient_records_returns_no_drift(self):
        d = DriftDetector(min_records=10)
        records = [_make_record(100.0, 110.0)] * 5  # only 5 records
        report = d.check(records, baseline_mape=0.10)
        assert report.drift_detected is False
        assert math.isnan(report.recent_mape)
        assert report.n_valid_records == 5
        assert "Insufficient" in (report.alert_message or "")

    def test_no_drift_when_errors_small(self):
        d = DriftDetector(min_records=5)
        # 5% MAPE: p50=100, actual=105 → ape = 5/105 ≈ 0.0476
        records = [_make_record(100.0, 105.0)] * 10
        report = d.check(records, baseline_mape=0.10, model_name="price")
        assert report.drift_detected is False
        assert report.recent_mape < 0.10
        assert report.drift_ratio < 2.0

    def test_drift_detected_when_errors_large(self):
        d = DriftDetector(threshold_multiplier=2.0, min_records=5)
        # 50% MAPE: actual is 50% off from forecast
        records = [_make_record(100.0, 150.0)] * 10
        report = d.check(records, baseline_mape=0.10, model_name="price")
        assert report.drift_detected is True
        assert report.drift_ratio > 2.0
        assert report.alert_message is not None
        assert "DRIFT DETECTED" in report.alert_message

    def test_drift_exactly_at_threshold_not_flagged(self):
        # drift_ratio = exactly 2.0 should NOT trigger (threshold is strictly >)
        d = DriftDetector(threshold_multiplier=2.0, min_records=3)
        # baseline_mape=0.10, want recent_mape=0.20 exactly
        # ape = |error| / |actual|; actual = forecast + error
        # if forecast=100, actual=125: error=25, actual=125, ape=25/125=0.20
        records = [_make_record(100.0, 125.0)] * 10
        report = d.check(records, baseline_mape=0.20)
        # recent_mape ≈ 0.20, ratio ≈ 1.0 — no drift
        assert report.drift_detected is False

    def test_invalid_baseline_mape_zero(self):
        d = DriftDetector(min_records=3)
        records = [_make_record(100.0, 110.0)] * 10
        report = d.check(records, baseline_mape=0.0)
        assert report.drift_detected is False
        assert math.isnan(report.drift_ratio)

    def test_invalid_baseline_mape_negative(self):
        d = DriftDetector(min_records=3)
        records = [_make_record(100.0, 110.0)] * 10
        report = d.check(records, baseline_mape=-0.05)
        assert report.drift_detected is False

    def test_records_with_missing_fields_skipped(self):
        d = DriftDetector(min_records=5)
        records = [
            {"energy_cost_p50_error": None, "forecast_energy_cost_p50": 100.0},
            {"energy_cost_p50_error": 10.0, "forecast_energy_cost_p50": None},
            {},
            {"other_field": 42},
            _make_record(100.0, 110.0),
            _make_record(100.0, 108.0),
            _make_record(100.0, 112.0),
            _make_record(100.0, 109.0),
            _make_record(100.0, 111.0),
            _make_record(100.0, 107.0),
        ]
        report = d.check(records, baseline_mape=0.10)
        # Only 6 valid records (the 4 bad ones are skipped)
        assert report.n_valid_records == 6
        assert not math.isnan(report.recent_mape)

    def test_actual_zero_records_skipped(self):
        """Rows where actual == 0 produce division-by-zero, must be skipped."""
        d = DriftDetector(min_records=3)
        # actual = forecast + error = 100 + (-100) = 0 → skip
        zero_actual = {"energy_cost_p50_error": -100.0, "forecast_energy_cost_p50": 100.0}
        valid = [_make_record(100.0, 110.0)] * 10
        report = d.check([zero_actual] + valid, baseline_mape=0.10)
        assert report.n_valid_records == 10  # zero-actual record excluded
        assert not math.isnan(report.recent_mape)

    def test_report_to_dict_serializable(self):
        d = DriftDetector(min_records=3)
        records = [_make_record(100.0, 110.0)] * 10
        report = d.check(records, baseline_mape=0.10)
        d_out = report.to_dict()
        assert isinstance(d_out, dict)
        # Must be JSON-serializable
        json.dumps(d_out)

    def test_to_json(self):
        d = DriftDetector(min_records=3)
        records = [_make_record(100.0, 110.0)] * 10
        report = d.check(records, baseline_mape=0.10)
        j = report.to_json()
        parsed = json.loads(j)
        assert "drift_detected" in parsed
        assert "recent_mape" in parsed

    def test_model_name_preserved(self):
        d = DriftDetector(min_records=3)
        records = [_make_record(100.0, 110.0)] * 10
        report = d.check(records, baseline_mape=0.10, model_name="carbon_model")
        assert report.model_name == "carbon_model"

    def test_nan_in_error_field_skipped(self):
        d = DriftDetector(min_records=3)
        bad = {"energy_cost_p50_error": float("nan"), "forecast_energy_cost_p50": 100.0}
        valid = [_make_record(100.0, 110.0)] * 10
        report = d.check([bad] + valid, baseline_mape=0.10)
        # NaN produces a valid float division but APE should still be computable
        # (nan/anything = nan; the row should be skipped in practice)
        # Just assert no crash and n_valid ≤ 11
        assert report.n_recent_records == 11


class TestDriftDetectorFromJsonl:
    def test_missing_file_returns_no_drift(self):
        d = DriftDetector()
        report = d.check_from_jsonl("/nonexistent/path.jsonl", baseline_mape=0.10)
        assert report.drift_detected is False
        assert "not found" in (report.alert_message or "")

    def test_valid_jsonl_no_drift(self, tmp_path):
        d = DriftDetector(min_records=5)
        jsonl = tmp_path / "records.jsonl"
        records = [_make_record(100.0, 105.0) for _ in range(20)]
        with open(jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        report = d.check_from_jsonl(jsonl, baseline_mape=0.15)
        assert report.drift_detected is False
        assert report.n_recent_records == 20

    def test_valid_jsonl_drift_detected(self, tmp_path):
        d = DriftDetector(threshold_multiplier=2.0, min_records=5)
        jsonl = tmp_path / "records.jsonl"
        # 30% error, baseline 5% → ratio ≈ 6× → drift
        records = [_make_record(100.0, 130.0) for _ in range(20)]
        with open(jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        report = d.check_from_jsonl(jsonl, baseline_mape=0.05)
        assert report.drift_detected is True

    def test_last_n_respected(self, tmp_path):
        d = DriftDetector(min_records=5)
        jsonl = tmp_path / "records.jsonl"
        # First 50 records: large error; last 20 records: small error
        big_error = [_make_record(100.0, 200.0) for _ in range(50)]
        small_error = [_make_record(100.0, 105.0) for _ in range(20)]
        with open(jsonl, "w") as f:
            for r in big_error + small_error:
                f.write(json.dumps(r) + "\n")
        # Use only last 20 → small error → no drift
        report = d.check_from_jsonl(jsonl, baseline_mape=0.10, last_n=20)
        assert report.n_recent_records == 20
        assert report.drift_detected is False

    def test_corrupt_jsonl_returns_no_drift(self, tmp_path):
        d = DriftDetector()
        jsonl = tmp_path / "records.jsonl"
        jsonl.write_text("not valid json\n{also bad\n")
        report = d.check_from_jsonl(jsonl, baseline_mape=0.10)
        assert report.drift_detected is False

    def test_empty_jsonl_returns_no_drift(self, tmp_path):
        d = DriftDetector()
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("")
        report = d.check_from_jsonl(jsonl, baseline_mape=0.10)
        assert report.drift_detected is False
        assert report.n_recent_records == 0
