"""Validation utilities for Aurelius."""

from .leakage_audit import DataLeakageError, assert_no_leakage

__all__ = [
    "DataLeakageError",
    "assert_no_leakage",
]


def __getattr__(name):
    """Lazy-load robustness module to avoid circular import issues."""
    _robustness_names = {
        "RobustnessTestHarness",
        "RobustnessReport",
        "RunMetrics",
        "AggregateMetrics",
        "format_cli_report",
        "report_to_dict",
        "save_report_json",
    }
    if name in _robustness_names:
        from .robustness import (
            RobustnessTestHarness,
            RobustnessReport,
            RunMetrics,
            AggregateMetrics,
            format_cli_report,
            report_to_dict,
            save_report_json,
        )
        globals().update({
            "RobustnessTestHarness": RobustnessTestHarness,
            "RobustnessReport": RobustnessReport,
            "RunMetrics": RunMetrics,
            "AggregateMetrics": AggregateMetrics,
            "format_cli_report": format_cli_report,
            "report_to_dict": report_to_dict,
            "save_report_json": save_report_json,
        })
        return globals()[name]
    raise AttributeError(f"module 'aurelius.validation' has no attribute {name!r}")
