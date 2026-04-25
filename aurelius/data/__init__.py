"""Data persistence utilities for Aurelius.

Provides minimal, dependency-free persistence for:
- Post-execution records (JSONL append-only)
- No external databases
- No network calls
- Safe for air-gapped environments
"""

from .persistence import JSONLWriter, RotatingJSONLWriter, get_default_post_execution_path

__all__ = [
    "JSONLWriter",
    "RotatingJSONLWriter",
    "get_default_post_execution_path",
]
