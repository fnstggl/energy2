"""Package entry point for retrain_forecasters.

Allows invocation as:
    python -m aurelius.scripts.retrain_forecasters --start 2023-01-01 --end 2024-01-01

The actual implementation lives in scripts/retrain_forecasters.py at the
repository root. This module re-exports it for package-level invocation.
"""
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so the scripts/ directory is importable.
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.retrain_forecasters import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
