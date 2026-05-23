"""DecisionRecorder — JSONL persistence for shadow-mode decisions.

Each line in the output file is one JSON-serialized DecisionRecord.
Files are append-safe: multiple shadow runs can write to the same archive.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .models import DecisionRecord

logger = logging.getLogger(__name__)


class DecisionRecorder:
    """Save, load, and update shadow-mode DecisionRecords.

    The JSONL format is chosen for:
    - Streaming reads (no need to load entire file into memory)
    - Append safety (each line is independent)
    - Human readability (grep-friendly)
    - Easy export to CSV/DataFrame for analysis
    """

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self.output_path = output_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        records: list[DecisionRecord],
        path: Optional[Path] = None,
        mode: str = "a",
    ) -> Path:
        """Append DecisionRecords to a JSONL file.

        Args:
            records: Records to save.
            path:    Override self.output_path. Directories are created.
            mode:    "a" (append, default) or "w" (overwrite).

        Returns:
            Path to the file that was written.
        """
        dest = path or self.output_path
        if dest is None:
            raise ValueError("output_path must be set or passed as 'path'")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, mode, encoding="utf-8") as fh:
            for record in records:
                fh.write(record.to_json() + "\n")

        logger.info(f"DecisionRecorder: saved {len(records)} records to {dest}")
        return dest

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, path: Optional[Path] = None) -> list[DecisionRecord]:
        """Load all DecisionRecords from a JSONL file.

        Args:
            path: Override self.output_path.

        Returns:
            List of DecisionRecord (in file order). Empty list if file missing.
        """
        src = path or self.output_path
        if src is None:
            raise ValueError("output_path must be set or passed as 'path'")
        src = Path(src)

        if not src.exists():
            logger.warning(f"DecisionRecorder.load: {src} not found, returning empty")
            return []

        records: list[DecisionRecord] = []
        errors = 0
        with open(src, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(DecisionRecord.from_json(line))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    logger.warning(f"DecisionRecorder: skipping line {lineno} ({exc})")
                    errors += 1

        if errors:
            logger.warning(f"DecisionRecorder.load: {errors} lines could not be parsed")
        logger.info(f"DecisionRecorder: loaded {len(records)} records from {src}")
        return records

    # ------------------------------------------------------------------
    # Update with realized data
    # ------------------------------------------------------------------

    def mark_realized(
        self,
        records: list[DecisionRecord],
        realized_updates: dict[str, dict],
    ) -> list[DecisionRecord]:
        """Apply realized data to matching records.

        Args:
            records:         List of DecisionRecord (may have realized_* = None).
            realized_updates:
                {job_id: {
                    "realized_rt_price": float,
                    "realized_energy_cost": float,
                    "realized_baseline_rt_price": float,
                    "realized_baseline_cost": float,
                    "realized_savings_pct": float,
                    "sla_met": bool,            (optional)
                    "realization_note": str,    (optional)
                }}

        Returns:
            Updated records (same list, in-place updates, returns for chaining).
        """
        for record in records:
            update = realized_updates.get(record.job_id)
            if update is None:
                continue
            for field_name, value in update.items():
                if hasattr(record, field_name):
                    setattr(record, field_name, value)
        return records

    # ------------------------------------------------------------------
    # Convenience: save_and_reload round-trip
    # ------------------------------------------------------------------

    def save_updated(
        self,
        records: list[DecisionRecord],
        path: Optional[Path] = None,
    ) -> Path:
        """Overwrite the file with current record state (write, not append)."""
        return self.save(records, path=path, mode="w")
