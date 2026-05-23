"""Single-host advisory file lock to prevent overlapping learning-loop runs.

A cron/Railway schedule can fire a new run while the previous one is still
going. This lock makes the loop idempotent at the process level: the second
process fails to acquire the lock and exits cleanly instead of racing on the
CSV store / model artifacts.

Single-host only (fcntl.flock). Multi-host deployments should additionally use
a DB advisory lock (e.g. Postgres pg_advisory_lock) — documented in
docs/DATA_MOAT_ARCHITECTURE.md.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LockNotAcquiredError(RuntimeError):
    """Raised when the lock is already held by another process."""


class FileLock:
    """Non-blocking advisory lock backed by fcntl.flock on a lockfile."""

    def __init__(self, lock_path: str) -> None:
        self.lock_path = Path(lock_path)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        import fcntl

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._fd)
            self._fd = None
            raise LockNotAcquiredError(
                f"learning-loop lock held by another process: {self.lock_path}"
            ) from exc
        os.ftruncate(self._fd, 0)
        os.write(self._fd, str(os.getpid()).encode())

    def release(self) -> None:
        if self._fd is None:
            return
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
