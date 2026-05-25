"""Frozen scenario hash checker for Phase 11.

Canonical scenarios under benchmarks/v1/ are immutable.
This module verifies that frozen scenarios have not been silently modified.

In CI, run:
    python -m aurelius.benchmarks.scenario_lock --check

To generate/update the lockfile after an intentional version bump:
    python -m aurelius.benchmarks.scenario_lock --generate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmarks"
_LOCKFILE = _BENCHMARKS_DIR / "v1" / ".scenario_hashes.json"


def _hash_file(path: Path) -> str:
    """Return SHA-256[:16] of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _collect_hashes(version: str = "v1") -> dict[str, str]:
    """Collect hashes for all YAML files in benchmarks/{version}/."""
    scenario_dir = _BENCHMARKS_DIR / version
    hashes: dict[str, str] = {}
    for yaml_path in sorted(scenario_dir.glob("*.yaml")):
        hashes[yaml_path.name] = _hash_file(yaml_path)
    return hashes


def generate_lockfile(version: str = "v1") -> None:
    """Write current hashes to .scenario_hashes.json."""
    hashes = _collect_hashes(version)
    lockfile = _BENCHMARKS_DIR / version / ".scenario_hashes.json"
    lockfile.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    print(f"Generated lockfile: {lockfile}")
    for name, h in hashes.items():
        print(f"  {name}: {h}")


def check_lockfile(version: str = "v1") -> tuple[bool, list[str]]:
    """Verify current hashes match the lockfile.

    Returns (ok, list_of_mismatches).
    If the lockfile doesn't exist, return (False, ["lockfile missing"]).
    """
    lockfile = _BENCHMARKS_DIR / version / ".scenario_hashes.json"
    if not lockfile.exists():
        return False, [f"Lockfile not found: {lockfile}. Run --generate to create it."]

    expected: dict[str, str] = json.loads(lockfile.read_text())
    current = _collect_hashes(version)

    mismatches: list[str] = []

    # Files that exist in lockfile but hash has changed
    for name, exp_hash in expected.items():
        cur_hash = current.get(name)
        if cur_hash is None:
            mismatches.append(f"DELETED: {name} (was {exp_hash})")
        elif cur_hash != exp_hash:
            mismatches.append(
                f"MODIFIED: {name} (expected {exp_hash}, got {cur_hash}). "
                "If this is intentional, bump the scenario version and run --generate."
            )

    # Files added without lockfile update
    for name in current:
        if name not in expected:
            mismatches.append(
                f"NEW_UNREGISTERED: {name} (hash {current[name]}). "
                "Run --generate to register it."
            )

    return (len(mismatches) == 0, mismatches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen scenario hash checker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Verify scenario hashes (CI)")
    group.add_argument("--generate", action="store_true", help="Generate/update lockfile")
    parser.add_argument("--version", default="v1", help="Scenario version directory")
    args = parser.parse_args()

    if args.generate:
        generate_lockfile(args.version)
        sys.exit(0)

    ok, mismatches = check_lockfile(args.version)
    if ok:
        print(f"Scenario hashes OK ({args.version})")
        sys.exit(0)
    else:
        print(f"Scenario hash check FAILED ({args.version}):")
        for m in mismatches:
            print(f"  ✗ {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
