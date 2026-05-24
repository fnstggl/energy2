"""SLA ingestion layer.

Loads SLA policy configs from JSON or YAML, validates them, applies priority
tier defaults, and exposes a clean registry that the optimizer queries to find
the policy governing a given workload/service.

Malformed configs are NEVER silently accepted — :class:`SLAValidationError`
(with the full list of problems) is raised instead.

Config format (JSON or YAML), top level may be either a single policy object
or ``{"policies": [ ... ]}``::

    policies:
      - name: inference-prod
        tier: critical
        applies_to_workloads: [inference-prod]
        applies_to_workload_types: [realtime_inference]
        hard:
          allowed_regions: [us-east, us-west]
          max_p99_latency_ms: 3000
          migration_allowed: false
        soft:
          preferred_regions: [us-east]
          max_acceptable_savings_tradeoff_pct: 5
          optimization_aggressiveness: conservative
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

from .schema import (
    HardSLA,
    OptimizationAggressiveness,
    PriorityTier,
    SLAPolicy,
    SLAValidationError,
    SoftSLA,
    apply_tier_defaults,
)

logger = logging.getLogger(__name__)

_HARD_FIELDS = {f for f in HardSLA().__dict__.keys()}
_SOFT_FIELDS = {f for f in SoftSLA().__dict__.keys()}


def _coerce_tier(value: Any, errs: list[str]) -> PriorityTier:
    if isinstance(value, PriorityTier):
        return value
    if value is None:
        return PriorityTier.STANDARD
    try:
        return PriorityTier(str(value).strip().lower())
    except ValueError:
        errs.append(
            f"unknown tier {value!r}; valid tiers: {[t.value for t in PriorityTier]}"
        )
        return PriorityTier.STANDARD


def _coerce_aggressiveness(value: Any, errs: list[str]) -> Optional[OptimizationAggressiveness]:
    if value is None or isinstance(value, OptimizationAggressiveness):
        return value
    try:
        return OptimizationAggressiveness(str(value).strip().lower())
    except ValueError:
        errs.append(
            f"unknown optimization_aggressiveness {value!r}; valid: "
            f"{[a.value for a in OptimizationAggressiveness]}"
        )
        return None


def _build_hard(raw: dict, errs: list[str]) -> HardSLA:
    if not isinstance(raw, dict):
        errs.append("'hard' section must be a mapping/object")
        return HardSLA()
    unknown = set(raw) - _HARD_FIELDS
    if unknown:
        errs.append(f"unknown hard SLA fields: {sorted(unknown)}")
    return HardSLA(**{k: v for k, v in raw.items() if k in _HARD_FIELDS})


def _build_soft(raw: dict, errs: list[str]) -> SoftSLA:
    if not isinstance(raw, dict):
        errs.append("'soft' section must be a mapping/object")
        return SoftSLA()
    unknown = set(raw) - _SOFT_FIELDS
    if unknown:
        errs.append(f"unknown soft SLA fields: {sorted(unknown)}")
    kwargs = {k: v for k, v in raw.items() if k in _SOFT_FIELDS}
    if "optimization_aggressiveness" in kwargs:
        kwargs["optimization_aggressiveness"] = _coerce_aggressiveness(
            kwargs["optimization_aggressiveness"], errs
        )
    return SoftSLA(**kwargs)


def policy_from_dict(raw: dict) -> SLAPolicy:
    """Build (and validate) a single :class:`SLAPolicy` from a raw mapping.

    Raises :class:`SLAValidationError` if the config is malformed.
    """
    errs: list[str] = []
    if not isinstance(raw, dict):
        raise SLAValidationError(["policy must be a mapping/object"])

    known_top = {
        "name", "tier", "hard", "soft", "applies_to_workloads",
        "applies_to_workload_types", "enabled", "description",
    }
    unknown_top = set(raw) - known_top
    if unknown_top:
        errs.append(f"unknown top-level fields: {sorted(unknown_top)}")

    name = raw.get("name")
    tier = _coerce_tier(raw.get("tier"), errs)
    hard = _build_hard(raw.get("hard", {}) or {}, errs)
    soft = _build_soft(raw.get("soft", {}) or {}, errs)

    def _as_str_list(key: str) -> list[str]:
        v = raw.get(key, []) or []
        if not isinstance(v, list):
            errs.append(f"{key} must be a list of strings")
            return []
        return [str(x) for x in v]

    policy = SLAPolicy(
        name=str(name) if name is not None else "",
        tier=tier,
        hard=hard,
        soft=soft,
        applies_to_workloads=_as_str_list("applies_to_workloads"),
        applies_to_workload_types=_as_str_list("applies_to_workload_types"),
        enabled=bool(raw.get("enabled", True)),
        description=str(raw.get("description", "")),
    )

    errs.extend(policy.validate())
    if errs:
        raise SLAValidationError(errs)

    # Fill unspecified fields from tier defaults (critical=safest, batch=cheapest).
    return apply_tier_defaults(policy)


def _parse_text(text: str, fmt: str) -> Any:
    fmt = fmt.lower()
    if fmt in ("yaml", "yml"):
        import yaml  # PyYAML is a declared dependency

        return yaml.safe_load(text)
    if fmt == "json":
        return json.loads(text)
    # Auto: try JSON first, then YAML.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml

        return yaml.safe_load(text)


class SLARegistry:
    """Holds loaded SLA policies and resolves the policy for a workload.

    Resolution order when looking up a workload:
      1. exact workload/service id match (``applies_to_workloads``)
      2. workload-type match (``applies_to_workload_types``)
      3. registry default policy (if one was set)
      4. ``None`` — no policy governs this workload (optimizer behaves as before)
    """

    def __init__(self, policies: Optional[list[SLAPolicy]] = None, enabled: bool = True):
        self.enabled = enabled
        self._policies: list[SLAPolicy] = []
        self._by_workload: dict[str, SLAPolicy] = {}
        self._by_type: dict[str, SLAPolicy] = {}
        self._default: Optional[SLAPolicy] = None
        for p in policies or []:
            self.add(p)

    def add(self, policy: SLAPolicy) -> None:
        self._policies.append(policy)
        for wl in policy.applies_to_workloads:
            if wl in self._by_workload:
                logger.warning(
                    "SLA workload assignment conflict for %r: %r overrides %r",
                    wl, policy.name, self._by_workload[wl].name,
                )
            self._by_workload[wl] = policy
        for wt in policy.applies_to_workload_types:
            self._by_type[wt] = policy

    def set_default(self, policy: SLAPolicy) -> None:
        self._default = policy

    @property
    def policies(self) -> list[SLAPolicy]:
        return list(self._policies)

    def get(self, name: str) -> Optional[SLAPolicy]:
        for p in self._policies:
            if p.name == name:
                return p
        return None

    def resolve(
        self,
        workload_id: Optional[str] = None,
        workload_type: Optional[str] = None,
    ) -> Optional[SLAPolicy]:
        """Return the governing policy for a workload, or None."""
        if workload_id and workload_id in self._by_workload:
            return self._by_workload[workload_id]
        if workload_type and workload_type in self._by_type:
            return self._by_type[workload_type]
        return self._default

    def __len__(self) -> int:
        return len(self._policies)


class SLALoader:
    """Loads and validates SLA configs into an :class:`SLARegistry`."""

    @staticmethod
    def load_text(text: str, fmt: str = "auto", enabled: bool = True) -> SLARegistry:
        data = _parse_text(text, fmt)
        return SLALoader._build_registry(data, enabled=enabled)

    @staticmethod
    def load_file(path: Union[str, Path], enabled: bool = True) -> SLARegistry:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SLA config not found: {p}")
        suffix = p.suffix.lower().lstrip(".")
        fmt = "yaml" if suffix in ("yaml", "yml") else "json" if suffix == "json" else "auto"
        return SLALoader.load_text(p.read_text(), fmt=fmt, enabled=enabled)

    @staticmethod
    def load_dir(path: Union[str, Path], enabled: bool = True) -> SLARegistry:
        """Load every *.json/*.yaml/*.yml file in a directory into one registry."""
        p = Path(path)
        if not p.is_dir():
            raise NotADirectoryError(f"Not a directory: {p}")
        registry = SLARegistry(enabled=enabled)
        all_errors: list[str] = []
        files = sorted(
            [f for f in p.iterdir() if f.suffix.lower() in (".json", ".yaml", ".yml")]
        )
        for f in files:
            try:
                sub = SLALoader.load_file(f, enabled=enabled)
                for pol in sub.policies:
                    registry.add(pol)
                if sub._default is not None and registry._default is None:
                    registry.set_default(sub._default)
            except SLAValidationError as e:
                all_errors.extend(f"{f.name}: {msg}" for msg in e.errors)
        if all_errors:
            raise SLAValidationError(all_errors)
        return registry

    @staticmethod
    def _build_registry(data: Any, enabled: bool) -> SLARegistry:
        if data is None:
            raise SLAValidationError(["empty SLA config"])

        # Accept either {"policies": [...]} (optionally with "default") or a
        # bare single-policy object, or a bare list of policies.
        default_name: Optional[str] = None
        if isinstance(data, dict) and "policies" in data:
            raw_policies = data.get("policies")
            default_name = data.get("default")
            registry_enabled = bool(data.get("enabled", enabled))
            if not isinstance(raw_policies, list):
                raise SLAValidationError(["'policies' must be a list"])
        elif isinstance(data, list):
            raw_policies = data
            registry_enabled = enabled
        elif isinstance(data, dict):
            raw_policies = [data]
            registry_enabled = enabled
        else:
            raise SLAValidationError(["SLA config must be an object or list"])

        all_errors: list[str] = []
        policies: list[SLAPolicy] = []
        seen_names: set[str] = set()
        for i, rp in enumerate(raw_policies):
            try:
                pol = policy_from_dict(rp)
                if pol.name in seen_names:
                    all_errors.append(f"duplicate policy name: {pol.name!r}")
                seen_names.add(pol.name)
                policies.append(pol)
            except SLAValidationError as e:
                all_errors.extend(f"policies[{i}]: {msg}" for msg in e.errors)

        if all_errors:
            raise SLAValidationError(all_errors)

        registry = SLARegistry(policies, enabled=registry_enabled)
        if default_name:
            dp = registry.get(default_name)
            if dp is None:
                raise SLAValidationError([f"default policy {default_name!r} not found"])
            registry.set_default(dp)
        return registry
