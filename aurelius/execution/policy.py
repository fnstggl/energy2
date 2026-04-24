"""Policy-controlled execution for Aurelius.

This module provides a clean, deterministic policy enforcement mechanism that:
- Allows free dry_run usage
- Controls live execution under a valid signed policy
- Produces auditable authorization decisions

The engine can:
- Read a policy bundle from disk
- Verify its signature
- Enforce policy constraints at execution time

The engine CANNOT mint valid policies.

This is NOT DRM. This is NOT anti-tamper.
This is authorization gating for production deployments.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional
import base64
import hashlib
import hmac
import json
import logging
import os

logger = logging.getLogger(__name__)

# ============================================================================
# EMBEDDED CRYPTOGRAPHIC KEYS (MANDATORY)
# ============================================================================
# These keys MUST be module-level constants.
# They MUST ship with the engine code.
# They MUST NOT be loaded from disk, environment variables, or network.
# The engine MUST NOT support key rotation without a code update.

# Ed25519 public key for policy verification (base64-encoded)
# This is the verification key only - the engine cannot sign.
# Replace with your actual public key when deploying.
_ED25519_PUBLIC_KEY_B64 = "MCowBQYDK2VwAyEAVGVzdFB1YmxpY0tleUZvckF1cmVsaXVzUG9saWN5"

# HMAC-SHA256 secret for v0 verification (NOT SECURE AGAINST KEY EXPOSURE)
# This is a temporary fallback until Ed25519 is fully deployed.
# Clearly labeled as insecure if key is exposed.
_HMAC_V0_SECRET = b"v0_hmac_not_secure_against_key_exposure_aurelius_policy_secret"


# ============================================================================
# POLICY BUNDLE DATA STRUCTURES
# ============================================================================

@dataclass
class PolicyConfig:
    """Policy configuration from the policy bundle.

    All fields correspond to the policy JSON specification.
    """
    policy_id: str
    issued_at: date  # Informational only, not used for validation
    valid_until: date
    allowed_modes: list[str]
    execution_enabled: bool
    constraint_profiles: list[str]
    max_latency_slack_hours: float
    quantile: float
    max_downside_risk_pct: float
    min_expected_savings_pct: float
    metric: Literal["energy_cost", "carbon", "both"]
    notes: Optional[str] = None


@dataclass
class SignatureInfo:
    """Signature information from the policy bundle."""
    alg: Literal["ed25519", "hmac_sha256_v0"]
    key_id: str
    sig_b64: str


@dataclass
class PolicyBundle:
    """Complete policy bundle with policy and signature."""
    policy: PolicyConfig
    signature: SignatureInfo
    raw_policy_json: dict  # Original policy dict for signature verification


@dataclass
class AuthorizationResult:
    """Result of policy authorization check.

    Attributes:
        allowed: Whether execution is allowed
        reason: Explanation of the decision
        policy_id: ID of the policy used (if any)
        key_id: Key ID used for verification (if any)
    """
    allowed: bool
    reason: str
    policy_id: Optional[str] = None
    key_id: Optional[str] = None


# ============================================================================
# CANONICAL JSON SERIALIZATION
# ============================================================================

def canonical_json_bytes(policy_dict: dict) -> bytes:
    """Produce canonical JSON bytes for signature verification.

    MANDATORY format:
    - sort_keys=True
    - separators=(",", ":")
    - ensure_ascii=False
    - UTF-8 encoded

    Args:
        policy_dict: The policy dictionary to serialize

    Returns:
        Canonical UTF-8 bytes
    """
    return json.dumps(
        policy_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


# ============================================================================
# SIGNATURE VERIFICATION
# ============================================================================

def _verify_ed25519(policy_bytes: bytes, signature_b64: str) -> bool:
    """Verify Ed25519 signature using the cryptography library.

    Args:
        policy_bytes: Canonical JSON bytes of the policy
        signature_b64: Base64-encoded signature

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_der_public_key
        from cryptography.exceptions import InvalidSignature

        # Decode the public key
        public_key_bytes = base64.b64decode(_ED25519_PUBLIC_KEY_B64)
        public_key = load_der_public_key(public_key_bytes)

        if not isinstance(public_key, Ed25519PublicKey):
            logger.debug("Loaded key is not an Ed25519 public key")
            return False

        # Decode the signature
        signature = base64.b64decode(signature_b64)

        # Verify
        try:
            public_key.verify(signature, policy_bytes)
            return True
        except InvalidSignature:
            return False

    except ImportError:
        logger.debug("cryptography library not available for Ed25519")
        return False
    except Exception as e:
        logger.debug(f"Ed25519 verification failed: {e}")
        return False


def _verify_hmac_v0(policy_bytes: bytes, signature_b64: str) -> bool:
    """Verify HMAC-SHA256 signature (v0 fallback).

    WARNING: v0_hmac_not_secure_against_key_exposure
    This is a temporary fallback. The secret is embedded in the code.
    If the code is exposed, the secret is exposed.

    Args:
        policy_bytes: Canonical JSON bytes of the policy
        signature_b64: Base64-encoded HMAC

    Returns:
        True if HMAC is valid, False otherwise
    """
    try:
        expected_hmac = hmac.new(
            _HMAC_V0_SECRET,
            policy_bytes,
            hashlib.sha256,
        ).digest()

        provided_hmac = base64.b64decode(signature_b64)

        return hmac.compare_digest(expected_hmac, provided_hmac)

    except Exception as e:
        logger.debug(f"HMAC-SHA256 v0 verification failed: {e}")
        return False


def verify_signature(
    policy_dict: dict,
    signature: SignatureInfo,
) -> bool:
    """Verify the signature of a policy bundle.

    Args:
        policy_dict: The raw policy dictionary
        signature: The signature information

    Returns:
        True if signature is valid, False otherwise
    """
    policy_bytes = canonical_json_bytes(policy_dict)

    if signature.alg == "ed25519":
        return _verify_ed25519(policy_bytes, signature.sig_b64)
    elif signature.alg == "hmac_sha256_v0":
        return _verify_hmac_v0(policy_bytes, signature.sig_b64)
    else:
        logger.warning(f"Unknown signature algorithm: {signature.alg}")
        return False


# ============================================================================
# POLICY LOADING
# ============================================================================

def get_default_policy_path() -> Path:
    """Get the default path for the policy bundle.

    Returns:
        Path to aurelius/data/policy/policy_bundle.json
    """
    package_dir = Path(__file__).parent.parent
    return package_dir / "data" / "policy" / "policy_bundle.json"


def get_policy_path() -> Path:
    """Get the policy bundle path, respecting environment override.

    Environment variable: AURELIUS_POLICY_BUNDLE_PATH

    Returns:
        Path to the policy bundle file
    """
    env_path = os.environ.get("AURELIUS_POLICY_BUNDLE_PATH")
    if env_path:
        return Path(env_path)
    return get_default_policy_path()


def _parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string

    Returns:
        date object
    """
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def load_policy_bundle(path: Optional[Path] = None) -> Optional[PolicyBundle]:
    """Load and parse a policy bundle from disk.

    Does NOT verify the signature - call verify_signature separately.

    Args:
        path: Path to policy bundle (uses default if None)

    Returns:
        PolicyBundle if loaded successfully, None if file missing or invalid
    """
    policy_path = path or get_policy_path()

    if not policy_path.exists():
        return None

    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse policy section
        policy_dict = data.get("policy", {})
        policy = PolicyConfig(
            policy_id=policy_dict["policy_id"],
            issued_at=_parse_date(policy_dict["issued_at"]),
            valid_until=_parse_date(policy_dict["valid_until"]),
            allowed_modes=policy_dict["allowed_modes"],
            execution_enabled=policy_dict["execution_enabled"],
            constraint_profiles=policy_dict["constraint_profiles"],
            max_latency_slack_hours=policy_dict["max_latency_slack_hours"],
            quantile=policy_dict["quantile"],
            max_downside_risk_pct=policy_dict["max_downside_risk_pct"],
            min_expected_savings_pct=policy_dict["min_expected_savings_pct"],
            metric=policy_dict["metric"],
            notes=policy_dict.get("notes"),
        )

        # Parse signature section
        sig_dict = data.get("signature", {})
        signature = SignatureInfo(
            alg=sig_dict["alg"],
            key_id=sig_dict["key_id"],
            sig_b64=sig_dict["sig_b64"],
        )

        return PolicyBundle(
            policy=policy,
            signature=signature,
            raw_policy_json=policy_dict,
        )

    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to parse policy bundle at {policy_path}: {e}")
        return None


# ============================================================================
# POLICY VALIDATION
# ============================================================================

def is_policy_expired(policy: PolicyConfig) -> bool:
    """Check if a policy has expired.

    Rules:
    - Compare using date objects (not strings)
    - Policy is valid if current_utc_date <= valid_until (inclusive)

    Args:
        policy: The policy configuration

    Returns:
        True if expired, False if still valid
    """
    current_date = datetime.utcnow().date()
    return current_date > policy.valid_until


def validate_constraint_profile(
    config_profile: str,
    policy: PolicyConfig,
) -> tuple[bool, str]:
    """Validate that the constraint profile is allowed by policy.

    Args:
        config_profile: The constraint profile from ExecutionConfig
        policy: The policy configuration

    Returns:
        (valid, reason) tuple
    """
    if config_profile in policy.constraint_profiles:
        return True, "constraint_profile_allowed"
    return False, f"constraint_profile_denied: {config_profile} not in {policy.constraint_profiles}"


def validate_latency_slack(
    config_slack: float,
    config_profile: str,
    policy: PolicyConfig,
) -> tuple[bool, str]:
    """Validate latency slack against policy ceiling.

    Only applies when constraint_profile == "latency_safe".

    Args:
        config_slack: latency_slack_threshold_hours from ExecutionConfig
        config_profile: constraint_profile from ExecutionConfig
        policy: The policy configuration

    Returns:
        (valid, reason) tuple
    """
    if config_profile != "latency_safe":
        return True, "latency_slack_not_applicable"

    if config_slack <= policy.max_latency_slack_hours:
        return True, "latency_slack_within_ceiling"

    return False, (
        f"latency_slack_violation: {config_slack}h > "
        f"policy.max_latency_slack_hours={policy.max_latency_slack_hours}h"
    )


def validate_quantile_config(
    quantile_config: Any,  # QuantileGateConfig
    policy: PolicyConfig,
) -> tuple[bool, str]:
    """Validate QuantileGateConfig against policy ceilings.

    Rules:
    - quantile must equal policy.quantile OR be more conservative (higher)
    - max_downside_risk_pct <= policy.max_downside_risk_pct
    - min_expected_savings_pct >= policy.min_expected_savings_pct
    - metric conservativeness: "both" is most conservative

    Args:
        quantile_config: QuantileGateConfig from safety gate
        policy: The policy configuration

    Returns:
        (valid, reason) tuple
    """
    if quantile_config is None:
        # No quantile config provided - skip validation
        return True, "quantile_config_not_provided"

    violations = []

    # Quantile validation (higher is more conservative)
    if quantile_config.quantile < policy.quantile:
        violations.append(
            f"quantile={quantile_config.quantile} < policy.quantile={policy.quantile}"
        )

    # max_downside_risk_pct validation (lower is more conservative)
    if quantile_config.max_downside_risk_pct > policy.max_downside_risk_pct:
        violations.append(
            f"max_downside_risk_pct={quantile_config.max_downside_risk_pct} > "
            f"policy.max_downside_risk_pct={policy.max_downside_risk_pct}"
        )

    # min_expected_savings_pct validation (higher is more conservative)
    if quantile_config.min_expected_savings_pct < policy.min_expected_savings_pct:
        violations.append(
            f"min_expected_savings_pct={quantile_config.min_expected_savings_pct} < "
            f"policy.min_expected_savings_pct={policy.min_expected_savings_pct}"
        )

    # Metric conservativeness validation
    # "both" is strictly more conservative than either single metric
    config_metric = quantile_config.metric
    policy_metric = policy.metric

    metric_valid = True
    if policy_metric == "both":
        # Policy requires both - config must be both
        if config_metric != "both":
            metric_valid = False
    elif policy_metric == "energy_cost":
        # Policy allows energy_cost - config may be energy_cost or both
        if config_metric not in ["energy_cost", "both"]:
            metric_valid = False
    elif policy_metric == "carbon":
        # Policy allows carbon - config may be carbon or both
        if config_metric not in ["carbon", "both"]:
            metric_valid = False

    if not metric_valid:
        violations.append(
            f"metric={config_metric} not allowed under policy.metric={policy_metric}"
        )

    if violations:
        return False, f"quantile_config_violation: {'; '.join(violations)}"

    return True, "quantile_config_valid"


# ============================================================================
# AUTHORIZATION LOGIC
# ============================================================================

# Track if we've already warned about missing policy (avoid spam)
_missing_policy_warned = False


def authorize_execution(
    execution_config: Any,  # ExecutionConfig
    quantile_config: Any = None,  # QuantileGateConfig | None
) -> AuthorizationResult:
    """Authorize execution based on policy bundle.

    This is the main integration point for policy enforcement.
    Call this before submitting jobs in live mode.

    Rules:
    - dry_run: Always allowed (policy violations logged as warnings)
    - live: Requires valid signed policy satisfying all constraints

    Args:
        execution_config: ExecutionConfig with mode, constraint_profile, etc.
        quantile_config: Optional QuantileGateConfig for quantile validation

    Returns:
        AuthorizationResult with allowed/denied and reason
    """
    global _missing_policy_warned

    mode = execution_config.mode
    is_dry_run = mode == "dry_run"

    # Load policy bundle
    bundle = load_policy_bundle()

    # Handle missing policy
    if bundle is None:
        if not _missing_policy_warned:
            logger.warning(
                "Policy bundle not found. dry_run allowed, live blocked. "
                f"Expected at: {get_policy_path()}"
            )
            _missing_policy_warned = True

        if is_dry_run:
            result = AuthorizationResult(
                allowed=True,
                reason="dry_run_allowed_no_policy",
            )
            _emit_authorization_audit(result, mode)
            return result
        else:
            result = AuthorizationResult(
                allowed=False,
                reason="missing_policy",
            )
            _emit_authorization_audit(result, mode)
            return result

    # Verify signature
    if not verify_signature(bundle.raw_policy_json, bundle.signature):
        if is_dry_run:
            _emit_policy_violation_dry_run("invalid_signature")
            result = AuthorizationResult(
                allowed=True,
                reason="dry_run_allowed_invalid_signature",
                policy_id=bundle.policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result
        else:
            result = AuthorizationResult(
                allowed=False,
                reason="invalid_signature",
                policy_id=bundle.policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result

    policy = bundle.policy

    # Check policy expiration
    if is_policy_expired(policy):
        if is_dry_run:
            _emit_policy_violation_dry_run(
                f"expired_policy: valid_until={policy.valid_until}"
            )
            result = AuthorizationResult(
                allowed=True,
                reason="dry_run_allowed_expired_policy",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result
        else:
            result = AuthorizationResult(
                allowed=False,
                reason="expired_policy",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result

    # Check execution_enabled
    if not policy.execution_enabled:
        if is_dry_run:
            _emit_policy_violation_dry_run("execution_disabled")
            result = AuthorizationResult(
                allowed=True,
                reason="dry_run_allowed_execution_disabled",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result
        else:
            result = AuthorizationResult(
                allowed=False,
                reason="execution_disabled",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result

    # Check allowed_modes (for live only)
    if not is_dry_run and mode not in policy.allowed_modes:
        result = AuthorizationResult(
            allowed=False,
            reason="mode_not_allowed",
            policy_id=policy.policy_id,
            key_id=bundle.signature.key_id,
        )
        _emit_authorization_audit(result, mode)
        return result

    # Validate constraint_profile
    profile_valid, profile_reason = validate_constraint_profile(
        execution_config.constraint_profile,
        policy,
    )
    if not profile_valid:
        if is_dry_run:
            _emit_policy_violation_dry_run(profile_reason)
            result = AuthorizationResult(
                allowed=True,
                reason=f"dry_run_allowed_{profile_reason}",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result
        else:
            result = AuthorizationResult(
                allowed=False,
                reason="constraint_profile_denied",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result

    # Validate latency slack
    slack_valid, slack_reason = validate_latency_slack(
        execution_config.latency_slack_threshold_hours,
        execution_config.constraint_profile,
        policy,
    )
    if not slack_valid:
        if is_dry_run:
            _emit_policy_violation_dry_run(slack_reason)
            result = AuthorizationResult(
                allowed=True,
                reason=f"dry_run_allowed_{slack_reason}",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result
        else:
            result = AuthorizationResult(
                allowed=False,
                reason="latency_slack_violation",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result

    # Validate quantile config
    quantile_valid, quantile_reason = validate_quantile_config(
        quantile_config,
        policy,
    )
    if not quantile_valid:
        if is_dry_run:
            _emit_policy_violation_dry_run(quantile_reason)
            result = AuthorizationResult(
                allowed=True,
                reason=f"dry_run_allowed_{quantile_reason}",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result
        else:
            result = AuthorizationResult(
                allowed=False,
                reason="quantile_config_violation",
                policy_id=policy.policy_id,
                key_id=bundle.signature.key_id,
            )
            _emit_authorization_audit(result, mode)
            return result

    # All checks passed
    result = AuthorizationResult(
        allowed=True,
        reason="policy_valid",
        policy_id=policy.policy_id,
        key_id=bundle.signature.key_id,
    )
    _emit_authorization_audit(result, mode)
    return result


# ============================================================================
# AUDIT LOGGING
# ============================================================================

def _emit_authorization_audit(result: AuthorizationResult, mode: str) -> None:
    """Emit structured audit log for authorization decision.

    One JSON line per run (not per job).
    """
    audit_record = {
        "event": "policy_authorization",
        "mode": mode,
        "allowed": result.allowed,
        "reason": result.reason,
        "policy_id": result.policy_id,
        "key_id": result.key_id,
    }
    logger.info(f"AUDIT: {json.dumps(audit_record)}")


def _emit_policy_violation_dry_run(violation: str) -> None:
    """Emit warning for policy violation in dry_run mode."""
    audit_record = {
        "event": "policy_violation_dry_run",
        "violation": violation,
    }
    logger.warning(f"AUDIT: {json.dumps(audit_record)}")


# ============================================================================
# UTILITY: GENERATE HMAC SIGNATURE (FOR TESTING ONLY)
# ============================================================================

def _generate_hmac_v0_signature(policy_dict: dict) -> str:
    """Generate HMAC-SHA256 signature for testing purposes.

    WARNING: This is for testing only. The engine should NOT generate
    signatures in production. This function exists only for inline tests.

    Args:
        policy_dict: The policy dictionary to sign

    Returns:
        Base64-encoded HMAC signature
    """
    policy_bytes = canonical_json_bytes(policy_dict)
    sig = hmac.new(_HMAC_V0_SECRET, policy_bytes, hashlib.sha256).digest()
    return base64.b64encode(sig).decode("ascii")


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    import tempfile
    from datetime import timedelta

    print("=" * 60)
    print("Policy-Controlled Execution Inline Tests")
    print("=" * 60)

    # Reset warning flag for tests
    _missing_policy_warned = False

    # Mock ExecutionConfig for testing
    @dataclass
    class MockExecutionConfig:
        mode: str = "dry_run"
        constraint_profile: str = "batch_optimized"
        latency_slack_threshold_hours: float = 0.05

    # Mock QuantileGateConfig for testing
    @dataclass
    class MockQuantileGateConfig:
        quantile: float = 0.9
        max_downside_risk_pct: float = 10.0
        min_expected_savings_pct: float = 5.0
        metric: str = "both"

    # Helper to create valid policy dict
    def create_valid_policy_dict(
        valid_days: int = 30,
        execution_enabled: bool = True,
        allowed_modes: list = None,
    ) -> dict:
        today = datetime.utcnow().date()
        return {
            "policy_id": "test-policy-001",
            "issued_at": today.isoformat(),
            "valid_until": (today + timedelta(days=valid_days)).isoformat(),
            "allowed_modes": allowed_modes or ["dry_run", "live"],
            "execution_enabled": execution_enabled,
            "constraint_profiles": ["batch_optimized", "latency_safe"],
            "max_latency_slack_hours": 0.05,
            "quantile": 0.9,
            "max_downside_risk_pct": 10.0,
            "min_expected_savings_pct": 5.0,
            "metric": "both",
            "notes": "Test policy",
        }

    # Helper to create policy bundle file
    def write_policy_bundle(tmpdir: str, policy_dict: dict, alg: str = "hmac_sha256_v0") -> str:
        path = Path(tmpdir) / "policy_bundle.json"

        if alg == "hmac_sha256_v0":
            sig_b64 = _generate_hmac_v0_signature(policy_dict)
        else:
            sig_b64 = "invalid_signature_for_testing"

        bundle = {
            "policy": policy_dict,
            "signature": {
                "alg": alg,
                "key_id": "test-key-001",
                "sig_b64": sig_b64,
            },
        }

        with open(path, "w") as f:
            json.dump(bundle, f)

        return str(path)

    # Test 1: Canonical JSON is deterministic
    print("\n[Test 1] Canonical JSON determinism")
    policy1 = {"b": 2, "a": 1, "c": {"z": 26, "y": 25}}
    policy2 = {"a": 1, "c": {"y": 25, "z": 26}, "b": 2}
    bytes1 = canonical_json_bytes(policy1)
    bytes2 = canonical_json_bytes(policy2)
    assert bytes1 == bytes2, "Canonical JSON should be deterministic"
    print(f"  PASSED: Identical canonical bytes for reordered dicts")

    # Test 2: dry_run allowed with no policy file
    print("\n[Test 2] dry_run allowed with no policy file")
    _missing_policy_warned = False
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = f"{tmpdir}/nonexistent.json"
        config = MockExecutionConfig(mode="dry_run")
        result = authorize_execution(config)
        assert result.allowed is True
        assert "no_policy" in result.reason
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 3: live blocked with missing policy file
    print("\n[Test 3] live blocked with missing policy file")
    _missing_policy_warned = False
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = f"{tmpdir}/nonexistent.json"
        config = MockExecutionConfig(mode="live")
        result = authorize_execution(config)
        assert result.allowed is False
        assert result.reason == "missing_policy"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 4: live blocked with expired policy
    print("\n[Test 4] live blocked with expired policy")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict(valid_days=-1)  # Expired yesterday
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live")
        result = authorize_execution(config)
        assert result.allowed is False
        assert result.reason == "expired_policy"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 5: live blocked when execution_enabled == false
    print("\n[Test 5] live blocked when execution_enabled == false")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict(execution_enabled=False)
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live")
        result = authorize_execution(config)
        assert result.allowed is False
        assert result.reason == "execution_disabled"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 6: live blocked when "live" not in allowed_modes
    print("\n[Test 6] live blocked when 'live' not in allowed_modes")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict(allowed_modes=["dry_run"])
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live")
        result = authorize_execution(config)
        assert result.allowed is False
        assert result.reason == "mode_not_allowed"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 7: live allowed when policy valid and signature valid
    print("\n[Test 7] live allowed when policy valid and signature valid")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict()
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live")
        quantile_config = MockQuantileGateConfig()
        result = authorize_execution(config, quantile_config)
        assert result.allowed is True
        assert result.reason == "policy_valid"
        assert result.policy_id == "test-policy-001"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 8: live blocked when constraint_profile not allowed
    print("\n[Test 8] live blocked when constraint_profile not allowed")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict()
        policy_dict["constraint_profiles"] = ["batch_optimized"]  # Remove latency_safe
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live", constraint_profile="latency_safe")
        result = authorize_execution(config)
        assert result.allowed is False
        assert result.reason == "constraint_profile_denied"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 9: live blocked when latency slack exceeds policy ceiling
    print("\n[Test 9] live blocked when latency slack exceeds policy ceiling")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict()
        policy_dict["max_latency_slack_hours"] = 0.01  # Very strict
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(
            mode="live",
            constraint_profile="latency_safe",
            latency_slack_threshold_hours=0.05,  # Exceeds 0.01
        )
        result = authorize_execution(config)
        assert result.allowed is False
        assert result.reason == "latency_slack_violation"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 10: dry_run logs warnings but does not block
    print("\n[Test 10] dry_run logs warnings but does not block")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict(valid_days=-1)  # Expired
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="dry_run")
        result = authorize_execution(config)
        assert result.allowed is True
        assert "dry_run_allowed" in result.reason
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 11: invalid signature blocks live
    print("\n[Test 11] invalid signature blocks live")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict()
        path = write_policy_bundle(tmpdir, policy_dict, alg="invalid_alg")
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live")
        result = authorize_execution(config)
        assert result.allowed is False
        assert result.reason == "invalid_signature"
        print(f"  PASSED: allowed={result.allowed}, reason={result.reason}")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 12: HMAC v0 verification works
    print("\n[Test 12] HMAC v0 verification works")
    policy_dict = create_valid_policy_dict()
    sig_b64 = _generate_hmac_v0_signature(policy_dict)
    sig_info = SignatureInfo(alg="hmac_sha256_v0", key_id="test", sig_b64=sig_b64)
    assert verify_signature(policy_dict, sig_info) is True
    print(f"  PASSED: HMAC v0 signature verified")

    # Test 13: HMAC v0 rejects tampered policy
    print("\n[Test 13] HMAC v0 rejects tampered policy")
    policy_dict = create_valid_policy_dict()
    sig_b64 = _generate_hmac_v0_signature(policy_dict)
    policy_dict["policy_id"] = "tampered"  # Tamper after signing
    sig_info = SignatureInfo(alg="hmac_sha256_v0", key_id="test", sig_b64=sig_b64)
    assert verify_signature(policy_dict, sig_info) is False
    print(f"  PASSED: Tampered policy rejected")

    # Test 14: Quantile config validation - more conservative allowed
    print("\n[Test 14] Quantile config - more conservative allowed")
    policy = PolicyConfig(
        policy_id="test",
        issued_at=datetime.utcnow().date(),
        valid_until=datetime.utcnow().date() + timedelta(days=30),
        allowed_modes=["dry_run", "live"],
        execution_enabled=True,
        constraint_profiles=["batch_optimized"],
        max_latency_slack_hours=0.05,
        quantile=0.9,
        max_downside_risk_pct=10.0,
        min_expected_savings_pct=5.0,
        metric="energy_cost",
    )
    # More conservative config (higher quantile, lower risk, higher savings, both metrics)
    quantile_config = MockQuantileGateConfig(
        quantile=0.95,  # More conservative
        max_downside_risk_pct=5.0,  # More conservative
        min_expected_savings_pct=10.0,  # More conservative
        metric="both",  # More conservative
    )
    valid, reason = validate_quantile_config(quantile_config, policy)
    assert valid is True
    print(f"  PASSED: More conservative config allowed")

    # Test 15: Quantile config validation - less conservative blocked
    print("\n[Test 15] Quantile config - less conservative blocked")
    quantile_config = MockQuantileGateConfig(
        quantile=0.9,
        max_downside_risk_pct=15.0,  # Less conservative than policy
        min_expected_savings_pct=5.0,
        metric="both",
    )
    valid, reason = validate_quantile_config(quantile_config, policy)
    assert valid is False
    assert "max_downside_risk_pct" in reason
    print(f"  PASSED: Less conservative config blocked: {reason}")

    # Test 16: Metric conservativeness - policy=both requires config=both
    print("\n[Test 16] Metric conservativeness - policy=both requires config=both")
    policy_both = PolicyConfig(
        policy_id="test",
        issued_at=datetime.utcnow().date(),
        valid_until=datetime.utcnow().date() + timedelta(days=30),
        allowed_modes=["dry_run", "live"],
        execution_enabled=True,
        constraint_profiles=["batch_optimized"],
        max_latency_slack_hours=0.05,
        quantile=0.9,
        max_downside_risk_pct=10.0,
        min_expected_savings_pct=5.0,
        metric="both",
    )
    quantile_config = MockQuantileGateConfig(
        quantile=0.9,
        max_downside_risk_pct=10.0,
        min_expected_savings_pct=5.0,
        metric="energy_cost",  # Less conservative than "both"
    )
    valid, reason = validate_quantile_config(quantile_config, policy_both)
    assert valid is False
    assert "metric" in reason
    print(f"  PASSED: policy=both rejects config=energy_cost")

    # Test 17: issued_at is informational only (old date should not block)
    print("\n[Test 17] issued_at is informational only")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict(valid_days=30)
        # Set issued_at to a very old date
        policy_dict["issued_at"] = "2020-01-01"
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live")
        quantile_config = MockQuantileGateConfig()
        result = authorize_execution(config, quantile_config)
        assert result.allowed is True
        print(f"  PASSED: Old issued_at does not block execution")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    # Test 18: valid_until boundary (exactly today is valid)
    print("\n[Test 18] valid_until boundary - exactly today is valid")
    with tempfile.TemporaryDirectory() as tmpdir:
        policy_dict = create_valid_policy_dict()
        policy_dict["valid_until"] = datetime.utcnow().date().isoformat()
        path = write_policy_bundle(tmpdir, policy_dict)
        os.environ["AURELIUS_POLICY_BUNDLE_PATH"] = path
        config = MockExecutionConfig(mode="live")
        quantile_config = MockQuantileGateConfig()
        result = authorize_execution(config, quantile_config)
        assert result.allowed is True
        print(f"  PASSED: valid_until = today is still valid")
    del os.environ["AURELIUS_POLICY_BUNDLE_PATH"]

    print("\n" + "=" * 60)
    print("All 18 tests passed!")
    print("=" * 60)
