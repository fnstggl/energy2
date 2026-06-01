#!/usr/bin/env python3
"""Phase-0 CC-traces strength expansion (80 MiB -> bounded 3000 MiB).

Re-uses the per-request flatten logic from
``scripts/ingest_hf_gap_datasets.py`` but with a 3000 MiB budget. Writes
the comparison summary expected by Phase 0 of the cache/prefix-reuse
forecaster mission spec.

Outputs:
- ``data/external/hf/semianalysisai__cc-traces-weka-no-subagents-051226/
  traces_3000mib/processed/normalized_sample.jsonl`` (gitignored, sized via
  --max-normalized-bytes / --max-normalized-rows)
- ``data/external/forecasting/cache_prefix_reuse_v1/
  cc_traces_strength_expansion.json`` (committed)

No raw, no analysis_sample.jsonl is committed. The normalized_sample
contains hashed block-id summaries only — no raw prompt/completion text
exists in CC-traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW_URL = (
    "https://huggingface.co/datasets/"
    "semianalysisai/cc-traces-weka-no-subagents-051226/"
    "resolve/main/traces.jsonl"
)
RAW_DIR = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "raw"
NEW_PROC_DIR = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "traces_3000mib" / "processed"
OLD_NORM_SAMPLE = REPO_ROOT / "data" / "external" / "hf" / (
    "semianalysisai__cc-traces-weka-no-subagents-051226"
) / "traces_head" / "processed" / "normalized_sample.jsonl"
OUT_SUMMARY = REPO_ROOT / "data" / "external" / "forecasting" / (
    "cache_prefix_reuse_v1"
) / "cc_traces_strength_expansion.json"

DEFAULT_MAX_BYTES = 3000 * 1024 * 1024  # 3000 MiB
DEFAULT_MAX_NORMALIZED_BYTES = 8 * 1024 * 1024  # 8 MiB cap for committed normalized
DEFAULT_MAX_NORMALIZED_ROWS = 5000  # rows in committed normalized sample
PROGRESS_INTERVAL_S = 15


def _bounded_download(url: str, dest: Path, *, max_bytes: int) -> dict:
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "aurelius-cc-traces-3000mib-expand/1.0",
        "Range": f"bytes=0-{int(max_bytes - 1)}",
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    written = 0
    truncated = False
    status = None
    err: str | None = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp:
            status = resp.getcode()
            last_log = time.monotonic()
            t0 = last_log
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    remaining = max_bytes - written
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(chunk) > remaining:
                        out.write(chunk[:remaining])
                        written += remaining
                        truncated = True
                        break
                    out.write(chunk)
                    written += len(chunk)
                    now = time.monotonic()
                    if now - last_log >= PROGRESS_INTERVAL_S:
                        mb = written / (1024 * 1024)
                        rate = (written / (now - t0)) / (1024 * 1024)
                        print(f"  [dl] {mb:.1f} MiB / {max_bytes/1024/1024:.0f} MiB  "
                              f"({rate:.2f} MiB/s)")
                        last_log = now
    except urllib.error.HTTPError as e:
        err = f"HTTPError {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        err = f"URLError: {e.reason}"
    except Exception as e:  # pragma: no cover
        err = f"{type(e).__name__}: {e}"
    return {
        "url": url, "dest": str(dest), "status": status,
        "downloaded_bytes": written, "truncated": truncated,
        "max_bytes": max_bytes, "error": err,
    }


def _read_jsonl_sessions(path: Path):
    """Yield session-level dicts from CC-traces, dropping a trailing partial line."""
    with open(path, "rb") as fh:
        data = fh.read()
    # Drop incomplete trailing line (Range download often truncates mid-line).
    last = data.rfind(b"\n")
    if last >= 0:
        data = data[:last]
    for raw_line in data.splitlines():
        if not raw_line.strip():
            continue
        try:
            yield json.loads(raw_line)
        except json.JSONDecodeError:
            continue


def _hash_block_ids(block_ids) -> tuple[str, int]:
    if not isinstance(block_ids, list):
        return ("", 0)
    n = len(block_ids)
    h = hashlib.blake2b(digest_size=8)
    for x in block_ids:
        h.update(str(x).encode("utf-8"))
        h.update(b",")
    return (h.hexdigest(), n)


def _flatten_session(sess: dict, max_rows: int | None) -> list[dict]:
    """Flatten one CC-traces session into per-request rows.

    Carries session-level keys (id, block_size, hash_id_scope, models) into
    each request. Hashes per-request block-id list and records its length.
    Drops the raw block_ids list (so committed sample contains no list).
    """
    if not isinstance(sess, dict):
        return []
    session_id = sess.get("id")
    block_size = sess.get("block_size")
    hash_scope = sess.get("hash_id_scope")
    models = sess.get("models") or []
    reqs = sess.get("requests") or []
    out: list[dict] = []
    for turn, req in enumerate(reqs):
        if not isinstance(req, dict):
            continue
        bh_hash, bh_count = _hash_block_ids(req.get("hash_ids"))
        row = {
            "session_id": session_id,
            "block_size_tokens": block_size,
            "hash_id_scope": hash_scope,
            "session_models": json.dumps(sorted(set(models))),
            "requests_count": len(reqs),
            "turn_index": int(turn),
            "request_arrival_delta_s": req.get("t"),
            "model_id": req.get("model"),
            "input_tokens": req.get("in"),
            "output_tokens": req.get("out"),
            "block_hashes_count": bh_count,
            "block_hashes_hash": bh_hash,
            "api_time_s": req.get("api_time"),
            "think_time_s": req.get("think_time"),
            "ttft_s": req.get("ttft"),
            "request_type": req.get("type"),
        }
        out.append(row)
        if max_rows is not None and len(out) >= max_rows:
            return out
    return out


def _safe_jsonable(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float):
            if v != v:  # NaN
                continue
            out[k] = v
        elif v is None:
            continue
        elif isinstance(v, (int, str, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _old_sample_stats() -> dict:
    """Compute the 80 MiB sample's reuse stats from the committed
    normalized sample (no re-download)."""
    rows = []
    if OLD_NORM_SAMPLE.exists():
        with open(OLD_NORM_SAMPLE) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return _compute_strength(rows, label="80_MiB_committed")


def _compute_strength(rows: list[dict], *, label: str) -> dict:
    """Reuse + diversity stats for a CC-traces request-flattened sample."""
    n = len(rows)
    sessions = set()
    hash_to_count: dict[str, int] = {}
    request_types: dict[str, int] = {}
    models: dict[str, int] = {}
    has_ttft = 0
    has_api_time = 0
    has_think_time = 0
    intra_session_reuse_examples = 0
    cache_loss_proxy_examples = 0
    # Per-session ordered turn list for in-session reuse detection.
    per_session: dict[str, list[dict]] = {}
    for r in rows:
        s = r.get("session_id")
        if s is not None:
            sessions.add(s)
        h = r.get("block_hashes_hash") or ""
        if h:
            hash_to_count[h] = hash_to_count.get(h, 0) + 1
        rt = r.get("request_type")
        if rt is not None:
            request_types[str(rt)] = request_types.get(str(rt), 0) + 1
        m = r.get("model_id")
        if m is not None:
            models[str(m)] = models.get(str(m), 0) + 1
        if r.get("ttft_s") is not None:
            has_ttft += 1
        if r.get("api_time_s") is not None:
            has_api_time += 1
        if r.get("think_time_s") is not None:
            has_think_time += 1
        per_session.setdefault(str(s), []).append(r)
    # In-session reuse: between consecutive turns of the same session, count
    # how many block_hashes_count values increase (prefix-grew but did not
    # reset). Count repeated hashes (likely retry / prefix-identical).
    for sid, turns in per_session.items():
        turns.sort(key=lambda r: (r.get("turn_index") or 0))
        prev_hash = None
        prev_count = None
        for r in turns:
            ch = r.get("block_hashes_count")
            hh = r.get("block_hashes_hash")
            if prev_count is not None and ch is not None:
                # cache-loss proxy: block count fell to a small value after a
                # large one (likely a cache eviction or new context)
                if prev_count >= 100 and ch < (prev_count * 0.25):
                    cache_loss_proxy_examples += 1
            if prev_hash is not None and hh is not None:
                # in-session reuse: same hash, or count grew (prefix extends)
                if hh == prev_hash:
                    intra_session_reuse_examples += 1
                elif (prev_count is not None and ch is not None
                      and ch >= prev_count and prev_count >= 1):
                    intra_session_reuse_examples += 1
            prev_hash = hh
            prev_count = ch
    repeated_hashes = sum(1 for c in hash_to_count.values() if c >= 2)
    return {
        "label": label,
        "session_count": len(sessions),
        "request_count": n,
        "unique_kv_block_hashes": len(hash_to_count),
        "repeated_kv_block_hashes": repeated_hashes,
        "intra_session_reuse_examples": intra_session_reuse_examples,
        "cache_loss_proxy_examples": cache_loss_proxy_examples,
        "request_type_distribution": dict(sorted(request_types.items())),
        "model_distribution": dict(sorted(models.items())),
        "ttft_coverage_rows": has_ttft,
        "api_time_coverage_rows": has_api_time,
        "think_time_coverage_rows": has_think_time,
        "ttft_coverage_pct": (has_ttft / n * 100.0) if n else 0.0,
        "api_time_coverage_pct": (has_api_time / n * 100.0) if n else 0.0,
        "think_time_coverage_pct": (has_think_time / n * 100.0) if n else 0.0,
    }


def _decide(old: dict, new: dict) -> tuple[str, str]:
    """Adaptive expansion rule per mission spec.

    Use 3000 MiB for training iff request_count >= 2500 OR session_count >= 20,
    AND unique_kv_block_hashes increases >= 3x, AND repeated_kv_block_hashes
    or reuse_example_count is nonzero.
    """
    req_ok = new["request_count"] >= 2500 or new["session_count"] >= 20
    if not req_ok:
        return ("not_worth_expansion",
                f"request_count={new['request_count']}, session_count="
                f"{new['session_count']} fails 2500 / 20 threshold")
    old_unique = max(1, old["unique_kv_block_hashes"])
    ratio = new["unique_kv_block_hashes"] / old_unique
    if ratio < 3.0:
        return ("diagnostic_only",
                f"unique_kv_block_hashes ratio "
                f"{ratio:.2f}x < 3.0x (old={old['unique_kv_block_hashes']},"
                f" new={new['unique_kv_block_hashes']})")
    if new["repeated_kv_block_hashes"] == 0 and new["intra_session_reuse_examples"] == 0:
        return ("diagnostic_only",
                "no repeated KV hashes and no intra-session reuse examples — "
                "no cache-residency signal even after expansion")
    return ("use_for_training",
            f"expanded sample passes all gates: {new['request_count']} "
            f"requests, {new['session_count']} sessions, unique-hash "
            f"ratio {ratio:.2f}x, repeated={new['repeated_kv_block_hashes']}, "
            f"intra_session_reuse={new['intra_session_reuse_examples']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    p.add_argument("--max-normalized-bytes", type=int,
                   default=DEFAULT_MAX_NORMALIZED_BYTES)
    p.add_argument("--max-normalized-rows", type=int,
                   default=DEFAULT_MAX_NORMALIZED_ROWS)
    p.add_argument("--max-flatten-rows", type=int, default=200_000,
                   help="cap on per-request rows after flatten (memory bound)")
    p.add_argument("--skip-download", action="store_true",
                   help="reuse existing raw file (for re-runs)")
    args = p.parse_args(argv)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NEW_PROC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "traces_3000mib.jsonl"

    t0 = time.monotonic()
    if args.skip_download and raw_path.exists():
        manifest = {
            "url": RAW_URL, "dest": str(raw_path),
            "downloaded_bytes": raw_path.stat().st_size,
            "truncated": True, "max_bytes": args.max_bytes,
            "status": None, "error": None, "reused_existing": True,
        }
    else:
        print(f"[phase0] download up to {args.max_bytes/1024/1024:.0f} MiB ...")
        manifest = _bounded_download(RAW_URL, raw_path, max_bytes=args.max_bytes)
        if manifest.get("error"):
            print(f"[phase0] download error: {manifest['error']}")
    print(f"[phase0] downloaded {manifest['downloaded_bytes']/1024/1024:.1f} MiB"
          f" in {time.monotonic()-t0:.1f}s")

    # Flatten sessions to per-request rows.
    print("[phase0] flattening sessions -> requests ...")
    t0 = time.monotonic()
    flat: list[dict] = []
    n_sessions = 0
    for sess in _read_jsonl_sessions(raw_path):
        n_sessions += 1
        flat.extend(_flatten_session(sess, max_rows=None))
        if len(flat) >= args.max_flatten_rows:
            break
    print(f"[phase0] flattened {n_sessions} sessions -> {len(flat)} requests "
          f"in {time.monotonic()-t0:.1f}s")

    # Strength stats vs the 80 MiB committed sample.
    old = _old_sample_stats()
    new = _compute_strength(flat, label="3000_MiB_cap")

    # Decision
    decision, reason = _decide(old, new)
    print(f"[phase0] decision = {decision}  reason: {reason}")

    # Committed normalized sample — small head of the flattened rows.
    safe_rows = [_safe_jsonable(r) for r in flat[:args.max_normalized_rows]]
    norm_path = NEW_PROC_DIR / "normalized_sample.jsonl"
    written = 0
    with open(norm_path, "w") as fh:
        for r in safe_rows:
            line = json.dumps(r, sort_keys=True) + "\n"
            line_bytes = len(line.encode("utf-8"))
            if written + line_bytes > args.max_normalized_bytes:
                break
            fh.write(line)
            written += line_bytes
    norm_rows = sum(1 for _ in open(norm_path))
    norm_sha = hashlib.sha256(norm_path.read_bytes()).hexdigest()
    print(f"[phase0] committed normalized sample: {norm_path}  "
          f"rows={norm_rows} bytes={written}")

    # Forbid commit of analysis_sample / raw — defensive: do NOT write
    # analysis_sample.jsonl in the new directory.
    analysis_path = NEW_PROC_DIR / "analysis_sample.jsonl"
    if analysis_path.exists():
        analysis_path.unlink()

    payload = {
        "doc_version": "cc_traces_strength_expansion_v1",
        "raw_url": RAW_URL,
        "raw_committed": False,
        "analysis_sample_committed": False,
        "normalized_sample_committed": True,
        "normalized_sample_path": str(norm_path.relative_to(REPO_ROOT)),
        "normalized_sample_bytes": written,
        "normalized_sample_rows": norm_rows,
        "normalized_sample_sha256": norm_sha,
        "normalized_sample_caps": {
            "max_bytes": args.max_normalized_bytes,
            "max_rows": args.max_normalized_rows,
        },
        "raw_download_manifest": manifest,
        "old_sample_stats_80_mib_committed": old,
        "new_sample_stats_3000_mib_cap": new,
        "deltas": {
            "session_count_delta": new["session_count"] - old["session_count"],
            "request_count_delta": new["request_count"] - old["request_count"],
            "unique_kv_block_hashes_delta": (
                new["unique_kv_block_hashes"] - old["unique_kv_block_hashes"]),
            "unique_kv_block_hashes_ratio": (
                new["unique_kv_block_hashes"] / max(1, old["unique_kv_block_hashes"])),
            "repeated_kv_block_hashes_delta": (
                new["repeated_kv_block_hashes"]
                - old["repeated_kv_block_hashes"]),
            "intra_session_reuse_examples_delta": (
                new["intra_session_reuse_examples"]
                - old["intra_session_reuse_examples"]),
            "cache_loss_proxy_examples_delta": (
                new["cache_loss_proxy_examples"]
                - old["cache_loss_proxy_examples"]),
        },
        "decision": decision,
        "decision_reason": reason,
        "production_claim": False,
        "modifies_controllers_or_defaults": False,
        "modifies_robust_energy_engine": False,
        "uses_oracle_as_headline": False,
        "shadow_only": True,
        "no_raw_prompt_completion_text_committed": True,
        "evaluated_at_s": time.time(),
    }
    OUT_SUMMARY.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[phase0] wrote {OUT_SUMMARY}")
    print(f"[phase0] decision={decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
