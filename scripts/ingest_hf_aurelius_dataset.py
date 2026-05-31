#!/usr/bin/env python3
"""Bounded ingestion of one Hugging Face dataset.

The script is **schema-first**: it inspects HF metadata before any download,
and refuses to write a sample if unknown columns are present and the caller
didn't pass ``--allow-unknown-columns``. Honours ``HF_TOKEN`` for gated
access. Writes:

- ``data/external/hf/<safe>/raw/<file>`` (gitignored)
- ``data/external/hf/<safe>/processed/sample.jsonl``
- ``data/external/hf/<safe>/processed/summary.json``
- ``tests/fixtures/hf/<safe>_sample.jsonl`` (tiny deterministic fixture)
- updates the canonical corpus registry +
  ``data/external/hf_discovery/hf_dataset_candidates.json`` entry.

Sources accepted (in order):

1. ``--from-local-json <path>``  — already-downloaded JSON / JSONL file
   (the test path; no network).
2. ``--from-hf-file <repo-path>`` — HTTP-Range download of a parquet /
   json file at ``https://huggingface.co/datasets/<id>/resolve/main/<repo-path>``.
   Bounded by ``--max-bytes``. ``pyarrow`` required for parquet.

If neither is supplied AND no cached raw file exists, the script prints a
schema-only summary and exits 0 (mirrors the discovery-only pattern in
``scripts/ingest_lmsys_chatbot_arena.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces.hf_corpus import discovery, ingestion, promotion  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CANDIDATES = os.path.join(
    REPO_ROOT, "data", "external", "hf_discovery", "hf_dataset_candidates.json")
DEFAULT_REGISTRY = os.path.join(
    REPO_ROOT, "data", "external", "hf_discovery", "canonical_corpus_registry.json")


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def _summary_path_relative(summary_path: str) -> str:
    return os.path.relpath(summary_path, REPO_ROOT).replace(os.sep, "/")


def _classify_and_signals(meta_dict) -> tuple[str, list, list]:
    cls = discovery.classify_dataset(meta_dict) if not isinstance(
        meta_dict, dict) else discovery.classify_dataset(meta_dict)
    return cls["trace_type"], [], []


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Bounded HF dataset ingestion.")
    p.add_argument("--dataset-id", required=True,
                   help="HF dataset id, e.g. agent-perf-bench/AgentPerfBench")
    p.add_argument("--trace-type", default=None,
                   help="Override the auto-classified canonical trace type.")
    p.add_argument("--from-local-json", default=None,
                   help="Path to a JSON / JSONL file to ingest (test path).")
    p.add_argument("--from-hf-file", default=None,
                   help="HF repo-relative file path to download via HTTP "
                        "Range, e.g. 'trace_replay/summary.parquet'.")
    p.add_argument("--config-name", default=None,
                   help="HF dataset config name (namespaces the output dir + "
                        "fixture name when one dataset has several configs).")
    p.add_argument("--max-rows", type=int, default=ingestion.DEFAULT_MAX_ROWS)
    p.add_argument("--max-bytes", type=int, default=ingestion.DEFAULT_MAX_BYTES)
    p.add_argument("--allow-unknown-columns", action="store_true",
                   help="Force ingestion when raw schema contains columns "
                        "not in RAW_TO_NORMALIZED. Promotion-gate test will "
                        "still flag the unknown columns.")
    p.add_argument("--fixtures-dir", default=None,
                   help="Use OfflineHFClient + this fixtures dir.")
    p.add_argument("--provenance", default=None,
                   help="Free-form provenance label. Defaults to "
                        "'<dataset_id>@<from>#<git_sha[:7]>'.")
    p.add_argument("--limitations", action="append", default=[],
                   help="Explicit limitation strings. Repeatable.")
    p.add_argument("--write-registry", action="store_true",
                   help="Append the new dataset to canonical_corpus_registry.json.")
    p.add_argument("--canonical-registry-path", default=DEFAULT_REGISTRY)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)s %(name)s: %(message)s")

    # 1. Schema inspection (always).
    if args.fixtures_dir:
        client = discovery.OfflineHFClient(args.fixtures_dir)
    else:
        client = discovery.HFAPIClient(token=os.environ.get("HF_TOKEN"))
    meta, raw_meta = ingestion.inspect_schema(client, args.dataset_id)
    if meta is None:
        print(f"[ingest] HF metadata unavailable for {args.dataset_id} "
              f"(gated/private/404). Marking GATED_BLOCKED.", file=sys.stderr)
        return 0

    classification = discovery.classify_dataset(meta)
    trace_type = args.trace_type or classification["trace_type"]
    available = discovery.available_signals(meta)
    missing = discovery.missing_signals(meta, available)

    if trace_type == "mixed_or_unknown_trace":
        print(f"[ingest] {args.dataset_id} classified mixed_or_unknown_trace; "
              f"specify --trace-type explicitly before ingestion.",
              file=sys.stderr)
        return 2

    # 2. Load raw rows.
    raw_records: list = []
    raw_source_descriptor = ""
    if args.from_local_json:
        raw_records = ingestion.try_load_json_rows(
            args.from_local_json, max_rows=args.max_rows)
        raw_source_descriptor = f"local_json:{os.path.basename(args.from_local_json)}"
    elif args.from_hf_file:
        url = ingestion._hf_file_url(args.dataset_id, args.from_hf_file)
        paths = ingestion.safe_sample_paths(REPO_ROOT, args.dataset_id, args.config_name)
        raw_path = os.path.join(paths["raw_dir"], os.path.basename(args.from_hf_file))
        manifest = ingestion.download_bounded(
            url, raw_path, max_bytes=args.max_bytes,
            token=os.environ.get("HF_TOKEN"))
        rows = ingestion.try_load_parquet_rows(raw_path, max_rows=args.max_rows)
        if rows is None:
            rows = ingestion.try_load_json_rows(raw_path, max_rows=args.max_rows)
        raw_records = rows
        raw_source_descriptor = (
            f"hf_file:{args.from_hf_file}#bytes={manifest['downloaded_bytes']}"
        )
    else:
        print(f"[ingest] no data source — schema-only summary for "
              f"{args.dataset_id}", file=sys.stderr)
        print(json.dumps({
            "dataset_id": args.dataset_id,
            "trace_type": trace_type,
            "available_signals": available,
            "missing_signals": missing,
            "configs": list(meta.configs),
            "splits": [{"config": c, "split": s, "num_examples": n}
                       for (c, s, n) in meta.splits],
            "feature_names": list(meta.feature_names),
        }, indent=2))
        return 0

    if not raw_records:
        print(f"[ingest] no rows loaded for {args.dataset_id}", file=sys.stderr)
        return 3

    provenance = args.provenance or (
        f"{args.dataset_id}@{raw_source_descriptor}#{(_git_sha() or '')[:7]}")

    limitations = list(args.limitations) or [
        f"Bounded ingestion: only first {args.max_rows} rows / "
        f"{args.max_bytes} bytes from {raw_source_descriptor}",
        "Not production telemetry. See docs/HF_DATASET_REGISTRY.md trust tiers.",
    ]

    try:
        result = ingestion.ingest_from_records(
            repo_root=REPO_ROOT,
            dataset_id=args.dataset_id,
            source_url=meta.dataset_url,
            license_str=meta.license,
            gated=meta.gated,
            raw_records=raw_records,
            trace_type=trace_type,
            provenance=provenance,
            available_signals_list=available,
            missing_signals_list=missing,
            limitations=limitations,
            max_rows=args.max_rows,
            max_bytes=args.max_bytes,
            allow_unknown_columns=args.allow_unknown_columns,
            git_sha=_git_sha(),
            config_name=args.config_name,
        )
    except (ingestion.IngestionUnknownColumns, ingestion.IngestionBoundsExceeded) as e:
        print(f"[ingest] REFUSED: {e}", file=sys.stderr)
        return 4

    print(f"[ingest] wrote {result.sample_rows} rows / {result.sample_bytes} bytes")
    print(f"[ingest] sample: {result.sample_path}")
    print(f"[ingest] summary: {result.summary_path}")
    print(f"[ingest] sha256: {result.sha256}")
    if result.unknown_columns:
        print(f"[ingest] unknown columns (recorded, ungated by flag): "
              f"{result.unknown_columns}")

    if args.write_registry:
        with open(result.summary_path) as fh:
            summary = json.load(fh)
        summary["summary_path_relative"] = _summary_path_relative(result.summary_path)
        summary["config_name"] = args.config_name
        decision = promotion.evaluate_promotion(summary)
        registry = promotion.load_canonical_registry(args.canonical_registry_path) or {
            "entries": []
        }
        dedupe_key = (args.dataset_id, args.config_name)
        entries = [
            e for e in registry.get("entries") or []
            if (e.get("dataset_id"), e.get("config_name")) != dedupe_key
        ]
        entries.append(promotion.build_registry_entry(summary, decision))
        promotion.write_canonical_registry(entries, args.canonical_registry_path)
        print(f"[ingest] registry updated: {args.canonical_registry_path} "
              f"-> {decision['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
