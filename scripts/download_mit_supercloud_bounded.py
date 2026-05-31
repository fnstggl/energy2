#!/usr/bin/env python3
"""Bounded MIT Supercloud download from the public S3 bucket.

Downloads ONLY the small high-level scheduler files in full (slurm-log
~ 98 MB, labelled_jobids ~ 80 KB, tres-mapping ~ 111 B,
labelled_job_stats ~ 339 B) and, optionally, a bounded sample of
``node-data.csv`` (~2.1 GB total — sampled via HTTP Range GET on the
first N rows) and a budget-capped sample of per-job GPU CSVs from
``gpu/``. The full ~1–2 TB dataset is NEVER fetched.

The bucket
``s3://mit-supercloud-dataset/datacenter-challenge/202201/`` is public
and supports anonymous HTTP GET / Range GET, so this script uses
stdlib ``urllib`` and an XML listing call — no AWS CLI required.

Honesty rules (asserted by tests):

- Total downloaded bytes are tracked + printed; the script aborts
  before exceeding the configured budget (``--max-gpu-bytes-mb``).
- A manifest of every download is written so the choice of what was
  downloaded / skipped / how-many-bytes is auditable.
- No raw archives are committed (``.gitignore`` excludes the raw
  tree).

Honesty / non-goals:

- This is NOT a full archive download. The full dataset is ~1–2 TB.
- The fetched files are the ones documented in the MIT README + the
  intro notebook (``scheduler-log.csv`` / ``labelled_jobids.csv`` /
  ``tres-mapping.txt`` / ``node-data.csv`` / ``gpu/<NN>/<job-id>.csv``).
- No production mutation, no ML training, no serving-frontier change,
  no robust-energy-engine change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable, Iterator, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW = os.path.join(REPO_ROOT, "data", "external",
                            "mit_supercloud", "raw")
DEFAULT_BUCKET_URL = (
    "https://mit-supercloud-dataset.s3.amazonaws.com/"
    "datacenter-challenge/202201/")

# Pre-registered list of the canonical small files (downloaded in full).
DEFAULT_FILES: tuple = (
    ("slurm-log.csv", "scheduler"),
    ("labelled_jobids.csv", "label"),
    ("labelled_job_stats.csv", "label_stats"),
    ("tres-mapping.txt", "tres_mapping"),
    ("LICENSE", "license"),
    ("README.md", "readme"),
)


# ---------------------------------------------------------------------------
# S3-style ListBucket helpers (anonymous; bucket is public).
# ---------------------------------------------------------------------------

def _http_get(url: str, *, byte_range: Optional[tuple] = None,
               timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url)
    if byte_range is not None:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_head(url: str, *, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return dict(resp.headers)


def s3_list(bucket_root_url: str, *, prefix: str = "",
             delimiter: str = "", max_keys: int = 1000,
             continuation_token: Optional[str] = None
             ) -> tuple[list, list, Optional[str]]:
    """Anonymous S3 ListObjectsV2 against a public bucket.

    Returns ``(files, common_prefixes, next_continuation_token)`` where
    ``files`` is a list of ``(key, size_bytes)`` and ``common_prefixes``
    is the subdirectory list.
    """
    # The bucket root URL is the standard virtual-hosted form
    # ``https://bucket.s3.amazonaws.com/path/``; rewrite to the
    # bucket-level URL ``https://bucket.s3.amazonaws.com/`` for ListObjects.
    parsed = urllib.parse.urlparse(bucket_root_url)
    bucket_url = f"{parsed.scheme}://{parsed.netloc}/"
    # The user passes the path-prefix as ``--bucket`` ending with
    # ``datacenter-challenge/202201/`` — split off the path component
    # and append it to the caller's ``prefix``.
    root_prefix = parsed.path.lstrip("/")
    full_prefix = root_prefix + prefix
    qs = {"list-type": "2", "prefix": full_prefix,
           "max-keys": str(max_keys)}
    if delimiter:
        qs["delimiter"] = delimiter
    if continuation_token:
        qs["continuation-token"] = continuation_token
    url = bucket_url + "?" + urllib.parse.urlencode(qs)
    body = _http_get(url).decode("utf-8")
    files = re.findall(
        r"<Contents>.*?<Key>(.*?)</Key>.*?<Size>(.*?)</Size>", body, re.S)
    files = [(k, int(s)) for k, s in files]
    prefixes = re.findall(
        r"<CommonPrefixes>.*?<Prefix>(.*?)</Prefix>", body, re.S)
    next_token = None
    m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>",
                  body)
    if m:
        next_token = m.group(1)
    return files, prefixes, next_token


def s3_iter_list(bucket_root_url: str, *, prefix: str = "",
                  delimiter: str = "", max_pages: Optional[int] = None
                  ) -> Iterator[tuple]:
    """Yield ``(file_key, file_size_bytes)`` lazily across paginated
    ListObjectsV2 responses."""
    token = None
    page = 0
    while True:
        files, _prefixes, token = s3_list(
            bucket_root_url, prefix=prefix, delimiter=delimiter,
            max_keys=1000, continuation_token=token)
        for f in files:
            yield f
        page += 1
        if not token or (max_pages is not None and page >= max_pages):
            break


# ---------------------------------------------------------------------------
# Download helpers.
# ---------------------------------------------------------------------------

def _sha256_file(path: str, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _download_file(url: str, dest: str, *, dry_run: bool = False,
                    timeout: float = 600.0) -> int:
    """Stream-download ``url`` to ``dest`` (chunked). Returns size_bytes.
    In ``dry_run`` mode, only HEADs to report the size."""
    if dry_run:
        head = _http_head(url, timeout=30.0)
        return int(head.get("Content-Length", 0))
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".",
                exist_ok=True)
    tmp = dest + ".part"
    n_bytes = 0
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp, \
            open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            n_bytes += len(chunk)
    os.replace(tmp, dest)
    return n_bytes


def _download_range(url: str, dest: str, *, n_bytes: int,
                     dry_run: bool = False) -> int:
    """HTTP Range GET — fetch the first ``n_bytes`` of ``url``.

    Used to sample the first chunk of a large CSV (e.g. node-data.csv)
    so we get a representative bounded sample without pulling the
    entire 2.1 GB file. The first row will always be the CSV header;
    the last partial row is trimmed.
    """
    if dry_run:
        return n_bytes
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".",
                exist_ok=True)
    payload = _http_get(url, byte_range=(0, n_bytes - 1))
    # Trim the trailing partial row (everything past the last newline).
    last_nl = payload.rfind(b"\n")
    if last_nl > 0:
        payload = payload[: last_nl + 1]
    with open(dest, "wb") as fh:
        fh.write(payload)
    return len(payload)


# ---------------------------------------------------------------------------
# Manifest builder.
# ---------------------------------------------------------------------------

def _manifest_entry(s3_uri: str, local_path: str, size_bytes: int,
                     downloaded: bool, reason: str,
                     sample_policy: str,
                     sha256: Optional[str] = None) -> dict:
    return {
        "s3_uri": s3_uri, "local_path": local_path,
        "size_bytes": size_bytes, "downloaded": downloaded,
        "reason": reason, "sample_policy": sample_policy,
        "sha256": sha256,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", default=DEFAULT_BUCKET_URL,
                   help="virtual-hosted bucket URL ending with "
                        "``datacenter-challenge/202201/``")
    p.add_argument("--raw-dir", default=DEFAULT_RAW)
    p.add_argument("--download-scheduler", default="true",
                   choices=("true", "false"))
    p.add_argument("--download-labels", default="true",
                   choices=("true", "false"))
    p.add_argument("--download-tres", default="true",
                   choices=("true", "false"))
    p.add_argument("--download-node-sample", default="true",
                   choices=("true", "false"),
                   help="HTTP-Range-GET the first N bytes of "
                        "node-data.csv (default ~50 MB sample) rather "
                        "than pulling the full 2.1 GB file")
    p.add_argument("--node-sample-rows", type=int, default=200_000,
                   help="cap on rows kept from the node-data sample "
                        "(after Range-GET truncation)")
    p.add_argument("--node-sample-bytes-mb", type=int, default=50,
                   help="HTTP-Range-GET budget for node-data.csv")
    p.add_argument("--download-gpu-sample", default="false",
                   choices=("true", "false"),
                   help="opt-in: also sample per-job GPU utilization "
                        "CSVs from gpu/")
    p.add_argument("--max-gpu-files", type=int, default=50,
                   help="cap on per-job GPU CSV files downloaded")
    p.add_argument("--max-gpu-bytes-mb", type=int, default=100,
                   help="total bytes budget for the per-job GPU sample")
    p.add_argument("--gpu-shard-prefix", default="0000",
                   help="which gpu/<shard>/ to sample from (the bucket "
                        "has shards 0000..0099)")
    p.add_argument("--gpu-job-id-sample", default=None,
                   help="path to a newline-delimited list of job_ids "
                        "to preferentially download from gpu/; if "
                        "omitted, the smallest files are sampled "
                        "first to maximize coverage per byte")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="HEAD only; print sizes; never write to disk")
    p.add_argument("--manifest-path", default=None)
    args = p.parse_args(argv)

    os.makedirs(args.raw_dir, exist_ok=True)
    manifest: list[dict] = []
    total_bytes = 0
    bucket_url = (args.bucket if args.bucket.endswith("/")
                  else args.bucket + "/")

    # --- canonical small files ---
    file_flags = {
        "slurm-log.csv": args.download_scheduler == "true",
        "labelled_jobids.csv": args.download_labels == "true",
        "labelled_job_stats.csv": args.download_labels == "true",
        "tres-mapping.txt": args.download_tres == "true",
        "LICENSE": True,
        "README.md": True,
    }
    for name, classification in DEFAULT_FILES:
        url = bucket_url + name
        s3_uri = "s3://" + urllib.parse.urlparse(bucket_url).netloc \
                  .split(".s3.")[0] + "/" \
                 + urllib.parse.urlparse(bucket_url).path.lstrip("/") + name
        local = os.path.join(args.raw_dir, name)
        if not file_flags.get(name, True):
            manifest.append(_manifest_entry(
                s3_uri, local, 0, False,
                f"skipped: --download-* flag disabled for {classification}",
                "full_file"))
            continue
        try:
            n = _download_file(url, local, dry_run=args.dry_run)
            sha = (_sha256_file(local) if not args.dry_run
                    and os.path.exists(local) else None)
            manifest.append(_manifest_entry(
                s3_uri, local, n, not args.dry_run,
                "downloaded in full", "full_file", sha))
            total_bytes += n
            print(f"[download] {'(dry) ' if args.dry_run else ''}"
                  f"{name:30s}  {n:>12,} B  -> {local}", flush=True)
        except urllib.error.HTTPError as exc:
            manifest.append(_manifest_entry(
                s3_uri, local, 0, False,
                f"http_error:{exc.code}:{exc.reason}", "full_file"))
            print(f"[download] FAILED {name}: {exc}", file=sys.stderr)

    # --- node-data.csv: HTTP-Range bounded sample ---
    if args.download_node_sample == "true":
        url = bucket_url + "node-data.csv"
        s3_uri = "s3://mit-supercloud-dataset/datacenter-challenge/" \
                  "202201/node-data.csv"
        local = os.path.join(args.raw_dir, "node-data.csv")
        budget = args.node_sample_bytes_mb * 1024 * 1024
        try:
            n = _download_range(url, local, n_bytes=budget,
                                 dry_run=args.dry_run)
            sha = (_sha256_file(local) if not args.dry_run
                    and os.path.exists(local) else None)
            manifest.append(_manifest_entry(
                s3_uri, local, n, not args.dry_run,
                f"bounded HTTP-Range GET of first "
                f"{args.node_sample_bytes_mb} MB (full file is "
                "~2.1 GB; sampling the head keeps the contiguous "
                "time window)",
                f"range_get_first_{args.node_sample_bytes_mb}MB", sha))
            total_bytes += n
            print(f"[download] {'(dry) ' if args.dry_run else ''}"
                  f"node-data.csv  (Range 0-{budget - 1}): "
                  f"{n:>12,} B  -> {local}", flush=True)
        except urllib.error.HTTPError as exc:
            manifest.append(_manifest_entry(
                s3_uri, local, 0, False,
                f"http_error:{exc.code}:{exc.reason}",
                f"range_get_first_{args.node_sample_bytes_mb}MB"))
            print(f"[download] FAILED node-data.csv: {exc}",
                  file=sys.stderr)
    else:
        manifest.append(_manifest_entry(
            "s3://mit-supercloud-dataset/datacenter-challenge/202201/"
            "node-data.csv",
            os.path.join(args.raw_dir, "node-data.csv"),
            0, False,
            "skipped: --download-node-sample false",
            "range_get_first_N_MB"))

    # --- gpu/ per-job sample (opt-in, budget-capped) ---
    if args.download_gpu_sample == "true":
        budget_bytes = args.max_gpu_bytes_mb * 1024 * 1024
        shard_prefix = f"gpu/{args.gpu_shard_prefix.rstrip('/')}/"
        print(f"[download] listing s3://.../{shard_prefix} for sample "
              f"(budget {args.max_gpu_files} files / "
              f"{args.max_gpu_bytes_mb} MB)", flush=True)
        candidates = list(s3_iter_list(bucket_url, prefix=shard_prefix,
                                         max_pages=10))
        if args.gpu_job_id_sample and os.path.exists(args.gpu_job_id_sample):
            wanted = {ln.strip() for ln in open(args.gpu_job_id_sample)
                       if ln.strip()}
            candidates = [(k, s) for (k, s) in candidates
                          if any(j in os.path.basename(k) for j in wanted)]
        else:
            # Default: smallest-first so we cover more jobs per byte.
            candidates.sort(key=lambda kv: kv[1])
        rng = random.Random(args.seed)
        sampled = candidates[: args.max_gpu_files]
        rng.shuffle(sampled)
        used = 0
        local_gpu_root = os.path.join(args.raw_dir, "gpu")
        for key, size in sampled:
            if used + size > budget_bytes:
                manifest.append(_manifest_entry(
                    "s3://mit-supercloud-dataset/" + key,
                    os.path.join(local_gpu_root,
                                 os.path.relpath(key, "datacenter-challenge/"
                                                      "202201/gpu/")),
                    size, False,
                    f"budget_exhausted ({used} + {size} > "
                    f"{budget_bytes})",
                    f"gpu_sample_n={args.max_gpu_files}_"
                    f"max_mb={args.max_gpu_bytes_mb}"))
                continue
            url = bucket_url + key.replace(
                "datacenter-challenge/202201/", "", 1)
            rel = os.path.relpath(
                key, "datacenter-challenge/202201/gpu/")
            local = os.path.join(local_gpu_root, rel)
            try:
                n = _download_file(url, local, dry_run=args.dry_run)
                used += n
                total_bytes += n
                sha = (_sha256_file(local) if not args.dry_run
                        and os.path.exists(local) else None)
                manifest.append(_manifest_entry(
                    "s3://mit-supercloud-dataset/" + key, local, n,
                    not args.dry_run, "downloaded in full (per-job CSV)",
                    f"gpu_sample_n={args.max_gpu_files}_"
                    f"max_mb={args.max_gpu_bytes_mb}", sha))
                print(f"[download] {'(dry) ' if args.dry_run else ''}"
                      f"gpu/{rel:50s}  {n:>10,} B  (budget "
                      f"{used/1024/1024:.1f}/{args.max_gpu_bytes_mb} MB)",
                      flush=True)
            except urllib.error.HTTPError as exc:
                manifest.append(_manifest_entry(
                    "s3://mit-supercloud-dataset/" + key, local, 0, False,
                    f"http_error:{exc.code}:{exc.reason}",
                    f"gpu_sample_n={args.max_gpu_files}_"
                    f"max_mb={args.max_gpu_bytes_mb}"))
    else:
        manifest.append({
            "s3_uri": "s3://mit-supercloud-dataset/datacenter-challenge/"
                       "202201/gpu/",
            "local_path": os.path.join(args.raw_dir, "gpu"),
            "size_bytes": 0, "downloaded": False,
            "reason": "skipped: --download-gpu-sample false",
            "sample_policy": "gpu_sample",
            "sha256": None,
        })

    # --- write manifest ---
    manifest_path = (args.manifest_path
                     or os.path.join(args.raw_dir,
                                      "bounded_download_manifest.json"))
    payload = {
        "bucket": bucket_url,
        "raw_dir": args.raw_dir,
        "total_downloaded_bytes": total_bytes,
        "total_downloaded_mb": round(total_bytes / 1024 / 1024, 3),
        "dry_run": args.dry_run,
        "config": vars(args),
        "files": manifest,
    }
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)) or ".",
                exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    print(f"\n[download] total downloaded: {total_bytes:,} bytes "
          f"({total_bytes / 1024 / 1024:.2f} MB)")
    print(f"[download] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
