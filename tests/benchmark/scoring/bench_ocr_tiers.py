#!/usr/bin/env python3
"""Per-tier end-to-end throughput bench for the PP-OCRv6 migration (plan §5.7).

Drives a running turbo-ocr server over `/ocr/raw` on a dense fixture and reports
*end-to-end* pages/s per OCR_MODEL tier — not the @100-crops micro-bench. The
non-negotiable floor is >=10 pages/s (GATE A/F; tiny clears it too per GATE G,
where only its accuracy regression is report-only).

Tiers
  OCR_MODEL in {medium, small, tiny} select the registry entries.
  OCR_MODEL=v5 has no registry entry; it is expressed by pointing REC_ONNX /
  DET_ONNX at the legacy v5 det/rec so the v6 tiers can be compared against the
  frozen baseline through the same harness.

Two ways to run
  Attached (default): start the server yourself with the tier's env, then
      bench_ocr_tiers.py --tiers medium
  Spawned: let the bench restart the server per tier (needs the built binary)
      bench_ocr_tiers.py --spawn build/turboocr-server --tiers medium,small,tiny

The v5 comparison resolves its paths from env overrides, e.g.
  V5_DET_ONNX=models/det.onnx V5_REC_ONNX=models/rec.onnx \
      bench_ocr_tiers.py --spawn build/turboocr-server --tiers v5,medium
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FLOOR_PAGES_PER_SEC = 10.0
TIER_ENV = "OCR_MODEL"

# Each v6 tier maps to a registry name resolved server-side via OCR_MODEL. The
# v5 comparison has no registry entry, so it carries explicit det/rec overrides
# (taken from V5_DET_ONNX / V5_REC_ONNX, defaulting to the flat slots).
V6_TIERS = ("medium", "small", "tiny")


def _dense_fixture() -> bytes:
    """A text-dense page so the det candidate budget and rec batch are exercised."""
    candidates = [
        REPO / "tests/fixtures/images/png/dense_text.png",
        REPO / "tests/fixtures/images/png/table.png",
        REPO.parent / "compare-ocrs/funsd_cache/funsd_000.png",
        REPO / "tests/fixtures/images/png/business_letter.png",
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    raise FileNotFoundError(
        "no dense image fixture found; expected tests/fixtures/images/png/dense_text.png"
    )


def _tier_env(tier: str) -> dict[str, str]:
    """Server env that selects `tier`. Inherits the caller env; overrides win."""
    env = dict(os.environ)
    # A clean slate so a stale selector from the caller can't shadow the tier.
    for k in (TIER_ENV, "OCR_LANG", "REC_ONNX", "DET_ONNX", "REC_DICT"):
        env.pop(k, None)
    if tier == "v5":
        env["DET_ONNX"] = os.environ.get("V5_DET_ONNX", "models/det.onnx")
        env["REC_ONNX"] = os.environ.get("V5_REC_ONNX", "models/rec.onnx")
        if v5_dict := os.environ.get("V5_REC_DICT"):
            env["REC_DICT"] = v5_dict
    else:
        env[TIER_ENV] = tier
    return env


def _wait_healthy(url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


def _spawn_server(binary: str, url: str, env: dict[str, str], boot_timeout_s: float):
    proc = subprocess.Popen([binary], env=env, cwd=str(REPO),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_healthy(url, boot_timeout_s):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError(f"server did not become healthy within {boot_timeout_s:.0f}s")
    return proc


def _post_raw(url: str, body: bytes, timeout_s: float) -> int:
    req = urllib.request.Request(f"{url}/ocr/raw", data=body, method="POST",
                                 headers={"Content-Type": "image/png"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        results = json.load(r).get("results", [])
        return len(results)


def measure_tier(url: str, body: bytes, iters: int, concurrency: int,
                 req_timeout_s: float) -> dict:
    """End-to-end pages/s for one already-configured server."""
    _post_raw(url, body, req_timeout_s)  # warmup
    _post_raw(url, body, req_timeout_s)

    boxes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        t0 = time.perf_counter()
        for n in pool.map(lambda _: _post_raw(url, body, req_timeout_s), range(iters)):
            boxes.append(n)
        elapsed = time.perf_counter() - t0
    return {
        "pages_per_sec": iters / elapsed if elapsed else 0.0,
        "iters": iters,
        "concurrency": concurrency,
        "elapsed_s": elapsed,
        "median_boxes": sorted(boxes)[len(boxes) // 2] if boxes else 0,
    }


def run(tiers: list[str], url: str, iters: int, concurrency: int, req_timeout_s: float,
        spawn: str | None, boot_timeout_s: float) -> dict:
    body = _dense_fixture()
    results: dict[str, dict] = {}
    for tier in tiers:
        proc = None
        if spawn:
            proc = _spawn_server(spawn, url, _tier_env(tier), boot_timeout_s)
        elif not _wait_healthy(url, 5):
            raise RuntimeError(
                f"no server at {url}; start one for tier '{tier}' or pass --spawn"
            )
        try:
            results[tier] = measure_tier(url, body, iters, concurrency, req_timeout_s)
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-url", default=os.environ.get("OCR_SERVER_URL",
                                                            "http://localhost:8080"))
    ap.add_argument("--tiers", default="medium",
                    help="comma list from {medium,small,tiny,v5}")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--req-timeout", type=float, default=60.0)
    ap.add_argument("--spawn", default=None,
                    help="server binary to (re)launch per tier instead of attaching")
    ap.add_argument("--boot-timeout", type=float, default=180.0)
    ap.add_argument("--floor", type=float, default=FLOOR_PAGES_PER_SEC)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in (*V6_TIERS, "v5")]
    if unknown:
        ap.error(f"unknown tier(s): {', '.join(unknown)}")

    results = run(tiers, args.server_url, args.iters, args.concurrency,
                  args.req_timeout, args.spawn, args.boot_timeout)

    print(f"{'tier':<8} {'pages/s':>9} {'c':>4} {'boxes':>6}  floor>={args.floor:.0f}")
    print("-" * 44)
    failures = []
    for tier, r in results.items():
        pps = r["pages_per_sec"]
        ok = pps >= args.floor
        # tiny clears the throughput floor too; only its accuracy regression is
        # report-only (GATE G), which this throughput bench does not measure.
        if not ok:
            failures.append(tier)
        print(f"{tier:<8} {pps:>9.1f} {r['concurrency']:>4} {r['median_boxes']:>6}  "
              f"{'PASS' if ok else 'FAIL'}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"floor": args.floor, "results": results, "failures": failures}, indent=2))

    if failures:
        print(f"\nFAIL: tiers below {args.floor:.0f} pages/s: {', '.join(failures)}",
              file=sys.stderr)
        return 1
    print(f"\nOK: all tiers >= {args.floor:.0f} pages/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
