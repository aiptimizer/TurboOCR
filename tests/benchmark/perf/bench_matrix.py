"""Single-entry benchmark orchestrator.

Runs latency → throughput → stress phases and emits LATEST.md + LATEST.json.
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import aiohttp

from _harness_import import _harness_mod as H

HERE = Path(__file__).resolve().parent


async def _warmup(session, base_url, seconds: float = 30.0):
    print(f"warming up for {seconds:.0f}s ...")
    import time
    image = H.load_image_bytes()
    stop_at = time.monotonic() + seconds
    n = 0
    while time.monotonic() < stop_at:
        try:
            async with session.post(
                f"{base_url}/ocr/raw", data=image, headers={"Content-Type": "image/png"}
            ) as r:
                await r.read()
                n += 1
        except Exception:
            break
    print(f"  warmup: {n} requests")


async def run(base_url: str, phases: list[str], quick: bool, duration_s: float):
    report = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "server_url": base_url,
        "quick": quick,
    }
    try:
        report["git_sha"] = subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        report["git_sha"] = "unknown"

    n_latency = 50 if quick else 200
    n_tp = 50 if quick else 200

    async with aiohttp.ClientSession() as session:
        await _warmup(session, base_url, seconds=5 if quick else 30)

        if "latency" in phases:
            print("\n=== latency ===")
            report["latency"] = []
            for ep in H.ENDPOINTS:
                r = await H.measure_latency(session, base_url, ep, n=n_latency, warmup=10)
                report["latency"].append(r)
                print(f"  {r['endpoint']:<24} p50={r['p50_ms']:.1f}ms p99={r['p99_ms']:.1f}ms")

        if "throughput" in phases:
            print("\n=== throughput ===")
            report["throughput"] = []
            for ep in H.ENDPOINTS:
                for c in ([1, 8] if quick else [1, 4, 16, 32]):
                    r = await H.measure_throughput(session, base_url, ep, concurrency=c, n=n_tp)
                    report["throughput"].append(r)
                    print(f"  {r['endpoint']:<24} c={c:>2} "
                          f"{r['reqs_per_s']:>8.1f} reqs/s "
                          f"{r['units_per_s']:>8.1f} {r['unit']}")

        if "stress" in phases:
            print(f"\n=== stress ({duration_s:.0f}s soak) ===")
            report["stress"] = []
            for ep in H.ENDPOINTS:
                r = await H.measure_stress(session, base_url, ep, duration_s=duration_s)
                report["stress"].append(r)
                print(f"  {r['endpoint']:<24} c={r['concurrency']:>2} "
                      f"{r['reqs_per_s']:>8.1f} reqs/s p99={r['p99_ms']:.0f}ms "
                      f"err={r['error_rate']*100:.2f}%")

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"))
    ap.add_argument("--phases", default="latency,throughput,stress",
                    help="comma-separated subset of latency,throughput,stress")
    ap.add_argument("--quick", action="store_true",
                    help="cut iterations 4x and stress duration to 10s")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--out-json", default=str(HERE / "LATEST.json"))
    ap.add_argument("--out-md", default=str(HERE / "LATEST.md"))
    args = ap.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    duration_s = 10.0 if args.quick else args.duration

    report = asyncio.run(run(args.server_url, phases, args.quick, duration_s))

    H.emit_json(report, Path(args.out_json))
    H.emit_markdown(report, Path(args.out_md))
    print(f"\nwrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
