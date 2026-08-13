"""Stress benchmark: sustained 60s soak per endpoint, p99 + error rate.

This runs as a benchmark (no assertions). For assert-mode use the
tests/stress/ suite.
"""

import argparse
import asyncio
import json
import os

import aiohttp

from _harness_import import _harness_mod as H


async def main(base_url: str, duration_s: float, only: list[str] | None):
    async with aiohttp.ClientSession() as session:
        rows = []
        print(f"{'endpoint':<24} {'c':>3} {'reqs/s':>9} {'units/s':>10} {'p99 ms':>9} {'err%':>7}")
        print("-" * 70)
        for ep in H.ENDPOINTS:
            if only and ep.name not in only:
                continue
            r = await H.measure_stress(session, base_url, ep, duration_s=duration_s)
            rows.append(r)
            print(f"{r['endpoint']:<24} {r['concurrency']:>3} "
                  f"{r['reqs_per_s']:>9.1f} {r['units_per_s']:>10.1f} "
                  f"{r['p99_ms']:>8.1f}ms {r['error_rate']*100:>6.2f}%")
        return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"))
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = asyncio.run(main(args.server_url, args.duration, args.only))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
