"""Latency benchmark: p50/p95/p99 per endpoint at c=1."""

import argparse
import asyncio
import json
import os

import aiohttp

from _harness_import import _harness_mod as H


async def main(base_url: str, n: int, warmup: int):
    async with aiohttp.ClientSession() as session:
        rows = []
        print(f"{'endpoint':<24} {'p50':>9} {'p95':>9} {'p99':>9} {'avg':>9} {'err':>4}")
        print("-" * 68)
        for ep in H.ENDPOINTS:
            r = await H.measure_latency(session, base_url, ep, n=n, warmup=warmup)
            rows.append(r)
            print(f"{r['endpoint']:<24} "
                  f"{r['p50_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms {r['p99_ms']:>7.1f}ms "
                  f"{r['avg_ms']:>7.1f}ms {r['err']:>4}")
        return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = asyncio.run(main(args.server_url, args.n, args.warmup))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
