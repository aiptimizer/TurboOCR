"""Throughput benchmark: reqs/s and units/s per endpoint × concurrency."""

import argparse
import asyncio
import json
import os

import aiohttp

from _harness_import import _harness_mod as H

CONCURRENCIES = [1, 4, 16, 32]


async def main(base_url: str, n: int):
    async with aiohttp.ClientSession() as session:
        rows = []
        print(f"{'endpoint':<24} {'c':>3} {'unit':<9} {'reqs/s':>9} {'units/s':>10} {'err':>4}")
        print("-" * 68)
        for ep in H.ENDPOINTS:
            for c in CONCURRENCIES:
                r = await H.measure_throughput(session, base_url, ep, concurrency=c, n=n)
                rows.append(r)
                print(f"{r['endpoint']:<24} {c:>3} {r['unit']:<9} "
                      f"{r['reqs_per_s']:>9.1f} {r['units_per_s']:>10.1f} {r['err']:>4}")
        return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = asyncio.run(main(args.server_url, args.n))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
