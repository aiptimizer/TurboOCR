#!/usr/bin/env python3
"""Analyze /tmp/vlm_profile.jsonl produced by VLM_PROFILE=1.

Usage:
    python3 tools/bench/analyze_vlm_profile.py [path_to_jsonl]

Produces a per-section p50/p95 table and answers the decision-gate questions.
"""

import json
import sys
import statistics
from pathlib import Path


def pct(data, p):
    if not data:
        return 0
    s = sorted(data)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]


def fmt_ms(us):
    return f"{us/1000:.1f}ms"


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def analyze(recs, backend_filter=None):
    if backend_filter:
        recs = [r for r in recs if r.get("backend") == backend_filter]
    if not recs:
        return None

    metrics = {
        "n_crops":              [],
        "t_mutex_wait_us":      [],
        "t_d2h_us":             [],
        "t_png_total_us":       [],
        "t_b64_total_us":       [],
        "t_json_build_us":      [],
        "t_http_wall_us":       [],
        "t_response_parse_us":  [],
        "t_total_us":           [],
        "bytes_sent_total":     [],
        "bytes_recv_total":     [],
    }

    # Per-crop flat lists
    http_per_crop  = []
    parse_per_crop = []
    png_per_crop   = []

    for r in recs:
        for k in metrics:
            if k in r:
                metrics[k].append(r[k])
        if "t_http_per_crop_us" in r:
            http_per_crop.extend(r["t_http_per_crop_us"])
        if "t_parse_per_crop_us" in r:
            parse_per_crop.extend(r["t_parse_per_crop_us"])
        if "t_png_per_crop_us" in r:
            png_per_crop.extend(r["t_png_per_crop_us"])

    return {
        "n_pages":     len(recs),
        "metrics":     metrics,
        "http_per_crop":  http_per_crop,
        "parse_per_crop": parse_per_crop,
        "png_per_crop":   png_per_crop,
    }


def print_section_table(label, data, recs_list=None):
    if data is None:
        print(f"  [no {label} records found]")
        return

    m = data["metrics"]
    print(f"\n{'='*60}")
    print(f"  {label}  ({data['n_pages']} page-calls)")
    print(f"{'='*60}")
    print(f"  {'Section':<26} {'p50':>10} {'p95':>10} {'mean':>10}")
    print(f"  {'-'*56}")

    def row(name, vals):
        if not vals:
            return
        p50 = pct(vals, 50)
        p95 = pct(vals, 95)
        mn  = statistics.mean(vals)
        print(f"  {name:<26} {fmt_ms(p50):>10} {fmt_ms(p95):>10} {fmt_ms(mn):>10}")

    row("t_mutex_wait",     m["t_mutex_wait_us"])
    row("t_d2h",            m["t_d2h_us"])
    row("t_png_total",      m["t_png_total_us"])
    row("t_b64_total",      m["t_b64_total_us"])
    row("t_json_build",     m["t_json_build_us"])
    row("t_http_wall (sum)", m["t_http_wall_us"])
    row("t_response_parse", m["t_response_parse_us"])
    row("t_total",          m["t_total_us"])

    if data["png_per_crop"]:
        print()
        row("  png/crop",      data["png_per_crop"])
    if data["http_per_crop"]:
        row("  http/crop",     data["http_per_crop"])
    if data["parse_per_crop"]:
        row("  parse/crop",    data["parse_per_crop"])

    # bytes
    if m["bytes_sent_total"]:
        sent_p50 = pct(m["bytes_sent_total"], 50)
        recv_p50 = pct(m["bytes_recv_total"], 50)
        print(f"\n  bytes_sent p50: {sent_p50/1024:.1f} KB")
        print(f"  bytes_recv p50: {recv_p50/1024:.1f} KB")

    # Decision gate analysis
    totals = m["t_total_us"]
    http   = m["t_http_wall_us"]
    parses = m["t_response_parse_us"]
    # Note: t_http_wall is SUM of per-crop times (crops run in parallel).
    # The real server-side parallel wall = max(t_http_per_crop_us).
    # We compute cpp_side = t_total - max_http_per_crop (per page).
    source_recs = recs_list if recs_list is not None else []
    max_http_per_page = []
    for r in source_recs:
        hcrop = r.get("t_http_per_crop_us", [])
        if hcrop:
            max_http_per_page.append(max(hcrop) / 1000)
        elif "t_http_wall_us" in r:
            max_http_per_page.append(r["t_http_wall_us"] / 1000)
    # totals is in us, max_http_per_page in ms — convert totals to ms
    totals_ms = [v / 1000 for v in totals]
    if totals_ms and max_http_per_page:
        cpp_sides = []
        for i in range(min(len(totals_ms), len(max_http_per_page))):
            server_compute = max_http_per_page[i]   # ms
            cpp_side = totals_ms[i] - server_compute  # ms
            if totals_ms[i] > 0:
                cpp_sides.append(cpp_side / totals_ms[i] * 100)
        if cpp_sides:
            cpp_pct_p50 = pct(cpp_sides, 50)
            cpp_pct_p95 = pct(cpp_sides, 95)
            print(f"\n  C++-side fraction (excl server compute):")
            print(f"    p50 = {cpp_pct_p50:.1f}%   p95 = {cpp_pct_p95:.1f}%")
            if cpp_pct_p50 > 30:
                print("  GATE: C++ > 30% -> Option A (curl_multi + parallel PNG) WORTH IT")
            elif cpp_pct_p50 < 10:
                print("  GATE: C++ < 10% -> bottleneck is server-side; pivot to streaming tensors")
            else:
                print(f"  GATE: C++ {cpp_pct_p50:.0f}% (borderline — profile individual pages)")


def find_max_crop_page(recs):
    if not recs:
        return None
    max_r = max(recs, key=lambda r: r.get("n_crops", 0))
    return max_r


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vlm_profile.jsonl"
    if not Path(path).exists():
        print(f"Profile file not found: {path}")
        sys.exit(1)

    recs = load_records(path)
    print(f"\nLoaded {len(recs)} records from {path}")

    formula_recs = [r for r in recs if r.get("backend") == "formula"]
    table_recs   = [r for r in recs if r.get("backend") == "table"]

    print_section_table("FORMULA backend", analyze(formula_recs), formula_recs)
    print_section_table("TABLE backend",   analyze(table_recs), table_recs)

    # Find the heaviest page (most crops)
    for backend, bname in [(formula_recs, "formula"), (table_recs, "table")]:
        heavy = find_max_crop_page(backend)
        if heavy:
            n = heavy.get("n_crops", 0)
            tot = heavy.get("t_total_us", 0)
            d2h = heavy.get("t_d2h_us", 0)
            png = heavy.get("t_png_total_us", 0)
            b64 = heavy.get("t_b64_total_us", 0)
            jsn = heavy.get("t_json_build_us", 0)
            parse = heavy.get("t_response_parse_us", 0)
            mu = heavy.get("t_mutex_wait_us", 0)
            http_crops = heavy.get("t_http_per_crop_us", [])
            # real parallel server wall = max per-crop HTTP (not sum)
            max_http_us = max(http_crops) if http_crops else heavy.get("t_http_wall_us", 0)
            print(f"\n{'='*60}")
            print(f"  Heaviest {bname} page: {n} crops, total={fmt_ms(tot)}")
            print(f"{'='*60}")
            print(f"  mutex_wait  : {fmt_ms(mu)}")
            print(f"  d2h         : {fmt_ms(d2h)}")
            print(f"  png_total   : {fmt_ms(png)}")
            print(f"  b64_total   : {fmt_ms(b64)}")
            print(f"  json_build  : {fmt_ms(jsn)}")
            print(f"  http/crop max (parallel wall): {fmt_ms(max_http_us)}")
            print(f"  parse       : {fmt_ms(parse)}")
            server_side = max_http_us  # parallel HTTP wall (server compute dominates)
            cpp_side = tot - server_side
            if tot > 0:
                print(f"  C++-side    : {fmt_ms(cpp_side)} ({100*cpp_side/tot:.1f}%)")
                print(f"  server-side : {fmt_ms(server_side)} ({100*server_side/tot:.1f}%)")
            png_crops = heavy.get("t_png_per_crop_us", [])
            if png_crops:
                print(f"  png/crop avg: {fmt_ms(statistics.mean(png_crops))}")
            if http_crops:
                print(f"  http/crop avg: {fmt_ms(statistics.mean(http_crops))}")
                print(f"  http/crop max: {fmt_ms(max(http_crops))}")
                print(f"  http/crop min: {fmt_ms(min(http_crops))}")


if __name__ == "__main__":
    main()
