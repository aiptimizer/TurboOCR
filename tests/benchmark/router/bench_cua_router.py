"""CUA-router benchmark orchestrator.

Runs three scenarios (text_only, formula_heavy, table_heavy) sequentially at
c=1, applies the rolling-baseline regression detector, and writes a JSON
report to --out. Exit code per plan 06 §6: 0=PASS, 1=INFRA, 3=ALERT, 4=HALT.

Invoked by scripts/bench/bench_cua_loop.sh.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

import _cua_scenarios as S
import _regression as R


# ---------------------------------------------------------------------------
# Per-fixture measurement
# ---------------------------------------------------------------------------

async def _measure_fixture(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    fx: S.FixtureSpec,
    n: int,
    warmup: int,
    sample_timing: bool,
) -> dict[str, Any]:
    body = fx.path.read_bytes()
    url = base_url.rstrip("/") + endpoint
    sender = S.send_pdf if fx.kind == "pdf" else S.send_image

    # warmup
    for _ in range(warmup):
        await sender(session, url, fx.content_type, body, timing=False)

    latencies: list[float] = []
    per_request_pages: list[int] = []
    ok = err = 0
    timings_first: dict | None = None
    for i in range(n):
        do_timing = sample_timing and i == 0
        t0 = time.perf_counter()
        status, pages, timings = await sender(session, url, fx.content_type, body, timing=do_timing)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if status == 200:
            ok += 1
            latencies.append(dt_ms)
            per_request_pages.append(max(pages, 1))
            if timings and timings_first is None:
                timings_first = timings
        else:
            err += 1

    # for PDF endpoints we report per-page latency.
    if fx.kind == "pdf":
        latencies = [
            lat / pages for lat, pages in zip(latencies, per_request_pages) if pages > 0
        ]

    if not latencies:
        return {"name": fx.path.name, "ok": ok, "err": err,
                "p50_ms": None, "p99_ms": None, "timings": None}

    latencies.sort()
    return {
        "name": fx.path.name,
        "ok": ok,
        "err": err,
        "p50_ms": latencies[int(len(latencies) * 0.50)],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)],
        "n": len(latencies),
        "timings": timings_first,
    }


async def _run_scenario(
    session: aiohttp.ClientSession,
    base_url: str,
    scn: S.Scenario,
    fixtures: list[S.FixtureSpec],
) -> dict[str, Any]:
    per_fixture: list[dict[str, Any]] = []
    all_p50s: list[float] = []
    all_p99s: list[float] = []
    total_ok = total_err = 0
    stage_breakdown: dict[str, float] | None = None
    fixture_hashes: dict[str, str] = {}

    for i, fx in enumerate(fixtures):
        endpoint = scn.endpoint
        # table_heavy uses /ocr/raw for image fixtures
        if scn.name == "table_heavy" and fx.kind == "image":
            endpoint = "/ocr/raw"

        sample_timing = (i == 0)  # one X-Turbo-Timing probe per scenario
        r = await _measure_fixture(
            session, base_url, endpoint, fx,
            n=scn.n_per_fixture, warmup=scn.warmup,
            sample_timing=sample_timing,
        )
        per_fixture.append({k: v for k, v in r.items() if k != "timings"})
        total_ok += r["ok"]
        total_err += r["err"]
        if r["p50_ms"] is not None:
            all_p50s.append(r["p50_ms"])
            all_p99s.append(r["p99_ms"])
        if stage_breakdown is None and r.get("timings"):
            stage_breakdown = r["timings"] if isinstance(r["timings"], dict) else None
        try:
            fixture_hashes[fx.path.name] = S.sha256_file(fx.path)
        except Exception:
            pass

    n_total = total_ok + total_err
    err_rate = (total_err / n_total) if n_total else 1.0
    return {
        "fixtures": [fx.path.name for fx in fixtures],
        "n_per_fixture": scn.n_per_fixture,
        "warmup": scn.warmup,
        "ok": total_ok,
        "err": total_err,
        "err_rate": err_rate,
        "p50_ms": statistics.median(all_p50s) if all_p50s else None,
        "p99_ms": max(all_p99s) if all_p99s else None,
        "per_fixture": per_fixture,
        "stage_breakdown_ms": stage_breakdown or {},
        "fixture_hashes": fixture_hashes,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out
    except Exception:
        return "unknown"


def _loadavg_1m() -> float:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return 0.0


async def main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"
    started = time.perf_counter()

    base_url = args.server_url.rstrip("/")
    scenarios = S.build_scenarios()

    # health probe (in addition to shell's): record the result, don't bail
    server_up = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url + "/health/ready",
                                   timeout=aiohttp.ClientTimeout(total=2.0)) as r:
                server_up = (r.status == 200)
    except Exception:
        server_up = False

    if not server_up:
        report = _stub_report(ts, base_url, started, reason="health/ready not OK")
        R.write_report(out_path, report)
        R.update_latest_symlink(out_dir, out_path)
        R.write_summary_line(out_dir, _summary_line(report))
        print(_summary_line(report))
        return R.EXIT_SERVER_DOWN

    async with aiohttp.ClientSession() as session:
        # text-only self-validation
        kept_text, warns = await S.validate_text_fixtures(
            session, base_url, scenarios[0].fixtures
        )
        if warns:
            for w in warns:
                print(f"[validate] {w}", file=sys.stderr)
        scenarios[0].fixtures = kept_text

        scenario_results: dict[str, dict[str, Any]] = {}
        for scn in scenarios:
            if not scn.fixtures:
                scenario_results[scn.name] = {
                    "fixtures": [], "n_per_fixture": scn.n_per_fixture,
                    "warmup": scn.warmup, "ok": 0, "err": 0, "err_rate": 1.0,
                    "p50_ms": None, "p99_ms": None, "per_fixture": [],
                    "stage_breakdown_ms": {}, "fixture_hashes": {},
                }
                continue
            scenario_results[scn.name] = await _run_scenario(
                session, base_url, scn, scn.fixtures
            )

    # build verdict
    baseline = R.load_baseline(out_dir)
    scn_results = [
        R.ScenarioResult(
            name=k,
            p50_ms=v["p50_ms"],
            p99_ms=v["p99_ms"],
            err_rate=v["err_rate"],
        )
        for k, v in scenario_results.items()
    ]
    verdict = R.compute_verdicts(scn_results, baseline)

    # persist PASS samples to rolling baseline
    text_p50 = scenario_results.get("text_only", {}).get("p50_ms")
    for sr in scn_results:
        sv = verdict.get(sr.name, "INFRA")
        if sv != "PASS":
            continue
        if sr.name == "text_only":
            R.append_pass(baseline, "text_only", sr.p50_ms, ts)
        elif sr.p50_ms is not None and text_p50 is not None:
            R.append_pass(baseline, sr.name, sr.p50_ms, ts,
                          delta_ms=sr.p50_ms - text_p50)
    R.save_baseline(out_dir, baseline)

    # baseline summary block
    text_baseline_p50 = R.baseline_p50(baseline, "text_only")
    text_entries = baseline.get("scenarios", {}).get("text_only", [])
    baseline_summary = {
        "text_only_p50_ms": text_baseline_p50,
        "samples_used": min(len(text_entries), R.BASELINE_WINDOW),
        "window": [e["timestamp"] for e in text_entries[-R.BASELINE_WINDOW:]],
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": _to_iso(ts),
        "git_sha": _git_sha(),
        "server_url": base_url,
        "server_up": True,
        "duration_s": round(time.perf_counter() - started, 2),
        "loadavg_1m": _loadavg_1m(),
        "scenarios": scenario_results,
        "baseline": baseline_summary,
        "verdict": verdict,
    }

    # downgrade verdict to INFRA under high load if not already worse
    if report["loadavg_1m"] > 2.0 and verdict["overall"] in ("ALERT",):
        verdict["overall"] = "INFRA"
        verdict["reasons"].append(f"loadavg_1m {report['loadavg_1m']:.2f} > 2.0; downgraded")

    R.write_report(out_path, report)
    R.update_latest_symlink(out_dir, out_path)
    line = _summary_line(report)
    R.write_summary_line(out_dir, line)
    print(line)

    return R.exit_code_for(verdict["overall"])


def _stub_report(ts: str, base_url: str, started_perf: float, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": _to_iso(ts),
        "git_sha": _git_sha(),
        "server_url": base_url,
        "server_up": False,
        "duration_s": round(time.perf_counter() - started_perf, 2),
        "loadavg_1m": _loadavg_1m(),
        "scenarios": {},
        "baseline": {},
        "verdict": {
            "overall": "INFRA",
            "text_only": "INFRA",
            "formula_heavy": "INFRA",
            "table_heavy": "INFRA",
            "consecutive_alerts": {},
            "reasons": [reason],
        },
    }


def _to_iso(ts: str) -> str:
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}Z"


def _summary_line(report: dict[str, Any]) -> str:
    v = report["verdict"]["overall"]
    if not report.get("server_up"):
        return f"{report['timestamp_utc']} {v}  server_up:false  dur={report['duration_s']}s"
    s = report["scenarios"]
    text = s.get("text_only", {}).get("p50_ms")
    form = s.get("formula_heavy", {}).get("p50_ms")
    tab = s.get("table_heavy", {}).get("p50_ms")
    B = report.get("baseline", {}).get("text_only_p50_ms")
    text_s = f"{text:.1f}ms" if text is not None else "n/a"
    if text is not None and B:
        text_s += f" (B={B:.1f}, {(text/B - 1)*100:+.1f}%)"
    elif text is not None:
        text_s += " (no-B)"
    form_s = (f"{form:.0f}ms (Δ+{form - text:.0f})"
              if (form is not None and text is not None) else "n/a")
    tab_s = (f"{tab:.0f}ms (Δ+{tab - text:.0f})"
             if (tab is not None and text is not None) else "n/a")
    return (f"{report['timestamp_utc']} {v}  text={text_s}  "
            f"form={form_s}  tab={tab_s}  dur={report['duration_s']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url",
                    default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"))
    ap.add_argument("--out", dest="out_path", default=None,
                    help="explicit JSON output path (overrides --out-dir/<ts>.json)")
    ap.add_argument("--out-dir", default="/tmp/cua_bench")
    args = ap.parse_args()
    if args.out_path:
        # caller wants a specific file; place it in out-dir for symlink consistency
        p = Path(args.out_path)
        args.out_dir = str(p.parent)
    sys.exit(asyncio.run(main(args)))
