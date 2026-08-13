#!/usr/bin/env python3
"""Sustained-concurrency soak with resource tracking.

Short benchmarks measure speed; this measures whether the process is still
healthy after being hammered. Throughput and latency come out of it, but the
point is the three resources a benchmark never looks at: RSS after the load
stops, file descriptors, and temp directories. The PDF path matters most — it
exercises the renderer's make_temp_dir/cleanup pair, where a leak fills the disk
instead of moving the latency.

    ./build/turboocr-server &
    python3 tests/e2e/soak.py --pid <server-pid> --url http://127.0.0.1:8080 \
        --image tests/fixtures/images/png/mixed_fonts.png \
        --pdf tests/fixtures/pdf/academic_paper.pdf --seconds 60 --concurrency 24

Exits non-zero on any error response, an fd delta over 16, a temp-dir delta over
2, or more than 256 MB of RSS still held 8 s after the load stops — so it works
as a gate, not only as a report.

--pid must be the SERVER, not a shell that launched it: `pgrep -f` will happily
match the wrapper, and tracking the wrapper is how a run reports a clean result
while measuring nothing. `pgrep -x turboocr-server` is the safe form.

A 60 s window catches per-request leaks (fds, temp dirs, handles) because those
scale with request count — several thousand requests is plenty. It does NOT
catch a slow leak of a few bytes per request; that needs hours — or, on macOS,
`leaks <pid>` run twice with load in between: identical leaked-byte counts mean
the leaks are one-time init, growth means per-request. Measured 2026-08-11:
26,144 bytes before and after doubling the request count — init-only.

KNOWN LIMIT — this harness is CLOSED-loop: every worker waits for a response
before sending again, so under saturation it self-throttles. Its latency
numbers are therefore queue time at absorbed rate (coordinated omission), and
it cannot measure overload behaviour at all. Use it for the resource gates it
exists for; for latency and overload use tests/e2e/load_openloop.sh, which
drives a constant arrival rate. The difference is not academic: on the same
server and image, this harness reported p50 306 ms at saturation while the
open-loop run showed 33 ms service time below capacity.
"""
import argparse, os, statistics, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests


def rss_kb(pid):
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                         capture_output=True, text=True, timeout=5).stdout.strip()
    if not out:
        # Reporting -1 here is how the first Linux run produced "RSS -0 MB,
        # delta +0" and looked like a clean result while measuring nothing at
        # all. A monitor that cannot read its target must say so.
        raise SystemExit(f"soak: cannot read RSS for pid {pid} — is it the right process?")
    return int(out.split()[0])


def fd_count(pid):
    # /proc is exact on Linux; lsof is the portable fallback (macOS).
    proc_fd = f"/proc/{pid}/fd"
    if os.path.isdir(proc_fd):
        return len(os.listdir(proc_fd))
    out = subprocess.run(["lsof", "-p", str(pid)], capture_output=True,
                         text=True, timeout=30)
    if out.returncode != 0 and not out.stdout:
        raise SystemExit(f"soak: cannot count fds for pid {pid}")
    return max(0, len(out.stdout.splitlines()) - 1)


def tmpdirs(prefix="turbo_pdf_"):
    base = os.environ.get("TMPDIR", "/tmp")
    try:
        return sum(1 for n in os.listdir(base) if n.startswith(prefix))
    except Exception:
        return -1


def soak(url, payload, ctype, pid, seconds, concurrency, label):
    stop = threading.Event()
    lat, errors, done = [], [], 0
    lock = threading.Lock()
    samples = []

    def sampler():
        while not stop.is_set():
            samples.append((time.time(), rss_kb(pid), tmpdirs()))
            time.sleep(2)

    def worker():
        nonlocal done
        s = requests.Session()
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                r = s.post(url, data=payload, headers={"Content-Type": ctype}, timeout=120)
                dt = (time.perf_counter() - t0) * 1000
                with lock:
                    if r.status_code != 200:
                        errors.append(r.status_code)
                    else:
                        lat.append(dt); done += 1
            except Exception as e:
                with lock:
                    errors.append(type(e).__name__)

    fd0, rss0, td0 = fd_count(pid), rss_kb(pid), tmpdirs()
    th = threading.Thread(target=sampler, daemon=True); th.start()
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker) for _ in range(concurrency)]
        time.sleep(seconds)
        stop.set()
        for f in futs: f.result()
    elapsed = time.time() - t_start
    th.join(timeout=3)

    # Settle: a real leak does not come back; caches and pools do.
    time.sleep(8)
    fd1, rss1, td1 = fd_count(pid), rss_kb(pid), tmpdirs()
    peak = max((s[1] for s in samples), default=rss1)

    lat.sort()
    def pct(p): return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else float("nan")
    print(f"\n=== {label} — {concurrency} concurrent, {elapsed:.0f}s ===")
    print(f"  completed        {done}   ({done/elapsed:.1f} req/s)")
    print(f"  errors           {len(errors)}" + (f"  {set(errors)}" if errors else ""))
    print(f"  latency ms       p50 {pct(.50):.0f}   p95 {pct(.95):.0f}   p99 {pct(.99):.0f}   max {lat[-1]:.0f}" if lat else "  latency          n/a")
    print(f"  RSS MB           start {rss0/1024:.0f}   peak {peak/1024:.0f}   after-settle {rss1/1024:.0f}   delta {(rss1-rss0)/1024:+.0f}")
    print(f"  file descriptors start {fd0}   end {fd1}   delta {fd1-fd0:+d}")
    print(f"  temp dirs        start {td0}   end {td1}   delta {td1-td0:+d}")
    # RSS trend across the load window: a leak climbs monotonically.
    if len(samples) >= 4:
        half = len(samples) // 2
        first = statistics.mean(s[1] for s in samples[:half])
        second = statistics.mean(s[1] for s in samples[half:])
        print(f"  RSS trend        first half {first/1024:.0f} MB -> second half {second/1024:.0f} MB "
              f"({(second-first)/1024:+.1f} MB)")
    return {"done": done, "errors": len(errors), "rss_delta_mb": (rss1 - rss0) / 1024,
            "fd_delta": fd1 - fd0, "td_delta": td1 - td0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18099")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--pdf")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()

    img = open(a.image, "rb").read()
    r1 = soak(f"{a.url}/ocr/raw", img, "image/png", a.pid, a.seconds, a.concurrency, "IMAGE /ocr/raw")
    r2 = None
    if a.pdf:
        pdf = open(a.pdf, "rb").read()
        r2 = soak(f"{a.url}/ocr/pdf?mode=ocr", pdf, "application/pdf", a.pid,
                  a.seconds, max(4, a.concurrency // 4), "PDF /ocr/pdf?mode=ocr (rasterizing)")

    bad = []
    for name, r in (("image", r1), ("pdf", r2)):
        if not r: continue
        if r["errors"]: bad.append(f"{name}: {r['errors']} errors")
        if r["fd_delta"] > 16: bad.append(f"{name}: fd leak {r['fd_delta']:+d}")
        if r["td_delta"] > 2: bad.append(f"{name}: temp-dir leak {r['td_delta']:+d}")
        if r["rss_delta_mb"] > 256: bad.append(f"{name}: RSS +{r['rss_delta_mb']:.0f} MB after settle")
    print("\n" + ("SOAK FAILED: " + "; ".join(bad) if bad else "SOAK CLEAN: no errors, no fd/temp-dir leak, RSS settled"))
    sys.exit(1 if bad else 0)
