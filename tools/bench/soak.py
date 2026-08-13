#!/usr/bin/env python3
"""Sustained-load soak against a running turboocr-server.

Mixed fire for --minutes: 4 image clients rotating text / text+layout /
layout-only, 2 PDF clients, 1 NDJSON stream client asserting exactly-once page
events, and a VRAM sampler (nvidia-smi or rocm-smi, auto-detected). Prints one
JSON stats line; the pass criterion is every counter key ending in _200 /
stream_ok and nothing else (no _500, no stream_BAD), plus a stable vram peak.

This is the same harness the NVIDIA RTX 5090 bring-up passed on 2026-08-02
(41,566 image + 5,973 PDF requests, all 200), promoted into the tree so every
backend bring-up runs the identical soak instead of reinventing it.

Usage:
  python3 tools/bench/soak.py --base http://127.0.0.1:18860 \
      --images ~/funsd_cache --pdf /tmp/test8.pdf --pdf-pages 8 --minutes 5
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:18860")
ap.add_argument("--images", required=True, help="dir of .png test pages")
ap.add_argument("--pdf", required=True, help="scanned test PDF")
ap.add_argument("--pdf-pages", type=int, default=8,
                help="expected page count of --pdf (stream exactly-once check)")
ap.add_argument("--minutes", type=float, default=5)
ap.add_argument("--img-clients", type=int, default=4)
ap.add_argument("--pdf-clients", type=int, default=2)
args = ap.parse_args()

IMGDIR = os.path.expanduser(args.images)
IMGS = [os.path.join(IMGDIR, f) for f in sorted(os.listdir(IMGDIR))[:20]
        if f.endswith(".png")]
assert IMGS, f"no .png files under {IMGDIR}"
END = time.time() + args.minutes * 60
stats = collections.Counter()
lock = threading.Lock()


def curl(cargs, timeout=120):
    return subprocess.run(
        ["curl", "-s", "-m", str(timeout), "-o", "/dev/null", "-w",
         "%{http_code}"] + cargs,
        capture_output=True, text=True).stdout


def img_client(i):
    n = 0
    while time.time() < END:
        img = IMGS[n % len(IMGS)]
        n += 1
        q = ["", "?layout=1", "?layout=1&text=0"][n % 3]
        code = curl(["--data-binary", "@" + img, "-H",
                     "Content-Type: application/octet-stream",
                     args.base + "/ocr/raw" + q])
        with lock:
            stats["img_" + code] += 1


def pdf_client(i):
    while time.time() < END:
        code = curl(["--data-binary", "@" + args.pdf, "-H",
                     "Content-Type: application/pdf",
                     args.base + "/ocr/pdf?layout=1"], 180)
        with lock:
            stats["pdf_" + code] += 1


def stream_client():
    while time.time() < END:
        out = subprocess.run(
            ["curl", "-s", "-N", "-m", "180", "--data-binary", "@" + args.pdf,
             "-H", "Content-Type: application/pdf", args.base + "/ocr/stream"],
            capture_output=True, text=True).stdout
        pages = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") == "page":
                pages.append(d.get("page_index"))
        ok = sorted(pages) == list(range(args.pdf_pages))
        with lock:
            stats["stream_ok" if ok else "stream_BAD"] += 1


def read_vram_mb():
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        return int(out.splitlines()[0])
    if shutil.which("rocm-smi"):
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True).stdout
        d = json.loads(out)
        card = d[sorted(d)[0]]
        used = next(v for k, v in card.items() if "Used" in k)
        return int(used) // (1024 * 1024)
    return None


def vram_sampler():
    peak = 0
    while time.time() < END:
        try:
            v = read_vram_mb()
            if v is not None:
                peak = max(peak, v)
        except Exception:
            pass
        time.sleep(5)
    with lock:
        stats["vram_peak_mb"] = peak


threads = [threading.Thread(target=img_client, args=(i,))
           for i in range(args.img_clients)]
threads += [threading.Thread(target=pdf_client, args=(i,))
            for i in range(args.pdf_clients)]
threads += [threading.Thread(target=stream_client),
            threading.Thread(target=vram_sampler)]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
stats["elapsed_s"] = int(time.time() - t0)
print(json.dumps(dict(stats)))
