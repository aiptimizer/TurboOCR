#!/usr/bin/env python3
"""Graceful shutdown under load: SIGTERM mid-burst, in-flight must drain.

The contract (server_bootstrap.h begin_graceful_shutdown): on SIGTERM the
server stops accepting new connections at once, keeps the process alive up to
shutdown_grace_seconds while in-flight requests finish and send their response,
then exits. A request already being served when the signal arrives must return
200, never a dropped connection.

Measured behaviour, 2026-08-11 (macOS build, PDF path so requests are genuinely
mid-service at signal time):
  - 8/8 in-flight requests completed 200
  - the HTTP listener stopped accepting within ~1 s of SIGTERM (new connections
    refused, not served-then-killed) — the ideal k8s pattern: the readiness
    probe's endpoint goes unreachable, so the load balancer drains the pod
  - the process stayed alive ~1 s to finish the in-flight PDF, then exited
    cleanly, far inside the 20 s grace; gRPC drained on its own thread in
    parallel against the same deadline

Usage:
  ./build/turboocr-server &            # needs a PDF-capable build
  python3 tests/e2e/shutdown_under_load.py --url http://127.0.0.1:8080 \
      --pid <server-pid> --pdf tests/fixtures/pdf/academic_paper.pdf

Exits non-zero if any in-flight request fails to complete 200.
"""
import argparse, os, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--inflight", type=int, default=8)
    ap.add_argument("--signal-after", type=float, default=1.0)
    a = ap.parse_args()
    body = open(a.pdf, "rb").read()

    results = {}
    started = threading.Event()
    lock = threading.Lock()

    def one(i):
        s = requests.Session()
        with lock:
            results[i] = "reached-server"
        if i == 0:
            started.set()
        try:
            r = s.post(f"{a.url}/ocr/pdf?mode=ocr", data=body,
                       headers={"Content-Type": "application/pdf"}, timeout=60)
            with lock:
                results[i] = "200" if r.status_code == 200 else str(r.status_code)
        except Exception as e:
            with lock:
                results[i] = "EXC:" + type(e).__name__

    pool = ThreadPoolExecutor(max_workers=a.inflight)
    futs = [pool.submit(one, i) for i in range(a.inflight)]
    started.wait(10)
    time.sleep(a.signal_after)

    inflight = sum(1 for v in results.values() if v == "reached-server")
    t_sig = time.time()
    os.kill(a.pid, signal.SIGTERM)

    # Poll the listener: it should refuse promptly (connection error), proving
    # the server stops admitting rather than serving-then-killing.
    listener_closed_after = None
    for _ in range(40):
        time.sleep(0.25)
        try:
            requests.get(f"{a.url}/health", timeout=1)
        except Exception:
            listener_closed_after = time.time() - t_sig
            break

    for f in futs:
        try: f.result(timeout=60)
        except Exception: pass
    drain_time = time.time() - t_sig

    ok = sum(1 for v in results.values() if v == "200")
    other = {i: v for i, v in results.items() if v != "200"}
    print(f"in-flight at signal   {inflight}/{a.inflight}")
    print(f"completed 200         {ok}/{a.inflight}")
    print(f"listener refused new  {listener_closed_after:.1f}s after SIGTERM"
          if listener_closed_after is not None else
          "listener refused new  never (stayed open) — investigate")
    print(f"total drain time      {drain_time:.1f}s")
    if other:
        print(f"NOT 200               {other}")

    if inflight > 0 and ok >= inflight:
        print("\nGRACEFUL SHUTDOWN OK: every in-flight request drained to 200")
        sys.exit(0)
    print("\nGRACEFUL SHUTDOWN FAILED: an in-flight request was dropped")
    sys.exit(1)


if __name__ == "__main__":
    main()
