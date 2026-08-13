#!/usr/bin/env python3
"""Score a FUNSD prediction JSON (list of per-image word lists) against a
ground-truth word list, using the SAME Counter-based bag-of-words F1 as
tests/benchmark/scoring/bench_funsd_local.py.

This is the GATE OF RECORD for OCR accuracy in this repo: tests/cpp/backends/turbo_bench
computes the same metric in-process (so it can never print a throughput number
without its accuracy), but ctest asserts on THIS script.

Usage
-----
    # backward-compatible positional form (unchanged)
    score_funsd.py preds.json

    # explicit paths — no hardcoded repo root any more
    score_funsd.py preds.json --gt tests/benchmark/funsd_gt_words.json

    # as a CI/ctest gate: nonzero exit below the floor
    score_funsd.py preds.json --assert-f1 85.7

    # assert accuracy AND throughput from one turbo_bench metrics JSON
    score_funsd.py preds.json --metrics run.json --assert-f1 85.2 --assert-throughput 90

Exit codes: 0 pass, 1 gate failed, 2 bad input.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Repo root inferred from THIS file's location (tools/bench/score_funsd.py),
# never hardcoded: the old absolute /Users/... path made the script unusable on
# the NVIDIA box the numbers are supposed to be compared against.
REPO = Path(__file__).resolve().parents[2]
DEFAULT_GT = REPO / "tests/benchmark/funsd_gt_words.json"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tok(text):
    return _TOKEN_RE.findall(text.lower()) if text else []


def word_f1(gt, pred):
    if not gt and not pred:
        return dict(recall=1.0, precision=1.0, f1=1.0)
    if not gt or not pred:
        return dict(recall=0.0, precision=0.0, f1=0.0)
    gb, pb = Counter(gt), Counter(pred)
    tp = sum((gb & pb).values())
    r = tp / sum(gb.values())
    p = tp / sum(pb.values())
    f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return dict(recall=r, precision=p, f1=f1)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", help="prediction JSON: list of per-image word lists")
    ap.add_argument("gt_pos", nargs="?", default=None,
                    help="ground-truth JSON (optional 2nd positional; same as --gt)")
    ap.add_argument("--gt", default=None,
                    help=f"ground-truth JSON (default: {DEFAULT_GT})")
    ap.add_argument("--metrics", default=None,
                    help="turbo_bench metrics JSON; enables --assert-throughput and "
                         "prints the run's provenance so two machines' numbers are "
                         "comparable")
    ap.add_argument("--assert-f1", type=float, default=None,
                    help="minimum mean F1 in PERCENT; exit 1 below it")
    ap.add_argument("--assert-throughput", type=float, default=None,
                    help="minimum img/s from --metrics; exit 1 below it")
    ap.add_argument("--worst", type=int, default=5, help="how many worst pages to list")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    gt_path = Path(args.gt or args.gt_pos or DEFAULT_GT)
    pred_path = Path(args.preds)
    for p, what in ((pred_path, "predictions"), (gt_path, "ground truth")):
        if not p.exists():
            print(f"error: {what} file not found: {p}", file=sys.stderr)
            return 2

    preds = json.loads(pred_path.read_text())   # list[list[str]] recognized text items
    gt = json.loads(gt_path.read_text())        # list[list[str]] raw GT words
    n = min(len(preds), len(gt))
    if n == 0:
        print("error: nothing to score (empty predictions or GT)", file=sys.stderr)
        return 2

    accs = []
    for i in range(n):
        accs.append(word_f1(tok(" ".join(gt[i])), tok(" ".join(preds[i]))))
    f1 = mean([a["f1"] for a in accs])
    p = mean([a["precision"] for a in accs])
    r = mean([a["recall"] for a in accs])

    print(f"{pred_path}")
    print(f"  gt={gt_path}")
    print(f"  pages={n}  F1={f1:.2%}  P={p:.2%}  R={r:.2%}")
    if args.worst and not args.quiet:
        order = sorted(range(n), key=lambda i: accs[i]["f1"])[: args.worst]
        print("  worst pages:", [(i, round(accs[i]["f1"], 2),
                                  f"pred={len(tok(' '.join(preds[i])))}w "
                                  f"gt={len(tok(' '.join(gt[i])))}w")
                                 for i in order])

    metrics = None
    if args.metrics:
        mp = Path(args.metrics)
        if not mp.exists():
            print(f"error: metrics file not found: {mp}", file=sys.stderr)
            return 2
        metrics = json.loads(mp.read_text())
        prov = metrics.get("provenance", {})
        thr = metrics.get("throughput", {})
        # PROVENANCE IS THE POINT: this is what makes a run on a GPU box
        # comparable to a run measured on the M3 Max.
        print(f"  run: backend={prov.get('backend')} device={prov.get('device')} "
              f"host={prov.get('hostname')} chip={prov.get('chip')}")
        print(f"       threads={prov.get('threads')} repeat={prov.get('repeat')} "
              f"tier={prov.get('tier')} images_sha={str(prov.get('images_sha256'))[:16]}")
        for name, m in (prov.get("models") or {}).items():
            print(f"       model {name} sha256={m.get('sha256', '')[:16]} "
                  f"bytes={m.get('bytes')}")
        print(f"       throughput={thr.get('img_per_s', 0):.1f} img/s over "
              f"{thr.get('window_s', 0):.1f}s  window_ok={thr.get('window_long_enough')} "
              f"wall_clock_ok={thr.get('wall_clock_agrees')}")
        # An in-process F1 that disagrees with this scorer means one of the two
        # metrics drifted — always a bug, never a rounding difference.
        acc = metrics.get("accuracy") or {}
        if acc.get("scored"):
            delta = abs(acc.get("f1", 0.0) * 100 - f1 * 100)
            print(f"       turbo_bench in-process F1={acc.get('f1', 0.0):.2%} "
                  f"(delta vs this scorer: {delta:.4f}pt)")
            if delta > 0.01:
                print("error: in-process F1 and score_funsd.py DISAGREE — the two "
                      "implementations of the metric have drifted", file=sys.stderr)
                return 1

    rc = 0
    if args.assert_f1 is not None:
        if f1 * 100 < args.assert_f1:
            print(f"GATE FAILED: F1 {f1:.2%} < required {args.assert_f1:.2f}%",
                  file=sys.stderr)
            rc = 1
        else:
            print(f"GATE OK: F1 {f1:.2%} >= {args.assert_f1:.2f}%")
    if args.assert_throughput is not None:
        if metrics is None:
            print("error: --assert-throughput needs --metrics <turbo_bench json>",
                  file=sys.stderr)
            return 2
        thr = (metrics.get("throughput") or {})
        got = thr.get("img_per_s", 0.0)
        # A rate the harness itself flagged as untrustworthy must never pass a
        # gate — this is discipline rules 1 and 2 (>=15s window, wall-clock
        # cross-check) surviving all the way out to CI.
        if not thr.get("window_long_enough", False) or not thr.get("wall_clock_agrees", False):
            print(f"GATE FAILED: turbo_bench rejected its own measurement "
                  f"(window_ok={thr.get('window_long_enough')}, "
                  f"wall_clock_ok={thr.get('wall_clock_agrees')}); "
                  f"{got:.1f} img/s is not a publishable number", file=sys.stderr)
            rc = 1
        elif got < args.assert_throughput:
            print(f"GATE FAILED: {got:.1f} img/s < required {args.assert_throughput:.1f}",
                  file=sys.stderr)
            rc = 1
        else:
            print(f"GATE OK: {got:.1f} img/s >= {args.assert_throughput:.1f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
