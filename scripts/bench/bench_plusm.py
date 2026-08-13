#!/usr/bin/env python3
"""PP-FormulaNet_plus-M throughput harness (crops/s).

Measures formula-recognition throughput on a directory of pre-cropped formula
images, with length-bucketed batching (sort by crop area so each batch's
auto-regressive decode isn't stalled by one long formula in a random batch).

This is the GATE-2 reference: it reports the PADDLE-inference crops/s baseline
that the C++ prototype must meet/beat. The natural OmniDocBench isolated-formula
distribution is long-tailed (median ~102 char, max ~800), so static batching
plateaus ~27/s; continuous (iteration-level) batching in the C++ host loop is
the lever to clear the 59 crops/s bar -- this script measures the static-batch
paddle reference only.

Usage (paddle venv, GPU; gate the GPU run with the shared lockfile):
  .venv-paddle/bin/python scripts/bench/bench_plusm.py --crops <dir> [--bs 32 48] \
      [--model-dir models/formula/ppformulanet_plus_m] [--order sorted|random]
"""
from __future__ import annotations
import argparse, glob, os, time
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True, help="dir of formula crop images")
    ap.add_argument("--model-dir", default=None,
                    help="local plus-M paddle model dir (else cached PP-FormulaNet_plus-M)")
    ap.add_argument("--bs", type=int, nargs="+", default=[16, 32, 48])
    ap.add_argument("--order", choices=["sorted", "random", "both"], default="both")
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.crops, "*.png")) +
                   glob.glob(os.path.join(args.crops, "*.jpg")))
    if not paths:
        raise SystemExit(f"no crops in {args.crops}")
    area = {p: Image.open(p).size[0] * Image.open(p).size[1] for p in paths}

    from paddlex import create_model
    kw = dict(model_name="PP-FormulaNet_plus-M", device=args.device)
    if args.model_dir:
        kw["model_dir"] = args.model_dir
    m = create_model(**kw)
    list(m.predict(paths[:args.warmup], batch_size=min(args.warmup, len(paths))))

    def run(seq, bs):
        t = time.perf_counter()
        for s in range(0, len(seq), bs):
            list(m.predict(seq[s:s + bs], batch_size=bs))
        return len(seq) / (time.perf_counter() - t)

    import random
    rnd = paths[:]; random.seed(0); random.shuffle(rnd)
    srt = sorted(paths, key=lambda p: area[p])
    orders = {"random": rnd, "sorted": srt}
    pick = ["random", "sorted"] if args.order == "both" else [args.order]

    print(f"=== plus-M throughput (n={len(paths)}, paddle {args.device}) ===")
    for bs in args.bs:
        line = f"  bs={bs:3d}:"
        for o in pick:
            r = run(orders[o], bs)
            line += f"  {o} {r:6.1f}/s"
            if r >= 59:
                line += "*"
        print(line)
    print("  (* >= 59 crops/s bar; '*' on 'sorted' approximates length-bucketed static batching)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
