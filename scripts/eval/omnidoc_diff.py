#!/usr/bin/env python3
"""Diff two omnidoc_run_and_score.py summary JSONs.

Edit-distance metrics: lower is better, so an *increase* is a regression.
TEDS / CDM: higher is better, so a *decrease* is a regression.

Exits non-zero if any metric regresses by more than --threshold (default
0.01, i.e. 1 percentage point on the relevant scale).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"

LOWER_IS_BETTER = {
    "text_block_edit_dist",
    "display_formula_edit_dist",
    "table_edit_dist",
    "reading_order_edit_dist",
    "text_block_edit",
    "display_formula_edit",
    "table_edit",
    "reading_order_edit",
}

HIGHER_IS_BETTER = {
    "display_formula_cdm",
    "table_teds",
    "table_teds_structure_only",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--threshold", type=float, default=0.01,
                   help="Absolute regression budget per metric.")
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def _direction(metric: str, delta: float) -> str:
    if metric in LOWER_IS_BETTER:
        regressed = delta > 0
    elif metric in HIGHER_IS_BETTER:
        regressed = delta < 0
    else:
        return "neutral"
    return "regression" if regressed else "improvement"


def _color(text: str, kind: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    if kind == "regression":
        return f"{ANSI_RED}{text}{ANSI_RESET}"
    if kind == "improvement":
        return f"{ANSI_GREEN}{text}{ANSI_RESET}"
    if kind == "dim":
        return f"{ANSI_DIM}{text}{ANSI_RESET}"
    if kind == "bold":
        return f"{ANSI_BOLD}{text}{ANSI_RESET}"
    return text


def _fmt_row(metric: str, base, cand, *, color_enabled: bool,
             threshold: float) -> tuple[str, bool]:
    if base is None or cand is None:
        return (f"  {metric:<32} {str(base):>10}  ->  {str(cand):>10}    n/a",
                False)
    delta = cand - base
    kind = _direction(metric, delta)
    over_budget = (
        (metric in LOWER_IS_BETTER and delta > threshold) or
        (metric in HIGHER_IS_BETTER and delta < -threshold)
    )
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
    sign = "+" if delta > 0 else ("" if delta == 0 else "")
    delta_str = f"{sign}{delta:+.4f}".replace("++", "+")
    line = (f"  {metric:<32} {base:>10.4f}  ->  {cand:>10.4f}   "
            f"{arrow} {delta_str}")
    if over_budget:
        line = _color(line + "   [OVER BUDGET]", "regression",
                      enabled=color_enabled)
    else:
        line = _color(line, kind, enabled=color_enabled)
    return line, over_budget


def main() -> int:
    args = parse_args()
    color_enabled = (not args.no_color) and sys.stdout.isatty()

    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())

    print(_color(
        f"baseline:  {args.baseline.name}  ({base.get('experiment')})",
        "bold", enabled=color_enabled))
    print(_color(
        f"candidate: {args.candidate.name}  ({cand.get('experiment')})",
        "bold", enabled=color_enabled))
    print()

    any_regression = False

    print(_color("== aggregate ==", "bold", enabled=color_enabled))
    base_sum = base["summary"]
    cand_sum = cand["summary"]
    for metric in sorted(base_sum.keys() | cand_sum.keys()):
        line, regressed = _fmt_row(metric, base_sum.get(metric),
                                   cand_sum.get(metric),
                                   color_enabled=color_enabled,
                                   threshold=args.threshold)
        print(line)
        any_regression = any_regression or regressed

    base_pd = {d["image"]: d for d in base.get("per_doc", [])}
    cand_pd = {d["image"]: d for d in cand.get("per_doc", [])}
    common = sorted(base_pd.keys() & cand_pd.keys())

    if common:
        print()
        print(_color("== per-doc ==", "bold", enabled=color_enabled))
        for img in common:
            print(f"\n  {_color(img, 'dim', enabled=color_enabled)}")
            for metric in ("text_block_edit", "display_formula_edit",
                           "table_edit", "reading_order_edit"):
                line, regressed = _fmt_row(metric,
                                           base_pd[img].get(metric),
                                           cand_pd[img].get(metric),
                                           color_enabled=color_enabled,
                                           threshold=args.threshold)
                print("  " + line)
                any_regression = any_regression or regressed

    base_rt = base.get("runtime_sec", {})
    cand_rt = cand.get("runtime_sec", {})
    if base_rt and cand_rt:
        print()
        print(_color("== runtime (sec) ==", "bold", enabled=color_enabled))
        for k in ("ocr", "md", "score", "total"):
            b = base_rt.get(k)
            c = cand_rt.get(k)
            if b is None or c is None:
                continue
            print(f"  {k:<10} {b:>8.2f}  ->  {c:>8.2f}   ({c - b:+.2f}s)")

    if any_regression:
        print()
        print(_color(
            f"FAIL: at least one metric regressed by > {args.threshold:.4f}",
            "regression", enabled=color_enabled))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
