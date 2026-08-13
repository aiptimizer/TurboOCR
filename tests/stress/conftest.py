"""Stress suite fixtures. Adds the benchmark perf dir to sys.path so we can
import the harness primitives (`_harness_import` / `_harness`, which live
next to the perf bench drivers that share them).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent / "benchmark" / "perf"
for d in (HERE, BENCH_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
