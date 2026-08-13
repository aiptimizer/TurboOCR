#!/usr/bin/env python3
"""Master test runner for the Turbo OCR test suite.

Suites (fast -> slow):
    cpp          — C++ Catch2 suite under tests/cpp (run via ctest)
    integration  — HTTP/gRPC endpoint correctness
    regression   — ordering + synthetic smoke regressions
    accuracy     — ground-truth F1/CER per fixture per endpoint
    stress       — 60s soak per endpoint (opt-in, excluded from default)

Usage:
    python tests/run_all.py                          # default suites only
    python tests/run_all.py --suite integration --suite accuracy
    python tests/run_all.py --suite cpp              # ctest, not pytest
    python tests/run_all.py --suite stress           # opt-in stress soak
    python tests/run_all.py --suite all              # everything

Benchmarks are NOT a suite: tests/benchmark holds standalone drivers, not
test_*.py, so pytest collects nothing there. Run one directly, e.g.
    python tests/benchmark/perf/bench_matrix.py --quick
    python tests/benchmark/scoring/bench_per_document.py
Likewise tests/e2e holds drivers that boot a real server or container.
"""

import argparse
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

SUITES = {
    "cpp":         TESTS_DIR / "cpp",
    "integration": TESTS_DIR / "integration",
    "regression":  TESTS_DIR / "regression",
    "accuracy":    TESTS_DIR / "accuracy",
    "stress":      TESTS_DIR / "stress",
}

# Suite order, once and only once — every list below is derived from it, so a
# new suite is added here alone.
ORDER = ["cpp", "integration", "regression", "accuracy", "stress"]
DEFAULT_ORDER = ["integration", "regression", "accuracy"]
OPT_IN = {"stress", "cpp"}


def main():
    parser = argparse.ArgumentParser(
        description="Turbo OCR test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite", action="append",
        choices=list(SUITES.keys()) + ["all"],
        help="Suite(s) to run. Default runs: " + ", ".join(DEFAULT_ORDER),
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--grpc-target",
        default=os.environ.get("OCR_GRPC_TARGET", "localhost:50051"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-x", "--exitfirst", action="store_true")
    parser.add_argument("-k", default=None)
    args = parser.parse_args()

    suites_req = args.suite or DEFAULT_ORDER
    ordered = list(ORDER) if "all" in suites_req else [s for s in ORDER if s in suites_req]

    # Handle the C++ suite separately (ctest, not pytest). Its exit code
    # must survive into the final status regardless of --exitfirst — a red
    # C++ suite exiting green here would be a false-pass gate.
    cpp_rc = 0
    if "cpp" in ordered:
        ordered.remove("cpp")
        print("=== cpp suite (ctest) ===")
        import shutil
        import subprocess
        build_dir = TESTS_DIR.parent / "build"
        if not build_dir.is_dir():
            print(f"cpp suite: build dir {build_dir} not found — "
                  f"configure & build first (cmake -B build ...)")
            cpp_rc = 1
        elif shutil.which("ctest") is None:
            print("cpp suite: ctest not on PATH — install CMake")
            cpp_rc = 1
        else:
            # --no-tests=error so a zero-registered suite fails instead of
            # silently passing (the false-green the add_test comment warns of).
            cpp_rc = subprocess.call(
                ["ctest", "--output-on-failure", "--no-tests=error"],
                cwd=build_dir)
        if cpp_rc != 0 and args.exitfirst:
            sys.exit(cpp_rc)

    dirs = [SUITES[s] for s in ordered if SUITES[s].exists()]
    if not dirs:
        print("no python suites selected")
        sys.exit(cpp_rc)

    pytest_args = [str(d) for d in dirs]
    pytest_args.extend([
        f"--server-url={args.server_url}",
        f"--grpc-target={args.grpc_target}",
        f"--rootdir={TESTS_DIR}",
    ])

    # Marker filtering: a soak is only ever run when asked for by name, even if
    # a stress-marked test were to live outside tests/stress/.
    if "stress" not in ordered:
        pytest_args.extend(["-m", "not stress"])

    if args.verbose or "benchmark" in ordered or "stress" in ordered:
        pytest_args.append("-v")
    if "benchmark" in ordered or "stress" in ordered:
        pytest_args.append("-s")
    if args.exitfirst:
        pytest_args.append("-x")
    if args.k:
        pytest_args.extend(["-k", args.k])

    try:
        import pytest
    except ImportError:
        print("pytest not installed. Run: pip install -r tests/requirements.txt")
        sys.exit(1)

    print(f"Running suites: {', '.join(ordered)}")
    print(f"Server: {args.server_url}")
    print(f"gRPC:   {args.grpc_target}")
    print()

    py_rc = int(pytest.main(pytest_args))
    sys.exit(py_rc if py_rc != 0 else cpp_rc)


if __name__ == "__main__":
    main()
