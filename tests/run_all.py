#!/usr/bin/env python3
"""Master test runner for the Turbo OCR test suite.

Suites (fast -> slow):
    unit         — pure-python unit tests
    cpp          — C++ unit tests (run via ctest)
    integration  — HTTP/gRPC endpoint correctness
    regression   — ordering + synthetic smoke regressions
    accuracy     — ground-truth F1/CER per fixture per endpoint
    stress       — 60s soak per endpoint (opt-in, excluded from default)
    benchmark    — performance measurement (opt-in, excluded from default)

Usage:
    python tests/run_all.py                          # default suites only
    python tests/run_all.py --suite unit
    python tests/run_all.py --suite integration --suite accuracy
    python tests/run_all.py --suite stress           # opt-in stress soak
    python tests/run_all.py --suite benchmark        # opt-in perf
    python tests/run_all.py --suite all              # everything
"""

import argparse
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

SUITES = {
    "unit":        TESTS_DIR / "unit",
    "cpp":         TESTS_DIR / "cpp",
    "integration": TESTS_DIR / "integration",
    "regression":  TESTS_DIR / "regression",
    "accuracy":    TESTS_DIR / "accuracy",
    "stress":      TESTS_DIR / "stress",
    "benchmark":   TESTS_DIR / "benchmark",
}

DEFAULT_ORDER = ["unit", "integration", "regression", "accuracy"]
OPT_IN = {"stress", "benchmark", "cpp"}


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
    if "all" in suites_req:
        ordered = ["unit", "cpp", "integration", "regression", "accuracy", "stress", "benchmark"]
    else:
        ordered = [s for s in ["unit", "cpp", "integration", "regression", "accuracy", "stress", "benchmark"]
                   if s in suites_req]

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

    # Marker filtering: default run excludes stress and benchmark.
    markers = []
    if "stress" not in ordered:
        markers.append("not stress")
    if "benchmark" not in ordered:
        markers.append("not benchmark")
    if markers:
        pytest_args.extend(["-m", " and ".join(markers)])

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
