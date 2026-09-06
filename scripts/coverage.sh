#!/usr/bin/env bash
# Line/branch coverage for the host C++ code.
#
# Builds an instrumented copy in build-coverage/ (-DTURBO_COVERAGE=ON, host C++
# only — CUDA .cu objects are not instrumented), runs the C++ unit suite, and
# aggregates the *.gcda with gcovr. The report's 0-hit lines/branches are the
# paths no test exercises — the class of gap that shipped #22 and #23.
#
# Usage:
#   scripts/coverage.sh                 # build + ctest + report
#   scripts/coverage.sh --no-build      # reuse build-coverage/, just re-run+report
#   COVERAGE_INTEGRATION=1 scripts/coverage.sh
#         # also boot the instrumented server and run the pytest integration
#         # suite against it, so /ocr/* route + decode-dispatch code is covered
#
# Env knobs: TENSORRT_DIR, CUDA_ARCH (default from the existing build cache).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BUILD=build-coverage
# TensorRT is discovered by CMake (tarball root, /usr/local/tensorrt, or the
# distribution packages); pass the hint through only when one is given, or
# reuse the existing build's.
TRT_DIR="${TENSORRT_DIR:-$(sed -n 's/^TENSORRT_DIR:PATH=//p' build/CMakeCache.txt 2>/dev/null)}"
TRT_FLAG=()
[[ -n "$TRT_DIR" ]] && TRT_FLAG=(-DTENSORRT_DIR="$TRT_DIR")
CUDA_ARCH="${CUDA_ARCH:-75}"
GCOVR=(python3 -m gcovr)

if [[ "${1:-}" != "--no-build" ]]; then
  cmake -S . -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTURBO_COVERAGE=ON \
    "${TRT_FLAG[@]}" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DTURBO_PDFIUM_DIR="$ROOT/third_party/pdfium"
  cmake --build "$BUILD" --target turbo_ocr_tests -j"$(nproc)"
  # The server target is only needed for route/decode-dispatch coverage.
  if [[ "${COVERAGE_INTEGRATION:-0}" == "1" ]]; then
    cmake --build "$BUILD" --target turboocr-server -j"$(nproc)"
  fi
fi

# Wipe stale counters so the report reflects only this run.
find "$BUILD" -name '*.gcda' -delete 2>/dev/null || true

echo "== running C++ unit suite (ctest) =="
ctest --test-dir "$BUILD" --output-on-failure || true

if [[ "${COVERAGE_INTEGRATION:-0}" == "1" ]]; then
  echo "== booting instrumented server for integration coverage =="
  PORT=8097 GRPC_PORT=50097 PIPELINE_POOL_SIZE=2 OCR_MODEL=tiny \
    "$BUILD/turboocr-server" >/tmp/coverage_server.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  for i in $(seq 1 300); do
    curl -sf http://localhost:8097/health >/dev/null 2>&1 && break; sleep 1; done
  ( cd tests && python3 -m pytest integration/ \
      --server-url http://localhost:8097 --grpc-target localhost:50097 \
      -q -p no:cacheprovider ) || true
  # gcov flushes counters on normal exit; terminate cleanly (not -9).
  kill -INT $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; trap - EXIT
fi

echo "== aggregating coverage (src/ + include/) =="
mkdir -p "$BUILD/coverage"
"${GCOVR[@]}" -r "$ROOT" \
  --filter 'src/' --filter 'include/turbo_ocr/' \
  --exclude '.*third_party.*' --exclude '.*/tests/.*' \
  --gcov-ignore-parse-errors \
  --html-details "$BUILD/coverage/index.html" \
  --txt "$BUILD/coverage/summary.txt" \
  --print-summary || true

echo
echo "== files with ZERO executed lines (never entered by any test) =="
"${GCOVR[@]}" -r "$ROOT" --filter 'src/' --filter 'include/turbo_ocr/' \
  --exclude '.*third_party.*' --exclude '.*/tests/.*' \
  --gcov-ignore-parse-errors --csv 2>/dev/null \
  | awk -F, 'NR>1 && $4==0 {print "  0% line  "$1}' | sort || true

echo
echo "HTML report: $BUILD/coverage/index.html"
echo "Text summary: $BUILD/coverage/summary.txt"
