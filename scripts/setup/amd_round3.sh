#!/usr/bin/env bash
# Round 3 = the ENTIRE residual AMD verification, hands-free, in one command:
#
#   bash scripts/setup/amd_pod_bootstrap.sh          # env + build (self-repairing)
#   bash scripts/setup/amd_round3.sh [mxr_cache.tgz] [rocm_wheel.whl]
#
# Runs, in order (each phase logs to /root/artifacts and is safe to re-run):
#   0. restore the .mxr compile cache (the round-2 tar) — THE step that turns
#      every measurement below from hours of MIGraphX compiling into minutes
#   1. correctness gates (probe, unit, goldens SEQUENTIALLY, conformance, FUNSD)
#   2. bench windows for ALL tiers (--threads 5 --repeat N; >= 15 s enforced
#      by turbo_bench itself; accuracy bar: tiny ~85, small/medium ~92 F1)
#   3. python GPU smoke + timed reads (3.12 env; LD_PRELOAD trap applied)
#   4. onnx-mode scored run (needs the build-onnx tree / ROCm ORT — skipped
#      with a WARNING if absent)
#   5. server: health, real image, corrupt-input must 4xx, 5-minute soak
#   6. wheel: pip-install the rocm wheel into a fresh venv and read one page
#      (the LAST unchecked box before PyPI publish)
# Artifacts land in ~/artifacts; rsync that directory home CONTINUOUSLY.
#
# Goldens are sequential and nothing here runs concurrently with anything
# else GPU-bound: concurrent MIGraphX first-runs serialize on the device
# compile lock (~10x per-variant, measured round 2).
set -euo pipefail

CACHE_TAR="${1:-}"
WHEEL="${2:-}"
cd "$(dirname "$0")/../.."
export SCCACHE_DISABLE=1 CCACHE_DISABLE=1
FUNSD="${FUNSD_CACHE:-$HOME/funsd_cache}"
ART="$HOME/artifacts"; mkdir -p "$ART"

phase() { echo; echo "==== $* ===="; }

# ---- 0. compile cache ------------------------------------------------------
if [ -n "$CACHE_TAR" ] && [ -f "$CACHE_TAR" ]; then
  phase "restoring MIGraphX compile cache"
  tar xzf "$CACHE_TAR" -C "$HOME"
  echo "cache: $(ls "$HOME/.cache/turbo-ocr" | wc -l) programs"
else
  phase "NO cache tar given — first runs will pay full MIGraphX compiles"
fi

# ---- 1. gates --------------------------------------------------------------
phase "gates"
./build/turbo_backend_probe --list | tee "$ART/probe.txt"
ctest --test-dir build -j8 -R 'test_|backend_probe' --output-on-failure
for g in det cls rec layout; do
  ctest --test-dir build -R "golden_amd_$g" --output-on-failure
  cp build/Testing/Temporary/LastTest.log "$ART/LastTest_golden_$g.log"
done
ctest --test-dir build -R 'backend_conformance' --output-on-failure
ctest --test-dir build -R 'funsd_amd_tiny' --output-on-failure
cp build/Testing/Temporary/LastTest.log "$ART/LastTest_funsd.log"

# ---- 2. bench, all tiers ---------------------------------------------------
phase "bench (tiny/small/medium, >= 15 s windows)"
for spec in "tiny 40" "small 20" "medium 10"; do
  set -- $spec
  ./build/turbo_bench --backend amd --tier "$1" --images "$FUNSD" \
    --threads 5 --repeat "$2" \
    --words "/tmp/amd_$1.words.json" --out "/tmp/amd_$1.metrics.json" \
    | tee "$ART/bench_$1.txt"
  cp "/tmp/amd_$1".*.json "$ART/"
done
tar czf "$ART/mxr_cache_$(/opt/rocm/bin/rocminfo | grep -om1 'gfx[0-9a-f]*').tgz" \
  -C "$HOME" .cache/turbo-ocr

# ---- 3. python -------------------------------------------------------------
phase "python suite + GPU smoke"
PY="${PYTHON312:-/opt/conda/envs/py312/bin/python}"
[ -x "$PY" ] || PY=python3
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
( cd python && "$PY" -m pytest tests/ -q | tail -3 | tee "$ART/py_suite.txt" )
"$PY" -u - <<'PYEOF' 2>&1 | tee "$ART/py_smoke.txt"
import glob, time
import cv2
import turboocr_engine as t
o = t.OCR(backend="amd")
imgs = [cv2.imread(p) for p in sorted(glob.glob(
    __import__("os").path.expanduser("~/funsd_cache/funsd_0*.png")))[:20]]
t0 = time.time(); r = o.read(imgs[0]); print(f"first read {time.time()-t0:.2f}s, lines={len(r.lines)}")
t0 = time.time()
n = sum(len(o.read(im).lines) for im in imgs)
dt = time.time() - t0
print(f"20 warm reads: {dt:.2f}s = {20/dt:.1f} img/s, lines={n}")
o.close()
PYEOF
unset LD_PRELOAD

# ---- 4. onnx-mode ----------------------------------------------------------
if [ -x build-onnx/turbo_bench ]; then
  phase "onnx-mode (ORT + MIGraphX EP)"
  TURBO_ENGINE_MODE=onnx ORT_EP=migraphx ./build-onnx/turbo_bench --backend amd \
    --tier tiny --images "$FUNSD" --threads 5 --repeat 10 \
    --words /tmp/onnx_tiny.words.json --out /tmp/onnx_tiny.metrics.json \
    | tee "$ART/bench_onnx_tiny.txt"
else
  phase "WARNING: no build-onnx tree (ROCm ORT not installed) — onnx-mode SKIPPED"
fi

# ---- 5. server + soak ------------------------------------------------------
phase "server smoke + 5-min soak"
TURBO_BACKEND=amd ./build/turboocr-server --http-port 18860 --grpc-port 50061 \
  > "$ART/server.log" 2>&1 &
SRV=$!; sleep 25
{ curl -sf localhost:18860/health && echo HEALTH-OK
  head -c 4096 /dev/urandom > /tmp/corrupt.bin
  C=$(curl -s -o /dev/null -w "%{http_code}" --data-binary @/tmp/corrupt.bin \
      -H "Content-Type: image/png" localhost:18860/ocr/raw)
  echo "corrupt-input HTTP $C (must be 4xx/5xx, server must survive)"
  curl -sf localhost:18860/health && echo HEALTH-AFTER-CORRUPT-OK
} | tee "$ART/server_smoke.txt"
python3 tools/bench/soak.py --base http://127.0.0.1:18860 \
  --images "$FUNSD" --pdf tests/fixtures/pdf/test8.pdf --pdf-pages 8 --minutes 5 \
  | tee "$ART/soak_stats.txt"
curl -sf localhost:18860/health && echo HEALTH-FINAL-OK
kill "$SRV" 2>/dev/null || true

# ---- 6. wheel --------------------------------------------------------------
if [ -n "$WHEEL" ] && [ -f "$WHEEL" ]; then
  phase "wheel install test (the last box before PyPI)"
  "$PY" -m venv /tmp/wheelenv
  /tmp/wheelenv/bin/pip install -q "$WHEEL"
  /tmp/wheelenv/bin/python - <<'PYEOF' 2>&1 | tee "$ART/wheel_test.txt"
import cv2, os
import turboocr_engine as t
o = t.OCR(backend="amd")
r = o.read(cv2.imread(os.path.expanduser("~/funsd_cache/funsd_000.png")))
print(f"WHEEL-OK lines={len(r.lines)} backend={o.info().get('backend')}")
o.close()
PYEOF
else
  phase "no wheel path given — wheel install test SKIPPED"
fi

phase "ROUND 3 COMPLETE — rsync ~/artifacts home before teardown"
