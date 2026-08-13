# AMD bring-up runbook — hourly-billed box, minimal paid time

Written 2026-08-02, before the first hardware session. The goal of this file is
that the paid GPU hours contain **only** the work that needs a GPU: every
compile-class risk was already burned down on the dev machine (see "Already
verified" below), so on the box you execute phases in order, check each gate,
and stop at the first genuine hardware question.

Target: **~2–2.5 paid hours** for a full pass (build → goldens → FUNSD →
bench → soak), assuming nothing hardware-surprising. Collect artifacts as you
go (phase H) — never leave results only on a rental.

---

## Already verified WITHOUT hardware (do not re-litigate on the meter)

Done 2026-08-02 in a `rocm/dev-ubuntu-24.04:7.1.1` container (x86_64, no GPU,
compile-only — HIP 7.1 / MIGraphX 2.14 / amdclang 20):

- `cmake -DTURBO_BACKENDS="cpu;amd"` **configures**. Requires the MIOpen/
  rocBLAS dev cmake configs (transitive deps of `migraphx-config.cmake`):
  `apt install miopen-hip-dev rocblas-dev hipblas-dev` beyond `migraphx-dev`.
- `turbo_ocr_backend_amd` **compiles clean**, including all five `.hip` device
  kernels for BOTH `gfx942` (CDNA/wave64) and `gfx1100` (RDNA/wave32), and
  `migraphx_engine.cpp` against the **real** MIGraphX headers (three
  from-documentation API errors were found and fixed:
  `run_async(pp, stream)` is a template over the stream type, and
  `migraphx::arguments` is not default-constructible).
- Link-level traps fixed: `hip::host` not `hip::device` (device injects
  `-x hip` into g++ TUs), and **IPO/LTO is auto-disabled for amd configures**
  (GCC slim-LTO archives are unreadable by ld.lld — symptoms look exactly like
  a link-order bug; the CMake message tells you it happened).
- The `.mxr` compile cache is implemented (`MIGRAPHX_ENGINE_CACHE`, default
  `~/.cache/turbo-ocr`): the 42-graph warmup ladder compiles once per
  (model, shape, gfx, ROCm), then loads from disk. **Do not delete
  ~/.cache/turbo-ocr between runs on the box — it is what makes restarts
  cheap.**
- The REAL `argmax_kernel` (the wavefront-agnostic rewrite, AMD-only code with
  no CUDA coverage) passes the tie-break contract executed on CPU threads via
  ROCm's HIP-CPU header library — all six engineered tie rows incl. the
  32/64 lane-boundary straddles. (HIP-CPU cannot run the CCL kernels — no
  cooperative groups — so det stays a Phase-D hardware question.)

What this does NOT cover (the actual hardware questions, in phase order):
kernel behaviour (CCL cooperative launch, wave64 argmax tie-break, LDS float
atomics), MIGraphX runtime semantics (`offload_copy=false` residency,
`run_async` stream discipline, `.mxr` save/load on a real device), fp16
accuracy, and every performance number.

## Choosing the rental

- **ROCm 7.1.x preferred** (exact parity with the verified toolchain); 6.4+
  acceptable — expect only small API drift, all confined to
  `migraphx_engine.cpp`.
- Any CDNA part (MI210/MI250/MI300X = gfx90a/gfx942) or RDNA3 (gfx1100). The
  kernels are wavefront-agnostic by construction, and CCL has a non-cooperative
  fallback, so consumer parts are usable — but MI300X is the headline target.
- ≥40 GB disk (ROCm + models + build), ≥16 CPU cores helps the build.

## Phase A — preflight (5 min)

```bash
rocminfo | grep -m1 gfx           # note the gfx arch
apt list --installed 2>/dev/null | grep -E 'migraphx|miopen-hip-dev|rocblas-dev|hipcc' 
df -h ~ ; nproc
python3 -c 'import json'; curl --version >/dev/null && echo curl-ok
```
Missing dev packages: `sudo apt install hipcc rocm-hip-runtime-dev migraphx
migraphx-dev half miopen-hip-dev rocblas-dev hipblas-dev cmake ninja-build
g++ libopencv-dev` (+ the Drogon/gRPC deps from docker/Dockerfile's amd stage
if building the server: `libgrpc++-dev protobuf-compiler-grpc libjsoncpp-dev
uuid-dev zlib1g-dev libssl-dev libcurl4-openssl-dev nginx gosu gettext-base`,
then build Drogon v1.9.12 from source as in the Dockerfile).

## Phase B — code + models + test data (10 min, parallel with C)

```bash
git clone <origin> turbo && cd turbo        # or push/pull per repo policy
bash scripts/models/fetch/fetch_release_models.sh   # OUT=models
# FUNSD cache + the 8-page scanned test PDF: copy them from whichever machine
# already has them, then point FUNSD_CACHE / TEST_PDF at the copies.
```

## Phase C — build (15–25 min)

```bash
TARGETARCH=amd64 bash scripts/setup/install_pdfium.sh
sudo cp third_party/pdfium/lib/libpdfium.so /usr/lib/ && sudo ldconfig
TARGETARCH=amd64 bash scripts/setup/install_fastpdf2png.sh   # PDF daemon binary

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON \
      -DFETCH_MODELS=OFF -DTURBO_BACKENDS="cpu;amd" \
      -DCMAKE_HIP_ARCHITECTURES="$(rocminfo | grep -om1 'gfx[0-9a-f]*')" \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DTURBO_FUNSD_CACHE="$HOME/funsd_cache"
ninja -C build turbo_backend_probe turbo_golden turbo_bench turbo_conformance \
      turboocr-server turbo_ocr_tests
```
Expect: configure prints the amd UNVERIFIED warning + "IPO/LTO disabled".
Any compile error here is NEW relative to the container run — suspect ROCm
version drift, and fix in `migraphx_engine.cpp` first.

## Phase D — probe + unit tests + goldens (30 min; STOP-gate)

```bash
./build/turbo_backend_probe --list          # MUST list amd (registrar survived)
ctest --test-dir build -R 'test_|backend_probe' --output-on-failure   # unit tests
ctest --test-dir build -R 'golden_amd' --output-on-failure            # 4 stages
ctest --test-dir build -R 'backend_conformance' --output-on-failure
```
First golden run pays the MIGraphX warmup compiles (minutes) — watch for
`HOT-PATH COMPILE` lines afterwards; in steady state there must be none, and
`~/.cache/turbo-ocr/mgx_*.mxr` files should appear.

Failure triage (from README "Known AMD-specific correctness risks"):
- **det golden diff / zero boxes** → CCL. Check the cooperative-launch probe
  (`hipDeviceAttributeCooperativeLaunch`); the two-pass fallback must engage on
  parts that report 0. This is risk #2 and it fails as "blank page".
- **rec/cls text wrong, boxes right** → argmax. Run the synthetic tie-break
  check (README Stage 1.4): equal logits must pick the LOWER class index on
  wave64 AND wave32.
- **table tensors drift ~ulp** → fp contraction; set `-ffp-contract` on both
  sides, do not edit expressions (risk #5).

## Phase E — FUNSD accuracy (15 min; STOP-gate)

```bash
ctest --test-dir build -R 'funsd_amd_tiny' --output-on-failure   # floor 85.0
```
The gate is PROVISIONAL (same models through MIGraphX ⇒ expect ~85.5). If it
lands materially below: re-run with fp16 disabled (edit the two
`set_fp16(true)` sites in `stages/rocm_stages.cpp` — there is deliberately no
env knob yet) to separate quantization drift from stage bugs. A big gap with
fp16 OFF is a stage bug, not a device difference.

## Phase F — throughput (20 min)

```bash
for t in tiny small medium; do
  ./build/turbo_bench --backend amd --tier $t --images ~/funsd_cache --count 50 \
      --words /tmp/amd_$t.words.json --out /tmp/amd_$t.metrics.json
  python3 tools/bench/score_funsd.py /tmp/amd_$t.words.json --metrics /tmp/amd_$t.metrics.json
done
```
Report throughput AND F1 together, ≥15 s windows. Sanity: `hot_path_compiles`
still 0. `--engine-mode onnx` (ORT MIGraphX/ROCm EP) is NOT expected to run —
there is no official C++ ORT-ROCm binary; the dual-path comparison is deferred
until one exists.

## Phase G — server + soak (40 min incl. 5-min soak)

```bash
TURBO_BACKEND=amd ./build/turboocr-server --http-port 18860 --grpc-port 50061 &
curl -sf localhost:18860/health && curl -s localhost:18860/capabilities | head -c 300
python3 tools/bench/soak.py --base http://127.0.0.1:18860 \
    --images ~/funsd_cache --pdf /tmp/test8.pdf --pdf-pages 8 --minutes 5
```
Pass = every counter `_200`/`stream_ok`, VRAM peak stable, zero error-class log
lines, healthy after. (Kill with `killall turboocr-server` — `pkill -f` matches
your own ssh command line.) Also exercise the HIP error policy once: a corrupt
image request must 4xx/5xx, never kill the server (README risk: two-tier
HIP_CHECK has never seen a real fault).

## Phase H — collect BEFORE teardown (10 min)

scp home: `build/golden_amd_*.json`, `build/conformance.json`,
`build/funsd_amd_*.json`, `/tmp/amd_*.json`, the soak stats line, the server
log, and `rocminfo | head`. Update `docs/` + the gap doc with measured numbers
and flip this file's status header. If the rental will be reused, also tar
`~/.cache/turbo-ocr/mgx_*.mxr` (per-gfx — only reusable on the same arch).
