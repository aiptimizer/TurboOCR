# AMD bring-up runbook — hourly-billed box, minimal paid time

**STATUS (updated 2026-08-26, after the second hardware session):**
**correctness RE-PROVEN and tiny throughput MEASURED on MI300X.** All gates
green on the first try (4 goldens, conformance, FUNSD F1 85.06–85.15%, unit
suite) and the Python binding is suite-green on hardware (123 passed / 0
failed, 3.12 stable-ABI build). The first PUBLISHABLE throughput number:
**tiny = 105.5 img/s, F1 85.15%** (18.95 s window, 0.03% clock skew,
threads=5) vs 415.8 on a 5090 and 63.7 on M3 Max. small/medium windows and
the onnx-mode run were eaten by MIGraphX compile time (see the round-2
session notes); `mxr_cache_gfx942.tgz` (~85 compiled programs) is archived —
**round 3 MUST restore it to ~/.cache/turbo-ocr first**, making those
measurements minutes, not hours. Bring-up itself is now ONE COMMAND (below).

**One command replaces phases A–F** (added round 2, 2026-08-26):

```bash
rsync -az --exclude '/.git' --exclude '/build*' --exclude '/tmp' \
      --exclude '/models' --exclude '/third_party/pdfium' \
      <repo>/ pod:~/turbo/ && ssh pod \
  'cd ~/turbo && bash scripts/setup/amd_pod_bootstrap.sh --gates --bench'
```

The script is idempotent by real probes (ROCm version, cmake version, a
`<format>` compile test, installed-file checks), so a well-chosen image skips
straight to the build and a failed run resumes where it stopped. Every trap
in this file is encoded in it. `--with-ort` adds the onnx-mode runtime. The
manual phases below remain as documentation of what it does and as the
failure-triage guide.

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

---

## Second hardware session — 2026-08-26, RunPod MI300X (same host), ROCm 7.1.1

~2 paid hours. Full detail + chronology: `tmp/amd-round2/SESSION_LOG.md`
(kept on the dev machine with every artifact, pulled continuously — nothing
was lost to pod death this time).

**Proven:** every gate green FIRST TRY on a fresh pod (unit, det/cls/rec/
layout goldens, backend_conformance, funsd_amd_tiny F1 85.06%); Python
binding suite-green on hardware (123 passed / 27 skipped / 0 failed; 3.12
stable-ABI `_turboocr.abi3.so`); first publishable throughput —
**tiny 105.5 img/s @ F1 85.15%** (threads=5, 18.95 s window, 0.03% skew).
The bring-up itself became `scripts/setup/amd_pod_bootstrap.sh` (one
command), and the ORT>=1.27 MIGraphX-EP link-probe landed in CMakeLists +
ort_engine.cpp (configure prints "ONNX Runtime exports the MIGraphX EP").

**Not measured (compile time ate the window):** small/medium throughput
windows, the onnx-mode scored run, the 5-min soak, and the wheel-on-hardware
check. The dominant cost was MIGraphX compilation: ~100 programs, with
det_small canvas graphs at MINUTES each. Round-3 rules: (1) restore
`mxr_cache_gfx942.tgz` into `~/.cache/turbo-ocr` BEFORE anything runs;
(2) NEVER run goldens/benches concurrently — parallel first-runs serialize
on the GPU's compile lock (measured ~10x per-variant slowdown at ctest -j8);
(3) `--count` clamps to the image set — size windows with
`--threads 5 --repeat N`.

**Round 3 is ONE command** — after the bootstrap, `scripts/setup/amd_round3.sh
[mxr_cache.tgz] [rocm_wheel.whl]` runs the entire residual verification
hands-free (cache restore -> gates -> all-tier bench -> python smoke ->
onnx-mode -> server smoke + soak -> wheel install test), with every rule
above baked in and all artifacts in ~/artifacts for the sync-home loop.

## First hardware session — 2026-08-24, RunPod MI300X (gfx942), ROCm 7.1.1

~80 paid minutes. The pod terminated before Phase F/G artifacts were synced
home, so every number below is from the live logs; nothing else survived.

### Proven (STOP-gates D and E cleared)

- `turbo_backend_probe --list`: `amd cpu`; auto-detect -> `backend='amd'
  device='hip' async=1 pool=5`.
- **All four goldens PASSED** vs the cpu reference: det 16.4s, cls 17.3s,
  rec 20.2s, layout 37.6s (times include first-compile warmup).
- **backend_conformance PASSED** (18.9s).
- **funsd_amd_tiny PASSED: F1 = 85.21%** against the provisional 85.0 floor
  set blind on 2026-08-02. fp16 stayed ON; no accuracy fallback was needed.

### NOT proven — round 2's entire job

- **Throughput**: no valid number exists. (A 45.8 img/s tiny figure was
  logged mid-session; it is contention-tainted — two benches plus goldens
  shared the GPU — and its window was 1.1s where the runbook demands >=15s.
  Do not cite it.) Reference bar: the same harness on a 5090 does 415.8 img/s.
- **Phase G** (server + corrupt-input survival + soak): never completed.
- **`--engine-mode onnx` on the GPU** (ORT 1.28.0 + MIGraphX EP): the ORT
  build SUCCEEDED (recipe below) but the verification never ran.
- The `.mxr` cache (~82 graphs) and the ORT build died with the pod.

### Hardware-forced fixes (all in the tree, see git log)

1. `MIGraphXEngine::load` no longer parses the model's DECLARED shapes —
   MIGraphX materializes static shapes AT PARSE, and rec's placeholder width
   fails its pooling ("POOLING: not enough padding"). I/O names now come from
   a parse-only probe with `set_default_dim_value`, and nothing compiles at
   load.
2. `offload_copy=false` programs expose outputs as parameters
   (`main:#output_N`) and `run_async` throws unless EVERY one is bound —
   each variant now owns device output buffers.
3. `.mxr` artifacts are validated against the requested input shapes on load
   AND after compile — a stale pre-pinning artifact under a batched key
   overran an 8-byte buffer (batch-64 cls key loading a batch-1 program).
4. **Layout cannot run on MIGraphX at all** — its parser rejects the
   PP-DocLayoutV3 export twice over (data-dependent post-NMS reshape;
   an uninitialized-size_t op-builder bug at pinned dims). Layout now routes
   through the shared host-ORT stage (`HostLayoutOnHip`, the Apple
   `HostLayoutOnDevice` pattern) — goldens stay byte-comparable, and an
   ORT+MIGraphX build can move it to GPU later inside the same design.
5. `pdf_ppm.cpp`/`page_image_encoder.cpp` now build without libturbojpeg
   (OpenCV JPEG fallback) — a missing optional dependency used to be a LINK
   FAILURE, first hit on this pod.

### Traps that burned paid time — round 2 must dodge every one

- **Pick a pod image with ROCm >= 6.4 preinstalled.** The default image had
  5.7; installing 7.1.1 via apt cost ~20 min.
- **RunPod images put `/opt/cache/bin` (sccache) first in PATH** and it
  serves STALE objects for rsynced, mtime-preserved sources — a "successful"
  build with new symbols silently missing. `export SCCACHE_DISABLE=1
  CCACHE_DISABLE=1` before every build, always.
- **jammy binutils 2.38 cannot assemble ORT 1.28 MLAS** (AVX-NE-CONVERT).
  Build ORT with `CC=amdclang CXX=amdclang++` and
  `--compile_no_warning_as_error` (clang 20 trips -Werror in ORT's own
  MIGraphX EP). Working invocation: `./build.sh --config Release
  --build_shared_lib --parallel 96 --skip_tests --compile_no_warning_as_error
  --use_migraphx --rocm_home /opt/rocm-X --migraphx_home /opt/rocm-X
  --allow_running_as_root`.
- **Prebuild ORT+MIGraphX and Drogon OFF-meter** (any x86 box with a ROCm
  container) and ship binaries; both together cost ~45 paid minutes here.
- **Sync artifacts home CONTINUOUSLY** (a 5-minute rsync loop), not at
  teardown — this session lost its bench/soak evidence to a dead pod.
- **Tar `~/.cache/turbo-ocr` home after warmup** — the .mxr set is per-gfx
  reusable and is the single most expensive artifact on the box.
- `--count 50` finishes in ~1s on an MI300X: size bench counts until the
  measured window is >= 15s or the numbers are noise.
- (round 2, same host) **`ssh.runpod.io` is a PTY-only proxy** — no exec, no
  rsync/scp. Demand the "SSH over exposed TCP" connection (root@ip -p port)
  before doing anything.
- (round 2) **Ubuntu 22.04 pod images ship cmake 3.18 and g++ 11.** The tree
  needs cmake >= 3.24 and `<format>` (= GCC 13 libstdc++). Fix in ~3 min:
  `pip install cmake ninja` (conda bin precedes /usr/local in PATH) and
  `add-apt-repository ppa:ubuntu-toolchain-r/test && apt install g++-13`,
  then configure with `-DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13`.
  (The 2026-08-02 prep container was Ubuntu 24.04, where GCC 13 is default —
  that is why compile-verification never saw this.)
- (round 2) When rsyncing the tree, `third_party/` must come along (clipper,
  catch2, cli11, nlohmann, simdutf, wuffs are vendored SOURCES) — exclude only
  `third_party/pdfium`, which `install_pdfium.sh` provisions pod-side.

### Detail addenda from the session (for round 2 and upstream filing)

- **Fix #1's failures were DISGUISED as plausible numbers**: with outputs
  unbound, cls reported "97.27% agreement" (the stage defaulting to 0 deg on
  mostly-upright pages) and rec a mean_abs_delta of 0.907 (CPU confidences vs
  zeroes). Golden thresholds caught it; eyeballs would not have. Trust gates,
  not vibes.
- **FUNSD identity block** (for exact reproduction): F1 85.21 / P 84.60 /
  R 85.93, 50 pages, threads=1 repeat=1, images_sha ab699c47004f1d4d,
  det_tiny 193bab7a04fca699, rec_tiny 9ef676d6ed3c8825, keys c5cbe34ef40c29c4,
  cls 5fcd13afa5bf4719. 5090 reference 85.37% — the 0.16pt delta is
  consistent with fp16; no fp16-off A/B was needed. Worst pages: 39, 19, 15,
  36, 21.
- **det saw THREE canvases** this session (96x96, 992x768, 992x800): the
  steady-state hot_path_compiles=0 check must cover the real canvas set, via
  cache carry-over or a det warmup policy.
- **File upstream at ROCm/AMDMIGraphX** (repros in hand): (1) Reshape with an
  EMPTY target shape (legal ONNX rank-0, 1 element) computes "0 elements" —
  a 98-byte single-node model passes onnx.checker and crashes
  migraphx-driver read; layout.onnx has three such nodes (paddle2onnx
  artifacts, Reshape.395/.398/.404). (2) At pinned dims, an
  uninitialized-size_t "Inconsistent strides" in the conv op builder. Also
  test MIGraphX 2.15 (has reshape shape-computation fixes, unverified).
- **Engine invariants worth knowing when editing migraphx_engine.cpp**: the
  probe parse ladder is {64,1} because no single default dim parses every
  model (rec pooling needs >=64 spatial; batched-NMS heads only parse small),
  and ShapeVariant::input_names exists because get_parameter_shapes().names()
  is NOT graph order (layout.onnx declares im_shape before image) —
  positional pinning mis-pins any multi-input model.
- **Python-on-ROCm round-2 notes**: `pip install nanobind` first (the
  BUILD_PYTHON block resolves it via the interpreter, no FetchContent); use
  Python >=3.12 so the abi3 wheel shape is exercised (3.11 builds a
  cpython-tagged .so — Development.SABIModule needs 3.12); conda pods need
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (conda libstdc++
  lacks GLIBCXX_3.4.31, and the failure hides until first native use);
  install opencv-python-headless + reportlab beyond the obvious deps. The
  Python construct path compiles ~31 rec-bucket .mxr graphs the C++ chain
  never touches (~5 min) — pre-warm or persist the cache before timing.
- **Gate-harness gap found**: the goldens ctest phase ran without
  --output-on-failure, and the conformance ctest after it OVERWRITES
  Testing/Temporary/LastTest.log — a failing golden's stdout is
  unrecoverable by design. Fix the phase script (and consider tee) before
  round 2.

