# `tests/cpp/backends/` — ONE cross-backend test harness

Three binaries. They run on **every** backend, and the *same* command produces
directly comparable numbers on your machine and on mine.

| binary | what it answers |
|---|---|
| `turbo_bench` | *How accurate and how fast is backend X here?* (accuracy + throughput, one protocol) |
| `turbo_conformance` | *Do the backends in this binary agree with each other?* (text + box IoU, diff table) |
| `turbo_golden` | *Which STAGE diverges?* (per-stage golden diff vs the CPU reference) |

They replace four per-backend forks — `funsd_unified_cpu.cpp`,
`funsd_unified_apple.mm`, `funsd_unified_apple_conc.mm`, `cls_golden_apple.mm` —
which were the same protocol written four times, each pinned to one backend, and
none of them recording *what* was run, so no number could be compared to another
machine's.

**Why one plain `.cpp` works for a Metal backend:** a driver that only calls
`backend::make_backend(name)` + `UnifiedOcrPipeline` names no vendor type. All
the Metal/MPSGraph lives inside `libturbo_ocr_backend_apple.a` behind the seam,
and all the CUDA will live inside `libturbo_ocr_backend_nvidia.a` the same way.
No Objective-C++, no `#ifdef __APPLE__`, nothing to port.

---

## 1. Build

```bash
# Apple + CPU (this Mac)
cmake -B build-rebuild -S . -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON -DFETCH_MODELS=OFF \
      -DONNXRUNTIME_INCLUDE_DIR=/opt/homebrew/include/onnxruntime \
      -DONNXRUNTIME_LIB=/opt/homebrew/lib/libonnxruntime.dylib \
      -DTURBO_BACKENDS="cpu;apple" \
      -DTURBO_FUNSD_CACHE=$HOME/compare-ocrs/funsd_cache
ninja -C build-rebuild turbo_bench turbo_conformance turbo_golden turbo_backend_probe
```

```bash
# CPU + NVIDIA (the GPU box) — NEVER COMPILED, read §1a before running this
#
# NOTE: no -DUSE_CPU_ONLY. That option selects the CPU arm of the ROOT
# CMakeLists.txt; the NVIDIA backend wraps turbo_ocr_gpu, which only exists in
# the other arm. TENSORRT_DIR is the SAME cache variable the existing GPU build
# uses — if your box already builds turboocr-server, it needs no new flags.
cmake -B build-rebuild -S . -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DFETCH_MODELS=OFF \
      -DTENSORRT_DIR=/usr/local/tensorrt \
      -DTURBO_BACKENDS="cpu;nvidia" \
      -DTURBO_FUNSD_CACHE=$HOME/compare-ocrs/funsd_cache
ninja -C build-rebuild turbo_bench turbo_conformance turbo_golden turbo_backend_probe
```

Optional, for a native (no PTX JIT) build of your own GPU — same variable and
same meaning as the main tree, default sm_75/Turing with forward-JITting PTX:

```bash
      -DCMAKE_CUDA_ARCHITECTURES=90     # 75 Turing · 80/86 Ampere · 89 Ada · 90 Hopper · 100/120 Blackwell
```

If the CUDA configure gives you trouble, `-DTURBO_BACKENDS="nvidia"` (no `cpu`)
is the smaller target: it drops `turbo_ocr_cpu_host` (§1a) and everything still
builds except `turbo_conformance`, which needs two backends in one binary and
exits **77 = SKIP** with only one.

### 1a. First-configure checklist — WORKED 2026-08-10, all nine steps pass

**Bring-up is done.** The list below is kept because it is the procedure for the
*next* machine, not because the target is still unproven.

| | |
|---|---|
| Machine | Arch Linux, RTX 5090 (sm_120), driver 610.43.02 |
| Toolkit | CUDA 13.3.33, TensorRT 10.15.1.29, ONNX Runtime 1.27 (CUDA 13) |
| Configure | `-DTURBO_BACKENDS="nvidia;cpu" -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15` |
| Step 6 | `available_backends() [2]: nvidia cpu` — the registrar survived `--gc-sections` |
| Step 7 | TRT engines built and cached under `~/.cache/turbo-ocr/` |
| Step 8 | `turbo_conformance --count 10` exits 0 with a real cpu-vs-nvidia diff table |
| Step 9 | FUNSD tiny **F1 85.37%** against the provisional 85.0 floor — the floor stands, unchanged |
| Suite | all pass — 620 cases / 5775 assertions when measured 2026-08-10; the 3 platform-independent `test_decode_contract` cases added since bring the current total to 623 |
| Throughput | 415.8 img/s in-process, 353.7 img/s over HTTP — both at **2 replicas**, VRAM-capped (see note) |

The throughput figures were taken with ~19.5 GB of the card already held by
another process, so the pool capped at 2 of the tier's 5 replicas and device
utilization sat at 84%. They are a floor, not this card's ceiling.

Two things the checklist predicted correctly and one it did not: the CUDA
configure could not link the test suite at all until three seam TUs were moved
next to the library that defines their symbols (see the root `CMakeLists.txt`
comment above `tests/cpp/backends/test_seam_conformance.cpp`), which is a defect
in the *root* build, not in this arm.

Historical context for the list that follows: it was written on an Apple M3 Max
with no CUDA toolkit, no `nvcc` and no TensorRT. Every flag, library and include
in it was copied from something that *was* proven — the main tree's GPU arm
(`CMakeLists.txt:513-853`) and the working `turbo_ocr_backend_cpu` /
`turbo_ocr_backend_apple` targets — and the combination had never been
exercised. That is what the run above finally exercised; the configure now
prints a `STATUS` line naming the machine it was proven on rather than a
`WARNING`.

Work this list **in order** on a new machine. Each item is a thing that can only
be checked on the hardware, with the exact symptom to expect.

| # | Check | Command / what you should see | If it fails |
|---|---|---|---|
| 1 | **CUDAToolkit discovery** | configure prints `src/backends/nvidia: CUDAToolkit <ver> @ <root>` | `find_package(CUDAToolkit REQUIRED)` failed — same failure the main GPU build would hit; fix `PATH`/`CUDACXX` first. |
| 2 | **TensorRT discovery** | configure prints `src/backends/nvidia: TensorRT /usr/local/tensorrt` | A `FATAL_ERROR` naming `NvInfer.h` means `TENSORRT_DIR` is wrong. It is the SAME cache var as the main build — if `turboocr-server` builds, reuse its value. |
| 3 | **CUDA arch flags** | configure prints `src/backends/nvidia: CUDA architectures = 75` (or your `-DCMAKE_CUDA_ARCHITECTURES`) | Empty or `OFF` means the root's `enable_language(CUDA)` did not pre-fill it. Pass `-DCMAKE_CUDA_ARCHITECTURES=<your sm>` explicitly. There are **no `.cu` files in `src/backends/nvidia/`** — every kernel is already an `nvcc` object inside `turbo_ocr_gpu` — so this only matters if you add one. |
| 4 | **ORT-CUDA provider present** | configure prints `ONNX Runtime (CUDA EP): …` (root) and does **not** stop in the backend arm | A CPU-only ORT configures and links fine and then fails at **runtime** in `OrtCudaEngine::load` / `NvFormulaRecognizer`. The backend arm re-asserts `ONNXRUNTIME_CUDA_EP` so this is a configure error, not a 3-a.m. one. |
| 5 | **the NVIDIA image-decode TU links** | `ninja turbo_backend_probe` finishes | Undefined `nvidia::probe_nvjpeg` / `nvidia::make_nv_image_decoder` means `src/backends/nvidia/support/nv_image_decode.cpp` is not on the backend library. (These used to be `server::` functions in a `src/service/server/cuda/stages_gpu.cpp` that the root compiled into the `turboocr-server` **executable** rather than a library — which is why the backend had to pull the TU in itself. They now live inside the vendor arm, where they belong.) |
| 6 | **BOTH backends survived the link — the important one** | `./build/turbo_backend_probe --list` prints **`cpu` AND `nvidia`** | This is the `WHOLE_ARCHIVE` / `--gc-sections` test. `nv_backend_registry.cpp` defines a `BackendRegistrar` that **nothing references**, so a static archive is entitled to drop the whole object and the backend silently vanishes — and Linux's `-Wl,--gc-sections` is exactly where that bites (it never could on the Mac, where `build.sh` passed the `.o` explicitly). `turbo_link_backends()` force-links it. If `nvidia` is missing from `--list`, the target linked but the registrar did not. |
| 7 | **TensorRT dlopen at first engine build** | the first `turbo_bench --backend nvidia` run builds/loads TRT engines instead of aborting on a missing `libnvinfer_builder_resource_sm*.so` | `turbo_nvidia_runtime_paths()` sets `BUILD_RPATH=$TENSORRT_DIR/lib` **and** `-Wl,--disable-new-dtags` (DT_RPATH, not DT_RUNPATH — RUNPATH is not consulted when `libnvinfer` dlopen()s by bare name). |
| 8 | **`turbo_conformance` cpu-vs-nvidia actually runs** | `turbo_conformance --images … --count 10` prints a diff table, exit 0 | Exit **77 (SKIP)** means only one backend is in the binary → go back to #6. A link error about duplicate symbols means `turbo_ocr_cpu_host` (§1a below) overlaps `turbo_ocr_gpu` → report which symbols. |
| 9 | **Accuracy** | `turbo_bench --backend nvidia --tier tiny … --count 50` lands near **85.5 %** F1 | The registered ctest gate `funsd_nvidia_tiny_gate` asserts ≥ **85.0**, a floor taken from the pre-rebuild NVIDIA FUNSD-tiny measurement. **Measured 2026-08-10: 85.37%** — the seam reproduces the pre-rebuild number, so the floor is no longer provisional and stays at 85.0. If a new machine differs, change the floor in the root `CMakeLists.txt` after establishing why — do not delete the gate. |

Two structural things that were decided here and could not be tested here:

* **`turbo_ocr_cpu_host`.** The root's `turbo_ocr_cpu` and `turbo_ocr_gpu` are the
  two arms of one `if(USE_CPU_ONLY)`, so a CUDA configure has **no
  `turbo_ocr_cpu`** for the `cpu` backend to wrap — and without it
  `turbo_conformance` cpu-vs-nvidia can never exist in one binary. The root
  therefore builds the 8 CPU-only TUs the cpu backend actually reaches
  (`cpu_engine`, `cpu_paddle_{det,rec,cls,layout}`, `cpu_doc_orientation`,
  `cpu_formula_recognizer`, `cpu_slanext_table`) as `turbo_ocr_cpu_host`, with
  `USE_CPU_ONLY`/`TURBO_CPU_ONLY` **private** to those TUs. The 9 TUs that
  `turbo_ocr_cpu` and `turbo_ocr_gpu` *both* compile (`ort_session`,
  `formula_tokenizer`, `latex_normalize`, `ppformulanet_preprocess`,
  `slanext_{host_decode,dict,postprocess}`, `cell_matcher`, `html_reconstruct`)
  are deliberately **not** rebuilt — they come from `turbo_ocr_gpu`, whose
  `ort_session.cpp` is a strict superset. A duplicate-symbol link error naming
  any of those nine is this decision being wrong; the fallback is two build dirs
  (see below).
* **The seam no longer defines `USE_CPU_ONLY`/`TURBO_CPU_ONLY` when
  `USE_CPU_ONLY=OFF`.** On a CUDA configure both are actively wrong:
  `TURBO_CPU_ONLY` compiles the CUDA execution provider out of `OrtSession`
  (`ort_session.h:40`), which makes `src/backends/nvidia/engine/ort_cuda_engine.cpp:29`
  return `false` unconditionally. Nothing changes for the cpu/apple builds, which
  always pass `-DUSE_CPU_ONLY=ON`. Verified on the Mac: the `cpu;apple`
  configure still puts both defines on every `src/backends/cpu` TU.

**Fallback if `cpu;nvidia` in one binary cannot be made to link.** Configure two
build dirs — `-DTURBO_BACKENDS="nvidia"` (GPU arm) and `-DUSE_CPU_ONLY=ON
-DTURBO_BACKENDS="cpu"` — and compare `turbo_bench` **metrics JSON** from each.
You keep the accuracy and throughput comparison and lose only
`turbo_conformance` / `turbo_golden`, which are single-binary by construction.

**Also skipped in a GPU configure:** `turboocr-server`.
`src/service/server/unified/unified_server.cmake` links `turbo_ocr_cpu` and the main tree's
`TURBO_HTTP_CPU_SRCS` by name, so it is a CPU-configure target as written;
`the root CMakeLists.txt` now skips it with a `STATUS` message instead of failing
at generate time. The test harness is unaffected.

> Check the link worked before anything else:
> ```bash
> ./build/turbo_backend_probe --list   # must list cpu AND nvidia
> ```

---

## 2. Run the identical tests on your box

All commands run **from the repo root** (model paths are relative to it).

### 2a. Accuracy (the FUNSD gate)

```bash
# NVIDIA
./build/turbo_bench --backend nvidia --tier tiny \
    --images ~/compare-ocrs/funsd_cache --count 50 \
    --words /tmp/nv_tiny.words.json --out /tmp/nv_tiny.metrics.json

python3 tools/bench/score_funsd.py /tmp/nv_tiny.words.json \
    --metrics /tmp/nv_tiny.metrics.json --assert-f1 85.2
```

**What to compare that number against** (full table in §3):

| baseline | value | how to read it |
|---|---|---|
| cpu tiny | **85.79 %** | **exact and deterministic** — the harness reproduces it to the last digit with `DISABLE_COREML=1`. If your box's cpu backend prints anything else, fix that before believing any nvidia number. |
| cpu medium | **92.56 %** | same: exact, deterministic. |
| apple tiny | **85.44 %** F1 @ ~**103.8 img/s** (K=16) | F1 is bit-stable; **the throughput is worth ±30 %**. Three runs of the identical command measured 103.8 → 71.1 → 93.6 img/s (`ANECompilerService` contending for a core). Never compare a single Apple img/s figure against a single NVIDIA one — see §5.4/§5.6. |

The NVIDIA pipeline recorded **~85.5 %** on FUNSD tiny before the rebuild; the
rebuild backend *wraps* those same `PaddleDet`/`PaddleRec` classes, so landing
materially away from 85.5 means the seam changed behaviour, not that the models
did.

Repeat with `--tier medium` (and `--tier small`) for the other two gates.
`--tier` selects `models/{det,rec}[_tiny|_small].onnx` + the matching keys file;
`--det/--rec/--keys/--cls` override individually.

### 2b. Throughput (with its accuracy, always)

```bash
./build/turbo_bench --backend nvidia --tier tiny \
    --images ~/compare-ocrs/funsd_cache --count 50 \
    --threads 16 --repeat 40 \
    --words /tmp/nv_tp.words.json --out /tmp/nv_tp.metrics.json
```

`--threads K` builds K independent `UnifiedOcrPipeline` replicas (one stage set +
one `DeviceQueue` each — the production pool shape). `--repeat R` runs the set R
times so the **timed window is ≥ 15 s**; the harness refuses to publish a shorter
one. Model load and warmup are always **outside** the timed window.

### 2c. Cross-backend conformance (the keystone)

```bash
./build/turbo_conformance --images ~/compare-ocrs/funsd_cache \
    --count 20 --out /tmp/conformance_nv.json
```

Runs the same pages through **every backend in the binary** and diffs them
against the CPU reference: box match rate, mean IoU, per-line text agreement,
pages identical, plus a table of the first disagreements. On a single-backend
build it exits 77 (**SKIP**), never a failure.

### 2d. Per-stage golden diff

```bash
./build/turbo_golden --backend nvidia --ref cpu --stage all \
    --images ~/compare-ocrs/funsd_cache --count 10
```

`--stage det|cls|rec|all`. `cls` and `rec` are fed the **reference backend's
boxes**, so a disagreement is that stage's own, not detection leaking downstream.

### 2e. Everything through ctest

```bash
cd build-rebuild && ctest -R "rebuild_" --output-on-failure
```

Registered tests: `backend_probe`, `backend_conformance`,
`golden_<backend>_{det,cls,rec}` for every **non-cpu** backend in the
binary (cpu is the reference, so `golden_nvidia_{det,cls,rec}` appear
automatically on the GPU box), and per-(backend, tier)
`funsd_{cpu_tiny,cpu_medium,apple_tiny,nvidia_tiny}_run` →
`..._gate` pairs (the `_run`
produces the transcript + metrics JSON, the `_gate` feeds both to
`tools/bench/score_funsd.py --assert-f1` and **fails the build below the floor**).
The FUNSD tests require `-DTURBO_FUNSD_CACHE=<dir>`. Only the gates whose
backend is in `TURBO_BACKENDS` are registered.

**On the GPU box the expected first `ctest -R rebuild_` result is not "all
green".** `funsd_nvidia_tiny_gate`'s 85.0 floor is provisional and
`golden_nvidia_*` has no measured baseline at all — see §1a #9.

---

## 3. Numbers measured on this machine — compare against these

Machine: **Apple M3 Max**, macOS 25.4 (Darwin), `arm64`.
Set: FUNSD 50 test pages (`~/compare-ocrs/funsd_cache`), GT
`tests/benchmark/funsd_gt_words.json`, metric = mean per-page bag-of-words F1.

| backend | tier | F1 | throughput | conditions |
|---|---|---|---|---|
| cpu | tiny | **85.79 %** | 5.1 img/s (K=1) | `DISABLE_COREML=1` (ORT CPU/MLAS), cls on |
| cpu | small | **90.97 %** | — | same |
| cpu | medium | **92.56 %** | 0.3 img/s (K=1, machine shared) | same |
| apple | tiny | **85.72 %** | 19.7 img/s (K=1) | MPSGraph exports, cls on, 9-bucket rec ladder |
| apple | tiny | **85.72 %** | 83.7 img/s (K=16, 23.9 s window) | 9-bucket ladder, CPU job running alongside |
| apple | tiny | **85.44 %** | **103.8 img/s** (K=16, 19.3 s window) | `TURBO_APPLE_REC_BUCKETS=320,480,800,1600`; GPU median 97 % |
| apple | tiny | 85.44 % | 71.1 img/s (K=16, back-to-back rerun) | *identical command*; GPU 93-100 %, ANE compiler idle |
| apple | tiny | 85.44 % | 93.6 img/s (K=16, after an idle gap) | *identical command*; GPU min 44 %, **ANE compiler at 98 %** |

The CPU numbers are **exact-match gates** — the harness reproduces 85.79 / 92.56
to the last digit. **103.8 img/s @ 85.44 % F1 with the 4-bucket ladder is the
throughput reference** and it reproduces exactly (the earlier hand-measured
figure was 103).

Look at the last three rows: **three runs of the identical command spread
103.8 → 71.1 → 93.6 img/s (±30 %) while F1 stayed bit-stable at 85.44 %.** The
saturation instrumentation explains, not excuses, the spread — the GPU was at
93-100 % median in all three, and the slowest-recovering run had
`ANECompilerService` pegging a core at 98 % *inside* the window with device
utilization dipping to 44 %. Consequence: **a single Apple throughput number is
worth ±30 %.** Quote it with its window, its utilization and its ANE-compiler
state, and make every comparison with `--ab`.

### Cross-backend baselines (what `turbo_conformance` / `turbo_golden` measured)

cpu (reference) vs apple, FUNSD pages 0-9, tiny tier:

| check | measured | ctest tripwire |
|---|---|---|
| box match rate (IoU ≥ 0.5) | 96.57 % | ≥ 90 % |
| mean IoU of matched boxes | 0.9357 | ≥ 0.88 |
| per-line exact text agreement | 73.08 % | ≥ 65 % |
| pages with identical text sets | 0 / 10 | — |
| golden `det` agreement / IoU | 0.9625 / 0.9283 | 0.93 / 0.88 |
| golden `cls` flip agreement | 0.9881 | 0.98 |
| golden `rec` exact-string agreement | 0.7223 | 0.65 |

Read that carefully: **two backends can disagree on ~27 % of individual lines and
still land 0.07 pt apart on F1** (85.79 vs 85.72). The disagreements are
per-character (`SPECIAL`/`SPECLAL`, `FAX NO.`/`FAXNO.`) and go in both
directions — device bilinear resampling, not a bug. The thresholds are
**tripwires for a backend that has started to diverge**, not equality assertions;
F1 remains the accuracy gate of record.

### Env that changes these numbers (set them, or your comparison is invalid)

| var | why |
|---|---|
| `DISABLE_COREML=1` | CPU backend: forces ORT CPU/MLAS. The CoreML EP is not deterministic across macOS versions and moves F1. |
| `TURBO_APPLE_REC_BUCKETS=320,480,800,1200,1600,2000,2500,3200,4000` | `MpsRecognizer` auto-discovers **every** `rec_b*` export dir on disk (42 exist). Unpinned you silently get the 42-bucket ladder: +0.15 pt F1 for **2× slower**. |
| `TURBO_APPLE_PROFILE` **unset** | the profiler takes a global mutex per scope and distorts high-K throughput. |
| idle gap before a run | `ANECompilerService` can still be pegging a core from the *previous* run; the metrics JSON records its `%CPU` so you can see when it was. |
| `TURBO_APPLE_METALLIB` | only needed if the metallib is not next to the binary. |

Every one of these — and anything matching `TURBO_*`, `OCR_*`, `ORT_*`, `CUDA_*`,
`TRT_*`, `HIP_*`, `ZE_*`, `OMP_*` — is recorded in the metrics JSON, so a
disagreement between two boxes can be traced instead of argued about.

### NVIDIA / AMD / Intel have never been run

Say it plainly: **no number in this repo for the NVIDIA, AMD or Intel rebuild
backends has ever been measured, and as of this commit the NVIDIA CMake target
has never even been configured** (§1a). `src/backends/nvidia/` is a wrapper of the
existing main-tree Paddle* classes and *should* reproduce the main tree's
~85.4 % tiny F1; `src/backends/amd/` and `src/backends/intel/` are compile-only scaffolds
with no CMake target at all (`TURBO_BACKENDS` still `FATAL_ERROR`s on them).
These tests are
how you find out — and `turbo_conformance` is how you find out *where* a backend
went wrong rather than just *that* it did. The known divergences it is designed
to catch (all found by hand, one backend at a time, before it existed) are listed
in the header comment of `turbo_conformance.cpp`.

---

## 4. What makes two machines' numbers comparable

`--out metrics.json` writes provenance next to every result:

```json
"provenance": {
  "hostname": "...", "os": "Darwin 25.4.0 arm64", "chip": "Apple M3 Max",
  "hw_concurrency": 16, "backend": "apple", "device": "metal",
  "recommended_pool_size": 8, "available_backends": ["apple","cpu"],
  "models": { "models/det_tiny.onnx": {"sha256":"193bab7a...","bytes":1780590} },
  "env": { "DISABLE_COREML": "1", "TURBO_APPLE_REC_BUCKETS": "..." },
  "images_sha256": "ab699c47...", "threads": 16, "repeat": 40, "tier": "tiny"
}
```

Model artefacts are SHA-256'd — **directories too** (Apple MPSGraph export dirs,
TensorRT plan dirs hash as the sorted (relpath, filehash) list), so "we ran the
same weights" is verifiable rather than assumed. `images_sha256` proves both
machines scored the same pages.

---

## 5. Measurement discipline (enforced, not documented-and-ignored)

These were all learned the hard way; they are in `harness.h` so no future harness
can quietly drop one.

1. **Windows ≥ 15 s.** A short window is dominated by model load and graph JIT —
   that is how a fabricated **288 img/s** reading was once produced. Throughput
   runs below 15 s exit non-zero unless `--allow-short-window`. A single-pass
   accuracy run is explicitly labelled "indicative" instead.
2. **Wall-clock cross-check.** Two independent clocks time the *same* region: one
   `steady_clock` span, and the summed per-image latencies ÷ threads. > 5 %
   disagreement means the window contains work that is not per-image OCR, so the
   rate is rejected. This is the check that caught the 288 artifact.
3. **Never a throughput number without its accuracy.** Every run scores its own
   transcript in-process (same metric as `tools/bench/score_funsd.py`, verified to
   agree to 0.0000 pt) and prints both together.
4. **Thermal drift is real.** Absolute throughput drifts ~12 % downward over a
   long session. A cross-session or cross-machine raw img/s comparison is
   untrustworthy — use interleaved paired mode:
   ```bash
   ./build/turbo_bench --ab cpu,apple --tier tiny \
       --images ~/compare-ocrs/funsd_cache --count 50 --repeat 3
   ```
   which runs A,B,A,B,… and reports the **ratio**, the only thing that survives
   drift.

5. **Device saturation is sampled during the window** and printed next to the
   rate, because a throughput number alone cannot tell *"this is the hardware
   limit"* from *"the device idled waiting on the host"*:
   ```
   device utilization    : median 97.0%  (min 93.0, max 100.0, n=24)  -> DEVICE-BOUND (saturated)
   ANECompilerService CPU: 0% (no ANE compilation during the window)
   ```
   Source per platform: `ioreg -r -c IOAccelerator` → `Device Utilization %`
   (Apple, **no root needed**), `nvidia-smi --query-gpu=utilization.gpu`
   (NVIDIA), `rocm-smi --showuse` (AMD). It lands in the metrics JSON under
   `saturation`.

   > **Two traps, both already paid for:**
   > * ioreg prints the entire `PerformanceStatistics` dict on ONE line. Read the
   >   number after `"Device Utilization %"=` specifically — grepping the line
   >   and taking the next `=` yields `lastRecoveryTime`=1029878500420, a 1e12
   >   "utilization" that still passes a ≥90 % saturation test.
   > * **ioreg's `ane0` "busy (N ms)" is NOT ANE utilization.** It is IOKit
   >   device-matching state; it does not move under ANE load and reading it as
   >   utilization produces a false "ANE at 0 %". There is no user-space ANE
   >   utilization counter, so this harness reports **no ANE utilization field at
   >   all** (`"ane_utilization_pct": null`) rather than a wrong one.

6. **ANE program compilation is a contended host resource, not a warmup cost.**
   `ANECompilerService` pegs a full CPU core and can run *during* a benchmark, so
   its `%CPU` is sampled and reported. Two consequences you must respect:
   * model load, `warmup()`, and ANE program compilation must be **outside** the
     timed window — the harness always excludes load and warmup, and flags the
     window when the compiler was active inside it;
   * **back-to-back long runs are not comparable to short ones.** Measured here,
     same binary, same models, bit-stable F1 (85.44 %): **103.8 → 71.1 → 93.6
     img/s** across three K=16 runs of the identical command, device at 93-100 %
     median utilization throughout, and `ANECompilerService` at 98 % inside the
     third window. A previously observed sustained run degraded to 55.1 img/s.
     This is *not* simply thermal drift — leave an idle gap between runs, check
     the `saturation` block, and make comparisons only with `--ab`.

### Proving the cross-check works

```bash
# (a) too-short window -> rejected
./build/turbo_bench --backend cpu --tier tiny \
    --images ~/compare-ocrs/funsd_cache --count 3 --repeat 2 \
    --assert-throughput 1 ; echo "exit=$?"     # exit=4, WINDOW TOO SHORT

# (b) untimed work inside the window -> rejected
#     --selftest-skew-ms injects wall time no per-image latency accounts for,
#     i.e. exactly the shape of "model load landed inside the measurement".
./build/turbo_bench --backend cpu --tier tiny \
    --images ~/compare-ocrs/funsd_cache --count 50 --repeat 2 \
    --selftest-skew-ms 20000 ; echo "exit=$?"  # exit=4, WALL-CLOCK CROSS-CHECK FAILED
```

---

## 6. Backward compatibility

The retired CLI still works, because existing gates and notes reference it:

```bash
# legacy positional form, accepted verbatim by turbo_bench
./build/turbo_bench ~/compare-ocrs/funsd_cache 50 out.json \
    --det models/det_tiny.onnx --rec models/rec_tiny.onnx --keys models/keys_tiny.txt

# and the old binary NAMES exist as shims in the build dir
./build/funsd_unified_cpu        ~/compare-ocrs/funsd_cache 50 out.json
./build/funsd_unified_apple      ~/compare-ocrs/funsd_cache 50 out.json
./build/funsd_unified_apple_conc ~/compare-ocrs/funsd_cache 50 out.json --repeat 40
```

The shims are generated by `the root CMakeLists.txt` and are one line each:
`exec turbo_bench --backend <vendor> --threads <K> "$@"`.

`tools/bench/score_funsd.py` also kept its positional form (`score_funsd.py preds.json`)
while gaining `--gt`, `--metrics`, `--assert-f1` and `--assert-throughput`; it no
longer hardcodes an absolute repo path, so it runs on your box.

---

## 7. Files

| file | role |
|---|---|
| `harness.h` | the ONE shared toolbox: args, JSON, SHA-256, provenance, image set, F1, timing discipline, IoU, device upload. Header-only, backend-neutral. |
| `turbo_bench.cpp` | accuracy + throughput, any backend, `--ab` paired mode |
| `turbo_conformance.cpp` | cross-backend text/box diff |
| `turbo_golden.cpp` | per-(backend, stage) golden diff vs the CPU reference |
| `vlm_factory_link_support.cpp` | nullptr stubs for the two seam VLM factories, so these offline binaries never link drogon/libcurl |

Only `default_models()` in `harness.h` mentions a backend by name, and it is a
**table of paths**, not logic: adding NVIDIA means adding rows, never editing a
driver.

### The one thing that was dropped, and why

`cls_golden_apple.mm` compared **three** paths: (A) Metal warp + MPSGraph,
(B) `cv::warpPerspective` crops + MPSGraph, (C) `cv::warpPerspective` crops +
ORT. The A-vs-B leg isolated *Metal resampling error* from *graph error* by
hand-feeding host crops into a Metal buffer — it can only be written against
Metal types, cannot exist for NVIDIA or AMD, and is therefore a per-backend fork
by definition. `turbo_golden --stage cls` keeps the leg that matters and that
generalises (A vs C: end-to-end flip agreement vs the CPU reference, which is
what both historical cls bugs tripped). If a future Metal-internal
resample-vs-graph question comes up, that is a *vendor debugging* exercise for
`src/backends/apple/`, not a test every backend has to carry.
