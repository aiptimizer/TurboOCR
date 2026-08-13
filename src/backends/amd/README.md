# TurboOCR — AMD backend (`turbo_ocr_amd`)

A **device-resident** OCR backend for AMD GPUs: **HIP** for pre/post kernels and
**MIGraphX** for inference. It mirrors the NVIDIA reference (TensorRT + CUDA
kernels + per-stage streams) one-to-one, swapping the vendor layer while keeping
data on the GPU end-to-end. It is **not** a CPU fallback with a swapped execution
provider — the image, normalized tensors, model logits, DB post-process, warp
batches, and argmax all live in HIP device memory; the host sees only the small
stage results (DB boxes, argmax indices for CTC collapse, the cls flip decision).

> ## ⚠ NEVER RUN ON AMD HARDWARE — read this first
> **Compiled and linked, never executed on an AMD GPU.** As of 2026-08-02 the
> full stack (backend library incl. all five `.hip` kernels for gfx942+gfx1100,
> MIGraphX engine against the real ROCm 7.1.1 headers, server, bench, golden,
> conformance, unit tests) builds clean in a `rocm/dev-ubuntu-24.04:7.1.1`
> container, the registrar registers, and the factory declines gracefully with
> no device. The argmax kernel additionally PASSES its tie-break contract
> executed on CPU threads via ROCm's HIP-CPU library. Everything else —
> kernels on a real wavefront, MIGraphX eval, accuracy, performance — is
> unmeasured, and any number quoted for it would be fabricated.
> Bring-up procedure: **[BRINGUP.md](BRINGUP.md)** (time-boxed for an
> hourly-billed box). It targets **Linux + ROCm** only. On-hardware TODOs are
> marked inline with `TODO(on-hardware)`.

---

## File layout

One directory per concern — the same layout every vendor under `src/backends/`
uses. The rule each directory answers to is in
[`../README.md`](../README.md), and `python3 src/backends/layout_check.py`
enforces it. Headers are included through the vendor-rooted
path off `-Isrc/backends`, e.g. `#include "amd/queue/hip_queue.h"`.

```
src/backends/amd/
├── README.md                           # this file
├── support/
│   └── hip_check.h                     # HIP_CHECK fail-fast (mirror of CUDA_CHECK)
├── queue/hip_queue.{h,cpp}             # HipStreamQueue/HipEvent : DeviceQueue/DeviceEvent
│                                       #   hipStream/hipEvent record/wait/synchronize
├── memory/hip_allocator.{h,cpp}        # HipAllocator : IDeviceAllocator
│                                       #   hipMalloc/hipHostMalloc/hipMemcpyAsync
├── engine/migraphx_engine.{h,cpp}      # MIGraphXEngine : IEngine (pImpl; hides MIGraphX)
│                                       #   parse ONNX -> compile per shape -> run_async(stream)
├── kernels_hip/                        # hipified CUDA kernels — SAME bodies as src/cuda/kernels/*.cu
│   ├── hip_kernels.{h,cpp}             # HipKernels : IKernels — the device pre/post op set
│   │                                   #   (wrapper + DB post chain + box build)
│   ├── kernels_hip.h                   # public hipified signatures (HipImage/GpuDetBox)
│   ├── kernels_internal_hip.h          # coop_grid_for (hip occupancy sizing)
│   ├── preprocess_kernels.hip          # resize/normalize (det/layout/param-driven) + ROI warp
│   ├── reduce_kernels.hip              # argmax (CTC) + threshold->u8   [WAVEFRONT-AGNOSTIC]
│   ├── ccl_kernels.hip                 # Block-based Union-Find CCL      [COOP-LAUNCH FALLBACK]
│   ├── jfa.hip                         # bounded Euclidean unclip + oriented PCA rects
│   └── table_kernels.hip               # fused region preproc (table cls / SLANeXt / layout sub-rect)
├── stages/rocm_stages.{h,cpp}          # RocmDetector/Recognizer/Classifier/Layout
└── backend/
    ├── rocm_backend.{h,cpp}            # RocmBackend : Backend — factories + load_stages
    └── amd_backend_registry.cpp        # registers "amd" in backend::make_backend()
```

The build lives in the root `CMakeLists.txt` (`turbo_ocr_backend_amd`, enabled
by `-DTURBO_BACKENDS=cpu;amd`); there is no per-directory CMakeLists and no
`build.sh` (invocations of it further down this file are stale). Three
non-obvious build facts, each found the hard way on the first real configure:
`find_package(migraphx)` transitively needs the MIOpen/rocBLAS dev cmake
configs; the host TUs link `hip::host` (NOT `hip::device`, which injects
`-x hip` into g++ TUs); and IPO/LTO is auto-disabled for amd configures because
ld.lld cannot read GCC slim-LTO archive members (the symptom is a wall of
"undefined symbol" for symbols that nm plainly shows).

### What compiles where (toolchain)

| Sources                          | Compiler        | Needs                                   |
|----------------------------------|-----------------|-----------------------------------------|
| `kernels_hip/*.hip`              | `hipcc` / HIP-lang (amdclang++) | ROCm HIP runtime, `__HIP_PLATFORM_AMD__` |
| everything else (`*.cpp`)        | host C++20 (amdclang++/clang/gcc) | HIP headers (host API), OpenCV, common  |
| `engine/migraphx_engine.cpp`     | host C++20      | **MIGraphX** dev package (only this TU sees it) |
| `turbo_ocr_common` (shared)      | host C++20      | device-free; provides interfaces + geometry |

- The **`.hip` kernel bodies are byte-for-byte the CUDA math** from
  `src/cuda/kernels/*.cu`. Only the runtime surface changed: `cuda_runtime.h ->
  hip/hip_runtime.h`, `cudaStream_t -> hipStream_t`, `cuda*` host API `-> hip*`,
  `cudaLaunchCooperativeKernel -> hipLaunchCooperativeKernel`,
  `cooperative_groups -> hip_cooperative_groups`. `uchar3/float3/make_*`, `__ldg`,
  `atomicCAS/Min/Max/Add`, `__shfl_down_sync`, and device math all exist in HIP.
- **MIGraphX is confined to `migraphx_engine.cpp`** via pImpl, exactly how the
  NVIDIA `OrtSession` hides ORT from nvcc TUs — stage/backend TUs never include
  `<migraphx/migraphx.hpp>`.
- **Build the .onnx models** the same ones the other backends use; MIGraphX
  parses ONNX directly (no separate plan build like TensorRT), but see per-gfx
  compile below.

---

## How each interface is satisfied

- **`IEngine` → `MIGraphXEngine`** — `parse_onnx -> compile(target("gpu"),
  offload_copy=false) -> run_async(params, hipStream)`. `offload_copy=false` is
  the device-resident lever: inputs are the caller's `hipMalloc` pointers (bound
  zero-copy as `migraphx::argument`), outputs come back as engine-owned **device**
  arguments surfaced through `OutputLease` (Hip space, valid until the next
  `run()` — the same lifetime as `CpuEngine::infer_batch_view`, on device).
  `caps()` reports `io_space=Hip, async=true, caller_owns_outputs=false,
  multi_io=true, dynamic_shapes=true, thread_safe_concurrent=false`.
  **Shape ladder (performance gate):** MIGraphX compiles for concrete shapes, so
  the engine keeps a **cache of compiled programs keyed by input shape**.
  `warmup(variants)` compiles the whole ladder at load; `run()` is a map hit. A
  cache miss still compiles (correctness first) but logs a loud
  `HOT-PATH COMPILE` line naming the shape — that line is a bug report, not
  noise. `hot_path_compiles()` must read 0 in a warmed pipeline.
- **`IKernels` → `HipKernels`** — wraps the hipified op set. Native:
  `resize_normalize` (param-driven — takes the caller's `NormParams` rather than
  sniffing which of two baked variants was meant, which is what the CUDA adapter
  still has to do), `warp_crops`, `threshold`, `argmax`, `preprocess_region`
  (all four `PreprocKind`s), and the full `db_postprocess` chain (GPU CCL →
  crack-perimeter → per-component expand → Euclidean unclip → axis-aligned /
  oriented extract). Host-fallback, advertised honestly as `false` in `caps()`:
  `decode_image` only (OpenCV imdecode + H2D; there is no rocJPEG path yet).
- **`DeviceQueue`/`DeviceEvent` → `HipStreamQueue`/`HipEvent`** — 1:1 with CUDA:
  stream = ordered lane, event = cross-lane token, `record/wait/synchronize` =
  `hipEventRecord/hipStreamWaitEvent/hipStreamSynchronize`. `begin/end_batch` is a
  no-op (a hipStream is already one submission lane), like CUDA.
- **`IDeviceAllocator` → `HipAllocator`** — `hipMalloc` / pinned `hipHostMalloc` /
  `hipMemcpyAsync` on the queue's stream.
- **Stage interfaces → `RocmDetector`/`RocmRecognizer`/`RocmClassifier`/
  `RocmLayout`** — each owns one `MIGraphXEngine`, borrows the entry's
  `HipKernels`+`HipAllocator`, and runs the device-resident flow, returning HOST
  types (`vector<Box>`, `vector<pair<string,float>>`, `int` flip count,
  `vector<LayoutBox>`). Detection = resize→forward→threshold→CCL+unclip→rescale.
  Recognition = per-box inverse-warp batch→forward→argmax→greedy CTC collapse.
  Classification = warp→forward→argmax→180° quad flip. Layout = multi-IO
  (image+im_shape+scale_factor)→forward→parse rows.
- **`Backend` → `RocmBackend`** — the factories (`make_queue/allocator/kernels/
  engine`), `load_stages(cfg)` (builds+loads the four stages, fills
  `StageAvailability`), the table/formula registry dispatch, and the service fns.
  `make_rocm_backend()` is the vendor entry the shared `make_backend("amd")` calls.

---

## What lives here vs. what is SHARED (dedup audit)

The plan's overriding rule is *"generic policy is SHARED; only device mechanics
are per-backend."* This directory is audited against it:

**Deliberately NOT here — consumed from the shared layer:**

| Concern | Shared owner | Used by |
|---|---|---|
| Detection resize policy (`limit_type`/`limit_side_len`/`max_side_limit`, `/32` rounding, `DET_*` env overrides) | `detection::read_det_resize` + `compute_det_resize` (`det_config.h`) | `RocmDetector::run` |
| DB thresholds (binarize / box score / unclip ratio) | `detection::read_db_params` (`det_config.h`) | `RocmDetector::run` |
| Rec crop width + width buckets | `recognition::rec_input_width`, `kRecWidthBuckets`, `kMaxRecWidth`, `kMinRecWidth` (`rec_geometry.h`) | `RocmRecognizer` |
| Rec routing, batch rungs, chunking | `recognition::plan_rec_batches` / `group_by_width_bucket` / `snap_batch` / `batch_ladder_for_width` (`rec_batching.h`) | `RocmRecognizer`, `RocmClassifier` |
| The (width, batch) shape matrix to pre-build | `recognition::rec_shape_matrix` (`rec_batching.h`) | `RocmRecognizer::load` warmup |
| Character dictionary layout (blank-at-0 + trailing space) | `recognition::load_label_dict` (`ctc_decode.h`) | `RocmRecognizer::load_dict` |
| Greedy CTC collapse | `recognition::ctc_greedy_decode` (`ctc_decode.h`) | `RocmRecognizer::run` |
| Crop perspective transform | `turbo_ocr::compute_crop_transform` (`perspective.h`) | rec + cls |
| det→cls→rec→layout→router orchestration | `pipeline::UnifiedOcrPipeline` + `pipeline::make_infer_func` | — (this backend deliberately has **no** `make_infer_func` override) |
| Table / formula recognizer selection | `table::make_table_recognizer`, `formula::make_formula_recognizer` | `RocmBackend` delegates |

**Legitimately here — genuinely device mechanics:**

| File | Why it must be per-backend |
|---|---|
| `kernels_hip/*.hip` | GPU kernels; there is no portable form of CCL/JFA/warp |
| `engine/migraphx_engine.cpp` | MIGraphX is the AMD inference runtime |
| `queue/hip_queue.cpp`, `memory/hip_allocator.cpp` | `hipStream_t`/`hipEvent_t`/`hipMalloc` glue behind `DeviceQueue`/`IDeviceAllocator` |
| `kernels_hip/hip_kernels.cpp` | translates `ImageView`/`NormParams` ↔ HIP kernel arguments |
| `stages/rocm_stages.cpp` | binds engine + kernels; holds **no** policy of its own |
| `backend/rocm_backend.cpp`, `backend/amd_backend_registry.cpp` | factories + vendor registration |

Note what is **absent by design**: no `make_infer_func`, no private width
ladder, no private CTC loop, no `set_max_side()`/`set_db()` knobs on the
detector. Every one of those would be a place for AMD to silently drift from the
other backends — which is exactly how the Apple rec-ladder clamping bug happened.

---

## What is and is not verified

**Verified 2026-08-02 in a ROCm 7.1.1 container (x86_64, no GPU):** full
configure + compile + link of every target with `TURBO_BACKENDS=cpu;amd`
(three from-documentation MIGraphX API errors fixed against the real headers:
`run_async` is a template over the stream type, `migraphx::arguments` is not
default-constructible); `turbo_backend_probe` lists amd and its factory
declines cleanly with no device; the argmax tie-break unit test
(`tests/cpp/backends/test_argmax_tiebreak.cpp`) passes on the host reference
and SKIPs without a HIP device; and the REAL `argmax_kernel` passes the same
tie rows executed via HIP-CPU (risk #1's fix is functionally confirmed —
hardware wavefront behaviour still pending).

**Verified on the dev machine (an M3 Max Mac, no ROCm):**

- `build.sh` is syntactically valid bash and its `--syntax-only-host` mode runs.
- The **shared-helper call sites** this backend depends on type-check: a
  host-only harness reproducing every `det_config` / `rec_batching` /
  `rec_geometry` / `ctc_decode` / `perspective` call from `rocm_stages.cpp`,
  with the same argument types, compiles clean under
  `c++ -std=gnu++20 -Wall -Wextra`. This is the check that matters most for the
  dedup work, because wrong shared-API usage is the likeliest compile break.
- The warmup ladder's size is a computed fact, not a guess: `rec_shape_matrix`
  over `kRecWidthBuckets` at `rec_image_h=48` yields **35** (width, batch)
  variants; the cls ladder at 160×80 yields **6**; layout adds **1**. So a full
  warm start is **42 MIGraphX graph compiles**. See "startup cost" below.

**NOT verified — requires a ROCm host:**

- Nothing in `kernels_hip/` has been compiled. hipcc is not installed here.
- Nothing in `src/` has been compiled: every TU includes `<hip/hip_runtime.h>`
  (directly or via `hip_check.h` / `hip_queue.h`) or `<migraphx/migraphx.hpp>`.
  There is **no** subset of this backend that a non-ROCm machine can compile,
  and `build.sh --syntax-only-host` says so rather than pretending otherwise.
- The MIGraphX API surface used (`onnx_options::set_input_parameter_shape`,
  `compile_options::set_offload_copy`, `program::run_async(..., "ihipStream_t")`,
  `argument::data()` lifetime) is written from documentation. Exact spellings
  move between ROCm versions; expect small fixes on first build.
- **Zero** throughput or accuracy measurements exist. Any figure quoted for this
  backend is fabricated.

---

## Bring-up plan on real AMD hardware (in this order)

**Stage 0 — it builds.** `ROCM_PATH=/opt/rocm HIP_ARCHS=<your gfx> ./build.sh`.
Fix MIGraphX packaging/API spellings. Nothing below is meaningful until this
passes.

**Stage 1 — per-stage golden diff vs the CPU backend.** This is the gate, and it
comes before any speed work. Fix an image set (start with 20 FUNSD pages), run
the CPU backend and the AMD backend over the same inputs through the same
`UnifiedOcrPipeline`, and diff **stage by stage**, because a whole-pipeline diff
cannot tell you which stage is wrong:

1. `resize_normalize` — dump the det input tensor from both; expect
   near-bit-equality (fp32, same bilinear form). A large diff means the fp
   contraction or the coordinate convention differs.
2. `threshold` + `db_postprocess` — compare the returned box lists. **This is
   where the two AMD-specific correctness risks land** (see below). A missing or
   duplicated component here is a CCL bug, not a model difference.
3. `warp_crops` — dump one rec batch tensor; compare against the CPU
   `warpPerspective` output.
4. `argmax` — compare indices AND scores against the CPU argmax on identical
   logits. **Assert the tie-break**: on equal values the LOWER class index must
   win. Construct a synthetic logits row with an exact tie to test it; a natural
   corpus will not hit it reliably.
5. `preprocess_region` — dump the SLANeXt/TableCls tensors and diff against the
   CUDA/CPU reference. These feed the table model directly.
6. `ctc_greedy_decode` — should be identical by construction (it is the shared
   host function), so a text difference here means the argmax above is wrong.

**Stage 2 — end-to-end accuracy.** Only once every stage diff is clean: run
FUNSD through `UnifiedOcrPipeline` over `RocmBackend`, score with
`tools/bench/score_funsd.py`. The target is the shared-pipeline CPU number on the same
models (tiny ≈ 85.5% F1) — AMD is running the same ONNX through a different
runtime, so a materially lower score means a stage bug, not a device difference.
Re-run with `set_fp16(false)` if FP16 is suspected; fp16 det/rec is an
unvalidated throughput lever here, not a default to trust.

**Stage 3 — throughput.** Last. Report **throughput and F1 together, always**;
a speed number without its accuracy is meaningless. Check, in this order:
`hot_path_compiles()` on every engine (must be 0), then whether cooperative
launch was available (a fallback to the two-kernel compaction is correct but
costs one extra launch per image), then per-stage timings.

---

## Known AMD-specific correctness risks and how they are handled

**1. Wavefront width 64 vs 32 — HANDLED, was the real trap.**
`argmax_kernel` (the CTC reduction) originally inherited CUDA's hand-rolled
"warp-synchronous" tail: an unsynchronized read of `s_vals[tid+32]` for
`tid < 32`, then `__shfl_down_sync(0xffffffff, v, offset)` with offsets
{16,8,4,2,1}. Both halves hard-code `warpSize == 32`, and on AMD that is wrong
in two independent ways:
  * the "no `__syncthreads()` needed, we're in one warp" claim is a statement
    about the hardware wave width. It happens to hold for lanes 0–31 of a
    **wave64** CDNA wavefront, and is **false on wave32 RDNA**, where lanes 0–31
    and 32–63 are different waves and the read races;
  * HIP ignores `__shfl_down_sync`'s mask — all active lanes participate and the
    width defaults to `warpSize`, so on wave64 lanes 32–63 shuffle their
    never-combined partials into lanes 0–31.
The two errors cancel by accident on wave64 (the combine only ever keeps a
larger value), which is precisely why it is dangerous: it would pass a smoke
test on an MI250 and fail on an RX 7900.
**Fix applied:** the hand-rolled tail is gone. The reduction is now a pure
shared-memory tree down to `stride == 1` with a `__syncthreads()` at every level
— correct for *any* wavefront width, no `warpSize` query, no shuffle. Cost is 5
extra barriers per timestep, negligible against the `num_classes`-wide load
loop. Tie-breaking (lower class index wins, matching
`src/analysis/recognition/ctc_decode.cpp`) is written to be order-independent, so
wavefront width cannot change the answer. **Verify with the synthetic tie test
in Stage 1.4.**

**2. Cooperative launch — HANDLED.** The CCL compaction used `grid.sync()` via
`hipLaunchCooperativeKernel`. Several consumer RDNA parts and older ROCm stacks
report `hipDeviceAttributeCooperativeLaunch == 0`; there, the launch simply
fails and **the detector silently returns no text**. There are now two
implementations — the cooperative kernel, and `ccl_buf_compact_assign_pass1` /
`_pass2` launched back-to-back on the same stream, where the launch boundary
*is* the grid-wide barrier (no `hipStreamSynchronize`, no host round-trip). The
host wrapper picks one from a cached device-attribute probe.

**3. Component-budget overflow — FIXED (was a real out-of-bounds).**
`hip_gpu_ccl_detect` returns the **raw** `id_counter`, which keeps incrementing
past `kMaxGpuComponents` (the kernel only clamps the ids it *stores*). That
value was being used directly as `N` to size the JFA extract and the final
`hipMemcpyAsync` out of scratch arrays sized `kMaxGpuComponents` — an OOB
read/write on any dense page above the budget. `N` is now clamped to the budget,
and to the caller's smaller `DbPostParams::max_components` when set.

**4. LDS float atomics (CCL fused extract) — performance risk only.**
`atomicMin`/`atomicMax` on shared **float** are emulated via CAS on some gfx and
shared-atomic throughput differs markedly from NVIDIA. The algorithm is correct;
if profiling shows the 32-slot LDS hash dominating, replace it with a
wavefront-ballot reduction keyed on `cid`. Same note for the 64-bit integer
moment atomics in the oriented path (`jfa.hip`).

**5. fp-contraction on the region preprocessors.** `table_kernels.hip` is
transcribed verbatim so its fp32 arithmetic matches the CUDA export. amdclang++
contracts to FMA by default like nvcc, but contraction is a compiler choice, not
a guarantee. If Stage 1.5 shows a ulp-class drift, set `-ffp-contract` explicitly
on **both** sides rather than editing the expressions.

---

## Remaining on-hardware TODOs

**Startup cost / caching (the biggest known ergonomics problem):**

1. **Per-gfx `.mxr` compile cache.** A warm start compiles **42** MIGraphX
   programs (35 rec + 6 cls + 1 layout), which will take real wall-clock time.
   This is the deliberate trade the performance gate demands — never compile
   during a request — but it should be paid once per fleet, not once per
   process. Persist each compiled program with `migraphx::save()` keyed on
   (model, shape, `gcnArchName`) and load the match. Hook:
   `Impl::compile_variant()` in `migraphx_engine.cpp`.
2. **Detection shape explosion.** Unlike rec, the det canvas is a function of
   the page aspect (`compute_det_resize` gives per-image `/32` dims), so there
   is no small ladder to pre-compile and a mixed corpus triggers one hot-path
   compile per new canvas (each logged). Preferred fix: MIGraphX **dynamic
   dimensions** for the det input, giving one program for all canvases; failing
   that, the `.mxr` cache above. **Do not** "fix" it by forcing a fixed det
   canvas — the resize policy is shared, and changing it here alone would give
   AMD different recall from every other backend.

**Fidelity / performance:**

3. **FP16.** `set_fp16(true)` is on for det and rec to match the TRT path's
   throughput. It is **unvalidated**: run Stage 2 with it off and on and compare
   F1 before keeping it.
4. **Device image decode.** `decode_image` is an OpenCV host decode + H2D, and
   `caps().decode_image` reports `false` so the shared layer knows. Wire
   rocJPEG/VAAPI to close the last host round-trip.
5. **Doc orientation.** `make_orient_func()` returns empty (autorotate off). A
   `RocmDocOrientation` (MIGraphX 224×224 classifier) mirrors the CUDA
   `DocOrientation`.
6. **Device-resident table/formula backends.** `preprocess_region` is now native,
   so a MIGraphX SLANeXt encoder and a PP-FormulaNet-S AR loop can be registered
   behind the existing `table::`/`formula::` factory keys and stay resident;
   until then those specs fall through to the shared host/VLM path.
7. **Layout output layout.** Confirm the exported PP-DocLayoutV3 ONNX output
   ordering (`[N,6]` vs split boxes/scores/num) and bind by name, and confirm
   the `scale_factor` axis order. `rocm_stages.cpp`.
8. **hipGraph capture in `begin_batch`/`end_batch`.** Optional launch-overhead
   win only — residency is already achieved by the stream. Note that capture
   forbids synchronizing calls inside the region, which today's `db_postprocess`
   violates (it syncs to read the component count); that would have to be lifted
   first. See the long comment in `hip_queue.h`.
