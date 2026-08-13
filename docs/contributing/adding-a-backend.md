# Adding a new GPU vendor backend

How to bring a new accelerator into TurboOCR: what a backend is, what it is
*not*, the shared policy it must call, the order to build it in, and the gates
that decide whether it is done.

There is a generator for the mechanical part:

```bash
python3 tools/new_backend.py --name foo          # see §4.9
```

It emits a **compiling, correct, host-delegated** backend and prints the exact
CMake lines to add. Read §2 before you edit what it emits — the shared-policy
rule is the thing that has broken every backend added to this tree so far.

Every claim below cites `file:line`. Anything unverified is marked
**[UNVERIFIED]**.

---

## 1. What a backend IS

A `Backend` is a **factory**, at the altitude the old `stages_gpu.cpp` /
`stages_cpu.cpp` sat at (`include/turbo_ocr/backend/backend.h:234`). It hands
the shared layer seven things and nothing else:

| Method | Returns |
|---|---|
| `caps()` | `BackendCaps` — what this backend is, on this machine, right now |
| `make_queue()` | one ordered lane of device work (`DeviceQueue`) |
| `allocator()` | device memory (`IDeviceAllocator`, may be a shared singleton) |
| `make_kernels()` | the device pre/post op set (`IKernels`) |
| `make_engine()` | a fresh model runner (`IEngine`) |
| `load_stages(cfg)` | the constructed `StageSet` + its `StageAvailability` |
| `make_image_decoder()` / `make_orient_func()` | the two genuinely device-shaped service functions |

Plus `make_table_recognizer` / `make_formula_recognizer`, which are **registry
dispatch**, not implementations (§4.8).

The interface owns **no vendor SDK type**. Everything device-shaped is behind
`ImageView` / `DeviceQueue` / `IEngine` / `IKernels`
(`include/turbo_ocr/backend/backend.h:16-17`).

### What is NOT a backend's job

Read this list as a set of things you must **delete** if you find yourself
writing them:

* **Orchestration.** There is deliberately no `make_infer_func()` on the
  interface (`backend.h:270-284`). The one det→cls→rec→layout→router flow is
  `pipeline::UnifiedOcrPipeline` + `pipeline::make_infer_func(pool)`. Every
  backend that once overrode `make_infer_func()` carried a private copy of that
  orchestration; NVIDIA's was deleted
  (`src/backends/nvidia/README.md`, bring-up item 1) and AMD/Intel were written
  without one on purpose (`src/backends/amd/README.md`,
  `src/backends/intel/README.md` — "there is deliberately **no**
  `make_infer_func()` override").
* **HTTP/gRPC routing, serialization, request validation.** Those live in
  `src/service/http/`, `src/service/grpc/`, `include/turbo_ocr/service/validation/`. A backend never
  sees a request.
* **Capability policy.** A backend *reports* two of the three capability axes
  (`include/turbo_ocr/core/capability.h:10-21`); it never decides what a
  client is allowed to ask for.
* **Pool management.** `build_backend_runtime` owns the pool
  (`src/service/server/unified/backend_stages.cpp:104-142`); the backend only *suggests* a size
  via `caps().recommended_pool_size`.
* **Model path policy.** `BackendConfig` arrives already resolved
  (`src/service/server/unified/backend_stages.cpp:19-60`). Table/formula bundle paths have their
  own shared resolvers (§2).

The seam is one-directional: **device mechanics below, everything generic
above.**

---

## 2. THE SHARED-POLICY RULE

> **Generic policy is SHARED. A backend may implement only device mechanics.**
> If two backends could ever disagree about a number, that number does not
> belong in either of them.

This is not style advice. Every header listed below exists because a backend
forked that policy and drifted, and the drift was **invisible** — the seam
reported success, the pipeline returned the right *shape* of answer, and only
the F1 moved, on one backend, quietly.

### The list — call these, never re-derive them

| Policy | The ONE header that owns it | Key entry points |
|---|---|---|
| Detection resize (`limit_type`, `limit_side_len`, `/32` rounding, `DET_*` env) | `turbo_ocr/analysis/detection/det_config.h` | `read_det_resize()` :89, `compute_det_resize()` :58 |
| DB thresholds (binarize / box score / unclip) | `turbo_ocr/analysis/detection/det_config.h` | `read_db_params()` :132, `kDbDefaults` :46 |
| DB post geometry limits, canvas sizing | `turbo_ocr/core/db_post_config.h` | `kDbDefaults` note :6 |
| Rec crop width + width buckets | `turbo_ocr/analysis/recognition/rec_geometry.h` | `rec_input_width()` :46, `kRecWidthBuckets` :34, `kMaxRecWidth` :23 |
| Rec routing, batch ladder, chunking, warmup shape matrix | `turbo_ocr/analysis/recognition/rec_batching.h` | `plan_rec_batches`, `batch_ladder_for_width` :64, `snap_batch` :78, `group_by_width_bucket` :86, `rec_shape_matrix` |
| CTC greedy decode + dictionary | `turbo_ocr/analysis/recognition/ctc_decode.h` | `ctc_greedy_decode()` :11, `load_label_dict()` :22 |
| Text-line angle cls geometry, threshold, 180° quad flip | `turbo_ocr/analysis/classification/cls_config.h` | `kClsImageH/W` :36-37, `kClsThresh` :41, `should_flip_180()` :76 |
| Normalization constants | `turbo_ocr/core/norm_params.h` | `norm::rec_norm()`, `cls_norm()`, `imagenet_bgr()`, `imagenet_rgb()` |
| Crop perspective transform | `turbo_ocr/base/geometry/perspective.h` | `compute_crop_transform()` |
| PicoDet (PP-DocLayoutV3) row decode | `turbo_ocr/analysis/layout/picodet_decode.h` | `decode_picodet_rows()` :48, `kPicodetMaxDet` :31 |
| Layout post-filter (NMS, oversized-image drop, containment, merge mode) | `turbo_ocr/analysis/layout/layout_postfilter.h` | `postfilter_layout_boxes()` :150 |
| SLANeXt table post-process + dict | `turbo_ocr/analysis/table/slanext/slanext_postprocess.h`, `slanext_dict.h`, `slanext_host_decode.h` | — |
| SLANeXt model path resolution | `turbo_ocr/analysis/table/slanext/slanext_paths.h` | `resolve_slanext_encoder()` :23 |
| Formula bundle resolution | `turbo_ocr/analysis/formula/formula_bundle_env.h` | `resolve_formula_bundle_env()` :27 |
| Engine-mode policy (Auto fallback / explicit-native error) | `src/backends/cpu/stages/cpu_stages.h` | `resolve_engine_mode()` :151 |
| The whole ONNX ("fast") stage set | `src/backends/cpu/stages/cpu_stages.h` | `make_vendor_onnx_stages()` :157 |
| Host ORT thread policy | `turbo_ocr/onnx/host_ort_threads.h` | `set_host_ort_intra_op_threads()` :44 |
| Host fallback kernels | `src/backends/cpu/kernels_host/host_kernels.h` | `cpu::HostKernels` |
| det→cls→rec→layout→router flow | `include/turbo_ocr/pipeline/unified/unified_ocr_pipeline.h`, `pipeline/make_infer_func.h` | — |

### The scars — what each rule cost before it existed

Read these. They are the argument.

**Normalization retyped in eight places → the same bug shipped three times.**
`norm_params.h:1-18` names them: `intel_stages.cpp`, `host_kernels.cpp`,
`rocm_stages.cpp` (×3), `mps_stages.mm`, `sycl_kernels.cpp` (×2). Three separate
backends fed **ImageNet mean/std to the text-line orientation classifier**,
which is trained on rec's `(x/127.5 − 1)`. Wrong-distribution input to cls means
mis-detected 180° lines means reversed text — on that backend only.
`norm_params.h:31-36` explicitly warns not to "fix" cls to ImageNet: the
regression has been introduced three times (Intel, an Apple variant, AMD).

**AMD's RGB/BGR swap, from retyping the norm params.** The kernels contract
records it first-hand: *"AMD dropped `NormParams::order` (R/B swapped on det)"*
(`include/turbo_ocr/backend/kernels.h:33-37`). The det/table convention is
BGR-positional — plane 0 (B) gets 0.485 — because that is what
`cuda_fused_resize_normalize_det` bakes in, *"so every other backend must match
it or R and B are swapped on that backend alone"*
(`norm_params.h:70-73`).

**The Apple rec-ladder clamp.** A backend-private width/batch ladder squashed
every line wider than the private ceiling. `rec_batching.h:12-18` states the
rule it produced: question 2 (which static batch size) *"used to be answered
per-backend, which is exactly how the Apple ladder-clamping bug happened: a
backend-private ladder drifts from the shared one, and a bug fixed on one path
silently persists on the other."* Intel's README records the same class of
defect found in its own scaffold — hardcoded `kMaxW = 320` with a private crop-
width formula, an inline CTC loop, a hand-rolled `findContours`+unclip DB post,
a private det canvas rule, and the wrong cls normalization and corner flip —
and notes the measured cost of the Apple original: **0.10 pt of F1 on one
backend only** (`src/backends/intel/README.md`, "A defect that was found and
removed").

**The CoreML-NaN guard that lived in one of three decode copies.** The CoreML EP
on ORT 1.24.4 returns NaN for every layout score and box. A NaN score compares
false against the threshold, so every row is dropped and *"a numerically broken
graph or execution provider … produces an empty layout indistinguishable from a
clean page — fast, silent, and wrong."* The guard *"briefly lived only in the CPU
copy, leaving Intel/AMD … unguarded against the same class of EP failure"*
(`picodet_decode.h:59-70`). It is now in the shared decoder, checked **per row**
— a first-row-only test passed while partial NaN output was silently dropped.

**Three PicoDet decoders, two of them wrong.** `picodet_decode.h:4-14`: Intel's
was correct; AMD's *"decoded only 6 columns, refused any tensor with `cols < 6`
… never read the authoritative `count` tensor — using the rows tensor's first
dim, which is DATA-DEPENDENT and documented to go stale across repeated
requests, silently dropping layout from every consecutive response — and did no
class-id range check."*

**Intel + AMD skipped the layout post-filter.** Both call sites now carry the
same comment: *"The SHARED postfilter (NMS + full-page-image drop +
containment/merge-mode reconciliation) must run on EVERY backend: CPU
(`cpu_paddle_layout.cpp:271`) and NVIDIA (`paddle_layout.cpp:223`) already
applied it, and these two arms did not — so Intel/AMD returned raw overlapping
boxes and their layout, reading order and every downstream block/table decision
diverged from the other two on the same page"*
(`src/backends/intel/stages/intel_stages.cpp:784-789`,
`src/backends/amd/stages/rocm_stages.cpp:684-689`).

**Five copies of three cls numbers.** `cls_config.h:7-21` lists them, plus: the
180° quad flip written four times two different ways (a cyclic rebuild on AMD, a
pair of `std::swap`s elsewhere — *"equal only by luck of the corner ordering"*),
and the flip comparison written `>` on three backends and `>=` on AMD.

**Kernel parameters silently ignored.** `kernels.h:27-50` is a formal contract
because of it: AMD dropped `order`; NVIDIA *sniffed* `params.mean` to pick
between two baked variants and discarded `order`/`inv_std`/`inv_scale`; NVIDIA's
`db_postprocess` ignored `oriented` and *"always emitted AABBs while the caller
asked for rotated quads"*; `letterbox` had four different meanings. The rule is
now: **honour the field, or declare it unsupported in `caps().params` and refuse
loudly.** Never substitute.

**Default `DbPostParams` were AMD's forked values.** `kernels.h:83-87`: they used
to be `box_thresh = 0.6 / unclip_ratio = 1.5`, *"AMD's old forked values, which
no backend actually wants"* — every default-constructed caller silently got
lower recall and fatter boxes than `detection::kDbDefaults` (0.45 / 1.4).

**Path policy forked between binaries.** `FORMULA_BACKEND=ppformulanet_s` loaded
formulas on the GPU server but left the unified server rejecting `?formulas=1`
with `FORMULA_BACKEND_DISABLED` — *"two binaries disagreeing on the same env"*
(`formula_bundle_env.h:15-19`). Same shape for SLANeXt: the path policy lived
only in the TRT loader, so `TABLE_BACKEND=slanext` booted the GPU server and
aborted the CPU one with the encoder sitting on disk
(`slanext_paths.h:17-21`).

**Error policy forked.** AMD's `HIP_CHECK` called `std::abort()` on every HIP
error, *"so a single bad launch killed the whole server on the AMD arm while the
identical failure on NVIDIA became a 5xx"* (`src/backends/amd/support/hip_check.h:3-10`).

If your new backend needs a policy that does not exist yet: **add it to the
shared layer**, in one commit, with every backend updated. Do not add it to your
directory. Intel's README says the same thing about a det canvas ladder: *"If
dynamic reshape measures expensive, add `detection::snap_det_canvas()` to the
**shared** layer … do not add a private ladder here."*

---

## 3. The minimum viable backend

**Required.** `load_stages()` must return a working detector and recognizer, or
the server refuses to start:

```
"backend load_stages() did not produce the required detector + recognizer
 stages — refusing to start"
```
(`src/service/server/unified/backend_stages.cpp:108-111`)

**Optional, all opt-in.**

| Stage | Where it is reported | Notes |
|---|---|---|
| text-line angle classifier | `StageAvailability::classifier` | `backend.h:189-191`: **not** an optional capability — it has no request flag, so nothing else in the system ever mentions it. A silent failure here just stops 180°-rotated lines from being corrected, forever. The shared builder logs it loudly (`cpu_stages.cpp:129-134`). |
| layout | `available.optional.set(CapabilityId::Layout, …)` | request-time `LAYOUT_DISABLED` when absent |
| doc orientation | `available.optional.set(CapabilityId::DocOrientation, …)` | it is a *service function*, not a `StageSet` member; the flag says whether `make_orient_func()` returns a live callable |
| tables | `CapabilityId::Table` | |
| formulas | `CapabilityId::Formula` | |

### Report honestly — the three axes

`capability.h:10-21` defines them and they must not be conflated:

* **IMPLEMENTED** (`BackendCaps::implemented`, `backend.h:168-175`) — what this
  backend **+ mode** could *ever* build given the right models. The axis an
  operator **cannot** fix by configuration. Defaults to `all()`; narrow it only
  when a stage is structurally impossible, so `/capabilities` can say
  "unsupported" instead of sending an operator hunting for a model path that
  would never be used.
* **LOADED** (`StageAvailability`) — what actually came up this boot. Usually
  fixable by config.
* **REQUESTED** (`InferOptions`) — not yours.

Build the first two with `set()` (literal facts), never `request()` — `set()`
applies no implications, and *"quietly asserting layout here would make a server
advertise a stage it cannot run"* (`capability.h:102-108`).

Two honesty rules with teeth:

* **`make_orient_func()` returns `{}` when you have no model.** Never a closure
  that always answers 0°: *"strictly better than a closure that always answers 0
  degrees, which would look like a working detector that thinks every page is
  upright"* (`src/backends/intel/backend/intel_backend.cpp:283-288`).
* **`KernelCaps` per-op flags are read by the shared layer.** An op you have not
  implemented must be a declared host fallback, not a no-op. Intel's no-SYCL
  build compiled its five native ops to no-ops and *"detection then emits ZERO
  boxes at full inference cost, which is exactly what an end-to-end F1 of 0.00%
  at 31 img/s looked like"* (`intel_backend.cpp:10-24`). The fix was to delegate
  to `cpu::HostKernels`, not to write an Intel-local host implementation.

---

## 4. Step by step

Order matters: each step is testable against the one before it.

### 4.0 Start from the host-delegated skeleton

```bash
python3 tools/new_backend.py --name foo
```

This gives you `backend/foo_backend.{h,cpp}` + `backend/foo_backend_registry.cpp`
+ a README stub, where every factory returns the shared host implementation and
`load_stages()` calls `cpu::make_vendor_onnx_stages("foo", cfg)`. It compiles
and it *works*. Everything below replaces one delegation at a time.

**The directory layout is fixed, and the names are the interface names.** A
vendor arm is an implementation of a known interface set, so which directory a
file belongs in follows from which header in `include/turbo_ocr/backend/` it
implements — it is not a matter of taste, and "where is this vendor's allocator"
has one answer across all five vendors:

```
src/backends/foo/
├── backend/        Backend impl + the one BackendRegistrar   (§4.6, §4.7)
├── memory/         IDeviceAllocator                          (§4.1)
├── queue/          DeviceQueue / DeviceEvent                 (§4.2)
├── engine/         IEngine                                   (§4.3)
├── kernels_foo/    IKernels + the device kernel sources      (§4.4)
├── stages/         IDetector/IRecognizer/IClassifier/ILayout (§4.5)
├── support/        used by >=2 of the above, implements no seam interface
├── probes/         standalone probes (see intel/probes/ov_engine_probe.cpp)
└── README.md       what IS and is NOT verified on real hardware
```

> **Copy `amd/kernels_hip/` or `apple/kernels_metal/` for the kernels row, not
> `nvidia/kernels_cuda/`.** NVIDIA is the one vendor that does not follow it: its
> `.cu` files sit in `src/backends/nvidia/kernels_cuda/` but are reached from
> **11 TUs outside `src/backends/`**, so they are a shared runtime primitive
> rather than a leaf. Every other vendor's device kernels are called only by
> their own `IKernels` adapter, which is what makes this row free for them. The
> full ruling is in
> `src/backends/nvidia/kernels_cuda/README.md` (in the source tree — repo
> READMEs are not part of this docs site).
> Note that `nvidia/README.md` calls itself the non-regression reference — that
> is about *behaviour*, not layout.

Directories appear as you implement them; the generator emits only `backend/`
and the README. Sibling headers are included through the **vendor-rooted** path
off `-Isrc/backends` — `#include "foo/memory/foo_allocator.h"`, never a bare
`"foo_allocator.h"`, which silently breaks the moment a file moves.

Only `support/` and `probes/` map to no interface, so they carry explicit
membership rules instead (`src/backends/README.md`): `support/` admits a file
only if **two or more** of the other directories use it *and* it implements
nothing from `include/turbo_ocr/backend/`; `probes/` is one `add_executable` per
file. Do not invent a directory you cannot write that one-line rule for.

Run the layout gate before you open a PR — stdlib only, no build needed:

```bash
python3 src/backends/layout_check.py
```

It catches the two things that have actually broken this tree: a bare sibling
include (breaks on any move) and a source file compiled by nothing — the Apple
target once used a non-recursive `file(GLOB .../apple/*.mm)`, which
subdirectories would have silently emptied, giving a **broken binary rather than
a build error**.

### 4.1 `IDeviceAllocator` — device memory

Model: `src/backends/cpu/memory/host_allocator.{h,cpp}` (simplest),
`src/backends/apple/memory/metal_allocator.{h,mm}` (unified memory),
`src/backends/intel/memory/l0_allocator.{h,cpp}` (USM),
`src/backends/amd/memory/hip_allocator.{h,cpp}` (discrete VRAM).

Implement `allocate/free`, `allocate_host/free_host` (pinned staging; plain host
is a legal answer), and `copy_h2d/d2h/d2d` on a queue
(`backend.h:74-110`).

Override `host_coherent()` if your pointers *are* host-dereferenceable once the
queue drains — a UMA APU, an iGPU, a CUDA managed-memory allocator. The default
comes from the device class (`image_view.h:65-74`). **The shared layer branches
on this method, never on `kind == DeviceKind::Metal`**
(`backend.h:83-89`).

> **The `DeviceKind` enum is closed** (`image_view.h:44-50`: Host, Cuda, Metal,
> Hip, L0). A genuinely new memory space needs an enumerator there *plus*
> `device_is_host_coherent()` and `device_kind_name()` — a shared edit, reviewed
> as such. Until it lands, run in `Host` space; the host-delegated skeleton
> already does.

### 4.2 `DeviceQueue` / `DeviceEvent` — one ordered lane

Model: `src/backends/cpu/queue/host_device_queue.h` (the degenerate synchronous lane),
`src/backends/apple/queue/metal_device_queue.{h,mm}` (the interesting one).

`record` / `wait` / `synchronize` / `make_event`, plus the one-submit batch
`begin_batch` / `end_batch` / `flush` (`device_queue.h:114-136`). Per-vendor
mapping is spelled out at `device_queue.h:13-28`.

Two contracts that bite:

* **`synchronize()` while a batch is open is a logic error** — the accumulated
  work has not been submitted, so the wait covers only earlier submissions and
  the caller reads stale results. *"A silent wrong-output bug, not a crash."*
  Diagnose it (`device_queue.h:100-107`). Use `flush()` then `synchronize()`.
* `begin_batch`/`end_batch` may be near-no-ops. Intel's are, deliberately: an
  in-order `sycl::queue` already coalesces, and *"`end_batch()` does **not** call
  `q.wait()`; a flush there would destroy the coalescing the seam is asking
  for"* (`src/backends/intel/README.md`).

### 4.3 `IEngine` — the model runner

Model: `src/backends/cpu/engine/cpu_engine_adapter.{h,cpp}` (host/ORT),
`src/backends/intel/engine/openvino_engine.{h,cpp}`,
`src/backends/amd/engine/migraphx_engine.{h,cpp}`,
`src/backends/apple/engine/mps_engine.{h,mm}`,
`src/backends/nvidia/engine/trt_engine_adapter.{h,cpp}`.

**This is the step that buys the speed, and it is the one to do first after
memory.** Answer `EngineCaps` (`engine.h:72-85`) *up front* rather than letting
callers assume: `io_space`, `async`, `caller_owns_outputs`, `multi_io`,
`dynamic_shapes`, `thread_safe_concurrent`. `engine.h:8-22` explains why those
six: they are exactly the things that genuinely differ between TRT, ORT-CUDA,
OrtEngine, MPSGraph, MIGraphX and OpenVINO.

`profiles()` and `graph()` return `nullptr` when absent — do not fake them
(`engine.h:124-129`).

If your runtime compiles per shape, **pre-build at `load()` from the shared
ladder** (`recognition::rec_shape_matrix`) and expose a miss counter. Both Intel
and AMD do, and both say why: *"a shape that was never prebuilt … increments
`shape_misses()`, which is exposed precisely so a wrong warmup matrix is
*observable* instead of silently costing reshape time"*
(`src/backends/intel/README.md`); AMD logs a `HOT-PATH COMPILE` line and
requires `hot_path_compiles() == 0` in a warmed pipeline
(`src/backends/amd/README.md`). Never compile during a request.

### 4.4 `IKernels` — one op at a time

Model: `src/backends/cpu/kernels_host/host_kernels.{h,cpp}` is the reference *and* the
fallback target — you may keep delegating to it forever for any op.

Ops: `decode_image`, `resize_normalize`, `warp_crops`, `threshold`,
`db_postprocess`, `argmax`, `preprocess_region`
(`kernels.h:297-402`).

Rules:

* `caps()` reports **per op** whether it is native or a host fallback
  (`kernels.h:198-212`). The op is always callable either way.
* `NormParams` and `DbPostParams` are **binding**. Honour every field, or
  declare it unsupported and refuse via `require_norm_supported()` /
  `require_db_supported()` / `refuse_unbaked_norm()`
  (`kernels.h:235-293`). A void op then returns without writing;
  `db_postprocess` returns `{}`.
* If your kernel bakes constants in, pass the right `NormPath`
  (`FullFrame` / `Warp`) — the per-path `ParamSupport` flag is *only read when
  the call site passes the path* (`kernels.h:118-127`), so omitting it silently
  downgrades to the generic check and your `false` flag enforces nothing.
* `db_postprocess` (CCL + unclip) is the op most likely to stay on the host
  forever, and that is the **right call**: Intel's README argues it costs zero
  accuracy, inherits every future fix, and hand-writing a SYCL union-find would
  be *"a second implementation of shared post-processing policy"*. Apple made the
  same choice.
* Implement `reserve_host_fallback()` if you stage maps back to host memory —
  the performance gate forbids hot-path allocation (`kernels.h:388-401`).

### 4.5 The four stage classes

Model: `src/backends/cpu/stages/cpu_stages.{h,cpp}` (thin wrapping, no logic),
`src/backends/intel/stages/intel_stages.{h,cpp}`,
`src/backends/amd/stages/rocm_stages.{h,cpp}`,
`src/backends/apple/stages/mps_stages.{h,mm}`.

`IDetector` / `IRecognizer` / `IClassifier` / `ILayout`
(`include/turbo_ocr/backend/stages.h`). Inputs are device-resident
`ImageView` + `DeviceQueue&`; outputs are **host POD** — that boundary is the
whole point (`stages.h:5-17`).

Opt-in extras, all defaulted so you can ignore them at first:

* `IDetector::supports_async()` + `enqueue()`/`collect()` — two images in
  flight. Contract at `stages.h:138-146`: at most `max_in_flight()` outstanding
  futures, and **`img` must stay alive and unmodified until `collect()`
  returns** (`ImageView` is non-owning).
* `IDetector::max_batch_size()` — the *capability*;
  `BackendCaps::preferred_batch_size` is the *policy*; the shared batcher takes
  the smaller (`stages.h:158-168`, `backend.h:141-156`). `run_batch()` must
  return exactly `imgs.size()` entries in input order and entry *i* must depend
  only on `imgs[i]` — callers pass images from unrelated requests
  (`stages.h:170-178`).
* `IRecognizer::last_dropped_crops()` — **implement this**, see §8.

### 4.6 The `Backend` class

Model: `src/backends/cpu/backend/cpu_backend.{h,cpp}` (the reference — 125 lines),
`src/backends/intel/backend/intel_backend.{h,cpp}` (the dual-path one).

Wire the factories, `load_stages()`, and the two service functions. See §6 for
mode handling and §8 for the `make_orient_func()` capture trap.

### 4.7 The registrar

Model: `src/backends/cpu/backend/cpu_backend_registry.cpp` (28 lines) and
`src/backends/intel/backend/intel_backend_registry.cpp` (the one with real
reasoning in it).

One namespace-scope `BackendRegistrar` and nothing else
(`backend_registry.h:68-74`). This TU must **not** define
`backend::make_backend` / `available_backends` — those are defined once in
`src/backend/backend_registry.cpp`, and a per-vendor definition is why only one
backend could ever be linked into a binary (`backend_registry.h:5-19`).

**Factory contract:** return `nullptr` when the device is absent. The registry
treats null and a thrown exception identically — *"compiled in, but not usable
on this machine"* — and auto-detect walks to the next priority, logging every
fall-through (`src/backend/backend_registry.cpp:118-145`). An **explicit**
`--backend foo` that finds no device now throws a clear error rather than
returning null, because null used to be indistinguishable from "no such backend"
and produced *"an error naming the backend inside its own available list"*
(`backend_registry.cpp:148-171`).

**Priority.** Constants at `backend_registry.h:48-52` (nvidia 100, amd 90, apple
80, intel 70, cpu 0). A new, unmeasured backend must **not** claim the auto slot
— set it below `kBackendPriorityCpu` until `turbo_bench` says otherwise. Intel
does exactly that and records the number: *"measured on Core Ultra 7 265T,
intel/OpenVINO runs 4.3 img/s where the ORT CpuBackend runs 8.8 … at
`kBackendPriorityIntel` it would then WIN by default and silently halve
throughput"* (`intel_backend_registry.cpp`).

### 4.8 Table / formula: dispatch, never a private table

Copy `cpu_backend.cpp:63-81` verbatim in shape:

```cpp
if (spec.kind == backend_routing::Kind::Openai)
  return backend::make_table_recognizer(spec);      // device-independent
if (spec.engine.empty() || spec.engine == "slanext")
  return std::make_unique<cpu::CpuTableRecognizer>(); // CUDA-free local
return backend::make_table_recognizer(spec);
```

The middle line is load-bearing: *"Handing a local spec to the common registry
asks for the CUDA-tied sibling, which a CPU-configured build never compiles: the
factory returns null and the server ABORTS AT BOOT rather than serve without
tables"* (`intel_backend.cpp:168-177`).

### 4.9 CMake target + `TURBO_BACKENDS`

`tools/new_backend.py` prints these with line numbers. Three edits to
`CMakeLists.txt`:

1. **Register the name** (`CMakeLists.txt:1025`) — add `foo` to
   `set(_turbo_known_backends cpu apple nvidia amd intel)`, or the configure
   `FATAL_ERROR`s with "unknown backend 'foo'" (`:1026-1032`).
2. **Pull in the shared ONNX stage set** (`CMakeLists.txt:1175-1176`) — add
   `OR foo IN_LIST TURBO_BACKENDS` to the guard that builds
   `turbo_ocr_backend_onnx`, or you get undefined references to
   `cpu::make_vendor_onnx_stages`.
3. **Add the target**, after the `turbo_ocr_backend_cpu` block
   (`CMakeLists.txt:1199-1207`) and inside the `if(TURBO_BACKENDS)` scope:

```cmake
    if(foo IN_LIST TURBO_BACKENDS)
        add_library(turbo_ocr_backend_foo STATIC
            src/backends/foo/backend/foo_backend.cpp
            src/backends/foo/backend/foo_backend_registry.cpp   # <- registrar; needs WHOLE_ARCHIVE
        )
        target_include_directories(turbo_ocr_backend_foo PUBLIC "${CMAKE_SOURCE_DIR}/src/backends")
        target_link_libraries(turbo_ocr_backend_foo PUBLIC
            turbo_ocr_pipeline
            turbo_ocr_backend_onnx      # THE shared fast path (host pre/post + ORT EP)
            turbo_ocr_host_kernels      # SHARED host fallback ops
            ${OpenCV_LIBS}
        )
        target_compile_options(turbo_ocr_backend_foo PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
        add_library(turbo_ocr::backend_foo ALIAS turbo_ocr_backend_foo)
    endif()
```

Nothing else is needed. `turbo_link_backends()` loops over `TURBO_BACKENDS` and
force-links `turbo_ocr_backend_${_b}` with
`$<LINK_LIBRARY:WHOLE_ARCHIVE,...>` (`CMakeLists.txt:1403-1410`), so
`turbo_bench` / `turbo_conformance` / `turbo_golden` / `turbo_backend_probe`
(`:1421-1452`) and `_turboocr` (`:1581`) pick the new backend up automatically,
and the `golden_foo_{det,cls,rec}` ctests register themselves for every non-cpu
backend in the list (`:1487-1500`).

**WHOLE_ARCHIVE is not an optimisation, it is the correctness requirement**
(`CMakeLists.txt:1400-1402`, and the same warning at `:1-4`): the registrar
defines no symbol anybody references, so a plain archive link legitimately drops
the whole object and your backend vanishes from `available_backends()` with no
error at all. Linux's `-Wl,--gc-sections` is exactly where this bites
(`tests/cpp/backends/README.md` §1a #6).

A FUNSD accuracy **gate** is *not* automatic — add a row to `_turbo_gates`
(`CMakeLists.txt:1508-1516`) once you have a measured floor.

Configure:

```bash
cmake -B build-foo -S . -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON -DFETCH_MODELS=OFF \
      -DTURBO_BACKENDS="cpu;foo" \
      -DTURBO_FUNSD_CACHE=$HOME/compare-ocrs/funsd_cache
ninja -C build-foo turbo_backend_probe turbo_bench turbo_conformance turbo_golden
./build-foo/turbo_backend_probe --list      # MUST print cpu AND foo
```

Keep `cpu` in the list. `turbo_conformance` needs two backends in one binary and
exits **77 (SKIP)** with one (`tests/cpp/backends/README.md` §2c).

---

## 5. The host-fallback strategy — you do not need kernels on day one

**Both Intel and AMD started here, and Intel still ships this way.**

The seam is designed so that a backend with *only* an `IEngine` is complete and
correct:

* `make_queue()` → `cpu::HostDeviceQueue`
* `allocator()` → `cpu::HostAllocator`
* `make_kernels()` → `cpu::HostKernels`
* `load_stages()` → `cpu::make_vendor_onnx_stages("foo", cfg)`

That last call is the shared "fast" path: det/cls/rec/layout + doc-orientation
on `engine::OrtEngine` over *your* execution provider
(`cpu_stages.h:125-141`, `cpu_stages.cpp:88-164`). NVIDIA-on-CUDA-EP,
Intel-on-OpenVINO, Apple-on-CoreML and AMD-on-MIGraphX are all **this same
code** with a different `EpConfig` — *"the per-vendor part is one provider
string, not a stage set each"* (`cpu_stages.h:130-131`).

Add your provider to `onnx_provider_for()`
(`include/turbo_ocr/backend/engine_mode.h:76-82`) and its fp16 story to
`fp16_support_for()` (`:98-105`) — the four fp16 mechanisms genuinely differ
(`Provider` / `Native` / `Model` / `None`) and *"pretending they do [agree] is
how 'fp16' silently becomes fp32"* (`:84-95`). Adding a provider string is a
shared edit to a shared header; make it as one.

`TURBO_EP_PROVIDER` overrides the vendor default, which matters more than it
sounds: *"the stock onnxruntime-linux-x64 build ships only the CPU provider, so
`--backend intel` in onnx mode asks for 'openvino' and gets a clean 'provider
not compiled in' failure on a perfectly good Intel box"*
(`cpu_stages.cpp:204-215`).

Then move work onto the device in the order of §4, re-running the gates after
each step. Intel's live example: with `TURBO_OCR_HAS_SYCL` undefined it uses
`cpu::HostKernels` and *"the OpenVINO engine — where the speed lives — becomes
usable with no oneAPI toolchain at all"* (`intel_backend.cpp:10-24`).

---

## 6. Mode handling — Native vs Onnx

`EngineMode` is the **second selection axis**, orthogonal to the vendor
(`engine_mode.h:3-27`):

* **Native** ("ultra") — the vendor graph engine (a built TRT plan, an exported
  MPSGraph, a compiled OpenVINO blob, a MIGraphX program). Fastest steady state,
  useless without an artefact that has to be built or exported.
* **Onnx** ("fast") — the plain `.onnx` through your ORT execution provider. No
  graph build; starts in seconds on the models already on disk.
* **Auto** — prefer Native, fall back to Onnx **loudly**.

Use the shared policy, do not re-invent it:

```cpp
const bool native_available = FooEngine::artefacts_present(cfg);
mode_ = cpu::resolve_engine_mode("foo", cfg, native_available);
```

`resolve_engine_mode()` (`cpu_stages.h:143-153`, impl `cpu_stages.cpp:169-198`)
encodes the corner that matters: an **explicit** `native` with nothing to load
throws, because *"silently serving the slower path would make 'my ultra engine is
not being used' indistinguishable from 'my ultra engine is slow'"*; `Auto` with
nothing to load logs and returns `Onnx`, because *"a fallback nobody can see is a
silent performance cliff."*

### `caps()` must report the mode you ACTUALLY came up on

`backend.h:158-161`: *"an Auto run that fell back from native to onnx must SAY
onnx."* And the device side must follow the mode (§8, trap 4).

> **[VERIFIED BY READING, NOT BY RUNNING] Resolve the mode before the server
> reads `caps()`.** `build_backend_runtime` snapshots `rt.caps =
> rt.backend->caps()` at `src/service/server/unified/backend_stages.cpp:78` and calls
> `load_stages()` only at `:107`. Everything the server reports afterwards —
> `device`, `async`, `mode`, `/capabilities`, the `is_gpu` routing flag, the HTTP
> thread count — comes from that single pre-load snapshot
> (`src/service/server/unified/server_main.cpp:111,168,174-176,185,205`). Apple and Intel both
> resolve their mode *inside* `load_stages()` and both default the member to
> `Native` (`src/backends/apple/backend/apple_backend.h:52`,
> `src/backends/intel/backend/intel_backend.cpp:58`), so on a fall-back boot the
> snapshot names the wrong path. The Python binding reads `caps().mode` live
> (`src/service/python/bindings.cpp:230`) and is unaffected. **For a new backend:
> resolve the mode in the factory/constructor** (read `TURBO_ENGINE_MODE` there)
> so `caps()` is right the first time it is asked. I have not reproduced this at
> runtime — treat it as a code-reading finding.

---

## 7. Correctness gates, in order

Do not skip ahead. An F1 computed on top of a broken kernel tells you nothing.

### 7.1 Syntax — before anything builds

For a vendor arm this machine cannot compile, `tools/syntax_shims/` declares just
enough of the SDK for `-fsyntax-only` to type-check it:

```bash
tools/syntax_shims/check.sh                                  # everything in sources.txt
tools/syntax_shims/check.sh src/backends/foo/backend/foo_backend.cpp # one file
```

Add your TUs to `tools/syntax_shims/sources.txt`. If your SDK is missing, add a
stub header — copy the **real** signature, because *"a stub that disagrees with
the real header is worse than no stub"* (`tools/syntax_shims/README.md`).

This proves declarations, overloads and template instantiation line up — i.e.
the code would compile. It does **not** prove it links or runs. Because your
backend class is `final` and gets instantiated, a clean run also proves **every
`override` matches the seam's pure virtuals** — the argument Intel's README
makes for its own check.

### 7.2 Link — the registrar survived

```bash
./build-foo/turbo_backend_probe --list      # MUST list cpu AND foo
ctest -R backend_probe
```

Missing name = the archive dropped the registrar (§4.9).

### 7.3 `turbo_conformance` — do the linked backends agree?

```bash
./build-foo/turbo_conformance --images ~/compare-ocrs/funsd_cache \
    --count 20 --out /tmp/conformance_foo.json
```

Same pages through every backend in the binary, diffed against the CPU
reference: box match rate, mean IoU, per-line text agreement, pages identical,
plus the first disagreements. Exits 77 (SKIP) on a single-backend build.

Calibration, from the measured cpu-vs-apple baseline
(`tests/cpp/backends/README.md` §3): box match **96.57 %**, mean IoU **0.9357**,
per-line exact text agreement **73.08 %**, pages identical **0/10**. Read that
carefully: *"two backends can disagree on ~27 % of individual lines and still
land 0.07 pt apart on F1"* — per-character diffs from device bilinear
resampling, in both directions. The ctest thresholds (0.90 / 0.88 / 0.65) are
**tripwires for a backend that has started to diverge**, not equality
assertions.

### 7.4 `turbo_golden` — which stage diverges?

```bash
./build-foo/turbo_golden --backend foo --ref cpu --stage all \
    --images ~/compare-ocrs/funsd_cache --count 10
# --stage det|cls|rec|all
```

`cls` and `rec` are fed the **reference backend's boxes**, so a disagreement is
that stage's own rather than detection leaking downstream
(`tests/cpp/backends/README.md` §2d). Registered as `golden_foo_{det,cls,rec}`
automatically.

Measured apple-vs-cpu baseline: `det` agreement/IoU **0.9625 / 0.9283**, `cls`
flip agreement **0.9881**, `rec` exact-string agreement **0.7223**.

If you are bringing kernels up, do the finer per-op diff both vendor READMEs
prescribe — dump and compare `resize_normalize`, `warp_crops`, `threshold`,
`db_postprocess`, `argmax`, `preprocess_region` against the host
implementations, in that order. Expected tolerances are tabulated in
`src/backends/intel/README.md` ("Validation on real hardware"), including the
one that must be **bit-identical** (`threshold`) and the one that must be
identical by construction (`db_postprocess`, because both sides call the same
shared function). Assert the argmax tie-break explicitly with a synthetic tie —
**lower class index wins** — because a natural corpus will not hit it reliably
(`src/backends/amd/README.md`, Stage 1.4).

### 7.5 `turbo_bench` — accuracy, then throughput

```bash
# accuracy
./build-foo/turbo_bench --backend foo --tier tiny \
    --images ~/compare-ocrs/funsd_cache --count 50 \
    --words /tmp/foo.words.json --out /tmp/foo.metrics.json
python3 tools/bench/score_funsd.py /tmp/foo.words.json \
    --metrics /tmp/foo.metrics.json --assert-f1 85.2

# throughput (WITH its accuracy — always)
./build-foo/turbo_bench --backend foo --tier tiny \
    --images ~/compare-ocrs/funsd_cache --count 50 \
    --threads 8 --repeat 40 \
    --words /tmp/foo_tp.words.json --out /tmp/foo_tp.metrics.json

# paired A/B — the only comparison that survives thermal drift
./build-foo/turbo_bench --ab cpu,foo --tier tiny \
    --images ~/compare-ocrs/funsd_cache --count 50 --repeat 3
```

Targets on the same models: cpu tiny **85.79 %** (exact and deterministic with
`DISABLE_COREML=1`), cpu medium **92.56 %**, apple tiny **85.44–85.72 %**
(`tests/cpp/backends/README.md` §3). Materially below means a stage is subtly wrong,
not that your device is less accurate — you are running the same ONNX through a
different runtime.

The harness **enforces** five disciplines (`tests/cpp/backends/README.md` §5), all
learned the hard way:

1. **Timed window ≥ 15 s.** Shorter windows are dominated by model load and
   graph JIT — *"that is how a fabricated 288 img/s reading was once produced."*
   Below 15 s the run exits non-zero unless `--allow-short-window`.
2. **Wall-clock cross-check.** Two independent clocks time the same region (a
   `steady_clock` span, and summed per-image latencies ÷ threads); > 5 %
   disagreement rejects the rate. This is what caught the 288 artifact. Prove it
   still works with `--selftest-skew-ms 20000` (expect exit 4).
3. **Never a throughput number without its accuracy.** Every run scores its own
   transcript in-process.
4. **Thermal drift is real** (~12 % over a session) — use `--ab`.
5. **Device saturation is sampled inside the window** and printed next to the
   rate, so "hardware limit" is distinguishable from "device idled waiting on the
   host".

Set the env the baselines were measured with, or the comparison is invalid:
`DISABLE_COREML=1`, `TURBO_APPLE_REC_BUCKETS=…`, `TURBO_APPLE_PROFILE` unset.
Everything matching `TURBO_*`, `OCR_*`, `ORT_*`, `CUDA_*`, `TRT_*`, `HIP_*`,
`ZE_*`, `OMP_*` lands in the metrics JSON, along with SHA-256 of every model
artefact (directories too) and of the image set — so a cross-machine
disagreement can be traced rather than argued about (`tests/cpp/backends/README.md`
§4).

### 7.6 Everything through ctest

```bash
cd build-foo && ctest -R "backend_|golden_|funsd_" --output-on-failure
```

---

## 8. The traps

Every one of these was paid for. Sources cited.

**1. Pre-sized result vectors hide partial failure.** Every recognizer pre-sizes
its output and writes an *empty* entry for a failed crop, so the vector is still
exactly `boxes.size()` long and *"the pipeline's under-return check
structurally cannot see the loss"* — a partial recognition failure came back as a
thin page with no warning at all. `IRecognizer::last_dropped_crops()` exists in
the **seam** for this reason: *"a per-backend fix would leave the other three
silently wrong"* (`stages.h:216-232`). Reset it at the top of every
`run()`/`run_multi()`. Live examples:
`src/backends/apple/stages/mps_stages.mm:450,555`,
`src/backends/intel/stages/intel_stages.cpp:430,505-511`,
`src/backends/amd/stages/rocm_stages.cpp:320`. The pipeline consumes it at
`src/pipeline/unified/unified_ocr_pipeline.cpp:343,687`.

**2. Free-then-allocate under a throwing check macro.** AMD's scratch caches were
`if (p) free(p); p = allocate(n); cap = n;` — fine while `HIP_CHECK` aborted, a
use-after-free once it started throwing (matching NVIDIA's recoverable
`CUDA_CHECK`): *"a throwing allocator would leave `p` holding a FREED pointer
with `cap` still claiming the old size: the next request on this POOLED,
long-lived stage would either skip the regrow … and write through a dangling
pointer, or free it a second time."* Release **and clear** before the call that
can throw (`src/backends/amd/stages/rocm_stages.cpp:58-79`). Related: make your error
macro *throw* on ordinary device errors and terminate only on sticky faults —
AMD's aborted on everything and *"a single bad launch killed the whole server on
the AMD arm while the identical failure on NVIDIA became a 5xx"*
(`src/backends/amd/support/hip_check.h:3-10`).

**3. Raw-pointer capture in `make_orient_func()`.** `build_backend_runtime` calls
`load_stages()` **once per pool entry** (`src/service/server/unified/backend_stages.cpp:106-107`)
and each call **replaces** the doc-orientation model; `UnifiedOcrPipeline`'s
constructor calls `make_orient_func()` for every entry
(`src/pipeline/unified/unified_ocr_pipeline.cpp:204`). A raw-pointer capture therefore
*"leaves every pool entry but the last holding a dangling pointer (latent UAF)"*.
Capture `this` — the backend outlives every pipeline and route
(`src/backends/cpu/backend/cpu_backend.cpp:107-122`,
`src/backends/intel/backend/intel_backend.cpp:271-282`).

**4. Device pointers handed to host stages in onnx mode.** In Onnx mode the
stages are the shared **host** ones: they take an `ImageView` in host memory and
dereference it on the CPU. Returning a device queue + device allocator makes the
shared pipeline upload every page into device memory and hand a device pointer to
host code. *"That is not a slow path, it is a wrong one — it aborted on the first
image"* (`src/backends/apple/backend/apple_backend.mm:70-81`; Intel does the same guard,
`intel_backend.cpp:133-158`). **All four device factories must check the mode.**

**5. Count-tensor vs `shape[0]` sizing.** For PP-DocLayoutV3, the model's own
count tensor is authoritative; `rows_dim0` is data-dependent and *"goes stale
across repeated requests on at least the TRT path, silently zeroing layout on
every response after the first"* (`picodet_decode.h:40-45`). The mirror-image bug:
sizing your host copy buffer from `rows_dim0` while letting `*count` drive the
loop is a **heap over-read** when `*count > rows_dim0` — copy the full
`kPicodetMaxDet` budget, which is always in bounds because the rows tensor is
allocated at the budget regardless (`src/backends/amd/stages/rocm_stages.cpp:668-679`).

**6. Thread-cap defaults are wrong for an accelerated backend.** Each host ORT
stage caps its intra-op pool assuming it competes with det/rec for cores —
`OrtEngine` 4, formula 4, `CpuPaddleLayout` **2**. That is right only when the
rest of the stage set is *also* on the CPU. If your det/rec run on the device,
call `set_host_ort_intra_op_threads(4)` once at bootstrap
(`include/turbo_ocr/onnx/host_ort_threads.h:37-46`). Measured on Apple native:
layout 1010 → 583 ms, formula 1.08 → 0.53 s, table 1.30 → 0.68 s
(`src/backends/apple/backend/apple_backend.mm:255-272`). Use **4**, not "as many as ORT
wants": pool replicas × a machine-sized pool oversubscribes and *"on layout that
cost 40% of throughput to buy 130 ms of latency."* `ORT_NUM_THREADS` still wins.

**7. ODR collisions with the old CUDA-typed world.** `ITableRecognizer` and
`IFormulaRecognizer` were declared in the *same namespaces* by both the old
`turbo_ocr/{table,formula}/…` headers and the new seam headers, so one TU cannot
see both generations. They were moved into `turbo_ocr::backend` specifically to
end that collision (`docs/architecture/multi-backend.md:88`;
`src/backends/nvidia/stages/nv_table_bridge.h:7`). NVIDIA still needs a
**pimpl-across-a-generation-gap** split for table and formula — a new-headers TU,
an old-headers TU, and a neutral bridge header including *neither* interface
(`src/backends/nvidia/README.md`, "The old/new interface collision"). det / rec /
cls / layout need no such split: `turbo_ocr::backend` does not collide with
`turbo_ocr::{detection,recognition,…}` (`src/backends/nvidia/stages/nv_stages.h:11`).
**`turbo_ocr::backend` is the new seam** — put your overrides there and stay out
of the old namespaces.

**8. A no-op kernel looks exactly like a working one.** Intel's no-SYCL build
compiled five native ops to no-ops; detection emitted zero boxes at full
inference cost — *"an end-to-end F1 of 0.00% at 31 img/s"*
(`intel_backend.cpp:10-24`). Delegate to `cpu::HostKernels` instead. Never ship
an op that returns without doing anything and without saying so.

**9. Auto-detect can silently claim a machine.** A factory that returns a backend
on hardware it cannot actually use makes the server boot onto it; every
fall-through in the shared registry is logged precisely because *"a CUDA
driver/library mismatch, a Metal device that would not open, or a failed Level
Zero init turns into a host-backend server — an order-of-magnitude throughput
loss on a machine that has the hardware, with zero operator signal"*
(`src/backend/backend_registry.cpp:124-128`). And do not gate availability on
the *wrong* thing: Intel's gate was `L0Allocator::has_device()`, false whenever
built without SYCL, so `--backend intel` returned null on a working iGPU
(`intel_backend_registry.cpp`). Gate on "does the runtime enumerate the device".

**10. Two clocks, or your throughput number is fiction.** See §7.5. Also: a
single Apple throughput figure is worth ±30 % — three runs of an identical
command measured 103.8 → 71.1 → 93.6 img/s with F1 bit-stable at 85.44 %
(`tests/cpp/backends/README.md` §3, §5.6). Quote the window, the utilization, and
use `--ab`.

**11. `ImageView` is non-owning, and async detection widens the gap.** With
`supports_async()`, the page buffer for image N+1 must stay alive and unmodified
until `collect()` returns (`stages.h:143-146`). The base `ILayout` even ships a
`pending_` member that stores a bare `ImageView` across the gap — flagged in the
seam as *"a dangling-view bug waiting for the first backend that sets
`supports_async() == true`"* (`stages.h:60-64`).

---

## 9. Reviewer checklist

Structure

- [ ] No `make_infer_func()` override, no private pipeline, no private pool.
- [ ] No HTTP/gRPC/serialization/validation code in `src/backends/<name>/`.
- [ ] Table/formula are registry **dispatch** in the `cpu_backend.cpp:63-81`
      shape, including the explicit CUDA-free local branch.
- [ ] Everything under `namespace turbo_ocr::<name>`; overrides target
      `turbo_ocr::backend`, not the old `turbo_ocr::{table,formula}`.

Shared policy (grep the directory — each of these appearing locally is a defect)

- [ ] No literal `mean` / `inv_std` / `inv_scale` — `norm::*` only.
- [ ] No private width buckets, `kMaxW`, or batch ladder — `rec_geometry.h` +
      `rec_batching.h`.
- [ ] No inline CTC loop — `ctc_greedy_decode` / `load_label_dict`.
- [ ] No private cls height/width/threshold/quad-flip — `cls_config.h`.
- [ ] No private det canvas rule or DB thresholds — `det_config.h`.
- [ ] No private crop-geometry math — `compute_crop_transform`.
- [ ] Layout calls `decode_picodet_rows()` **and** `postfilter_layout_boxes()`.
- [ ] SLANeXt / formula paths come from `slanext_paths.h` /
      `formula_bundle_env.h`.
- [ ] `resolve_engine_mode()` is used; the fallback rule is not re-implemented.

Honesty

- [ ] `caps().mode` is the mode actually in use, resolved **before** the server
      snapshots `caps()` (§6).
- [ ] `caps().device` / `.async` follow the mode (Host + sync in onnx mode).
- [ ] `caps().implemented` narrowed only for structurally impossible stages.
- [ ] `StageAvailability` reflects what loaded; a configured-but-failed model is
      logged, not silently skipped.
- [ ] `KernelCaps` reports each op's real native/fallback status; no silent
      no-ops.
- [ ] `ParamSupport` matches reality; unsupported fields refuse via
      `require_*_supported()` with the right `NormPath`.
- [ ] `EngineCaps` (io_space / async / ownership / multi_io / shapes /
      thread-safety) is accurate; `profiles()`/`graph()` return `nullptr` when
      absent.
- [ ] `make_orient_func()` returns `{}` rather than a stub answering 0°.

Traps

- [ ] `last_dropped_crops()` implemented and reset per run.
- [ ] Scratch regrow clears the pointer **before** the throwing allocate.
- [ ] `make_orient_func()` captures `this`, never a raw model pointer.
- [ ] All four device factories branch on the mode.
- [ ] Layout row count uses the count tensor; host buffers sized at
      `kPicodetMaxDet`.
- [ ] `set_host_ort_intra_op_threads()` called iff det/rec are off-CPU.
- [ ] Registrar factory returns `nullptr` when the device is absent.
- [ ] Auto-detect priority below `cpu` until `turbo_bench` justifies otherwise —
      with the number in a comment.

Build + gates

- [ ] Three CMake edits done (§4.9); registry TU is in the source list.
- [ ] `turbo_backend_probe --list` shows the new backend (WHOLE_ARCHIVE worked).
- [ ] TUs added to `tools/syntax_shims/sources.txt`; `check.sh` green.
- [ ] `turbo_conformance` runs (not 77) and is inside the tripwires.
- [ ] `turbo_golden --stage all` inside the tripwires, or every deviation
        explained per stage.
- [ ] `turbo_bench` F1 at parity with cpu on the same tier; throughput quoted
      **with** its F1, its ≥ 15 s window, and its saturation block.
- [ ] `_turbo_gates` row added with a measured floor.
- [ ] The backend's `README.md` states plainly what is verified and what is not.

---

## 10. Files to read, in order

1. `include/turbo_ocr/backend/backend.h` — the seam.
2. `src/backends/cpu/` — the simplest complete implementation.
3. `src/backends/cpu/stages/cpu_stages.h` / `.cpp` — the shared ONNX path and the mode
   policy every backend calls.
4. `src/backends/intel/README.md` — a host-fallback-first backend, and the
   honest verified/not-verified split to copy.
5. `src/backends/amd/README.md` — device-resident bring-up, and vendor-specific
   correctness risks written down before the hardware existed.
6. `src/backends/apple/README.md` — a real device-resident backend with measured
   numbers, including the levers that **lost**.
7. `tests/cpp/backends/README.md` — the acceptance path and the measurement
   discipline.
8. `docs/architecture/multi-backend.md` — why the seam is shaped this way.
