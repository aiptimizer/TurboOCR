> **NOTE (2026-07):** the native layer has since been unified — `CpuOcrPipeline`
> was deleted in the multi-backend restructure. The binding now wraps
> `UnifiedOcrPipeline` over the backend seam (`src/service/python/bindings.cpp`,
> Python-visible class `Pipeline`; construction goes through
> `backend::make_backend("cpu")` + `Backend::load_stages`). Code excerpts below
> that reference `CpuOcrPipeline` describe the ORIGINAL design and are kept as
> history; the current source is authoritative.

# TurboOCR Python bindings — native design (nanobind) + cross-platform wheels

**Decision (this doc):** the Python library is a **thin wrapper over the C++
engine**, bound with **nanobind**. No pipeline logic in Python — detection,
recognition, CTC, warp, layout/table/formula all run your existing
`libturbo_ocr_cpu.a` / `libturbo_ocr_gpu.a`. "Full speed" means C++/CUDA/Metal/
TRT/AVX2 speed. The pure-Python `detection.py`/`recognition.py`/
`classification.py` are removed.

---

## 1. The one idea that makes the wheel matrix simple

`turbo_ocr_cpu` (the whole ONNX/ORT pipeline: det → rec → cls → layout → table →
formula) links **`${ONNXRUNTIME_LIB}`** — *whatever ONNX Runtime you point CMake
at* (`-DONNXRUNTIME_LIB=... -DONNXRUNTIME_INCLUDE_DIR=...`). The execution
provider inside it is chosen by `OrtEngine::apply_execution_provider()` from the
`ORT_EP` env var (coreml / openvino / migraphx / rocm / dml / cpu).

So the backend axis is **not** N codebases. It is:

> **one nanobind extension × one C++ pipeline × N ONNX Runtime builds.**

Each wheel = the same extension linked against a different ORT variant. That is
exactly the Tier-A / Tier-B seam from the architecture notes, now surfaced to
Python.

```
                         ┌──────────────────────────────┐
   Python (thin glue)    │ OCR · doctor · models · pdf  │  ← reused, no pipeline logic
                         │ result · cli                 │
                         └───────────────┬──────────────┘
                                         │ nanobind
                         ┌───────────────▼──────────────┐
   native _turboocr.so   │  CpuOcrPipeline (Tier B)      │  fast-setup ONNX, EP switch
                         │  OcrPipeline    (Tier A, opt) │  TensorRT "turbo"
                         └───────────────┬──────────────┘
                                         │ links
              ┌──────────────────────────┼───────────────────────────┐
        libturbo_ocr_cpu.a        libturbo_ocr_common.a         ${ONNXRUNTIME_LIB}
        libturbo_ocr_gpu.a          (+ OpenCV, pdfium)      (cpu│cuda│openvino│dml│rocm│coreml)
```

---

## 2. Native binding surface (`src/service/python/bindings.cpp`, nanobind)

Small and boring on purpose — marshal a BGR uint8 image in, marshal result items
out. Zero-copy input via `nb::ndarray`.

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "turbo_ocr/pipeline/ocr/cpu_ocr_pipeline.h"

namespace nb = nanobind;
using turbo_ocr::pipeline::CpuOcrPipeline;

static cv::Mat as_bgr(nb::ndarray<uint8_t, nb::ndim<3>, nb::c_contig> a) {
  // a: [H,W,3] uint8 BGR, no copy — wrap the caller's buffer
  return cv::Mat((int)a.shape(0), (int)a.shape(1), CV_8UC3, (void*)a.data());
}

NB_MODULE(_turboocr, m) {
  nb::class_<turbo_ocr::OCRResultItem>(m, "Item")
      .def_ro("text", &turbo_ocr::OCRResultItem::text)
      .def_ro("confidence", &turbo_ocr::OCRResultItem::confidence)
      .def_prop_ro("box", [](const turbo_ocr::OCRResultItem& r){
          return std::array{r.box[0][0],r.box[0][1], r.box[1][0],r.box[1][1],
                            r.box[2][0],r.box[2][1], r.box[3][0],r.box[3][1]}; })
      .def_ro("layout_id", &turbo_ocr::OCRResultItem::layout_id);

  nb::class_<CpuOcrPipeline>(m, "CpuPipeline")
      .def(nb::init<>())
      .def("init", &CpuOcrPipeline::init,
           "det"_a, "rec"_a, "dict"_a, "cls"_a = "", /* det_cfg via env */)
      .def("load_layout_model", &CpuOcrPipeline::load_layout_model)
      .def("load_table_backend", &CpuOcrPipeline::load_table_backend)
      .def("load_formula_model", &CpuOcrPipeline::load_formula_model)
      .def("warmup", &CpuOcrPipeline::warmup)
      .def("run", [](CpuOcrPipeline& p, nb::ndarray<uint8_t,nb::ndim<3>,nb::c_contig> img){
          nb::gil_scoped_release nogil;                 // release GIL during C++
          return p.run(as_bgr(img));
      })
      .def("run_with_layout", [](CpuOcrPipeline& p, /*img*/, bool layout,
                                 bool order, bool tables, bool formulas){ ... });

  // GPU build only (compiled when USE_CPU_ONLY is off):
  // nb::class_<OcrPipeline>(m, "GpuPipeline") ...  // TensorRT "turbo"
}
```

Notes
- **GIL released** around every `run()` so Python threads overlap with C++.
- **EP selection**: Python sets `ORT_EP` (+ `OPENVINO_DEVICE`, `ROCM_DEVICE_ID`,
  `CLS_ONNX`, `DET_*`, `REC_BATCH_N`, …) in `os.environ` **before** constructing
  the pipeline — the C++ already reads all of these. No C++ API churn.
- **det config**: pass through the existing `DET_*` env overrides; a typed
  `DetInferConfig` arg can be added later if we want it non-env.

### One small, high-value C++ addition
`OrtEngine::apply_execution_provider()` today supports xnnpack/dnnl/coreml/
openvino/migraphx/rocm/dml but **not CUDA**. To honor "fast-setup ONNX default
on NVIDIA" (run the ONNX graph on the GPU with **no TensorRT engine build**), add
a `cuda` (and optional `tensorrt`) branch (~10 lines) that appends
`CUDAExecutionProvider`. Then the NVIDIA story is:
- `backend="cuda"`  → CpuOcrPipeline + ORT CUDA EP (instant start)  ← **default on NVIDIA**
- `backend="turbo"` → OcrPipeline + TensorRT (peak throughput, slow first build)

---

## 3. CMake integration (additive; nothing existing changes)

```cmake
option(BUILD_PYTHON "Build the nanobind Python extension" OFF)
if(BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  # nanobind ships its CMake via pip; scikit-build-core puts it on CMAKE_PREFIX_PATH
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_turboocr STABLE_ABI NB_STATIC src/service/python/bindings.cpp)
  if(USE_CPU_ONLY)
    target_link_libraries(_turboocr PRIVATE turbo_ocr_cpu)      # Tier B (+ORT variant)
  else()
    target_link_libraries(_turboocr PRIVATE turbo_ocr_gpu turbo_ocr_cpu)  # +Tier A TRT
  endif()
  install(TARGETS _turboocr LIBRARY DESTINATION turboocr)
endif()
```

`STABLE_ABI` → one extension binary works across CPython 3.12+ (fewer wheels).

---

## 4. Wheel matrix (the deliverable you asked to design first)

Same extension, N ORT variants. Names mirror onnxruntime's own distribution so
`doctor` maps 1:1.

| PyPI wheel | Hardware / OS | ORT variant linked (`${ONNXRUNTIME_LIB}`) | EPs exposed | fast-setup default | Bundled into wheel |
|---|---|---|---|---|---|
| **`turboocr-engine-cpu`** (base) | any CPU; **+ Apple GPU** on macOS arm64 | onnxruntime (CPU; mac wheel has CoreML) | CPU, XNNPACK, **CoreML** (mac) | ✅ **CPU/MLAS** | ORT, OpenCV, pdfium(Linux) |
| **`turboocr-engine-cuda12`** / **`-cuda13`** | NVIDIA, Linux + Win | onnxruntime-gpu (CUDA+TRT) | **CUDA**, TensorRT | ⚠️ **auto → TensorRT (first run builds)** | ORT-GPU, OpenCV vendored; CUDA/cuDNN/TensorRT **host-provided** — excluded by the repair step and NOT declared as deps |
| **`turboocr-engine-openvino`** | Intel CPU/iGPU/Arc/NPU | onnxruntime (plain CPU) | CPU (+ native OpenVINO engine, `backend="openvino"`) | ✅ native OpenVINO | ORT, OpenCV vendored; OpenVINO via the wheel's own `openvino` pip dep (preloaded) |
| **`turboocr-directml`** (not built) | any DX12 GPU, Windows | onnxruntime-directml | DirectML | ✅ DirectML | ORT-DML |
| **`turboocr-engine-rocm`** | AMD, Linux + ROCm | onnxruntime-migraphx (AMD index) | MIGraphX, ROCm | ✅ MIGraphX | ORT-ROCm (from repo.radeon.com) |

- **`turbo` (TensorRT)** is a *mode*, not a wheel: `OCR(model, backend="turbo")`
  on the `turboocr-engine-cuda12`/`-cuda13` wheels uses the Tier-A `OcrPipeline` with an on-disk
  engine cache. It IS auto-selected on these wheels (slow first build) — matching the
  fast-setup default.
- **Install exactly one** `turboocr*` wheel per environment (they all provide the
  `turboocr` import and link mutually-incompatible ORT builds — same rule
  onnxruntime itself has).
- CUDA/ROCm wheels do **not** fat-bundle the vendor toolkit; they piggy-back the
  runtime libs the ORT-GPU / ORT-ROCm wheels already ship (call
  `onnxruntime.preload_dlls()` at import). Keeps wheels sane-sized.

---

## 5. Build tooling

- **Build backend:** `scikit-build-core` (drives CMake from `pip install`) +
  `nanobind`.
  ```toml
  [build-system]
  requires = ["scikit-build-core>=0.10", "nanobind>=2"]
  build-backend = "scikit_build_core.build"
  [tool.scikit-build]
  cmake.args = ["-DBUILD_PYTHON=ON", "-DUSE_CPU_ONLY=ON", "-DFETCH_MODELS=OFF"]
  ```
  The `-DONNXRUNTIME_LIB`/`-DUSE_CPU_ONLY` values differ per wheel (env-templated
  in CI).
- **Matrix:** `cibuildwheel` — `manylinux_2_28` (x86_64/aarch64), macOS
  (arm64 + x86_64), Windows (amd64). One cibuildwheel config per backend wheel.
- **Repair / vendoring:** `auditwheel` (Linux), `delocate` (macOS),
  `delvewheel` (Windows) fold `libonnxruntime`, OpenCV, `libpdfium` into the
  wheel. Exclude CUDA/cuDNN/ROCm sonames from repair (provided by the ORT-GPU/
  ROCm dependency, not bundled).

---

## 6. PDF (your pdfium requirement)

Two options; recommend **A** as default:

- **A. `pypdfium2`** (default, cross-platform): the maintained Python binding to
  *the same PDFium*. Renders each page → BGR array → native `run()`. Works on
  macOS too (the C++ PDF daemon is Linux-only). Already implemented in `pdf.py`.
- **B. native `PdfRenderer`** (Linux fast-path, optional): bind the pooled
  fastpdf2png/PDFium daemon for max render throughput on servers. Add later
  behind `pdf_backend="native"`.

Either way it's PDFium. Rendering is not the OCR bottleneck, so A ships first.

---

## 7. What changes in the current `python/` tree

| File | Fate |
|---|---|
| `detection.py`, `recognition.py`, `classification.py` | **delete** (native does this) |
| `pipeline.py` | **rewrite**: `OCR` calls `_turboocr.Pipeline` (renamed from the planned `CpuPipeline` when the binding moved onto `UnifiedOcrPipeline` — see `native.py`), sets EP env, marshals `Item`→`TextLine` |
| `session.py`, `providers.py` | **repurpose**: `providers` now = hardware→**wheel** map; EP knob = `ORT_EP` env, not `InferenceSession` |
| `doctor.py` | **keep**, retarget to recommend a `turboocr*` **wheel** (data already structured) |
| `models.py`, `result.py`, `pdf.py`, `imaging.py`, `catalog.py`, `cli.py` | **keep** as-is (thin glue) |
| `__init__.py` | keep; fix `_default_engine` to key its cache on `(model, backend)` (confirmed bug) |

---

## 8. Milestones

1. **Bindings + local proof (macOS CPU/ONNX):** `src/service/python/bindings.cpp`,
   `BUILD_PYTHON` CMake, build `_turboocr` against the existing
   `libturbo_ocr_cpu.a`; `OCR().read("receipt.png")` runs *your* C++ (~7× vs the
   NumPy port per the Phase-0 notes). Rewrite `pipeline.py`, delete the reimpl.
2. **CUDA EP in `OrtEngine`** (+`tensorrt`) so NVIDIA fast-setup = ONNX-on-GPU,
   no engine build. Verify on a Linux+NVIDIA box.
3. **scikit-build-core packaging**; `pip install ./python` builds the extension.
4. **cibuildwheel**: ship `turboocr-engine-cpu` (CPU+CoreML) first, then
   `turboocr-engine-cuda12`/`-cuda13`, then openvino/rocm.
5. **Retarget `doctor`** to wheels; wire `backend=`→`ORT_EP`; `turbo`→Tier A.
6. **Optional:** native `PdfRenderer` Linux fast-path; layout/table/formula
   surfaced (the C++ already supports them via `run_with_layout`).
```

---

## Status (loops 1–6, native binding on macOS CPU/ONNX)

Done and tested (`python/tests/test_smoke.py`, 11 passing):
- ✅ Milestone 1 — nanobind `_turboocr` over `libturbo_ocr_cpu.a`; `OCR().read()` runs the real C++ engine (receipt/PDF verified).
- ✅ Milestone 2 — guarded CUDA branch in `OrtEngine::apply_execution_provider` (`#ifdef TURBO_HAVE_CUDA`) + clean Python rejection on non-CUDA builds via `build_info()`.
- ✅ Milestone 5 (partial) — `doctor` reads native `build_info()` providers; install matrix retargeted to per-backend wheels.
- ✅ Milestone 6 (partial) — `run_with_layout` bound (`PageResult.layout` populated with labeled regions + reading order). Tables/formulas not yet marshaled.
- ✅ DX — `draw()`/`save_overlay()`/`crop()`, TSV/hOCR/pandas, `filter()`, `page[i]`, lang/tier selection, autorotate, PDF/batch `progress=`, CLI (globs, `--format`, `-o`, `--overlay`, `--layout`, `--lang/--tier`).
- ✅ Safety — construction lock (env race), per-instance run lock (data race), quiet-by-default stdout.

Not yet done (next):
- ✅ Milestone 3 (2026-08-04) — `scikit-build-core` build backend: `pip wheel python/` drives the top-level CMake (`BUILD_PYTHON=ON`, install component `python` → `_turboocr` + `turbo_apple.metallib`, nothing else), and `delocate-wheel` produces a fully self-contained 28 MB macOS wheel (ORT 1.28 + OpenCV + PDFium vendored; verified in a clean venv from a neutral cwd, native Metal mode). The RPATH landmine was real: `_turboocr` needs explicit `INSTALL_RPATH` entries for the vendored PDFium and ORT dirs or delocate cannot chase `@rpath` deps. hatch_build.py deleted.
- 🚧 Milestone 4 — per-backend wheels: `scripts/python/build_backend_wheel.sh <cpu|cuda|openvino|rocm>` stages a renamed dist (PEP 621 forbids dynamic names) and repairs it. **the CUDA engine wheel built and validated on the RTX 5090 box (2026-08-04)**: clean venv, `OCR(backend="cuda")` runs on the CUDA EP with correct output. Blockers before shipping it: (a) ~~ORT-CUDA slower than CPU~~ FIXED 2026-08-05: scalar-rec default root-caused, device-EP batching defaults land 7.8 img/s (~2x CPU); the remaining ~84 ms/page is ORT's per-shape-switch replanning (proven with a bare-ORT repro — see BUGS.md), so beating it needs per-shape CUDA graphs or the native-TRT wheel; (b) a teardown heap corruption on the CUDA path; (c) portable manylinux repair must run in the cibuildwheel container (build hosts' glibc outruns auditwheel policies; the dlopen'd provider libs are injected post-auditwheel by the script). The cibuildwheel CI matrix now covers all four engine wheels, and the PyPI naming decision landed 2026-08-13: the published `turboocr` client SDK continues as the pure-Python umbrella (python-sdk/) whose extras pin `turboocr-engine-*`.
- ⏳ Bind the real `PipelinePool` for cross-instance concurrency (currently single-flight per instance).
- ⏳ Searchable-PDF export (needs visual verification), tables/formulas marshaling.

---

## Packaging next-step (architect-verified recipe — do WITH the user, needs CI)

The wheel build must NOT drag in Drogon/gRPC/protobuf (server-only deps). Verified
that `turbo_ocr_cpu`/`turbo_ocr_common` include no protobuf/grpc headers and link
none — only the two server executables do. Do NOT fork a parallel `python/CMakeLists`
or link prebuilt `.a`s (both drift/staleness risks). Instead gate the root CMake:

1. Add `option(BUILD_SERVER "Build the HTTP/gRPC servers" ON)` (default ON ⇒ every
   existing Docker/dev invocation is byte-identical).
2. Wrap `find_package(Drogon CONFIG REQUIRED)` (CMakeLists.txt:~203) in `if(BUILD_SERVER)`.
3. Move `turbo_setup_grpc_codegen()` (currently ~line 405, before `add_library(turbo_ocr_cpu)`)
   DOWN to immediately before `add_executable(turboocr-cpu-server ...)`, and wrap that
   executable block in `if(BUILD_SERVER)`. Same for the GPU-mode server (~772).
4. Gate `add_executable(turbo_ocr_tests ...)` (~276) behind `BUILD_TESTING`.
5. Switch `python/pyproject.toml` build-system to `scikit-build-core` with
   `cmake.args = ["-DBUILD_PYTHON=ON","-DBUILD_SERVER=OFF","-DUSE_CPU_ONLY=ON","-DFETCH_MODELS=OFF"]`
   and `cmake.source-dir = ".."` (the wheel builds the ext from the repo root C++).

Landmines the architect flagged for the cibuildwheel stage:
- **STABLE_ABI vs requires-python>=3.9**: nanobind `abi3` only covers CPython ≥3.12;
  3.9–3.11 need per-interpreter wheels. Decide: bump floor to 3.12 (1 wheel/platform)
  or build cp39/310/311 explicitly.
- **RPATH/vendoring**: `$ORIGIN`-relative RPATH for libpdfium/libturbojpeg/libonnxruntime
  so auditwheel/delocate/delvewheel actually bundle them; else it imports on the build
  box but not a clean machine. Add a per-wheel `otool -L`/`ldd` symbol-version check in CI
  (the protobuf dylib skew we hit will recur per wheel).
- **Collision**: two `turboocr-*` wheels in one site-packages silently last-wins; add an
  import-time build_info self-check.

Until this lands, local install is: `cmake --build build-cpu --target _turboocr` then
`pip install ./python` (hatchling packages the built `.so`).
