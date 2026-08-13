# Per-backend wheels

Four engine PyPI distributions, **one source tree**, **one C++ engine**:

| Distribution | Hardware | Config |
|---|---|---|
| `turboocr-engine-cpu` | any CPU, + Metal/ANE on macOS arm64 | [`../pyproject.toml`](../pyproject.toml) |
| `turboocr-engine-cuda12` | NVIDIA, CUDA 12 (driver R525+) | [`cuda12/pyproject.toml`](cuda12/pyproject.toml) |
| `turboocr-engine-cuda13` | NVIDIA, CUDA 13 (driver R580+) | [`cuda13/pyproject.toml`](cuda13/pyproject.toml) |
| `turboocr-engine-openvino` | Intel CPU / iGPU / Arc / NPU | [`openvino/pyproject.toml`](openvino/pyproject.toml) |
| `turboocr-engine-rocm` | AMD (ROCm) | [`rocm/pyproject.toml`](rocm/pyproject.toml) |

They are **mutually exclusive**. Every one of them installs the same
`turboocr_engine` import package, the same `_turboocr` extension module and the
same `turboocr` console script; the only differences are the distribution name
and the CMake args. Two of them in one `site-packages` is last-writer-wins, not
a merge — install exactly one, and let `turboocr doctor` (which reads the
native `build_info()`) tell you which engine you actually have. (The
pure-Python `turboocr` umbrella in `python-sdk/` is a separate, compatible
distribution: its `[cpu]`/`[cuda]`/`[openvino]`/`[rocm]` extras are the normal
way to install exactly one of these.)

## Why separate wheels and not one wheel with a plugin

The execution provider is **statically compiled into `_turboocr`**: the
ONNX Runtime the extension links, and the vendor backends co-linked beside it,
are fixed at build time. So a wheel *is* an engine. This is the same rule
`onnxruntime` / `onnxruntime-gpu` / `onnxruntime-openvino` follow, for the same
reason.

## How a variant config works: STAGING

A variant is **never built from this directory**. `<backend>/pyproject.toml` is
the canonical config for that variant, but the build copies it **over**
`python/pyproject.toml` (base backed up first) and builds `python/`:

```bash
scripts/python/build_backend_wheel.sh cuda12      # -> build-wheels/cuda12/fixed/
```

The script restores the base config on exit — success, failure or `^C`.

Every path in a variant config is therefore written **relative to `python/`**,
byte-identical to the base config:

* `cmake.source-dir = ".."` → the repo root, the same C++ tree the base wheel
  builds.
* `wheel.packages = ["turboocr_engine"]` → the **one** pure-Python package at
  `python/turboocr_engine`; there is no copy of it under `wheels/`.
* `[tool.scikit-build.metadata.version] input = "turboocr_engine/_version.py"` → the
  same single source of truth as the base wheel. Bump
  `python/turboocr_engine/_version.py` and all four distributions move together.
* `wheel.py-api = "cp312"` → one `cp312-abi3` wheel per platform. See
  [abi3](#abi3-one-wheel-per-platform-not-per-interpreter).
* `build-dir` is **per variant**. `USE_CPU_ONLY` and `TURBO_BACKENDS` are CMake
  *cache* variables: sharing one build tree across variants would silently
  rebuild the previous variant's configuration.

**Why staging and not `pip wheel python/wheels/cuda`.** PEP 621 forbids a
dynamic `[project].name`, so the four names need four files — but a config
*inside* `wheels/<backend>/` can only reach the package and the C++ tree through
`../../` escapes, and cibuildwheel mounts **only the package directory** into
its container, where such a path resolves to nothing. Staging keeps one set of
paths that works identically for a local build, for cibuildwheel, and for CI.

**Wheels only.** None of these configs can produce a usable *sdist*: the C++
sources live above the project directory (true of the base wheel too). Use
`pip wheel` / `cibuildwheel`, which build the directory in place — not
`python -m build`, whose default path goes through an sdist.

## abi3: one wheel per platform, not per interpreter

`nanobind_add_module(_turboocr STABLE_ABI)` already compiles the extension
against CPython's 3.12 limited API and emits `_turboocr.abi3.so`. That alone
does **not** change the wheel *tag*: without `wheel.py-api` scikit-build-core
still tags the wheel `cp3XY-cp3XY`, so the matrix ships one wheel per
interpreter for a binary that needs one. All four configs set
`wheel.py-api = "cp312"` (nanobind's floor for the stable ABI); each build then
produces a single `cp312-abi3` wheel per platform, installable on 3.12 and every
later CPython.

Building on an older interpreter is not a trap: scikit-build-core drops a
`py-api` it cannot satisfy and falls back to the plain `cp3XY-cp3XY` tag, so a
from-source build under `requires-python = ">=3.9"` is still tagged honestly.

*(Windows note: `nanobind` STABLE_ABI links `python3.dll`, which needs
`find_package(Python COMPONENTS ... Development.SABIModule)` — scikit-build-core
exposes the right component in `SKBUILD_SABI_COMPONENT`. The root `CMakeLists.txt`
asks only for `Development.Module`; no Windows wheel is in the matrix today, so
this has not had to matter yet.)*

## Backend → CMake args

| Wheel | `TURBO_BACKENDS` | Build path | Engine reached at run time |
|---|---|---|---|
| `turboocr-engine-cpu` | *(default: `cpu` on Linux/Windows, `cpu;apple` on macOS)* | `USE_CPU_ONLY=ON` — ONNX Runtime host path | ORT CPU/XNNPACK, CoreML + Metal on macOS |
| `turboocr-engine-cuda12` / `-cuda13` | `cpu;nvidia` | `USE_CPU_ONLY=OFF` — native CUDA + TensorRT | ORT CUDA EP (`backend="cuda"`), TensorRT (`backend="turbo"`) |
| `turboocr-engine-openvino` | `cpu;intel` | `USE_CPU_ONLY=ON` | native OpenVINO Runtime, ORT OpenVINO EP; device via `OV_DEVICE=CPU\|GPU\|NPU` |
| `turboocr-engine-rocm` | `cpu;amd` | `USE_CPU_ONLY=ON` | native MIGraphX, ORT MIGraphX/ROCm EP |

Two things that look surprising and are not:

* **`USE_CPU_ONLY` is a misnomer** (`CMakeLists.txt:262-266`). It selects the
  ONNX-Runtime *host* build path, which `cpu`, `apple`, `intel` **and** `amd`
  all take. Only `nvidia` needs the native CUDA/TensorRT path, because
  `turbo_ocr_backend_nvidia` wraps `turbo_ocr_gpu`, which does not exist in a
  host-path configure (`CMakeLists.txt:1766-1772`). CMake already infers
  `USE_CPU_ONLY=ON` from any backend list without `nvidia`
  (`CMakeLists.txt:279-287`); the configs state it anyway so all four read the
  same way.
* **`cpu` is in every list.** `TURBO_BACKENDS` is a list that co-links into one
  binary, and the vendor is picked at *run* time (`--backend` / `TURBO_BACKEND`,
  empty = auto-detect). Keeping `cpu` beside the vendor costs nothing and
  leaves a working fallback when the device is missing or unusable.

## Build inputs CI must provide (not pip dependencies)

The matching **ONNX Runtime build is a build input**, not a Python dependency.
Nothing in a `pyproject.toml` can fetch it: it is a C++ library the extension
links, discovered by CMake in `third_party/onnxruntime` or `/usr/local`, or
pointed at explicitly with `-DONNXRUNTIME_LIB=... -DONNXRUNTIME_INCLUDE_DIR=...`.
The same goes for the vendor SDKs. Per variant:

| Wheel | Required on the build image |
|---|---|
| `turboocr-engine-cuda12` / `-cuda13` | the MATCHING CUDA-enabled ORT (`gpu_cuda12` or `gpu_cuda13`) (`onnxruntime-linux-x64-gpu_cuda*`; the configure hard-fails without `libonnxruntime_providers_cuda`), CUDA toolkit, TensorRT at `-DTENSORRT_DIR=`, and `-DCMAKE_CUDA_ARCHITECTURES=` for the SMs to ship |
| `turboocr-engine-openvino` | OpenVINO Runtime dev package (`find_package(OpenVINO)`), ORT with the OpenVINO EP |
| `turboocr-engine-rocm` | ROCm (`hip` + `migraphx` + hipcc), `-DCMAKE_HIP_ARCHITECTURES=`, ORT with the MIGraphX/ROCm EP |

Pass them through the `CMAKE_ARGS` environment variable — scikit-build-core
appends it *after* `cmake.args`, so it wins on conflicts, and the build script
forwards it:

```bash
CMAKE_ARGS="-DTENSORRT_DIR=/usr/local/tensorrt -DCMAKE_CUDA_ARCHITECTURES=80-real;90-real;120-real" \
  scripts/python/build_backend_wheel.sh cuda12    # or cuda13
```

## Repair (vendoring) is deliberately not fully expressed here

Each config carries a minimal `[tool.cibuildwheel]` (cp312 abi3 build, Linux
x86_64), but **not** a `repair-wheel-command`, because the correct repair is
more than one command:

* CUDA/cuDNN/TensorRT and ROCm sonames must be **excluded** from vendoring —
  they come from the host driver/toolkit, exactly as `onnxruntime-gpu` ships.
* ORT's execution-provider libraries are **`dlopen`'d, not `DT_NEEDED`**, so
  `auditwheel` drops them; they have to be re-injected into the wheel's libs
  directory afterwards, with their original sonames.
* `auditwheel` refuses to run when the build host's glibc is newer than any
  policy it knows, which is why the repair belongs inside the manylinux
  container.

That logic already exists, once, in `scripts/python/build_backend_wheel.sh`.
Point `CIBW_REPAIR_WHEEL_COMMAND` at it rather than duplicating a partial copy
per variant.

## Two traps worth remembering

* **Stale local extension.** `wheel.packages` copies `python/turboocr_engine` as it is
  on disk. A locally built `python/turboocr_engine/_turboocr*.so` (or
  `turbo_apple.metallib`) sitting there can be picked up by the copy; clean
  them before building a release wheel. (Same caveat as the base wheel;
  `scripts/python/build_backend_wheel.sh` handles it by staging a clean copy.)
* **Server dependencies.** These configures do not set `-DBUILD_SERVER=OFF`, so
  the root CMake still runs `find_package(Drogon CONFIG REQUIRED)` even though
  only the `_turboocr` target is built. Either install Drogon on the build
  image or add `-DBUILD_SERVER=OFF` via `CMAKE_ARGS` — the wheel never links
  the servers either way.
