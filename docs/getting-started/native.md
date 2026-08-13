# Native build

!!! abstract "TL;DR"
    This page is the NVIDIA build in depth — reference platform Arch Linux,
    CUDA 13.x, RTX 5090 (Blackwell, sm_120); `scripts/setup/install_native.sh`
    is the supported one-shot installer, and this page documents what it does
    so you can reproduce it by hand on any distribution. For the other
    backends the build is one CMake invocation — see
    [Other backends](#other-backends) below or the
    [install selector](install.md).

## System packages

`scripts/setup/install_native.sh:45` invokes:

```bash
sudo pacman -S --needed --noconfirm \
    cmake opencv protobuf grpc jsoncpp openssl c-ares nginx curl libjpeg-turbo
```

On Debian/Ubuntu the equivalents include `libcurl4-openssl-dev` and
`libturbojpeg0-dev` — both are hard link dependencies of the GPU build (the
VLM table/formula clients use libcurl even though they are runtime-opt-in,
and inline page-image export uses libjpeg-turbo as the CPU fallback encoder).
The GPU build additionally needs a **CUDA-enabled** ONNX Runtime (with
`libonnxruntime_providers_cuda.so`). On x86_64 a clean clone fetches the
official pinned `onnxruntime-linux-x64-gpu_cuda13` release tarball
automatically at configure time (same pin as the Docker `nvidia` target); a
pre-installed copy in `third_party/onnxruntime/{include,lib}` or `/usr/local`
is preferred when present. The plain `onnxruntime-linux-<arch>-<ver>.tgz`
asset is CPU-only and is rejected at configure time with an explanatory
error; aarch64 has no official GPU tarball and needs a custom CUDA-enabled
ORT dropped in manually.

!!! warning "CUDA prerequisite"
    The installer refuses to proceed if `nvcc --version` reports a
    release below 13.0 (`install_native.sh:33`). On Arch the package is
    `cuda`; ensure `nvcc` is on `$PATH` before running the installer.

## Drogon HTTP framework

Pinned to **v1.9.12**, built from source — there is no Arch package.
`install_native.sh:48-66` clones the upstream repo into a tempdir,
configures with every optional ORM / Redis / SQLite backend disabled,
and `sudo cmake --install build`s into `/usr/local`.

```bash
cmake -B build \
      -DBUILD_EXAMPLES=OFF -DBUILD_CTL=OFF -DBUILD_ORM=OFF \
      -DBUILD_POSTGRESQL=OFF -DBUILD_MYSQL=OFF -DBUILD_SQLITE=OFF \
      -DBUILD_REDIS=OFF -DBUILD_TESTING=OFF
cmake --build build -j"$(nproc)"
sudo cmake --install build
```

## TensorRT

Version is mapped to the host CUDA version
(`install_native.sh:82-99`):

| CUDA   | TensorRT       | Tar suffix    |
|--------|----------------|---------------|
| 13.0   | 10.14.1.16     | `cuda-13.0`   |
| 13.1   | 10.15.1.29     | `cuda-13.1`   |
| 13.2   | 10.16.0.72     | `cuda-13.2`   |

The tarball is fetched from
`https://developer.download.nvidia.com/compute/machine-learning/tensorrt/...`,
extracted to `/usr/local/`, and symlinked at `/usr/local/tensorrt`. The
installer appends

```bash
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}
```

to your shell rc so `turboocr-server` finds `libnvinfer.so.10` at run
time.

!!! danger "LD_LIBRARY_PATH is mandatory"
    Without `/usr/local/tensorrt/lib` on `LD_LIBRARY_PATH` the server
    will fail to dlopen the TRT runtime at startup. The installer wires
    this into `~/.bashrc` / `~/.zshrc` automatically; if you bypass the
    installer, set it yourself — and note most distro bashrc files return
    early for non-interactive shells, so a server launched via ssh, cron
    or systemd does **not** inherit the export. Set it in the unit/launch
    environment explicitly.

    A wrong or missing path can also stay hidden: the RUNTIME may resolve
    via the binary's rpath while the engine cache satisfies every start —
    until the first fresh engine build, which fails with
    `Unable to load library: libnvinfer_builder_resource_sm*…`. That error
    means the *builder resource* dlopen missed: point `LD_LIBRARY_PATH` at
    the directory that actually contains your `TensorRT-*/lib`.

!!! tip "sm_120 builder lib"
    TensorRT 10.15's `Builder` library has a hard sm_120 (Blackwell)
    dependency when JIT-building engines from ONNX on first launch. On
    older GPUs CUDA omits the kernels silently and you'll see
    `kernels not found` at engine-build time — drop
    `-DCMAKE_CUDA_ARCHITECTURES=120` to your host's compute capability
    (e.g. `89` for Ada).

## Configure & build

Direct CMake invocation, matching what `install_native.sh:160-163`
runs:

```bash
cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTURBO_BACKENDS="nvidia;cpu" \
      -DTENSORRT_DIR=$HOME/TensorRT-10.15.1.29 \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc) --target turboocr-server
```

!!! warning "`TURBO_BACKENDS` is not optional here"
    `turboocr-server` is defined in `unified_server.cmake`, which the root list
    includes only inside `if(TURBO_BACKENDS)`. That variable defaults to **empty**
    on the CUDA configure, so leaving it out configures cleanly, builds the
    libraries and then fails with *no rule to make target `turboocr-server`* —
    the server target was never created. Name the backends you want.

`CMAKE_CUDA_ARCHITECTURES` defaults to the toolkit's oldest supported arch
(`sm_75`), whose embedded PTX JITs forward onto anything newer. `120` is native
Blackwell (RTX 50-series) and skips that JIT — set the value matching your card.

`TENSORRT_DIR` can also point to `/usr/local/tensorrt` (the installer's
default symlink). If `nvcc` is not on `PATH`, add `-DCUDAToolkit_ROOT=/opt/cuda`;
CMake's `FindCUDAToolkit` will otherwise stop with *Could not find `nvcc`*.
Recent distributions may ship a GCC newer than the CUDA release supports —
`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15` pins a supported one.

The default `target` builds everything including the unit-test binary;
restricting to `turboocr-server` shaves ~30 s on a clean build.

!!! success "Verified from a clean tree"
    Run end to end on 2026-08-10 — Arch, CUDA 13.3.33, TensorRT 10.15.1.29,
    RTX 5090 at `sm_120` — starting from a tree with **no models and no
    `third_party/`**:

    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
          -DTURBO_BACKENDS="nvidia;cpu" -DCMAKE_CUDA_ARCHITECTURES=120 \
          -DCUDAToolkit_ROOT=/opt/cuda -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
          -DTENSORRT_DIR=$HOME/TensorRT-10.15.1.29
    cmake --build build -j$(nproc)          # also fetches ORT + the model set
    bash scripts/setup/install_fastpdf2png.sh
    export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
    ./build/turboocr-server
    ```

    Result: 0 build failures, 620/620 tests, server up on `backend=nvidia
    device=cuda`, and `POST /ocr/raw` on `tests/fixtures/images/png/mixed_fonts.png`
    returned 508 text lines. First start spends ~90 s building TensorRT engines.

## PDF renderer (required to start the server)

The server rasterizes PDF pages through a separate `fastpdf2png` binary, and it
treats a missing one as **fatal at startup** — not as a disabled feature:

```
{"level":"error","msg":"Fatal error during startup","error":"fastpdf2png binary not found"}
```

It is not a CMake target; install it once, into `bin/` next to the build:

```bash
bash scripts/setup/install_fastpdf2png.sh
```

The server searches `/app/bin`, `/usr/local/bin`, `./build/` and `./bin/`, in
that order; `FASTPDF2PNG_PATH` overrides the search outright and fails fast if
the path you name does not exist.

## Smoke test

```bash
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
./build/turboocr-server &      # native build binds port 8080 (the 8000→8080 nginx hop is Docker-only)
curl -fsS http://localhost:8080/health/ready
curl -X POST http://localhost:8080/ocr/raw \
     --data-binary @tests/fixtures/images/png/receipt.png \
     -H 'Content-Type: image/png'
```

!!! note "First-start cost"
    First start spends ~90 s building TensorRT engines from the bundled
    ONNX files; engines are cached under `~/.cache/turbo-ocr/` so
    subsequent runs are instant. `/health/ready` returns `503 NOT_READY`
    during the build — the `curl -fsS` above fails fast on that 503, so retry
    it (or poll in a loop) until it returns 200.

## Build output

| Path | What |
|---|---|
| `build/turboocr-server` | HTTP + gRPC server (GPU build) |
| `build/turbo_ocr_tests` | Catch2 unit suite (always built) |
| `build/proto_gen/` | Generated `ocr.{pb,grpc.pb}.{h,cc}` stubs |
| `build/turbo_ocr_common.a`, `build/turbo_ocr_gpu.a` | Internal libs |

## CPU-only variant

For a no-GPU build pass `-DUSE_CPU_ONLY=ON` and target
`turboocr-server`; that build does **not** need TensorRT (only
ONNX Runtime 1.28.0, which CMake fetches on demand if not already in
`/usr/local/lib`).

```bash
cmake -B build_cpu -DUSE_CPU_ONLY=ON
cmake --build build_cpu -j$(nproc) --target turboocr-server
```

### PDFium

Two PDFium binaries are vendored side by side under `third_party/pdfium`,
sharing one `include/` tree:

| Target | File | Source |
|---|---|---|
| `linux-x64` | `lib/libpdfium.so` | vendored |
| `mac-arm64` | `lib/libpdfium.dylib` | vendored |
| `linux-arm64`, `mac-x64` | fetched | `scripts/setup/install_pdfium.sh` |

CMake picks the right one per platform. `scripts/setup/install_pdfium.sh` fetches the
missing targets from the pinned bblanchon release, verifies the SHA256, and
installs *only* that platform's library — the other vendored binary is left in
place, so one tree can serve both. On macOS it also rewrites the dylib's
install name to `@rpath` and re-signs it ad-hoc, which arm64 requires.

### macOS (CPU + Apple backend)

One-time prerequisites on Apple Silicon — the build fails at configure (or
worse, silently misbehaves) without them:

1. **Xcode + Metal toolchain.** The Apple backend compiles
   `src/backends/apple/kernels_metal/shaders.metal` at build time via
   `xcrun metal`, which the Command Line Tools alone do **not** provide.
   Install full Xcode; on Xcode 26+ the Metal compiler is a separate
   component: `xcodebuild -downloadComponent MetalToolchain`.
2. **Homebrew packages.**

    ```bash
    brew install cmake opencv drogon jsoncpp protobuf grpc c-ares jpeg-turbo
    ```

    Drogon has a Homebrew formula — the from-source Drogon build above is a
    Linux-only step.
3. **ONNX Runtime ≥ 1.27 (osx-arm64).** The automatic ORT fetch above is
   **Linux-only**, so drop the official
   `onnxruntime-osx-arm64-<ver>.tgz` contents into
   `third_party/onnxruntime/{include,lib}` (preferred by CMake on macOS), or
   use a Homebrew `onnxruntime` **≥ 1.28**. Do not build against an old brew
   1.24: its CoreML EP silently returns NaN for every layout score, so every
   page gets an EMPTY layout that looks like a blank document rather than an
   error.
4. **Models.** `scripts/models/fetch/fetch_release_models.sh`, same as Linux.

Then the build is the standard CPU invocation — `TURBO_BACKENDS` defaults to
`cpu;apple` on macOS, so the Metal/MPSGraph/ANE backend is compiled in
automatically (runtime selection via `--backend apple`; commands in the
[install selector](install.md)).

PDFium needs nothing on mac-arm64 (vendored; mac-x64 runs
`install_pdfium.sh`, see above). All PDF work — rasterization, text-layer
extraction, `mode=auto_verified`, and the `?output=pdf` searchable-PDF writer —
builds and runs on macOS. Only the *daemon* renderer is Linux-bound
(`pdf_daemon` drives worker processes with `sigtimedwait`/`pipe2`/`inotify`);
platforms without inotify compile the in-process renderer instead and serve
the same routes.

## Sanitizer build (ASan + LSan)

Allocation-level leak and memory-safety checking, Linux and macOS. The CPU
configure builds clean under AddressSanitizer:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CPU_ONLY=ON \
      -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build build-asan -j"$(nproc)"
LSAN_OPTIONS=suppressions=tools/checks/lsan.supp ./build-asan/turbo_ocr_tests
```

`RelWithDebInfo`, not `Release`: it skips the Release-only IPO/LTO path (which
fights sanitizer instrumentation) and keeps symbols so leak stacks are legible.
The suppression file covers exactly one thing — the prebuilt PDFium — and its
header documents the measured evidence for that classification; nothing in the
`turbo_ocr::` namespace is ever suppressed. Expect the suite to run 2–4×
slower under ASan.

For the CUDA configure, host code sanitizes the same way but the CUDA runtime
needs `ASAN_OPTIONS=protect_shadow_gap=0` to initialize (its allocations
collide with ASan's shadow region otherwise), and `.cu` device code is not
instrumented — nvcc has no ASan.

## Other backends

Every backend is one CMake invocation away — `TURBO_BACKENDS` picks the set
(default: `cpu;apple` on macOS, `cpu` elsewhere). The canonical per-backend
commands, including the AMD HIP-architecture line and the Python wheel, live
in the [install selector](install.md); this page stays the NVIDIA deep dive.

!!! info "See also"
    - [Install](install.md) — the pick-your-hardware selector.
    - [Build → Docker](docker.md) — production-ready images with the
      same dependency pins.
    - [Build → Models](model-bundle.md) — what `fetch_release_models.sh`
      lays down.
    - [Dev → Testing](../contributing/testing.md) — how to exercise the binary
      after build.
