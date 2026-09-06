# Native build (Arch Linux + CUDA)

!!! abstract "TL;DR"
    Reference platform is Arch Linux with CUDA 13.x and an RTX 5090
    (Blackwell, sm_120). `scripts/install_native.sh` is the supported
    one-shot installer; this page documents what it does so you can
    reproduce it by hand or adapt to a different distribution.

## System packages

`scripts/install_native.sh:45` invokes:

```bash
sudo pacman -S --needed --noconfirm \
    cmake opencv protobuf grpc jsoncpp openssl c-ares nginx curl libjpeg-turbo
```

OpenCV 4 and OpenCV 5 both work: OpenCV 5 moved the contour and 2-D
transform helpers into its `geometry` module, which CMake links when
present (`-- OpenCV <version>: …` in the configure log says which was found;
`-DOpenCV_DIR=<prefix>/lib/cmake/opencv4` selects one when several are
installed).

On Debian/Ubuntu the equivalents include `libcurl4-openssl-dev` and
`libturbojpeg0-dev` — both are hard link dependencies of the GPU build (the
VLM table/formula clients use libcurl even though they are runtime-opt-in,
and inline page-image export uses libjpeg-turbo as the CPU fallback encoder).
The GPU build additionally needs a **CUDA-enabled** ONNX Runtime (with
`libonnxruntime_providers_cuda.so`). On x86_64 a clean clone fetches the
official pinned `onnxruntime-linux-x64-gpu_cuda13` release tarball
automatically at configure time (same pin as `docker/Dockerfile.gpu`); a
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
    installer, set it yourself.

!!! tip "sm_120 builder lib"
    TensorRT 10.15's `Builder` library has a hard sm_120 (Blackwell)
    dependency when JIT-building engines from ONNX on first launch. On
    older GPUs CUDA omits the kernels silently and you'll see
    `kernels not found` at engine-build time — drop
    `-DCMAKE_CUDA_ARCHITECTURES=120` to your host's compute capability
    (e.g. `89` for Ada).

## Configure & build

Direct CMake invocation:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc) --target turboocr-server
```

CMake finds the toolchain on its own:

- **CUDA**: `nvcc` on `PATH`, else `CUDAToolkit_ROOT`, `CUDA_HOME`,
  `CUDA_PATH`, `/usr/local/cuda`, the newest `/usr/local/cuda-<version>`,
  `/opt/cuda`. `-DCMAKE_CUDA_COMPILER=<toolkit>/bin/nvcc` overrides.
- **TensorRT**: a tarball root (`/usr/local/tensorrt`, the installer's
  symlink) or the distribution packages (`NvInfer.h` in the system include
  directory, `libnvinfer.so` in the multiarch library directory).
  `-DTENSORRT_DIR=<root or lib dir>` is an optional hint, e.g.
  `-DTENSORRT_DIR=$HOME/TensorRT-10.16.0.72` for an unpacked tarball.

The configure log prints what was chosen (`-- CUDA toolkit: …`,
`-- TensorRT: headers …, libraries …`). The default `target` builds
everything including the unit-test binary; restricting to `turboocr-server`
shaves ~30 s on a clean build.

## Smoke test

```bash
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH
./build/turboocr-server &      # native build binds port 8080 (the 8000→8080 nginx hop is Docker-only)
curl -fsS http://localhost:8080/health/ready
curl -X POST http://localhost:8080/ocr/raw \
     --data-binary @tests/test_data/png/receipt.png \
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
| `build/fastpdf2png`, `build/libpdfium.so` | PDF page renderer, built from its pinned source for this machine's architecture (`TURBO_BUILD_FASTPDF2PNG=OFF` skips it) |
| `build/proto_gen/` | Generated `ocr.{pb,grpc.pb}.{h,cc}` stubs |
| `build/turbo_ocr_common.a`, `build/turbo_ocr_gpu.a` | Internal libs |

The server looks for the renderer next to its own executable first, then in
`/app/bin`, `/usr/local/bin`, `./build` and `./bin`; `FASTPDF2PNG_PATH`
overrides the search. A binary built for another CPU is reported as such
("fastpdf2png at … is built for x86-64; this machine is aarch64"), not as
missing.

## CPU-only variant

For a no-GPU build pass `-DUSE_CPU_ONLY=ON` and target
`turboocr-cpu-server`; that build does **not** need TensorRT (only
ONNX Runtime 1.22.0, which CMake fetches on demand if not already in
`/usr/local/lib`).

```bash
cmake -B build_cpu -DUSE_CPU_ONLY=ON
cmake --build build_cpu -j$(nproc) --target turboocr-cpu-server
```

!!! info "See also"
    - [Build → Docker](docker.md) — production-ready images with the
      same dependency pins.
    - [Build → Models](models.md) — what `fetch_release_models.sh`
      lays down.
    - [Dev → Testing](../dev/testing.md) — how to exercise the binary
      after build.

## aarch64 (Jetson, GB10 / DGX Spark, Grace)

The repository vendors **x86-64** binaries: `third_party/pdfium/lib/libpdfium.so`
and the renderer in `bin/` (`fastpdf2png`, `libpdfium.so`). The build never
links or runs them on another architecture:

- **PDFium**: CMake reads the vendored library's ELF header; when it is not the
  target's, it fetches `pdfium-linux-arm64.tgz` of the pinned release
  (`chromium/7857`, from `bblanchon/pdfium-binaries`) into
  `build/_deps/pdfium-linux-arm64/` and verifies its SHA-256 before use. The
  configure log says which SDK was chosen (`-- PDFium: … (fetched …)`).
- **fastpdf2png**: built from its pinned commit with the same compiler as the
  server (`cmake/TurboFastpdf2png.cmake`) and placed at `build/fastpdf2png`.

So a plain configure works on aarch64 once the toolchain is found:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build -j$(nproc) --target turboocr-server
```

Notes for this platform:

- **CUDA and TensorRT** are found as described under *Configure & build*
  (`/usr/local/cuda-13.0` without a `/usr/local/cuda` link, TensorRT from
  the `tensorrt-dev` packages under `/usr/lib/aarch64-linux-gnu`, both work
  without flags).
- **TensorRT 11 is not supported yet** (it removed the builder flag the engine
  builder uses). Use TensorRT 10.16.0.72; there is no official aarch64 tarball
  for it, so install NVIDIA's aarch64 package for your CUDA version.
- **A CUDA-enabled ONNX Runtime is the one thing you build yourself.** The
  formula stage runs on ONNX Runtime's CUDA provider, and Microsoft publishes
  that build for x86-64 only (fetched automatically there). On aarch64 the
  configure step stops with the exact commands; in short:

    ```bash
    git clone --depth 1 -b v1.27.0 https://github.com/microsoft/onnxruntime
    cd onnxruntime
    ./build.sh --config Release --build_shared_lib --parallel --skip_tests \
               --use_cuda --cuda_home /usr/local/cuda-13.0 --cudnn_home /usr
    mkdir -p ../TurboOCR/third_party/onnxruntime/{include,lib}
    cp include/onnxruntime/core/session/*.h ../TurboOCR/third_party/onnxruntime/include/
    cp build/Linux/Release/libonnxruntime.so* build/Linux/Release/libonnxruntime_providers_*.so \
       ../TurboOCR/third_party/onnxruntime/lib/
    ```

    (about an hour on a GB10; cuDNN for your CUDA version must be installed).
    The same layout under `/usr/local` is found too.
- **Page size**: the bblanchon aarch64 PDFium aborts at startup on kernels with
  a page size other than 4 KiB or 16 KiB (some RHEL/CentOS aarch64 configs use
  64 KiB). Ubuntu/Debian and NVIDIA L4T use 4 KiB and work.
- **Offline builds**: put an aarch64 PDFium SDK (`include/`, `lib/libpdfium.so`)
  somewhere and pass `-DTURBO_PDFIUM_DIR=<dir>`; put a checkout of the pinned
  fastpdf2png commit somewhere and pass `-DTURBO_FASTPDF2PNG_SOURCE_DIR=<dir>`.
  The Docker route also works natively: `scripts/install_fastpdf2png.sh`
  (which runs `scripts/install_pdfium.sh` first) replaces the x86-64 copies in
  `third_party/pdfium` and `bin/` with ones for this machine — both scripts
  check the ELF architecture of what is installed and never keep a mismatch;
  configure with `-DTURBO_BUILD_FASTPDF2PNG=OFF` afterwards.
- The Docker images already do all of this per `TARGETARCH`; the published
  `turboocr-cpu` image is multi-arch (amd64 + arm64).
