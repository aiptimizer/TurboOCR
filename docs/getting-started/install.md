# Install

> **v4.0.0-alpha.1 — nothing is published yet. You build it yourself.**
>
> There is **no PyPI release** (`pip install turboocr` will not get you v4) and
> **no published Docker image**. Every path below builds from this checkout —
> either a `cmake` build, a `docker build`, or a wheel build. The `pip install
> "turboocr[...]"` one-liners appear once the wheels are published; until then
> use `scripts/python/build_backend_wheel.sh`.

Select your hardware and how you want to run it — the command updates below.

<!-- The pills below are built at runtime by docs/javascripts/install-selector.js
     and only work on the docs site. Every command is ALSO written out statically
     under "All install commands" so this page is usable on GitHub, where no
     JavaScript runs. Keep the two IN SYNC: the CONFIG object in the .js file and
     the <details> blocks below must carry the same commands. -->

<div class="phc-installer">
<div class="phc-sel-row">
<div class="phc-sel-label"></div>
<div class="phc-sel-pills" data-row="hw"></div>
</div>
<div class="phc-sel-row">
<div class="phc-sel-label"></div>
<div class="phc-sel-pills" data-row="method"></div>
</div>
<div class="phc-sel-cmdwrap">
<div class="phc-sel-cmd"><code></code></div>
<button type="button" class="phc-sel-copy"></button>
</div>
<div class="phc-sel-meta">
<span class="phc-sel-status"></span>
<span class="phc-sel-note"></span>
</div>
</div>

## Choosing a backend

One binary contains every backend you built, and you pick one when you start the
server:

```bash
./build/turboocr-server --backend apple   # nvidia | apple | intel | amd | cpu
```

`TURBO_BACKEND=apple` does the same thing if you would rather set it in the
environment.

Without the flag the server chooses for you, in this order: **nvidia, amd, apple,
cpu** — the first one that is built in and actually works on the machine.

`intel` is not in that list. It is never chosen automatically, so `--backend
intel` is the only way to run OpenVINO.

`GET /capabilities` on a running server reports which backend it is using.

<h2 class="phc-static">All install commands</h2>

<details class="phc-static" markdown="1" open>
<summary><b>NVIDIA GPU</b> — shipped</summary>

Linux, driver 595+, Turing or newer. ~4 GB VRAM text-only, ~8 GB full pipeline.
(That floor is for **this server image**, built against CUDA 13.3. The Python
wheels are separate artifacts with a lower DRIVER floor — `cuda12` needs only
R525+, same Turing GPU floor — see [Python packages](#python-packages).)

**Docker** (built from this repo):

```bash
docker build -f docker/Dockerfile --target nvidia -t turboocr:nvidia .
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  turboocr:nvidia
```

First start builds TensorRT engines (~90 s on a 5090; `TRT_OPT_LEVEL=3` cuts it
3–5x on older cards) and caches them in the volume. Add stages with
`-e TABLE_BACKEND=slanext`, `-e FORMULA_BACKEND=ppformulanet_s`, `-e OCR_MODEL=medium`.

**Build from source:**

```bash
cmake -B build -DTENSORRT_DIR=/usr/local/tensorrt
cmake --build build -j$(nproc)
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/turboocr-server
```

Needs GCC 13.3+/C++20, CUDA + TensorRT 10.2+, OpenCV 4.x, Drogon 1.9+, gRPC.
Models are auto-fetched into `./models/` on first build.

**Python library:**

```bash
scripts/python/build_backend_wheel.sh cuda12    # or cuda13
pip install build-wheels/cuda12/fixed/*.whl
python -c "import turboocr_engine; print(turboocr_engine.OCR(backend='cuda').read('doc.png'))"
```

Builds the `turboocr-engine-cuda12` wheel from this checkout — the helper also
repairs it, because a bare `pip wheel` only runs on the machine that built it.
NVIDIA ships as **two** wheels, one per CUDA major: `cuda12` needs driver
R525+, `cuda13` needs R580+. Both carry TensorRT 10.15.1.29 and the same
engine; pick by the driver you have (`nvidia-smi`), or let `turboocr doctor`
name it.
`backend='cuda'` is the instant-start CUDA execution provider; `backend='turbo'`
is native TensorRT with a one-time cached engine build.

</details>

<details class="phc-static" markdown="1">
<summary><b>Apple Silicon</b> — testing</summary>

No Docker option: macOS containers have no GPU passthrough, so the Apple backend
runs natively.

**Build from source:**

```bash
brew install cmake opencv drogon jsoncpp protobuf grpc c-ares jpeg-turbo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBO_BACKENDS="cpu;apple"
cmake --build build -j"$(sysctl -n hw.ncpu)"
./build/turboocr-server --backend apple
```

One-time prereqs: full Xcode + Metal toolchain
(`xcodebuild -downloadComponent MetalToolchain`) and an osx-arm64 ONNX Runtime
≥ 1.27 — see [Native build → macOS](native.md). Detection runs on Metal +
MPSGraph; recognition is a GPU + Neural Engine hybrid (narrow crops go to the ANE
via CoreML, in parallel). `TURBO_APPLE_ANE_MAXW` tunes the split (default 800).

**Python library:**

```bash
scripts/python/build_backend_wheel.sh cpu
pip install build-wheels/cpu/fixed/*.whl
python -c "import turboocr_engine; print(turboocr_engine.OCR(backend='apple', replicas=3).read('doc.png'))"
```

Needs the same one-time macOS prereqs as the source build — the wheel compiles
the same C++ tree. The macOS arm64 build of `turboocr-engine-cpu` bundles the
Metal shader library; models auto-download per tier (~6 MB for tiny).
`replicas=3` measured at 94% of the server's multi-replica throughput.

</details>

<details class="phc-static" markdown="1">
<summary><b>Intel CPU / iGPU / Arc</b> — testing</summary>

**Docker** — built from this repo, no published Intel image yet:

```bash
docker build -f docker/Dockerfile --target intel -t turboocr:intel .
docker run --device /dev/dri -p 8000:8000 -p 50051:50051 turboocr:intel
```

`--device /dev/dri` passes the iGPU/Arc into the container; `OV_DEVICE=CPU|GPU|NPU`
picks the device.

**Build from source:**

```bash
cmake -S . -B build-intel -DTURBO_BACKENDS="cpu;intel"
cmake --build build-intel -j$(nproc)
./build-intel/turboocr-server --backend intel
```

**`--backend intel` is required.** With it, the server runs OpenVINO. Without it,
it runs the ONNX Runtime CPU path — even though you just built the Intel backend.

`OV_DEVICE=CPU|GPU|NPU` picks which Intel device OpenVINO runs on; unset it
targets the iGPU/Arc. The OpenVINO runtime must be on `CMAKE_PREFIX_PATH`.

**Python library:**

```bash
scripts/python/build_backend_wheel.sh openvino
pip install build-wheels/openvino/fixed/*.whl
python -c "import turboocr_engine; print(turboocr_engine.OCR(backend='openvino').read('doc.png'))"
```

Builds the `turboocr-engine-openvino` wheel from this checkout (needs the
OpenVINO runtime installed). `OV_DEVICE=CPU|GPU|NPU` picks the device at run time.

</details>

<details class="phc-static" markdown="1">
<summary><b>AMD GPU</b> — not yet hardware-tested</summary>

**Docker** — built from this repo, no published AMD image yet:

```bash
docker build -f docker/Dockerfile --target amd -t turboocr:amd .
docker run --device /dev/kfd --device /dev/dri --group-add video \
  -v ocr-cache:/home/ocr/.cache/turbo-ocr \
  -p 8000:8000 -p 50051:50051 turboocr:amd
```

`/dev/kfd` + `/dev/dri` expose the GPU to ROCm inside the container. First run
compiles ~42 MIGraphX graphs; the named volume persists that cache so only the
first start pays.

**Build from source:**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTURBO_BACKENDS="cpu;amd" \
      -DCMAKE_HIP_ARCHITECTURES="$(rocminfo | grep -om1 'gfx[0-9a-f]*')" \
      -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j$(nproc)
./build/turboocr-server --backend amd
```

The two `CMAKE_*` lines are genuinely required: HIP needs your exact gfx arch,
and ROCm is not on the default prefix path.

HIP kernels + MIGraphX engine. The first run compiles the graphs and caches them
under `~/.cache/turbo-ocr/mgx_*.mxr`; steady state starts instantly.
First-machine checklist: `src/backends/amd/BRINGUP.md`.

**Python library:**

```bash
scripts/python/build_backend_wheel.sh rocm
pip install build-wheels/rocm/fixed/*.whl
python -c "import turboocr_engine; print(turboocr_engine.OCR(backend='rocm').read('doc.png'))"
```

Builds the `turboocr-engine-rocm` wheel from this checkout (needs ROCm on the
host). It compiles clean but has **not** been validated on AMD hardware.

</details>

<details class="phc-static" markdown="1">
<summary><b>CPU only</b> — shipped</summary>

**Docker** — built from this repo:

```bash
docker build -f docker/Dockerfile --target cpu -t turboocr:cpu .
docker run -p 8000:8000 -p 50051:50051 turboocr:cpu
```

No devices to pass through — runs anywhere Docker runs.

**Build from source:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBO_BACKENDS="cpu"
cmake --build build -j$(nproc)
./build/turboocr-server
```

ONNX Runtime + OpenCV, no GPU required. Runs anywhere.

**Python library:**

```bash
scripts/python/build_backend_wheel.sh cpu
pip install build-wheels/cpu/fixed/*.whl
python -c "import turboocr_engine; print(turboocr_engine.OCR().read('doc.png'))"
```

The portable default — works on any machine.

</details>

## First request — identical on every backend

```bash
# Docker serves on 8000 (nginx-fronted); a native/source build binds 8080.
curl -X POST http://localhost:8000/ocr/raw \
  --data-binary @document.png -H "Content-Type: image/png"
```

```json
{"results": [{"text": "Invoice Total", "confidence": 0.97, "bounding_box": [[42,10],[210,10],[210,38],[42,38]]}]}
```

Stages are opt-in per request: `?layout=1`, `?tables=1`, `?formulas=1`
(tables and formulas auto-enable layout). `GET /capabilities` reports what a
running server has loaded; requesting a stage the server was not started with
returns a clear `400`.

## Operating systems

All three build from one CMake tree — `TURBO_BACKENDS` picks the arms, and the
platform differences are handled by feature detection, not by an OS switch.
What differs is **which dependencies you install** and **which GPU paths are
proven**.

| OS | Builds | Unit suite | GPU path proven on hardware |
|---|---|---|---|
| **Linux** x64 | yes | all pass (623) | **NVIDIA** — the reference platform (Arch, CUDA 13.3, TensorRT 10.15.1, RTX 5090 at `sm_120`); see [Build natively](native.md) |
| **Linux** aarch64 | yes | all pass (566) | CPU only — no GPU arm tested on ARM |
| **macOS** arm64 | yes | all pass (572) | **Apple** — Metal + MPSGraph + ANE |
| **Windows** x64 | yes | see [the Windows page](../guides/windows-native-build.md) | **NVIDIA via the ORT CUDA EP** — real inference confirmed on an RTX 5090 |

Counts are the gtest case totals. The GPU-arm figures (620/563) were measured on
hardware 2026-08-10; the `+3` since is the platform-independent
`test_decode_contract` cases added 2026-08-11, re-verified only on macOS (572).
The number moves whenever a test is added — the load-bearing claim is **all
pass**, not the exact total.

The Linux x64 row is `-DTURBO_BACKENDS="nvidia;cpu"`. The plain GPU configure
(`TURBO_BACKENDS` empty — the native TensorRT arm) builds 605/605 instead: it
omits the three tests that exercise the multi-backend seam, because it does not
build the seam.

!!! note "aarch64 needs one extra step"
    `third_party/pdfium/` vendors **linux-x64** and mac-arm64 only, so on ARM
    Linux you must run `scripts/setup/install_pdfium.sh` before configuring — it
    fetches the `linux-arm64` build. Everything else is automatic: the ONNX
    Runtime download follows `TURBO_ARCH`, and the arch flags become
    `-march=armv8.2-a`. The aarch64 row (566 cases as of 2026-08-12) was measured on Ubuntu 24.04,
    `-DTURBO_BACKENDS="cpu" -DBUILD_SERVER=OFF`.

!!! warning "Read this before pointing any build at a GPU"
    A GPU request that quietly runs on the CPU is the failure mode to guard
    against — same numbers, an order of magnitude slower, no error. The engine
    now refuses instead (`session.disable_cpu_ep_fallback`), but the version
    pairing is on you:
    [GPU providers fail loudly](../reference/configuration.md#gpu-providers-fail-loudly).

**Not yet proven on hardware**, and honestly labelled as such rather than
implied by the table above: **TensorRT** on Windows (the NVIDIA seam backend
still uses POSIX headers — a separate port), and **AMD ROCm** anywhere
(`src/backends/amd/BRINGUP.md`).

The Intel **NPU** now has real measurements, and they are discouraging enough
to state up front: the NPU plugin accepts **only fully static shapes** (our
det/rec are exported dynamic and fail to compile as-is), the **layout model is
rejected outright** on op support even when reshaped, and once static, det/rec
run *slower* than both the CPU and the iGPU on every model except the medium
detector. The NPU is also unreachable from WSL2 entirely — native Windows or
bare-metal Linux only. Details and the full matrix:
`src/backends/intel/SETUP.md` §0b. Treat `OV_DEVICE=NPU` as unproven for
production throughput.

### Linux

The supported path is one script — it installs the system packages, Drogon,
ONNX Runtime, PDFium and TensorRT, then configures:

```bash
bash scripts/setup/install_native.sh
```

[Build natively](native.md) documents everything that script does, so you can
reproduce it by hand on a non-Arch distribution.

### macOS (arm64)

Needs full Xcode (for the Metal toolchain — Command Line Tools alone is not
enough), then:

```bash
brew install cmake ninja opencv drogon jpeg-turbo
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTURBO_BACKENDS="cpu;apple"
cmake --build build -j
./build/turboocr-server --backend apple
```

That is the whole thing — verified end to end from a clean clone on
2026-08-10: it configures, downloads ONNX Runtime and the ~1.7 GB PP-OCRv6
model set itself, builds `turboocr-server`, passes the full suite (572 cases), and answers
`POST /ocr/raw`. The first build is long mostly because of that download.

`onnxruntime` is deliberately **not** in the brew line. The build fetches its
own pinned `osx-arm64` 1.28.0, and will reject a system ONNX Runtime older than
API 27 even if you install one — Homebrew's formula is 1.24, whose CoreML EP
returns NaN for every layout score, so a build against it reports every page as
having no layout, silently. Install it anyway if you like; the version check
decides.

`TURBO_BACKENDS` defaults to `cpu;apple` here, so the Metal backend is in by
default. Without exported MPSGraph artefacts the Apple backend runs its ONNX
fast path on CoreML, which is a normal, supported configuration. The PDF page
renderer uses in-process PDFium (no daemon pool) — there is no inotify on
darwin, and CMake selects that arm automatically.

### Windows (x64)

The full walkthrough, including every platform fix the port needed, is
[Build on Windows](../guides/windows-native-build.md). The short version:

```powershell
winget install Kitware.CMake
winget install Microsoft.VisualStudio.2022.BuildTools `
  --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Then unpack OpenCV, ONNX Runtime (`onnxruntime-win-x64`) and PDFium
(`pdfium-win-x64` — note it is `pdfium.dll.lib`, an import library), and take
libcurl + libjpeg-turbo from vcpkg:

```powershell
git clone --depth 1 https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
C:\vcpkg\vcpkg.exe install curl:x64-windows libjpeg-turbo:x64-windows
```

Configure from an **x64 Native Tools Command Prompt** (otherwise `cl.exe` is
not on `PATH`):

```powershell
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DTURBO_BACKENDS="cpu" -DBUILD_SERVER=OFF ^
  -DOpenCV_DIR=C:/opencv/build ^
  -DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows
cmake --build build-win --target turbo_ocr_tests
```

`BUILD_SERVER=OFF` skips `turboocr-server`, which needs Drogon + gRPC — neither
is required for the library, the backends or the test suite. Windows **can**
rasterize PDFs: it compiles the in-process renderer (the same one macOS uses),
and a `BUILD_SERVER=ON` build serves `/ocr/pdf` — see
[Build on Windows](../guides/windows-native-build.md). Only the multi-process
*daemon* renderer is Linux-only (it needs `inotify`/`pipe2`).

The runtime DLLs must sit beside the `.exe` or on `PATH`: `pdfium.dll`,
`onnxruntime.dll`, `opencv_world*.dll`, and vcpkg's `libcurl.dll` /
`turbojpeg.dll` / `z.dll`.

## Python packages

The Python side is **one pure-Python package plus one engine wheel per hardware
target**. `turboocr` is the umbrella: the typed client for a TurboOCR server,
and the facade over the in-process engine. Its backend extras pin the matching
engine wheel — install exactly one backend; the engine wheels are mutually
exclusive, each baking a different ONNX Runtime execution provider into the
same native extension.

| Hardware | Install | Engine wheel it pins |
|---|---|---|
| *(none — client only)* | `pip install turboocr` | — |
| CPU — any x86-64 / ARM64 (**the default**) | `pip install "turboocr[cpu]"` | `turboocr-engine-cpu` |
| Apple Silicon — Metal + Neural Engine | `pip install "turboocr[cpu]"` | `turboocr-engine-cpu` (its macOS arm64 build) |
| NVIDIA GPU — driver R525+ | `pip install "turboocr[cuda12]"` | `turboocr-engine-cuda12` |
| NVIDIA GPU — driver R580+ | `pip install "turboocr[cuda13]"` | `turboocr-engine-cuda13` |
| Intel CPU / iGPU / Arc / NPU | `pip install "turboocr[openvino]"` | `turboocr-engine-openvino` |
| AMD GPU (ROCm) | `pip install "turboocr[rocm]"` | `turboocr-engine-rocm` |

There is no Apple engine wheel: the macOS arm64 `turboocr-engine-cpu` wheel is
built with the Apple backend and bundles the Metal shader library.
`turboocr doctor` inspects the machine and prints the right line for it.

Feature extras — `[pdf]` (read PDFs, write searchable PDFs), `[rich]` (prettier
`doctor` panel), `[pandas]` (`PageResult.to_pandas()`), `[all]` — combine with
any backend: `pip install "turboocr[cuda,pdf]"`, or on a bare engine wheel,
`pip install "turboocr-engine-cpu[all]"`.

!!! warning "The engine wheels are not published yet"
    `pip install turboocr` today resolves to this project's published 0.3.0
    client SDK, which the umbrella continues — but the `turboocr-engine-*`
    distributions have no PyPI release until the first release run of
    `.github/workflows/wheels.yml`, so the backend extras do not resolve yet.

    **Build from source** — always current, matches the host exactly. Use the
    helper script: it builds *and* repairs the wheel, which is the part that
    makes it installable anywhere.

    ```bash
    # <variant> is one of: cpu | cuda | openvino | rocm
    # (cpu builds turboocr-engine-cpu — also the Apple wheel on macOS arm64)
    scripts/python/build_backend_wheel.sh cpu
    pip install build-wheels/cpu/fixed/*.whl
    ```

    A bare `pip wheel python/` is **not** enough on its own — see
    [the repair step](#the-repair-step-is-not-optional) below.

!!! note "`4.0.0a1` is a pre-release — pip skips it by default"
    Once the wheels are published, a plain `pip install "turboocr[cpu]"` still
    resolves to the newest *stable* release, not the alpha. Ask for it:

    ```bash
    pip install --pre "turboocr[cpu]"
    pip install "turboocr[cpu]==4.0.0a1"    # equivalent, explicit
    ```

    Verify what you got: `pip show turboocr-engine-cpu` (or your variant) and
    `turboocr doctor` report the installed version and the provider actually
    selected.

### The repair step is not optional

`pip wheel python/` produces a wheel that runs **only on the machine that built
it**. It bundles zero shared libraries, and its RPATH points back into your dev
checkout — move it to another machine, or just delete `build/`, and the import
fails. Making it self-contained is a separate *repair* step that vendors the
~114 dylibs/shared objects it depends on (OpenCV, ONNX Runtime, PDFium, …).

`scripts/python/build_backend_wheel.sh` does build **and** repair, which is why
it is the recommended path. If you drive `pip wheel` by hand, run the repair
yourself and install the repaired wheel:

```bash
# macOS
pip wheel python/ --no-deps -w dist/
pip install delocate
delocate-wheel -w dist/fixed -v dist/turboocr_engine_*.whl
pip install dist/fixed/turboocr_engine_*.whl

# Linux
pip wheel python/ --no-deps -w dist/
pip install auditwheel
auditwheel repair -w dist/fixed dist/turboocr_engine_*.whl
pip install dist/fixed/turboocr_engine_*.whl
```

For the `cuda` and `rocm` variants the repair must **exclude** the driver and
toolkit sonames (`libcuda.so.1`, `libcudart`, `libcudnn`, `libnvinfer`,
`libamdhip64`, `libmigraphx`, …) so they come from the host, exactly as
`onnxruntime-gpu` ships — and the CUDA wheel needs a second pass, because ORT
`dlopen`s its provider libraries and `auditwheel` cannot see them in the
`DT_NEEDED` graph. The helper script already encodes both; hand-rolling them is
the part that goes wrong.

## Backend details

??? info "NVIDIA — `turboocr-engine-cuda12` / `-cuda13`, and what the first run costs"

    Pick the wheel by **GPU generation first, then driver**:

    | | `cuda12` | `cuda13` |
    |---|---|---|
    | oldest GPU | Turing (sm_75) | Turing (sm_75) |
    | driver | **R525+** | **R580+** |

    Both wheels need **Turing or newer** — the same floor as the server. It is
    not a CUDA-major difference: TurboOCR's own kernels require it (the
    connected-components pass uses a cooperative-groups grid sync, which needs
    compute capability 6.0+, and the project pins sm_75 as its floor), and
    CUDA 13 drops pre-Turing support outright. Both are compiled for
    sm_75/80/86/89/90/120, so every card from an RTX 20 to a Blackwell is
    native, and the TensorRT engine is built at run time for whichever is
    present.

    **So choose by driver, not by card.** An RTX 3090 on driver R535 needs
    `cuda12`; the same card on R580 can take either.

    The CUDA, cuDNN and TensorRT runtimes are **not** bundled: the repair step
    excludes every one of those sonames, exactly as `onnxruntime-gpu` ships.
    They come from the system CUDA install, or from the matching pip packages,
    which the wheel **finds automatically** — no `LD_LIBRARY_PATH` needed. On a
    driver-only machine:

    ```bash
    pip install tensorrt-cu12-libs==10.15.1.29 nvidia-cuda-runtime-cu12 nvidia-nvjpeg-cu12   # or the -cu13 equivalents
    ```

    They are not declared as dependencies of the engine wheel, so nothing
    CUDA-sized lands on machines that install it for its API surface alone.

    It carries two NVIDIA paths, and which one you get is a `backend=` choice
    with very different startup behaviour:

    | `backend=` | Start-up | Steady state |
    |---|---|---|
    | `"auto"` (**the default**) → resolves to `"turbo"` | **Slow on the FIRST run only** — builds a TensorRT engine | Fastest: peak throughput |
    | `"cuda"` | **Instant** — nothing is compiled | Fast: the ONNX graph on the CUDA execution provider |
    | `"turbo"` (aliases `"tensorrt"`, `"trt"`) | **Slow on the first run only** — builds a TensorRT engine | Fastest: peak throughput |

    The nvidia backend is compiled into these wheels, so `backend="auto"` picks
    the native TensorRT path — it does **not** start instantly. The first
    `OCR()` call builds an engine (~90 s on an RTX 5090, longer on older cards)
    and caches it; the process prints a one-time notice while it does. Ask for
    `backend="cuda"` when you need an instant first start.

    ```python
    turboocr.OCR(backend="cuda")    # instant start, ONNX Runtime CUDA EP
    turboocr.OCR(backend="turbo")   # TensorRT — first run builds, then cached
    ```

    `backend="turbo"` compiles a TensorRT engine specialised for your exact GPU,
    driver and model. That build is a **one-time cost**, not a per-run one: the
    engine is written to an on-disk cache and every later run loads it in a
    fraction of a second. Budget roughly ~90 s on an RTX 5090 and up to an hour
    on older cards; `TRT_OPT_LEVEL=3` cuts it 3–5x.

    The cache directory is `TRT_ENGINE_CACHE`, defaulting to
    `~/.cache/turbo-ocr`. Point it somewhere persistent — or mount it as a
    volume in a container — and the build is paid once per machine rather than
    once per process. The cache is invalidated by a GPU, driver, TensorRT or
    model change, which correctly triggers one more rebuild.

    Note the asymmetry with the **server**, whose native NVIDIA arm uses
    TensorRT by default — that is why the Docker quick-start warns about a slow
    first start. The Python wheel deliberately defaults the other way: no build,
    running the second you have the wheel. Opt in with `backend="turbo"` when
    you want peak throughput and can absorb one slow start.

??? info "Apple Silicon — Metal + Neural Engine"

    One-time setup before the build command above: full Xcode with the Metal
    toolchain, the Homebrew package line from the selector, and an osx-arm64
    ONNX Runtime ≥ 1.27 — the checklist lives in
    [Native build → macOS](native.md#macos-cpu-apple-backend).

    Detection and warp run on the GPU (Metal + MPSGraph); recognition is a
    GPU + Neural Engine hybrid, with narrow crops on the ANE through CoreML
    in parallel with the GPU. `turbo_apple.metallib` is compiled next to the
    binary and found automatically. Layout, tables, formulas and autorotate
    all work once their models are supplied (`--layout-onnx
    models/layout.onnx`, `DOC_ORI_ONNX=…`, `TABLE_BACKEND=…`,
    `FORMULA_BACKEND=…`).

    | Knob | Meaning |
    |---|---|
    | `TURBO_APPLE_ANE_MAXW` | GPU/ANE split point (default 800; `0` = GPU only) |

    → `src/backends/apple/README.md`

??? info "Intel — OpenVINO"

    One backend covers Intel CPUs, integrated GPUs, Arc and NPUs. By default
    it runs on the **integrated GPU / Arc** — `OV_DEVICE=GPU`, which resolves
    to `GPU.0`. Set `OV_DEVICE=CPU` or `OV_DEVICE=NPU` to pin a different
    device.

    If the machine also has a **discrete card**, note that OpenVINO enumerates
    it under the same GPU plugin (`GPU.1`), and plain `GPU` always means
    `GPU.0` — the integrated one. Pass the index (`OV_DEVICE=GPU.1`) when you
    want the other one; otherwise you may benchmark silicon you did not
    intend. Run `python -c "import openvino as ov; c=ov.Core(); print([(d,
    c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices])"` to
    see which index is which.

    The models can execute two ways — `TURBO_ENGINE_MODE` picks (default
    `auto` = use `native` when it comes up):

    - **`native`** (alias `ultra`): OpenVINO compiles and runs the models
      itself. Fastest on Intel silicon; the first start pays a one-time model
      compile — set `OV_CACHE_DIR` to pay it once.
    - **`onnx`** (alias `fast`): ONNX Runtime running with the OpenVINO
      execution provider underneath. Starts faster but runs measurably
      slower on the same chip — the fallback, not the recommendation.

    Leave `OV_PERF_HINT` on its `latency` default: `throughput` makes this
    server *slower* (measured 2.4 vs 5.5 img/s — the server issues one
    inference at a time, and the throughput hint parks that single request
    on one shared stream).

    → `src/backends/intel/README.md`

??? info "AMD — ROCm"

    HIP kernels plus a MIGraphX inference engine, with a per-architecture
    `.mxr` compile cache so model compilation is paid once. Not yet
    hardware-tested; the first-machine checklist is
    `src/backends/amd/BRINGUP.md`.

??? info "Python library"

    The same C++ pipeline behind a native wheel (nanobind, GIL released
    during inference) — models auto-download per tier (~6 MB for `tiny`)
    with SHA256 verification. `backend=` picks `"cuda"`, `"apple"`,
    `"openvino"`, `"rocm"`, `"cpu"`; `OCR(replicas=N)` fans work across a
    built-in replica pool. Which providers are available depends on which
    wheel you installed — see [Python wheels](#python-wheels).

    → `python/README.md` · `python/DESIGN.md`

→ [Docker & deployment in depth](docker.md) · [Build guide](native.md) · [HTTP API](../reference/http.md) · [Configuration](../reference/configuration.md)
