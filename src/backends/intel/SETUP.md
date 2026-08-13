# Intel backend — bring-up on real Intel hardware

Everything here was executed and verified on **2026-07-22** on an Intel Tiber AI
Cloud bare-metal instance: **Core Ultra 7 265T** (Arrow Lake, 20 cores / 20
threads), Intel Graphics `[0x7d67]` iGPU, 31.4 GB RAM, Windows 11 Pro build
26200 with **WSL2 / Ubuntu 26.04 / kernel 6.18**.

This is the "port it there, not here" document that `the root CMakeLists.txt`
points at when it hard-fails on `TURBO_BACKENDS=intel`. Machine-specific access
credentials deliberately live **outside the repo**, in `~/.turboocr_intel/ACCESS.md`.

---

## 0. What the hardware actually exposes

Measure this first on any new Intel box — it decides which of the three engines
you can target.

| Engine | Available under WSL2? | How to confirm |
|---|---|---|
| 20 CPU cores | yes | `nproc` |
| Intel iGPU | **yes** | `clinfo -l` → `Intel(R) Graphics [0x7d67]` |
| AI Boost NPU | **no** | `/dev/accel` missing |

Re-confirmed 2026-08-12 on a second Arrow Lake box (Core Ultra 9 285K, Windows
11 + WSL2/Ubuntu 24.04): `/dev/accel` and `/dev/dri` both absent, only
`/dev/dxg`; a bare `pip install openvino` in WSL enumerates `['CPU']`. Same
conclusion, independently: the NPU never reaches WSL, and a CPU-only device
list there is **not** evidence about the iGPU (see the driver note below).

Two counter-intuitive results worth recording, because both cost time:

**The iGPU works even though `/dev/dri` does not exist.** On native Linux the
Intel compute runtime binds `/dev/dri/renderD128`. WSL2 never creates that node —
it exposes `/dev/dxg` instead — so the reasonable expectation is that Level Zero
and OpenCL are dead in WSL. They are not: the 26.05 compute runtime reaches the
GPU through the dxcore shim. **Do not conclude "no GPU" from a missing
`/dev/dri`.** Enumerate and see.

Note also that `/usr/lib/wsl/lib/` contains only `libd3d12.so`, `libd3d12core.so`
and `libdxcore.so` — no Level-Zero or OpenCL libraries. Unlike NVIDIA, which
injects `libcuda.so` there, the Intel stack comes entirely from the distro
packages below. An empty-looking `/usr/lib/wsl/lib` is not a problem either.

**The NPU is genuinely unavailable under WSL2**, and no package fixes it. There
is no `intel_vpu` device to bind and an open Microsoft feature request
([microsoft/WSL#40842](https://github.com/microsoft/WSL/issues/40842)) asking for
NPU passthrough. Reaching AI Boost requires a **native Windows** OpenVINO build
or a bare-metal Linux install — not WSL.

---

## 0b. What the NPU does once you reach it (measured, native Windows)

Measured **2026-08-12** on the Core Ultra 9 285K box, native-Windows Python +
`pip install openvino` 2026.3, driving **our own exported ONNX** directly (no
TurboOCR build needed). `Core().available_devices` natively returns
`['CPU', 'GPU.0', 'GPU.1', 'NPU']` — all three families reachable outside WSL.

!!! warning "`GPU` is ambiguous on a box with a discrete card — pin the index"
    `FULL_DEVICE_NAME` on this machine (verified directly, twice):

    ```
    CPU   | Intel(R) Core(TM) Ultra 9 285K
    GPU.0 | Intel(R) Graphics (iGPU)
    GPU.1 | NVIDIA GeForce RTX 5090 (dGPU)
    NPU   | Intel(R) AI Boost
    ```

    OpenVINO's GPU plugin is usually described as Intel-only, but on native
    Windows it **also enumerates the NVIDIA discrete card** (presumably through
    the DXCore/D3D12 interop path, not a native NVIDIA plugin). `OV_DEVICE=GPU`
    resolves to `GPU.0` — the iGPU — so the GPU column of the matrix below is
    the Xe iGPU. On any box with a dGPU, **name the index explicitly**
    (`GPU.0`/`GPU.1`) rather than trusting the bare `GPU` alias, or you may
    silently benchmark the wrong silicon.

**Finding 1 — the NPU requires fully static shapes. This is a hard gate, not a
tuning matter.** Every dynamic-shape compile failed, on all four models, with
the identical error:

```
[NPU_VCL] Compiler returned msg:
Upper bounds were not specified, got the default value - '9223372036854775807'
  (Level0 pfnCreate2 -> ZE_RESULT_ERROR_INVALID_ARGUMENT, code 0x78000004)
```

Our det (`[?,3,?,?]`) and rec (`[?,3,48,?]`) are both exported dynamic, so
**neither compiles on the NPU as-is**. CPU and GPU accept the dynamic form
happily. After `reshape()` to a static shape, all four **OCR** models compiled
on the NPU without error — for det/rec this is a shape contract, not an
unsupported-op wall.

**Finding 2 — `layout.onnx` cannot run on the NPU at all.** Reshaping does NOT
save it, unlike det/rec; the NPU compiler rejects an operator outright:

```
[NPU_VCL] Compiler returned msg:
String attribute reduction is not supported
```

So PP-DocLayoutV3 (DETR-family) would have to fall back to CPU/GPU in any
NPU-routed pipeline. Any "run the pipeline on the NPU" plan is a non-starter
for the layout stage specifically.

**Finding 3 — the NPU is not a throughput win on the models it does run.**
Warm single-stream forward pass, static shapes:

| Model | shape | CPU | GPU (iGPU) | **NPU** |
|---|---|---:|---:|---:|
| `det_tiny` | 1×3×480×480 | **379.9 img/s** (2.63 ms) | 302.2 (3.31 ms) | 118.7 (8.42 ms) |
| `rec_tiny` | 1×3×48×320 | 441.2 (2.27 ms) | **530.6 img/s** (1.88 ms) | 373.6 (2.68 ms) |
| `det` (medium) | 1×3×960×960 | 7.32 (136.6 ms) | 8.16 (122.5 ms) | **8.32 img/s** (120.2 ms) |
| `rec` (medium) | 1×3×48×320 | 105.1 (9.51 ms) | **130.5 img/s** (7.66 ms) | 92.5 (10.81 ms) |
| `layout` | 1×3×800×800 | **6.85 img/s** (145.9 ms) | 5.31 (188.2 ms) | **rejected** |

The NPU is **3.2× slower than the CPU** on the tiny detector, loses on both
recognizers, and only edges ahead on the medium detector (+14% over CPU, +2%
over the iGPU — inside noise). NPU compile time is, however, much cheaper than
the GPU's (1.5 s vs 6.2 s for `det_tiny`; the GPU's cold compile runs 6–26 s).

**Read:** an NPU path is *feasible for det/rec* — the static-shape machinery
this backend already has for the GPU (`prebuild()` / per-canvas static compile)
is exactly what the NPU needs — but on throughput grounds it does not pay for
itself, and layout can never join it. Its plausible value is power draw and
leaving the CPU free for the host-side pipeline, neither of which is measured
here. **Do not spend integration effort on NPU for speed.** If someone does
pursue it, the work is: bounded/static shapes for det canvases and the rec
width ladder, one compiled blob per shape, plus a CPU/GPU fallback for layout.

!!! note "Reading the matrix"
    Only the **dynamic-compile pass/fail** and the **static benchmark** numbers
    above are trustworthy. The probe's dynamic-*bench* cells failed on every
    device including CPU/GPU with `to_shape was called on a dynamic shape` —
    that was a bug in the probe (introspecting `.shape` on a still-dynamic
    compiled model), not a device limitation.

### Would WSL 3 change the NPU verdict? (researched 2026-08-12)

Probably not for us, and it is not available on these boxes today.

- **WSL 2 has no NPU passthrough at all** — confirmed by
  [microsoft/WSL#40842](https://github.com/microsoft/WSL/issues/40842) (open
  feature request) and
  [microsoft/WSL#40445](https://github.com/microsoft/WSL/issues/40445), whose
  reported symptom is exactly ours: *"OpenVINO inside WSL2 reports
  available_devices = ['CPU'] even though the NPU is enumerable at the D3DKMT
  layer."* The NPU is visible but not driveable. The fix people want — surface
  it as `/dev/accel/accel0` so the in-tree IVPU driver
  (`CONFIG_DRM_ACCEL_IVPU`) can claim it — is a proposal, not shipped.
- **WSL 3** (Build 2026 preview, Windows Insiders Dev/Beta) replaces the
  Hyper-V backend with a paravirtualized machine and does add GPU **and NPU**
  passthrough, incl. Intel Core Ultra Series 3.
- **The catch:** WSL 3 routes the NPU through **DirectML 2.0**, not the
  IVPU/Level-Zero stack that **OpenVINO's NPU plugin** binds to. So WSL 3
  would likely unlock NPU for DirectML-based runtimes (ORT DirectML EP), not
  necessarily `OV_DEVICE=NPU`. Nobody in those threads claims OpenVINO-NPU
  works inside WSL.

Both test boxes are on Windows 25H2 (build 26200) with WSL 2.6.3.0 — not
Insider builds, so WSL 3 is not installable there as-is. Given Finding 3, none
of this is worth chasing for throughput anyway.

---

## 1. WSL2 itself

Skip if you are on native Linux.

`wsl.exe` in `System32` may be only the **inbox stub** (check `wsl --version`;
a stub reports `10.0.26100.1` and answers every command with "The Windows
Subsystem for Linux is not installed"). In that state `wsl --install` **cannot
bootstrap itself** — it just reprints the same message. Install explicitly:

```powershell
# 1. enable the two features (Administrator), then REBOOT
dism /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
shutdown /r /t 5

# 2. after the reboot, install the real WSL package from GitHub
#    (find the current asset via api.github.com/repos/microsoft/WSL/releases/latest)
Invoke-WebRequest -UseBasicParsing -OutFile $env:TEMP\wsl.msi `
  https://github.com/microsoft/WSL/releases/download/2.7.10/wsl.2.7.10.0.x64.msi
msiexec /i $env:TEMP\wsl.msi /quiet /norestart

# 3. distro
wsl --install -d Ubuntu --no-launch
```

Run everything below as root inside the distro: `wsl -d Ubuntu -u root -e bash <script>`.

### Driving WSL over SSH

!!! warning "The WSL VM shuts down between idle SSH calls — `/tmp` does not survive"
    Observed 2026-08-12: scripting the box over one-shot SSH commands lets the
    lightweight WSL VM idle-shut-down between calls, which **wipes anything
    under `/tmp`** and makes a multi-step script look like it randomly lost
    state. Put working files under `/home/<user>/…` instead.

The Windows OpenSSH default shell is `cmd.exe`, so commands chain with `&`, not
`;`, and quoting a bash script through `ssh host "wsl -e bash -lc '...'"` breaks
on almost anything non-trivial. Write the script to a file and execute it by
path. `~/.turboocr_intel/rx` does exactly that and is worth recreating on any new
machine. PowerShell output also carries UTF-16 nulls — pipe through `tr -d '\000'`.

---

## 2. Fix the apt mirror FIRST

On this instance `archive.ubuntu.com` returned **0 B/s** while
`azure.archive.ubuntu.com` sustained **12.6 MB/s**. The symptom is apt appearing
to hang for tens of minutes with no error. Always measure before assuming a slow
link:

```bash
curl -s -o /dev/null -w "%{speed_download} B/s\n" --max-time 15 \
  http://archive.ubuntu.com/ubuntu/ls-lR.gz
```

If it is slow:

```bash
sed -i 's|http://archive.ubuntu.com/ubuntu/|http://azure.archive.ubuntu.com/ubuntu/|g; \
        s|http://security.ubuntu.com/ubuntu/|http://azure.archive.ubuntu.com/ubuntu/|g' \
  /etc/apt/sources.list.d/ubuntu.sources
apt-get update
```

---

## 3. Build dependencies

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get install -y \
  build-essential cmake ninja-build pkg-config rsync curl unzip \
  libopencv-dev python3-pip python3-venv \
  libdrogon-dev libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev protobuf-compiler \
  libturbojpeg0-dev libjsoncpp-dev uuid-dev zlib1g-dev libssl-dev \
  libpq-dev libmariadb-dev libsqlite3-dev libhiredis-dev \
  libbrotli-dev libyaml-cpp-dev libc-ares-dev \
  libcurl4-openssl-dev
```

The long tail after `libdrogon-dev` is not optional: Ubuntu's Drogon package ships
a CMake config with `find_dependency` on PostgreSQL, MariaDB, SQLite, Brotli,
hiredis and yaml-cpp, and **configure fails on each one in turn** until all are
present. Read `/usr/lib/*/cmake/Drogon/DrogonConfig.cmake` rather than
discovering them one build at a time.

ONNX Runtime needs no manual step — CMake fetches the Linux x64 prebuilt.

---

## 4. Intel GPU compute runtime

```bash
apt-get install -y intel-opencl-icd libze-intel-gpu1 libze1 clinfo intel-gpu-tools
```

Verify — **both** must succeed before any GPU work is meaningful:

```bash
clinfo -l
#   Platform #0: Intel(R) OpenCL Graphics
#    `-- Device #0: Intel(R) Graphics [0x7d67]

python3 -m venv /root/ovenv && /root/ovenv/bin/pip install openvino
/root/ovenv/bin/python -c "import openvino as ov; print(ov.Core().available_devices)"
#   ['CPU', 'GPU']
```

Before these packages are installed OpenVINO reports `['CPU']` only, which reads
exactly like "the GPU is not available in WSL". It is a missing driver, not a
missing capability.

---

## 5. Build

```bash
cmake -S . -B build-intel -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DTURBO_BACKENDS=cpu \
  -DFETCH_MODELS=OFF \
  -DTURBO_FUNSD_CACHE=/root/funsd_cache
cmake --build build-intel -j"$(nproc)"
```

Two flags matter (`-DUSE_CPU_ONLY=ON` is no longer among them — naming a
non-nvidia `TURBO_BACKENDS` now selects the host build path automatically, so
the configure no longer hunts for `nvcc`):

- `TURBO_BACKENDS=cpu` — **without it the backend arm returns early and silently
  builds no harness at all**. The main tree still builds, so this looks like
  success.
- `FETCH_MODELS=OFF` — stops it re-downloading models you already have.

Products land in `build-intel/`: `turbo_bench`, `turbo_conformance`,
`turbo_golden`, `turbo_backend_probe`, `turboocr-server`.

`TURBO_BACKENDS=intel` builds `turbo_ocr_backend_intel` (root `CMakeLists.txt`).
It requires SYCL: the configure hard-fails when `TURBO_OCR_HAS_SYCL` is off,
rather than silently producing a backend with no device kernels. The off-hardware
functional test of the engine is `probes/ov_engine_probe.cpp`.

---

## 6. Verify

```bash
./build-intel/turbo_backend_probe
#   available_backends() [1]: cpu
#   cpu -> backend='cpu' device='host' async=0 pool=20

./build-intel/turbo_bench --backend cpu --tier tiny \
  --images /root/funsd_cache --count 50 --repeat 4 --threads 10
```

`turbo_bench` refuses to report a throughput number from a window shorter than
15 s and cross-checks two independent clocks, so use `--repeat` to lengthen the
window rather than trusting a fast single pass. It is the same binary and the
same protocol on Intel, Apple and NVIDIA — that is what makes these numbers
comparable across machines.

---

## 7. Measured baseline (2026-07-22, ORT CPU backend)

FUNSD 50, tiny tier, 200 images/run, all windows ≥ 15 s with wall-clock skew
within tolerance.

| replicas × intra-op threads | img/s | mean latency |
|---|---|---|
| 10 × 2 | **9.0** | 1089 ms |
| 20 × 1 | 8.1 | 2365 ms |
| 20 × 2 | 7.5 | 2574 ms |
| 5 × 4 | 7.1 | 694 ms |
| 20 × 4 (default) | 5.0 | 3843 ms |
| 1 × 20 | 2.3 | 433 ms |

**F1 = 85.78% in every single configuration** — identical to 2 decimal places.
Threading changes speed only; accuracy is invariant. For reference the Apple
backend reaches 85.70% and the NVIDIA reference ~85.4%, so the shared pipeline
is landing on the same accuracy across a third platform.

Two conclusions:

- `CpuEngine` defaults to `SetIntraOpNumThreads(4)`, so the default 20-replica
  configuration requests **80 ORT threads on 20 cores** and loses 45% of
  throughput to oversubscription. Keep `replicas × ORT_NUM_THREADS ≈ nproc`.
- At the 10×2 optimum the CPU measures **95–99% busy**, so 9 img/s is a real
  compute ceiling for ORT-CPU on this part, not a scheduling artifact. Beating it
  requires a different engine (OpenVINO CPU/GPU), not more tuning.

`ORT_EP=xnnpack` is **not usable**: the official ONNX Runtime prebuilt is not
compiled with XNNPACK, and every model load fails with "XNNPACK execution
provider is not supported in this build."

---

## 8. The OpenVINO backend: measured result (2026-07-22)

`TURBO_BACKENDS=intel` now builds and runs. Add `-DOpenVINO_DIR=<pip>/openvino/cmake`
— the OpenVINO **pip wheel ships a complete C++ SDK** (headers, libs,
`OpenVINOConfig.cmake`, and the GPU *and* NPU plugins), so no oneAPI install is
needed. SYCL stays off; pre/post runs on the shared host kernels.

Runtime configuration (all env, no rebuild — the same binary runs CPU-heavy on an
iGPU part and GPU-heavy on a discrete Arc):

| var | values | note |
|---|---|---|
| `OV_DEVICE` | CPU / GPU / NPU | NPU unavailable under WSL2 |
| `OV_PERF_HINT` | throughput / latency / none | default **latency** (2026-08-03: throughput starved the sync engine — one request = one stream; 2.4 vs 5.5 img/s on a 13600K) |
| `OV_NUM_STREAMS` | int | `OV_PERF_HINT=none` to use it |
| `OV_INFER_PRECISION` | f16 / f32 | GPU: f16 measured 1629 vs 1147 crops/s |
| `OV_REC_DYNAMIC_BATCH` | 0 / 1 | one artefact per width instead of per rung |
| `TURBO_INTEL_DEBUG` | 0 / 1 | per-stage tensor stats + timings |

### Correctness: PASSES

Per-stage golden diff vs the CPU reference (`turbo_golden --backend intel --ref cpu`):

| stage | agreement | tolerance | |
|---|---|---|---|
| det | 1.0000 | 0.93 | OK |
| cls | 1.0000 | 0.98 | OK |
| rec | 0.7570 | 0.65 | OK (bilinear resampling differs per device) |

End-to-end **F1 = 85.52%** (ORT 85.78%, Apple 85.70%, NVIDIA ref ~85.4%), stable
across every configuration below — threading and artefact strategy change speed
only, never accuracy.

### Speed: OpenVINO is currently SLOWER end-to-end than ORT

| config | img/s | peak RSS |
|---|---|---|
| ORT (`--backend cpu`, 10 replicas x 2 threads) | **8.8** | — |
| OpenVINO static per-rung, r1 | 4.3 | 4692 MB |
| OpenVINO dynamic-batch, r1 | 4.1 | 2917 MB |
| OpenVINO dynamic-batch, r4 | 4.1 | 11041 MB |
| OpenVINO dyn + throughput hint, r1 | 3.2 | 12468 MB |
| OpenVINO dyn + throughput hint, r2 | 3.9 | 24790 MB |

**Do not read this as "OpenVINO is slow on Intel".** Per-model it is much faster
than ORT. The gap is that the engine is called SYNCHRONOUSLY and cannot use the
parallelism its own throughput figures depend on.

Per-page breakdown at 249 ms (`TURBO_INTEL_DEBUG=1`, single replica):

```
det infer   31 ms
det post     6 ms
rec infer  145 ms   (5.9 batches/page)
host pre/post ~67 ms  (warp_crops, resize_normalize, argmax, CTC)
```

The 145 ms is NOT an engine defect. `benchmark_app` measured rec at **2144
crops/s with the throughput hint** but only **476 crops/s at `streams=1`** — and
476 crops/s x 53 crops/page = 111 ms, which is what we observe. The headline
number needs multiple streams and several in-flight requests.
`OpenVINOEngine::run()` still RETURNS synchronously (`caps().async == false`,
see the header: OpenVINO's GPU plugin runs on its own internal stream, so
returning early would race the caller's DeviceQueue contract) — but WITHIN a
call it now fans a dynamic-batch request across the InferRequest pool with
`start_async` and joins before returning, which is what those in-flight
requests are. These paragraphs' measurements predate that; re-measure on
hardware (Open items).

That is also why the throughput hint makes things WORSE here (3.2 vs 4.1): it
allocates per-stream scratch for every artefact — hence the 12-25 GB — while a
synchronous caller keeps only one stream busy. Paying the memory, getting none of
the parallelism.

### FULL GPU path: works, correct, and it SCALES

`OV_DEVICE=GPU` puts det + cls + rec entirely on the Intel GPU. No hybrid, no
device splitting — one device runs the whole pipeline, which is also what makes
the same binary correct on a discrete Arc.

| config | img/s | F1 | peak RSS |
|---|---|---|---|
| GPU, 1 replica | 2.5 | **85.60%** | 2639 MB |
| GPU, 2 replicas | 3.9 | 85.60% | 2480 MB |
| GPU f16, 1 replica | 2.8 | 85.60% | 1454 MB |
| CPU, 1 replica | 4.0 | 85.52% | 2917 MB |

**F1 85.60% on the GPU** — marginally ABOVE the CPU path (85.52%) and within
0.2 pt of ORT. The GPU path is production-correct; only its absolute speed on
this particular silicon is unimpressive, and that is a property of a 2 GB
integrated part sharing bandwidth with 20 cores, not of the code.

Re-verified 2026-08-02 on a second part + release: 13600K / UHD Graphics 770
(Gen12, bare-metal Arch host, `openvino/ubuntu24_dev:2026.2.1` container with
`--device /dev/dri`): ov_engine_probe PASSES on CPU and GPU,
`golden_intel_{det,cls,rec,layout}` 4/4 vs the cpu reference, FUNSD tiny
F1 85.52% (CPU) / **85.59% (GPU)** — first hardware run of the multi-request
batch-split engine, and second-silicon confirmation of the NATIVE GPU path's
correctness. (The 2026-07-25 ~23-point detection-loss defect lives on the ORT
OpenVINO-EP route — see the accuracy-defect section — and was NOT exercised
here; that retest still needs an ORT built with the OpenVINO EP.)

**2026-08-03 — the two speed fixes** (measured on the 13600K/UHD770 part,
FUNSD-50 ×2, per-stage profile via `PROFILE_STAGES=1`):

1. **`OV_PERF_HINT` default flipped to `latency`.** The old `throughput`
   default partitions cores into streams that pay only with several requests
   in flight; the sync engine's one-at-a-time request ran on ONE stream.
   CPU device: 2.4 → **5.5 img/s** (rec_infer 317 → 115 ms/page), which puts
   native OpenVINO AHEAD of the ORT-CPU backend (4.9) for the first time.
2. **Detection canvas snapping + per-canvas static compile**
   (`detection::snap_det_canvas` + `EngineCaps::per_shape_jit`, GPU/NPU only).
   The fully dynamic det variant runs shape-agnostic GPU kernels ~9× slower
   than static ones (143 vs 15.4 ms/img — and FUNSD-50 has only TWO distinct
   canvases, so this was never per-shape JIT). The detector now letterboxes
   into a 128-grid canvas and compiles one static variant per canvas on first
   sight. GPU device: 2.6 → **3.7 img/s**, det_infer 143 → ~15 ms steady; F1
   85.61%, goldens clean; CPU exactly unchanged (snapping stays off there).
   3.7 is within ~5% of the silicon-proportional ceiling — benchmark_app puts
   the UHD 770 at ~0.7× the 13600K on these models, so CPU rightfully wins on
   this part; an Arc-class GPU is where the GPU device should flip ahead.

**fp16 is the DEFAULT on GPU** (plugin default kept on CPU, where forcing f16 is
emulated and slower). Measured against an explicit f32 control at 4 replicas:

| GPU, r4 | img/s | F1 | peak RSS |
|---|---|---|---|
| default (fp16) | **4.3** | 85.60% | 4456 MB |
| forced f32 | 2.1 | 85.52% | 7518 MB |
| CPU default (control) | 4.0 | 85.52% | 2940 MB |

**2.05x faster and 41% less memory**, with F1 marginally HIGHER (85.60 vs 85.52).
Note the gap WIDENS with concurrency — at one replica fp16 was only +12% (2.8 vs
2.5), because f32 doubles the per-replica footprint and this iGPU then runs out of
capacity. A single-replica measurement would have badly understated this. The CPU
control confirms the default is GPU-only.

Per-model, the same direction: rec 1629 vs 1147 crops/s, det 43.7 vs 38.7 FPS.

Full GPU f16 scaling curve (replicas -> img/s), the number that transfers to
other hardware:

| replicas | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| img/s | 2.8 | 3.9 | 4.3 | FAIL (5747 MB) |

It climbs to r4 and runs out of DEVICE capacity at r6 — this iGPU shares 2 GB
with the system. The CPU path by contrast is FLAT at ~4.0 for every replica
count: the CPU is saturated, the GPU is capacity-limited. A discrete card removes
the capacity limit; it does NOT remove the two software ceilings below.

The number that TRANSFERS to other hardware is the scaling shape: the GPU goes
2.5 -> 3.9 img/s from 1 to 2 replicas (+56%) while the CPU path is FLAT at ~4.0
regardless of replica count. The CPU is saturated; the GPU still has parallel
headroom the pipeline can already use. `OV_INFER_PRECISION=f16` is also worth
having on a GPU: +12% throughput and nearly half the memory (2639 -> 1454 MB).

Two ceilings that do NOT improve with a faster GPU:

* **The synchronous engine.** `run()` issues one request and waits. GPUs need
  several in flight to fill, so this bites a discrete Arc HARDER than it bites
  the iGPU. Bring-up item 1.
* **~67 ms/page of host pre/post** (`warp_crops` = one `cv::warpPerspective` per
  crop) is device-independent. On a fast Arc it becomes the dominant term. That
  is what the unported SYCL kernels are for.

### Fixed along the way (all behaviour-preserving, F1 unchanged)

1. **`cv::threshold` silent no-write** in the SHARED `HostKernels` — passing a
   `CV_32F` src with a `CV_8U` dst makes OpenCV reallocate the destination
   instead of converting, so the caller's bitmap stayed all-zero, detection found
   zero boxes, and end-to-end F1 read 0.00% at full inference cost. Now
   `cv::compare`. Latent because the CPU backend reaches DB post through the
   main-tree `CpuPaddleDet` and never called this op.
2. **GPU declared unavailable** — the registry gated on `L0Allocator::has_device()`
   ("built with SYCL AND has a Level-Zero context"), which is the ZERO-COPY
   question, not the can-it-infer question. Now `OpenVINOEngine::device_available()`
   asks `ov::Core::get_available_devices()`.
3. **No-SYCL builds ran no-op kernels** — `SyclKernels` compiles its five device
   ops to empty bodies without DPC++. The build now uses the SHARED `HostKernels`
   instead (new `turbo_ocr_host_kernels` target), so the OpenVINO engine is
   usable with no oneAPI toolchain at all.
4. **`compile_model(m, dev, {})`** — an empty config is not neutral; OpenVINO
   applies the LATENCY hint. Now configurable (table above).
5. **Empty `bucket_rungs` crash** — capping the prebuild ladder left uncovered
   widths with no legal batch rungs, which is an invalid plan for the shared
   planner, not a degraded one. Uncovered widths now advertise the full
   width-legal ladder and run on the dynamic variant.
6. **Artefact explosion** — up to 35 CompiledModels (5 widths x 7 rungs), each
   with plugin-packed weights. `OV_REC_DYNAMIC_BATCH=1` compiles one per width
   with a dynamic batch dim (width is what changes kernel shapes) plus a small
   static probe per width for geometry, since a dynamic artefact reports its
   batch dim as dynamic and cannot size the logits scratch.
7. `reserve_host_fallback` moved onto the seam (`IKernels`), so stages hold the
   interface rather than a concrete `SyclKernels`.

### To actually beat ORT here, in evidence order

1. **Make the engine asynchronous / multi-request.** LANDED (batch-split
   concurrency in `OpenVINOEngine::run`): a dynamic-batch, single-input,
   caller-owned-output call is cut into per-request slices that run
   `start_async` across an `ov::optimal_number_of_infer_requests`-sized pool
   and are joined before returning — the seam contract (outputs valid on
   return) is unchanged, so `caps().async` stays false. This was the 476 ->
   2144 crops/s (4.5x) gap for rec, which carries the bulk of the work.
   PENDING RUNTIME VALIDATION on Intel hardware: re-run the SETUP bench and
   compare crops/s and ms/page against the streams=1 numbers above. Det
   (batch-1) and layout (multi-input) still run one synchronous request; they
   are bounded by the same L0-queue item as `caps().async` below.
2. **SYCL kernels.** ~67 ms/page is host pre/post — `warp_crops` runs a
   `cv::warpPerspective` per crop. This is what the unported device half is for.
3. **CPU+GPU width split.** Measured 1.40x concurrent (3061 vs 2190 crops/s), and
   the GPU wins detection (80.4 vs 73.2 FPS) and EVERY wide rec bucket
   (480/800/1600). Mirror image of Apple, where the second engine won narrow
   crops and lost wide ones.

## 9. Open items

1. **Validate the batch-split multi-request engine on Intel hardware.** The
   code is in (see "To actually beat ORT here" item 1); the backend was last
   hardware-verified before it landed (F1 85.5-85.6% on the synchronous
   engine). Validate: (a) F1 unchanged — the split is a pointer-offset slicing
   of batch-major tensors, so golden diffs must be bit-identical; (b) rec
   crops/s approaches the 2144 throughput-hint number rather than the 476
   streams=1 number; (c) ms/page drops accordingly.
2. **SYCL kernels for the host pre/post**, per item 2 above.
3. **Flip `caps().async` to true** once the OpenVINO remote context shares the
   `L0DeviceQueue` (README "async is deliberate" section) — that is what
   unlocks det/layout overlap and removes the per-run barrier.
4. **NPU** needs native Windows or bare-metal Linux; decide whether it is worth
   leaving WSL for.
