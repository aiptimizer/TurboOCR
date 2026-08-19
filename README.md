<p align="center">
  <sub>v4 alpha · backends: NVIDIA, Apple (Metal + Neural Engine), Intel (OpenVINO), AMD (ROCm), CPU · AMD is not yet hardware-tested</sub>
</p>

<p align="center">
  <img src="tests/benchmark/comparison/images/hero_banner.svg" alt="TurboOCR — the fastest GPU document parser. 650+ img/s whole-page OCR, 200+ img/s dense documents, 20 pages/s full parse to Markdown, on one RTX 5090." width="100%">
</p>

<p align="center">
  <strong>English</strong> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <strong>The fastest GPU document parser — OCR · layout · tables · formulas → Markdown, at 650+ images/s on one GPU.</strong><br>
  C++ / CUDA / TensorRT / PP-OCRv6 &mdash; Linux + NVIDIA GPU
</p>

<h3 align="center">v4.0-alpha — one pipeline, many backends</h3>
<p align="center">
  <sub>One unified engine behind a device seam: NVIDIA · Apple Metal + Neural Engine · Intel OpenVINO · AMD ROCm · a native Python library · PP-OCRv6 with <code>tiny</code>/<code>small</code>/<code>medium</code> tiers · <a href="docs/guides/upgrading-v4.md">what changed in v4</a></sub>
</p>

<p align="center">
  <a href="https://github.com/aiptimizer/TurboOCR"><strong>⭐ Star TurboOCR on GitHub</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/throughput-650%2B_img%2Fs-blue?style=flat-square&logo=speedtest&logoColor=white" alt="650+ img/s">
  <a href="https://turboocr.com"><img src="https://img.shields.io/badge/website-turboocr.com-3B82F6?style=flat-square&logo=googlechrome&logoColor=white" alt="turboocr.com"></a>
  <a href="https://github.com/aiptimizer/TurboOCR/releases/latest"><img src="https://img.shields.io/github/v/release/aiptimizer/TurboOCR?style=flat-square&logo=github&logoColor=white" alt="Release"></a>
  <a href="https://ghcr.io/aiptimizer/turboocr"><img src="https://img.shields.io/badge/docker-ghcr.io-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"></a>
  <img src="https://img.shields.io/badge/C%2B%2B20-00599C?style=flat-square&logo=cplusplus&logoColor=white" alt="C++20">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-10.15-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="TensorRT 10.15">
  <img src="https://img.shields.io/badge/Metal_%2B_ANE-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple Metal + Neural Engine">
  <img src="https://img.shields.io/badge/OpenVINO-1E7BD9?style=flat-square" alt="Intel OpenVINO">
  <img src="https://img.shields.io/badge/ROCm-ED1C24?style=flat-square&logo=amd&logoColor=white" alt="AMD ROCm">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python library">
  <img src="https://img.shields.io/badge/gRPC-4285F4?style=flat-square&logo=google&logoColor=white" alt="gRPC">
  <a href="https://github.com/PaddlePaddle/PaddleOCR"><img src="https://img.shields.io/badge/PP--OCRv6-PaddleOCR-0053D6?style=flat-square&logo=paddlepaddle&logoColor=white" alt="PaddleOCR"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="MIT License">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#benchmarks">Benchmarks</a> &middot;
  <a href="#getting-higher-accuracy">Accuracy</a> &middot;
  <a href="#models">Models</a> &middot;
  <a href="#python">Python</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="docs/index.md">Docs</a>
</p>

---

TurboOCR is a complete GPU document parser. It runs PP-OCRv6 text detection
and recognition together with layout analysis, table recognition (HTML),
formula recognition (LaTeX) and reading-order Markdown export in a single
device-resident pipeline, served over HTTP and gRPC. All inference is local;
no vision-language model and no external API is involved.

Measured on a single RTX 5090, whole-page OCR runs at over 650 images per
second on FUNSD forms and full structured parsing at 20 pages per second,
where VLM-based document parsers process roughly one page per second
([benchmarks](#benchmarks)).

- **Models.** One PP-OCRv6 model covers Latin, Chinese and Japanese in three tiers (`tiny`, `small`, `medium` — `tiny` omits kana, so use `small`/`medium` for Japanese). Dedicated recognizers add Arabic, Cyrillic, Korean, Thai and Greek.
- **Document structure.** PP-DocLayoutV3 layout, SLANet+ tables to HTML, PP-FormulaNet_plus-S formulas to LaTeX, class-aware reading order. Every stage is opt-in per request; the default path pays nothing for stages it does not use.
- **PDF.** Pages are rendered and OCR'd natively, with optional auto-rotation, per-page streaming, and whole-PDF Markdown export.
- **Backends.** The same pipeline runs on NVIDIA CUDA/TensorRT, Apple Metal plus the Neural Engine, Intel OpenVINO, AMD ROCm, and plain CPU.
- **Python.** The C++ pipeline is available as a native Python library with a built-in replica pool ([Python](#python)).
- **Operations.** Single-line Docker deployment, Prometheus metrics, and HTTP plus gRPC from one binary.

Full documentation: **[docs/](docs/index.md)**

---

## Quick Start

> **v4.0.0-alpha** — the Python wheels for CPU (Linux/macOS/Windows), Apple
> Silicon (Metal GPU + Neural Engine) and Intel/OpenVINO are **live on PyPI**:
> `pip install --pre "turboocr[cpu]"` / `[apple]` / `[openvino]`. The NVIDIA wheels are
> awaiting a PyPI file-size approval, and there is **no published Docker
> image** — those paths build from this checkout. Full details in
> **[the install guide](docs/getting-started/install.md)**.

Picking a backend is two steps, the same on every path below:

1. **Build time — what gets compiled in.** `-DTURBO_BACKENDS="cpu;intel"` is a
   semicolon-separated list telling CMake which backends to compile into the
   one server binary (`cpu`, `apple`, `intel`, `amd`). Run without the flag,
   `cmake -B build` builds the native NVIDIA CUDA/TensorRT server on Linux.
   **Docker does this step for you** — each `--target` below is an image with
   the right backends already compiled in and started.
2. **Start time — which one runs.** `--backend nvidia|apple|intel|amd|cpu`
   picks one of the compiled-in backends when the server starts. Left unset,
   the server auto-picks, and auto does not always mean the vendor you built —
   the Intel block below says so explicitly.

Everything else (`OV_DEVICE`, `TRT_OPT_LEVEL`, …) tunes the backend that is
already selected; each block explains the ones it needs.

<details open>
<summary><strong>NVIDIA GPU</strong> &nbsp;·&nbsp; shipped</summary>

**Docker** (built from this repo):

```bash
docker build -f docker/Dockerfile --target nvidia -t turboocr:nvidia .
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  turboocr:nvidia
```

**From source** (no `TURBO_BACKENDS` needed — the default Linux configure *is*
the native CUDA/TensorRT server):

```bash
cmake -B build -DTENSORRT_DIR=/usr/local/tensorrt
cmake --build build -j$(nproc)
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/turboocr-server --backend nvidia
```

First start builds TensorRT engines (~90 s on a 5090, longer on older cards;
`TRT_OPT_LEVEL=3` cuts it 3–5x) and caches them. Needs GCC 13.3+/C++20,
CUDA + TensorRT 10.2+, OpenCV 4.x, Drogon 1.9+, gRPC.
</details>

<details>
<summary><strong>Apple Silicon</strong> &nbsp;·&nbsp; in testing</summary>

No Docker — macOS containers have no GPU passthrough.

**Python library** (the fastest way to try it — live on PyPI):

```bash
pip install --pre "turboocr[apple]"
python -c "import turboocr; print(turboocr.OCR(backend='apple').read('doc.png').text)"
```

The macOS arm64 wheel runs the full native mode out of the box: detection and
recognition on the **Metal GPU** with the narrow recognition buckets on the
**Neural Engine** in parallel — the export bundles auto-download with SHA256
verification on first use. Detection adapts to any page shape (the engine
specializes per shape at runtime), and `aread`/`aread_batch`/`aread_pdf` give
you asyncio concurrency across the built-in replica pool.

**Server, from source:**

```bash
brew install cmake opencv drogon jsoncpp protobuf grpc c-ares jpeg-turbo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBO_BACKENDS="cpu;apple"   # step 1: compile cpu + apple in
cmake --build build -j"$(sysctl -n hw.ncpu)"
./build/turboocr-server --backend apple                                       # step 2: run the apple one
```

Full Xcode + Metal toolchain required. Details: `src/backends/apple/README.md`.
</details>

<details>
<summary><strong>Intel CPU / iGPU / Arc</strong> &nbsp;·&nbsp; in testing</summary>

One backend, two names: the server calls it `intel` (the vendor), the Python
side calls it `openvino` (Intel's runtime — `turboocr[openvino]`,
`backend="openvino"`); `OV_DEVICE`/`device=` then picks CPU, iGPU/Arc or NPU.
Its CPU device is also the fastest way to run TurboOCR on any x86 CPU.

Both paths run the same OpenVINO backend and differ only in who performs the
two steps: the Docker image has both baked in (it sets `TURBO_BACKEND=intel`
internally — that is why its run line passes no `--backend`), while from
source you pass them yourself.

**Docker** (built from this repo):

```bash
docker build -f docker/Dockerfile --target intel -t turboocr:intel .

# OpenVINO on the CPU device — works everywhere, nothing to pass through:
docker run -p 8000:8000 -p 50051:50051 turboocr:intel

# OpenVINO on the iGPU/Arc — pass the device through AND select it:
docker run --device /dev/dri -e OV_DEVICE=GPU -p 8000:8000 -p 50051:50051 turboocr:intel
```

**From source:**

```bash
cmake -S . -B build -DTURBO_BACKENDS="cpu;intel"   # step 1: compile cpu + intel in
cmake --build build -j$(nproc)
./build/turboocr-server --backend intel            # step 2: run the intel one — REQUIRED,
                                                   # auto-pick starts plain CPU without it
```

The one knob after that is `OV_DEVICE=CPU|GPU|NPU` — which Intel silicon
OpenVINO runs on. Its default differs by context for one physical reason: a
bare binary can see the host's iGPU, so it defaults to `GPU`; a container
only sees the iGPU if you pass `--device /dev/dri`, so the image defaults to
`CPU`. Details: `src/backends/intel/README.md`.
</details>

<details>
<summary><strong>AMD GPU (ROCm)</strong> &nbsp;·&nbsp; not yet hardware-tested</summary>

**Docker** (built from this repo):

```bash
docker build -f docker/Dockerfile --target amd -t turboocr:amd .
docker run --device /dev/kfd --device /dev/dri --group-add video \
  -v ocr-cache:/home/ocr/.cache/turbo-ocr \
  -p 8000:8000 -p 50051:50051 turboocr:amd
```

**From source:**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTURBO_BACKENDS="cpu;amd" \
      -DCMAKE_HIP_ARCHITECTURES="$(rocminfo | grep -om1 'gfx[0-9a-f]*')" \
      -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j$(nproc)
./build/turboocr-server --backend amd
```

First run compiles MIGraphX graphs and caches them under
`~/.cache/turbo-ocr/mgx_*.mxr`. First-machine checklist: `src/backends/amd/BRINGUP.md`.
</details>

<details>
<summary><strong>CPU only</strong> &nbsp;·&nbsp; shipped</summary>

**Docker** (built from this repo):

```bash
docker build -f docker/Dockerfile --target cpu -t turboocr:cpu .
docker run -p 8000:8000 -p 50051:50051 turboocr:cpu
```

**From source** (`--backend` not needed — cpu is the only backend in this build):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBO_BACKENDS="cpu"
cmake --build build -j$(nproc)
./build/turboocr-server
```
</details>

<details>
<summary><strong>Python library</strong> &nbsp;·&nbsp; one package, one extra per backend</summary>

`turboocr` is the pure-Python package — the typed client for a TurboOCR server,
plus the in-process engine facade. Its extras pick the engine wheel for your
hardware (install exactly one backend; the engine wheels are mutually
exclusive):

> **Alpha status: `[cpu]`, `[apple]` and `[openvino]` install from PyPI
> today** (with `--pre` — see below). The NVIDIA `[cuda12]` / `[cuda13]` extras resolve once
> PyPI approves the file-size requests for those wheels; `[rocm]` is
> deliberately unpublished. For a backend that is not on PyPI yet, this is the
> working path:

```bash
# The helper builds AND repairs the wheel — a bare `pip wheel python/`
# bundles no libraries and only runs on the machine that built it.
scripts/python/build_backend_wheel.sh cpu     # cpu | cuda12 | cuda13 | openvino | rocm
pip install build-wheels/cpu/fixed/*.whl
```

The engine wheel is self-sufficient: `import turboocr_engine` gives you the
full pipeline and the `turboocr` CLI without the umbrella package installed.

Once the wheels are published, the umbrella becomes the front door — one extra
per backend, and they are mutually exclusive, so install exactly one:

```bash
pip install turboocr              # client only — talk to a running server
pip install "turboocr[cpu]"       # + in-process engine, CPU
pip install "turboocr[apple]"     # + Apple engine — Metal GPU + Neural Engine (macOS arm64)
pip install "turboocr[cuda12]"    # + NVIDIA engine, CUDA 12 (driver R525+)
pip install "turboocr[cuda13]"    # + NVIDIA engine, CUDA 13 (driver R580+)
pip install "turboocr[openvino]"  # + Intel engine (CPU / iGPU / Arc / NPU)
pip install "turboocr[rocm]"      # + AMD engine
```

`turboocr doctor` prints the right line for your machine — on NVIDIA it also
picks between `cuda12` and `cuda13` from your driver. Feature extras combine:
`"turboocr[cuda12,pandas]"` (PDF support is built in since `4.0.0a6` — no
extra needed). Because `4.0.0a6` is a pre-release, pip will
not select it by default even after publication — ask for it explicitly:

```bash
pip install --pre "turboocr[cpu]"        # or pin: turboocr[cpu]==4.0.0a6
```

On NVIDIA, the engine wheel needs only an NVIDIA **driver** (no CUDA toolkit).
Its default `backend="auto"` resolves to the native TensorRT engine: the
**first** run builds an engine (~90 s on a 5090, longer on older cards) and
caches it under `TRT_ENGINE_CACHE` (default `~/.cache/turbo-ocr`), so every
later run starts fast. `backend="cuda"` is the instant-start ONNX Runtime
path — nothing is compiled.
</details>

First request, identical on every backend (native builds bind `8080`; the
Docker quick-start above maps `8000`):

```bash
curl -X POST http://localhost:8080/ocr/raw \
  --data-binary @document.png -H "Content-Type: image/png"
```

```json
{"results": [{"text": "Invoice Total", "confidence": 0.97, "bounding_box": [[42,10],[210,10],[210,38],[42,38]]}]}
```

Stages are opt-in per request: `?layout=1`, `?tables=1`, `?formulas=1`
(tables and formulas auto-enable layout). PDF goes to `POST /ocr/pdf`, Markdown
export to `?markdown=1` or `POST /ocr/markdown`, gRPC to port 50051.
`GET /capabilities` reports what a running server has loaded; asking for a
stage the server was not started with is a hard `400`, never a silent empty
result.

Build dependencies, GPU-architecture notes and deployment detail live in the
docs: [build guide](docs/getting-started/native.md) ·
[Docker & compose](docs/getting-started/docker.md). Coming from v2.x, read
[Upgrading to v3](docs/guides/upgrading-v3.md) first; what v4 adds is in
[What changed in v4](docs/guides/upgrading-v4.md).

---

## Benchmarks

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tests/benchmark/comparison/images/bench_hero_dark.svg">
    <img src="tests/benchmark/comparison/images/bench_hero_light.svg" alt="Whole-page OCR throughput on one RTX 5090: TurboOCR 678 img/s on FUNSD vs 2–6 img/s for PaddleOCR, EasyOCR, RapidOCR and Tesseract" width="100%">
  </picture>
</p>

Accuracy: the `medium` tier reaches **91.9% / 93.4%** word-F1 on FUNSD / CORD,
the highest of every engine measured, and full structured parsing scores
**0.90** Overall on OmniDocBench, within ~5 points of PaddleOCR-VL at roughly
twenty times its speed. All numbers come from the in-repo harness on one
RTX 5090: identical pages for every engine, ≥15 s timed windows, dual-clock
cross-check.

→ [Full benchmarks & methodology](docs/benchmarks/comparison.md)

---

## Getting higher accuracy

The defaults maximize throughput. Three levers trade speed for accuracy:

1. **Bigger tier** — `-e OCR_MODEL=small` (fixes most of tiny's misreads at ~half the speed) or `medium` (most accurate).
2. **Higher detection resolution** — `-e DET_MAX_SIDE_LIMIT=2560` for small text on phone screenshots or dense scans (one-time engine rebuild, then cached).
3. **Orientation on every line** — `-e CLS_ALL_BOXES=1` for scans with mixed or upside-down horizontal lines.

Measured cost of every lever and combination:

<p align="center">
  <img src="tests/benchmark/comparison/images/lever_cost.png" alt="Measured throughput cost of each accuracy lever per model tier, and of full parsing" width="88%">
</p>

---

## Models

Text detection + recognition + line orientation always run; everything else is
opt-in and only loads when configured.

| Stage | Model / arch | Size | Selected by | Docs |
|---|---|---:|---|---|
| **Text detection** | PP-OCRv6 det (DB, three tiers) | 1.7 / 9.4 / 59 MB | `OCR_MODEL` tier | [detection](docs/models/detection.md) |
| **Text recognition** | PP-OCRv6 rec (Latin + Chinese + Japanese) | 4.3 / 20 / 73 MB | `OCR_MODEL` tier — default `tiny` | [recognition](docs/models/recognition.md) |
| **Line orientation** | PP-LCNet 0°/180° per line | ~1 MB | always on (`CLS_ALL_BOXES=1` for every line) | [classification](docs/models/classification.md) |
| **Page orientation** | PP-LCNet doc_ori 0/90/180/270 | ~7 MB | `/ocr/pdf?autorotate=1` | [http api](docs/reference/http.md) |
| **Layout** | PP-DocLayoutV3 (RT-DETR-L, 25 classes) | ~124 MB | `?layout=1` | [layout](docs/models/layout.md) |
| **Table → HTML** | SLANet-Plus (TRT encoder + C++ GRU decoder) | ~5 MB | `TABLE_BACKEND=slanext` + `?tables=1` | [table](docs/models/table.md) |
| **Formula → LaTeX** | PP-FormulaNet_plus-S (pure C++, no Python) | ~294 MB | `FORMULA_BACKEND=ppformulanet_s` + `?formulas=1` | [formula](docs/models/formula.md) |

Tiers trade speed for accuracy, and mostly share a charset — with one exception
that matters: `tiny` drops the Japanese kana and most CJK ideographs, so for
Japanese use `small` or `medium` (see [model selection](docs/models/selection.md)).
Other scripts (`arabic`, `eslav`, `korean`, `thai`, `greek`) use retained
PP-OCRv5 recognizers.
`tables=1`/`formulas=1` auto-enable layout; the default path pays nothing for
stages it doesn't use.

→ [Model selection guide](docs/models/selection.md)

---

## Python

The `python/` package wraps the same C++ pipeline (nanobind, GIL released
during inference) — not a reimplementation. Models auto-download per tier
(~6 MB for tiny) with SHA256 verification. It ships as the `turboocr` umbrella
(client + engine facade) plus one engine wheel per backend, picked by an extra.

**What is on PyPI today, and what is not.** `turboocr-engine-cpu` (Linux,
Windows, and macOS arm64 — where it carries the full **Apple backend**: Metal
GPU + Neural Engine native mode with auto-downloaded export bundles, installed
via the `[apple]` extra), `turboocr-engine-openvino` and the `turboocr`
umbrella are **live**. The NVIDIA wheels (`-cuda12` / `-cuda13`)
are built and verified but await PyPI's file-size approval — until then their
extras do not resolve and the working path is building from this checkout;
`-rocm` is deliberately unpublished. A `0.0.0` release under an engine name
is an empty placeholder from the PyPI project setup, not installable
software. A plain `pip install turboocr` still resolves to the old **0.3.0
client** (no engine): `--pre` is required because `4.0.0a6` is a pre-release:

```bash
pip install --pre "turboocr[cpu]"     # or [apple] | [cuda12] | [cuda13] | [openvino] | [rocm]
```

`turboocr doctor` names the right one for your machine.

**Installing manually today**, from this checkout:

```bash
# Build AND repair the engine wheel for your hardware — a bare
# `pip wheel python/` bundles no libraries and only runs on the machine
# that built it.
scripts/python/build_backend_wheel.sh cpu     # cpu | cuda12 | cuda13 | openvino | rocm
pip install build-wheels/cpu/fixed/*.whl

python -c "import turboocr_engine; print(turboocr_engine.OCR().read('doc.png').text)"
```

The engine wheel is self-sufficient: `import turboocr_engine` gives you the
full pipeline and the `turboocr` CLI without the umbrella package installed.
With the umbrella (once published), the same API is `import turboocr`:

```python
import turboocr

ocr = turboocr.OCR(tier="tiny", replicas=3)   # built-in replica pool
page = ocr.read("invoice.png")                # one image → PageResult
doc = ocr.read_batch(images)                  # fans out across replicas
doc = ocr.read_pdf("report.pdf")              # PDF → DocumentResult (pages fan out too)

for page in ocr.read_pdf_stream("report.pdf"):    # stream pages as they're ready
    ...                                            # ordered=False → completion order

page = await ocr.aread("invoice.png")         # async twins: aread / aread_batch /
async for page in ocr.aread_pdf_stream(pdf):  #   aread_pdf / aread_pdf_stream
    ...
```

One `OCR(replicas=3)` object reaches the server's multi-replica throughput
(measured: 94% of it on Apple silicon) with no user-side threading; a
multi-page PDF reads ~2.4× faster on accelerator backends. `backend=`
picks `"cuda"`, `"apple"`, `"openvino"`, `"cpu"`, … — same seam as the server.
Since 4.0.0a6: `read_pdf`/`read_batch` drop page rasters by default (pass
`keep_image=True` when you need `save_searchable_pdf()`/`draw()`),
`on_error="skip"` contains a failing page to a `page_failed` warning instead
of aborting the document, `autorotate=True` also straightens PDF pages,
`password=` opens encrypted PDFs. `read_pdf` still defaults to `mode="ocr"` —
it is an OCR engine, so every page goes through the recognizer — and the new
opt-in `mode="auto"` reads a page's embedded text layer instead when one is
there and passes a quality gate, which is ~10x faster and byte-exact on
born-digital PDFs (`mode="text"` never OCRs at all).

→ [Python library reference](docs/reference/python.md) · [python/README.md](python/README.md) · [design](python/DESIGN.md)

---

## API

One binary serves HTTP and gRPC from a shared pipeline pool.

| Endpoint | Purpose |
|---|---|
| `POST /ocr/raw` | OCR raw image bytes (fastest) |
| `POST /ocr` | OCR base64 image in JSON |
| `POST /ocr/pixels` | Zero-decode raw pixel buffer |
| `POST /ocr/batch` | Batch of images |
| `POST /ocr/pdf` | PDF → text; `?markdown=1` → whole PDF as Markdown |
| `POST /ocr/markdown` | Page → faithful Markdown (requires layout) |
| `POST /ocr/stream` | PDF → newline-delimited JSON, one event per page |
| `POST /infer` | One crop through a chosen table/formula backend |
| `GET /capabilities` | Runtime feature & route discovery |
| `GET /metrics` · `/profile` · `/health` | Prometheus · per-stage timings · probes |

The OCR and recognition endpoints have gRPC twins on port 50051, parsed through
the same validation core, so the two transports cannot drift apart (the mapping,
and the few HTTP-only endpoints, are in [the gRPC reference](docs/reference/grpc.md)).
All OCR endpoints accept `?layout=1`, `?tables=1`, `?formulas=1`.

→ [HTTP API](docs/reference/http.md) · [gRPC API](docs/reference/grpc.md) · [Monitoring](docs/reference/monitoring.md)

---

## Configuration

Everything is an environment variable (with an equivalent CLI flag). Common ones:

| Variable | Default | Description |
|---|---|---|
| `OCR_MODEL` | `tiny` | `tiny` / `small` / `medium`, or a PP-OCRv5 script model |
| `DISABLE_LAYOUT` | `0` | `1` skips the layout model (~300–500 MB VRAM) |
| `CLS_ALL_BOXES` | `0` | `1` runs the 0°/180° classifier on every line |
| `DET_MAX_SIDE_LIMIT` | `1280` | Detection resolution cap (raise for dense scans) |
| `REQUEST_TIMEOUT_MS` | `60000` | Queueing deadline before `504` |
| `SHUTDOWN_GRACE_SECONDS` | `30` | Real drain bound on SIGTERM — queued work past it is shed, in-flight finishes |
| `PIPELINE_POOL_SIZE` | auto | Concurrent GPU pipelines |

→ [Full configuration reference (35+ variables)](docs/reference/configuration.md)

---

## Acknowledgements

Built on open-source work:

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (Baidu) — PP-OCRv6 / PP-OCRv5 detection, recognition, and classification models, plus PP-DocLayoutV3 layout detection. This project would not exist without their research and pre-trained weights.
- **[Drogon](https://drogon.org)** — high-performance async C++ HTTP framework.
- **[Wuffs](https://github.com/google/wuffs)** — fast PNG decoder by Google (vendored).
- **[PDFium](https://pdfium.googlesource.com/pdfium/)** — PDF rendering and text extraction (vendored).
- **[Clipper](http://www.angusj.com/delphi/clipper.php)** — polygon clipping for text-detection post-processing (vendored).

## License

MIT. See [LICENSE](LICENSE).

<p align="center">
  <a href="https://github.com/aiptimizer/TurboOCR"><strong>⭐ Star TurboOCR on GitHub</strong></a><br>
  <sub>Sponsored by <a href="https://miruiq.com"><strong>Miruiq</strong></a> — AI-powered data extraction from PDFs and documents — and <a href="https://diaiq.com"><strong>DiaIQ</strong></a>.</sub>
</p>
