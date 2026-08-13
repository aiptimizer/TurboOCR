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

Pick your hardware. The full click-through installation guide, with every
backend and the Python library, is at
**[Install — pick your hardware](docs/getting-started/install.md)**.

<details open>
<summary><strong>NVIDIA GPU</strong> &nbsp;·&nbsp; shipped</summary>

Linux, driver 595+, Turing or newer. ~4 GB VRAM text-only, ~8 GB full pipeline.

```bash
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  ghcr.io/aiptimizer/turboocr:latest
```

First start builds TensorRT engines (~90 s on a 5090, up to an hour on older
cards; `TRT_OPT_LEVEL=3` cuts that 3–5x). The named volume caches them, so
later starts are instant. All weights are baked into the image; env vars pick
what loads:

```bash
-e TABLE_BACKEND=slanext              # tables to HTML
-e FORMULA_BACKEND=ppformulanet_s     # formulas to LaTeX
-e OCR_MODEL=medium                   # tiny (default) | small | medium | arabic | eslav | korean | thai | greek
```

→ [Docker & deployment](docs/getting-started/docker.md) · [Native build](docs/getting-started/native.md)
</details>

<details>
<summary><strong>Apple Silicon</strong> &nbsp;·&nbsp; in testing &nbsp;·&nbsp; Metal + Neural Engine, native only</summary>

Detection and warp run on the GPU (Metal + MPSGraph); recognition is a GPU +
Neural Engine hybrid, with narrow crops on the ANE via CoreML in parallel.
No container can run it: macOS virtualization exposes no GPU.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON
cmake --build build -j"$(sysctl -n hw.ncpu)"

# Pin the 9-bucket rec ladder (auto-discovery otherwise builds a 42-bucket
# ladder and roughly halves throughput):
export TURBO_APPLE_REC_BUCKETS=320,480,800,1200,1600,2000,2500,3200,4000

./build/turboocr-server --backend apple
```

Layout, tables, formulas and autorotate work too when their models are
supplied. Details: `src/backends/apple/README.md`.
</details>

<details>
<summary><strong>Intel CPU / iGPU / Arc</strong> &nbsp;·&nbsp; in testing &nbsp;·&nbsp; OpenVINO</summary>

```bash
cmake -S . -B build-intel -DTURBO_BACKENDS="cpu;intel"
cmake --build build-intel -j$(nproc)
./build-intel/turboocr-server --backend intel      # OV_DEVICE=CPU|GPU|NPU
```

The native OpenVINO path beats the ONNX Runtime path on the same silicon.
Details: `src/backends/intel/README.md`.
</details>

<details>
<summary><strong>AMD GPU (ROCm)</strong> &nbsp;·&nbsp; not yet hardware-tested</summary>

Runs through ROCm: HIP kernels plus a MIGraphX inference engine with a
per-architecture `.mxr` compile cache, so model compilation is paid once.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTURBO_BACKENDS="cpu;amd" \
      -DCMAKE_HIP_ARCHITECTURES="$(rocminfo | grep -om1 'gfx[0-9a-f]*')" \
      -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j$(nproc)
./build/turboocr-server --backend amd
```

The first run compiles the MIGraphX graphs and caches them under
`~/.cache/turbo-ocr/mgx_*.mxr`. Checklist for a first machine:
`src/backends/amd/BRINGUP.md`.
</details>

<details>
<summary><strong>CPU only</strong> &nbsp;·&nbsp; portable fallback, runs anywhere</summary>

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON
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

```bash
pip install turboocr              # client only — talk to a running server
pip install "turboocr[cpu]"       # + in-process engine, CPU (and Apple Silicon)
pip install "turboocr[cuda]"      # + NVIDIA engine
pip install "turboocr[openvino]"  # + Intel engine (CPU / iGPU / Arc / NPU)
pip install "turboocr[rocm]"      # + AMD engine
```

`turboocr doctor` prints the right line for your machine. Feature extras
combine: `"turboocr[cuda,pdf]"`.

**Pre-release status:** `4.0.0a1` is an alpha, so those commands install the
current stable client (0.3.0) until you ask for the pre-release explicitly:

```bash
pip install --pre "turboocr[cpu]"        # or pin: turboocr[cpu]==4.0.0a1
```

The engine wheels are not published yet at all — until the first release run,
build one from this checkout:

```bash
# The helper builds AND repairs the wheel — a bare `pip wheel python/`
# bundles no libraries and only runs on the machine that built it.
scripts/python/build_backend_wheel.sh cpu     # cpu | cuda | openvino | rocm
pip install build-wheels/cpu/fixed/*.whl
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
(client + engine facade) plus one engine wheel per backend, picked by an extra:
`pip install "turboocr[cpu]"` / `[cuda]` / `[openvino]` / `[rocm]`. The engine
wheels are not published yet — see [Quick Start](#quick-start) for the
build-from-source path.

```python
import turboocr

ocr = turboocr.OCR(tier="tiny", replicas=3)   # built-in replica pool
page = ocr.read("invoice.png")                # one image → PageResult
doc = ocr.read_batch(images)                  # fans out across replicas
ocr.read_pdf("report.pdf")                    # PDF → DocumentResult
```

One `OCR(replicas=3)` object reaches the server's multi-replica throughput
(measured: 94% of it on Apple silicon) with no user-side threading. `backend=`
picks `"cuda"`, `"apple"`, `"openvino"`, `"cpu"`, … — same seam as the server.

→ [python/README.md](python/README.md) · [design](python/DESIGN.md)

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
