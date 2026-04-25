<p align="center">
  <img src="tests/benchmark/comparison/images/banner.png" alt="Turbo OCR — Fast GPU OCR server. 270 img/s on FUNSD." width="100%">
</p>

<!--
Turbo OCR — Fast GPU OCR server. C++ / CUDA / TensorRT. 270 img/s on FUNSD.
-->

<p align="center">
  <strong>GPU-accelerated OCR server. 50x faster than PaddleOCR Python.</strong><br>
  C++ / CUDA / TensorRT / PP-OCRv5 &mdash; Linux + NVIDIA GPU
</p>

<p align="center">
  <img src="https://img.shields.io/badge/throughput-270_img%2Fs-blue?style=flat-square&logo=speedtest&logoColor=white" alt="270 img/s">
  <a href="https://github.com/aiptimizer/TurboOCR/releases/latest"><img src="https://img.shields.io/github/v/release/aiptimizer/TurboOCR?style=flat-square&logo=github&logoColor=white" alt="Release"></a>
  <a href="https://ghcr.io/aiptimizer/turboocr"><img src="https://img.shields.io/badge/docker-ghcr.io-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"></a>
  <img src="https://img.shields.io/badge/C%2B%2B20-00599C?style=flat-square&logo=cplusplus&logoColor=white" alt="C++20">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-10.16-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="TensorRT 10.16">
  <a href="https://drogon.org"><img src="https://img.shields.io/badge/Drogon-1.9-009688?style=flat-square&logo=cplusplus&logoColor=white" alt="Drogon"></a>
  <img src="https://img.shields.io/badge/nginx-009639?style=flat-square&logo=nginx&logoColor=white" alt="nginx">
  <img src="https://img.shields.io/badge/gRPC-4285F4?style=flat-square&logo=google&logoColor=white" alt="gRPC">
  <a href="https://github.com/PaddlePaddle/PaddleOCR"><img src="https://img.shields.io/badge/PP--OCRv5-PaddleOCR-0053D6?style=flat-square&logo=paddlepaddle&logoColor=white" alt="PaddleOCR"></a>
  <a href="#monitoring"><img src="https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=prometheus&logoColor=white" alt="Prometheus"></a>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="MIT License">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#benchmarks">Benchmarks</a> &middot;
  <a href="#monitoring">Monitoring</a> &middot;
  <a href="#configuration">Configuration</a> &middot;
  <a href="#building-from-source">Build</a> &middot;
  <a href="#roadmap">Roadmap</a>
</p>

---

![Turbo-OCR vs alternatives on FUNSD](tests/benchmark/comparison/images/hero.png)

<!--
The image above is a chart that may be blocked by robots.txt. Below is the chart data as a text table for LLMs, scrapers, and other automated readers.
Turbo-OCR vs PaddleOCR · EasyOCR · VLMs — FUNSD (50 pages, RTX 5090)
| Engine | F1 (%) | Throughput (img/s) |
|---|---:|---:|
| Turbo-OCR (C++/TRT) | 90.2 | 268.4 |
| PaddleOCR mobile latin | 86.6 | 5.1 |
| Qwen3-VL-2B | 84.3 | 1.3 |
| PaddleOCR-VL (pipeline) | 83.1 | 2.0 |
| EasyOCR (Python) | 63.0 | 2.8 |
-->

### Highlights

- 🚀 **270 img/s** on FUNSD A4 forms (c=16) &mdash; **1,200+ img/s** on sparse images
- ⚡ **11 ms p50 latency**, single request
- 🎯 **F1 = 90.2%** on FUNSD &mdash; higher accuracy than PaddleOCR Python with the same weights
- 🖨️ **Prints & handwriting** &mdash; PP-OCRv5 handles both out of the box
- 📄 **PDF native** &mdash; pages rendered and OCR'd in parallel
- 🔒 **4 PDF modes** &mdash; pure OCR, native text layer, auto-dispatch, detection-verified hybrid
- 🧩 **Layout detection** &mdash; PP-DocLayoutV3 with 25 region classes, per-request `?layout=1` toggle
- 🌐 **HTTP + gRPC** from a single binary, sharing the same GPU pipeline pool
- 🐳 **One-line Docker deploy** &mdash; `docker run` with auto TRT engine build on first start
- 📊 **Prometheus metrics** &mdash; request counters, latency histograms, VRAM usage on `/metrics`

*RTX 5090, PP-OCRv5 mobile latin, TensorRT FP16, pool=5. Prints, handwriting, layout detection. This is the fast lane.*

### 🗺️ Roadmap

- ✅ Configurable languages — all 7 bundles (Latin, Chinese, Greek, Russian, Arabic, Korean, Thai) baked into the image
- 🔍 Structured extraction
- 📝 Markdown output
- 📊 Table parsing

---

## Quick Start

**Requirements:** Linux, NVIDIA driver 595+, Turing or newer GPU (RTX 20-series / GTX 16-series+).

```bash
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  ghcr.io/aiptimizer/turboocr:v2.1.1
```

First startup builds TensorRT engines from ONNX (~90s). The volume caches them for instant restarts. nginx (port 8000) reverse-proxies to Drogon (port 8080) for connection buffering — both start automatically.

```bash
curl -X POST http://localhost:8000/ocr/raw \
  --data-binary @document.png -H "Content-Type: image/png"
```

```json
{
  "results": [
    {"text": "Invoice Total", "confidence": 0.97, "bounding_box": [[42,10],[210,10],[210,38],[42,38]]}
  ]
}
```

---

## API

HTTP on port 8000, gRPC on port 50051 — single binary, shared GPU pipeline pool.

> **Important:** Use persistent connections (HTTP keep-alive). Sending many short-lived connections (e.g. one `curl` per request in a loop) can overwhelm the server and cause it to stall. All standard HTTP client libraries (`requests.Session`, `aiohttp`, Go `http.Client`, etc.) reuse connections by default.

### Endpoints

| Endpoint | Input | Description |
|----------|-------|-------------|
| `/health` | — | Returns `"ok"` |
| `/health/live` | — | Kubernetes liveness probe |
| `/health/ready` | — | Readiness probe — verifies GPU pipeline is responsive |
| `/ocr/raw` | Raw image bytes | Fastest path — PNG, JPEG, etc. |
| `/ocr` | `{"image": "<base64>"}` | For clients that can only send JSON |
| `/ocr/batch` | `{"images": ["<b64>", ...]}` | Multiple images in one request |
| `/ocr/pixels` | Raw BGR bytes + `X-Width`/`X-Height` headers | Zero-decode path |
| `/ocr/pdf` | Raw bytes, `{"pdf": "<b64>"}`, or `multipart/form-data` | All pages OCR'd in parallel |
| `/metrics` | — | Prometheus metrics (text exposition format) |
| gRPC | Raw bytes (protobuf) | Port 50051 — see `proto/ocr.proto` |

### Query Parameters

| Parameter | Endpoints | Values | Default |
|-----------|-----------|--------|---------|
| `layout` | all | `0` / `1` | `0` — include [layout regions](#layout-detection) (~20% throughput cost) |
| `mode` | `/ocr/pdf` | `ocr` / `geometric` / `auto` / `auto_verified` | `ocr` |
| `dpi` | `/ocr/pdf` | `50`–`600` | `100` — render resolution |

### Examples

```bash
# Image — raw bytes (fastest)
curl -X POST http://localhost:8000/ocr/raw \
  --data-binary @doc.png -H "Content-Type: image/png"

# Image — base64 JSON
curl -X POST http://localhost:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{"image":"'$(base64 -w0 doc.png)'"}'

# PDF — raw bytes
curl -X POST http://localhost:8000/ocr/pdf \
  --data-binary @document.pdf

# PDF — multipart (works from any client, including browsers)
curl -X POST http://localhost:8000/ocr/pdf \
  -F "file=@document.pdf"

# PDF — with layout + auto mode
curl -X POST "http://localhost:8000/ocr/pdf?layout=1&mode=auto" \
  --data-binary @document.pdf

# gRPC (grpcurl uses base64 for CLI; real clients send raw bytes)
grpcurl -plaintext -d '{"image":"'$(base64 -w0 doc.png)'"}' \
  localhost:50051 ocr.OCRService/Recognize
```

### Response Format

**Image endpoints** return:
```json
{"results": [{"text": "Invoice Total", "confidence": 0.97, "bounding_box": [[42,10],[210,10],[210,38],[42,38]]}]}
```

**With `?layout=1`**, a `layout` array is added. Each OCR result gets a `layout_id` linking it to the containing layout region:
```json
{
  "results": [{"text": "...", "confidence": 0.97, "id": 0, "layout_id": 2, "bounding_box": [...]}],
  "layout": [{"id": 0, "class": "header", "confidence": 0.91, "bounding_box": [...]},
             {"id": 2, "class": "table", "confidence": 0.95, "bounding_box": [...]}]
}
```

**PDF endpoint** wraps results per page:
```json
{
  "pages": [{
    "page": 1, "page_index": 0, "dpi": 100, "width": 1047, "height": 1389,
    "mode": "ocr", "results": [...]
  }]
}
```
Coordinate conversion: `x_pdf = x_px * 72 / dpi`.

### PDF Extraction Modes

| Mode | What it does | Speed |
|------|-------------|-------|
| `ocr` | Render + full OCR pipeline | Baseline |
| `geometric` | PDFium text layer only, no rasterization | ~10x faster |
| `auto` | Per-page: text layer if available, else OCR | Fastest for mixed PDFs |
| `auto_verified` | Full pipeline + replace with native text where sanity check passes | Slightly slower than OCR |

> [!CAUTION]
> **PDF text-layer trust model.** Modes other than `ocr` read the PDF's native text layer, which the PDF author controls. A malicious PDF can embed invisible text, remap glyphs via ToUnicode, or inject arbitrary strings that differ from what's visually rendered.
>
> **When to use each mode:**
> | Scenario | Recommended mode | Why |
> |----------|-----------------|-----|
> | Untrusted uploads (user-submitted PDFs) | `ocr` | Only trusts pixel data — immune to text-layer manipulation |
> | Internal/trusted documents | `auto` or `geometric` | Safe when you control the PDF source; much faster |
> | High-accuracy with verification | `auto_verified` | OCR runs first, then results are cross-checked against the text layer. Accepts native text only if it passes heuristic validation (character count, non-printable ratio < 10%, replacement char ratio < 5%, no rotation) |
>
> **Default:** `mode=ocr` (safest). Override per-request via `?mode=` query parameter or globally via `ENABLE_PDF_MODE` env var.
>
> **Deployment recommendation:** If your service accepts PDFs from untrusted sources, do **not** set `ENABLE_PDF_MODE` to `geometric` or `auto` globally. Keep the default `ocr` and only use text-layer modes for trusted internal workflows.

### Layout Detection

All endpoints accept `?layout=1` to detect document regions using [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) (25 classes):

`abstract` · `algorithm` · `aside_text` · `chart` · `content` · `display_formula` · `doc_title` · `figure_title` · `footer` · `footer_image` · `footnote` · `formula_number` · `header` · `header_image` · `image` · `inline_formula` · `number` · `paragraph_title` · `reference` · `reference_content` · `seal` · `table` · `text` · `vertical_text` · `vision_footnote`

<p align="center">
  <img src="tests/benchmark/comparison/images/layout_example.png" alt="Layout detection overlay" width="500">
  <br><sub>Layout detection overlay — color-coded regions: <span style="color:#9C27B0">paragraph_title</span>, <span style="color:#2196F3">text</span>, <span style="color:#00BCD4">chart</span>, <span style="color:#FFC107">figure_title</span>, <span style="color:#F44336">header</span>, <span style="color:#607D8B">footer</span>, <span style="color:#646464">number</span></sub>
</p>

---

## Benchmarks

FUNSD form-understanding dataset (50 pages, ~170 words/page). Same word-level F1 metric for all engines. Single RTX 5090.

![Accuracy](tests/benchmark/comparison/images/accuracy_v2.png)

<!--
OCR Accuracy — FUNSD · 50 images · ~174 words/img
| Engine | F1 (%) | Recall (%) | Precision (%) |
|---|---:|---:|---:|
| Turbo-OCR (C++/TRT) | 90.2 | 91.6 | 88.8 |
| PaddleOCR mobile latin | 86.6 | 85.5 | 88.2 |
| Qwen3-VL-2B | 84.3 | 82.8 | 87.5 |
| PaddleOCR-VL (pipeline) | 83.1 | 82.5 | 85.0 |
| EasyOCR (Python) | 63.0 | 66.2 | 60.4 |
-->

![Throughput](tests/benchmark/comparison/images/throughput_v2.png)

<!--
OCR Throughput — FUNSD Dataset · Higher is Better
| Engine | Throughput (img/s) |
|---|---:|
| Turbo-OCR (C++/TRT) | 268.4 |
| PaddleOCR mobile latin | 5.1 |
| EasyOCR (Python) | 2.8 |
| PaddleOCR-VL (pipeline) | 2.0 |
| Qwen3-VL-2B | 1.3 |
-->

![Latency](tests/benchmark/comparison/images/latency_v2.png)

<!--
OCR Latency — FUNSD Dataset · Lower is Better
| Engine | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Turbo-OCR (C++/TRT) | 11 | 16 |
| PaddleOCR mobile latin | 182 | 352 |
| Qwen3-VL-2B | 2859 | 6191 |
| PaddleOCR-VL (pipeline) | 1513 | 6517 |
| EasyOCR (Python) | 559 | 948 |
-->

<details>
<summary>Benchmark caveats</summary>

- **Crude accuracy metric.** Bag-of-words F1 ignores order and duplicate counts. CER or reading-order metrics would likely help VLM systems.
- **VLMs could run faster.** Served via off-the-shelf vLLM in fp16. Quantization, speculative decoding, or a dedicated stack would push throughput higher.
- **VLM prompts are untuned.** With prompt engineering both VLMs would likely surpass every CTC engine here.
- **Single domain.** FUNSD is English business forms; other document types would look different.

Reproduce: `python tests/benchmark/comparison/bench_turbo_ocr.py` (requires running server + `datasets` library).
</details>

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_LANG` | *(unset = latin)* | Language bundle: `latin`, `chinese`, `greek`, `eslav`, `arabic`, `korean`, `thai`. All bundles are baked into the image at build time — no runtime download. |
| `OCR_SERVER` | *(unset)* | With `OCR_LANG=chinese`, set to `1` to use the 84 MB PP-OCRv5 server rec instead of the 16 MB mobile rec. Ignored for other languages. |
| `PIPELINE_POOL_SIZE` | auto | Concurrent GPU pipelines (~1.4 GB each) |
| `DISABLE_LAYOUT` | `0` | Set to `1` to disable PP-DocLayoutV3 layout detection and save ~300-500 MB VRAM |
| `ENABLE_PDF_MODE` | `ocr` | Default PDF mode: `ocr` / `geometric` / `auto` / `auto_verified` |
| `DISABLE_ANGLE_CLS` | `0` | Skip angle classifier (~0.4 ms savings) |
| `DET_MAX_SIDE` | `960` | Max detection input size |
| `PORT` / `GRPC_PORT` | `8000` / `50051` | Server ports |
| `PDF_DAEMONS` / `PDF_WORKERS` | `16` / `4` | PDF render parallelism |
| `HTTP_THREADS` | `pool * 32` | Work pool threads for blocking inference |
| `MAX_PDF_PAGES` | `2000` | Maximum pages per PDF request |
| `LOG_LEVEL` | `info` | Log level: `debug` / `info` / `warn` / `error` |
| `LOG_FORMAT` | `json` | Log format: `json` (structured) / `text` (human-readable) |

Layout detection is **enabled by default**. The model is loaded at startup but only runs when a request includes `?layout=1`. Requests without `?layout=1` have zero overhead. Requests with `?layout=1` reduce throughput by ~20%. Set `DISABLE_LAYOUT=1` to skip loading the model entirely and save ~300-500 MB VRAM.

```bash
docker run --gpus all -p 8000:8000 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  -e PIPELINE_POOL_SIZE=3 \
  ghcr.io/aiptimizer/turboocr:v2.1.1
```

Add `MAX_PDF_PAGES` (default `2000`) to limit the number of pages processed per PDF request. `LOG_LEVEL` (`debug`/`info`/`warn`/`error`) and `LOG_FORMAT` (`json`/`text`) control structured logging output.

---

## Monitoring

### Prometheus Metrics

Scrape `GET /metrics` for Prometheus-compatible metrics:

```
turbo_ocr_requests_total{route="/ocr/raw",status="2xx"} 1042
turbo_ocr_request_duration_seconds_bucket{route="/ocr/raw",le="0.025"} 980
turbo_ocr_request_duration_seconds_sum{route="/ocr/raw"} 12.345
turbo_ocr_request_duration_seconds_count{route="/ocr/raw"} 1042
turbo_ocr_gpu_vram_used_bytes 9052815360
turbo_ocr_gpu_vram_total_bytes 33661911040
turbo_ocr_pipeline_pool_size 5
turbo_ocr_pool_exhaustions_total 0
turbo_ocr_request_bytes_total 49493243
turbo_ocr_request_body_avg_bytes 9407
```

### Response Headers

Every response includes:

| Header | Description |
|--------|-------------|
| `X-Request-Id` | UUID v7 (or propagated from client `X-Request-Id` header) |
| `X-Inference-Time-Ms` | End-to-end processing time in milliseconds |
| `Retry-After` | Seconds to wait (only on 503 responses) |

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic liveness check |
| `GET /health/live` | Kubernetes liveness probe |
| `GET /health/ready` | Readiness probe &mdash; verifies GPU pipeline is responsive |

### Structured Errors

All error responses return JSON with `Content-Type: application/json`:

```json
{"error": {"code": "EMPTY_BODY", "message": "Empty body"}}
```

Error codes: `EMPTY_BODY`, `INVALID_JSON`, `MISSING_IMAGE`, `BASE64_DECODE_FAILED`, `IMAGE_DECODE_FAILED`, `INVALID_PARAMETER`, `UNSUPPORTED_PARAMETER`, `INVALID_DPI`, `INVALID_DIMENSIONS`, `DIMENSIONS_TOO_LARGE`, `BODY_SIZE_MISMATCH`, `MISSING_HEADER`, `INVALID_HEADER`, `EMPTY_BATCH`, `MISSING_FILE`, `MISSING_PDF`, `INVALID_MULTIPART`, `PDF_RENDER_FAILED`, `PDF_TOO_LARGE`, `EMPTY_PDF`, `SERVER_BUSY`, `NOT_READY`, `INFERENCE_ERROR`.

---

## Building from Source

| Dependency | GPU | CPU |
|-----------|:---:|:---:|
| GCC 13.3+ / C++20 | x | x |
| CUDA + TensorRT 10.2+ | x | |
| OpenCV 4.x | x | x |
| Drogon 1.9+ | x | x |
| gRPC + Protobuf | x | |
| ONNX Runtime 1.22+ | | x |

Wuffs, Clipper, PDFium vendored in `third_party/`.

```bash
# Docker (recommended)
docker build -f docker/Dockerfile.gpu -t turboocr .
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr turboocr

# CPU only (Docker) — ~2-3 img/s, mainly for testing
docker build -f docker/Dockerfile.cpu -t turboocr-cpu .
docker run -p 8000:8000 turboocr-cpu

# Native build — PP-OCRv5 models auto-fetched into ./models/ on first build
cmake -B build -DTENSORRT_DIR=/usr/local/tensorrt
cmake --build build -j$(nproc)
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/paddle_highspeed_cpp

# CPU-only native
cmake -B build_cpu -DUSE_CPU_ONLY=ON
cmake --build build_cpu -j$(nproc)
./build_cpu/paddle_cpu_server

# If your distro's gRPC CMake config conflicts with system protobuf,
# add -DCMAKE_DISABLE_FIND_PACKAGE_gRPC=ON to fall back to pkg-config.
# To skip the model auto-fetch (e.g. in CI), add -DFETCH_MODELS=OFF.
```

---

## Supported Languages

Set via the `OCR_LANG` environment variable. Every supported language bundle is baked into the image at build time from the pinned PP-OCRv5 GitHub Release (SHA256-verified). No runtime downloads, no network dependency at container start.

| `OCR_LANG` | Script / family | Notes |
|---|---|---|
| *(unset)* / `latin` | Latin + basic Greek (English, German, French, Italian, Polish, Czech, …) | 836-char dict; what powers the benchmarks above |
| `chinese` | Simplified + Traditional Chinese | 18,385-class mobile rec (16 MB); set `OCR_SERVER=1` for the 84 MB server variant |
| `greek` | dedicated Greek rec | 356-class Greek-specialized rec (7.8 MB) — higher accuracy than Latin's combined dict |
| `korean` | Hangul + basic Latin | 11,947-class rec (13 MB) |
| `arabic`, `eslav`, `thai` | per-script PP-OCRv5 | 7-8 MB each |

```bash
# Chinese
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  -e OCR_LANG=chinese \
  ghcr.io/aiptimizer/turboocr:v2.1.1
```

> **Volume tip:** use a **named** volume (`trt-cache:`) as shown above, not a
> host bind-mount. Named volumes auto-populate from the image on first use,
> so the baked language bundles survive. A bind-mount of an empty host
> directory would shadow `/home/ocr/.cache/turbo-ocr` and leave the server
> with nothing to load.

Run `tests/language_smoketest.py` to verify any language end-to-end on your
hardware (renders a short phrase, OCRs it, checks char-recall against a
per-language threshold).

---

## Acknowledgements

This project builds on the work of several open-source projects:

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (Baidu) — PP-OCRv5 detection, recognition, and classification models. PP-DocLayoutV3 layout detection model. This project would not exist without their research and pre-trained weights.
- **[Drogon](https://drogon.org)** — high-performance async C++ HTTP framework
- **[Wuffs](https://github.com/google/wuffs)** — fast PNG decoder by Google (vendored)
- **[PDFium](https://pdfium.googlesource.com/pdfium/)** — PDF rendering and text extraction (vendored)
- **[Clipper](http://www.angusj.com/delphi/clipper.php)** — polygon clipping for text detection post-processing (vendored)

## License

MIT. See [LICENSE](LICENSE).

<p align="center">
  <sub>Main Sponsor: <a href="https://miruiq.com"><strong>Miruiq</strong></a> — AI-powered data extraction from PDFs and documents.</sub>
</p>
