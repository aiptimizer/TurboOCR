# OCR Benchmark Suite

Scripts that produce the head-to-head comparison shown in the root README.
Every engine is scored with the **same word-level F1 metric** against the
FUNSD test split (first 50 images), so accuracy numbers are directly
comparable.

## Layout

```
tests/benchmark/comparison/
├── README.md                        (this file)
├── bench_turbo_ocr.py               Turbo-OCR via HTTP
├── bench_paddleocr_latin.py         PaddleOCR Python with matching weights
├── bench_easyocr.py                 EasyOCR (GPU)
├── bench_vlm_async.py               Generic async VLM client (Qwen3-VL etc.)
├── bench_paddleocr_vl_pipeline.py   PaddleOCR-VL 2-stage pipeline
├── bench_endpoints.py               Turbo-OCR /ocr, /ocr/raw, /ocr/pdf… cross-check
├── generate_plots.py                Rebuild the PNGs in images/
└── images/                          Plots rendered into the root README
```

## Environment

All scripts share one `uv` project — PyTorch nightly cu130 + vLLM nightly for
Blackwell SM120. From this directory:

```bash
uv sync
```

## Running the benchmarks

**Turbo-OCR (this repo)** — start the server on port 8000 first, then:

```bash
.venv/bin/python bench_turbo_ocr.py
```

**PaddleOCR Python** — same PP-OCRv5 mobile latin weights, apples-to-apples:

```bash
.venv/bin/python bench_paddleocr_latin.py
```

**EasyOCR** — tuned thresholds (`text_threshold=0.5, low_text=0.3`) to catch
small form text; beamsearch was dropped because it triggers numeric overflow
on FUNSD.

```bash
.venv/bin/python bench_easyocr.py
```

**Qwen3-VL-2B via vLLM** — separate terminal for the server:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8077 --host 0.0.0.0 \
  --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.75

.venv/bin/python bench_vlm_async.py \
  Qwen/Qwen3-VL-2B-Instruct "Qwen3-VL-2B" 8077 "OCR:"
```

**PaddleOCR-VL** — 2-stage pipeline (layout detector on local GPU, VLM via vLLM):

```bash
vllm serve PaddlePaddle/PaddleOCR-VL --port 8077 --host 0.0.0.0 \
  --served-model-name PaddleOCR-VL-1.5-0.9B \
  --max-num-seqs 64 --gpu-memory-utilization 0.75

.venv/bin/python bench_paddleocr_vl_pipeline.py
```

**Regenerate plots:**

```bash
.venv/bin/python generate_plots.py
```

## Metric

Word-level F1 per image, computed identically for every engine:

```
pred = { tok.lower() for tok in re.split(r"[^A-Za-z0-9]+", output) if len(tok) >= 2 }
gt   = { tok.lower() for tok in re.split(r"[^A-Za-z0-9]+", " ".join(sample["words"])) if len(tok) >= 2 }
f1   = 2 · |pred ∩ gt| / (|pred| + |gt|)
```

Tokens shorter than 2 chars are dropped to avoid scoring punctuation noise.

## What is and isn't measured

Every script pre-loads images, runs a **warmup** (2 calls), and only then
starts the per-image timer. These costs are therefore **excluded**:

- Dataset download and FUNSD loading
- Model download, ONNX→TRT engine build, Python library import
- GPU / CUDA / cuDNN initialization
- First-call lazy allocation inside each framework

Each script times the **per-image predict path only**. Scope per engine:

| Engine           | What the timer covers                                                         |
|------------------|-------------------------------------------------------------------------------|
| Turbo-OCR        | HTTP POST PNG bytes → server PNG decode → det + cls + rec → JSON response    |
| PaddleOCR Python | `engine.predict(path)` — disk read + decode + det + cls + rec                |
| EasyOCR          | `reader.readtext(path)` — disk read + decode + det + rec                     |
| Qwen3-VL-2B      | HTTP POST base64 PNG → vLLM inference → response                             |
| PaddleOCR-VL     | `pipe.predict(path)` — disk read + local layout + HTTP to vLLM for rec      |

Known asymmetries (all minor, all documented):

- **Turbo-OCR is the only engine with HTTP overhead** in its measurement
  (~1.1 ms localhost round-trip, measured separately). That's roughly 10% of
  its 11 ms p50 — i.e. the in-process gap vs PaddleOCR Python would be
  *slightly larger* than the numbers shown.
- **PaddleOCR Python and EasyOCR read the same file from disk on every call.**
  The OS page cache makes this effectively free (<0.1 ms) after the first
  iteration.
- **Warmup count is 2 for every engine except PaddleOCR-VL** (1, because its
  first call initializes the full pipeline). Any residual first-call cost
  would hurt PaddleOCR-VL's reported latency, not help it.

Throughput columns are measured with whatever concurrency the engine can
safely support from a single process:

- Turbo-OCR: **c=16** concurrent HTTP requests (server runs a 3-pipeline pool).
- VLMs: **c=8** via async OpenAI client + semaphore.
- PaddleOCR Python, EasyOCR: **sequential** — neither library is thread-safe
  from a single Python process (`predict()` raises `AssertionError` under
  `ThreadPoolExecutor`; EasyOCR's reader shares mutable GPU state).

## Endpoint cross-check

`bench_endpoints.py` sends 5 FUNSD pages through every Turbo-OCR ingress path
and verifies they return equivalent output. Summary (5 pages):

| Endpoint            |    F1 | Word-set overlap vs `/ocr/raw PNG` |
|---------------------|------:|-----------------------------------:|
| `/ocr` (base64 PNG) | 92.3% |                               100% |
| `/ocr/raw` PNG      | 92.3% |                               100% |
| `/ocr/raw` JPEG     | 93.1% |                ~94% (quality 95)   |
| `/ocr/pixels` BGR   | 92.3% |                               100% |
| `/ocr/batch` PNG    | 92.3% |                               100% |
| `/ocr/pdf`          | 92.4% |           ~90% (PDFium re-render)  |

PNG paths are bit-identical. JPEG and PDF differ only from lossy
compression / page re-rendering.

## Notes

- **Tesseract** is not included — the CPU binary was not installed on the
  benchmark host, and its accuracy on dense FUNSD-style forms is known to
  trail every engine in this comparison.
- **GLM-OCR** was tested but removed — it enters a repetition loop on
  Blackwell/vLLM on ~28% of pages regardless of flags (`TRITON_ATTN`,
  `enforce-eager`, `bf16`). Known upstream compatibility issue with SM120.
- All VLM benchmarks use `temperature=0.0` and `max_tokens=2048`.
