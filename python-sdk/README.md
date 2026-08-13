# turboocr

Python for [TurboOCR](https://github.com/aiptimizer/TurboOCR): the typed client
for a TurboOCR server (sync + async, HTTP + gRPC, layout-aware Markdown
rendering, searchable-PDF generation) — and, via one extra, the **in-process
native engine**, so the same package OCRs locally with no server at all.

[![PyPI](https://img.shields.io/pypi/v/turboocr.svg?v=1)](https://pypi.org/project/turboocr/)
[![Python](https://img.shields.io/pypi/pyversions/turboocr.svg?v=1)](https://pypi.org/project/turboocr/)
[![Typed](https://img.shields.io/badge/typed-PEP_561-blue.svg)](https://peps.python.org/pep-0561/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

- [Install](#install) · [Quickstart](#quickstart) · [What you get](#what-you-get)
- [Examples](examples/) · [API reference](docs/) · [CLI](#cli) · [Errors](#errors)

## Install

```bash
pip install turboocr             # HTTP client + CLI + searchable-PDF
pip install 'turboocr[grpc]'     # add the gRPC transport
pip install 'turboocr[all]'      # everything optional (currently == [grpc])
```

To also run OCR **in-process** (no server), add exactly one backend extra —
it pins the matching native `turboocr-engine-*` wheel:

```bash
pip install 'turboocr[cpu]'       # any CPU — and Apple Silicon (Metal/ANE build)
pip install 'turboocr[cuda12]'    # NVIDIA (TensorRT + CUDA EP), driver R525+
pip install 'turboocr[cuda13]'    # same, CUDA 13 — driver R580+
pip install 'turboocr[openvino]'  # Intel CPU / iGPU / Arc / NPU
pip install 'turboocr[rocm]'      # AMD (ROCm)
```

`turboocr doctor` names the right one for your machine. With an engine
installed, `turboocr.OCR()` runs locally:

```python
import turboocr

page = turboocr.OCR().read("invoice.png")   # in-process, no server
print(page.text)
```

Without one, the client API works everywhere and `turboocr.OCR()` raises an
ImportError naming the extra to install. Feature extras combine with any
backend: `'turboocr[cuda,pdf]'` (PDF in/out), `[pandas]`.

Requires Python 3.12+.

## Quickstart

Start a [TurboOCR](https://github.com/aiptimizer/TurboOCR) server (the C++/CUDA
OCR engine; for serverless in-process OCR see the backend extras above):

```bash
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  -e TABLE_BACKEND=slanext -e FORMULA_BACKEND=ppformulanet_s \
  ghcr.io/aiptimizer/turboocr:latest
```

The default `OCR_MODEL=tiny` covers Latin + Chinese + Japanese; `small`/`medium`
trade speed for accuracy, and `arabic`, `eslav`, `korean`, `thai`, `greek` are
baked in too. The two backend env vars enable table → HTML and formula → LaTeX
recognition (strict per-request opt-ins). See the
[TurboOCR repo](https://github.com/aiptimizer/TurboOCR) for build-from-source,
benchmarks, and the full set of server env vars.

Then recognise an image and turn a PDF into Markdown:

```python
from pathlib import Path

from turboocr import Client

with Client(base_url="http://localhost:8000") as client:
    # Image OCR
    img = client.recognize_image("page.png", layout=True, include_blocks=True)
    print(f"{len(img.results)} text items, {len(img.blocks)} blocks")
    print(img.text)

    # PDF → Markdown file (rendered server-side: tables → HTML, formulas →
    # LaTeX, figures embedded as data URIs — a real, self-contained .md)
    Path("paper.md").write_text(client.pdf_markdown("paper.pdf", dpi=150))

    # Tables + formulas as structured fields (strict opt-in, v3.1+ server)
    rich = client.recognize_image("paper.png", tables=True, formulas=True)
    for table in rich.tables:
        print(table.html)
    for formula in rich.formulas:
        print(formula.latex)

    # Searchable PDF (invisible text overlay)
    overlay = client.make_searchable_pdf("scan.pdf", dpi=200)
    open("scan.searchable.pdf", "wb").write(overlay)
```

That's the 80% case. Full runnable examples for async, gRPC, batch, retries,
custom `httpx.Client`, hooks, Markdown styling, folder pipelines, and more live
in [`examples/`](examples/) — every script runs end-to-end against the bundled
ACME invoice fixture.

## What you get

- **Sync + async, HTTP + gRPC.** Four clients (`Client`, `AsyncClient`,
  `GrpcClient`, `AsyncGrpcClient`) with identical method surfaces.
- **Typed, immutable responses (pydantic v2).** IDE autocomplete, and if a newer
  server adds a field your SDK doesn't know about, parsing still succeeds — the
  extra lands on `.model_extra` instead of crashing.
- **Layout-aware Markdown.** `render_to_markdown(...)` walks the reading order
  and maps each layout class (`doc_title`, `display_formula`, `table`, …) to a
  Markdown construct. Pluggable via `MarkdownStyle`.
- **Searchable PDFs.** `make_searchable_pdf(...)` overlays an invisible text
  layer aligned to the page geometry. Auto-discovers a Unicode font for
  non-Latin scripts, or pass `font_path=`.
- **Production-friendly.** Configurable retry policy (HTTP status + gRPC status
  + `Retry-After`), per-request timeouts, custom `httpx.Client`, `on_request` /
  `on_response` event hooks, uuid7 `X-Request-ID` per call.
- **Tables → HTML, formulas → LaTeX.** `tables=True` / `formulas=True`
  (server v3.1+, strict opt-in) populate `response.tables[*].html` and
  `response.formulas[*].latex`; `client.capabilities()` tells you what the
  running server has loaded.
- **Server-side Markdown.** `client.pdf_markdown(...)` converts a whole PDF in
  one call (`as_pages=True` for per-page chunks — the RAG-friendly shape);
  `client.page_markdown(...)` does a single image. `render_to_markdown(...)`
  stays for client-side, style-customizable rendering.
- **Per-page streaming.** `client.stream(...)` yields NDJSON events as each
  page completes, so you can start consuming page 1 while page N is still
  being OCR'd.
- **Precise exception hierarchy.** Maps the server's error codes to typed
  exceptions — see [Errors](#errors).
- **`turbo-ocr` CLI** included in the default install.

## Configuration

```python
from turboocr import Client, RetryPolicy

client = Client(
    base_url="http://localhost:8000",   # or TURBO_OCR_BASE_URL env
    api_key="sk-...",                   # or TURBO_OCR_API_KEY env
    auth_scheme="bearer",               # "bearer" | "x-api-key"
    timeout=30.0,
    default_headers={"X-Tenant": "acme"},
    retry=RetryPolicy(attempts=5, backoff=0.5),
)
```

Pass `http_client=httpx.Client(...)` for custom TLS, connection limits, or
proxies — see [`examples/08_custom_httpx_client.py`](examples/08_custom_httpx_client.py).

Retry defaults: HTTP `{429, 502, 503, 504}`, gRPC
`{UNAVAILABLE, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED}`, 3 attempts, exponential
backoff + jitter, `Retry-After` honoured. Tune via `RetryPolicy(...)` — see
[`examples/07_retry_and_timeout.py`](examples/07_retry_and_timeout.py).

## Errors

```
TurboOcrError
├── APIConnectionError       # transport-level
│   ├── Timeout
│   ├── NetworkError
│   └── ProtocolError
├── InvalidParameter         # 4xx: bad params / headers / dims
├── EmptyBody                # 4xx: empty body / batch / PDF
├── BackendDisabled          # tables/formulas/autorotate without that backend
│   └── LayoutDisabled       # layout requested with DISABLE_LAYOUT=1
├── ImageDecodeError         # bad bytes / bad base64
├── DimensionsTooLarge       # image / PDF / batch over server limits
├── PoolExhausted            # "Server at capacity" / SERVER_BUSY
├── PdfRenderError           # PDF rasterization failed
├── InferenceTimeout         # per-request deadline elapsed (504)
└── ServerError              # 5xx, no specific code
```

Server-side exceptions carry `.code`, `.status_code`, and `.payload`. Transport
exceptions inherit from `APIConnectionError`.

| Symptom | Cause | Fix |
|---|---|---|
| `NetworkError: Connection refused` | server not running | start the docker container (above) |
| `DimensionsTooLarge` | image > `MAX_IMAGE_DIM` (default 16384) | downscale, or raise the server limit |
| `LayoutDisabled` | server started with `DISABLE_LAYOUT=1` | restart without that env var |
| `BackendDisabled` | `tables=True`/`formulas=True` without the backend | start with `TABLE_BACKEND=slanext` / `FORMULA_BACKEND=ppformulanet_s` |
| `PoolExhausted` | server queue full | retry with backoff, or scale `PIPELINE_POOL_SIZE` |
| `Timeout` | per-request timeout hit | pass `timeout=N`, or raise `RetryPolicy.attempts` |

## CLI

```bash
turbo-ocr ocr page.png --output markdown --tables --formulas
turbo-ocr pdf doc.pdf --dpi 150 --output json
turbo-ocr markdown doc.pdf -o doc.md        # server-side PDF → Markdown
turbo-ocr searchable-pdf doc.pdf -o out.pdf --font-path /path/to/font.ttf
turbo-ocr capabilities
turbo-ocr health --ready
```

`--output` accepts `json | blocks | text | markdown`. Reads `TURBO_OCR_BASE_URL`
and `TURBO_OCR_API_KEY` from the environment. Run `turbo-ocr --help`
for the full surface.

## Logging

```python
import logging
logging.getLogger("turboocr").setLevel(logging.DEBUG)
```

Emits `method path -> status (Xms) [req=<short-id>]` per HTTP request. Retry
warnings go to `turboocr.retry` / `turboocr.grpc.retry`. Searchable-PDF font
resolution logs to `turboocr.searchable_pdf`. Every HTTP request sends a uuid7
`X-Request-ID` header (gRPC uses `x-request-id` metadata).

## Learn more

- [`examples/`](examples/) — 14 runnable scripts (each runs against the bundled
  ACME invoice fixture, no server config needed beyond `TURBO_OCR_BASE_URL`)
- [`docs/`](docs/) — full docs source (MkDocs + mkdocstrings, deployed at
  https://aiptimizer.github.io/TurboOCR-python/). Preview locally with
  `uv run --extra docs mkdocs serve -f docs/mkdocs.yml`
- Server compatibility: `SERVER_API_VERSION_MIN` /
  `SERVER_API_VERSION_MAX_EXCLUSIVE` document the supported server range;
  `extra="allow"` on response models means additive server changes don't break
  parsing

## Testing

```bash
pytest -q                                                # offline (respx)
TURBO_OCR_BASE_URL=http://localhost:8000 pytest tests/integration -v
python examples/03_searchable_pdf.py                         # smoke test
```

## License

MIT. See [LICENSE](LICENSE).

<p align="center">
  <a href="https://turboocr.com"><strong>turboocr.com</strong></a> · <a href="https://github.com/aiptimizer/TurboOCR"><strong>⭐ Star TurboOCR on GitHub</strong></a><br>
  <sub>Sponsored by <a href="https://miruiq.com"><strong>Miruiq</strong></a> — AI-powered data extraction from PDFs and documents — and <a href="https://diaiq.com"><strong>DiaIQ</strong></a>.</sub>
</p>
