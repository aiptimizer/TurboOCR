# Upgrading to v3 (breaking changes)

v3 moves the default engine from PP-OCRv5 to **PP-OCRv6** and turns the server into
a full document parser — adding layout-aware **tables → HTML** and **formulas →
LaTeX** (both new in v3). The changes since v2.3 sort into three buckets; only the
first needs any action.

## Breaking / config-incompatible (action required)

- **Server binary renamed.** The GPU server is now `turboocr-server` (was `paddle_highspeed_cpp`) and the CPU server `turboocr-server` (was `paddle_cpu_server`). Update any direct launch command, systemd unit, or wrapper script. Docker users are unaffected — the image entrypoint/`CMD` handle it.
- **Clear the TensorRT engine cache and pull the new models on upgrade.** PP-OCRv6 ships new det/rec ONNX from the `models-v3.0.0-ppocrv6` release, so cached v5 engines must rebuild — wipe `~/.cache/turbo-ocr` (or the mounted `trt-cache` volume) once. The Docker image and the native build fetch the new release automatically; pull the new image (or rebuild) to get it.

## Default-behaviour changes (no config change, but output or runtime differ)

- **PP-OCRv6 is the default engine** (was PP-OCRv5), shipped as three tiers (`tiny`/`small`/`medium`) from the new `models-v3.0.0-ppocrv6` release. Recognition output changes vs v5.
- **Default tier is `tiny`** (max throughput). Set `OCR_MODEL=small` or `medium` for higher accuracy.
- **`LAYOUT_MERGE_MODE` default changed to `all`** (was effectively `large` / keep-outer). `all` keeps every detected box and drops nothing, so formulas/tables/titles nested inside a larger region survive (≈ +0.008 table TEDS, ≈ −0.006 formula CDM on OmniDocBench). Set `LAYOUT_MERGE_MODE=outer` to restore the previous behaviour. The mode *names* also changed (`outer`/`inner`/`all`, formerly `large`/`small`/`union`), but the old names still work as **deprecated aliases** — so the rename itself is not breaking. Modes: `outer` keeps the outer/container box and drops boxes nested inside it; `inner` keeps the innermost boxes and drops the pure containers; `all` keeps both.
- **Requests now time out at 60 s** instead of hanging unbounded. `REQUEST_TIMEOUT_MS` default changed `0` → `60000`: a wedged GPU slot returns `504 INFERENCE_TIMEOUT` and frees itself. Set `REQUEST_TIMEOUT_MS=0` to opt back into the old unbounded wait. A companion watchdog (`PIPELINE_HARD_KILL_MS`, default `600000` = 10 min) `_Exit`s the process for the orchestrator to restart **only** if a worker stays wedged mid-CUDA long after a recycle was already requested — so a genuine hang can now terminate the process instead of leaking a slot forever (this watchdog is inert when `REQUEST_TIMEOUT_MS=0`).
- **Detection resize defaults changed** (max-side `960` → `limit_type=min`, `limit_side_len=64`, `max_side_limit=1280`), so detection boxes — and therefore OCR output — differ slightly. Tune via `DET_MAX_SIDE_LIMIT` / `DET_LIMIT_TYPE` / `DET_LIMIT_SIDE_LEN` (`DET_MAX_SIDE` still honored).
- **GPU out-of-memory now returns `500 INFERENCE_ERROR`** instead of a blank `200`, and a sticky CUDA fault `_Exit`s the process for a clean restart. Under sustained overload, queued work whose client deadline already elapsed is dropped (the caller gets its `504`) rather than processed late. Clients should handle 5xx/504 and retry.
- **A bare launch announces text-only mode.** With neither `FORMULA_BACKEND` nor `TABLE_BACKEND` set, the server runs text-only (tables/formulas empty) and now logs a one-time `[Pipeline] NOTE: table + formula stages are DISABLED — running TEXT-ONLY …`, so a text-only run can't be silently mistaken for a full-document one.
- **New input-size caps** (previously-accepted requests are now rejected): `/ocr/batch` and gRPC `RecognizeBatch` cap at **1024 images** → `400 BATCH_TOO_LARGE` (split the batch or raise `MAX_BATCH_IMAGES`); `/ocr/pdf` rendered pages cap at **~40 MP/page** → very large pages at high DPI fail to render (lower `?dpi=` or raise `MAX_PDF_PAGE_PIXELS_MP`); and `/ocr`, `/ocr/raw`, `/ocr/batch`, `/infer` now also reject images over **128 MP total area** → `400 PIXELS_TOO_LARGE`, in addition to the existing per-side `MAX_IMAGE_DIM` guard (downscale, or raise `MAX_IMAGE_PIXELS_MP`).

## Additive / transparent (nothing to do)

- **`OCR_MODEL` is the new selector name; `OCR_LANG` still works** as a deprecated alias (warns on use), so this is backward-compatible. Select by tier/model name (`tiny`/`small`/`medium`, or `arabic`/`eslav`/`korean`/`thai`/`greek`).
- **New: formula recognition → LaTeX.** `FORMULA_BACKEND=ppformulanet_s` adds an **in-process pure-C++ PP-FormulaNet-S recognizer** (ORT-CUDA-13 on the GPU build, ORT-CPU on the CPU build — no Python, no sidecar). Use `ppformulanet_plus_m` for Chinese-capable formulas, or `vlm` to route to a VLM. Opt-in; off by default.
- **New: table recognition → HTML.** `TABLE_BACKEND=slanext` adds the SLANet-Plus structure model (TRT FP16 encoder + hand-written C++ decoder); the encoder auto-resolves from the bundled model, so no extra path env is needed. Opt-in; off by default.
- **New `POST /ocr/markdown` route** (GPU build) exports a parsed page as faithful Markdown. Purely additive — existing routes are unchanged.
- **Oversized-image guard on `/infer`.** Like the other image routes, `/infer` now rejects inputs whose dimensions exceed `MAX_IMAGE_DIM` (default `16384`) with `400 DIMENSIONS_TOO_LARGE` (a decompression-bomb guard). Only affects callers that were sending images larger than 16384 px on a side.
- **New `*_degraded` response signals.** When a configured stage produces nothing, the JSON now carries `text_degraded` / `table_degraded` / `formula_degraded` (+ a `*_warning` string) on `/ocr`, `/ocr/raw`, `/ocr/batch` and `/ocr/pdf`, and `/ocr/markdown` sets an `X-OCR-Degraded` header — so a partial result is never a silent clean `200` (a configured-but-failed stage also now fails at boot rather than serving empties). New fields only; ignore them and nothing changes.
- **New: document auto-rotation.** `?autorotate=1` straightens rotated/skewed pages with a PP-LCNet orientation model before OCR (opt-in per request).
- **New `GET /capabilities`** (runtime feature/route discovery) — opt-in and additive.

## v3.5.0 → next: error-code precedence on multi-defect requests

Request validation is now one shared, table-driven gate
(`include/turbo_ocr/core/capability_table.def`), which changes **which**
`400` a client gets when a single request has several defects at once. Status
codes and the codes themselves are unchanged; only the precedence moved.
Requests with exactly one defect get the same rejection as v3.5.0.

- Capability availability (`LAYOUT_DISABLED`, `TABLE_BACKEND_DISABLED`,
  `FORMULA_BACKEND_DISABLED`, `AUTOROTATE_DISABLED`) is now checked **first**,
  before routing-override validation, `text=0` combination checks, and the
  PDF-specific checks (`INVALID_DPI`, mode, image params). Example:
  `?tables=1&route_table=bogus` with no table backend returns
  `TABLE_BACKEND_DISABLED` (was `ROUTING_UNKNOWN_OVERRIDE`).
- Availability rejections triggered by an implied dependency
  (`?reading_order=1` without the layout model) keep their v3.5.0 code
  (`LAYOUT_DISABLED`) with a reworded message: `layout is required for this
  request but …` instead of claiming `layout=1` was sent.

These are deliberate: one gate, one rejection per condition, identical across
HTTP query, HTTP JSON body, and gRPC. Automation should key on the error `code`
field of the single defect it is probing, never on which of several coexisting
defects wins.
