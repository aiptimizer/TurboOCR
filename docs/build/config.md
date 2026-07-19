# Configuration

!!! abstract "TL;DR"
    Every runtime knob is an **environment variable**, and most are also a
    **CLI flag** that overrides the env value. Configuration is loaded once
    at startup, strict-validated, and the server **refuses to bind** on any
    malformed or out-of-range value. Inspect the resolved config without
    booting the pipeline with `--print-config` or `--check-config`.

Configuration is parsed by `ServerConfig::load_or_die`
(`include/turbo_ocr/server/server_config.h`) in two passes: environment
variables first, then CLI flags (flags win). Both override the built-in
defaults. The same call site serves the GPU (`turboocr-server`) and CPU
(`turboocr-cpu-server`) binaries; a handful of defaults differ by profile
and are called out below.

!!! tip "Inspect the resolved config"

    ```bash
    turboocr-server --help            # full CLI flag listing
    turboocr-server --print-config    # resolved config as JSON, exit 0
    turboocr-server --check-config    # validate only: exit 0 if valid, 2 on errors
    ```

    `--print-config` and `--check-config` both run the full env + flag
    parse and cross-field validation, then exit before any model loads —
    safe to run against a production config.

## Precedence

1. CLI flag (highest)
2. Environment variable
3. Built-in default (lowest)

Out-of-range integers, unknown enum values, and `PORT == GRPC_PORT` are
**fatal**: the loader prints a `[config error]` list and `exit(2)`. Some
conditions are advisory **warnings** (logged, non-fatal), e.g.
`PDF_WORKERS > PDF_DAEMONS` (excess workers idle) and
`MAX_BODY_MEMORY_MB > MAX_BODY_MB` (clamped to the body cap).

## Model selection

| Variable | Default | Description |
|---|---|---|
| `OCR_MODEL` | `tiny` | OCR model registry entry. PP-OCRv6: `tiny` / `small` / `medium`. PP-OCRv5: `arabic` / `eslav` / `korean` / `thai` / `greek`. Unknown value is fatal and prints the valid list. |
| `OCR_LANG` | *(unset)* | **Deprecated** alias of `OCR_MODEL`; warns on use. `OCR_MODEL` wins when both are set. |

## Detection tuning

These override the selected model's per-model detection config. Each is
read where the detector is constructed (`read_det_resize` / `read_db_params`),
so the effective value reported by `--print-config` already folds them in.

| Variable | Default | Description |
|---|---|---|
| `DET_LIMIT_TYPE` | per-model (`min`) | Resize policy: `min` grows the shorter side to `DET_LIMIT_SIDE_LEN`; `max` shrinks the longer side. |
| `DET_LIMIT_SIDE_LEN` | per-model (`64`) | Target side length for the resize policy. |
| `DET_MAX_SIDE_LIMIT` | per-model (`1280`) | Caps the longer resized side. Official PaddleOCR uses 4000, but that OOMs the pre-allocated pool; `1280` runs the vast majority of documents at native resolution. |
| `DET_MAX_SIDE` | *(derived from `DET_MAX_SIDE_LIMIT`)* | Single-knob override of the TRT engine optimization-profile MAX side. Bounds `[32, 4096]`. Changing it invalidates the cached engine and forces a one-time rebuild. CLI: `--det-max-side`. |
| `DET_DB_THRESH` | per-model (`0.2`) | DB probability-map binarization threshold. |
| `DET_BOX_THRESH` | per-model (`0.45`, `0.40` for `tiny`) | Per-box mean-score cutoff. |
| `DET_UNCLIP` | per-model (`1.4`) | Polygon expansion ratio. |

## Model path overrides

Explicit per-stage paths win independently over `OCR_MODEL`. On the GPU
build the variables are `*_ONNX`; on the CPU build they are `*_MODEL`.

| Variable | Default | Description |
|---|---|---|
| `DET_ONNX` / `DET_MODEL` | per-model | Detection model path. CLI: `--det-onnx`. |
| `REC_ONNX` / `REC_MODEL` | per-model | Recognition model path. |
| `CLS_ONNX` / `CLS_MODEL` | `models/cls.onnx` | Angle-classifier model: a path, or a shorthand variant name — `x0_25` (default tiny) / `x1_0` (full-width PP-LCNet, better on rotated lines; expects `models/cls_x1_0.onnx`). When set explicitly, a missing/unloadable file refuses to start instead of silently disabling the classifier. CLI: `--cls-onnx`. |
| `CLS_ALL_BOXES` | `0` | `1` runs the 0°/180° orientation classifier on **every** text crop instead of only vertical-looking ones (h ≥ 1.5·w). Detection geometry gives each line's axis but cannot spot an upside-down horizontal line — enable for scans with mixed per-line orientations (0/90/180/270 on one page). Upright documents gain nothing; leave off for speed. |
| `REC_DICT` | per-model | Recognition character dictionary. |
| `DOC_ORI_ONNX` | `models/doc_ori.onnx` | Document-orientation model (PP-LCNet_x1_0_doc_ori) for `/ocr/pdf?autorotate=1`. If the file is absent, autorotate requests return `400 AUTOROTATE_DISABLED`; nothing else is affected. |

## Layout

| Variable | Default | Description |
|---|---|---|
| `DISABLE_LAYOUT` | `0` | `1` skips loading PP-DocLayoutV3 entirely (smaller startup, ~300–500 MB less VRAM, no `?layout=1`). CLI: `--disable-layout`. |
| `LAYOUT_ONNX` | `models/layout/layout.onnx` | Layout-detection model path. CLI: `--layout-onnx`. |
| `LAYOUT_TRT` | *(unset)* | Pre-built layout TensorRT engine (GPU only); overrides the `LAYOUT_ONNX` JIT build. CLI: `--layout-trt`. |
| `LAYOUT_MERGE_MODE` | `all` | How nested layout boxes are reconciled. `all` (default) keeps every box, so formulas/tables/titles the model nests inside a larger region are never dropped. `outer` keeps the outer regions and drops boxes nested inside them; `inner` keeps the innermost boxes and drops the pure containers. The old `large`/`small`/`union` names are still accepted as deprecated aliases of `outer`/`inner`/`all`. `outer` collapses **forms**, where every field sits inside an outer frame — use `all` or `inner` there. |

!!! warning "Migration: `ENABLE_LAYOUT` was removed"
    `ENABLE_LAYOUT` is no longer supported and is **fatal if set**. Layout
    is on by default — to disable it set `DISABLE_LAYOUT=1`, or simply
    remove the variable.

## Tables & formulas

Both are **opt-in**: a stage loads only when its backend env var is set at
startup, and runs only when the request passes `?tables=1` / `?formulas=1`.
Weights are baked into the image, so setting the backend var is enough — the
model paths auto-resolve to `models/table/...` / `models/formula/<engine>/` and
the `*_ONNX` overrides below are only needed for a non-default location.

| Variable | Default | Description |
|---|---|---|
| `TABLE_BACKEND` | *(unset)* | `slanext` enables SLANet-Plus table → HTML; auto-resolves the baked encoder. (`vlm` routes to a VL endpoint.) |
| `TABLE_SLANEXT_ENCODER_ONNX` | `models/table/slanext_encoder/SLANeXt_wired_encoder.onnx` | Override the table encoder path; decoder `.bin` + dict are derived next to it. |
| `FORMULA_BACKEND` | *(unset)* | `ppformulanet_s` (English/Latin, default engine), `ppformulanet_plus_m` (Chinese-capable, GPU only), or `auto` (GPU only; runs -S then re-runs plus-M on CJK-context crops — EN pages keep -S speed, CJK pages get plus-M accuracy) enables formula → LaTeX; auto-resolves the baked weights. |
| `FORMULA_ONNX` | `models/formula/<engine>` | Override the formula model dir/file. Only needed for a non-baked location. |
| `FORMULA_TOKENIZER` | `models/formula/<engine>/tokenizer.json` | Override the formula tokenizer path. |

!!! note "CPU build"
    On the CPU build, `FORMULA_BACKEND` selects only `ppformulanet_s` (plus-M is
    GPU only). `TABLE_BACKEND=slanext` / `FORMULA_BACKEND=ppformulanet_s` both
    auto-resolve the same baked paths as the GPU build.

## PDF

| Variable | Default | Description |
|---|---|---|
| `ENABLE_PDF_MODE` | `ocr` | Default PDF extraction mode: `ocr` / `geometric` / `auto` / `auto_verified`. CLI: `--default-pdf-mode`. |
| `MAX_PDF_PAGES` | `2000` | Max pages per `/ocr/pdf` request; over the limit → `400 PDF_TOO_LARGE`. Bounds `[1, 100000]`. CLI: `--max-pdf-pages`. |
| `MAX_PDF_PAGE_PIXELS_MP` | `40` | Max rendered megapixels per PDF page (decompression-bomb guard). Bounds `[1, 268]`. |
| `TURBO_PDF_IMAGE_ENCODER` | `gpu` | Inline-JPEG page-image encoder: `gpu` (nvJPEG) or `cpu` (libjpeg-turbo). GPU-only path; reported but inert on the CPU build. |
| `TURBO_PPM_SWAP` | `simd` | PPM channel-swap path: `simd` or `scalar`. |

## TensorRT / engine

| Variable | Default | Description |
|---|---|---|
| `TRT_OPT_LEVEL` | `5` | TensorRT builder optimization level. `0` = fastest build, `5` = fastest runtime (`3` builds ~3–5× faster with <5% runtime regression). Part of the engine cache key. Bounds `[0, 5]`. CLI: `--trt-opt-level`. |
| `TRT_ENGINE_CACHE` | `~/.cache/turbo-ocr` | Directory for cached TensorRT engines (empty value resolves to the default). Mount it to share engines across restarts. CLI: `--trt-engine-cache`. |
| `TRT_DET_WORKSPACE_GB` | `4` | Ceiling (GiB, `[1, 24]`) for the detection engine's TensorRT build workspace. The 4 GiB default fits 16 GB cards, but the `medium` detector at `DET_MAX_SIDE_LIMIT=2560` needs ~4.1 GiB — on cards with headroom set `8` or the build fails with "Could not find any implementation". Out-of-range values warn and keep the default. |
| `TURBO_OCR_CUDA_GRAPHS` | `1` (on) | Bake CUDA graphs for the recognition batch shapes at warmup. **Default changed to ON in v3.1.0**: +10–16% throughput and lower p50 latency (recognition is launch-bound), identical accuracy, at ~0.5 GiB extra VRAM per pipeline. Set `0` to opt out on VRAM-constrained cards (or lower `PIPELINE_POOL_SIZE`). |

## Performance / threading

| Variable | Default | Description |
|---|---|---|
| `PIPELINE_POOL_SIZE` | auto | Concurrent GPU pipelines (~1.4 GB VRAM each). Unset → GPU auto-detects from VRAM, CPU uses 4. Bounds `[1, 4096]`. CLI: `--pool-size` (`0` = auto). |
| `HTTP_THREADS` | `max(pool*32, 128)` | Work-pool threads for blocking inference (GPU consumer only). Bounds `[1, 4096]`. CLI: `--http-threads` (`0` = auto). |
| `PDF_DAEMONS` | `16` (CPU: `4`) | PDF render daemon processes. Bounds `[1, 1024]`. CLI: `--pdf-daemons`. |
| `PDF_WORKERS` | `4` (CPU: `2`) | PDF render workers. Bounds `[1, 1024]`. Exceeding `PDF_DAEMONS` warns (excess idle). CLI: `--pdf-workers`. |
| `GRPC_CQS` | `10` | gRPC completion-queue count. Bounds `[1, 1024]`. CLI: `--grpc-cqs`. |
| `GRPC_BATCH_WORKERS` | `8` | Parallel workers in gRPC `RecognizeBatch`. Bounds `[1, 256]`. CLI: `--grpc-batch-workers`. |

## Request lifecycle

| Variable | Default | Description |
|---|---|---|
| `REQUEST_TIMEOUT_MS` | `60000` | Per-request inference deadline (ms). On overrun a single-image / batch / gRPC request returns `504 INFERENCE_TIMEOUT` and frees its GPU slot; PDF jobs bound their per-page join by the same value (scaled by page count). `0` = **disabled** (unbounded wait — the pre-v3 behaviour). Bounds `[0, 3600000]`. CLI: `--request-timeout-ms`. |
| `PIPELINE_HARD_KILL_MS` | `600000` | Hard-kill margin (ms) for the dispatcher watchdog. After the deadline+grace trips and a recycle is requested, if the worker stays wedged mid-CUDA this long the process `_Exit`s so an orchestrator restarts it. Inert when `REQUEST_TIMEOUT_MS=0` (the watchdog only scans once a deadline is set). |

!!! note "Watchdog is on by default"
    Because `REQUEST_TIMEOUT_MS` now defaults to `60000` (was `0`), the
    per-request 504 path **and** the hard-kill watchdog are active out of the
    box. Set `REQUEST_TIMEOUT_MS=0` to restore the pre-v3 unbounded-wait
    behaviour (which also disables the watchdog).

## Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_BODY_MB` | `100` | Max request body (MB), enforced at nginx, Drogon, and gRPC. Bounds `[1, 102400]`. CLI: `--max-body-mb`. |
| `MAX_BODY_MEMORY_MB` | `1024` | In-memory body buffer cap (MB); always clamped to `MAX_BODY_MB`, so the effective default is `min(1024, MAX_BODY_MB)`. Raising it above the body cap warns. Bounds `[1, 102400]`. CLI: `--max-body-memory-mb`. |
| `MAX_BATCH_IMAGES` | `1024` | Max images per `/ocr/batch` (HTTP + gRPC `RecognizeBatch`); over the limit → `400 BATCH_TOO_LARGE`. Bounds `[1, 1000000]`. CLI: `--max-batch-images`. |
| `MAX_IMAGE_DIM` | `16384` | Max image width/height (px) accepted on decode routes (`/ocr/pixels`, etc.). Bounds `[64, 65535]`. CLI: `--max-image-dim`. |

## Server / network

| Variable | Default | Description |
|---|---|---|
| `TURBO_OCR_HOST` | `0.0.0.0` | Bind address for HTTP and gRPC. `127.0.0.1` = loopback only; `::` = IPv6. CLI: `--host`. |
| `PORT` | `8080` | HTTP backend port. In Docker, nginx fronts the binary on `8000` and proxies to this port. Bounds `[1, 65535]`. CLI: `--http-port`. |
| `GRPC_PORT` | `50051` | gRPC bind port. Must differ from `PORT` (fatal otherwise). Bounds `[1, 65535]`. CLI: `--grpc-port`. |
| `GRPC_RESPONSE_MODE` | `json_bytes` | gRPC response format: `json_bytes` or `structured`. CLI: `--grpc-response-mode`. |
| `SHUTDOWN_GRACE_SECONDS` | `30` | Drain time for inflight requests on SIGTERM/SIGINT before teardown. Bounds `[0, 600]`. CLI: `--shutdown-grace`. |
| `DISABLE_ANGLE_CLS` | `0` | `1` skips the angle classifier (~0.4 ms savings). CLI: `--disable-angle-cls`. |

## Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `info` | Log level: `debug` / `info` / `warn` / `error`. CLI: `--log-level`. |
| `LOG_FORMAT` | `json` | Log output format: `json` / `text`. CLI: `--log-format`. |

At startup the server emits one structured INFO line (`Effective server
config`) containing every resolved value — a single grep target for
post-mortems. Recorded warnings are logged immediately after.


## Expert / subsystem knobs

These are read directly by their subsystem (not via `ServerConfig`, so they
do not appear in `--print-config`). Defaults are tuned; override only with a
measured reason.

### Recognition / detection tuning

| Variable | Default | Description |
|---|---|---|
| `REC_BATCH_N` | `32` | Recognition batch size per inference call. |
| `REC_BUCKET_STEP` | `16` | CPU recognizer: snap crop widths UP to this step so batches pad each crop by at most step-1 columns. |
| `REC_ZEROCOPY` | `1` | CPU recognizer: zero-copy batch view into ORT (`0` = copy path). |
| `REC_SELFTEST` | `0` | CPU recognizer: one-shot batch-consistency self-test on first batch. |
| `SIMD_CTC` | `1` | SIMD CTC argmax decode (`0` = scalar fallback). |
| `DET_OPT_BATCH` | `8` | Batch dimension the det TRT profile is optimized for. |
| `TURBO_DET_FUSED_PRE` | `1` | Fused GPU resize+normalize det preprocess (`0` = OpenCV path). |
| `GPU_CCL` | `1` | Det post-process: `2` all-GPU JFA, `1` GPU CCL + CPU contours, `0` CPU contours. |
| `GPU_BOX_THRESH` | model default | Override DB box threshold on the GPU path. |
| `GPU_UNCLIP_SCALE` | `1.0` | Multiplier on the DB unclip ratio (GPU path). |
| `CLS_BATCH` | `32` | Angle-classifier batch size. |
| `MAX_IMAGE_PIXELS_MP` | `128` | Decompression-bomb cap: max decoded image area in megapixels. |
| `MAX_BATCH_PIXELS_MP` | `2048` | Aggregate pixel cap across one /ocr/batch request. |

### ONNX Runtime (CPU / formula backends)

| Variable | Default | Description |
|---|---|---|
| `ORT_EP` | `cpu` | Execution provider for the CPU engine (`cpu` / `coreml`). |
| `ORT_NUM_THREADS` | auto | Intra-op thread count per ORT session. |
| `ORT_GLOBAL_THREADS` | auto | Shared global thread-pool size (with `ORT_SHARED_POOL=1`). |
| `ORT_SHARED_POOL` | `1` | One shared ORT thread pool across sessions instead of per-session pools. |
| `ORT_REC_OPT_CAP` | unset | Cap ORT graph-optimization level for the recognizer. |
| `DISABLE_COREML` / `COREML_FLAGS` | unset | macOS CoreML EP opt-out / flags. |

### Structure stages (tables / formulas / VLM sidecar)

| Variable | Default | Description |
|---|---|---|
| `TABLE_CROP_MODE` | `layout` | `detunion` snaps each table region to the tight AABB of its det boxes. |
| `TABLE_CROP_MARGIN` | `0.03` | Fractional expansion per table-region side before structure decode. |
| `TABLE_MATCH_INTER` | `1` | Cell matcher: intersection-based OCR-fragment assignment. |
| `TABLE_MATCH_FALLBACK` | `1` | Cell matcher: nearest-cell fallback for unmatched fragments. |
| `TABLE_CLS_TRT`, `TABLE_SLANEXT_DICT`, `TABLE_SLANEXT_DECODER_BIN`, `TABLE_SLANEXT_WIRELESS_ENCODER_ONNX`, `TURBO_OCR_TABLE_DICT_PATH` | bundled paths | Override individual SLANeXt model/dict file locations. |
| `PPFNS_CHUNK` | `0` | PP-FormulaNet-S decode chunk size (0 = single pass). |
| `PPFNS_DROP_COLLAPSE` | `1` | Guard that drops collapsed (repeating) formula decodes. |
| `VLM_BACKEND` | `pool` | `legacy` selects the per-request curl path instead of the shared async pool. |
| `VLM_GLOBAL_CONCURRENCY` | `50` | Max in-flight VLM crop requests across the whole process. |
| `VLM_MAX_RETRIES` | `2` | Retries per VLM crop on transient transport failures. |
| `VLM_PNG_THREADS` | `4` | Threads PNG-encoding crops before VLM submit. |
| `VLLM_BASE_URL` / `VLLM_MODEL` | `http://localhost:8000` / `PaddleOCR-VL-1.6-0.9B` | VLM sidecar endpoint and model id. |
| `VLLM_FORMULA_PROMPT` / `VLLM_FORMULA_BATCH` / `VLLM_FORMULA_TIMEOUT_S` / `VLLM_FORMULA_MAX_TOKENS` | `Formula Recognition:` / `8` / `30` / `512` | Formula sidecar request shape. |
| `VLLM_TABLE_BASE_URL` / `VLLM_TABLE_MODEL` / `VLLM_TABLE_PROMPT` / `VLLM_TABLE_BATCH` / `VLLM_TABLE_TIMEOUT_S` / `VLLM_TABLE_MAX_TOKENS` | formula equivalents / `Table Recognition:` / `8` / `60` / `4096` | Table sidecar request shape (falls back to the `VLLM_*` values). |
| `TURBO_ROUTING_CONFIG` | env-synthesized | Path to a routing table JSON replacing the env-derived backend routing. |
| `TURBO_ALLOW_ADHOC_BACKENDS` | `0` | Allow per-request backends outside the routing table. |

### Server / PDF / misc

| Variable | Default | Description |
|---|---|---|
| `BIND_HOST` | `0.0.0.0` | Bind address override. |
| `GRPC_BATCH_GLOBAL_WORKERS` | `16` | Process-wide ceiling on extra gRPC batch fanout threads (each RPC keeps one guaranteed worker). |
| `BATCH_FANOUT_GLOBAL_WORKERS` | `64` | Process-wide ceiling on extra CPU `/ocr/batch` fanout threads (each request keeps one guaranteed worker). |
| `FINALIZE_DEFERRED_TIMEOUT_MS` | request timeout | Await budget for deferred (async VLM) structure results. |
| `PDF_RENDER_REPLY_TIMEOUT_MS` | `120000` | Cap on waiting for a PDF daemon reply. |
| `FASTPDF2PNG_PATH` | bundled | Path to the fastpdf2png daemon binary. |
| `NVJPEG_DEVICE_COPY` | `1` | nvJPEG page-image encode keeps data device-side. |
| `LAYOUT_KEEP_NESTED_CHILDREN` | `0` | Keep child layout blocks nested inside their parents. |
| `TURBO_LAYOUT_DEBUG` | `0` | Verbose layout-stage debug output. |
| `TURBO_OCR_STRICT_QUERY_PARAMS` | `1` | Reject unknown query parameters with 400 (set `0` to ignore them). |
| `TURBO_OCR_DISABLE_MALLOC_REAPER` | `0` | Disable the periodic malloc_trim reaper thread. |
| `ENABLE_TIMING` / `PROFILE_STAGES` | `0` | Per-stage timing output / CPU-path stage profiler. |
| `TOCR_LOG_RATELIMIT` | `10:1000` | Per-call-site log rate limit `N[:WINDOW_MS]`; `0` disables. |

!!! info "See also"
    - [Build → Docker](docker.md) — image env vars and the nginx front
      (`TURBO_OCR_PORT`, `MAX_BODY_MB`).
    - [Build → Native](native.md) — `LD_LIBRARY_PATH` and first-start
      engine build.
    - [API → HTTP](../api/http.md) — per-request query parameters.
