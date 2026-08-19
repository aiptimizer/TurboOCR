# Configuration

!!! abstract "TL;DR"
    Every runtime knob is an **environment variable**, and most are also a
    **CLI flag** that overrides the env value. Configuration is loaded once
    at startup, strict-validated, and the server **refuses to bind** on any
    malformed or out-of-range value. Inspect the resolved config without
    booting the pipeline with `--print-config` or `--check-config`.

Configuration is parsed by `ServerConfig::load_or_die`
(`include/turbo_ocr/service/server/bootstrap/server_config.h`) in two passes:
environment variables first, then CLI flags (flags win). Both override the
built-in defaults. The same call site serves both servers — the GPU
(CUDA/TensorRT) build and the unified multi-backend build, each of which
produces a binary named `turboocr-server`; a handful of defaults differ by
profile and are called out below.

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

Explicit per-stage paths win independently over `OCR_MODEL`. Both spellings
(`*_ONNX` and `*_MODEL`) are accepted on **every** build, so a config carries
unchanged between the NVIDIA image and an Intel/Apple/CPU deployment. `*_ONNX`
is canonical and wins if both are set to different values (with a warning).

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
| `FORMULA_BACKEND` | *(unset)* | `ppformulanet_s` (PP-FormulaNet_plus-S weights; English/Latin default engine — alias `ppformulanet_plus_s`), `ppformulanet_plus_m` (Chinese-capable, GPU only), or `auto` (GPU only; runs plus-S then re-runs plus-M on CJK-context crops — EN pages keep plus-S speed, CJK pages get plus-M accuracy) enables formula → LaTeX; auto-resolves the baked weights. |
| `FORMULA_CROP_PAD` | `4` | Pixels of page context added around every formula layout box before recognition (clamped 0–64). PP-FormulaNet is margin-sensitive: tight crops scramble its decoder, over-padding pulls in neighbor glyphs — 4 is the measured optimum. Reported boxes are unchanged. |
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
| `PDF_DEFAULT_DPI` | `100` | Render DPI when a request omits `?dpi=` / `OCRPDFRequest.dpi`. Applies to `/ocr/pdf`, `/ocr/stream` and `RecognizePDF` alike. Bounds `[50, 600]` — the same range a per-request `?dpi=` is validated against, so a default can never be set to a value a request could not ask for. CLI: `--pdf-default-dpi`. |
| `MAX_PDF_PAGE_PIXELS_MP` | `40` | Max rendered megapixels per PDF page (decompression-bomb guard). Bounds `[1, 268]`. |
| `TURBO_PDF_IMAGE_ENCODER` | `gpu` | Inline-JPEG page-image encoder: `gpu` (nvJPEG) or `cpu` (libjpeg-turbo). GPU-only path; reported but inert on the CPU build. |
| `TURBO_PDF_PAGE_WORKERS` | `3` | Rendered pages OCR'd concurrently per PDF job; each in-flight page takes its own pipeline-pool lease, so values beyond the pool size only add idle threads. `1` restores strictly sequential pages. Measured 2.2x on an 8-page document at 3 replicas. |
| `TURBO_PPM_SWAP` | `simd` | PPM channel-swap path: `simd` or `scalar`. |

## TensorRT / engine

| Variable | Default | Description |
|---|---|---|
| `TRT_OPT_LEVEL` | `5` | TensorRT builder optimization level. `0` = fastest build, `5` = fastest runtime (`3` builds ~3–5× faster with <5% runtime regression). Part of the engine cache key. Bounds `[0, 5]`. CLI: `--trt-opt-level`. |
| `TRT_FP16` | `1` | `0` builds fp32 engines instead of the default weakly-typed fp16 — an escape hatch for graphs whose fp16 compilation fails on a given TRT version, not a tuning knob (~2× runtime on the affected engine). Part of the engine cache key. |
| `TRT_ENGINE_CACHE` | `~/.cache/turbo-ocr` | Directory for cached TensorRT engines (empty value resolves to the default). Mount it to share engines across restarts. CLI: `--trt-engine-cache`. |
| `MIGRAPHX_ENGINE_CACHE` | `~/.cache/turbo-ocr` | AMD backend: directory for cached compiled MIGraphX programs (`.mxr`), keyed by model + shape + gfx arch + ROCm version. The first start pays ~42 graph compiles; cached starts load them from disk. `off` disables caching. |
| `TRT_DET_WORKSPACE_GB` | `4` | Ceiling (GiB, `[1, 24]`) for the detection engine's TensorRT build workspace. The 4 GiB default fits 16 GB cards, but the `medium` detector at `DET_MAX_SIDE_LIMIT=2560` needs ~4.1 GiB — on cards with headroom set `8` or the build fails with "Could not find any implementation". Out-of-range values warn and keep the default. |
| `TURBO_OCR_CUDA_GRAPHS` | `1` (on) | Bake CUDA graphs for the recognition batch shapes at warmup. **Default changed to ON in v3.1.0**: +10–16% throughput and lower p50 latency (recognition is launch-bound), identical accuracy, at ~0.5 GiB extra VRAM per pipeline. Set `0` to opt out on VRAM-constrained cards (or lower `PIPELINE_POOL_SIZE`). |

## Performance / threading

| Variable | Default | Description |
|---|---|---|
| `HTTP_THREADS` | `clamp(pool*4, 16, 64)` | Work-pool threads for blocking host work (decode, JSON, PDF joins) in front of the replica pool — measured flat from 20–48 threads on a 5090, so the rule targets that region (the old pool*32 rule put 160 threads on a 20-core box). Bounds `[1, 4096]`. CLI: `--http-threads` (`0` = auto). |
| `WORK_QUEUE_DEPTH` | `8192` | Work-pool QUEUE depth — the admission bound in front of the whole server. When full, requests are rejected with `503 SERVER_BUSY` rather than queued unboundedly. `0` keeps the default. Note this is the queue, not `HTTP_THREADS` (the worker count). Bounds `[0, 1048576]`. CLI: `--work-queue-depth`. **Size it to a latency budget: `capacity_req_s × max_acceptable_latency_s`.** At the default, an overloaded server never sheds in practice — the queue fills at (offered − capacity) req/s, so a 10 req/s excess needs ~14 min to reach 8192 while `REQUEST_TIMEOUT_MS` (60 s) fires first; latency just climbs toward the timeout. Measured on an RTX 5090 at 90/s offered vs ~80/s capacity: depth 8192 → p50 2.3 s and rising, all 200s; depth 64 → 91% served at p99 1.0 s (= 64/80), 9% shed as sub-ms 503s, zero false 503s below capacity. See `tests/e2e/load_openloop.sh`. |
| `PDF_DAEMONS` | `16` (CPU: `4`) | PDF render daemon processes. Bounds `[1, 1024]`. CLI: `--pdf-daemons`. |
| `PDF_WORKERS` | `4` (CPU: `2`) | PDF render workers. Bounds `[1, 1024]`. Exceeding `PDF_DAEMONS` warns (excess idle). CLI: `--pdf-workers`. |
| `GRPC_CQS` | `10` | gRPC completion-queue count. Bounds `[1, 1024]`. CLI: `--grpc-cqs`. |
| `GRPC_BATCH_WORKERS` | `8` | Parallel workers in gRPC `RecognizeBatch`. Bounds `[1, 256]`. CLI: `--grpc-batch-workers`. |

## Request lifecycle

| Variable | Default | Description |
|---|---|---|
| `REQUEST_TIMEOUT_MS` | `60000` | How long a request may wait for a pipeline replica before `504` (a QUEUEING deadline — a stage already running is never interrupted; see `make_infer_func.h`). PDF jobs bound their per-page join by the same value (scaled by page count). `0` = **disabled** (unbounded wait). Bounds `[0, 3600000]`. CLI: `--request-timeout-ms`. |

!!! note "What the deadline does and does not cover"
    The 504 path bounds pool ACQUISITION only. There is no hard-kill
    watchdog and no forced recycle of a wedged replica — the stuck-lease
    sweep on `/metrics` reports the condition; recovery is an orchestrator
    restart (`make_infer_func.h` documents why pretending to interrupt a
    running stage is worse than reporting it).

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
| `SHUTDOWN_GRACE_SECONDS` | `30` | Real drain bound on SIGTERM/SIGINT: in-flight requests get this long to finish; queued-but-unstarted work past it is shed (counted on `/metrics` as `workpool_discarded_tasks_total`). Bounds `[0, 600]`. CLI: `--shutdown-grace`. |
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
| `SIMD_CTC` | `0` | Opt-IN SIMD CTC argmax decode (set `1` to enable; default is the scalar path). |
| `DET_OPT_BATCH` | `8` | Batch dimension the det TRT profile is optimized for. |
| `TURBO_DET_FUSED_PRE` | `0` | Opt-IN fused single-pass resize+normalize in the CPU/ORT detector (set `1`; default is the OpenCV two-step). |
| `GPU_CCL` | `1` | Det post-process: `2` all-GPU JFA, `1` GPU CCL + CPU contours, `0` CPU contours. |
| `GPU_BOX_THRESH` | model default | Override DB box threshold on the GPU path. |
| `GPU_UNCLIP_SCALE` | `1.0` | Multiplier on the DB unclip ratio (GPU path). |
| `CLS_BATCH` | `0` | Opt-IN batched angle classification (boolean; set `1` to run cls in batches instead of per-crop). |
| `MAX_IMAGE_PIXELS_MP` | `128` | Decompression-bomb cap: max decoded image area in megapixels. |
| `MAX_BATCH_PIXELS_MP` | `2048` | Aggregate pixel cap across one /ocr/batch request. |

### ONNX Runtime (CPU / formula backends)

| Variable | Default | Description |
|---|---|---|
| `ORT_EP` | `cpu` | Execution provider for the CPU engine. Recognized: `cpu`, `coreml`, `xnnpack`, `dnnl`, `openvino`, `migraphx`, `rocm`, `dml`, `cuda`. An unrecognized value is **not** passed through to ORT — it fails engine load with `Unknown ORT_EP='…'`. A recognized provider this onnxruntime build does not ship fails the same way: `apply_execution_provider()` checks `Ort::GetAvailableProviders()` **before** appending and refuses, naming the EP, the providers this build does have, and the build you need. See [GPU providers fail loudly](#gpu-providers-fail-loudly) for why that check exists and what it does not cover. |
| `ORT_NUM_THREADS` | per-stage cap | Intra-op threads per ORT session. Top rung of the shared host-thread policy (`include/turbo_ocr/onnx/host_ort_threads.h`), so it overrides both a backend's "my host is idle" hint and each stage's own built-in cap. Under `ORT_EP=xnnpack` it sizes XNNPACK's own pool instead, where the default is all cores. |
| `ORT_GLOBAL_THREADS` | all cores | Size of the shared global intra-op pool. Read only when `ORT_SHARED_POOL=1`. |
| `TURBO_STRICT_EP` | `auto` | Controls ORT's `disable_cpu_ep_fallback` guard. `auto` enables it only for providers verified to claim a whole graph (`cuda`); `1` requires it on every provider; `0` disables it everywhere. See [Which providers it applies to](#which-providers-it-applies-to). |
| `ORT_SHARED_POOL` | `0` | `1` draws every session's threads from one process-wide intra-op pool instead of a per-session one. Fixed at first-session time, so it is a process-wide decision. Forced off under `ORT_EP=xnnpack`, which manages its own threads. |
| `ORT_REC_OPT_CAP` | `1` | Caps the *recognizer*'s ORT graph optimization at EXTENDED: v6 rec fp16 decodes wrong under `ORT_ENABLE_ALL` on ORT 1.26 (SimplifiedLayerNormFusion). `0` lifts the cap on a fixed ORT without a recompile. |
| `DISABLE_COREML` | unset | macOS: `1` keeps the CoreML EP off. CoreML is on by default only for engines configured from the environment — a backend that passes an explicit EP config gets it only when that config asks for provider `coreml`, so `--backend cpu` on a Mac is plain MLAS either way. |
| `COREML_FLAGS` | `0x020` | CoreML provider flags. `0x020` is CPU + GPU, which includes the Neural Engine where present. |
| `OPENVINO_DEVICE` | `CPU` | OpenVINO target device (`CPU` / `GPU` / `NPU`). **`GPU` measurably degrades detection accuracy on these models** — the engine warns at startup. Leave on `CPU` unless you have measured otherwise. |
| `OPENVINO_PRECISION` | unset | OpenVINO inference precision hint (`FP16` / `FP32`). Only `GPU*`/`NPU*` accept `FP16`; on `CPU` it takes the whole provider down and surfaces as "openvino unavailable", so the engine warns before appending it. |
| `OPENVINO_CACHE_DIR` | unset | Directory for OpenVINO's compiled-blob cache (big first-run win, no effect on results). |
| `OPENVINO_EP_OPTS` | unset | Extra `key=value,key=value` options passed straight to the OpenVINO EP. Merged over the three keys above, so it wins — and merged *before* the device/precision guardrails, so a `device_type` set here is checked like any other. |
| `CUDA_DEVICE_ID` | `0` | GPU ordinal for the CUDA EP. |
| `ROCM_DEVICE_ID` | `0` | GPU ordinal for the ROCm / MIGraphX EPs. |
| `DML_DEVICE_ID` | `0` | Adapter ordinal for the DirectML EP. |

The three `*_DEVICE_ID` variables are the lowest-priority rung: a device named
through the vendor seam (`TURBO_EP_DEVICE`, or the Python `device=`) wins over
them, so one knob selects the device on every vendor arm.

### Vendor seam backends (the `--backend` engines, not ORT EPs)

| Variable | Default | Description |
|---|---|---|
| `TURBO_EP_PROVIDER` | vendor default | Override the ORT execution provider a vendor's ONNX path uses (see `src/backends/cpu/stages/cpu_stages.cpp`). |
| `TURBO_EP_DEVICE` | unset | Device selector handed to that engine (e.g. the OpenVINO device string). This is the name the engine reads — `TURBO_DEVICE` is not read by anything. |
| `TURBO_EP_FP16` | `1` | fp16 weights/activations where the engine supports it; set `0` to force fp32. |
| `TURBO_ENGINE_MODE` | auto | `native` / `onnx` / `auto` — which engine path a vendor backend brings up (also `--engine-mode`). |

### Structure stages (tables / formulas / VLM sidecar)

| Variable | Default | Description |
|---|---|---|
| `TABLE_CROP_MODE` | `layout` | `detunion` snaps each table region to the tight AABB of its det boxes. |
| `TABLE_CROP_MARGIN` | `0.03` | Fractional expansion per table-region side before structure decode. |
| `TABLE_MATCH_INTER` | `1` | Cell matcher: intersection-based OCR-fragment assignment. |
| `TABLE_MATCH_FALLBACK` | `0.15` | Cell matcher: overlap FLOOR (float) for rescuing an unmatched fragment into its argmax cell; `0` disables the rescue. |
| `TABLE_CLS_TRT`, `TABLE_SLANEXT_DICT`, `TABLE_SLANEXT_DECODER_BIN`, `TABLE_SLANEXT_WIRELESS_ENCODER_ONNX`, `TURBO_OCR_TABLE_DICT_PATH` | bundled paths | Override individual SLANeXt model/dict file locations. |
| `PPFNS_CHUNK` | `8` | PP-FormulaNet decode chunk size (bounds `[1, 32]`). |
| `PPFNS_DROP_COLLAPSE` | unset | Opt-IN guard (presence-checked) that drops collapsed (repeating) formula decodes. |
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
| `TURBO_OCR_STRICT_QUERY_PARAMS` | `0` | Opt-in: set `1` to reject with 400 any unknown parameter AND any known parameter the endpoint does not support. The default (including in this v4.0.0-alpha) tolerates both and marks the request with `X-Ignored-Params` + `X-Deprecation` response headers; a future release may flip the default to strict. Routing overrides, `text=0`, and `embed=0` on endpoints that cannot honor them are ALWAYS a 400 — ignoring those would falsify the response. |
| `TURBO_OCR_DISABLE_MALLOC_REAPER` | `0` | Disable the periodic malloc_trim reaper thread. |
| `ENABLE_TIMING` / `PROFILE_STAGES` | `0` | Per-stage timing output / CPU-path stage profiler. |
| `TOCR_LOG_RATELIMIT` | `10:1000` | Per-call-site log rate limit `N[:WINDOW_MS]`; `0` disables. |

!!! info "See also"
    - [Build → Docker](../getting-started/docker.md) — image env vars and the nginx front
      (`PORT`, `MAX_BODY_MB`).
    - [Build → Native](../getting-started/native.md) — `LD_LIBRARY_PATH` and first-start
      engine build.
    - [API → HTTP](http.md) — per-request query parameters.

## Backend selection

| Variable | Default | Meaning |
|---|---|---|
| `TURBO_BACKEND` | *(empty = auto-detect)* | Which vendor backend to run: `cpu`, `apple`, `nvidia`, `intel`, `amd`. Also `--backend`. Empty auto-detects by priority among the backends compiled in (`TURBO_BACKENDS` at build time). **This is the single most important operator knob and was previously undocumented.** |
| `OCR_BACKEND` | *(unset)* | Legacy alias retained for compatibility; prefer `TURBO_BACKEND`. |
| `TURBO_ENGINE_MODE` | `auto` | `native` / `onnx` / `auto` — which engine path a vendor brings up. Also `--engine-mode`. |

## Pool / admission control

Applies to every backend. The lease pool is what serialises on the device; the
`WorkPool` queue in front of it is a separate, larger bound.

| Variable | Default | Meaning |
|---|---|---|
| `PIPELINE_POOL_SIZE` | auto (VRAM-derived) | Number of pipeline replicas. Also `--pool-size` (`0` = auto); bounds `[1, 4096]`; CPU build defaults to 4. Auto-sizing (`pipeline/pool_sizing.h`) tiers on total VRAM (cap 5) and then lowers to what FREE VRAM fits, using a measured 4.5 GiB per replica (worst shipping tier; `tiny` ≈ 2.9 GiB) **plus 4 GiB when the routed formula engine is `ppformulanet_plus_m` or `auto`** (their per-replica decode buffers). |
| `TURBO_POOL_SIZE` | *(unset)* | **Intel-backend-only** replica override, read by `intel_backend.cpp`. Two names for one concept — prefer `PIPELINE_POOL_SIZE`; this exists for historical reasons and is a known wart. |
| `TURBO_POOL_MAX_WAITERS` | `8 × pool size` | Requests allowed to queue for a replica before the pool sheds with `PoolExhaustedError` → HTTP 503. Prevents a saturated device from parking a `WorkPool` thread per request. |
| `TURBO_POOL_ACQUIRE_TIMEOUT_MS` | `30000` | Deadline for acquiring a replica; `0` restores the old unbounded wait. Covers **queueing only** — once leased, a wedged stage is unbounded, which is what `TURBO_POOL_STUCK_LEASE_MS` below makes visible. |
| `TURBO_POOL_STUCK_LEASE_MS` | `0` (off) | How long a replica may be leased before it is reported wedged: logs once per stuck lease and increments `turbo_ocr_pool_stuck_leases`. Set to a small multiple of your slowest legitimate request — too low turns a large PDF into a false alarm. **Detection only:** a stuck lease is usually stuck *inside* a device call, and destroying a pipeline whose work is still in flight would crash rather than recover, so the replica is not rebuilt. Evaluated on each `/metrics` scrape. |
| `TURBO_DET_BATCH` | *(unset)* | Force cross-request detection batch size. Batching is inert unless set — no backend advertises `preferred_batch_size` yet. |
| `TURBO_DET_BATCH_DELAY_US` | `0` | Coalescing window for the detection batcher. `0` is Triton's zero-delay default. |

## Apple (Metal / MPSGraph) — native only, no container

See the *Apple Silicon* section of the README for why there is no Docker image.

| Variable | Default | Meaning |
|---|---|---|
| `TURBO_APPLE_REC_BUCKETS` | *(auto-discover)* | **Effectively mandatory.** Comma-separated recognizer width ladder. Unset, `MpsRecognizer` discovers every `rec_b*` export on disk and builds a ~42-bucket ladder that roughly **halves throughput**. Use `320,480,800,1200,1600,2000,2500,3200,4000`. |
| `TURBO_APPLE_METALLIB` | *(next to the binary)* | Path to `turbo_apple.metallib`. CMake emits it beside the executable; only needed if relocated. |
| `TURBO_APPLE_REC_TIER` | *(from model path)* | `tiny`/`small`/`medium`. Pins the ANE package tier; a wrong tier decodes against the wrong dictionary, so it is DISABLED rather than guessed when undeterminable. |
| `TURBO_APPLE_REC_FP16` | on | fp16 recognizer weights. |
| `TURBO_APPLE_DET_ASYNC` | on | Detector `enqueue`/`collect` overlap (its own private command queue). |
| `TURBO_APPLE_DET_JIT` | on | Per-page detector specialization: the runtime compiles the det engine for each page's shape (shared resize policy, 128-px-grid snapped, one-time ~50–350 ms per new shape, then full speed). `0` pins detection to the exported canvas(es) instead. |
| `TURBO_APPLE_DET_CANVAS_CACHE` | `6` | Live det engine specializations kept (LRU, clamped 2–32). Bounds detector memory on shape-diverse corpora. |
| `TURBO_APPLE_DET_COREML` | off | Path to a fixed-canvas det `.mlpackage`; runs the det forward on CoreML(GPU) instead of MPSGraph (implies `TURBO_APPLE_DET_JIT=0` — the package is baked at one shape). Measured +4–6% at 3 replicas on FUNSD (CoreML's conv kernels beat MPSGraph's for DBNet); the resize/normalize kernel and DB post-process are unchanged. The package's input shape must match the det canvas or it falls back, loudly. |
| `TURBO_APPLE_ANE_WORKERS` | auto | ANE worker threads; `0` disables the ANE entirely. |
| `TURBO_APPLE_ANE_MAXW` | `800` | Widest bucket routed to the ANE; wider falls back to MPSGraph. |
| `TURBO_APPLE_ANE_TIMEOUT_MS` | — | Per-predict ANE timeout. |
| `TURBO_APPLE_ANE_SHAPE_IDX` | — | Pin the ANE input shape index. |
| `TURBO_APPLE_COREML_DIR` | `~/.apple_ocr_ml/coreml` | `rec_ane_<W>.mlpackage` location. |
| `TURBO_APPLE_GPU_TIMEOUT_MS` | `30000` | Timeline-event wait before a command buffer is declared lost. |
| `TURBO_APPLE_PROFILE` | off | Per-stage profiler (takes a process-global mutex — measures itself at high K). |
| `TURBO_APPLE_CONTENTION` | off | Lock-free contention counters, dumped at exit. |
| `TURBO_APPLE_PAGE_AUDIT` | off | Fingerprints page bytes on texture pack and re-checks on cache hit. Use to rule a texture-aliasing bug in or out. |
| `FFDETR_COREML` | off | Route the FFDetr form-field model through CoreML (measured **slower**). |
| `MPS_DEBUG`, `MPS_OUT` | off | MPSGraph debug dumps. |

## Intel (OpenVINO)

| Variable | Default | Meaning |
|---|---|---|
| `OV_DEVICE` | `GPU` (= `GPU.0`) | `CPU` / `GPU` / `NPU`, **optionally with a device index** (`GPU.1`). Unset, the Intel backend targets the iGPU/Arc (and does not register at all if OpenVINO cannot enumerate that device); an explicit value pins the device. **On a machine with both an iGPU and a discrete card, plain `GPU` means `GPU.0` — the integrated one.** OpenVINO enumerates the discrete card too (verified: Core Ultra 9 285K + RTX 5090 → `GPU.0` = Intel Xe, `GPU.1` = RTX 5090), so name the index explicitly if you want the other one. An unrecognized value logs an error and falls back to the default rather than failing silently. Distinct from `OPENVINO_DEVICE` above, which selects the device for the ORT OpenVINO *execution provider* path. **`NPU` is not production-ready**: the NPU plugin compiles only fully static shapes (our dynamically-exported det/rec fail outright), the **layout model is rejected outright** on op support even when static, and once reshaped det/rec measured *slower* than CPU and iGPU on every model but the medium detector — `src/backends/intel/SETUP.md` §0b. **NPU is also unreachable from WSL2** (no `/dev/accel` passthrough); it needs native Windows or bare-metal Linux. |
| `OV_PERF_HINT` | `latency` | OpenVINO performance hint (`throughput`/`latency`/`none`). Default flipped to `latency` 2026-08-03: the sync engine's one-at-a-time requests ran on a single stream under the throughput hint — 2.4 vs 5.5 img/s on a 13600K (see `src/backends/intel/SETUP.md`). |
| `OV_NUM_STREAMS`, `OV_NUM_REQUESTS` | — | Inference stream / request counts. |
| `OV_INFER_PRECISION` | — | e.g. `f16`. |
| `OV_CACHE_DIR` | — | Compiled-blob cache directory. |
| `OV_REC_DYNAMIC_BATCH` | — | Dynamic recognizer batch. |
| `OV_REC_MAX_PREBUILD_WIDTH` | — | Widest pre-built recognizer width. |
| `TURBO_INTEL_DEBUG` | off | Intel backend debug logging. |

## Miscellaneous

| Variable | Default | Meaning |
|---|---|---|
| `REC_IMAGE_H` | model-derived | Recognizer input height override. |
| `HOME` | — | Read for the default cache/export paths (`~/.cache/turbo-ocr`, `~/.apple_ocr_ml`). Not a knob; listed so the dependency is visible. |

## Per-stage device placement

A machine with more than one accelerator is now the normal case — an Intel Core
Ultra has a CPU, an iGPU **and** an NPU, and a desktop adds a discrete GPU. One
`OV_DEVICE` for the whole pipeline leaves the rest idle.

Each stage that owns an inference engine can be pinned independently:

| Variable | Stage |
|---|---|
| `DET_DEVICE` | detection |
| `REC_DEVICE` | recognition |
| `CLS_DEVICE` | angle classification |
| `LAYOUT_DEVICE` | layout |

Values are whatever the backend understands (`CPU` / `GPU` / `NPU` for
OpenVINO). Precedence: `<STAGE>_DEVICE` → the backend's global (`OV_DEVICE`) →
the backend default. **With none set, nothing changes** — every stage resolves
exactly as before. An unparseable or unavailable value logs a warning and falls
back to the backend device rather than failing the server.

### Why this is not just "pick the fastest device"

The interesting win is **overlap**, not per-stage speed. Stages are pipelined
across pages, so end-to-end throughput is bounded by the slowest *device*, not
by the sum of stage latencies. Measured on a Core Ultra 9 285K
(det_tiny @ 640×640, rec_tiny @ batch 8×48×320, latency hint):

| device | det | rec |
|---|---|---|
| CPU | 4.75 ms | 6.65 ms |
| iGPU | 5.03 ms | 4.95 ms |
| NPU | 13.79 ms | 20.51 ms |

Running det and rec **concurrently**, per-page wall time:

| placement | ms/page | |
|---|---|---|
| `DET_DEVICE=CPU REC_DEVICE=CPU` | 10.97 | both contend on one device ≈ the sum |
| `DET_DEVICE=CPU REC_DEVICE=NPU` | 12.31 | worse — rec is the heavier stage |
| **`DET_DEVICE=NPU REC_DEVICE=CPU`** | **9.64** | **faster than all-CPU, despite the NPU being 2.9× slower at det** |
| **`DET_DEVICE=GPU REC_DEVICE=CPU`** | **8.32** | best on this machine — 24% better than all-CPU |

The NPU row is the point. In isolation it is the *worst* device for every
stage — but moving detection onto it frees the CPU for recognition, and the
pipeline gets faster. A device being slower is not a reason to leave it idle;
what matters is which device the pipeline is *waiting* on.

Pick the split by measuring, not by device reputation: the best placement
depends on the relative cost of your stages, which changes with model tier and
page size.

## GPU providers fail loudly

A GPU request that quietly runs on the CPU is the worst outcome this engine can
produce: throughput collapses by an order of magnitude, every result is still
*correct*, and nothing in the logs says why. There is no metric that
distinguishes it from "the GPU is slow today".

So `ORT_EP` is checked against `Ort::GetAvailableProviders()` **before** the
provider is appended (`src/onnx/ort_engine.cpp`, `apply_execution_provider()`).
A provider this onnxruntime does not ship is refused:

```
[OrtEngine] ORT_EP='cuda' requires CUDAExecutionProvider, which this onnxruntime
does not provide (available: CoreMLExecutionProvider, WebGpuExecutionProvider,
CPUExecutionProvider). Refusing rather than running on the CPU under a GPU
request — install the matching onnxruntime build, or set ORT_EP=cpu to ask for
CPU explicitly.
```

`load()` returns false; the stage never comes up. To ask for CPU, ask for it:
`ORT_EP=cpu`.

### How the guarantee works

Two mechanisms, and only one of them is the guarantee:

1. **`session.disable_cpu_ep_fallback`** (ORT's own
   `kOrtSessionOptionsDisableCPUEPFallback`) makes node assignment to the CPU EP
   an **error** instead of a silent degradation, so a provider that appends
   cleanly and then cannot claim the graph fails at session construction.
   **This is the guarantee** — on the providers it applies to; see
   [Which providers it applies to](#which-providers-it-applies-to).
2. The `GetAvailableProviders()` pre-check is only **error quality** — without
   it a missing provider surfaces as ORT's generic append failure; with it you
   are told which provider was needed and which ones this build has.

The distinction matters because the second one is not sufficient on its own.
Measured on a CUDA-13 host with a CUDA-12 `onnxruntime-gpu` wheel — the provider
is reported as available, so no pre-check can catch it:

```
available: True

--- WITHOUT disable_cpu_ep_fallback ---
providers: ['CPUExecutionProvider']          <- silently CPU

--- WITH disable_cpu_ep_fallback ---
REFUSED: FAIL : This session contains graph nodes that are assigned to the
default CPU EP, but fallback to CPU EP has been explicitly disabled by the user.
```

### Which providers it applies to

The flag fails session construction if **any** node lands on the CPU EP. Whether
that is the right bar depends on how the provider partitions, and they differ:

| provider | guard | why |
|---|---|---|
| `cuda` | **on** | Verified on hardware (RTX 5090 / CUDA 13.3, and the ORT CUDA EP on Windows). Claims these graphs whole, and this is the case the guarantee exists for. |
| `coreml` | off | Partitions by design — ORT reports *"partitions supported by CoreML: 30 … nodes supported: 46"* of 76 as **normal**. With the guard on, every CoreML session fails and the server will not start on any machine without exported MPSGraph artefacts. |
| `openvino`, `tensorrt`, `rocm`, `migraphx`, `dml` | off | Partition the same way CoreML does, and none has been run with the guard on real hardware from this tree. Defaulting it on would ship the CoreML breakage to Intel and AMD users unseen. |
| `xnnpack`, `dnnl` | off | CPU-side providers; *"must not touch the CPU EP"* is a contradiction. |

Where the guard is off, the engine says so once at load:

```
[OrtEngine] provider 'coreml' partitions its graph, so the CPU-fallback guard is
OFF (untested for this provider). Set TURBO_STRICT_EP=1 to require a full-graph
claim.
```

`TURBO_STRICT_EP` overrides the table: `1` requires a full-graph claim on any
provider, `0` disables the guard everywhere, `auto` (default) is the table above.
Turn it on once you have tested your own provider and model combination — on a
provider that partitions, expect the same refusal CoreML gives.

### CUDA / onnxruntime / cuDNN must agree

Three independent version axes, and a mismatch in any of them degrades silently
rather than failing. Verified working on an RTX 5090 (Blackwell, sm_120):

| component | version | note |
|---|---|---|
| NVIDIA driver | 596.36 | Blackwell needs a recent one |
| CUDA Toolkit | **13.3** | correct for Blackwell; 12.8+ is the floor |
| onnxruntime-gpu | 1.26.0 **cu13 build** | see below — the default wheel is CUDA 12 |
| cuDNN | **9.25** for cuda13 | ORT's CUDA EP hard-requires it |

Three traps, each of which cost a debugging cycle:

1. **`pip install onnxruntime-gpu` gives you the CUDA 12 build.** It reports
   `CUDAExecutionProvider` as available on a CUDA 13 host, then runs on the CPU.
   The CUDA 13 build comes from a separate index:
   ```
   pip install onnxruntime-gpu --index-url \
     https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/
   ```
   Tell them apart by the DLL it asks for: `cudart64_12.dll` vs `cudart64_13.dll`.

2. **CUDA 13 moved its DLLs to `bin\x64\`, not `bin\`.** Looking for
   `cudart64_13.dll` in `…\CUDA\v13.3\bin` finds nothing and reads as a broken
   install. It is one directory deeper.

3. **cuDNN is not part of the CUDA Toolkit and has no winget package.** Take it
   from NVIDIA's public redist — no login needed:
   ```
   https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/
   ```
   Pick the `_cuda13-archive.zip` matching your CUDA major version.

Both `bin\x64` directories (CUDA's and cuDNN's) must be on `PATH` — ORT's
`preload_dlls()` searches its own package directories, not `PATH`, so a
successful `preload_dlls()` is not evidence that the loader will find them.
