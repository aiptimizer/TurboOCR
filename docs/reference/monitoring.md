# Monitoring & Observability

!!! abstract "TL;DR"
    Scrape `GET /metrics` for Prometheus text-format metrics (zero external
    dependencies, served as `text/plain; version=0.0.4`). Every response
    carries `X-Request-Id` (UUID v7) and `X-Inference-Time-Ms`; 503s add
    `Retry-After`. Errors use the shared JSON envelope
    `{"error": {"code": ..., "message": ...}}`.

The metrics implementation lives in `include/turbo_ocr/service/server/metrics.h`
(`turbo_ocr::server::Metrics`); the response headers are injected by the
observability middleware in `include/turbo_ocr/service/server/server_types.h:187`
(`register_observability_middleware()`).

## `GET /metrics`

Prometheus text exposition (format version `0.0.4`). The handler refreshes
GPU VRAM via `cudaMemGetInfo()` on every scrape (cheap syscall) and is
registered by `register_metrics_route()`. The endpoint itself is excluded
from request accounting.

```bash
curl http://localhost:8000/metrics
```

### Exposed metrics

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `turbo_ocr_requests_total` | counter | `route`, `status` | HTTP requests by route and status class. |
| `turbo_ocr_request_duration_seconds_bucket` | histogram | `route`, `le` | Cumulative latency buckets (see boundaries below). |
| `turbo_ocr_request_duration_seconds_sum` | histogram | `route` | Total observed latency, seconds. |
| `turbo_ocr_request_duration_seconds_count` | histogram | `route` | Number of observations (== `+Inf` bucket). |
| `turbo_ocr_pipeline_pool_size` | gauge | — | Number of pipeline slots. |
| `turbo_ocr_pool_available` | gauge | — | Free pipeline replicas. |
| `turbo_ocr_pool_waiting` | gauge | — | Requests blocked waiting for a replica. |
| `turbo_ocr_pool_exhaustions_total` | counter | — | Times the pipeline pool was full and a request was rejected with `503 SERVER_BUSY`. |
| `turbo_ocr_pool_oldest_lease_ms` | gauge | — | Age of the longest-held pipeline lease — a value far above your slowest request means a wedged replica. |
| `turbo_ocr_pool_stuck_lease_threshold_ms` | gauge | — | `TURBO_POOL_STUCK_LEASE_MS`; `0` = stuck-lease detection disabled (the counter below is then meaningless, not reassuring). |
| `turbo_ocr_pool_stuck_leases` | gauge | — | Leases held past the threshold — each is a replica effectively lost until restart. |
| `turbo_ocr_dispatcher_queue_depth` | gauge | — | Requests queued for the GPU pipeline dispatcher. |
| `turbo_ocr_workpool_queue_depth` | gauge | — | Tasks waiting in the WorkPool queue. |
| `turbo_ocr_workpool_inflight` | gauge | — | Tasks currently executing on WorkPool threads. |
| `turbo_ocr_workpool_max_depth` | gauge | — | Configured WorkPool queue ceiling (`503` above it). |
| `turbo_ocr_workpool_escaped_tasks_total` | counter | — | Tasks still running when shutdown's grace expired. |
| `turbo_ocr_workpool_discarded_tasks_total` | counter | — | Queued-but-unstarted tasks shed at shutdown after `SHUTDOWN_GRACE_SECONDS`. |
| `turbo_ocr_gpu_vram_used_bytes` | gauge | — | Whole-device GPU memory in use — all processes, not just this server (GPU build only). |
| `turbo_ocr_gpu_vram_total_bytes` | gauge | — | Total GPU memory (GPU build only). |
| `turbo_ocr_request_bytes_total` | counter | — | Total request body bytes received. |
| `turbo_ocr_request_body_avg_bytes` | gauge | — | Mean request body size. |

!!! note "Label values"
    `route` is one of `/ocr`, `/ocr/raw`, `/ocr/batch`, `/ocr/pixels`,
    `/ocr/pdf`, `/ocr/markdown`, `/infer`, `/ocr/stream`, `/health`, `other`. `status` is bucketed by class —
    `2xx`, `4xx`, or `5xx`, **not** the exact code. The `/health` route is
    intentionally omitted from the latency histogram so probe traffic can't
    skew percentiles (`metrics.h`, the `kHealth` skip in `serialize()`).

!!! info "GPU-only gauges"
    `turbo_ocr_gpu_vram_used_bytes` and `turbo_ocr_gpu_vram_total_bytes` are
    emitted only when total VRAM is known (`> 0`), i.e. the GPU build. The
    CPU-only build (`USE_CPU_ONLY`) omits both.

### Histogram buckets

`turbo_ocr_request_duration_seconds_bucket` uses nine fixed `le` boundaries
(seconds), plus the implicit `+Inf` bucket:

```
0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 5.000
```

### Sample output

```
# HELP turbo_ocr_requests_total Total HTTP requests by route and status.
# TYPE turbo_ocr_requests_total counter
turbo_ocr_requests_total{route="/ocr/raw",status="2xx"} 1042
turbo_ocr_requests_total{route="/ocr/raw",status="4xx"} 3
turbo_ocr_requests_total{route="/ocr/raw",status="5xx"} 0
# HELP turbo_ocr_request_duration_seconds Request latency histogram.
# TYPE turbo_ocr_request_duration_seconds histogram
turbo_ocr_request_duration_seconds_bucket{route="/ocr/raw",le="0.025"} 980
turbo_ocr_request_duration_seconds_bucket{route="/ocr/raw",le="+Inf"} 1042
turbo_ocr_request_duration_seconds_sum{route="/ocr/raw"} 21.480000
turbo_ocr_request_duration_seconds_count{route="/ocr/raw"} 1042
# HELP turbo_ocr_pipeline_pool_size Number of pipeline slots.
# TYPE turbo_ocr_pipeline_pool_size gauge
turbo_ocr_pipeline_pool_size 5
# HELP turbo_ocr_pool_exhaustions_total Times pipeline pool was full (503).
# TYPE turbo_ocr_pool_exhaustions_total counter
turbo_ocr_pool_exhaustions_total 0
# HELP turbo_ocr_gpu_vram_used_bytes GPU memory currently in use.
# TYPE turbo_ocr_gpu_vram_used_bytes gauge
turbo_ocr_gpu_vram_used_bytes 9052815360
```

---

## Response headers

Injected by the observability middleware on every response.

| Header | When | Value |
|---|---|---|
| `X-Request-Id` | always | UUID v7 (timestamp-ordered). Propagated verbatim if the client sends one, otherwise generated. |
| `X-Inference-Time-Ms` | always | Wall-clock handling time in whole milliseconds (`steady_clock`). |
| `Retry-After` | `503` only | Backoff hint, seconds. The server emits `1`. |

!!! info "Retry-After: server vs nginx"
    The server sets `Retry-After: 1` on its own `503` responses. The bundled
    `docker/config/nginx.conf.template` PRESERVES upstream **502** (it does
    NOT remap to 503) and returns a JSON body; it sets no `Retry-After`
    header of its own — see [HTTP API → Health probes](http.md#health-probes).

---

## Error envelope

Every error response is `Content-Type: application/json` and carries the
shared envelope:

```json
{"error": {"code": "EMPTY_BODY", "message": "Empty body"}}
```

The same `code` strings are surfaced by the gRPC server in the
`x-error-code` trailing metadata field, so clients can branch on one set of
identifiers regardless of transport (`proto/ocr.proto:5-22`).

### Error codes

| Code | HTTP | Meaning |
|---|---|---|
| `EMPTY_BODY` | 400 | Request body was empty. |
| `INVALID_JSON` | 400 | JSON body failed to parse. |
| `MISSING_IMAGE` | 400 | Expected image field absent from the payload. |
| `MISSING_DIMENSIONS` | 400 | `/ocr/pixels` width/height absent from both query params and `X-*` headers. |
| `DIMENSION_CONFLICT` | 400 | `/ocr/pixels` query param and `X-*` header give different width/height/channels. |
| `BASE64_DECODE_FAILED` | 400 | Base64-encoded input could not be decoded. |
| `IMAGE_DECODE_FAILED` | 400 | Image bytes could not be decoded by any codec path. |
| `INVALID_PARAMETER` | 400 | A query parameter value was rejected. |
| `UNSUPPORTED_PARAMETER` | 400 | A parameter is not supported by this build/endpoint. |
| `INVALID_DPI` | 400 | PDF `dpi` outside the accepted 50–600 range. |
| `INVALID_DIMENSIONS` | 400 | Pixel width/height/channels invalid. |
| `DIMENSIONS_TOO_LARGE` | 400 | Image exceeds `MAX_IMAGE_DIM`. |
| `BODY_SIZE_MISMATCH` | 400 | `/ocr/pixels` body length != `width × height × channels`. |
| `EMPTY_BATCH` | 400 | `/ocr/batch` received an empty `images` array. |
| `LAYOUT_DISABLED` | 400 | A layout-dependent feature requested against a `DISABLE_LAYOUT=1` server. |
| `TABLE_BACKEND_DISABLED` | 400 | `tables=1` requested but no table backend loaded — start with `TABLE_BACKEND=slanext` (weights auto-resolve). |
| `FORMULA_BACKEND_DISABLED` | 400 | `formulas=1` requested but no formula backend loaded — start with `FORMULA_BACKEND=ppformulanet_s` (weights auto-resolve). |
| `STRUCTURED_MODE_NO_STRUCTURE` | 400/UNIMPLEMENTED | gRPC `tables`/`formulas` requested under `structured` response mode (use `json_bytes`). |
| `AUTOROTATE_DISABLED` | 400 | `autorotate=1` requested but the doc-orientation model is not loaded. |
| `BATCH_TOO_LARGE` | 400 | `/ocr/batch` / gRPC batch exceeds `MAX_BATCH_IMAGES` (default 1024). |
| `PIXELS_TOO_LARGE` | 400 | Image total area exceeds `MAX_IMAGE_PIXELS_MP` (default 128 MP). |
| `MISSING_FILE` | 400 | Multipart upload missing the `file`/`pdf` field. |
| `MISSING_PDF` | 400 | PDF payload absent. |
| `MISSING_HEADER` / `INVALID_HEADER` | 400 | `/ocr/pixels` dimension header absent or malformed. |
| `INVALID_MULTIPART` | 400 | Multipart body could not be parsed. |
| `EMPTY_PDF` | 400 | PDF decoded to zero bytes. |
| `INVALID_PDF` | 400 | Payload is not a parseable PDF. |
| `PDF_TOO_LARGE` | 400 | PDF page count exceeds `MAX_PDF_PAGES`. |
| `PDF_NOT_AVAILABLE` | 501 | This build has no PDF page renderer — only a build configured `-DTURBO_ENABLE_PDF=OFF`. Every standard build (Linux, macOS, Windows) has one. |
| `PAGE_DECODE_FAILED` / `PAGE_FAILED` | 500 | A PDF page failed to decode / process. |
| `ROUTING_*` | 400 | Routing-config/override rejections: `ROUTING_UNKNOWN_OVERRIDE`, `ROUTING_UNKNOWN_ENGINE`, `ROUTING_BAD_KIND`, `ROUTING_MODALITY_MISMATCH`, `ROUTING_MISSING_SECRET`, `ROUTING_DANGLING_REF`, `ROUTING_EMPTY_PROMPT`, `ROUTING_TEXT_NOT_ROUTABLE`. |
| `ADHOC_BACKENDS_DISABLED` / `ADHOC_LOCAL_DISABLED` | 403 | `/infer` inline backend specs disabled by policy. |
| `BACKEND_UNAVAILABLE` | 503 | Routed remote backend is not reachable. |
| `PDF_RENDER_FAILED` | 500 | PDFium failed to render a page. |
| `PDF_WRITE_FAILED` | 500 | `?output=pdf` searchable-PDF writer failed to produce the output document. |
| `SERVER_BUSY` | 503 | Pipeline pool exhausted; retry after `Retry-After`. |
| `NOT_READY` | 503 | Pipeline not yet ready (e.g. TensorRT engines still building). |
| `INFERENCE_ERROR` | 500 | An error occurred during pipeline inference. |
| `INFERENCE_TIMEOUT` | 504 | Request exceeded `REQUEST_TIMEOUT_MS` (default 60 s). |

!!! note "Transport parity caveat"
    The gRPC "known codes" comment in `proto/ocr.proto` additionally lists
    `PDF_NOT_AVAILABLE` (returned when the build lacks PDF support) and does
    not enumerate the HTTP-only client-validation codes
    (`UNSUPPORTED_PARAMETER`, `MISSING_DIMENSIONS`, `DIMENSION_CONFLICT`,
    `INVALID_JSON`). The `x-error-code` field is additive over the primary
    `grpc::StatusCode`, so older clients can ignore it.

!!! info "See also"
    - [HTTP API](http.md) — endpoints, query parameters, per-endpoint error sets.
    - [gRPC API](grpc.md) — same error codes via `x-error-code` metadata.
    - [Build → Docker](../getting-started/docker.md) — env-var matrix and the nginx `Retry-After` remap.
