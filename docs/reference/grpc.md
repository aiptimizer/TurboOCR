# gRPC API

!!! abstract "TL;DR"
    Six unary RPCs (`Recognize`, `RecognizeBatch`, `RecognizePDF`,
    `RecognizeMarkdown`, `InferOne`, `Health`) plus one server-streaming
    RPC (`RecognizeStream`) — one per HTTP endpoint, so the two transports
    expose the same surface. Default bind is `0.0.0.0:50051` (all interfaces — the server has no auth layer by design, so put it behind one before exposing it); override with `TURBO_OCR_HOST`.
    Every response carries a `json_response` byte blob — the same JSON the
    HTTP API returns — as a zero-allocation fast path. Error codes
    live in trailing metadata under `x-error-code`.

The service definition lives in `proto/ocr.proto`; generated stubs land
in `build/proto_gen/{ocr.pb,ocr.grpc.pb}.{h,cc}`.

```proto
service OCRService {
  rpc Recognize         (OCRRequest)         returns (OCRResponse);
  rpc RecognizeBatch    (OCRBatchRequest)    returns (OCRBatchResponse);
  rpc RecognizePDF      (OCRPDFRequest)      returns (OCRPDFResponse);
  rpc RecognizeMarkdown (OCRMarkdownRequest) returns (OCRMarkdownResponse);
  rpc InferOne          (InferOneRequest)    returns (InferOneResponse);
  rpc RecognizeStream   (OCRStreamRequest)   returns (stream OCRStreamEvent);
  rpc Health            (HealthRequest)      returns (HealthResponse);
}
```

Each maps to exactly one HTTP endpoint — `Recognize` ↔ `POST /ocr`,
`RecognizeBatch` ↔ `/ocr/batch`, `RecognizePDF` ↔ `/ocr/pdf`,
`RecognizeMarkdown` ↔ `/ocr/markdown`, `InferOne` ↔ `/infer`,
`RecognizeStream` ↔ `/ocr/stream`, `Health` ↔ `/health` + `/capabilities` —
and both transports parse their options through the same validation core
(`include/turbo_ocr/service/validation/options_core.h`), so a flag cannot mean
one thing over HTTP and another over gRPC.

## Error semantics

!!! note "x-error-code trailing metadata"
    Non-OK responses carry the same string identifier the HTTP API
    returns inside `{"error":{"code":"..."}}` — but in gRPC it lives in
    trailing metadata under the key `x-error-code`. Read it via
    `ClientContext::GetServerTrailingMetadata()` after the call
    completes. The `grpc::StatusCode` + message remain primary for
    older clients; `x-error-code` is purely additive.

Known codes (from `proto/ocr.proto:12-18`): `LAYOUT_DISABLED`,
`INVALID_PARAMETER`, `BASE64_DECODE_FAILED`, `IMAGE_DECODE_FAILED`,
`DIMENSIONS_TOO_LARGE`, `EMPTY_BATCH`, `MISSING_IMAGE`, `MISSING_PDF`,
`INVALID_DPI`, `INVALID_DIMENSIONS`, `BODY_SIZE_MISMATCH`,
`PDF_RENDER_FAILED`, `PDF_NOT_AVAILABLE`, `EMPTY_PDF`, `SERVER_BUSY`,
`INFERENCE_ERROR`, `NOT_READY`, `TABLE_BACKEND_DISABLED`,
`FORMULA_BACKEND_DISABLED`, `STRUCTURED_MODE_NO_STRUCTURE`.

## Response mode (json_bytes vs. structured)

!!! tip "Use the JSON fast path"
    `OCRResponse.json_response` contains the same pre-serialized JSON
    body the HTTP API returns. It avoids ~455 heap allocations per
    response on a 35-detection page (see `proto/ocr.proto:64-68`).

The response mode is `--grpc-response-mode` / `GRPC_RESPONSE_MODE` with
exactly two values (see `./turboocr-server --print-config`), and they are
**mutually exclusive**: the default `json_bytes` fills only
`json_response`; `structured` fills only the repeated `results` fields
(text-only — see the warning below). `response_mode` in every reply names
which one the server ran, so a client can hard-fail on a mismatch instead
of reading an empty field.

!!! warning "`tables`/`formulas` require json_bytes mode"
    The structured `results` field carries text only — the proto has no
    table/formula message. So in `structured` response mode a `tables=1`/
    `formulas=1` request is **rejected** with `UNIMPLEMENTED` +
    `x-error-code: STRUCTURED_MODE_NO_STRUCTURE` (it is not silently
    degraded). Use the default `json_bytes` mode and read `tables`/`formulas`
    from `json_response`, exactly like the HTTP API.

`reading_order` is duplicated as a top-level repeated field for clients
that don't parse the JSON.

---

## `Recognize(OCRRequest) → OCRResponse`

Single-image recognition. Request carries either an encoded image **or**
already-decoded BGR pixels.

```proto
message OCRRequest {
  bytes image         = 1;   // encoded JPEG/PNG/... bytes
  bool  layout        = 5;
  bytes pixels        = 6;   // raw BGR (alternative to `image`)
  int32 width         = 7;
  int32 height        = 8;
  int32 channels      = 9;
  bool  reading_order = 10;  // auto-enables layout
  bool  as_blocks     = 11;  // auto-enables layout + reading_order
  bool  tables        = 12;  // strict opt-in; auto-enables layout
  bool  formulas      = 13;  // strict opt-in; auto-enables layout
}
```

`tables`/`formulas` are strict opt-in (like HTTP `?tables=1`/`?formulas=1`): a
configured backend is necessary but the field must be set for the stage to run.
Setting the field with no backend configured fails loud — `INVALID_ARGUMENT` with
`x-error-code: TABLE_BACKEND_DISABLED` / `FORMULA_BACKEND_DISABLED`.
They surface in `json_response` (json_bytes mode). The encoded path mirrors HTTP
`/ocr/raw`; the pixels path mirrors
`/ocr/pixels`. Layout / reading-order / blocks have the same
auto-promotion rules as the HTTP query parsers — see
`include/turbo_ocr/service/grpc/grpc_service.h:217-225`.

```proto
message OCRResponse {
  repeated OCRResult results    = 1;
  int32   num_detections        = 2;
  bytes   json_response         = 3;  // pre-serialized JSON, fast path
  repeated int32 reading_order  = 4;
}

message OCRResult {
  string text        = 1;
  float  confidence  = 2;
  repeated BoundingBox bounding_box = 3;
}

message BoundingBox { repeated float x = 1; repeated float y = 2; }
```

### Example call

=== "python"

    ```python
    import json, grpc
    import ocr_pb2, ocr_pb2_grpc

    with grpc.insecure_channel("localhost:50051") as ch:
        stub = ocr_pb2_grpc.OCRServiceStub(ch)
        with open("page.jpg", "rb") as f:
            resp = stub.Recognize(ocr_pb2.OCRRequest(
                image=f.read(), layout=True, reading_order=True))
        page = json.loads(resp.json_response)
        print(page["results"][0]["text"])
    ```

=== "cpp"

    ```cpp
    #include <grpcpp/grpcpp.h>
    #include "ocr.grpc.pb.h"

    auto chan = grpc::CreateChannel("localhost:50051",
                                    grpc::InsecureChannelCredentials());
    auto stub = ocr::OCRService::NewStub(chan);

    ocr::OCRRequest req;
    req.set_image(jpeg_bytes.data(), jpeg_bytes.size());
    req.set_layout(true);
    req.set_reading_order(true);

    ocr::OCRResponse resp;
    grpc::ClientContext ctx;
    auto status = stub->Recognize(&ctx, req, &resp);
    if (!status.ok()) {
      const auto &trail = ctx.GetServerTrailingMetadata();
      auto it = trail.find("x-error-code");
      // it->second is the same string the HTTP error envelope returns.
    }
    ```

---

## `RecognizeBatch(OCRBatchRequest) → OCRBatchResponse`

```proto
message OCRBatchRequest {
  repeated bytes images   = 1;
  int32 det_batch_num     = 2;
  bool  layout            = 3;
  bool  reading_order     = 4;  // auto-enables layout
  bool  as_blocks         = 5;  // auto-enables layout + reading_order
  bool  tables            = 6;  // strict opt-in; auto-enables layout
  bool  formulas          = 7;  // strict opt-in; auto-enables layout
}

message OCRBatchResponse {
  repeated OCRResponse batch_results = 1;
  int32 total_images                 = 2;
}
```

!!! note "Per-slot errors"
    Per-slot error handling matches HTTP `/ocr/batch`: a failing slot
    populates an empty `results` array and an `errors[]` entry inside
    that slot's `json_response`. The index remains aligned with the
    input order — one bad image never silently drops the rest.

---

## `RecognizePDF(OCRPDFRequest) → OCRPDFResponse`

```proto
message OCRPDFRequest {
  bytes  pdf_data  = 1;
  string mode      = 2;   // "ocr" | "geometric" | "auto" | "auto_verified"
  int32  dpi       = 3;   // default 100, clamped to [50, 600]
  bool   layout    = 4;
  bool   as_blocks = 5;   // auto-enables layout
  bool   tables    = 6;   // strict opt-in; auto-enables layout
  bool   formulas  = 7;   // strict opt-in; auto-enables layout
}

message OCRPDFResponse {
  repeated OCRPageResult pages = 1;
}

message OCRPageResult {
  int32  page_number          = 1;
  repeated OCRResult results  = 2;
  int32  width                = 3;
  int32  height               = 4;
  int32  dpi                  = 5;
  string mode                 = 6;   // resolved per-page mode
  string text_layer_quality   = 7;   // "absent" | "rejected" | "trusted"
  bytes  json_response        = 8;   // pre-serialized per-page JSON
}
```

!!! warning "auto_verified is GPU-only"
    On a CPU build `auto_verified` silently downgrades to `auto` — the
    per-page `mode` field in the response reflects what actually ran,
    so honest clients can branch on it.

---

## `RecognizeMarkdown` · `InferOne` · `RecognizeStream`

The remaining three RPCs mirror their HTTP twins exactly; the shapes worth
knowing:

```proto
message OCRMarkdownRequest  { bytes image = 1; }   // no per-request flags —
    // the export always runs every stage the server LOADED (matches HTTP)
message OCRMarkdownResponse { string markdown = 1; string degraded = 2; }
    // `degraded` = HTTP's X-OCR-Degraded: failed regions are DROPPED from the
    // markdown body, so this field is the only trace of a partial page

message InferOneRequest  { bytes image = 1; string modality = 2;  // "table"|"formula"
                           string backend = 3; }   // routing-table NAME only —
    // inline backend specs are deliberately unrepresentable over gRPC
message InferOneResponse { string modality = 1; string html = 2; string latex = 3; }

message OCRStreamRequest { bytes data = 1;  // PDF or image, content-sniffed
                           string mode = 2; int32 dpi = 3; bool layout = 4;
                           bool reading_order = 5; bool as_blocks = 6;
                           bool tables = 7; bool formulas = 8; bool autorotate = 9; }
message OCRStreamEvent   { string event = 1;  // "meta"|"page"|"page_error"|"error"|"end"
                           int32 page_index = 6;  // 0-based; pages arrive OUT OF ORDER
                           bytes json_response = 7; /* … kind/pages/dpi/mode/w/h */ }
```

Each `page` event's `json_response` is byte-identical to the corresponding
`/ocr/pdf` `pages[]` element; match events by `page_index`, never by arrival
order.

---

## `Health(HealthRequest) → HealthResponse`

```proto
message HealthRequest {}
message HealthResponse { string status = 1; }
```

Returns `status = "ok"` when ready. While TensorRT engines are still
being built on first startup the RPC returns `UNAVAILABLE` with
`x-error-code: NOT_READY`, matching HTTP `/health/ready`'s 503.

---

## Regenerating stubs

The Python stubs at `tests/_grpc_generated/` are committed for the test
suite; production clients should regenerate against `proto/ocr.proto`:

=== "python"

    ```bash
    python -m grpc_tools.protoc -I proto \
        --python_out=. --grpc_python_out=. proto/ocr.proto
    ```

=== "cpp"

    ```bash
    protoc -I proto --cpp_out=. proto/ocr.proto
    protoc -I proto --grpc_out=. \
        --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin)" proto/ocr.proto
    ```

=== "javascript"

    ```bash
    npx grpc_tools_node_protoc -I proto \
        --js_out=import_style=commonjs:. \
        --grpc_out=grpc_js:. proto/ocr.proto
    ```

!!! info "See also"
    - [HTTP API](http.md) — same surface, plain JSON.
    - [Build → Docker](../getting-started/docker.md) — `TURBO_OCR_GRPC_PORT` and
      the `--grpc-response-mode` flag.
    - [Architecture → Pipeline](../architecture/pipeline.md) — what
      happens inside each RPC.
