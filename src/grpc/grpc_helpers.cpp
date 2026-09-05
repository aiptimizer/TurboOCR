// Free gRPC helper definitions shared by the RPC TUs; declarations in
// turbo_ocr/grpc/grpc_service.h.
#include "turbo_ocr/grpc/grpc_service.h"

#include "turbo_ocr/decode/cpu_image_decode.h"
#include "turbo_ocr/decode/size_classify.h"
#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/jpeg_infer.h"
#endif

namespace turbo_ocr::server {

// Helper: stamp the structured HTTP-parity error code into gRPC trailing
// metadata under "x-error-code" and return the status. Keeps the legacy
// StatusCode/message untouched so existing clients keep working while
// new clients can branch on the structured code (matches HTTP's
// {"error":{"code":...}} payload one-for-one).
[[nodiscard]] grpc::Status
grpc_error(grpc::ServerContext *ctx, grpc::StatusCode code,
           const char *error_code, std::string message) {
  if (ctx) ctx->AddTrailingMetadata("x-error-code", error_code);
  return grpc::Status(code, std::move(message));
}

// Same as grpc_error but sources the wire string + gRPC status from the shared
// error_codes.h table, so the code/status pairing can't drift from HTTP. Used
// for codes that have no hand-written literal at the call site (e.g. the C4
// inference-timeout -> DEADLINE_EXCEEDED mapping).
[[nodiscard]] grpc::Status
grpc_error(grpc::ServerContext *ctx, ErrorCode code, std::string message) {
  std::string_view name = error_code_str(code);
  if (ctx) ctx->AddTrailingMetadata("x-error-code", std::string(name));
  return grpc::Status(error_grpc_status(code), std::move(message));
}

// Mirror parse_query_options() in server_types.h: when the client asks for
// layout-derived output but the server was started without the layout
// model, HTTP rejects the request with LAYOUT_DISABLED (the one stable
// code for the condition, whichever flag triggered it). gRPC used to
// silently zero those flags, leaving callers wondering why they got a
// y/x fallback they did not ask for. Returns nullopt on success.
[[nodiscard]] std::optional<grpc::Status>
grpc_check_layout_request(grpc::ServerContext *ctx, bool req_layout,
                           bool req_reading_order, bool layout_available) {
  if (req_layout && !layout_available) {
    // LAYOUT_DISABLED for every "layout feature unavailable" rejection —
    // one stable code per condition, matching the HTTP routes and the
    // reading_order case below.
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "LAYOUT_DISABLED",
                      "Layout requested but the layout model is not loaded. "
                      "Either models/layout/layout.onnx is missing from the "
                      "image, or the server was started with DISABLE_LAYOUT=1.");
  }
  if (req_reading_order && !layout_available) {
    // `req_reading_order` here folds in reading_order/as_blocks/tables/formulas
    // — every layout-derived feature — so keep the message generic.
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "LAYOUT_DISABLED",
                      "reading_order/as_blocks/tables/formulas require the "
                      "layout model: start the server without DISABLE_LAYOUT=1 "
                      "(layout is on by default)");
  }
  return std::nullopt;
}

// Fail loud over gRPC when the client opts into a structure stage the server
// can't do (parity with the HTTP check_structure_backends).
[[nodiscard]] std::optional<grpc::Status>
grpc_check_structure_backends(grpc::ServerContext *ctx, bool want_tables,
                              bool want_formulas, bool table_available,
                              bool formula_available,
                              bool json_bytes_mode,
                              bool want_layout,
                              bool want_blocks) {
  // Structured response mode carries only `results` + `reading_order` (the
  // proto has no table/formula/layout/blocks message). Running any of those
  // stages then dropping the output is a silent failure — reject loudly so a
  // structured-mode client knows to use json_bytes. reading_order stays
  // allowed: it HAS a dedicated proto field.
  if ((want_tables || want_formulas || want_layout || want_blocks) &&
      !json_bytes_mode)
    return grpc_error(ctx, grpc::StatusCode::UNIMPLEMENTED,
                      "STRUCTURED_MODE_NO_STRUCTURE",
                      "layout/blocks/tables/formulas require the json_bytes "
                      "gRPC response mode (structured mode returns only text "
                      "results + reading_order)");
  if (want_tables && !table_available)
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "TABLE_BACKEND_DISABLED",
                      "tables=1 requested but no table backend is configured "
                      "(start the server with TABLE_BACKEND=...)");
  if (want_formulas && !formula_available)
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "FORMULA_BACKEND_DISABLED",
                      "formulas=1 requested but no formula backend is configured "
                      "(start the server with FORMULA_BACKEND=...)");
  return std::nullopt;
}

// The gRPC emission adapter over the shared decompression-bomb verdict
// (decode/size_classify.h): nullopt when the size is acceptable, else the
// INVALID_ARGUMENT status with the shared code + message.
[[nodiscard]] std::optional<grpc::Status>
grpc_check_image_size(grpc::ServerContext *ctx, int w, int h) {
  auto v = decode::classify_image_size(w, h);
  if (v == decode::ImageSizeVerdict::kOk) return std::nullopt;
  return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                    decode::image_size_error_code(v),
                    decode::image_size_error_message(v, w, h));
}

// Returns nullopt on success, or a status carrying DIMENSIONS_TOO_LARGE when
// the encoded image's PNG/JPEG/WebP header advertises width or height beyond
// MAX_IMAGE_DIM. Caller checks before paying the decode cost — same
// decompression-bomb defense the HTTP routes apply.
[[nodiscard]] std::optional<grpc::Status>
grpc_pre_decode_dim_check(grpc::ServerContext *ctx,
                           std::string_view image_data) {
  auto *data = reinterpret_cast<const unsigned char *>(image_data.data());
  if (auto d = decode::peek_image_dimensions(data, image_data.size()))
    return grpc_check_image_size(ctx, d->width, d->height);
  return std::nullopt;
}

// Pure-CPU decoder for the non-JPEG branch of the gRPC handlers. JPEGs are
// routed via grpc_jpeg_decode_and_infer (decode happens on a dispatcher
// worker thread); reaching this with JPEG bytes would be a caller bug.
cv::Mat grpc_decode_image(std::string_view image_data) {
  return decode::decode_cpu_fallback(
      reinterpret_cast<const unsigned char *>(image_data.data()),
      image_data.size());
}

#ifndef USE_CPU_ONLY
// Decode + infer on a dispatcher worker thread so nvJPEG's async NVDEC work
// runs on the pipeline's own stream — matches /ocr/raw and avoids the
// cross-thread DMA race that poisoned the CUDA context.
std::future<pipeline::OcrPipelineResult>
grpc_jpeg_decode_and_infer(pipeline::PipelineDispatcher &dispatcher,
                           std::string_view image_bytes,
                           bool want_layout, bool want_reading_order,
                           bool want_tables, bool want_formulas,
                           const backend_routing::RequestRouting &routing,
                           bool layout_only) {
  std::string owned(image_bytes);
  pipeline::JpegRunOpts run_opts{
      .want_layout = want_layout,
      .want_reading_order = want_reading_order,
      .want_tables = want_tables,
      .want_formulas = want_formulas,
      .routing = routing,
      .defer_external = false,
      .layout_only = layout_only,
  };
  return dispatcher.submit(
      [owned = std::move(owned), run_opts](auto &e) {
        return pipeline::decode_jpeg_and_run(
            e, reinterpret_cast<const unsigned char *>(owned.data()),
            owned.size(), run_opts);
      });
}
#endif

} // namespace turbo_ocr::server
