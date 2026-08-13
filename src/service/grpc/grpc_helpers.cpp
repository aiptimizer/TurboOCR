// Free gRPC helper definitions shared by the RPC TUs; declarations in
// turbo_ocr/service/grpc/grpc_service.h.
#include "turbo_ocr/service/grpc/grpc_service.h"

#include "turbo_ocr/image/cpu_image_decode.h"
#include "turbo_ocr/image/size_classify.h"

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

// (grpc_check_layout_request lived here. It opened "Mirror
// parse_query_options()" and was exactly that: a second implementation of the
// layout-availability gate, maintained by hand alongside the HTTP one. It is
// gone — all three RPCs now call parse_proto_options (validation/proto_options.h),
// which runs the SAME parse_options_core the HTTP routes run, so the rejection
// code for a given condition comes from capability_table.def once instead of
// from two places that have to be kept in step.)

// Fail loud over gRPC when the client opts into a structure stage the server
// can't do (parity with the HTTP check_structure_backends).
[[nodiscard]] std::optional<grpc::Status>
grpc_check_structure_backends(grpc::ServerContext *ctx,
                              const capability::CapabilityMask &requested,
                              const capability::CapabilityMask &loaded,
                              bool json_bytes_mode,
                              bool want_blocks, bool raw_layout) {
  using capability::CapabilityId;
  const bool want_tables   = requested.get(CapabilityId::Table);
  const bool want_formulas = requested.get(CapabilityId::Formula);
  // Structured-mode gate: the RAW layout flag, never the implied one — see the
  // declaration. reading_order-only requests imply layout internally but have
  // a dedicated structured response field, so they must pass this gate.
  const bool want_layout   = raw_layout;
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
  // The same table-driven rejection the HTTP gate uses, so a capability added
  // to capability_table.def is refused identically on both transports and the
  // codes cannot drift apart.
  if (const auto missing = requested.without(loaded).first()) {
    const auto &info = capability::capability_info(*missing);
    const std::string code(info.error_code);
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, code.c_str(),
                      std::format("{} is required for this request but {}.",
                                  info.name, info.hint));
  }
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

// Host decoder for the gRPC handlers' non-encoded path — used when the server
// has no EncodedInferFunc (no pipeline pool), so the transport must materialize
// a cv::Mat itself. Every format lands here, JPEG included: the old
// "JPEGs are routed elsewhere" split disappeared with the vendor-specific
// dispatcher decode branch.
cv::Mat grpc_decode_image(std::string_view image_data) {
  return decode::decode_cpu_fallback(
      reinterpret_cast<const unsigned char *>(image_data.data()),
      image_data.size());
}


} // namespace turbo_ocr::server
