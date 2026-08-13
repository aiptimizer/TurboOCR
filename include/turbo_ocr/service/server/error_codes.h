#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <string_view>

// Single source of truth for the structured error codes both transports emit:
// HTTP returns them inside {"error":{"code":"..."}}, gRPC stamps them into the
// "x-error-code" trailing metadata. Each code carries the HTTP status and the
// gRPC StatusCode it maps to so the two surfaces stay in lockstep — change a
// mapping here and both adopt it.
//
// The gRPC status is stored as the canonical grpc::StatusCode integer value so
// this header has no dependency on <grpcpp/grpcpp.h> and compiles standalone.
// Callers that already include grpcpp can pass the int straight into
// grpc::Status / static_cast it to grpc::StatusCode; grpc_status_code() is a
// thin, optionally-compiled convenience for that cast.

namespace turbo_ocr::server {

enum class ErrorCode {
  kMissingImage,
  kMissingPdf,
  kMissingFile,
  kMissingHeader,
  kInvalidHeader,
  kInvalidJson,
  kInvalidMultipart,
  kInvalidDpi,
  kInvalidDimensions,
  kInvalidParameter,
  kBase64DecodeFailed,
  kImageDecodeFailed,
  kDimensionsTooLarge,
  kBodySizeMismatch,
  kEmptyBody,
  kEmptyBatch,
  kBatchTooLarge,
  kEmptyPdf,
  kPdfTooLarge,
  kPdfRenderFailed,
  kAutorotateDisabled,
  kLayoutDisabled,
  kNotReady,
  kServerBusy,
  kPdfNotAvailable,
  kInferenceError,
  kPageDecodeFailed,
  kInferenceTimeout,
  // ── Codes the HTTP routes used to spell with a hand-written status ──────
  // These had no row here, which is why the routes could not derive their
  // status and hard-coded it instead. Their gRPC column is the status the
  // equivalent condition would carry; no RPC emits them today.
  kPixelsTooLarge,
  kDimensionConflict,
  kMissingDimensions,
  kInvalidPdf,
  kTableBackendDisabled,
  kFormulaBackendDisabled,
  kRoutingUnknownOverride,
  kRoutingBadKind,
  kRoutingDanglingRef,
  kRoutingEmptyPrompt,
  kRoutingMissingSecret,
  kRoutingModalityMismatch,
  kRoutingTextNotRoutable,
  kRoutingUnknownEngine,
  kAdhocBackendsDisabled,
  kAdhocLocalDisabled,
  kBackendUnavailable,
  kPageFailed,
  kPdfWriteFailed,
};

// Canonical grpc::StatusCode integer values (grpc/status.h), inlined so this
// header stays free of the grpcpp dependency.
enum class GrpcStatusValue : int {
  kOk = 0,
  kCancelled = 1,
  kUnknown = 2,
  kInvalidArgument = 3,
  kDeadlineExceeded = 4,
  kNotFound = 5,
  kAlreadyExists = 6,
  kPermissionDenied = 7,
  kResourceExhausted = 8,
  kFailedPrecondition = 9,
  kAborted = 10,
  kOutOfRange = 11,
  kUnimplemented = 12,
  kInternal = 13,
  kUnavailable = 14,
  kDataLoss = 15,
  kUnauthenticated = 16,
};

struct ErrorCodeEntry {
  ErrorCode      code = ErrorCode::kInferenceError;
  std::string_view name;   // wire string ("MISSING_IMAGE", …)
  int            http_status = 500;   // HTTP status code (e.g. 400, 503, 504)
  GrpcStatusValue grpc_status = GrpcStatusValue::kInternal;
};

// One row per ErrorCode, declaration order matching the enum so an index lookup
// is O(1). HTTP/gRPC mappings reproduce the literals previously scattered
// across the routes, grpc_service.h, server_types.h and proto/ocr.proto.
inline constexpr std::array<ErrorCodeEntry, 47> kErrorCodeTable{{
    {ErrorCode::kMissingImage,       "MISSING_IMAGE",        400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kMissingPdf,         "MISSING_PDF",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kMissingFile,        "MISSING_FILE",         400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kMissingHeader,      "MISSING_HEADER",       400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidHeader,      "INVALID_HEADER",       400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidJson,        "INVALID_JSON",         400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidMultipart,   "INVALID_MULTIPART",    400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidDpi,         "INVALID_DPI",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidDimensions,  "INVALID_DIMENSIONS",   400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidParameter,   "INVALID_PARAMETER",    400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kBase64DecodeFailed, "BASE64_DECODE_FAILED", 400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kImageDecodeFailed,  "IMAGE_DECODE_FAILED",  400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kDimensionsTooLarge, "DIMENSIONS_TOO_LARGE", 400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kBodySizeMismatch,   "BODY_SIZE_MISMATCH",   400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kEmptyBody,          "EMPTY_BODY",           400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kEmptyBatch,         "EMPTY_BATCH",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kBatchTooLarge,      "BATCH_TOO_LARGE",      400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kEmptyPdf,           "EMPTY_PDF",            400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kPdfTooLarge,        "PDF_TOO_LARGE",        400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kPdfRenderFailed,    "PDF_RENDER_FAILED",    400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kAutorotateDisabled, "AUTOROTATE_DISABLED",  400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kLayoutDisabled,     "LAYOUT_DISABLED",      400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kNotReady,           "NOT_READY",            503, GrpcStatusValue::kUnavailable},
    {ErrorCode::kServerBusy,         "SERVER_BUSY",          503, GrpcStatusValue::kResourceExhausted},
    {ErrorCode::kPdfNotAvailable,    "PDF_NOT_AVAILABLE",    501, GrpcStatusValue::kUnimplemented},
    {ErrorCode::kInferenceError,     "INFERENCE_ERROR",      500, GrpcStatusValue::kInternal},
    {ErrorCode::kPageDecodeFailed,   "PAGE_DECODE_FAILED",   500, GrpcStatusValue::kInternal},
    {ErrorCode::kInferenceTimeout,   "INFERENCE_TIMEOUT",    504, GrpcStatusValue::kDeadlineExceeded},
    {ErrorCode::kPixelsTooLarge,     "PIXELS_TOO_LARGE",     400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kDimensionConflict,  "DIMENSION_CONFLICT",   400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kMissingDimensions,  "MISSING_DIMENSIONS",   400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kInvalidPdf,         "INVALID_PDF",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kTableBackendDisabled,   "TABLE_BACKEND_DISABLED",   400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kFormulaBackendDisabled, "FORMULA_BACKEND_DISABLED", 400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingUnknownOverride,  "ROUTING_UNKNOWN_OVERRIDE",  400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingBadKind,          "ROUTING_BAD_KIND",          400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingDanglingRef,      "ROUTING_DANGLING_REF",      400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingEmptyPrompt,      "ROUTING_EMPTY_PROMPT",      400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingMissingSecret,    "ROUTING_MISSING_SECRET",    400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingModalityMismatch, "ROUTING_MODALITY_MISMATCH", 400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingTextNotRoutable,  "ROUTING_TEXT_NOT_ROUTABLE", 400, GrpcStatusValue::kInvalidArgument},
    {ErrorCode::kRoutingUnknownEngine,    "ROUTING_UNKNOWN_ENGINE",    400, GrpcStatusValue::kInvalidArgument},
    // 403, not 400: the request is well-formed and it is OPERATOR policy
    // (TURBO_ALLOW_ADHOC_BACKENDS / the ad-hoc-local guard) that refuses it.
    // ADHOC_LOCAL_DISABLED historically answered 400; aligned to 403 with its
    // sibling 2026-08-02 (deliberate wire change — both are the same class of
    // policy refusal, and answering two statuses for it taught clients nothing).
    {ErrorCode::kAdhocBackendsDisabled, "ADHOC_BACKENDS_DISABLED", 403, GrpcStatusValue::kPermissionDenied},
    {ErrorCode::kAdhocLocalDisabled,    "ADHOC_LOCAL_DISABLED",    403, GrpcStatusValue::kPermissionDenied},
    {ErrorCode::kBackendUnavailable, "BACKEND_UNAVAILABLE",  503, GrpcStatusValue::kUnavailable},
    {ErrorCode::kPageFailed,         "PAGE_FAILED",          500, GrpcStatusValue::kInternal},
    {ErrorCode::kPdfWriteFailed,     "PDF_WRITE_FAILED",     500, GrpcStatusValue::kInternal},
}};

static_assert(kErrorCodeTable.size() ==
                  static_cast<std::size_t>(ErrorCode::kPdfWriteFailed) + 1,
              "kErrorCodeTable must have exactly one row per ErrorCode, in "
              "enum order — error_entry() indexes it by enum value");

namespace detail {
// Rows are looked up BY INDEX, so a row inserted in the wrong slot silently
// hands every caller a different code's status. The `code` field is the
// redundancy that makes that a build error instead.
[[nodiscard]] inline constexpr bool error_table_is_in_enum_order() {
  for (std::size_t i = 0; i < kErrorCodeTable.size(); ++i)
    if (static_cast<std::size_t>(kErrorCodeTable[i].code) != i) return false;
  return true;
}
} // namespace detail
static_assert(detail::error_table_is_in_enum_order(),
              "kErrorCodeTable row order does not match the ErrorCode enum");

[[nodiscard]] inline constexpr const ErrorCodeEntry &error_entry(ErrorCode code) {
  // Index by enum value; the table is declared in enum order.
  return kErrorCodeTable[static_cast<std::size_t>(code)];
}

// Wire string for a code, e.g. "MISSING_IMAGE". The returned view points into
// the static table and is valid for the program lifetime.
[[nodiscard]] inline constexpr std::string_view error_code_str(ErrorCode code) {
  return error_entry(code).name;
}

[[nodiscard]] inline constexpr int error_http_status(ErrorCode code) {
  return error_entry(code).http_status;
}

// ── Lookup by wire name ─────────────────────────────────────────────────
// For the sites that only ever hold the code as a runtime string: a
// ValidationError from validation/, a capability_table.def code, or the
// ROUTING_* prefix parsed out of a RoutingConfigError message. They must reach
// the SAME mapping as the sites that name an ErrorCode, or the table stops
// being the single source of truth for exactly the codes it cannot see.
[[nodiscard]] inline constexpr const ErrorCodeEntry *
error_entry_by_name(std::string_view name) {
  for (const auto &e : kErrorCodeTable)
    if (e.name == name) return &e;
  return nullptr;
}

namespace detail {
// An unmapped name is a MISSING ROW, not a client condition. Answering 400
// would label a server-side omission as bad input and hide it forever, so the
// fallback is 500 and it is loud — same contract as size_classify.h's
// report_ok_verdict_misuse: assert in debug, stderr in release.
inline int report_unmapped_error_code(std::string_view name) {
  std::fprintf(stderr,
               "[error_codes] no row for error code '%.*s' — add it to "
               "kErrorCodeTable; answering 500.\n",
               static_cast<int>(name.size()), name.data());
  assert(false && "error code has no kErrorCodeTable row — see stderr");
  return 500;
}
} // namespace detail

[[nodiscard]] inline int error_http_status(std::string_view name) {
  const auto *e = error_entry_by_name(name);
  return e ? e->http_status : detail::report_unmapped_error_code(name);
}

// gRPC status as the canonical grpc::StatusCode integer value. Callers holding
// grpcpp can static_cast<grpc::StatusCode>(...) it.
[[nodiscard]] inline constexpr int error_grpc_status_value(ErrorCode code) {
  return static_cast<int>(error_entry(code).grpc_status);
}

} // namespace turbo_ocr::server

#ifdef GRPCPP_GRPCPP_H
// Only available where grpcpp is already in scope; keeps this header standalone
// for translation units that never touch gRPC.
namespace turbo_ocr::server {
[[nodiscard]] inline grpc::StatusCode error_grpc_status(ErrorCode code) {
  return static_cast<grpc::StatusCode>(error_grpc_status_value(code));
}
} // namespace turbo_ocr::server
#endif
