// error_codes.h is the ONE place a wire code's HTTP status and gRPC StatusCode
// are decided. Half the codes reach it as a runtime string — a ValidationError
// from validation/, a capability_table.def code, the ROUTING_* prefix parsed out
// of a RoutingConfigError — so the compiler cannot check that those have a row.
// A code with no row answers 500 (error_codes.h::report_unmapped_error_code),
// which for a validation failure means a client sees a server fault. These
// tests are what makes the missing row fail here instead of in production.
#include "catch_amalgamated.hpp"

#include <string_view>

#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/image/size_classify.h"
#include "turbo_ocr/service/server/error_codes.h"

using turbo_ocr::server::ErrorCode;
using turbo_ocr::server::error_entry;
using turbo_ocr::server::error_entry_by_name;
using turbo_ocr::server::error_http_status;
using turbo_ocr::server::GrpcStatusValue;
using turbo_ocr::server::kErrorCodeTable;

TEST_CASE("every ErrorCode round-trips through its wire name", "[error_codes]") {
  for (const auto &e : kErrorCodeTable) {
    CAPTURE(e.name);
    REQUIRE(error_entry_by_name(e.name) != nullptr);
    REQUIRE(error_entry_by_name(e.name)->code == e.code);
    REQUIRE(error_http_status(e.name) == error_http_status(e.code));
  }
}

TEST_CASE("wire names are unique", "[error_codes]") {
  // Two rows with the same name would make the by-name lookup answer whichever
  // came first, silently, for every runtime-string site.
  for (std::size_t i = 0; i < kErrorCodeTable.size(); ++i)
    for (std::size_t j = i + 1; j < kErrorCodeTable.size(); ++j) {
      CAPTURE(kErrorCodeTable[i].name, kErrorCodeTable[j].name);
      REQUIRE(kErrorCodeTable[i].name != kErrorCodeTable[j].name);
    }
}

TEST_CASE("HTTP status and gRPC status agree on the class", "[error_codes]") {
  // The two columns describe the SAME condition to two transports. They are
  // hand-written, so nothing but this stops a 400 row from carrying INTERNAL.
  for (const auto &e : kErrorCodeTable) {
    CAPTURE(e.name, e.http_status);
    const bool client_fault = e.http_status >= 400 && e.http_status < 500;
    const bool grpc_client_fault =
        e.grpc_status == GrpcStatusValue::kInvalidArgument ||
        e.grpc_status == GrpcStatusValue::kPermissionDenied ||
        e.grpc_status == GrpcStatusValue::kNotFound ||
        e.grpc_status == GrpcStatusValue::kFailedPrecondition;
    REQUIRE(client_fault == grpc_client_fault);
  }
}

TEST_CASE("every capability rejection code has a row", "[error_codes]") {
  // capability_table.def decides the code a client gets for a capability the
  // server did not load, and validation/ hands it straight to error_response as
  // a string. Adding a capability without a row here would 500 that rejection.
  using turbo_ocr::capability::CapabilityId;
  using turbo_ocr::capability::capability_info;
#define X(Enum, name, implies, code, hint)                                     \
  {                                                                            \
    const auto &info = capability_info(CapabilityId::Enum);                     \
    CAPTURE(info.error_code);                                                  \
    REQUIRE(error_entry_by_name(info.error_code) != nullptr);                  \
    REQUIRE(error_http_status(info.error_code) == 400);                        \
  }
  TURBO_CAPABILITY_TABLE(X)
#undef X
}

TEST_CASE("every size verdict code has a row", "[error_codes]") {
  // decode::image_size_error_code() is the shared decompression-bomb verdict;
  // size_guards.h passes its return value to error_response verbatim.
  using turbo_ocr::decode::image_size_error_code;
  using turbo_ocr::decode::ImageSizeVerdict;
  for (auto v : {ImageSizeVerdict::kDimTooLarge, ImageSizeVerdict::kPixelsTooLarge}) {
    const std::string_view code = image_size_error_code(v);
    CAPTURE(code);
    REQUIRE(error_entry_by_name(code) != nullptr);
    REQUIRE(error_http_status(code) == 400);
  }
}

TEST_CASE("every runtime-string validation code has a row", "[error_codes]") {
  // The closed set the validation layer and /infer can hand to error_response as
  // a std::string: validation/pixel_dims.h, validation/options_core.h,
  // validation/request_validation.h, and the ROUTING_* prefixes
  // backend/routing_config.cpp puts at the front of a RoutingConfigError.
  // Adding one of those without a row here is exactly the drift this catches.
  for (std::string_view code : {
           "INVALID_PARAMETER",
           "INVALID_DIMENSIONS",
           "DIMENSION_CONFLICT",
           "MISSING_DIMENSIONS",
           "BODY_SIZE_MISMATCH",
           "ROUTING_UNKNOWN_OVERRIDE",
           "ROUTING_BAD_KIND",
           "ROUTING_DANGLING_REF",
           "ROUTING_EMPTY_PROMPT",
           "ROUTING_MISSING_SECRET",
           "ROUTING_MODALITY_MISMATCH",
           "ROUTING_TEXT_NOT_ROUTABLE",
           "ROUTING_UNKNOWN_ENGINE",
       }) {
    CAPTURE(code);
    REQUIRE(error_entry_by_name(code) != nullptr);
    REQUIRE(error_http_status(code) == 400);
  }
}

TEST_CASE("the statuses the routes used to hard-code are preserved",
          "[error_codes]") {
  // The sweep that removed error_response()'s status parameter must not have
  // changed a single answer. These are the non-400 pairs the routes spelled by
  // hand before it.
  REQUIRE(error_http_status(ErrorCode::kNotReady) == 503);
  REQUIRE(error_http_status(ErrorCode::kServerBusy) == 503);
  REQUIRE(error_http_status(ErrorCode::kBackendUnavailable) == 503);
  REQUIRE(error_http_status(ErrorCode::kInferenceTimeout) == 504);
  REQUIRE(error_http_status(ErrorCode::kInferenceError) == 500);
  REQUIRE(error_http_status(ErrorCode::kPageDecodeFailed) == 500);
  REQUIRE(error_http_status(ErrorCode::kPageFailed) == 500);
  REQUIRE(error_http_status(ErrorCode::kPdfWriteFailed) == 500);
  REQUIRE(error_http_status(ErrorCode::kPdfNotAvailable) == 501);
  REQUIRE(error_http_status(ErrorCode::kAdhocBackendsDisabled) == 403);
  // Aligned with its sibling 2026-08-02 (was 400): both ADHOC_* codes are the
  // same class of operator-policy refusal of a well-formed request, so they
  // answer the same status. Deliberate wire change — see error_codes.h.
  REQUIRE(error_http_status(ErrorCode::kAdhocLocalDisabled) == 403);
}
