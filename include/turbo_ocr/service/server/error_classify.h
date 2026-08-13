#pragma once

// ONE classifier for the exception in flight, shared by both transports.
//
// The HTTP and gRPC inference wrappers each hand-rolled the same catch ladder,
// and they had already drifted: on an ImageDecodeError the HTTP path forwarded
// e.what() while the gRPC path discarded it and returned the literal
// "Decode failed", and PdfRenderError was caught by neither. Every new typed
// error had to be added in two places or it silently degraded to a generic
// INTERNAL on one surface. That is the exact class of drift error_codes.h was
// built to prevent — one table, both transports render — so the CLASSIFICATION
// belongs in one place too.
//
// Policy, decided once here: a typed error forwards its own (controlled)
// message; an untyped std::exception is logged server-side and reduced to a
// generic message so implementation internals never reach the client.

#include <string>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/service/server/error_codes.h"

namespace turbo_ocr::server {

struct ExceptionClass {
  ErrorCode   code;
  std::string message;  // client-facing text
  bool        log;      // true => log server-side (untyped / unknown only)
};

// Call ONLY from inside a catch block: it re-throws the in-flight exception to
// match it by type. The transports then render `code` through the single
// error_codes.h table — HTTP via error_response(code, message), gRPC via
// grpc_error(ctx, code, message) — so they cannot diverge again.
inline ExceptionClass classify_current_exception() {
  try {
    throw;
  } catch (const TimeoutError &e) {
    return {ErrorCode::kInferenceTimeout, e.what(), false};
  } catch (const PoolExhaustedError &e) {
    return {ErrorCode::kServerBusy, e.what(), false};
  } catch (const ImageTooLargeError &e) {
    return {ErrorCode::kDimensionsTooLarge, e.what(), false};
  } catch (const ImageDecodeError &e) {
    return {ErrorCode::kImageDecodeFailed, e.what(), false};
  } catch (const PdfRenderError &e) {
    return {ErrorCode::kPdfRenderFailed, e.what(), false};
  } catch (const std::exception &) {
    return {ErrorCode::kInferenceError, "Inference error", true};
  } catch (...) {
    return {ErrorCode::kInferenceError, "Inference error", true};
  }
}

} // namespace turbo_ocr::server
