#pragma once

// Umbrella for the server request/response plumbing. The concern slices are
// service_fns.h (function aliases, leaf), http_responses.h, infer_result.h,
// validation/query_options.h, common/uuid.h, decode/cpu_image_decode.h and
// work_pool.h; the remaining includes below are their transitive spine
// (serialization, routing config, metrics, logging). Route TUs should include
// the slice they need; this umbrella keeps the historical include surface for
// TUs that genuinely use most of it.

#include <functional>
#include <memory>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/common/uuid.h"
#include "turbo_ocr/decode/cpu_image_decode.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/server/http_responses.h"
#include "turbo_ocr/server/infer_result.h"
#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/service_fns.h"
#include "turbo_ocr/validation/query_options.h"
#include "turbo_ocr/validation/request_validation.h"

namespace turbo_ocr::server {

using ::turbo_ocr::generate_uuid_v7;

[[nodiscard]] inline cv::Mat cpu_decode_image(const unsigned char *data, size_t len) {
  return decode::decode_cpu_fallback(data, len);
}

} // namespace turbo_ocr::server

#include "turbo_ocr/server/work_pool.h"

namespace turbo_ocr::server {

// ── Work submission ─────────────────────────────────────────────────────

/// Submit blocking work to a WorkPool safely.
/// Callback is wrapped in shared_ptr so it survives if submit() throws.
/// Observability headers (X-Request-Id, X-Inference-Time-Ms, Retry-After)
/// are injected by the middleware registered in register_observability_middleware().
template <typename F>
void submit_work(WorkPool &pool, DrogonCallback &&callback, F &&work) {
  auto cb = std::make_shared<DrogonCallback>(std::move(callback));
  try {
    pool.submit([cb, w = std::forward<F>(work)]() mutable { w(*cb); });
  } catch (const turbo_ocr::PoolExhaustedError &e) {
    Metrics::instance().record_pool_exhaustion();
    (*cb)(error_response(drogon::k503ServiceUnavailable, "SERVER_BUSY", e.what()));
  }
}

/// Register Drogon middleware for observability headers and metrics.
/// Call once before drogon::app().run().
///
/// Pre-handling:  generates X-Request-Id (or propagates from client),
///                records request start time in request attributes.
/// Post-handling: injects X-Request-Id, X-Inference-Time-Ms, Retry-After
///                headers; records metrics.
void register_observability_middleware();

} // namespace turbo_ocr::server
