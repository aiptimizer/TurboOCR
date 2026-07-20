// Observability middleware (request-id, timing headers, metrics), extracted
// from server_types.h; declaration stays there.
#include <charconv>
#include <chrono>
#include <random>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/server_types.h"

namespace turbo_ocr::server {

void register_observability_middleware() {
  // Pre-request: assign request ID + start time
  drogon::app().registerPreHandlingAdvice(
      [](const drogon::HttpRequestPtr &req) {
        auto id = req->getHeader("X-Request-Id");
        if (id.empty()) id = generate_uuid_v7();
        req->addHeader("X-Request-Id", id);  // store for post-handler
        // Store start time as attribute (nanoseconds since epoch)
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        req->addHeader("X-Start-Ns", std::to_string(now));
      });

  // Post-request: inject response headers + record metrics
  drogon::app().registerPostHandlingAdvice(
      [](const drogon::HttpRequestPtr &req,
         const drogon::HttpResponsePtr &resp) {
        // X-Request-Id
        auto req_id = req->getHeader("X-Request-Id");
        if (!req_id.empty())
          resp->addHeader("X-Request-Id", req_id);

        // X-Inference-Time-Ms
        auto start_ns_str = req->getHeader("X-Start-Ns");
        double duration_s = 0.0;
        if (!start_ns_str.empty()) {
          // Best-effort timing only: a malformed X-Start-Ns just omits the
          // X-Inference-Time-Ms header — never fail the response over an
          // observability detail.
          long long start_ns = 0;
          const char *first = start_ns_str.data();
          const char *last = first + start_ns_str.size();
          if (auto [ptr, ec] = std::from_chars(first, last, start_ns);
              ec == std::errc{} && ptr == last) {
            auto now_ns =
                std::chrono::steady_clock::now().time_since_epoch().count();
            auto ms = (now_ns - start_ns) / 1'000'000;
            resp->addHeader("X-Inference-Time-Ms", std::to_string(ms));
            duration_s = static_cast<double>(now_ns - start_ns) / 1e9;
          }
        }

        // Deprecated param-tolerance relay (set by the request gate): tell
        // the client which unsupported params were ignored and that v4
        // rejects them.
        auto ignored = req->getHeader("X-Ignored-Params");
        if (!ignored.empty()) {
          resp->addHeader("X-Ignored-Params", ignored);
          resp->addHeader("X-Deprecation",
                          "unsupported-param-tolerance; removed-in=v4");
        }

        // Retry-After on 503
        if (resp->statusCode() == drogon::k503ServiceUnavailable)
          resp->addHeader("Retry-After", "1");

        // Metrics
        auto path = req->path();
        if (path != "/metrics") {
          auto route = Metrics::route_from_path(path);
          int status = static_cast<int>(resp->statusCode());
          Metrics::instance().record_request(route, status, duration_s);
          Metrics::instance().record_request_size(req->body().size());
        }
      });
}
} // namespace turbo_ocr::server
