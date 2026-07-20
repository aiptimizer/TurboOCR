#include "turbo_ocr/http/common_routes.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/validation/request_gate.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>


namespace turbo_ocr::routes {

void register_health_route(std::function<bool()> readiness_check,
                           server::WorkPool *pool) {
  auto health_ok = [](const drogon::HttpRequestPtr &,
                      std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
    callback(server::make_response(drogon::k200OK, "ok"));
  };
  drogon::app().registerHandler("/health", health_ok, {drogon::Get});
  drogon::app().registerHandler("/health/live", health_ok, {drogon::Get});

  // /health/ready — verifies the pipeline is actually responsive. The check
  // may run a real GPU inference (cache miss), so offload it to the WorkPool
  // when one is available; running it inline would block the event-loop
  // thread for the GPU-queue-drain duration.
  auto ready_check = std::make_shared<std::function<bool()>>(std::move(readiness_check));
  auto respond = [](server::DrogonCallback &cb, bool ready) {
    if (ready) cb(server::make_response(drogon::k200OK, "ok"));
    else       cb(server::error_response(drogon::k503ServiceUnavailable,
                                          "NOT_READY", "Pipeline not ready"));
  };
  drogon::app().registerHandler(
      "/health/ready",
      [ready_check, pool, respond](
          const drogon::HttpRequestPtr &,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        const bool has_check = static_cast<bool>(*ready_check);
        if (pool && has_check) {
          // Not submit_work: its pool-exhaustion arm answers 503 SERVER_BUSY,
          // which a readiness prober reads as not-ready and pulls a loaded but
          // healthy pod out of rotation — the exact flap the probe's own
          // last-verdict logic exists to prevent. A full queue means busy,
          // not broken: answer ready without running the probe.
          auto cb = std::make_shared<server::DrogonCallback>(std::move(callback));
          try {
            pool->submit([cb, ready_check, respond]() {
              respond(*cb, (*ready_check)());
            });
          } catch (const turbo_ocr::PoolExhaustedError &) {
            (*cb)(server::make_response(drogon::k200OK, "ok"));
          }
          return;
        }
        // No pool (cheap CPU check) or no check configured: run inline.
        callback(has_check && !(*ready_check)()
                     ? server::error_response(drogon::k503ServiceUnavailable,
                                               "NOT_READY", "Pipeline not ready")
                     : server::make_response(drogon::k200OK, "ok"));
      },
      {drogon::Get});
}

} // namespace turbo_ocr::routes
