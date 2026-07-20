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

#include "../size_guards.h"

namespace turbo_ocr::routes {

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available,
                             bool table_available,
                             bool formula_available) {
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &infer, &decode, layout_available, table_avail, formula_avail](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        if (req->body().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
          return;
        }

        server::InferOptions opts;
        server::EndpointSpec spec;
        spec.routing = server::kBuildRoutingSupport;
        if (!server::validate_request(req, spec, layout_available, table_avail,
                                      formula_avail, /*valid_route_table=*/{},
                                      /*valid_route_formula=*/{}, &opts,
                                      callback))
          return;

        server::submit_work(pool, std::move(callback),
            [req, &infer, &decode, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/raw", [&] {
            const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();

            if (reject_if_too_large_pre(data, len, cb)) return;

            cv::Mat img = decode(data, len);
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }
            if (reject_if_too_large_post(img, cb)) return;

            auto inf = infer(img, opts);
            cb(server::json_response(
                turbo_ocr::server::emit_infer_result_json(inf, opts.want_blocks)));
          });
        });
      },
      {drogon::Post});
}

void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available,
                             bool table_available,
                             bool formula_available,
                             std::function<bool()> readiness_check) {
  // Forward the pool so /health/ready offloads the readiness probe off the
  // event loop — the CPU readiness check calls pool->acquire(), which blocks
  // on a condition variable (unbounded for the CPU pool) when all pipelines
  // are busy; running it inline would wedge a Drogon IO thread under load.
  register_health_route(std::move(readiness_check), &pool);
  register_ocr_base64_route(pool, infer, decode, layout_available,
                            table_available, formula_available);
  register_ocr_raw_route(pool, infer, decode, layout_available,
                         table_available, formula_available);
}

} // namespace turbo_ocr::routes
