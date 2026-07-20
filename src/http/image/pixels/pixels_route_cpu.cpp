#include "turbo_ocr/http/cpu_image_routes.h"

#include <atomic>
#include <format>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/log/stage_profiler.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/validation/pixel_dims.h"
#include "turbo_ocr/validation/request_gate.h"

using turbo_ocr::base64_decode;

namespace turbo_ocr::routes {

void register_ocr_pixels_route_cpu(server::WorkPool &work_pool,
                                   const server::InferFunc &infer,
                                   bool layout_available,
                                   bool table_available,
                                   bool formula_available) {
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&work_pool, &infer, layout_available, table_available,
       formula_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        server::InferOptions opts;
        server::EndpointSpec spec;
        spec.routing = server::kBuildRoutingSupport;
        spec.pixel_dims = true;
        if (!server::validate_request(req, spec, layout_available,
                                      table_available, formula_available,
                                      /*valid_route_table=*/{},
                                      /*valid_route_formula=*/{}, &opts,
                                      callback))
          return;

        // Shared payload validation (pixel_dims.h): dims resolution, per-side
        // and area caps, exact body-size — identical on the GPU handler.
        const auto dims = server::validate_pixel_payload(req);
        if (!dims.ok()) {
          callback(server::error_response(drogon::k400BadRequest,
                                          dims.error_code.c_str(),
                                          dims.error));
          return;
        }
        const bool used_legacy_dim_header = dims.used_legacy_header;

        server::submit_work(work_pool, std::move(callback),
            [req, &infer, dims, opts,
             used_legacy_dim_header](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/pixels", [&] {
            cv::Mat img = server::pixel_body_to_bgr(req, dims);
            auto inf = infer(img, opts);
            auto resp = server::json_response(
                server::emit_infer_result_json(inf, opts.want_blocks));
            if (used_legacy_dim_header)
              server::stamp_pixel_dim_deprecation(resp);
            cb(resp);
          });
        });
      },
      {drogon::Post});
}

} // namespace turbo_ocr::routes
