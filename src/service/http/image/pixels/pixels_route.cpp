#include "turbo_ocr/service/http/image_routes.h"

#include <atomic>
#include <format>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/base/log/stage_profiler.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/image_dims.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/service/validation/pixel_dims.h"
#include "turbo_ocr/service/validation/request_gate.h"

using turbo_ocr::base64_decode;

namespace turbo_ocr::routes {

void register_ocr_pixels_route(server::WorkPool &work_pool,
                                   const server::InferFunc &infer,
                                   const capability::CapabilityMask &loaded) {
  // The REAL configured routing names, same source as every sibling route.
  // This used to pass empty sets while declaring kBuildRoutingSupport, so a
  // route_table/route_formula override that works on /ocr got a 400
  // ROUTING_UNKNOWN_OVERRIDE on the pixel encoding of the same request —
  // every legal override name misses an empty set.
  const server::RoutingNameSets routing = server::routing_name_sets();
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&work_pool, &infer, loaded, routing](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        server::InferOptions opts;
        server::EndpointSpec spec;
        // The seam (make_infer_func::maybe_autorotate) now rotates the page
        // when ?autorotate=1 is requested, so this endpoint genuinely ACTS on
        // DocOrientation — include it in acts_on so the shared gate parses
        // and availability-checks it instead of classifying it as an ignored
        // param. (/ocr/batch keeps the default exclusion: its batched det
        // path has no per-image rotation, and the exclusion surfaces
        // ?autorotate=1 in X-Ignored-Params rather than claiming it.)
        spec.acts_on = turbo_ocr::capability::CapabilityMask::all();
        spec.routing = server::kBuildRoutingSupport;
        spec.pixel_dims = true;
        if (!server::validate_request(req, spec, loaded, routing.table,
                                      routing.formula, &opts, callback))
          return;

        // Shared payload validation (pixel_dims.h): dims resolution, per-side
        // and area caps, exact body-size — identical on the GPU handler.
        const auto dims = server::validate_pixel_payload(req);
        if (!dims.ok()) {
          callback(server::error_response(dims.error_code, dims.error));
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
