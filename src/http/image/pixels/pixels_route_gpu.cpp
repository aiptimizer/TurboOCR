#include "turbo_ocr/http/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/markdown/markdown_export.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/validation/pixel_dims.h"

using turbo_ocr::decode::NvJpegDecoder;

#include "../image_internal.h"

namespace turbo_ocr::routes {
// --- /ocr/pixels: raw BGR pixel data, zero decode overhead ---
void register_ocr_pixels_route_gpu(server::WorkPool &pool,
                                    pipeline::PipelineDispatcher &dispatcher,
                                    bool layout_available,
                                    bool table_available,
                                    bool formula_available) {
  // Same Tier-A routing-override validation set as /ocr/raw — this handler
  // forwards opts.routing_override to the identical run_with_layout call, so
  // omitting the parse here silently dropped a caller's route_table/
  // route_formula while the sibling endpoint honored them.
  const auto rtbl = backend_routing::load_routing_config();
  const std::set<std::string> valid_table   = backend_routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = backend_routing::routable_backend_names(rtbl, "formula");
  // Availability from the warmed pipeline (single source of truth), threaded
  // from main(); not re-derived from config.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&pool, &dispatcher, valid_table, valid_formula, layout_available,
       table_avail, formula_avail](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    server::InferOptions opts;
    server::EndpointSpec spec;
    spec.routing = server::kBuildRoutingSupport;
    spec.pixel_dims = true;
    if (!server::validate_request(req, spec, layout_available, table_avail,
                                  formula_avail, valid_table, valid_formula,
                                  &opts, callback))
      return;

    // Shared payload validation (pixel_dims.h): dims resolution, per-side
    // and area caps, exact body-size — identical on the CPU handler.
    const auto dims = server::validate_pixel_payload(req);
    if (!dims.ok()) {
      callback(server::error_response(drogon::k400BadRequest,
          dims.error_code.c_str(), dims.error));
      return;
    }
    const bool used_legacy_dim_header = dims.used_legacy_header;

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, dims, opts,
         used_legacy_dim_header](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/pixels", [&] {
        cv::Mat img = server::pixel_body_to_bgr(req, dims);

        // C4: a 3-channel `img` is a non-owning view into req->body(), so the
        // task must hold both `img` (by value) and `req` (keeps the pixel
        // buffer alive) to survive an abandoned-on-timeout run.
        pipeline::OcrPipelineResult out;
        try {
          out = dispatcher.submit_for_default([img, req, opts](auto &e) {
            if (!opts.want_text)
              return e.pipeline->run_layout_only(img, e.stream);
            return e.pipeline->run_with_layout(img, e.stream,
                                                opts.want_layout,
                                                opts.want_reading_order,
                                                opts.routing_override,
                                                /*defer_external=*/false,
                                                opts.want_tables,
                                                opts.want_formulas);
          });
        } catch (const turbo_ocr::TimeoutError &) {
          cb(timeout_response());
          return;
        }
        auto resp = server::json_response(
            emit_pipeline_result_json(out, opts.want_blocks));
        if (used_legacy_dim_header)
          server::stamp_pixel_dim_deprecation(resp);
        cb(resp);
      });
    });
  }, {drogon::Post});
}


} // namespace turbo_ocr::routes
