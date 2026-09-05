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
#include "turbo_ocr/pipeline/jpeg_infer.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/validation/pixel_dims.h"

using turbo_ocr::decode::NvJpegDecoder;

#include "../image_internal.h"
#include "../size_guards.h"

namespace turbo_ocr::routes {
// --- /ocr/raw: GPU-direct JPEG decode, Wuffs PNG ---
void register_ocr_raw_route_gpu(server::WorkPool &pool,
                                 pipeline::PipelineDispatcher &dispatcher,
                                 const server::ImageDecoder &decode,
                                 bool nvjpeg_available,
                                 bool layout_available,
                                 bool table_available,
                                 bool formula_available) {
  // Per-request routing override (Tier-A) validation set: computed ONCE from
  // the same routing config the pipeline loaded, so route-layer validation and
  // the pipeline's recognizer registry never drift (both via
  // backend_routing::routable_backend_names). Captured by value into the handler.
  const auto rtbl = backend_routing::load_routing_config();
  const std::set<std::string> valid_table   = backend_routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = backend_routing::routable_backend_names(rtbl, "formula");
  // Fail-loud availability comes from the warmed pipeline (single source of
  // truth), threaded down from main(); not re-derived from config here.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available,
       valid_table, valid_formula, table_avail, formula_avail](
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
                                  formula_avail, valid_table, valid_formula,
                                  &opts, callback))
      return;

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, &decode, nvjpeg_available, opts](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/raw", [&] {
        const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
        size_t len = req->body().size();

        // Pre-decode header sniff (PNG IHDR / JPEG SOFn). Refuses
        // decompression bombs (per-side AND total pixel area) without ever
        // calling the decoder. Cap knobs: MAX_IMAGE_DIM / MAX_IMAGE_PIXELS_MP,
        // one predicate for every image route (decode/size_classify.h).
        if (reject_if_too_large_pre(data, len, cb)) return;

        // JPEG with nvJPEG: submit GPU-direct decode + infer as one work item
        // (pipeline::decode_jpeg_and_run — shared with every JPEG path).
        // Layout-only requests (?text=0) take it too: run_layout_only has a
        // device-image overload, so no host copy of the pixels is ever made.
        if (nvjpeg_available && NvJpegDecoder::is_jpeg(data, len)) {
          pipeline::JpegRunOpts run_opts{
              .want_layout = opts.want_layout,
              .want_reading_order = opts.want_reading_order,
              .want_tables = opts.want_tables,
              .want_formulas = opts.want_formulas,
              .routing = opts.routing_override,
              // defer_external=true: remote formula/table crops are submitted
              // non-blocking on the GPU worker, awaited later in
              // finalize_deferred() off the worker. Local backends ignore it.
              .defer_external = true,
              .layout_only = !opts.want_text,
          };
          // C4: capture `req` by value so the JPEG bytes (data/len point into
          // req->body()) stay alive if the future is abandoned on timeout and
          // the task keeps running after this handler returns.
          run_default_and_emit(
              dispatcher,
              [req, data, len, run_opts](auto &e) {
                return pipeline::decode_jpeg_and_run(e, data, len, run_opts);
              },
              opts.want_blocks, cb);
          return;
        }

        // Non-JPEG (PNG, etc.) or nvJPEG not available
        cv::Mat img = decode(data, len);
        if (img.empty()) {
          cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
          return;
        }
        // Post-decode safety net for formats we didn't sniff (BMP/TIFF/WEBP).
        if (reject_if_too_large_post(img, cb)) return;

        // C4: capture `img` BY VALUE (cv::Mat copy is a cheap refcount bump)
        // so an abandoned-on-timeout task never dereferences the freed local.
        run_default_and_emit(
            dispatcher,
            [img, opts](auto &e) {
              if (!opts.want_text)
                return e.pipeline->run_layout_only(img, e.stream);
              return e.pipeline->run_with_layout(img, e.stream,
                                                 opts.want_layout,
                                                 opts.want_reading_order,
                                                 opts.routing_override,
                                                 /*defer_external=*/true,
                                                 opts.want_tables,
                                                 opts.want_formulas);
            },
            opts.want_blocks, cb);
      });
    });
  }, {drogon::Post});
}


} // namespace turbo_ocr::routes
