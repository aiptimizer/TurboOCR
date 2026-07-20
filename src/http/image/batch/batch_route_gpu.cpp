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
#include "batch_internal.h"

namespace turbo_ocr::routes {

using namespace batchdetail;

void register_ocr_batch_route_gpu(server::WorkPool &pool,
                                   pipeline::PipelineDispatcher &dispatcher,
                                   const server::ImageDecoder &decode,
                                   bool nvjpeg_available,
                                   bool layout_available,
                                   bool table_available,
                                   bool formula_available,
                                   int max_batch_images) {
  // Same Tier-A routing-override validation set as /ocr/raw: the batch
  // pipeline forwards opts.routing_override into every chunk, so the batch
  // endpoint honors the same per-request backend selection as its siblings.
  const auto rtbl = backend_routing::load_routing_config();
  const std::set<std::string> valid_table   = backend_routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = backend_routing::routable_backend_names(rtbl, "formula");
  // Availability from the warmed pipeline (single source of truth), threaded
  // from main(); not re-derived from config.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/batch",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available,
       valid_table, valid_formula, table_avail, formula_avail,
       max_batch_images](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    server::InferOptions opts;
    server::EndpointSpec spec;
    spec.routing = server::kBuildRoutingSupport;
    // The batched det/rec path has no layout-only equivalent; the gate
    // rejects text=0 with a redirect instead of silently running full OCR.
    spec.layout_only_allowed = false;
    if (!server::validate_request(req, spec, layout_available, table_avail,
                                  formula_avail, valid_table, valid_formula,
                                  &opts, callback))
      return;

    auto json = req->getJsonObject();
    if (!json) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
      return;
    }
    if (!json->isMember("images") || !(*json)["images"].isArray()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Missing images array"));
      return;
    }

    auto &images_json = (*json)["images"];
    size_t n = images_json.size();
    if (n == 0) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BATCH", "Empty images array"));
      return;
    }
    // Cap before the O(n) per-slot allocations + n*1KB JSON reserve below —
    // an unbounded images[] (tens of millions of tiny strings in a <100MB
    // body) is otherwise a memory-amplification OOM lever.
    if (n > static_cast<size_t>(max_batch_images)) {
      callback(server::error_response(drogon::k400BadRequest, "BATCH_TOO_LARGE",
          std::format("images array has {} entries, max is {}", n, max_batch_images)));
      return;
    }

    // asString() throws Json::LogicError on a non-string element ({}/[]);
    // guard the type so a malformed slot becomes an empty string the per-slot
    // decode path tags, instead of an unhandled exception / 500.
    auto b64_strings = std::make_shared<std::vector<std::string>>(n);
    for (size_t i = 0; i < n; ++i) {
      const auto &el = images_json[static_cast<int>(i)];
      if (el.isString())
        (*b64_strings)[i] = el.asString();
    }

    server::submit_work(pool, std::move(callback),
        [b64_strings, n, &dispatcher, &decode, nvjpeg_available, opts](server::DrogonCallback &cb) {
      const bool want_layout = opts.want_layout;
      server::run_with_error_handling(cb, "/ocr/batch", [&] {
        // Per-slot error tags for the response. Empty string == success.
        // Mirrors the CPU /ocr/batch contract: callers see which slots
        // failed (empty / decode_failed / dimensions_too_large) instead
        // of getting a silently-zero result list.
        std::vector<std::string> errors(n);
        std::vector<std::string> raw_bytes(n);

        batch_decode_base64(*b64_strings, raw_bytes, errors);

        const int kMaxImageDim = decode::max_image_dim();
        batch_check_dims_pre(raw_bytes, kMaxImageDim, errors);

        std::vector<cv::Mat> imgs(n);
        batch_decode_images(raw_bytes, nvjpeg_available, decode, imgs, errors);
        batch_check_dims_post(imgs, kMaxImageDim, errors);

        std::vector<cv::Mat> valid_imgs;
        std::vector<size_t> valid_indices;
        valid_imgs.reserve(n);
        valid_indices.reserve(n);
        for (size_t i = 0; i < n; ++i) {
          if (errors[i].empty() && !imgs[i].empty()) {
            valid_imgs.push_back(std::move(imgs[i]));
            valid_indices.push_back(i);
          }
        }

        std::vector<BatchItem> all_items(n);
        // C4: hand ownership of the decoded images to the submitted task so an
        // abandoned-on-timeout run never touches this frame's vectors.
        try {
          batch_run_pipeline(dispatcher, std::move(valid_imgs),
                              std::move(valid_indices),
                              want_layout, opts, all_items, errors);
        } catch (const turbo_ocr::TimeoutError &) {
          cb(timeout_response());
          return;
        }

        cb(server::json_response(batch_emit_json(all_items, errors, want_layout, opts.want_blocks)));
      });
    });
  }, {drogon::Post});
}



} // namespace turbo_ocr::routes
