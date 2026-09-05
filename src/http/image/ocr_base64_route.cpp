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
#include <memory>
#include <string_view>

#include "turbo_ocr/decode/jpeg_codec.h"
#include "turbo_ocr/decode/json_image_field.h"

#include "size_guards.h"

namespace turbo_ocr::routes {

namespace {

// One request body's work, shared by the scanner fast path and the parsed
// path: base64 -> bytes -> (JPEG on the replica | host decode + infer) -> JSON.
void run_base64_image(std::string_view b64, const server::InferFunc &infer,
                      const server::JpegInferFunc &jpeg_infer,
                      const server::ImageDecoder &decode,
                      const server::InferOptions &opts, server::DrogonCallback &cb) {
  std::string decoded_bytes = turbo_ocr::base64_decode(b64);
  if (decoded_bytes.empty()) {
    cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64"));
    return;
  }
  const auto *bytes = reinterpret_cast<const unsigned char *>(decoded_bytes.data());
  if (reject_if_too_large_pre(bytes, decoded_bytes.size(), cb)) return;

  // JPEG takes the same GPU-direct path as /ocr/raw: decoded on the replica
  // that runs inference, no host pixel buffer on this thread, identical
  // results across the two routes by construction.
  if (jpeg_infer && decode::looks_like_jpeg(bytes, decoded_bytes.size())) {
    auto owned = std::make_shared<const std::string>(std::move(decoded_bytes));
    auto inf = jpeg_infer(owned, opts);
    cb(server::json_response(
        turbo_ocr::server::emit_infer_result_json(inf, opts.want_blocks)));
    return;
  }

  cv::Mat img = decode(bytes, decoded_bytes.size());
  if (img.empty()) {
    cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
    return;
  }
  if (reject_if_too_large_post(img, cb)) return;

  auto inf = infer(img, opts);
  cb(server::json_response(
      turbo_ocr::server::emit_infer_result_json(inf, opts.want_blocks)));
}

} // namespace

// `jpeg_infer` is copied into the handler (std::function copy, once at
// registration): callers may pass a temporary (the CPU build passes an empty
// one), and a reference to it would dangle by the first request.
void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
                                const server::JpegInferFunc &jpeg_infer,
                                const server::ImageDecoder &decode,
                                bool layout_available,
                                bool table_available,
                                bool formula_available) {
  // Tier-A override validation set (see register_ocr_raw_route_gpu); computed
  // once from the same routing config the pipeline loaded. Availability for the
  // fail-loud gate is passed in (build-specific: what actually loaded), NOT
  // re-derived here — see check_structure_backends.
  const auto rtbl = backend_routing::load_routing_config();
  const std::set<std::string> valid_table   = backend_routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = backend_routing::routable_backend_names(rtbl, "formula");
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr",
      [&pool, &infer, jpeg_infer, &decode, layout_available, valid_table, valid_formula,
       table_avail, formula_avail](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        server::InferOptions opts;
        server::EndpointSpec spec;
        spec.routing = server::kBuildRoutingSupport;
        spec.routing_unsupported_reason = server::kRoutingUnsupportedCpu;
        if (!server::validate_request(req, spec, layout_available, table_avail,
                                      formula_avail, valid_table, valid_formula,
                                      &opts, callback))
          return;

        // Fast path: read the base64 text straight out of the request body.
        // The scanner refuses anything it is not sure about (escapes, odd
        // shapes, a routing member), and the full parser below takes over.
        // A multi-MB image then exists once on this thread (its raw bytes)
        // instead of also as a Json::Value copy of the text.
        if (auto fast = decode::find_json_image_field(req->body());
            fast && !fast->has_routing) {
          const std::string_view b64 = fast->base64;
          server::submit_work(pool, std::move(callback),
              [req, b64, &infer, &jpeg_infer, &decode, opts](server::DrogonCallback &cb) {
            server::run_with_error_handling(cb, "/ocr", [&] {
              run_base64_image(b64, infer, jpeg_infer, decode, opts, cb);
            });
          });
          return;
        }

        auto json = req->getJsonObject();
        if (!json) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
          return;
        }
        // Borrow the base64 text from the parsed JSON rather than copying
        // it: the body already exists twice (request buffer, Json::Value),
        // and each extra copy of a multi-MB string is host memory the
        // allocator keeps as a high-water mark on this thread's arena.
        const char *b64_begin = nullptr;
        const char *b64_end = nullptr;
        if (!json->isMember("image") || !(*json)["image"].isString()
            || !(*json)["image"].getString(&b64_begin, &b64_end)
            || b64_begin == b64_end) {
          callback(server::error_response(drogon::k400BadRequest, "MISSING_IMAGE", "Empty or missing image field"));
          return;
        }
        const std::string_view b64(b64_begin, static_cast<size_t>(b64_end - b64_begin));
        // Tier-A: optional JSON `routing:{table,formula}` (backend NAMEs).
        // Validated against the registered backends; unknown => 400.
        {
          std::string rt, rf;
          if (json->isMember("routing") && (*json)["routing"].isObject()) {
            const auto &ro = (*json)["routing"];
            if (ro.isMember("table") && ro["table"].isString())   rt = ro["table"].asString();
            if (ro.isMember("formula") && ro["formula"].isString()) rf = ro["formula"].asString();
          }
          // Same single core policy as the query params: on the CPU build a
          // non-empty override is a loud reject (no routing plumbing there),
          // on GPU it validates against the registered backend names.
          if (auto e = server::apply_routing_override(
                  rt, rf, spec, valid_table, valid_formula,
                  &opts.routing_override);
              !e.ok()) {
            callback(server::error_response(drogon::k400BadRequest,
                                             e.code.c_str(), e.message));
            return;
          }
        }

        // `json` keeps the text alive for the duration of the work item.
        server::submit_work(pool, std::move(callback),
            [json, b64, &infer, &jpeg_infer, &decode, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr", [&] {
            run_base64_image(b64, infer, jpeg_infer, decode, opts, cb);
          });
        });
      },
      {drogon::Post});
}

} // namespace turbo_ocr::routes
