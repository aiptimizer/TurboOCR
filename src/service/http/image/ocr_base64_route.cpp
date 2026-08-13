#include "turbo_ocr/service/http/common_routes.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/image_dims.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/service/validation/request_gate.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>

#include "size_guards.h"

namespace turbo_ocr::routes {

void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
                                const server::ImageDecoder &decode,
                                const capability::CapabilityMask &loaded,
                                const server::EncodedInferFunc &encoded_infer) {
  // Tier-A override validation set (see register_ocr_raw_route_gpu); computed
  // once from the same routing config the pipeline loaded. Availability for the
  // fail-loud gate is passed in (build-specific: what actually loaded), NOT
  // re-derived here — see check_structure_backends.
  const server::RoutingNameSets routes = server::routing_name_sets();
  const std::set<std::string> &valid_table = routes.table;
  const std::set<std::string> &valid_formula = routes.formula;
  drogon::app().registerHandler(
      "/ocr",
      [&pool, &infer, &decode, &encoded_infer, loaded, valid_table, valid_formula](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        // The body is PARSED before validation — the endpoint's flags can
        // arrive in it, and reading the body only afterwards is precisely what
        // made `{"image":..., "layout":true}` return HTTP 200 with no layout.
        // But its ERRORS are reported after: v3.5.0 ran validation first, so a
        // request that is broken in both ways must still get the validation
        // code (an unparseable body simply contributes no flags — nullptr).
        auto json = req->getJsonObject();

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
        spec.routing_unsupported_reason = server::kRoutingUnsupportedCpu;
        if (!server::validate_request(req, spec, loaded, valid_table,
                                      valid_formula, &opts, callback,
                                      /*allow_image_only=*/false,
                                      json ? json.get() : nullptr))
          return;

        if (!json) {
          callback(server::error_response(server::ErrorCode::kInvalidJson, "Invalid JSON"));
          return;
        }
        if (!json->isMember("image") || !(*json)["image"].isString()
            || (*json)["image"].asString().empty()) {
          callback(server::error_response(server::ErrorCode::kMissingImage,
                                          "Empty or missing image field"));
          return;
        }
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
            callback(server::error_response(e.code, e.message));
            return;
          }
        }

        auto b64_str = std::make_shared<std::string>((*json)["image"].asString());

        server::submit_work(pool, std::move(callback),
            [b64_str, &infer, &decode, &encoded_infer, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr", [&] {
            std::string decoded_bytes = turbo_ocr::base64_decode(*b64_str);
            if (decoded_bytes.empty()) {
              cb(server::error_response(server::ErrorCode::kBase64DecodeFailed,
                                        "Failed to decode base64"));
              return;
            }

            const auto *bytes = reinterpret_cast<const unsigned char *>(decoded_bytes.data());
            // Device-decode fast path: hand the pipeline the STILL-ENCODED
            // bytes so a backend with an on-device decoder (nvJPEG, vImage)
            // avoids a host decode plus a full-frame upload. The post-decode
            // bomb guard is NOT skipped — it moved behind run_encoded() AND
            // the encoded-infer host-decode fallback (make_infer_func.cpp), so
            // BOTH branches of the deferred decode re-check the real dimensions.
            if (reject_if_too_large_pre(bytes, decoded_bytes.size(), cb)) return;

            if (encoded_infer) {
              auto inf = encoded_infer(bytes, decoded_bytes.size(), opts);
              cb(server::json_response(
                  turbo_ocr::server::emit_infer_result_json(inf, opts.want_blocks)));
              return;
            }

            cv::Mat img = decode(bytes, decoded_bytes.size());
            if (img.empty()) {
              cb(server::error_response(server::ErrorCode::kImageDecodeFailed,
                                        "Failed to decode image"));
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

} // namespace turbo_ocr::routes
