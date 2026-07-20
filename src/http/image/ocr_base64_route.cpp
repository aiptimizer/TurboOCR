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

#include "size_guards.h"

namespace turbo_ocr::routes {

void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
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
      [&pool, &infer, &decode, layout_available, valid_table, valid_formula,
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

        auto json = req->getJsonObject();
        if (!json) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
          return;
        }
        if (!json->isMember("image") || !(*json)["image"].isString()
            || (*json)["image"].asString().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "MISSING_IMAGE", "Empty or missing image field"));
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
            callback(server::error_response(drogon::k400BadRequest,
                                             e.code.c_str(), e.message));
            return;
          }
        }

        auto b64_str = std::make_shared<std::string>((*json)["image"].asString());

        server::submit_work(pool, std::move(callback),
            [b64_str, &infer, &decode, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr", [&] {
            std::string decoded_bytes = turbo_ocr::base64_decode(*b64_str);
            if (decoded_bytes.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64"));
              return;
            }

            const auto *bytes = reinterpret_cast<const unsigned char *>(decoded_bytes.data());
            if (reject_if_too_large_pre(bytes, decoded_bytes.size(), cb)) return;

            cv::Mat img = decode(bytes, decoded_bytes.size());
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

} // namespace turbo_ocr::routes
