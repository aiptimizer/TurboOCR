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

#include "../size_guards.h"

namespace turbo_ocr::routes {

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             const capability::CapabilityMask &loaded,
                             const server::EncodedInferFunc &encoded_infer) {
  // Tier-A override name sets, computed once from the same routing config the
  // pipeline loaded. THE single derivation — /ocr, /infer and the gRPC
  // registrar all call the same helper, so a name one transport accepts cannot
  // be one another rejects. Passing EMPTY sets on a kSupported
  // build is not a no-op: it 400s every legal ?route_table=/?route_formula= with
  // ROUTING_UNKNOWN_OVERRIDE while /capabilities lists the backend.
  const server::RoutingNameSets routes = server::routing_name_sets();
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &infer, &decode, &encoded_infer, loaded, routes](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        if (req->body().empty()) {
          callback(server::error_response(server::ErrorCode::kEmptyBody, "Empty body"));
          return;
        }

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
        if (!server::validate_request(req, spec, loaded, routes.table,
                                      routes.formula, &opts, callback))
          return;

        server::submit_work(pool, std::move(callback),
            [req, &infer, &decode, &encoded_infer, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/raw", [&] {
            const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();

            // Device-decode fast path: hand the pipeline the STILL-ENCODED
            // bytes so a backend with an on-device decoder (nvJPEG, vImage)
            // avoids a host decode plus a full-frame upload. The post-decode
            // bomb guard is NOT skipped — it moved behind run_encoded() AND
            // the encoded-infer host-decode fallback (make_infer_func.cpp), so
            // BOTH branches of the deferred decode re-check the real dimensions.
            if (reject_if_too_large_pre(data, len, cb)) return;

            if (encoded_infer) {
              auto inf = encoded_infer(data, len, opts);
              cb(server::json_response(
                  turbo_ocr::server::emit_infer_result_json(inf, opts.want_blocks)));
              return;
            }

            cv::Mat img = decode(data, len);
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

void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             const capability::CapabilityMask &loaded,
                             std::function<bool()> readiness_check,
                             const server::EncodedInferFunc &encoded_infer) {
  // Forward the pool so /health/ready offloads the readiness probe off the
  // event loop — the CPU readiness check calls pool->acquire(), which blocks
  // on a condition variable (unbounded for the CPU pool) when all pipelines
  // are busy; running it inline would wedge a Drogon IO thread under load.
  register_health_route(std::move(readiness_check), &pool);
  register_ocr_base64_route(pool, infer, decode, loaded, encoded_infer);
  register_ocr_raw_route(pool, infer, decode, loaded, encoded_infer);
}

} // namespace turbo_ocr::routes
