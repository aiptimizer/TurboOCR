#include "turbo_ocr/http/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <memory>
#include <optional>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/markdown/markdown_export.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/decode/size_classify.h"
#include "turbo_ocr/decode/jpeg_codec.h"
#include "turbo_ocr/pipeline/jpeg_infer.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/validation/pixel_dims.h"

using turbo_ocr::decode::NvJpegDecoder;

#include "image_internal.h"
#include "size_guards.h"

namespace turbo_ocr::routes {
// --- POST /infer (Tier-B): run ONE crop through a chosen backend ----------
// Body JSON: { "image": "<base64>", "modality": "table"|"formula",
//              "backend": "<registry-name>"  |  { inline BackendSpec } }
// Inline kind:openai (operator-supplied base_url => SSRF surface) is REJECTED
// unless TURBO_ALLOW_ADHOC_BACKENDS=1. Additive: touches no existing route.
void register_infer_route_gpu(server::WorkPool &pool,
                              pipeline::PipelineDispatcher &dispatcher,
                              const server::ImageDecoder &decode,
                              bool nvjpeg_available) {
  const auto rtbl = backend_routing::load_routing_config();
  const std::set<std::string> valid_table   = backend_routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = backend_routing::routable_backend_names(rtbl, "formula");
  drogon::app().registerHandler(
      "/infer",
      [&pool, &dispatcher, &decode, nvjpeg_available, valid_table, valid_formula](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        // All /infer inputs travel in the JSON body; no query params exist.
        {
          server::InferOptions inf_opts;
          server::EndpointSpec spec;
          spec.ocr_options = false;
          spec.routing_unsupported_reason = server::kRoutingUnsupportedEndpoint;
          if (!server::validate_request(req, spec, /*layout_available=*/false,
                                        /*table_available=*/false,
                                        /*formula_available=*/false, {}, {},
                                        &inf_opts, callback))
            return;
        }
        auto json = req->getJsonObject();
        if (!json) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON body"));
          return;
        }
        if (!json->isMember("image") || !(*json)["image"].isString() ||
            (*json)["image"].asString().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "MISSING_IMAGE", "Missing 'image' (base64)"));
          return;
        }
        if (json->isMember("modality") && !(*json)["modality"].isString()) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "'modality' must be a string ('table' or 'formula')"));
          return;
        }
        const std::string modality =
            json->isMember("modality") ? (*json)["modality"].asString() : "";
        if (modality != "table" && modality != "formula") {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "'modality' must be 'table' or 'formula'"));
          return;
        }
        if (!json->isMember("backend")) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "Missing 'backend' (a registered backend name or an inline spec object)"));
          return;
        }

        // Resolve backend: named (registry) or inline spec.
        std::string backend_name;
        std::optional<backend_routing::BackendSpec> inline_spec;
        const auto &be = (*json)["backend"];
        const auto &valid = (modality == "table") ? valid_table : valid_formula;
        if (be.isString()) {
          backend_name = be.asString();
          if (valid.find(backend_name) == valid.end()) {
            callback(server::error_response(drogon::k400BadRequest, "ROUTING_UNKNOWN_OVERRIDE",
                std::format("backend '{}' is not a configured {} backend (see /capabilities)",
                            backend_name, modality)));
            return;
          }
        } else if (be.isObject()) {
          // SSRF gate: an inline openai endpoint lets the caller name an
          // arbitrary base_url. Off by default; the operator opts in per-deploy.
          if (be.isMember("kind") && be["kind"].isString() &&
              be["kind"].asString() == "openai" &&
              !env::env_enabled("TURBO_ALLOW_ADHOC_BACKENDS")) {
            callback(server::error_response(drogon::k403Forbidden, "ADHOC_BACKENDS_DISABLED",
                "inline kind:openai backends are disabled; set TURBO_ALLOW_ADHOC_BACKENDS=1 "
                "to allow operator-supplied endpoint URLs (SSRF surface)"));
            return;
          }
          Json::StreamWriterBuilder w;
          w["indentation"] = "";
          const std::string spec_text = Json::writeString(w, be);
          try {
            inline_spec = backend_routing::parse_inline_backend(modality, spec_text);
          } catch (const backend_routing::RoutingConfigError &e) {
            // .what() begins with the ROUTING_* code; surface it verbatim.
            std::string msg = e.what();
            std::string code = "ROUTING_BAD_KIND";
            if (auto c = msg.find(':'); c != std::string::npos) code = msg.substr(0, c);
            callback(server::error_response(drogon::k400BadRequest, code.c_str(), msg));
            return;
          }
          // Reject inline kind:local: building a local engine on the request
          // thread spins up ORT-CUDA sessions + hundreds of MB of cudaMalloc,
          // unguarded — a resource-exhaustion vector. Local backends must be
          // named (already loaded at startup); only kind:openai (a cheap HTTP
          // client, gated by TURBO_ALLOW_ADHOC_BACKENDS above) may be inline.
          if (inline_spec && inline_spec->kind == backend_routing::Kind::Local) {
            callback(server::error_response(drogon::k400BadRequest, "ADHOC_LOCAL_DISABLED",
                "inline kind:local backends are not allowed; name an already-loaded "
                "backend (see /capabilities) instead"));
            return;
          }
        } else {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
              "'backend' must be a string (name) or an object (inline spec)"));
          return;
        }

        auto b64 = std::make_shared<std::string>((*json)["image"].asString());
        server::submit_work(pool, std::move(callback),
            [b64, &dispatcher, &decode, nvjpeg_available, modality, backend_name, inline_spec](
                server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/infer", [&] {
            auto bytes = std::make_shared<const std::string>(turbo_ocr::base64_decode(*b64));
            if (bytes->empty()) {
              cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64"));
              return;
            }
            // Decompression-bomb guard, same two-stage check as the other
            // image routes: a header sniff before decode, then a post-decode
            // check for formats the sniff can't parse. Without it a
            // 60000x60000 PNG decodes to a ~10 GB host Mat and OOMs the worker.
            const auto *raw = reinterpret_cast<const unsigned char *>(bytes->data());
            if (reject_if_too_large_pre(raw, bytes->size(), cb)) return;
            const bool gpu_jpeg = nvjpeg_available && decode::looks_like_jpeg(raw, bytes->size());
            cv::Mat img;
            if (!gpu_jpeg) {
              img = decode(raw, bytes->size());
              if (img.empty()) {
                cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
                return;
              }
              if (reject_if_too_large_post(img, cb)) return;
            }
            std::string result;
            try {
              // Capture inline_spec by value into the GPU task; pass its address
              // (the worker builds the transient recognizer in its own context).
              // A JPEG crop is decoded on the replica (its own decoder) to the
              // host image the region backends take; `bytes` is shared so an
              // abandoned task never reads freed memory.
              result = dispatcher.submit_for_default(
                  [img, bytes, gpu_jpeg, modality, backend_name, inline_spec](auto &e) {
                    cv::Mat crop = img;
                    if (gpu_jpeg) {
                      crop = pipeline::decode_jpeg_on_replica(
                          e, reinterpret_cast<const unsigned char *>(bytes->data()),
                          bytes->size());
                      decode::throw_if_image_too_large(crop.cols, crop.rows);
                    }
                    return e.pipeline->infer_one(
                        crop, e.stream, modality, backend_name,
                        inline_spec ? &*inline_spec : nullptr);
                  });
            } catch (const turbo_ocr::TimeoutError &) {
              cb(timeout_response());
              return;
            } catch (const turbo_ocr::BackendUnavailableError &e) {
              cb(server::error_response(drogon::k503ServiceUnavailable,
                                        "BACKEND_UNAVAILABLE", e.what()));
              return;
            }
            const char *key = (modality == "table") ? "html" : "latex";
            std::string body = "{\"modality\":\"" + modality + "\",\"" + key + "\":\"";
            turbo_ocr::detail::append_escaped_string(body, result);
            body += "\"}";
            cb(server::json_response(body));
          });
        });
      },
      {drogon::Post});
}


} // namespace turbo_ocr::routes
