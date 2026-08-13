// POST /infer (Tier-B) — run ONE crop through a chosen table/formula backend.
//
// RESTORED. Deleted with src/cuda/ (as src/cuda/http/infer_route_gpu.cpp) when
// the duplicate CUDA-native HTTP layer was removed. It was a ROUTE, not CUDA
// plumbing. UnifiedOcrPipeline::infer_one — which this is the only caller of —
// survived on every backend and even names "/infer" in its own diagnostics, so
// the pipeline has been carrying a half of this feature with nothing to call it.
//
// Body JSON: { "image": "<base64>", "modality": "table"|"formula",
//              "backend": "<registry-name>" | { inline BackendSpec } }
//
// Inline kind:openai (operator-supplied base_url => SSRF surface) is REJECTED
// unless TURBO_ALLOW_ADHOC_BACKENDS=1. Every security gate below is carried over
// verbatim — they were transport policy, not device policy.
#include "turbo_ocr/service/http/common_routes.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>
#include <set>

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/base/encoding.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/service/validation/request_gate.h"

#include "size_guards.h"

namespace turbo_ocr::routes {

// THE ADMISSION HALF of POST /infer. Same split as /ocr/batch and /ocr/stream:
// every rejection below is a synchronous 400 on the event loop, and what it
// returns is the only state the worker needs. /infer is the endpoint with the
// most parsing per request — modality, backend name, an optional INLINE backend
// spec — so it is the one where reading validation and execution as a single
// block cost the most.
//
// Returns nullopt having ALREADY answered `callback`.
struct InferRequest {
  std::string modality;
  std::string backend_name;
  std::optional<backend_routing::BackendSpec> inline_spec;
  std::shared_ptr<std::string> b64;
};

std::optional<InferRequest>
admit_infer_request(const drogon::HttpRequestPtr &req,
                    const std::set<std::string> &valid_table,
                    const std::set<std::string> &valid_formula,
                    server::DrogonCallback &callback) {
  // All /infer inputs travel in the JSON body; no query params exist.
  {
    server::InferOptions inf_opts;
    server::EndpointSpec spec;
    spec.ocr_options = false;
    spec.routing_unsupported_reason = server::kRoutingUnsupportedEndpoint;
    if (!server::validate_request(req, spec,
                                  /*loaded=*/capability::CapabilityMask::none(),
                                  {}, {}, &inf_opts, callback))
      return std::nullopt;
  }
  auto json = req->getJsonObject();
  if (!json) {
    callback(server::error_response(server::ErrorCode::kInvalidJson, "Invalid JSON body"));
    return std::nullopt;
  }
  if (!json->isMember("image") || !(*json)["image"].isString() ||
      (*json)["image"].asString().empty()) {
    callback(server::error_response(server::ErrorCode::kMissingImage, "Missing 'image' (base64)"));
    return std::nullopt;
  }
  if (json->isMember("modality") && !(*json)["modality"].isString()) {
    callback(server::error_response(server::ErrorCode::kInvalidParameter,
                                    "'modality' must be a string ('table' or 'formula')"));
    return std::nullopt;
  }
  const std::string modality =
      json->isMember("modality") ? (*json)["modality"].asString() : "";
  if (modality != "table" && modality != "formula") {
    callback(server::error_response(server::ErrorCode::kInvalidParameter,
                                    "'modality' must be 'table' or 'formula'"));
    return std::nullopt;
  }
  if (!json->isMember("backend")) {
    callback(server::error_response(server::ErrorCode::kInvalidParameter,
        "Missing 'backend' (a registered backend name or an inline spec "
                                    "object)"));
    return std::nullopt;
  }

  // Resolve backend: named (registry) or inline spec.
  std::string backend_name;
  std::optional<backend_routing::BackendSpec> inline_spec;
  const auto &be = (*json)["backend"];
  const auto &valid = (modality == "table") ? valid_table : valid_formula;
  if (be.isString()) {
    backend_name = be.asString();
    if (valid.find(backend_name) == valid.end()) {
      callback(server::error_response(
          server::ErrorCode::kRoutingUnknownOverride,
          std::format("backend '{}' is not a configured {} backend (see "
                      "/capabilities)",
                      backend_name, modality)));
      return std::nullopt;
    }
  } else if (be.isObject()) {
    // SSRF gate: an inline openai endpoint lets the caller name an
    // arbitrary base_url. Off by default; the operator opts in per-deploy.
    if (be.isMember("kind") && be["kind"].isString() &&
        be["kind"].asString() == "openai" &&
        !env::env_enabled("TURBO_ALLOW_ADHOC_BACKENDS")) {
      callback(server::error_response(server::ErrorCode::kAdhocBackendsDisabled,
                                      "inline kind:openai backends are disabled; set "
                                      "TURBO_ALLOW_ADHOC_BACKENDS=1 to allow operator-supplied "
                                      "endpoint URLs (SSRF surface)"));
      return std::nullopt;
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
      if (auto c = msg.find(':'); c != std::string::npos)
        code = msg.substr(0, c);
      callback(server::error_response(code, msg));
      return std::nullopt;
    }
    // Reject inline kind:local: building a local engine on the request
    // thread spins up inference sessions and hundreds of MB of device
    // allocation, unguarded — a resource-exhaustion vector. Local backends
    // must be named (already loaded at startup); only kind:openai (a cheap
    // HTTP client, gated above) may be inline.
    if (inline_spec && inline_spec->kind == backend_routing::Kind::Local) {
      callback(server::error_response(server::ErrorCode::kAdhocLocalDisabled,
                                      "inline kind:local backends are not allowed; name an "
                                      "already-loaded backend (see /capabilities) instead"));
      return std::nullopt;
    }
  } else {
    callback(server::error_response(server::ErrorCode::kInvalidParameter,
        "'backend' must be a string (name) or an object (inline spec)"));
    return std::nullopt;
  }

  auto b64 = std::make_shared<std::string>((*json)["image"].asString());
  return InferRequest{std::move(modality), std::move(backend_name),
                      std::move(inline_spec), std::move(b64)};
}


void register_infer_route(server::WorkPool &pool,
                          const server::InferOneFunc &infer_one,
                          const server::ImageDecoder &decode) {
  // Nothing to serve without the seam (no pipeline pool — the offline drivers).
  if (!infer_one) return;
  const server::RoutingNameSets routes = server::routing_name_sets();
  const std::set<std::string> &valid_table = routes.table;
  const std::set<std::string> &valid_formula = routes.formula;
  drogon::app().registerHandler(
      "/infer",
      [&pool, &infer_one, &decode, valid_table, valid_formula](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        auto admitted = admit_infer_request(req, valid_table, valid_formula,
                                            callback);
        if (!admitted) return;
        const std::string modality = std::move(admitted->modality);
        const std::string backend_name = std::move(admitted->backend_name);
        auto inline_spec = std::move(admitted->inline_spec);
        auto b64 = std::move(admitted->b64);

        server::submit_work(
            pool, std::move(callback),
            [b64, &infer_one, &decode, modality, backend_name, inline_spec](
                server::DrogonCallback &cb) {
              server::run_with_error_handling(cb, "/infer", [&] {
                std::string bytes = turbo_ocr::base64_decode(*b64);
                if (bytes.empty()) {
                  cb(server::error_response(server::ErrorCode::kBase64DecodeFailed,
                                            "Failed to decode base64"));
                  return;
                }
                // Decompression-bomb guard, the same two-stage check the other
                // image routes apply: a header sniff before decode, then a
                // post-decode check for formats the sniff cannot parse. Without
                // it a 60000x60000 PNG decodes to a ~10 GB host Mat and OOMs the
                // worker.
                const auto *raw =
                    reinterpret_cast<const unsigned char *>(bytes.data());
                if (reject_if_too_large_pre(raw, bytes.size(), cb)) return;
                cv::Mat img = decode(raw, bytes.size());
                if (img.empty()) {
                  cb(server::error_response(server::ErrorCode::kImageDecodeFailed,
                                            "Failed to decode image"));
                  return;
                }
                if (reject_if_too_large_post(img, cb)) return;

                std::string result;
                try {
                  result = infer_one(img, modality, backend_name,
                                     inline_spec ? &*inline_spec : nullptr);
                } catch (const turbo_ocr::BackendUnavailableError &e) {
                  cb(server::error_response(server::ErrorCode::kBackendUnavailable, e.what()));
                  return;
                }
                const char *key = (modality == "table") ? "html" : "latex";
                std::string body =
                    "{\"modality\":\"" + modality + "\",\"" + key + "\":\"";
                turbo_ocr::detail::append_escaped_string(body, result);
                body += "\"}";
                cb(server::json_response(body));
              });
            });
      },
      {drogon::Post});
}

} // namespace turbo_ocr::routes
