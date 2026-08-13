#include "turbo_ocr/service/http/common_routes.h"
#include "turbo_ocr/service/grpc/grpc_response_mode.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/image_dims.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/service/validation/request_gate.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>


namespace turbo_ocr::routes {

CapabilitiesInfo make_capabilities_info(
    const server::ServerConfig &cfg, bool is_gpu,
    const capability::CapabilityMask &loaded, bool profile_endpoint,
    bool honored_auto_verified, int pdf_default_dpi) {
  CapabilitiesInfo caps;
  caps.is_gpu = is_gpu;
  caps.loaded = loaded;
  caps.profile_endpoint = profile_endpoint;
  caps.grpc_response_mode =
      std::string(server::detail::grpc_mode_str(cfg.grpc_response_mode));
  caps.honored_auto_verified = honored_auto_verified;
  caps.pdf_default_dpi = pdf_default_dpi;
  caps.max_pdf_pages = cfg.max_pdf_pages;
  caps.max_body_mb = cfg.max_body_mb;
  caps.max_image_dim = cfg.max_image_dim;
  caps.max_batch_images = cfg.max_batch_images;
  return caps;
}

std::string build_capabilities_json(const CapabilitiesInfo &info) {
  // Build the stable advertisement once at startup — nothing here is
  // request- or runtime-varying (the availability bools are fixed once
  // startup finished). The SAME document is served by GET /capabilities and
  // carried in the gRPC HealthResponse.capabilities_json, so the two
  // transports can never advertise different capabilities.
  std::string body;
  body.reserve(640);
  const auto b = [](bool v) { return v ? "true" : "false"; };

  body += R"({"build":")";
  body += info.is_gpu ? "gpu" : "cpu";
  body += R"(","features":{)";
  // One entry per capability, in table order — including the FALSE ones, which
  // are the whole point: a client must be able to see that a capability exists
  // but is not loaded here, and tell that apart from one this build never had.
  {
    bool first = true;
    for (const auto &cap : capability::kCapabilities) {
      if (!first) body += ',';
      first = false;
      body += '"'; body.append(cap.name); body += R"(":)";
      body += b(info.loaded.get(cap.id));
    }
  }
  // profile_endpoint and grpc_response_mode are NOT capabilities — they are
  // facts about this build that have always lived in `features`. Kept here for
  // wire compatibility, appended after the generated rows.
  body += R"(,"profile_endpoint":)"; body += b(info.profile_endpoint);
  body += R"(,"grpc_response_mode":")";
  body += info.grpc_response_mode;
  body += R"("})";
  // The IMPLEMENTED axis, same capability keys and nothing else.
  // "supported but not loaded" is a config problem an operator can fix;
  // "not supported" is not — reporting only `features` conflates the two and
  // leaves them with no idea which knob to reach for.
  body += R"(,"supported":{)";
  {
    bool first = true;
    for (const auto &cap : capability::kCapabilities) {
      if (!first) body += ',';
      first = false;
      body += '"'; body.append(cap.name); body += R"(":)";
      body += b(info.implemented.get(cap.id));
    }
  }
  body += R"(})";
  // Which backend actually came up. Previously reachable only via the separate
  // GET /capabilities/backend endpoint (see server_main.cpp).
  body += R"(,"backend":")";
  body += info.backend_name;
  body += R"(","device":")";
  body += info.device_name;
  body += R"(","engine_mode":")";
  body += info.engine_mode;
  body += R"(","has_native_engine":)";
  body += b(info.has_native_engine);
  body += R"(,"has_onnx_engine":)";
  body += b(info.has_onnx_engine);
  body += R"(,"pdf":{"modes":[)";
  // auto_verified is advertised only when the build runs it as its own path
  // (GPU). The CPU build aliases it to auto, so listing it there would be a
  // false promise of verification it doesn't perform.
  body += R"("ocr","geometric","auto")";
  if (info.honored_auto_verified) body += R"(,"auto_verified")";
  body += R"(],"default_dpi":)";  body += std::to_string(info.pdf_default_dpi);
  body += R"(,"max_pages":)";      body += std::to_string(info.max_pdf_pages);
  body += R"(},"limits":{"max_body_mb":)";
  body += std::to_string(info.max_body_mb);
  body += R"(,"max_image_dim":)";  body += std::to_string(info.max_image_dim);
  body += R"(,"max_batch_images":)"; body += std::to_string(info.max_batch_images);
  // Every endpoint the unified server registers, in stable order.
  //
  // /ocr/markdown and /infer are back here because the ROUTES are back. They had
  // been dropped from this list when the CUDA-native server's route set was
  // deleted and its endpoints were not ported — and removing them from the
  // contract, rather than restoring the routes, is what made the regression
  // invisible: /capabilities stayed self-consistent while the server quietly
  // served three fewer endpoints than it had the release before.
  //
  // Both run on the device-agnostic InferFunc/InferOneFunc seam, so every
  // backend serves them; the GPU flag that used to gate them was an artifact of
  // their old home, not a property of the feature.
  body += R"(},"endpoints":["/health","/health/live","/health/ready",)"
          R"("/metrics","/capabilities","/ocr","/ocr/raw","/ocr/batch",)"
          R"("/ocr/pixels","/ocr/pdf","/ocr/markdown","/infer","/ocr/stream",)"
          R"("/capabilities/backend")";
  if (info.profile_endpoint) body += R"(,"/profile")";
  body += "]";

  // Resolved routing table for operator introspection. NAMES + kinds only —
  // never base_url/api_key/model: secrets must not serialize (verify audit).
  // Read-only; load_routing_config() already succeeded at pipeline load by the
  // time this registers, but guard anyway so a bad config can't crash startup.
  body += R"(,"routing":)";
  try {
    // Operator-supplied routing.json backend/route names may contain " or \,
    // which would produce malformed JSON here. Route every name through the
    // shared JSON-string escaper (quotes written explicitly around it).
    const auto esc = [](const std::string &s) {
      std::string q;
      turbo_ocr::detail::append_escaped_string(q, s);
      return q;
    };
    const auto tbl = backend_routing::load_routing_config();
    body += R"({"routes":{)";
    bool first = true;
    for (const auto &kv : tbl.routes) {
      if (!first) body += ",";
      first = false;
      body += "\"" + esc(kv.first) + "\":\"" + esc(kv.second) + "\"";
    }
    body += R"(},"backends":{)";
    first = true;
    for (const auto &kv : tbl.backends) {
      if (!first) body += ",";
      first = false;
      body += "\"" + esc(kv.first) + R"(":{"kind":")" +
              std::string(backend_routing::kind_name(kv.second.kind)) + "\"}";
    }
    body += "}}";
  } catch (const std::exception &) {
    body += R"({"error":"invalid"})";
  }
  body += "}";
  return body;
}

void register_capabilities_route(const CapabilitiesInfo &info,
                                 std::string prebuilt_json) {
  // Serve the caller's prebuilt document when given — the SAME bytes the gRPC
  // HealthResponse carries — instead of re-deriving (see the header note).
  auto shared = std::make_shared<std::string>(
      prebuilt_json.empty() ? build_capabilities_json(info)
                            : std::move(prebuilt_json));
  drogon::app().registerHandler(
      "/capabilities",
      [shared](const drogon::HttpRequestPtr &,
               std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        callback(server::json_response(*shared));
      },
      {drogon::Get});
}

} // namespace turbo_ocr::routes
