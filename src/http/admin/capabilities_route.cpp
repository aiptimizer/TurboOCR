#include "turbo_ocr/http/common_routes.h"
#include "turbo_ocr/grpc/grpc_response_mode.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/validation/request_gate.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>


namespace turbo_ocr::routes {

CapabilitiesInfo make_capabilities_info(
    const server::ServerConfig &cfg, bool is_gpu, bool layout_available,
    bool table_available, bool formula_available, bool autorotate_available,
    bool profile_endpoint, bool honored_auto_verified, int pdf_default_dpi) {
  CapabilitiesInfo caps;
  caps.is_gpu = is_gpu;
  caps.layout_available = layout_available;
  caps.table_available = table_available;
  caps.formula_available = formula_available;
  caps.autorotate_available = autorotate_available;
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
  body += R"("layout":)";          body += b(info.layout_available);
  body += R"(,"tables":)";         body += b(info.table_available);
  body += R"(,"formulas":)";       body += b(info.formula_available);
  body += R"(,"autorotate":)";     body += b(info.autorotate_available);
  body += R"(,"profile_endpoint":)"; body += b(info.profile_endpoint);
  body += R"(,"grpc_response_mode":")";
  body += info.grpc_response_mode;
  body += R"("},"pdf":{"modes":[)";
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
  // Endpoints both builds register, in stable order. Build-specific endpoints are
  // appended only when this build registered them: /ocr/markdown is GPU-only
  // (register_ocr_markdown_route_gpu), /profile is CPU-only.
  body += R"(},"endpoints":["/health","/health/live","/health/ready",)"
          R"("/metrics","/capabilities","/ocr","/ocr/raw","/ocr/batch",)"
          R"("/ocr/pixels","/ocr/pdf")";
  if (info.is_gpu) body += R"(,"/ocr/markdown","/infer","/ocr/stream")";
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

void register_capabilities_route(const CapabilitiesInfo &info) {
  auto shared = std::make_shared<std::string>(build_capabilities_json(info));
  drogon::app().registerHandler(
      "/capabilities",
      [shared](const drogon::HttpRequestPtr &,
               std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        callback(server::json_response(*shared));
      },
      {drogon::Get});
}

} // namespace turbo_ocr::routes
