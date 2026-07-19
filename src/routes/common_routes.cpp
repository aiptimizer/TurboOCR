#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/routing/routing_config.h"
#include "turbo_ocr/common/serialization.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>

namespace turbo_ocr::routes {

namespace {
using decode::max_image_dim;

// L3 strict-query-params (opt-in). Default OFF preserves the historical lenient
// behavior (unknown params silently ignored). When
// TURBO_OCR_STRICT_QUERY_PARAMS=1, an unrecognized query parameter is a 400
// INVALID_PARAMETER instead. Read once; cached for the process lifetime.
[[nodiscard]] bool strict_query_params_enabled() noexcept {
  static const bool v = server::env_enabled("TURBO_OCR_STRICT_QUERY_PARAMS");
  return v;
}

// When strict mode is on, reject the request if any query parameter is not in
// `allowed`. Returns true (and invokes `callback`) on rejection. No-op (returns
// false) when strict mode is off, so default behavior is byte-identical.
// NOTE: /ocr takes a JSON body and /ocr/raw a binary body — neither populates
// x-www-form-urlencoded form params, so Drogon's getParameters() is the query
// string only here.
[[nodiscard]] bool reject_unknown_query_params(
    const drogon::HttpRequestPtr &req,
    std::initializer_list<std::string_view> allowed,
    server::DrogonCallback &callback) {
  if (!strict_query_params_enabled()) return false;
  for (const auto &kv : req->getParameters()) {
    const std::string &key = kv.first;
    bool known = false;
    for (auto a : allowed)
      if (a == key) { known = true; break; }
    if (!known) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
          std::format("Unknown query parameter '{}' "
                      "(TURBO_OCR_STRICT_QUERY_PARAMS=1)", key)));
      return true;
    }
  }
  return false;
}

// Two-stage check for /ocr and /ocr/raw:
//   1. Pre-decode header sniff (PNG / JPEG): refuses oversized inputs
//      without ever calling the decoder, defending against decompression
//      bombs (a 1 KB PNG can claim 100k×100k → 30 GB decode buffer).
//   2. Post-decode check on the resulting cv::Mat: catches formats we
//      don't sniff (BMP, TIFF, WEBP).
// Returns true if the request was rejected (caller should return).
[[nodiscard]] inline bool reject_if_too_large_pre(
    const unsigned char *data, size_t len, server::DrogonCallback &cb) {
  if (auto d = decode::peek_image_dimensions(data, len)) {
    if (d->width > max_image_dim() || d->height > max_image_dim()) {
      cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
          std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                      d->width, d->height, max_image_dim(), max_image_dim())));
      return true;
    }
    if (decode::exceeds_pixel_cap(d->width, d->height)) {
      cb(server::error_response(drogon::k400BadRequest, "PIXELS_TOO_LARGE",
          std::format("Image area {}x{} exceeds maximum of {} pixels",
                      d->width, d->height, decode::max_image_pixels())));
      return true;
    }
  }
  return false;
}
[[nodiscard]] inline bool reject_if_too_large_post(
    const cv::Mat &img, server::DrogonCallback &cb) {
  if (img.cols > max_image_dim() || img.rows > max_image_dim()) {
    cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
        std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                    img.cols, img.rows, max_image_dim(), max_image_dim())));
    return true;
  }
  if (decode::exceeds_pixel_cap(img.cols, img.rows)) {
    cb(server::error_response(drogon::k400BadRequest, "PIXELS_TOO_LARGE",
        std::format("Image area {}x{} exceeds maximum of {} pixels",
                    img.cols, img.rows, decode::max_image_pixels())));
    return true;
  }
  return false;
}
} // namespace

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
    const auto tbl = routing::load_routing_config();
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
              std::string(routing::kind_name(kv.second.kind)) + "\"}";
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

void register_health_route(std::function<bool()> readiness_check,
                           server::WorkPool *pool) {
  auto health_ok = [](const drogon::HttpRequestPtr &,
                      std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
    callback(server::make_response(drogon::k200OK, "ok"));
  };
  drogon::app().registerHandler("/health", health_ok, {drogon::Get});
  drogon::app().registerHandler("/health/live", health_ok, {drogon::Get});

  // /health/ready — verifies the pipeline is actually responsive. The check
  // may run a real GPU inference (cache miss), so offload it to the WorkPool
  // when one is available; running it inline would block the event-loop
  // thread for the GPU-queue-drain duration.
  auto ready_check = std::make_shared<std::function<bool()>>(std::move(readiness_check));
  auto respond = [](server::DrogonCallback &cb, bool ready) {
    if (ready) cb(server::make_response(drogon::k200OK, "ok"));
    else       cb(server::error_response(drogon::k503ServiceUnavailable,
                                          "NOT_READY", "Pipeline not ready"));
  };
  drogon::app().registerHandler(
      "/health/ready",
      [ready_check, pool, respond](
          const drogon::HttpRequestPtr &,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        const bool has_check = static_cast<bool>(*ready_check);
        if (pool && has_check) {
          // Not submit_work: its pool-exhaustion arm answers 503 SERVER_BUSY,
          // which a readiness prober reads as not-ready and pulls a loaded but
          // healthy pod out of rotation — the exact flap the probe's own
          // last-verdict logic exists to prevent. A full queue means busy,
          // not broken: answer ready without running the probe.
          auto cb = std::make_shared<server::DrogonCallback>(std::move(callback));
          try {
            pool->submit([cb, ready_check, respond]() {
              respond(*cb, (*ready_check)());
            });
          } catch (const turbo_ocr::PoolExhaustedError &) {
            (*cb)(server::make_response(drogon::k200OK, "ok"));
          }
          return;
        }
        // No pool (cheap CPU check) or no check configured: run inline.
        callback(has_check && !(*ready_check)()
                     ? server::error_response(drogon::k503ServiceUnavailable,
                                               "NOT_READY", "Pipeline not ready")
                     : server::make_response(drogon::k200OK, "ok"));
      },
      {drogon::Get});
}

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
  const auto rtbl = routing::load_routing_config();
  const std::set<std::string> valid_table   = routing::routable_backend_names(rtbl, "table");
  const std::set<std::string> valid_formula = routing::routable_backend_names(rtbl, "formula");
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr",
      [&pool, &infer, &decode, layout_available, valid_table, valid_formula,
       table_avail, formula_avail](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        server::InferOptions opts;
        if (auto r = server::parse_query_options(req, layout_available, &opts);
            !r.error.empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                           r.error_code.c_str(), r.error));
          return;
        }
        if (reject_unknown_query_params(
                req, {"layout", "reading_order", "as_blocks", "tables", "formulas", "text"}, callback))
          return;
        if (auto r = server::check_structure_backends(opts, table_avail, formula_avail);
            !r.error.empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                           r.error_code.c_str(), r.error));
          return;
        }

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
#ifdef USE_CPU_ONLY
          // The CPU pipeline has no routing plumbing (run_with_layout takes no
          // routing argument), so an override here would validate and then do
          // nothing. Fail loud instead of silently serving the default backend.
          if (!rt.empty() || !rf.empty()) {
            callback(server::error_response(
                drogon::k400BadRequest, "INVALID_PARAMETER",
                "per-request routing overrides are not supported on the CPU "
                "build (the configured backend is always used)"));
            return;
          }
#else
          if (auto r = server::validate_routing_override(
                  rt, rf, valid_table, valid_formula, &opts.routing_override);
              !r.error.empty()) {
            callback(server::error_response(drogon::k400BadRequest,
                                             r.error_code.c_str(), r.error));
            return;
          }
#endif
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

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available,
                             bool table_available,
                             bool formula_available) {
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &infer, &decode, layout_available, table_avail, formula_avail](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        if (req->body().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
          return;
        }

        server::InferOptions opts;
        if (auto r = server::parse_query_options(req, layout_available, &opts);
            !r.error.empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                           r.error_code.c_str(), r.error));
          return;
        }
        // Recognized-but-unsupported here: this registrar serves the CPU
        // build's /ocr/raw, and the CPU pipeline has no routing plumbing.
        // Reject explicitly (same policy as /ocr's JSON routing field) rather
        // than let the params fall through as silently-ignored unknowns.
        if (!req->getParameter("route_table").empty() ||
            !req->getParameter("route_formula").empty()) {
          callback(server::error_response(
              drogon::k400BadRequest, "INVALID_PARAMETER",
              "per-request routing overrides are not supported on the CPU "
              "build (the configured backend is always used)"));
          return;
        }
        if (reject_unknown_query_params(
                req, {"layout", "reading_order", "as_blocks", "tables", "formulas",
                      "text", "route_table", "route_formula"}, callback))
          return;
        if (auto r = server::check_structure_backends(opts, table_avail, formula_avail);
            !r.error.empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                           r.error_code.c_str(), r.error));
          return;
        }

        server::submit_work(pool, std::move(callback),
            [req, &infer, &decode, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/raw", [&] {
            const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();

            if (reject_if_too_large_pre(data, len, cb)) return;

            cv::Mat img = decode(data, len);
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

void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available,
                             bool table_available,
                             bool formula_available,
                             std::function<bool()> readiness_check) {
  // Forward the pool so /health/ready offloads the readiness probe off the
  // event loop — the CPU readiness check calls pool->acquire(), which blocks
  // on a condition variable (unbounded for the CPU pool) when all pipelines
  // are busy; running it inline would wedge a Drogon IO thread under load.
  register_health_route(std::move(readiness_check), &pool);
  register_ocr_base64_route(pool, infer, decode, layout_available,
                            table_available, formula_available);
  register_ocr_raw_route(pool, infer, decode, layout_available,
                         table_available, formula_available);
}

} // namespace turbo_ocr::routes
