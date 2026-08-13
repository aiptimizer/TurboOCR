// /ocr/markdown — faithful Markdown export of one page image.
//
// RESTORED. This route was deleted with src/cuda/ (as
// src/cuda/http/ocr_markdown_route_gpu.cpp) when the second, CUDA-native HTTP
// layer was removed. It was a ROUTE, not CUDA plumbing: it only lived there
// because the GPU server carried its own copy of the HTTP surface. Deleting the
// duplicate implementation should have meant porting this to the unified server,
// and instead the endpoint was lost on every backend — with
// capabilities_route.cpp edited to stop advertising it, which hid the loss.
//
// The port is a one-line change of substance: the dispatcher submit becomes the
// device-agnostic InferFunc. Everything else — the layout gate, the embed
// contract, the faithful-export defaults, the degraded-header policy — is
// carried over verbatim, because none of it was ever device-specific.
#include "turbo_ocr/service/http/common_routes.h"

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/service/validation/request_gate.h"

#include "size_guards.h"

namespace turbo_ocr::routes {

// THE ADMISSION HALF of /ocr/markdown — the same boundary the other restored
// endpoints now draw. Markdown's own wrinkle is that it DERIVES options rather
// than only validating them (tables/formulas default from what the server
// loaded), so the derivation lives here with the parsing it depends on instead
// of trailing the validation call inside the handler.
//
// Returns nullopt having ALREADY answered `callback`.
std::optional<server::InferOptions>
admit_markdown_request(const drogon::HttpRequestPtr &req,
                       const capability::CapabilityMask &loaded,
                       server::DrogonCallback &callback) {
  if (req->body().empty()) {
    callback(server::error_response(server::ErrorCode::kEmptyBody, "Empty body"));
    return std::nullopt;
  }

  server::InferOptions opts;
  {
    // ocr_options=false: this endpoint does not take the per-request
    // layout/tables/formulas flags — it always runs the faithful export
    // (see the defaults below), so accepting them would imply a control
    // it does not offer. markdown_embed=true admits ?embed=, which is
    // then rejected for the one value it cannot honour.
    server::EndpointSpec spec;
    spec.ocr_options = false;
    spec.markdown_embed = true;
    spec.routing_unsupported_reason = server::kRoutingUnsupportedEndpoint;
    if (!server::validate_request(req, spec, loaded,
                                  /*valid_route_table=*/{},
                                  /*valid_route_formula=*/{}, &opts,
                                  callback))
      return std::nullopt;
  }
  if (!loaded.get(capability::CapabilityId::Layout)) {
    // Code from capability_table.def, which owns it — not a literal a
    // rename would leave stale.
    const std::string code(
        capability::capability_info(capability::CapabilityId::Layout)
            .error_code);
    callback(server::error_response(code,
                                    "/ocr/markdown requires the layout model (do not start with "
                                    "DISABLE_LAYOUT=1)"));
    return std::nullopt;
  }
  // Always self-contained base64 data: URIs over HTTP. The legacy ?embed=0
  // file-ref mode has render_markdown_with_assets write crop PNGs to the
  // server CWD — which the client can never retrieve, and which is a
  // swallowed no-op under a read-only root filesystem (the markdown would
  // then reference nonexistent files). A value the endpoint cannot honor
  // is a loud 400, never a silent override.
  if (auto p = req->getParameter("embed"); p == "0" || p == "false") {
    callback(server::error_response(server::ErrorCode::kInvalidParameter,
                                    "embed=0 (file-ref markdown) is not supported over HTTP; "
                                    "assets are always embedded as data: URIs"));
    return std::nullopt;
  }

  // Faithful export gates structure on what the server actually LOADED:
  // request table/formula recognition only when their backends exist, so
  // a text-only server produces honest text markdown rather than silently
  // dropping sections it claimed to have attempted. Matches /capabilities
  // and /ocr/pdf?markdown=1, which applies the identical defaults.
  opts.want_layout = true;
  opts.want_reading_order = true;
  opts.want_tables = loaded.get(capability::CapabilityId::Table);
  opts.want_formulas = loaded.get(capability::CapabilityId::Formula);
  // Keep the REQUESTED mask in step with the bools — request(), not set(),
  // so a capability's dependencies come with it (capability.h).
  opts.requested.request(capability::CapabilityId::Layout);
  opts.requested.request(capability::CapabilityId::Table,
                         opts.want_tables);
  opts.requested.request(capability::CapabilityId::Formula,
                         opts.want_formulas);
  return opts;
}


void register_ocr_markdown_route(server::WorkPool &pool,
                                 const server::InferFunc &infer,
                                 const server::ImageDecoder &decode,
                                 const capability::CapabilityMask &loaded) {
  drogon::app().registerHandler(
      "/ocr/markdown",
      [&pool, &infer, &decode, loaded](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        auto admitted = admit_markdown_request(req, loaded, callback);
        if (!admitted) return;
        const server::InferOptions opts = std::move(*admitted);


        server::submit_work(
            pool, std::move(callback),
            [req, &infer, &decode, opts](server::DrogonCallback &cb) {
              server::run_with_error_handling(cb, "/ocr/markdown", [&] {
                const auto *data = reinterpret_cast<const unsigned char *>(
                    req->body().data());
                const size_t len = req->body().size();

                if (reject_if_too_large_pre(data, len, cb)) return;

                cv::Mat img = decode(data, len);
                if (img.empty()) {
                  cb(server::error_response(server::ErrorCode::kImageDecodeFailed,
                                            "Failed to decode image"));
                  return;
                }
                if (reject_if_too_large_post(img, cb)) return;

                // The device seam. make_infer_func already runs the pipeline
                // with defer_external=true and calls finalize_deferred, so the
                // external-recognizer handling the dispatcher version did by
                // hand is covered here and is not repeated.
                auto out = server::to_pipeline_result(infer(img, opts));

                turbo_ocr::assign_layout_ids(out.results, out.layout);
                std::string md =
                    turbo_ocr::markdown::render_markdown_with_assets(
                        out, img, /*base_dir=*/".", /*embed_images=*/true);

                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setStatusCode(drogon::k200OK);
                // No-silent-failure: the markdown body intentionally drops
                // failed/garbage regions, so a degraded stage is INVISIBLE in
                // it. Surface degradation in a header so a caller can detect a
                // configured-but-failed stage instead of seeing a clean page.
                if (out.text_degraded || out.table_degraded ||
                    out.formula_degraded) {
                  std::string warn;
                  const auto add = [&](bool d, const char *name,
                                       const std::string &w) {
                    if (!d) return;
                    if (!warn.empty()) warn += "; ";
                    warn += name;
                    if (!w.empty()) warn += ":" + w;
                  };
                  add(out.text_degraded, "text", out.text_warning);
                  add(out.table_degraded, "table", out.table_warning);
                  add(out.formula_degraded, "formula", out.formula_warning);
                  resp->addHeader("X-OCR-Degraded", warn);
                }
                resp->setBody(std::move(md));
                resp->setContentTypeString("text/markdown; charset=utf-8");
                cb(resp);
              });
            });
      },
      {drogon::Post});
}

} // namespace turbo_ocr::routes
