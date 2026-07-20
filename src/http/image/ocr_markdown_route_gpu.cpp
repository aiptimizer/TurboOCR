#include "turbo_ocr/http/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/markdown/markdown_export.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/validation/pixel_dims.h"

using turbo_ocr::decode::NvJpegDecoder;

#include "image_internal.h"

namespace turbo_ocr::routes {
// --- /ocr/markdown: faithful Markdown export ---
void register_ocr_markdown_route_gpu(server::WorkPool &pool,
                                     pipeline::PipelineDispatcher &dispatcher,
                                     const server::ImageDecoder &decode,
                                     bool layout_available,
                                     bool table_available,
                                     bool formula_available) {
  drogon::app().registerHandler(
      "/ocr/markdown",
      [&pool, &dispatcher, &decode, layout_available, table_available,
       formula_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        if (req->body().empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                          "EMPTY_BODY", "Empty body"));
          return;
        }
        {
          server::InferOptions md_opts;
          server::EndpointSpec spec;
          spec.ocr_options = false;
          spec.markdown_embed = true;
          spec.routing_unsupported_reason = server::kRoutingUnsupportedEndpoint;
          if (!server::validate_request(req, spec, layout_available,
                                        table_available, formula_available,
                                        /*valid_route_table=*/{},
                                        /*valid_route_formula=*/{}, &md_opts,
                                        callback))
            return;
        }
        if (!layout_available) {
          callback(server::error_response(drogon::k400BadRequest,
              "LAYOUT_DISABLED",
              "/ocr/markdown requires the layout model (do not start with "
              "DISABLE_LAYOUT=1)"));
          return;
        }
        // Always self-contained base64 data: URIs over HTTP. The legacy ?embed=0 file-ref
        // mode has render_markdown_with_assets write crop PNGs to the server CWD — which the
        // client can never retrieve and which is a swallowed no-op under a read-only root
        // filesystem (the markdown would then reference nonexistent files). A value the
        // endpoint cannot honor is a loud 400, never a silent override.
        if (auto p = req->getParameter("embed"); p == "0" || p == "false") {
          callback(server::error_response(
              drogon::k400BadRequest, "INVALID_PARAMETER",
              "embed=0 (file-ref markdown) is not supported over HTTP; "
              "assets are always embedded as data: URIs"));
          return;
        }
        const bool embed = true;

        // Faithful export gates structure on what the server actually loaded:
        // request table/formula recognition only when their backends exist, so
        // a text-only server produces honest text markdown rather than silently
        // dropping table/formula sections it claimed (via the hardcoded flags)
        // to have attempted. Matches /capabilities and the fail-loud routes.
        const bool md_want_tables = table_available;
        const bool md_want_formulas = formula_available;
        server::submit_work(pool, std::move(callback),
            [req, &dispatcher, &decode, embed, md_want_tables,
             md_want_formulas](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/markdown", [&] {
            const auto *data =
                reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();
            cv::Mat img = decode(data, len);
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest,
                  "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }
            const int kMaxImageDim = decode::max_image_dim();
            if (img.cols > kMaxImageDim || img.rows > kMaxImageDim) {
              cb(server::error_response(drogon::k400BadRequest,
                  "DIMENSIONS_TOO_LARGE", "Image dimensions exceed maximum"));
              return;
            }
            if (decode::exceeds_pixel_cap(img.cols, img.rows)) {
              cb(server::error_response(drogon::k400BadRequest,
                  "PIXELS_TOO_LARGE", "Image area exceeds maximum pixel count"));
              return;
            }

            pipeline::OcrPipelineResult out;
            try {
              out = dispatcher.submit_for_default(
                  [img, md_want_tables, md_want_formulas](auto &e) {
                return e.pipeline->run_with_layout(
                    img, e.stream, /*want_layout=*/true,
                    /*want_reading_order=*/true, /*routing=*/{},
                    /*defer_external=*/true,
                    md_want_tables, md_want_formulas);
              });
            } catch (const turbo_ocr::TimeoutError &) {
              cb(timeout_response());
              return;
            }
            pipeline::finalize_deferred(out);

            turbo_ocr::assign_layout_ids(out.results, out.layout);
            std::string md = turbo_ocr::markdown::render_markdown_with_assets(
                out, img, /*base_dir=*/".", /*embed_images=*/embed);

            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k200OK);
            // no-silent-failure: the markdown body intentionally drops failed/garbage regions
            // (so a degraded stage is invisible in it). Surface degradation in a header so a
            // caller can detect a configured-but-failed stage rather than seeing a clean page.
            if (out.text_degraded || out.table_degraded || out.formula_degraded) {
              std::string warn;
              auto add = [&](bool d, const char *name, const std::string &w) {
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
      }, {drogon::Post});
}


} // namespace turbo_ocr::routes
