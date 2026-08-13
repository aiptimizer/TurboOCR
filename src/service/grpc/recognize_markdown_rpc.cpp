// gRPC RecognizeMarkdown + InferOne — the transport twins of HTTP
// /ocr/markdown and /infer.
//
// Both endpoints existed only over HTTP until now, and only because their
// routes had been deleted with the CUDA HTTP layer; restoring them there made
// these RPCs possible. Neither reimplements anything: markdown goes through the
// same render_markdown_with_assets + faithful-export defaults the route applies,
// and InferOne runs the same InferOneFunc seam.
#include "turbo_ocr/service/grpc/grpc_service.h"

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/size_classify.h"

namespace turbo_ocr::server {

grpc::Status
OCRServiceImpl::RecognizeMarkdown(grpc::ServerContext *ctx,
                                  const ocr::OCRMarkdownRequest *request,
                                  ocr::OCRMarkdownResponse *response) {
  if (request->image().empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "MISSING_IMAGE",
                      "Empty image");
  if (!loaded_.get(capability::CapabilityId::Layout)) {
    // Code from capability_table.def, which owns it — same rejection the HTTP
    // route emits for the same condition.
    const std::string code(
        capability::capability_info(capability::CapabilityId::Layout)
            .error_code);
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, code.c_str(),
                      "RecognizeMarkdown requires the layout model (do not "
                      "start with DISABLE_LAYOUT=1)");
  }
  if (auto err = grpc_pre_decode_dim_check(ctx, request->image()); err)
    return *err;

  cv::Mat img = grpc_decode_image(request->image());
  if (img.empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "IMAGE_DECODE_FAILED", "Decode failed");
  if (auto st = grpc_check_image_size(ctx, img.cols, img.rows)) return *st;

  // Faithful export: the stages the server actually LOADED run, so a text-only
  // server produces honest text markdown rather than silently dropping sections
  // it claimed to attempt. Byte-identical policy to /ocr/markdown and to
  // /ocr/pdf?markdown=1 — stated once per surface, derived from `loaded_`.
  const bool want_tables = loaded_.get(capability::CapabilityId::Table);
  const bool want_formulas = loaded_.get(capability::CapabilityId::Formula);

  // ONE exception mapping for every RPC (guarded_infer, grpc_service.h): this
  // route used to hand-roll its own catch chain, which omitted ImageDecode/
  // ImageTooLarge/PdfRender and catch(...) and duplicated the ladder the shared
  // classifier already owns. Everything that can throw — inference AND the
  // markdown render — runs inside it.
  return guarded_infer(ctx, "gRPC markdown error", [&] {
    pipeline::OcrPipelineResult out =
        run_infer(img, /*want_layout=*/true, /*want_reading_order=*/true,
                  want_tables, want_formulas);

    turbo_ocr::assign_layout_ids(out.results, out.layout);
    response->set_markdown(turbo_ocr::markdown::render_markdown_with_assets(
        out, img, /*base_dir=*/".", /*embed_images=*/true));

    // The markdown body intentionally drops failed regions, so a degraded stage
    // is INVISIBLE in it — this field is the only signal a client gets. Same
    // content as the HTTP route's X-OCR-Degraded header.
    std::string warn;
    const auto add = [&warn](bool d, const char *name, const std::string &w) {
      if (!d) return;
      if (!warn.empty()) warn += "; ";
      warn += name;
      if (!w.empty()) warn += ":" + w;
    };
    add(out.text_degraded, "text", out.text_warning);
    add(out.table_degraded, "table", out.table_warning);
    add(out.formula_degraded, "formula", out.formula_warning);
    response->set_degraded(warn);
  });
}

grpc::Status OCRServiceImpl::InferOne(grpc::ServerContext *ctx,
                                      const ocr::InferOneRequest *request,
                                      ocr::InferOneResponse *response) {
  if (!infer_one_fn_)
    return grpc_error(ctx, grpc::StatusCode::UNIMPLEMENTED,
                      "BACKEND_UNAVAILABLE",
                      "single-crop inference is not available on this server");
  if (request->image().empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "MISSING_IMAGE",
                      "Empty image");
  const std::string modality = request->modality();
  if (modality != "table" && modality != "formula")
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "INVALID_PARAMETER",
                      "modality must be 'table' or 'formula'");
  if (request->backend().empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "INVALID_PARAMETER",
                      "backend must name a configured backend (see "
                      "Health.capabilities_json -> routing.backends)");
  // Validated against the SAME registered-name sets the HTTP route uses, which
  // start_grpc_server threads in from the routing config. Inline specs are not
  // representable on this transport at all (see the proto comment) — the SSRF
  // surface they open is HTTP-only and opt-in.
  const auto &valid =
      (modality == "table") ? valid_route_table_ : valid_route_formula_;
  if (valid.find(request->backend()) == valid.end())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "ROUTING_UNKNOWN_OVERRIDE",
                      std::format("backend '{}' is not a configured {} backend "
                                  "(see Health.capabilities_json)",
                                  request->backend(), modality));

  if (auto err = grpc_pre_decode_dim_check(ctx, request->image()); err)
    return *err;
  cv::Mat img = grpc_decode_image(request->image());
  if (img.empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "IMAGE_DECODE_FAILED", "Decode failed");
  if (auto st = grpc_check_image_size(ctx, img.cols, img.rows)) return *st;

  try {
    std::string result =
        infer_one_fn_(img, modality, request->backend(), /*inline_spec=*/nullptr);
    response->set_modality(modality);
    if (modality == "table") response->set_html(std::move(result));
    else response->set_latex(std::move(result));
    return grpc::Status::OK;
  } catch (const turbo_ocr::BackendUnavailableError &e) {
    return grpc_error(ctx, grpc::StatusCode::UNAVAILABLE,
                      "BACKEND_UNAVAILABLE", e.what());
  } catch (const turbo_ocr::PoolExhaustedError &e) {
    return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED, "SERVER_BUSY",
                      e.what());
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("gRPC infer_one error", "error", e.what());
    return grpc_error(ctx, grpc::StatusCode::INTERNAL, "INFERENCE_ERROR",
                      "Inference error");
  }
}

} // namespace turbo_ocr::server
