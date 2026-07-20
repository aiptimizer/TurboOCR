// OCRServiceImpl core: health, response fill, routing validation, the shared inference entry.
#include "turbo_ocr/grpc/grpc_service.h"

namespace turbo_ocr::server {

grpc::Status OCRServiceImpl::Health(grpc::ServerContext *ctx,
                    const ocr::HealthRequest *,
                    ocr::HealthResponse *response) {
  // Readiness view of the pipeline, so a wedged/corrupt-engine pod fails its
  // k8s gRPC readiness probe. readiness_check_ MUST be cache-only on the GPU
  // path (set in main.cpp): running the GPU probe inline here would stall the
  // CQ poller thread and every RPC queued behind it (H2). Liveness stays
  // GPU-free — a process that answers this RPC at all is live, and a busy GPU
  // cannot block this call to flap the process out of service (M2).
  // H7: surface the active response mode so a client can discover whether to
  // read OCRResponse.json_response (json_bytes) or .results (structured)
  // without inferring it from an empty field. Additive; default unchanged.
  response->set_response_mode(mode_ == GrpcResponseMode::json_bytes
                                  ? "json_bytes"
                                  : "structured");
  response->set_capabilities_json(capabilities_json_);
  if (readiness_check_ && !readiness_check_()) {
    response->set_status("not_ready");
    return grpc_error(ctx, grpc::StatusCode::UNAVAILABLE,
                      "NOT_READY", "Pipeline not ready");
  }
  response->set_status("ok");
  return grpc::Status::OK;
}

void OCRServiceImpl::mark_empty_slot(ocr::OCRResponse *entry, const char *err) {
  entry->set_num_detections(0);
  if (err) entry->set_error(err);
  if (mode_ == GrpcResponseMode::json_bytes) {
    std::vector<OCRResultItem> empty;
    entry->set_json_response(results_to_json(empty));
  }
}

void OCRServiceImpl::fill_response(ocr::OCRResponse *response,
                   pipeline::OcrPipelineResult &out,
                   bool want_blocks) {
  response->set_num_detections(static_cast<int>(out.results.size()));
  if (mode_ == GrpcResponseMode::json_bytes) {
    response->set_json_response(
        turbo_ocr::emit_pipeline_result_json(out, want_blocks));
  } else {
    response->mutable_results()->Reserve(static_cast<int>(out.results.size()));
    for (const auto &item : out.results) {
      auto *result = response->add_results();
      result->set_text(item.text);
      result->set_confidence(item.confidence);
      result->mutable_bounding_box()->Reserve(4);
      for (int k = 0; k < 4; ++k) {
        auto *bbox = result->add_bounding_box();
        bbox->mutable_x()->Reserve(1);
        bbox->mutable_y()->Reserve(1);
        bbox->add_x(static_cast<float>(item.box[k][0]));
        bbox->add_y(static_cast<float>(item.box[k][1]));
      }
    }
  }
  // Always populate the dedicated reading_order field so non-JSON
  // clients can read it without parsing json_response.
  if (!out.reading_order.empty()) {
    response->mutable_reading_order()->Reserve(
        static_cast<int>(out.reading_order.size()));
    for (int idx : out.reading_order) response->add_reading_order(idx);
  }
  // Structured mode has no json envelope to carry the degraded flags, but a
  // degraded page must never look clean (json_bytes mode and every HTTP
  // route surface it). The per-slot `error` field is additive: empty on
  // healthy responses, so existing clients are unaffected.
  if (mode_ != GrpcResponseMode::json_bytes && out.text_degraded)
    response->set_error(out.text_warning.empty() ? "text_degraded"
                                                 : out.text_warning);
}

std::optional<grpc::Status>
OCRServiceImpl::grpc_validate_routing(grpc::ServerContext *ctx, const std::string &table,
                      const std::string &formula,
                      backend_routing::RequestRouting *out) {
  EndpointSpec spec;
  spec.routing = kBuildRoutingSupport;
  if (auto e = apply_routing_override(table, formula, spec,
                                      valid_route_table_,
                                      valid_route_formula_, out);
      !e.ok())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      e.code.c_str(), e.message);
  return std::nullopt;
}

pipeline::OcrPipelineResult OCRServiceImpl::run_infer(const cv::Mat &img, bool want_layout,
                                       bool want_reading_order,
                                       bool want_tables,
                                       bool want_formulas,
                                       const backend_routing::RequestRouting &routing,
                                       bool layout_only) {
  if (want_reading_order || want_tables || want_formulas)
    want_layout = want_layout || layout_available_;
  if (infer_fn_) {
    if (layout_only)
      // Rejected with INVALID_PARAMETER before ever reaching here; a logic
      // change that removed that gate must fail loud, not run full OCR
      // against the caller's stated intent.
      throw std::logic_error("layout_only reached the CPU InferFunc");
    InferOptions opts;
    opts.want_layout = want_layout;
    opts.want_reading_order = want_reading_order;
    opts.want_tables = want_tables;
    opts.want_formulas = want_formulas;
    auto r = infer_fn_(img, opts);
    pipeline::OcrPipelineResult res;
    res.results          = std::move(r.results);
    res.layout           = std::move(r.layout);
    res.reading_order    = std::move(r.reading_order);
    res.tables           = std::move(r.tables);
    res.formulas         = std::move(r.formulas);
    // Carry the no-silent-failure degradation signals too — without these a
    // failed table/formula/text stage would return a clean 200 over gRPC.
    res.formula_degraded = r.formula_degraded;
    res.formula_warning  = std::move(r.formula_warning);
    res.table_degraded   = r.table_degraded;
    res.table_warning    = std::move(r.table_warning);
    res.text_degraded    = r.text_degraded;
    res.text_warning     = std::move(r.text_warning);
    return res;
  }
#ifndef USE_CPU_ONLY
  // BY-VALUE capture of img (cheap cv::Mat refcount bump): submit_for_default
  // may abandon the task on timeout, so it must not reference caller stack.
  return dispatcher_->submit_for_default(
      [img, want_layout, want_reading_order, want_tables, want_formulas,
       routing, layout_only](auto &e) {
        if (layout_only)
          return e.pipeline->run_layout_only(img, e.stream);
        return e.pipeline->run_with_layout(img, e.stream, want_layout,
                                           want_reading_order, routing,
                                           /*defer_external=*/false,
                                           want_tables, want_formulas);
      });
#else
  throw std::logic_error("No inference backend configured");
#endif
}

} // namespace turbo_ocr::server
