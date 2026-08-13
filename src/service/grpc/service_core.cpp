// OCRServiceImpl core: health, response fill, routing validation, the shared inference entry.
#include "turbo_ocr/service/grpc/grpc_service.h"

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
  //
  // ALL THREE, in one place. Checking only text_degraded implemented a third of
  // the rule stated above; the table/formula arms were saved only by the
  // structured-mode gate in grpc_helpers.cpp rejecting tables/formulas requests
  // outright — a guard in a different file that nothing ties to this one, so
  // relaxing it (or adding a structured table field) would silently reintroduce
  // a clean-looking degraded page here.
  if (mode_ != GrpcResponseMode::json_bytes) {
    std::string err;
    const auto add = [&err](bool degraded, const char *token,
                            const std::string &warning) {
      if (!degraded) return;
      if (!err.empty()) err += "; ";
      err += warning.empty() ? std::string(token) : warning;
    };
    add(out.text_degraded, "text_degraded", out.text_warning);
    add(out.table_degraded, "table_degraded", out.table_warning);
    add(out.formula_degraded, "formula_degraded", out.formula_warning);
    if (!err.empty()) response->set_error(err);
  }
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
    want_layout =
        want_layout || loaded_.get(capability::CapabilityId::Layout);
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
    opts.routing_override = routing;
    // to_pipeline_result, NOT a hand-rolled field-by-field copy: this function
    // carried its own eleven-line transcription of the exact conversion
    // infer_result.h:40 declares itself "the single conversion site between the
    // two shapes". A forked copy is one forgotten line away from dropping a
    // degradation signal and making a failed stage look like a clean 200 — the
    // same class of defect already fixed in combine_recognition and
    // set_stage_degraded. Generic policy is shared, never per transport.
    return to_pipeline_result(infer_fn_(img, opts));
  }
  // Reaching here is a wiring bug: start_grpc_server always supplies an
  // InferFunc, so a caller that constructed OCRServiceImpl without one asked
  // for a service that cannot infer. Throw — same posture as the twin
  // run_infer_encoded below — rather than return an empty OcrPipelineResult{},
  // which would answer every RPC with a clean blank page: the silent-blank
  // failure this codebase refuses to ship anywhere else.
  throw std::logic_error("run_infer called without an InferFunc");
}

pipeline::OcrPipelineResult OCRServiceImpl::run_infer_encoded(
    const std::uint8_t *data, std::size_t len, bool want_layout,
    bool want_reading_order, bool want_tables, bool want_formulas,
    const backend_routing::RequestRouting &routing) {
  if (!encoded_infer_fn_)
    // Throw rather than fall back to a host decode: the caller is expected to
    // test encoded_infer_fn_ first, so reaching here is a wiring bug. A silent
    // fallback would turn "the device-decode path was never connected" into an
    // invisible performance regression — which is precisely how
    // make_encoded_infer_func came to have a definition, a declaration, and
    // zero call sites.
    throw std::logic_error("run_infer_encoded called without an EncodedInferFunc");
  // Identical implication to run_infer: reading-order/tables/formulas are
  // computed over layout regions, so they imply layout when the model is
  // loaded. Duplicating the RULE here would be the fork this whole change is
  // removing, so it stays one expression evaluated the same way in both.
  if (want_reading_order || want_tables || want_formulas)
    want_layout = want_layout || loaded_.get(capability::CapabilityId::Layout);
  InferOptions opts;
  opts.want_layout = want_layout;
  opts.want_reading_order = want_reading_order;
  opts.want_tables = want_tables;
  opts.want_formulas = want_formulas;
  opts.routing_override = routing;
  return to_pipeline_result(encoded_infer_fn_(data, len, opts));
}

} // namespace turbo_ocr::server
