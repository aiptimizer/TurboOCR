#pragma once

#include <atomic>
#include <cstring>
#include <format>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <string_view>

#include <grpcpp/grpcpp.h>

#include "turbo_ocr/base/log/logger.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/base/encoding.h"
#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/core/types.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/image_dims.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/service/server/error_codes.h"
#include "turbo_ocr/service/server/error_classify.h"
#include "turbo_ocr/service/grpc/grpc_response_mode.h"
#include "turbo_ocr/service/server/bootstrap/server_config.h"
#include "turbo_ocr/image/fast_png_decoder.h"
// (No nvjpeg_decoder.h: the RPCs no longer sniff for JPEG. Device decode is
// selected by the BACKEND behind EncodedInferFunc, not by the transport
// inspecting magic bytes for one vendor's decoder.)
#include "turbo_ocr/pipeline/job/pdf_job.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/analysis/layout/order/reading_order.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/service/server/server_types.h"
// The gRPC adapter over the SHARED option-parsing core — the same core the HTTP
// routes reach through query_options.h. This is what keeps the two transports'
// gates from drifting; see options_core.h for the three times they did.
#include "turbo_ocr/service/validation/proto_options.h"
#include "ocr.grpc.pb.h"

namespace turbo_ocr::server {

// Free helpers shared by the gRPC RPC TUs (definitions in
// src/service/grpc/grpc_helpers.cpp). grpc_error stamps the HTTP-parity code into
// trailing metadata under "x-error-code".
[[nodiscard]] grpc::Status grpc_error(grpc::ServerContext *ctx,
                                      grpc::StatusCode code,
                                      const char *error_code,
                                      std::string message);
[[nodiscard]] grpc::Status grpc_error(grpc::ServerContext *ctx, ErrorCode code,
                                      std::string message);

// ONE exception mapping for every inference path in the Recognize* RPCs.
//
// There were three copies — pixels, device-decode, host-decode — and they had
// drifted apart, which is the argument for collapsing them:
//   * the device-decode path did not map TimeoutError, so an execution deadline
//     there surfaced as a generic INFERENCE_ERROR instead of DEADLINE_EXCEEDED;
//   * two of the three had no catch(...) at all, so a non-std exception escaped
//     the handler entirely rather than becoming an INTERNAL status.
// Each copy was individually plausible. Only side by side is it clear they
// answer the same question differently.
//
// `body` runs the inference AND fills the response; it is a closure so this
// helper needs no access to OCRServiceImpl's members.
//
// NOT for the per-slot paths: RecognizeBatch and RecognizeStream convert a
// failure into an in-band per-item error inside a SUCCESSFUL response rather
// than returning a status. That is a real structural difference, not
// duplication, and folding it in here would change batch semantics.
template <class Body>
grpc::Status guarded_infer(grpc::ServerContext *ctx, const char *what,
                           Body &&body) {
  // Shared classifier with the HTTP wrapper (error_classify.h): one type→code
  // decision, rendered here through grpc_error's ErrorCode overload — which
  // reads the SAME error_codes.h row the HTTP status comes from. This is what
  // stopped the two from drifting (gRPC used to discard ImageDecodeError's
  // message and answer the literal "Decode failed").
  try {
    body();
    return grpc::Status::OK;
  } catch (...) {
    ExceptionClass ec = classify_current_exception();
    if (ec.log) {
      std::string detail;
      try { throw; } catch (const std::exception &e) { detail = e.what(); }
                     catch (...) { detail = "unknown exception"; }
      TOCR_LOG_ERROR_RL(what, "error", detail);
    }
    return grpc_error(ctx, ec.code, ec.message);
  }
}

// (grpc_check_layout_request is gone — the layout-availability gate is now the
// shared parse_options_core, reached via parse_proto_options. See grpc_helpers.cpp.)
//
// `requested` vs `loaded` — two CapabilityMasks that cannot be swapped for the
// bools they replace. The previous signature took (want_tables, want_formulas,
// table_available, formula_available) positionally: transposing a request flag
// with an availability flag compiled cleanly and silently disabled a feature.
// `raw_layout` is the layout flag AS THE CLIENT SENT IT (request->layout(),
// plus layout_only where applicable) — NOT the derived/implied value. The
// structured-mode gate must see the raw flag: reading_order=1 implies layout
// internally, but reading_order has its own proto response field and v3.5.0
// expressly served it in structured mode; gating on the implied value rejected
// exactly that request with STRUCTURED_MODE_NO_STRUCTURE.
[[nodiscard]] std::optional<grpc::Status> grpc_check_structure_backends(
    grpc::ServerContext *ctx, const capability::CapabilityMask &requested,
    const capability::CapabilityMask &loaded, bool json_bytes_mode,
    bool want_blocks = false, bool raw_layout = false);
[[nodiscard]] std::optional<grpc::Status>
grpc_check_image_size(grpc::ServerContext *ctx, int w, int h);
[[nodiscard]] std::optional<grpc::Status>
grpc_pre_decode_dim_check(grpc::ServerContext *ctx,
                          std::string_view image_data);
[[nodiscard]] cv::Mat grpc_decode_image(std::string_view image_data);

class OCRServiceImpl final : public ocr::OCRService::Service {
public:

  /// CPU-friendly constructor: takes an InferFunc instead of a dispatcher.
  OCRServiceImpl(InferFunc infer_fn,
                 const ServerConfig &cfg,
                 render::PdfRenderer *pdf_renderer,
                 const capability::CapabilityMask &loaded)
      : infer_fn_(std::move(infer_fn)),
        mode_(cfg.grpc_response_mode),
        pdf_renderer_(pdf_renderer),
        default_pdf_mode_(cfg.default_pdf_mode),
        loaded_(loaded),
        grpc_batch_workers_(cfg.grpc_batch_workers),
        max_pdf_pages_(cfg.max_pdf_pages),
        max_batch_images_(cfg.max_batch_images),
        // From cfg, like every neighbour here. It was the one literal in this
        // init list, so the render DPI default was the one PDF limit an
        // operator could not change (PDF_DEFAULT_DPI).
        default_pdf_dpi_(cfg.pdf_default_dpi) {}

  /// Optional page-orientation hook for RecognizePDF's autorotate (the same
  /// server::OrientFunc the HTTP /ocr/pdf route uses). When absent on the
  /// InferFunc path, start_grpc_server clears DocOrientation from `loaded` so
  /// autorotate=true fails loud (AUTOROTATE_DISABLED) instead of silently not
  /// rotating.
  void set_orient_fn(OrientFunc fn) { orient_fn_ = std::move(fn); }

  /// Encoded-bytes inference (server::EncodedInferFunc). OPTIONAL: when set,
  /// the image RPCs hand the pipeline the still-encoded bytes so a backend with
  /// an on-device decoder (nvJPEG, vImage) never pays a host decode + full-frame
  /// H2D. Falls back to the host decoder inside the pipeline when the backend
  /// has no device decode, so it is safe to prefer unconditionally.
  ///
  /// This REPLACES the deleted dispatcher-side nvJPEG branch these RPCs used to
  /// carry: that branch referenced a `dispatcher_` member removed with the CUDA
  /// pipeline, so it had stopped compiling on the only configure that builds it
  /// (USE_CPU_ONLY=OFF — the nvidia bring-up configure). The device-decode
  /// capability it provided now comes through the backend seam instead of a
  /// transport-local special case, which is why the RPC bodies below no longer
  /// sniff for JPEG at all.
  void set_encoded_infer_fn(EncodedInferFunc fn) {
    encoded_infer_fn_ = std::move(fn);
  }

  /// Set the readiness probe used by Health(). Called once per Health RPC on
  /// the gRPC CQ poller thread, so it MUST be cheap and non-blocking — the GPU
  /// server passes a CACHE-ONLY view of the HTTP /health/ready verdict (it
  /// never runs a fresh GPU pass here). nullptr (default) means "always ready".
  void set_readiness_check(std::function<bool()> check) {
    readiness_check_ = std::move(check);
  }

  /// Capability document carried in HealthResponse.capabilities_json — the
  /// exact JSON GET /capabilities serves (built once by
  /// routes::build_capabilities_json), so both transports advertise
  /// identical capabilities.
  void set_capabilities_json(std::string json) {
    capabilities_json_ = std::move(json);
  }

  /// Valid Tier-A routing-override backend names (from
  /// backend_routing::routable_backend_names), threaded from startup so gRPC
  /// validates route_table/route_formula against the same registry as HTTP.
  void set_routing_validation(std::set<std::string> valid_table,
                              std::set<std::string> valid_formula) {
    valid_route_table_ = std::move(valid_table);
    valid_route_formula_ = std::move(valid_formula);
  }

  // ---- Health ----
  grpc::Status Health(grpc::ServerContext *ctx,
                      const ocr::HealthRequest *,
                      ocr::HealthResponse *response) override;

  // ---- Recognize (single image + pixels + layout + reading_order) ----
  grpc::Status Recognize(grpc::ServerContext *ctx,
                         const ocr::OCRRequest *request,
                         ocr::OCRResponse *response) override;

  // ---- RecognizeBatch ----
  grpc::Status RecognizeBatch(grpc::ServerContext *ctx,
                              const ocr::OCRBatchRequest *request,
                              ocr::OCRBatchResponse *response) override;

  // ---- RecognizePDF ----
  grpc::Status RecognizePDF(grpc::ServerContext *ctx,
                            const ocr::OCRPDFRequest *request,
                            ocr::OCRPDFResponse *response) override;

  // ---- RecognizeMarkdown (HTTP /ocr/markdown) ----
  grpc::Status RecognizeMarkdown(grpc::ServerContext *ctx,
                                 const ocr::OCRMarkdownRequest *request,
                                 ocr::OCRMarkdownResponse *response) override;

  // ---- InferOne (HTTP /infer) ----
  grpc::Status InferOne(grpc::ServerContext *ctx,
                        const ocr::InferOneRequest *request,
                        ocr::InferOneResponse *response) override;

  // ---- RecognizeStream (HTTP /ocr/stream), server-streaming ----
  grpc::Status RecognizeStream(
      grpc::ServerContext *ctx, const ocr::OCRStreamRequest *request,
      grpc::ServerWriter<ocr::OCRStreamEvent> *writer) override;

  /// Single-crop inference seam for InferOne — the same InferOneFunc the HTTP
  /// /infer route runs on. Unset (the offline drivers) makes InferOne answer
  /// UNIMPLEMENTED rather than crash.
  void set_infer_one_fn(InferOneFunc fn) { infer_one_fn_ = std::move(fn); }

  /// PDF page-count cap + default DPI are already members; the streaming RPC
  /// reuses them so its limits cannot drift from RecognizePDF's.

private:
  // Mark a batch slot as having no detections. In json_bytes mode this also
  // sets json_response to a valid empty document ('{"results":[]}') so a
  // client uniformly parsing json_response per slot doesn't choke on an
  // empty bytes field for a failed/undecodable image (a successful blank
  // page already produces valid empty JSON via fill_response).
  void mark_empty_slot(ocr::OCRResponse *entry, const char *err = nullptr);

  // Takes the full pipeline result so json_bytes mode emits the SAME body as the
  // HTTP routes — including `tables`/`formulas` (+ degradation flags) when the
  // request opted in (?tables=1 / ?formulas=1). Structured mode carries only
  // `results` (the proto has no table/formula message), same as before.
  void fill_response(ocr::OCRResponse *response,
                     pipeline::OcrPipelineResult &out,
                     bool want_blocks = false);

  void fill_page_results(ocr::OCRPageResult *page,
                         const std::vector<OCRResultItem> &results);

  /// Unified inference through the shared InferFunc seam (there is no
  /// dispatcher fallback — every backend hands the server one InferFunc).
  /// `want_reading_order` auto-enables `want_layout` because reading-order
  /// is computed over layout regions — the contract matches the HTTP
  /// `?reading_order=1` query handler.
  // Tier-A routing-override validation — the same core policy the HTTP gate
  // enforces (request_validation.h): on the CPU build any non-empty override
  // is a loud reject, on GPU the names validate against the registered sets.
  [[nodiscard]] std::optional<grpc::Status>
  grpc_validate_routing(grpc::ServerContext *ctx, const std::string &table,
                        const std::string &formula,
                        backend_routing::RequestRouting *out);

  pipeline::OcrPipelineResult run_infer(const cv::Mat &img, bool want_layout,
                                         bool want_reading_order = false,
                                         bool want_tables = false,
                                         bool want_formulas = false,
                                         const backend_routing::RequestRouting &routing = {},
                                         bool layout_only = false);

  /// Encoded-bytes twin of run_infer: same options, same result shape, but the
  /// decode happens inside the pipeline (on-device where the backend can).
  /// Callers must check encoded_infer_fn_ first — this throws std::logic_error
  /// when it is unset rather than silently falling back, so a missing wire-up
  /// surfaces as a fault instead of a quiet performance regression.
  pipeline::OcrPipelineResult
  run_infer_encoded(const std::uint8_t *data, std::size_t len, bool want_layout,
                    bool want_reading_order = false, bool want_tables = false,
                    bool want_formulas = false,
                    const backend_routing::RequestRouting &routing = {});

  std::function<bool()> readiness_check_;
  InferFunc infer_fn_;
  // Optional; see set_encoded_infer_fn. Null on servers built without a
  // pipeline pool (the offline drivers), so every use site is guarded.
  EncodedInferFunc encoded_infer_fn_;
  // Optional; see set_infer_one_fn. Null => InferOne answers UNIMPLEMENTED.
  InferOneFunc infer_one_fn_;
  GrpcResponseMode mode_;
  render::PdfRenderer *pdf_renderer_ = nullptr;
  pdf::PdfMode default_pdf_mode_ = pdf::PdfMode::Ocr;
  // LOADED (capability/capability.h): one value, not three parallel bools, so
  // the RPCs and the HTTP routes gate on the identical thing.
  capability::CapabilityMask loaded_;
  OrientFunc orient_fn_;
  int grpc_batch_workers_ = 8;
  std::string capabilities_json_;
  std::set<std::string> valid_route_table_;
  std::set<std::string> valid_route_formula_;
  int max_pdf_pages_ = 2000;
  int max_batch_images_ = 1024;
  // Default render DPI when the request doesn't specify one.
  int default_pdf_dpi_ = 100;
  // NOTE: the request deadline is not a member here. It is applied once, in
  // the shared pool-acquisition layer (REQUEST_TIMEOUT_MS -> TimeoutError,
  // which the RPCs map to DEADLINE_EXCEEDED); a per-service copy of the value
  // sat here for a while feeding a deleted dispatcher join and then nothing.
};

// Owning handle for a running gRPC server (definitions in
// src/service/grpc/server_launch.cpp). The jthread joins on destruction, after
// Shutdown() has been driven by the bootstrap drain.
struct GrpcHandle {
  std::unique_ptr<grpc::Server> server;
  std::jthread thread;
};

/// Start the gRPC server on an InferFunc (every backend, both configures).
/// `encoded_infer_fn` is optional: pass make_encoded_infer_func(pool) so the
/// image RPCs can keep the bytes encoded until the pipeline decodes them
/// on-device. Omitting it is correct but costs the device-decode fast path.
[[nodiscard]] GrpcHandle start_grpc_server(
    InferFunc infer_fn, const ServerConfig &cfg,
    render::PdfRenderer *pdf_renderer, const capability::CapabilityMask &loaded,
    std::function<bool()> readiness_check = {},
    std::string capabilities_json = {}, OrientFunc orient_fn = {},
    EncodedInferFunc encoded_infer_fn = {}, InferOneFunc infer_one_fn = {});

} // namespace turbo_ocr::server
