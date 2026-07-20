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

#include "turbo_ocr/common/log/logger.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/grpc/grpc_response_mode.h"
#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/decode/fast_png_decoder.h"
#ifndef USE_CPU_ONLY
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#endif
#include "turbo_ocr/pipeline/pdf/pdf_job.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"
#include "ocr.grpc.pb.h"

namespace turbo_ocr::server {

// Free helpers shared by the gRPC RPC TUs (definitions in
// src/grpc/grpc_helpers.cpp). grpc_error stamps the HTTP-parity code into
// trailing metadata under "x-error-code".
[[nodiscard]] grpc::Status grpc_error(grpc::ServerContext *ctx,
                                      grpc::StatusCode code,
                                      const char *error_code,
                                      std::string message);
[[nodiscard]] grpc::Status grpc_error(grpc::ServerContext *ctx, ErrorCode code,
                                      std::string message);
[[nodiscard]] std::optional<grpc::Status>
grpc_check_layout_request(grpc::ServerContext *ctx, bool req_layout,
                          bool req_reading_order, bool layout_available);
[[nodiscard]] std::optional<grpc::Status> grpc_check_structure_backends(
    grpc::ServerContext *ctx, bool want_tables, bool want_formulas,
    bool table_available, bool formula_available, bool json_bytes_mode,
    bool want_layout = false, bool want_blocks = false);
[[nodiscard]] std::optional<grpc::Status>
grpc_check_image_size(grpc::ServerContext *ctx, int w, int h);
[[nodiscard]] std::optional<grpc::Status>
grpc_pre_decode_dim_check(grpc::ServerContext *ctx,
                          std::string_view image_data);
[[nodiscard]] cv::Mat grpc_decode_image(std::string_view image_data);
#ifndef USE_CPU_ONLY
[[nodiscard]] std::future<pipeline::OcrPipelineResult>
grpc_jpeg_decode_and_infer(pipeline::PipelineDispatcher &dispatcher,
                           std::string_view image_bytes, bool want_layout,
                           bool want_reading_order, bool want_tables = false,
                           bool want_formulas = false,
                           const backend_routing::RequestRouting &routing = {},
                           bool layout_only = false);
#endif


class OCRServiceImpl final : public ocr::OCRService::Service {
public:
#ifndef USE_CPU_ONLY
  OCRServiceImpl(pipeline::PipelineDispatcher &dispatcher,
                 const ServerConfig &cfg,
                 render::PdfRenderer *pdf_renderer,
                 bool layout_available)
      : dispatcher_(&dispatcher),
        mode_(cfg.grpc_response_mode),
        pdf_renderer_(pdf_renderer),
        default_pdf_mode_(cfg.default_pdf_mode),
        layout_available_(layout_available),
        grpc_batch_workers_(cfg.grpc_batch_workers),
        max_pdf_pages_(cfg.max_pdf_pages),
        max_batch_images_(cfg.max_batch_images),
        default_pdf_dpi_(100),
        request_timeout_ms_(cfg.request_timeout_ms) {}
#endif

  /// CPU-friendly constructor: takes an InferFunc instead of a dispatcher.
  OCRServiceImpl(InferFunc infer_fn,
                 const ServerConfig &cfg,
                 render::PdfRenderer *pdf_renderer,
                 bool layout_available)
      : infer_fn_(std::move(infer_fn)),
        mode_(cfg.grpc_response_mode),
        pdf_renderer_(pdf_renderer),
        default_pdf_mode_(cfg.default_pdf_mode),
        layout_available_(layout_available),
        grpc_batch_workers_(cfg.grpc_batch_workers),
        max_pdf_pages_(cfg.max_pdf_pages),
        max_batch_images_(cfg.max_batch_images),
        default_pdf_dpi_(100),
        request_timeout_ms_(cfg.request_timeout_ms) {}

  /// Set the readiness probe used by Health(). Called once per Health RPC on
  /// the gRPC CQ poller thread, so it MUST be cheap and non-blocking — the GPU
  /// server passes a CACHE-ONLY view of the HTTP /health/ready verdict (it
  /// never runs a fresh GPU pass here). nullptr (default) means "always ready".
  void set_readiness_check(std::function<bool()> check) {
    readiness_check_ = std::move(check);
  }

  /// Advertise which structure backends are configured so the RPCs can fail
  /// loud (TABLE_BACKEND_DISABLED / FORMULA_BACKEND_DISABLED) when a client
  /// asks for tables/formulas this server can't produce. Default: both false.
  void set_structure_availability(bool table_available, bool formula_available) {
    table_available_ = table_available;
    formula_available_ = formula_available;
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

  /// Unified inference: uses InferFunc if set, otherwise dispatcher.
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

#ifndef USE_CPU_ONLY
  pipeline::PipelineDispatcher *dispatcher_ = nullptr;
#endif
  std::function<bool()> readiness_check_;
  InferFunc infer_fn_;
  GrpcResponseMode mode_;
  render::PdfRenderer *pdf_renderer_ = nullptr;
  pdf::PdfMode default_pdf_mode_ = pdf::PdfMode::Ocr;
  bool layout_available_ = false;
  bool table_available_ = false;
  bool formula_available_ = false;
  int grpc_batch_workers_ = 8;
  std::string capabilities_json_;
  std::set<std::string> valid_route_table_;
  std::set<std::string> valid_route_formula_;
  int max_pdf_pages_ = 2000;
  int max_batch_images_ = 1024;
  // Default render DPI when the request doesn't specify one.
  int default_pdf_dpi_ = 100;
  // Per-request inference deadline (C4) from cfg.request_timeout_ms; 0 = wait
  // unbounded (legacy). Applied to every GPU future .get() so a wedged worker
  // surfaces as DEADLINE_EXCEEDED instead of hanging an RPC. CPU path leaves
  // it unused (InferFunc is synchronous, no dispatcher/wedge risk).
  long request_timeout_ms_ = 30000;
};

// Owning handle for a running gRPC server (definitions in
// src/grpc/server_launch.cpp). The jthread joins on destruction, after
// Shutdown() has been driven by the bootstrap drain.
struct GrpcHandle {
  std::unique_ptr<grpc::Server> server;
  std::jthread thread;
};

#ifndef USE_CPU_ONLY
/// Start the gRPC server on a PipelineDispatcher (GPU path).
[[nodiscard]] GrpcHandle start_grpc_server(
    pipeline::PipelineDispatcher &dispatcher, const ServerConfig &cfg,
    render::PdfRenderer *pdf_renderer = nullptr, bool layout_available = false,
    std::function<bool()> readiness_check = {}, bool table_available = false,
    bool formula_available = false, std::string capabilities_json = {});
#endif
/// Start the gRPC server on an InferFunc (CPU path, also usable from GPU).
[[nodiscard]] GrpcHandle start_grpc_server(
    InferFunc infer_fn, const ServerConfig &cfg,
    render::PdfRenderer *pdf_renderer = nullptr, bool layout_available = false,
    std::function<bool()> readiness_check = {}, bool table_available = false,
    bool formula_available = false, std::string capabilities_json = {});

} // namespace turbo_ocr::server
