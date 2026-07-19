#pragma once

#include <atomic>
#include <cstring>
#include <format>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <semaphore>
#include <string_view>

#include <grpcpp/grpcpp.h>

#include "turbo_ocr/common/logger.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/server/grpc_response_mode.h"
#include "turbo_ocr/server/server_config.h"
#include "turbo_ocr/decode/fast_png_decoder.h"
#ifndef USE_CPU_ONLY
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#endif
#include "turbo_ocr/pipeline/pdf_job.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/layout/reading_order.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"
#include "ocr.grpc.pb.h"

namespace turbo_ocr::server {

// Helper: stamp the structured HTTP-parity error code into gRPC trailing
// metadata under "x-error-code" and return the status. Keeps the legacy
// StatusCode/message untouched so existing clients keep working while
// new clients can branch on the structured code (matches HTTP's
// {"error":{"code":...}} payload one-for-one).
[[nodiscard]] inline grpc::Status
grpc_error(grpc::ServerContext *ctx, grpc::StatusCode code,
           const char *error_code, std::string message) {
  if (ctx) ctx->AddTrailingMetadata("x-error-code", error_code);
  return grpc::Status(code, std::move(message));
}

// Same as grpc_error but sources the wire string + gRPC status from the shared
// error_codes.h table, so the code/status pairing can't drift from HTTP. Used
// for codes that have no hand-written literal at the call site (e.g. the C4
// inference-timeout -> DEADLINE_EXCEEDED mapping).
[[nodiscard]] inline grpc::Status
grpc_error(grpc::ServerContext *ctx, ErrorCode code, std::string message) {
  std::string_view name = error_code_str(code);
  if (ctx) ctx->AddTrailingMetadata("x-error-code", std::string(name));
  return grpc::Status(error_grpc_status(code), std::move(message));
}

// Mirror parse_query_options() in server_types.h: when the client asks for
// layout-derived output but the server was started without the layout
// model, HTTP rejects the request with LAYOUT_DISABLED (the one stable
// code for the condition, whichever flag triggered it). gRPC used to
// silently zero those flags, leaving callers wondering why they got a
// y/x fallback they did not ask for. Returns nullopt on success.
[[nodiscard]] inline std::optional<grpc::Status>
grpc_check_layout_request(grpc::ServerContext *ctx, bool req_layout,
                           bool req_reading_order, bool layout_available) {
  if (req_layout && !layout_available) {
    // LAYOUT_DISABLED for every "layout feature unavailable" rejection —
    // one stable code per condition, matching the HTTP routes and the
    // reading_order case below.
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "LAYOUT_DISABLED",
                      "Layout requested but the layout model is not loaded. "
                      "Either models/layout/layout.onnx is missing from the "
                      "image, or the server was started with DISABLE_LAYOUT=1.");
  }
  if (req_reading_order && !layout_available) {
    // `req_reading_order` here folds in reading_order/as_blocks/tables/formulas
    // — every layout-derived feature — so keep the message generic.
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "LAYOUT_DISABLED",
                      "reading_order/as_blocks/tables/formulas require the "
                      "layout model: start the server without DISABLE_LAYOUT=1 "
                      "(layout is on by default)");
  }
  return std::nullopt;
}

// Fail loud over gRPC when the client opts into a structure stage the server
// can't do (parity with the HTTP check_structure_backends).
[[nodiscard]] inline std::optional<grpc::Status>
grpc_check_structure_backends(grpc::ServerContext *ctx, bool want_tables,
                              bool want_formulas, bool table_available,
                              bool formula_available,
                              bool json_bytes_mode) {
  // Structured response mode carries only `results` (the proto has no
  // table/formula message). Running the stage then dropping it is a silent
  // failure — reject loudly so a structured-mode client knows to use json_bytes.
  if ((want_tables || want_formulas) && !json_bytes_mode)
    return grpc_error(ctx, grpc::StatusCode::UNIMPLEMENTED,
                      "STRUCTURED_MODE_NO_STRUCTURE",
                      "tables/formulas require the json_bytes gRPC response mode "
                      "(structured mode returns only text results)");
  if (want_tables && !table_available)
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "TABLE_BACKEND_DISABLED",
                      "tables=1 requested but no table backend is configured "
                      "(start the server with TABLE_BACKEND=...)");
  if (want_formulas && !formula_available)
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "FORMULA_BACKEND_DISABLED",
                      "formulas=1 requested but no formula backend is configured "
                      "(start the server with FORMULA_BACKEND=...)");
  return std::nullopt;
}

// Returns nullopt on success, or a status carrying DIMENSIONS_TOO_LARGE when
// the encoded image's PNG/JPEG/WebP header advertises width or height beyond
// MAX_IMAGE_DIM. Caller checks before paying the decode cost — same
// decompression-bomb defense the HTTP routes apply.
[[nodiscard]] inline std::optional<grpc::Status>
grpc_pre_decode_dim_check(grpc::ServerContext *ctx,
                           std::string_view image_data) {
  auto *data = reinterpret_cast<const unsigned char *>(image_data.data());
  if (auto d = decode::peek_image_dimensions(data, image_data.size())) {
    int cap = decode::max_image_dim();
    if (d->width > cap || d->height > cap) {
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
          "DIMENSIONS_TOO_LARGE",
          std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                      d->width, d->height, cap, cap));
    }
    if (decode::exceeds_pixel_cap(d->width, d->height)) {
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
          "PIXELS_TOO_LARGE",
          std::format("Image area {}x{} exceeds maximum of {} pixels",
                      d->width, d->height, decode::max_image_pixels()));
    }
  }
  return std::nullopt;
}

// Pure-CPU decoder for the non-JPEG branch of the gRPC handlers. JPEGs are
// routed via grpc_jpeg_decode_and_infer (decode happens on a dispatcher
// worker thread); reaching this with JPEG bytes would be a caller bug.
inline cv::Mat grpc_decode_image(std::string_view image_data) {
  auto *data = reinterpret_cast<const unsigned char *>(image_data.data());
  auto len = image_data.size();
  if (decode::FastPngDecoder::is_png(data, len))
    return decode::FastPngDecoder::decode(data, len);
  if (len > static_cast<size_t>(INT_MAX)) return {};
  return cv::imdecode(cv::Mat(1, static_cast<int>(len), CV_8UC1,
                              const_cast<unsigned char *>(data)),
                      cv::IMREAD_COLOR);
}

#ifndef USE_CPU_ONLY
// Decode + infer on a dispatcher worker thread so nvJPEG's async NVDEC work
// runs on the pipeline's own stream — matches /ocr/raw and avoids the
// cross-thread DMA race that poisoned the CUDA context.
inline std::future<pipeline::OcrPipelineResult>
grpc_jpeg_decode_and_infer(pipeline::PipelineDispatcher &dispatcher,
                           std::string_view image_bytes,
                           bool want_layout, bool want_reading_order,
                           bool want_tables = false, bool want_formulas = false) {
  std::string owned(image_bytes);
  return dispatcher.submit(
      [owned = std::move(owned), want_layout, want_reading_order,
       want_tables, want_formulas](
          auto &e) -> pipeline::OcrPipelineResult {
        const auto *d =
            reinterpret_cast<const unsigned char *>(owned.data());
        size_t n = owned.size();
        const int cap = decode::max_image_dim();
        auto &nvjpeg = e.get_nvjpeg();
        if (nvjpeg.available()) {
          auto [w, h] = nvjpeg.get_dimensions(d, n);
          // Bomb guard for JPEGs the caller's pre-decode sniff couldn't
          // parse: reject (per-side AND total area) before allocating GPU
          // memory for them.
          if (w > cap || h > cap)
            throw turbo_ocr::ImageTooLargeError(std::format(
                "Image dimensions {}x{} exceed maximum of {}x{}", w, h, cap, cap));
          if (decode::exceeds_pixel_cap(w, h))
            throw turbo_ocr::ImageTooLargeError(std::format(
                "Image area {}x{} exceeds maximum of {} pixels", w, h,
                decode::max_image_pixels()));
          if (w > 0 && h > 0) {
            auto [d_buf, pitch] = e.pipeline->ensure_gpu_buf(h, w);
            if (nvjpeg.decode_to_gpu(d, n, d_buf, pitch, w, h, e.stream)) {
              turbo_ocr::GpuImage gi{
                  .data = d_buf, .step = pitch, .rows = h, .cols = w};
              try {
                return e.pipeline->run_with_layout(
                    gi, e.stream, want_layout, want_reading_order, /*routing=*/{},
                    /*defer_external=*/false, want_tables, want_formulas);
              } catch (const std::exception &) {
                // Best-effort GPU zero-copy fast path: if inference on the
                // nvJPEG-decoded GPU buffer fails, fall through to the CPU
                // decode + retry below (which re-runs run_with_layout and
                // surfaces a genuine error there) rather than failing here.
              }
            }
          }
        }
        cv::Mat img = nvjpeg.decode(d, n);
        if (img.empty() && n <= static_cast<size_t>(INT_MAX)) {
          img = cv::imdecode(
              cv::Mat(1, static_cast<int>(n), CV_8UC1,
                      const_cast<unsigned char *>(d)),
              cv::IMREAD_COLOR);
        }
        if (img.empty())
          throw turbo_ocr::ImageDecodeError("Failed to decode JPEG");
        // CPU-fallback / nvjpeg-unavailable bomb guard: re-check the decoded
        // size, since get_dimensions={0,0} and the 64KB sniff can both miss.
        if (img.cols > cap || img.rows > cap)
          throw turbo_ocr::ImageTooLargeError(std::format(
              "Image dimensions {}x{} exceed maximum of {}x{}",
              img.cols, img.rows, cap, cap));
        if (decode::exceeds_pixel_cap(img.cols, img.rows))
          throw turbo_ocr::ImageTooLargeError(std::format(
              "Image area {}x{} exceeds maximum of {} pixels",
              img.cols, img.rows, decode::max_image_pixels()));
        return e.pipeline->run_with_layout(img, e.stream, want_layout,
                                           want_reading_order, /*routing=*/{},
                                           /*defer_external=*/false,
                                           want_tables, want_formulas);
      });
}
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

  // ---- Health ----
  grpc::Status Health(grpc::ServerContext *ctx,
                      const ocr::HealthRequest *,
                      ocr::HealthResponse *response) override {
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
    if (readiness_check_ && !readiness_check_()) {
      response->set_status("not_ready");
      return grpc_error(ctx, grpc::StatusCode::UNAVAILABLE,
                        "NOT_READY", "Pipeline not ready");
    }
    response->set_status("ok");
    return grpc::Status::OK;
  }

  // ---- Recognize (single image + pixels + layout + reading_order) ----
  grpc::Status Recognize(grpc::ServerContext *ctx,
                         const ocr::OCRRequest *request,
                         ocr::OCRResponse *response) override {
    if (auto err = grpc_check_layout_request(ctx, request->layout(),
            request->reading_order() || request->as_blocks() ||
            request->tables() || request->formulas(),
            layout_available_); err)
      return *err;
    bool want_layout = request->layout();
    bool want_reading_order = request->reading_order();
    const bool want_blocks = request->as_blocks();
    const bool want_tables = request->tables();
    const bool want_formulas = request->formulas();
    if (want_blocks) {
      want_reading_order = true;
      want_layout = true;
    }
    if (want_reading_order || want_tables || want_formulas) want_layout = true;
    if (auto err = grpc_check_structure_backends(ctx, want_tables, want_formulas,
            table_available_, formula_available_,
            mode_ == GrpcResponseMode::json_bytes); err)
      return *err;

    // Pixels path: raw BGR pixel data
    if (!request->pixels().empty()) {
      int width = request->width();
      int height = request->height();
      int channels = request->channels();
      if (channels == 0) channels = 3;

      if (width <= 0 || height <= 0 || (channels != 1 && channels != 3))
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                          "INVALID_DIMENSIONS",
                          "Invalid dimensions or channels for pixels input");

      const int cap = decode::max_image_dim();
      if (width > cap || height > cap)
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
            "DIMENSIONS_TOO_LARGE",
            std::format("Dimensions {}x{} exceed maximum of {}x{}",
                        width, height, cap, cap));
      if (decode::exceeds_pixel_cap(width, height))
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
            "PIXELS_TOO_LARGE",
            std::format("Image area {}x{} exceeds maximum of {} pixels",
                        width, height, decode::max_image_pixels()));

      size_t expected = static_cast<size_t>(width) * height * channels;
      if (request->pixels().size() != expected)
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
            "BODY_SIZE_MISMATCH",
            std::format("Pixels size mismatch: expected {} bytes ({}x{}x{}), got {}",
                        expected, width, height, channels, request->pixels().size()));

      // Copy out of request->pixels() into an owning Mat. The dispatcher
      // worker thread reads img.data; even though run_infer() blocks on
      // .get() and the GPU pipeline syncs after its H2D memcpy, we don't
      // want this contract to depend on knowledge of pipeline internals.
      // One memcpy at request boundary keeps lifetime trivially correct.
      cv::Mat img = cv::Mat(height, width, channels == 3 ? CV_8UC3 : CV_8UC1,
                            const_cast<char *>(request->pixels().data()))
                        .clone();
      // The pipeline is BGR-only; a 1-channel Mat trips the degenerate-input
      // guard and returns empty. Expand grayscale up front, matching the
      // HTTP /ocr/pixels handler.
      if (channels == 1)
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

      try {
        auto out = run_infer(img, want_layout, want_reading_order,
                             want_tables, want_formulas);
        fill_response(response, out, want_blocks);
        return grpc::Status::OK;
      } catch (const turbo_ocr::PoolExhaustedError &e) {
        return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "SERVER_BUSY", e.what());
#ifndef USE_CPU_ONLY
        // TimeoutError only exists on the GPU dispatcher path; the CPU
        // InferFunc is synchronous and never throws it (and the type isn't
        // even declared in that build), so the catch is GPU-only.
      } catch (const turbo_ocr::TimeoutError &e) {
        return grpc_error(ctx, ErrorCode::kInferenceTimeout, e.what());
#endif
      } catch (const std::exception &e) {
        TOCR_LOG_ERROR_RL("gRPC pixels inference error", "error", e.what());
        return grpc_error(ctx, grpc::StatusCode::INTERNAL,
                          "INFERENCE_ERROR", "Inference error");
      }
    }

    // Image path: encoded image bytes
    if (request->image().empty()) [[unlikely]]
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                        "MISSING_IMAGE", "Empty image");

    if (auto err = grpc_pre_decode_dim_check(ctx, request->image()); err)
      return *err;

#ifndef USE_CPU_ONLY
    {
      const auto *bytes =
          reinterpret_cast<const unsigned char *>(request->image().data());
      const size_t blen = request->image().size();
      if (dispatcher_ &&
          decode::NvJpegDecoder::is_jpeg(bytes, blen)) {
        try {
          // grpc_jpeg_decode_and_infer's lambda owns its bytes by value, so a
          // timed-out future is safe to abandon (C4).
          auto fut = grpc_jpeg_decode_and_infer(*dispatcher_, request->image(),
                                                 want_layout,
                                                 want_reading_order,
                                                 want_tables, want_formulas);
          auto out = pipeline::get_with_timeout(fut, request_timeout_ms_);
          fill_response(response, out, want_blocks);
          return grpc::Status::OK;
        } catch (const turbo_ocr::ImageTooLargeError &e) {
          return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            "DIMENSIONS_TOO_LARGE", e.what());
        } catch (const turbo_ocr::ImageDecodeError &) {
          return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            "IMAGE_DECODE_FAILED", "Decode failed");
        } catch (const turbo_ocr::PoolExhaustedError &e) {
          return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED,
                            "SERVER_BUSY", e.what());
        } catch (const turbo_ocr::TimeoutError &e) {
          return grpc_error(ctx, ErrorCode::kInferenceTimeout, e.what());
        } catch (const std::exception &e) {
          TOCR_LOG_ERROR_RL("gRPC jpeg inference error", "error", e.what());
          return grpc_error(ctx, grpc::StatusCode::INTERNAL,
                            "INFERENCE_ERROR", "Inference error");
        }
      }
    }
#endif

    // Non-JPEG (PNG/etc.) path: CPU decode on this thread is safe, then
    // hand the materialized cv::Mat to the dispatcher.
    cv::Mat img = grpc_decode_image(request->image());
    if (img.empty()) [[unlikely]]
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                        "IMAGE_DECODE_FAILED", "Decode failed");

    {
      const int cap = decode::max_image_dim();
      if (img.cols > cap || img.rows > cap)
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
            "DIMENSIONS_TOO_LARGE",
            std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                        img.cols, img.rows, cap, cap));
      if (decode::exceeds_pixel_cap(img.cols, img.rows))
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
            "PIXELS_TOO_LARGE",
            std::format("Image area {}x{} exceeds maximum of {} pixels",
                        img.cols, img.rows, decode::max_image_pixels()));
    }

    try {
      auto out = run_infer(img, want_layout, want_reading_order,
                           want_tables, want_formulas);
      fill_response(response, out, want_blocks);
      return grpc::Status::OK;
    } catch (const turbo_ocr::PoolExhaustedError &e) {
      return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED,
                        "SERVER_BUSY", e.what());
#ifndef USE_CPU_ONLY
    } catch (const turbo_ocr::TimeoutError &e) {  // GPU-only type (see above)
      return grpc_error(ctx, ErrorCode::kInferenceTimeout, e.what());
#endif
    } catch (const std::exception &e) {
      TOCR_LOG_ERROR_RL("gRPC inference error", "error", e.what());
      return grpc_error(ctx, grpc::StatusCode::INTERNAL,
                        "INFERENCE_ERROR", "Inference error");
    } catch (...) {
      TOCR_LOG_ERROR_RL("gRPC inference error", "error", "unknown exception");
      return grpc_error(ctx, grpc::StatusCode::INTERNAL,
                        "INFERENCE_ERROR", "Inference error");
    }
  }

  // ---- RecognizeBatch ----
  grpc::Status RecognizeBatch(grpc::ServerContext *ctx,
                              const ocr::OCRBatchRequest *request,
                              ocr::OCRBatchResponse *response) override {
    int n = request->images_size();
    if (n == 0)
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                        "EMPTY_BATCH", "Empty images array");
    // Cap before the O(n) per-slot vectors + n response sub-messages below —
    // an unbounded repeated images field is a memory-amplification OOM lever.
    if (n > max_batch_images_)
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "BATCH_TOO_LARGE",
          std::format("images has {} entries, max is {}", n, max_batch_images_));

    if (auto err = grpc_check_layout_request(ctx, request->layout(),
            request->reading_order() || request->as_blocks() ||
            request->tables() || request->formulas(),
            layout_available_); err)
      return *err;
    bool want_layout = request->layout();
    bool want_reading_order = request->reading_order();
    const bool want_blocks = request->as_blocks();
    const bool want_tables = request->tables();
    const bool want_formulas = request->formulas();
    if (want_blocks) {
      want_reading_order = true;
      want_layout = true;
    }
    if (want_reading_order || want_tables || want_formulas) want_layout = true;
    if (auto err = grpc_check_structure_backends(ctx, want_tables, want_formulas,
            table_available_, formula_available_,
            mode_ == GrpcResponseMode::json_bytes); err)
      return *err;

    // Per-slot oversize handling: an oversized image is dropped to an empty
    // slot (0 detections), NOT a whole-RPC abort — one decompression-bomb in
    // a batch must not deny service to every co-batched valid image. Mirrors
    // the per-slot contract of HTTP /ocr/batch and of this handler's own
    // decode-failure path. Pre-decode header sniff refuses bombs before any
    // decode cost.
    const int dim_cap = decode::max_image_dim();
    // Aggregate decoded-pixel budget: the per-image caps below bound a single
    // slot, but every non-JPEG slot is decoded up front and held alive at once,
    // so a batch of highly-compressible bomb PNGs can still OOM the host. Tag
    // sniffable slots once the running sum would exceed the budget so they are
    // never decoded. Mirrors batch_check_dims_pre on the HTTP /ocr/batch routes
    // (image_routes.cpp / cpu_main.cpp), keeping all three batch surfaces in
    // lockstep.
    int64_t cumulative_pixels = 0;
    const int64_t batch_pixel_budget = decode::max_batch_pixels();
    std::vector<bool> too_large(n, false);
    for (int i = 0; i < n; ++i) {
      auto *p = reinterpret_cast<const unsigned char *>(request->images(i).data());
      if (auto d = decode::peek_image_dimensions(p, request->images(i).size())) {
        if (d->width > dim_cap || d->height > dim_cap ||
            decode::exceeds_pixel_cap(d->width, d->height)) {
          too_large[i] = true;
        } else {
          const int64_t pix = static_cast<int64_t>(d->width) * d->height;
          if (cumulative_pixels + pix > batch_pixel_budget)
            too_large[i] = true;
          else
            cumulative_pixels += pix;
        }
      }
    }

    // JPEGs decode inside the dispatcher lambda (see grpc_jpeg_decode_and_infer);
    // PNG/other decode here on CPU and ship the materialized cv::Mat.
    std::vector<cv::Mat> imgs(n);
    std::vector<bool> is_jpeg(n, false);
    for (int i = 0; i < n; ++i) {
      if (too_large[i]) continue;
      const auto &bytes = request->images(i);
      const auto *p = reinterpret_cast<const unsigned char *>(bytes.data());
#ifndef USE_CPU_ONLY
      if (dispatcher_ && decode::NvJpegDecoder::is_jpeg(p, bytes.size())) {
        is_jpeg[i] = true;
        continue; // decode happens inside the dispatcher lambda
      }
#endif
      imgs[i] = grpc_decode_image(bytes);
    }

    // Post-decode safety net for residual formats we don't sniff (BMP/PNM).
    for (int i = 0; i < n; ++i) {
      if (imgs[i].empty()) continue;
      if (imgs[i].cols > dim_cap || imgs[i].rows > dim_cap ||
          decode::exceeds_pixel_cap(imgs[i].cols, imgs[i].rows)) {
        too_large[i] = true;
        imgs[i].release();  // drop to empty slot
      }
    }

    // Check we have at least one valid candidate. JPEGs are still encoded
    // bytes at this point — we trust the pre-decode dim sniff and decode
    // failures will surface per-slot below. A mixed batch proceeds (oversized
    // slots emit empty results via mark_empty_slot). Only an all-invalid
    // batch aborts — and then the code reflects WHY: if any slot was
    // oversized, report DIMENSIONS_TOO_LARGE (matching single Recognize)
    // rather than the misleading IMAGE_DECODE_FAILED.
    bool any_valid = false, any_too_large = false;
    for (int i = 0; i < n; ++i) {
      if (is_jpeg[i] || !imgs[i].empty()) { any_valid = true; break; }
      if (too_large[i]) any_too_large = true;
    }
    if (!any_valid)
      return any_too_large
          ? grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                       "DIMENSIONS_TOO_LARGE",
                       "All images exceed the maximum dimension")
          : grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                       "IMAGE_DECODE_FAILED", "No valid images");

    // RepeatedPtrField is not thread-safe for concurrent add_*, so pre-allocate.
    response->set_total_images(n);
    std::vector<ocr::OCRResponse *> entries;
    entries.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      auto *e = response->add_batch_results();
      if (!is_jpeg[i] && imgs[i].empty())
        mark_empty_slot(e, too_large[i] ? "IMAGE_TOO_LARGE" : "IMAGE_DECODE_FAILED");
      entries.push_back(e);
    }

#ifndef USE_CPU_ONLY
    if (dispatcher_) {
      std::vector<std::future<pipeline::OcrPipelineResult>> futs(n);
      for (int i = 0; i < n; ++i) {
        try {
          if (is_jpeg[i]) {
            futs[i] = grpc_jpeg_decode_and_infer(
                *dispatcher_, request->images(i), want_layout,
                want_reading_order, want_tables, want_formulas);
          } else if (!imgs[i].empty()) {
            cv::Mat img_owned = std::move(imgs[i]);
            futs[i] = dispatcher_->submit(
                [img_owned = std::move(img_owned), want_layout,
                 want_reading_order, want_tables, want_formulas](auto &e) {
                  return e.pipeline->run_with_layout(
                      img_owned, e.stream, want_layout, want_reading_order,
                      /*routing=*/{}, /*defer_external=*/false,
                      want_tables, want_formulas);
                });
          }
        } catch (const turbo_ocr::PoolExhaustedError &e) {
          return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED,
                            "SERVER_BUSY", e.what());
        }
      }
      // Single overall deadline for the whole batch (C4): the slots run
      // concurrently on the dispatcher, so the per-request window applies to
      // the batch as a whole, not to each slot in turn (otherwise a wedged GPU
      // could block up to n * request_timeout_ms). Each .get() waits only the
      // time remaining until that one deadline. timeout<=0 means disabled, so
      // block (matches submit_for_default's disabled path).
      const bool batch_deadline_on = request_timeout_ms_ > 0;
      const auto batch_deadline =
          std::chrono::steady_clock::now() +
          std::chrono::milliseconds(batch_deadline_on ? request_timeout_ms_ : 0);
      for (int i = 0; i < n; ++i) {
        if (!futs[i].valid()) continue;
        try {
          // A wedged slot is abandoned and tagged empty rather than hanging the
          // whole batch RPC. The submit lambdas above own their inputs by
          // value, so an abandoned future is safe. TimeoutError derives from
          // std::exception, so the per-slot catch below marks it empty.
          pipeline::OcrPipelineResult out;
          if (batch_deadline_on) {
            long remaining_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    batch_deadline - std::chrono::steady_clock::now())
                    .count();
            if (remaining_ms <= 0) {
              // Batch deadline already elapsed (an earlier slot consumed the
              // window): abandon this slot empty. Must NOT call
              // get_with_timeout(fut, 0) — 0 means "disabled" there and would
              // block on future.get() forever, hanging the whole RPC.
              mark_empty_slot(entries[i], "INFERENCE_TIMEOUT");
              continue;
            }
            out = pipeline::get_with_timeout(futs[i], remaining_ms);
          } else {
            out = futs[i].get();
          }
          fill_response(entries[i], out, want_blocks);
        } catch (const std::exception &e) {
          TOCR_LOG_ERROR_RL("gRPC batch image error", "index", i, "error", e.what());
          mark_empty_slot(entries[i], "INFERENCE_ERROR");
        } catch (...) {
          mark_empty_slot(entries[i], "INFERENCE_ERROR");
        }
      }
      return grpc::Status::OK;
    }
#endif

    // CPU-only fanout: bounded jthread pool, each thread calls run_infer
    // (which is synchronous through the InferFunc on this build).
    // grpc_batch_workers_ bounds ONE RPC; without a process-wide ceiling, N
    // concurrent RPCs would spawn N*grpc_batch_workers_ threads (resource-
    // exhaustion vector, unlike the HTTP WorkPool which is globally bounded).
    // Every RPC keeps one guaranteed worker (progress under contention);
    // extra workers need a permit from the shared pool.
    // GRPC_BATCH_GLOBAL_WORKERS caps the process-wide total of extra workers.
    static std::counting_semaphore<1024> extra_worker_permits{
        static_cast<std::ptrdiff_t>(
            env::env_int("GRPC_BATCH_GLOBAL_WORKERS", 16, 1, 1024) - 1)};
    const int num_workers = std::min(n, grpc_batch_workers_);
    std::atomic<int> next_idx{0};
    {
      const auto worker_fn = [&]() {
        while (true) {
          const int i = next_idx.fetch_add(1);
          if (i >= n) break;
          if (imgs[i].empty()) continue;
          try {
            auto out = run_infer(imgs[i], want_layout, want_reading_order,
                                 want_tables, want_formulas);
            fill_response(entries[i], out, want_blocks);
          } catch (const std::exception &e) {
            TOCR_LOG_ERROR_RL("gRPC batch image error", "index", i, "error", e.what());
            mark_empty_slot(entries[i], "INFERENCE_ERROR");
          } catch (...) {
            mark_empty_slot(entries[i], "INFERENCE_ERROR");
          }
        }
      };
      std::vector<std::jthread> workers;
      workers.reserve(static_cast<size_t>(num_workers));
      workers.emplace_back(worker_fn);  // guaranteed worker, no permit
      for (int w = 1; w < num_workers; ++w) {
        if (!extra_worker_permits.try_acquire()) break;
        workers.emplace_back([&worker_fn]() {
          struct Release {
            ~Release() { extra_worker_permits.release(); }
          } release_on_exit;
          worker_fn();
        });
      }
    }
    return grpc::Status::OK;
  }

  // ---- RecognizePDF ----
  grpc::Status RecognizePDF(grpc::ServerContext *ctx,
                            const ocr::OCRPDFRequest *request,
                            ocr::OCRPDFResponse *response) override {
    if (!pdf_renderer_)
      return grpc_error(ctx, grpc::StatusCode::UNIMPLEMENTED,
                        "PDF_NOT_AVAILABLE",
                        "PDF rendering not available on this server");

    if (request->pdf_data().empty())
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                        "MISSING_PDF", "Empty PDF data");

    if (auto err = grpc_check_layout_request(ctx, request->layout(),
            /*reading_order=*/request->as_blocks() ||
            request->tables() || request->formulas(),
            layout_available_); err)
      return *err;

    const auto *pdf_data = reinterpret_cast<const uint8_t *>(request->pdf_data().data());
    size_t pdf_len = request->pdf_data().size();

    bool want_layout = request->layout();
    const bool want_blocks = request->as_blocks();
    const bool want_reading_order = want_blocks;
    const bool want_tables = request->tables();
    const bool want_formulas = request->formulas();
    if (want_blocks || want_tables || want_formulas) want_layout = true;
    if (auto err = grpc_check_structure_backends(ctx, want_tables, want_formulas,
            table_available_, formula_available_,
            mode_ == GrpcResponseMode::json_bytes); err)
      return *err;

    int dpi = request->dpi();
    if (dpi == 0) dpi = default_pdf_dpi_;
    if (dpi < pipeline::kMinPdfDpi || dpi > pipeline::kMaxPdfDpi)
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "INVALID_DPI",
                        std::format("DPI must be between {} and {}",
                                    pipeline::kMinPdfDpi, pipeline::kMaxPdfDpi));

    pdf::PdfMode req_mode = default_pdf_mode_;
    if (!request->mode().empty())
      req_mode = pdf::parse_pdf_mode(request->mode(), default_pdf_mode_);

    // MAX_PDF_PAGES guard — same env var and limit as HTTP /ocr/pdf
    // (default 2000). Mirror the route's reject_if_too_many_pages: open the
    // doc once just for the page count.
    {
      pdf::PdfDocument probe(pdf_data, pdf_len);
      if (probe.ok() && probe.page_count() > max_pdf_pages_)
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
            "PDF_TOO_LARGE",
            std::format("PDF has {} pages, maximum is {} "
                        "(set MAX_PDF_PAGES to increase)",
                        probe.page_count(), max_pdf_pages_));
    }

    // Transport-agnostic orchestrator (H9 + H3): the shared run_pdf_job runs
    // the exact same per-page pipeline the HTTP /ocr/pdf route runs, submitting
    // page work DIRECTLY onto the bounded dispatcher (no per-page std::async /
    // counting_semaphore). gRPC's PDF surface is text-only: it never requests
    // autorotate or inline page images, so those options stay default-off and
    // the serialised output matches today's byte-for-byte.
    pipeline::PdfJobOptions job_opts;
    job_opts.dpi = dpi;
    job_opts.mode = req_mode;
    job_opts.want_layout = want_layout;
    job_opts.want_reading_order = want_reading_order;
    job_opts.want_blocks = want_blocks;
    job_opts.want_tables = want_tables;
    job_opts.want_formulas = want_formulas;
    // Bound the GPU per-page future join with the configured request deadline so
    // a wedged page can't hang the RPC (no-op on the sequential CPU overload).
    job_opts.request_timeout_ms = request_timeout_ms_;

    pipeline::PdfJobResult job;
    try {
#ifndef USE_CPU_ONLY
      if (dispatcher_) {
        job = pipeline::run_pdf_job(*dispatcher_, *pdf_renderer_, pdf_data,
                                    pdf_len, job_opts);
      } else
#endif
      {
        // CPU build: AutoVerified is GPU-only, alias to Auto (parity with the
        // CPU HTTP route). No orientation backend here (gRPC has no autorotate).
        if (job_opts.mode == pdf::PdfMode::AutoVerified)
          job_opts.mode = pdf::PdfMode::Auto;
        job = pipeline::run_pdf_job(
            [this](const cv::Mat &img, const InferOptions &o) {
              auto r = run_infer(img, o.want_layout, o.want_reading_order,
                                 o.want_tables, o.want_formulas);
              return InferResult{
                  .results          = std::move(r.results),
                  .layout           = std::move(r.layout),
                  .reading_order    = std::move(r.reading_order),
                  .tables           = std::move(r.tables),
                  .formulas         = std::move(r.formulas),
                  .formula_degraded = r.formula_degraded,
                  .formula_warning  = std::move(r.formula_warning),
                  .table_degraded   = r.table_degraded,
                  .table_warning    = std::move(r.table_warning),
                  .text_degraded    = r.text_degraded,
                  .text_warning     = std::move(r.text_warning),
              };
            },
            *pdf_renderer_, pdf_data, pdf_len, job_opts,
            server::OrientFunc{});
      }
    } catch (const turbo_ocr::PoolExhaustedError &e) {
      return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED,
                        "SERVER_BUSY", e.what());
    } catch (const std::exception &e) {
      TOCR_LOG_ERROR_RL("gRPC pdf error", "error", e.what());
      return grpc_error(ctx, grpc::StatusCode::INTERNAL, "INFERENCE_ERROR",
                        "Inference error");
    }

    switch (job.status) {
      case pipeline::PdfJobStatus::Ok: break;
      case pipeline::PdfJobStatus::RenderFailed:
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                          "PDF_RENDER_FAILED", "PDF render failed");
      case pipeline::PdfJobStatus::EmptyPdf:
        return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                          "EMPTY_PDF", "PDF contains no pages");
      case pipeline::PdfJobStatus::Dropped:
        return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED, "SERVER_BUSY",
            std::format("GPU queue full: {} of {} pages could not be processed. "
                        "Retry with backoff.", job.dropped_pages, job.num_pages));
      case pipeline::PdfJobStatus::DecodeFailed:
        return grpc_error(ctx, grpc::StatusCode::INTERNAL, "PAGE_DECODE_FAILED",
            std::format("{} of {} rendered pages could not be decoded; retry",
                        job.decode_failures, job.num_pages));
      case pipeline::PdfJobStatus::PageFailed:
        return grpc_error(ctx, grpc::StatusCode::INTERNAL, "PAGE_FAILED",
            std::format("{} of {} pages failed during OCR; retry",
                        job.page_failures, job.num_pages));
      case pipeline::PdfJobStatus::TimedOut:
        return grpc_error(ctx, grpc::StatusCode::DEADLINE_EXCEEDED,
            "INFERENCE_TIMEOUT", "PDF job exceeded the request deadline");
    }

    // Build response
    for (int i = 0; i < job.num_pages; ++i) {
      auto *page = response->add_pages();
      auto &pg = job.pages[static_cast<size_t>(i)];
      page->set_page_number(i + 1);
      page->set_width(pg.width);
      page->set_height(pg.height);
      page->set_dpi(pg.effective_dpi > 0 ? pg.effective_dpi : dpi);
      page->set_mode(std::string(pdf::mode_name(pg.resolved_mode)));
      page->set_text_layer_quality(std::string(pg.text_layer_quality));

      if (mode_ == GrpcResponseMode::json_bytes) {
        // Shared per-page serializer (H7) — identical body to HTTP's per-page
        // JSON, so the two transports can't drift on result/layout shape.
        page->set_json_response(
            pipeline::serialize_page_results(pg, want_blocks));
      } else {
        fill_page_results(page, pg.results);
      }
    }

    return grpc::Status::OK;
  }

private:
  // Mark a batch slot as having no detections. In json_bytes mode this also
  // sets json_response to a valid empty document ('{"results":[]}') so a
  // client uniformly parsing json_response per slot doesn't choke on an
  // empty bytes field for a failed/undecodable image (a successful blank
  // page already produces valid empty JSON via fill_response).
  void mark_empty_slot(ocr::OCRResponse *entry, const char *err = nullptr) {
    entry->set_num_detections(0);
    if (err) entry->set_error(err);
    if (mode_ == GrpcResponseMode::json_bytes) {
      std::vector<OCRResultItem> empty;
      entry->set_json_response(results_to_json(empty));
    }
  }

  // Takes the full pipeline result so json_bytes mode emits the SAME body as the
  // HTTP routes — including `tables`/`formulas` (+ degradation flags) when the
  // request opted in (?tables=1 / ?formulas=1). Structured mode carries only
  // `results` (the proto has no table/formula message), same as before.
  void fill_response(ocr::OCRResponse *response,
                     pipeline::OcrPipelineResult &out,
                     bool want_blocks = false) {
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
  }

  void fill_page_results(ocr::OCRPageResult *page,
                         const std::vector<OCRResultItem> &results) {
    page->mutable_results()->Reserve(static_cast<int>(results.size()));
    for (const auto &item : results) {
      auto *result = page->add_results();
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

  /// Unified inference: uses InferFunc if set, otherwise dispatcher.
  /// `want_reading_order` auto-enables `want_layout` because reading-order
  /// is computed over layout regions — the contract matches the HTTP
  /// `?reading_order=1` query handler.
  pipeline::OcrPipelineResult run_infer(const cv::Mat &img, bool want_layout,
                                         bool want_reading_order = false,
                                         bool want_tables = false,
                                         bool want_formulas = false) {
    if (want_reading_order || want_tables || want_formulas)
      want_layout = want_layout || layout_available_;
    if (infer_fn_) {
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
        [img, want_layout, want_reading_order, want_tables, want_formulas](auto &e) {
          return e.pipeline->run_with_layout(img, e.stream, want_layout,
                                             want_reading_order, /*routing=*/{},
                                             /*defer_external=*/false,
                                             want_tables, want_formulas);
        });
#else
    throw std::logic_error("No inference backend configured");
#endif
  }

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

/// Start gRPC server on a background thread. Returns the server and thread.
/// Caller must keep both alive. Call server->Shutdown() to stop.
struct GrpcHandle {
  std::unique_ptr<grpc::Server> server;
  std::jthread thread;
};

namespace detail {

inline GrpcHandle launch_grpc_server(std::shared_ptr<OCRServiceImpl> service,
                                      int port, const ServerConfig &cfg) {
  // MAX_BODY_MB and GRPC_CQS now sourced from ServerConfig — the HTTP path
  // pulls from the same cfg so gRPC and HTTP body caps cannot drift.
  const int max_body_mb = cfg.max_body_mb;
  // Compute in int64 so MAX_BODY_MB=2048 (= 2^31 bytes) doesn't wrap
  // signed int. gRPC's SetMax{Receive,Send}MessageSize takes int, so
  // clamp to INT_MAX (~2 GiB) — operators wanting more must split
  // requests at the application layer.
  const int64_t max_msg64 = static_cast<int64_t>(max_body_mb) * 1024 * 1024;
  const int max_msg = static_cast<int>(
      std::min<int64_t>(max_msg64, std::numeric_limits<int>::max()));
  const int cqs = cfg.grpc_cqs;

  auto address = std::format("{}:{}", cfg.host, port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  builder.SetMaxReceiveMessageSize(max_msg);
  builder.SetMaxSendMessageSize(max_msg);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, cqs * 2);
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 1);
  builder.AddChannelArgument(GRPC_ARG_MINIMAL_STACK, 1);

  auto server = builder.BuildAndStart();
  std::cout << std::format("gRPC server listening on {} (max_body_mb={})\n",
                            address, max_body_mb);

  auto thread = std::jthread([srv = server.get(), svc = std::move(service)]() {
    srv->Wait();
  });

  return {std::move(server), std::move(thread)};
}

} // namespace detail

#ifndef USE_CPU_ONLY
/// Start gRPC server using a PipelineDispatcher (GPU path).
/// `readiness_check` is invoked from Health() so gRPC probes match
/// HTTP /health/ready behaviour. Pass {} to keep Health unconditionally OK.
inline GrpcHandle start_grpc_server(pipeline::PipelineDispatcher &dispatcher,
                                     const ServerConfig &cfg,
                                     render::PdfRenderer *pdf_renderer = nullptr,
                                     bool layout_available = false,
                                     std::function<bool()> readiness_check = {},
                                     bool table_available = false,
                                     bool formula_available = false) {
  auto service = std::make_shared<OCRServiceImpl>(
      dispatcher, cfg, pdf_renderer, layout_available);
  service->set_readiness_check(std::move(readiness_check));
  service->set_structure_availability(table_available, formula_available);
  return detail::launch_grpc_server(std::move(service), cfg.grpc_port, cfg);
}
#endif

/// Start gRPC server using an InferFunc (CPU path, also usable from GPU).
inline GrpcHandle start_grpc_server(InferFunc infer_fn,
                                     const ServerConfig &cfg,
                                     render::PdfRenderer *pdf_renderer = nullptr,
                                     bool layout_available = false,
                                     std::function<bool()> readiness_check = {},
                                     bool table_available = false,
                                     bool formula_available = false) {
  auto service = std::make_shared<OCRServiceImpl>(
      std::move(infer_fn), cfg, pdf_renderer, layout_available);
  service->set_readiness_check(std::move(readiness_check));
  service->set_structure_availability(table_available, formula_available);
  return detail::launch_grpc_server(std::move(service), cfg.grpc_port, cfg);
}

} // namespace turbo_ocr::server
