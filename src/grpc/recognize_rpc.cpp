// gRPC Recognize: single image / raw pixels, layout flags, routing.
#include "turbo_ocr/grpc/grpc_service.h"

namespace turbo_ocr::server {

grpc::Status OCRServiceImpl::Recognize(grpc::ServerContext *ctx,
                       const ocr::OCRRequest *request,
                       ocr::OCRResponse *response) {
  backend_routing::RequestRouting routing;
  if (auto err = grpc_validate_routing(ctx, request->route_table(),
                                       request->route_formula(), &routing);
      err)
    return *err;
  const bool layout_only = request->layout_only();
  if (layout_only) {
#ifdef USE_CPU_ONLY
    // Same contract as HTTP: parse_query_options rejects text=0 on the CPU
    // build (no layout-only pipeline path).
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "INVALID_PARAMETER",
                      "layout_only is not supported on the CPU build");
#else
    // Mirrors the HTTP text=0 combination rules: everything text-derived is
    // meaningless without recognition — fail loud, never silently empty.
    if (request->reading_order() || request->as_blocks() ||
        request->tables() || request->formulas())
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                        "INVALID_PARAMETER",
                        "layout_only returns layout regions only; it cannot "
                        "be combined with reading_order/as_blocks/tables/"
                        "formulas");
#endif
  }
  if (auto err = grpc_check_layout_request(ctx,
          request->layout() || layout_only,
          request->reading_order() || request->as_blocks() ||
          request->tables() || request->formulas(),
          layout_available_); err)
    return *err;
  bool want_layout = request->layout() || layout_only;
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
          mode_ == GrpcResponseMode::json_bytes,
          request->layout() || layout_only, request->as_blocks()); err)
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

    if (auto st = grpc_check_image_size(ctx, width, height)) return *st;

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
                           want_tables, want_formulas, routing, layout_only);
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
                                               want_tables, want_formulas,
                                               routing, layout_only);
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

  if (auto st = grpc_check_image_size(ctx, img.cols, img.rows)) return *st;

  try {
    auto out = run_infer(img, want_layout, want_reading_order,
                         want_tables, want_formulas, routing, layout_only);
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

} // namespace turbo_ocr::server
