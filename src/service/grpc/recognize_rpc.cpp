// gRPC Recognize: single image / raw pixels, layout flags, routing.
#include "turbo_ocr/service/capability/proto_capability_bridge.h"
#include "turbo_ocr/service/grpc/grpc_service.h"

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
  // THE SHARED GATE. Capability sweep, reading_order/as_blocks implications,
  // the availability rejection and every text=0 (here: layout_only)
  // combination rule come from parse_options_core — the same function the HTTP
  // routes run. What stood here instead was grpc_check_layout_request plus a
  // hand-written copy of the layout_only rules; both have been deleted.
  InferOptions opts;
  if (auto r = parse_proto_options(*request, layout_only, loaded_, &opts);
      !r.error.empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      r.error_code.c_str(), r.error);
  const bool want_reading_order = opts.want_reading_order;
  const bool want_blocks = opts.want_blocks;
  const bool want_layout = opts.want_layout;
  const bool want_tables = opts.want_tables;
  const bool want_formulas = opts.want_formulas;
  // The one genuinely gRPC-specific gate: structured response mode carries no
  // table/formula/layout/blocks message, so running those stages and dropping
  // the output would be a silent failure. `raw_layout` is the flag AS SENT, not
  // the implied value — see the declaration.
  if (auto err = grpc_check_structure_backends(
          ctx, opts.requested, loaded_, mode_ == GrpcResponseMode::json_bytes,
          want_blocks, /*raw_layout=*/request->layout() || layout_only); err)
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

    return guarded_infer(ctx, "gRPC pixels inference error", [&] {
      auto out = run_infer(img, want_layout, want_reading_order, want_tables,
                           want_formulas, routing, layout_only);
      fill_response(response, out, want_blocks);
    });
  }

  // Image path: encoded image bytes
  if (request->image().empty()) [[unlikely]]
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "MISSING_IMAGE", "Empty image");

  if (auto err = grpc_pre_decode_dim_check(ctx, request->image()); err)
    return *err;

  // Device-decode fast path: hand the pipeline the STILL-ENCODED bytes so a
  // backend with an on-device decoder never pays a host decode plus a
  // full-frame H2D (a ~200 KB JPEG would otherwise arrive as ~25 MB of pixels).
  // Backends without one decode on the host inside the pipeline, so this is
  // safe to prefer for every format — which is why it replaced the JPEG sniff
  // that used to guard the (now-deleted) dispatcher branch here. layout_only is
  // excluded: it has no EncodedInferFunc equivalent and run_infer asserts on it.
  if (encoded_infer_fn_ && !layout_only) {
    return guarded_infer(ctx, "gRPC encoded inference error", [&] {
      auto out = run_infer_encoded(
          reinterpret_cast<const std::uint8_t *>(request->image().data()),
          request->image().size(), want_layout, want_reading_order,
          want_tables, want_formulas, routing);
      fill_response(response, out, want_blocks);
    });
  }

  // Host-decode path: decode on this thread, then hand the materialized
  // cv::Mat to the pipeline.
  cv::Mat img = grpc_decode_image(request->image());
  if (img.empty()) [[unlikely]]
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "IMAGE_DECODE_FAILED", "Decode failed");

  if (auto st = grpc_check_image_size(ctx, img.cols, img.rows)) return *st;

  return guarded_infer(ctx, "gRPC inference error", [&] {
    auto out = run_infer(img, want_layout, want_reading_order, want_tables,
                         want_formulas, routing, layout_only);
    fill_response(response, out, want_blocks);
  });
}

} // namespace turbo_ocr::server
