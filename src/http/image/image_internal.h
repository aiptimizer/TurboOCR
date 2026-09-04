#pragma once

#include "turbo_ocr/http/image_routes.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/server/server_types.h"

// Internal to src/routes/: the per-endpoint GPU registrars (one TU each) and
// the helper they share. Public consumers use register_image_routes()
// (include/turbo_ocr/http/image_routes.h); these exist so each endpoint
// lives in its own file instead of one god-TU.
namespace turbo_ocr::routes {

// 504 response for a per-request inference deadline (C4). The dispatcher
// abandons the still-running task and throws turbo_ocr::TimeoutError; map it
// to the single shared kInferenceTimeout code so HTTP and gRPC stay in
// lockstep.
[[nodiscard]] inline drogon::HttpResponsePtr timeout_response() {
  using server::ErrorCode;
  return server::error_response(
      static_cast<drogon::HttpStatusCode>(
          server::error_http_status(ErrorCode::kInferenceTimeout)),
      server::error_code_str(ErrorCode::kInferenceTimeout).data(),
      "Inference exceeded the configured request deadline");
}

// The dispatch-and-emit tail every GPU image route repeats: submit to the
// dispatcher under the configured request deadline, map TimeoutError to the
// shared 504, await any deferred VLM crops OFF the GPU worker, emit the
// standard pipeline JSON. `fn` must own its inputs by value (timeout-abandon
// safety — see submit_for_default).
template <typename SubmitFn>
void run_default_and_emit(pipeline::PipelineDispatcher &dispatcher,
                          SubmitFn &&fn, bool want_blocks,
                          server::DrogonCallback &cb) {
  pipeline::OcrPipelineResult out;
  try {
    out = dispatcher.submit_for_default(std::forward<SubmitFn>(fn));
  } catch (const turbo_ocr::TimeoutError &) {
    cb(timeout_response());
    return;
  }
  pipeline::finalize_deferred(out);
  cb(server::json_response(emit_pipeline_result_json(out, want_blocks)));
}

void register_ocr_raw_route_gpu(server::WorkPool &pool,
                                pipeline::PipelineDispatcher &dispatcher,
                                const server::ImageDecoder &decode,
                                bool nvjpeg_available, bool layout_available,
                                bool table_available, bool formula_available);
// `nvjpeg` is the shared decoder pool (nullptr = nvJPEG unavailable, CPU
// decode). The raw route takes only the availability bit: its JPEG fast path
// decodes on the replica's own decoder (pipeline::decode_jpeg_and_run).
void register_ocr_batch_route_gpu(server::WorkPool &pool,
                                  pipeline::PipelineDispatcher &dispatcher,
                                  const server::ImageDecoder &decode,
                                  decode::NvJpegDecoderPool *nvjpeg,
                                  bool layout_available,
                                  bool table_available, bool formula_available,
                                  int max_batch_images);
void register_ocr_pixels_route_gpu(server::WorkPool &pool,
                                   pipeline::PipelineDispatcher &dispatcher,
                                   bool layout_available, bool table_available,
                                   bool formula_available);
void register_ocr_markdown_route_gpu(server::WorkPool &pool,
                                     pipeline::PipelineDispatcher &dispatcher,
                                     const server::ImageDecoder &decode,
                                     bool layout_available,
                                     bool table_available,
                                     bool formula_available);
void register_infer_route_gpu(server::WorkPool &pool,
                              pipeline::PipelineDispatcher &dispatcher,
                              const server::ImageDecoder &decode);

} // namespace turbo_ocr::routes
