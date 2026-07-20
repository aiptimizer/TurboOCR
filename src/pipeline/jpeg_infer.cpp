#include "turbo_ocr/pipeline/jpeg_infer.h"

#include <climits>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/decode/size_classify.h"
#include "turbo_ocr/pipeline/pool/gpu_pipeline_pool.h"

namespace turbo_ocr::pipeline {

OcrPipelineResult decode_jpeg_and_run(GpuPipelineEntry &e,
                                      const unsigned char *data, size_t len,
                                      const JpegRunOpts &o) {
  auto &nvjpeg = e.get_nvjpeg();
  // layout_only takes the cv::Mat decode below: run_layout_only has no
  // GpuImage overload, and the layout-only pass is cheap enough that the
  // zero-copy device-decode fast path buys nothing.
  if (!o.layout_only && nvjpeg.available()) {
    auto [w, h] = nvjpeg.get_dimensions(data, len);
    // Decompression-bomb guard for JPEGs the caller's pre-decode header sniff
    // couldn't parse (progressive / unusual SOF markers): nvJPEG's own dims
    // are authoritative — reject before allocating GPU memory for them.
    decode::throw_if_image_too_large(w, h);
    if (w > 0 && h > 0) {
      auto [d_buf, pitch] = e.pipeline->ensure_gpu_buf(h, w);
      if (nvjpeg.decode_to_gpu(data, len, d_buf, pitch, w, h, e.stream)) {
        turbo_ocr::GpuImage gpu_img{
            .data = d_buf, .step = pitch, .rows = h, .cols = w};
        try {
          return e.pipeline->run_with_layout(
              gpu_img, e.stream, o.want_layout, o.want_reading_order,
              o.routing, o.defer_external, o.want_tables, o.want_formulas);
        } catch (const std::exception &ex) {
          // Best-effort GPU zero-copy fast path: fall through to the CPU
          // decode + retry below (which re-runs run_with_layout and surfaces
          // a genuine error there) rather than failing here. Don't silently
          // re-run ~2x work invisibly on HTTP — rate-limited log so a
          // persistent GPU-path fault stays visible.
          if (o.log_fastpath_errors)
            TOCR_LOG_ERROR_RL("nvJPEG GPU fast-path failed; CPU fallback",
                              "error", ex.what());
        }
      }
    }
  }
  cv::Mat img = nvjpeg.decode(data, len);
  if (img.empty() && len <= static_cast<size_t>(INT_MAX)) {
    img = cv::imdecode(cv::Mat(1, static_cast<int>(len), CV_8UC1,
                               const_cast<unsigned char *>(data)),
                       cv::IMREAD_COLOR);
  }
  if (img.empty())
    throw turbo_ocr::ImageDecodeError("Failed to decode JPEG");
  // CPU-fallback / nvjpeg-unavailable bomb guard: re-check the decoded size,
  // since get_dimensions={0,0} and the 64KB sniff can both miss.
  decode::throw_if_image_too_large(img.cols, img.rows);
  if (o.layout_only)
    return e.pipeline->run_layout_only(img, e.stream);
  return e.pipeline->run_with_layout(img, e.stream, o.want_layout,
                                     o.want_reading_order, o.routing,
                                     o.defer_external, o.want_tables,
                                     o.want_formulas);
}

} // namespace turbo_ocr::pipeline
