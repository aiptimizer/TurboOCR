#include "nvidia/support/nv_image_decode.h"

#include <opencv2/core.hpp>

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/image/cpu_image_decode.h"   // decode::decode_cpu_fallback
#include "nvidia/support/nvjpeg_decoder.h"          // decode::NvJpegDecoder

namespace turbo_ocr::nvidia {

namespace {
// Per-thread decoder state. LOAD-BEARING: nvJPEG handles are not safe to share
// across threads, and the WorkPool threads decode concurrently. Ported verbatim
// from the deleted src/cuda/server/stages_gpu.cpp, where a dedicated regression
// (tools/checks/test_nvjpeg_race.cpp) exists for exactly this.
thread_local decode::NvJpegDecoder tl_nvjpeg;
} // namespace

bool probe_nvjpeg() {
  TOCR_LOG_INFO("Initializing nvJPEG decoders");
  const bool nvjpeg_available = tl_nvjpeg.available();
  if (nvjpeg_available)
    TOCR_LOG_INFO("nvJPEG GPU-accelerated JPEG decode enabled");
  else
    TOCR_LOG_WARN("nvJPEG not available, using OpenCV JPEG decode");
  return nvjpeg_available;
}

server::ImageDecoder make_nv_image_decoder(bool nvjpeg_available) {
  // JPEG via nvJPEG (GPU), everything else (PNG via Wuffs, WebP/BMP/TIFF/GIF
  // via cv::imdecode) through the shared host fallback.
  return [nvjpeg_available](const unsigned char *data,
                            std::size_t len) -> cv::Mat {
    if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8 && nvjpeg_available) {
      cv::Mat img = tl_nvjpeg.decode(data, len);
      if (!img.empty()) return img;
    }
    return decode::decode_cpu_fallback(data, len);
  };
}

} // namespace turbo_ocr::nvidia
