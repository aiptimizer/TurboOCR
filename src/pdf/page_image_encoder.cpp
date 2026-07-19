#include "turbo_ocr/pdf/page_image_encoder.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// libjpeg-turbo C API — faster than OpenCV's JPEG path for BGR images.
#include <turbojpeg.h>

#ifndef USE_CPU_ONLY
#include "turbo_ocr/encode/nvjpeg_encoder.h"
#endif

namespace turbo_ocr::pdf {

#ifndef USE_CPU_ONLY
namespace {
// GPU JPEG encode is opt-out via TURBO_PDF_IMAGE_ENCODER=cpu (validated to
// {cpu, gpu} at startup by ServerConfig). Default is the GPU path so a
// CPU-starved host offloads entropy coding off its scarce core.
bool gpu_jpeg_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("TURBO_PDF_IMAGE_ENCODER");
    return !(e && std::strcmp(e, "cpu") == 0);
  }();
  return enabled;
}

// Encode BGR -> JPEG on the GPU. One encoder per worker thread (nvJPEG encoder
// state is single-threaded). Returns empty on any failure -> CPU fallback.
std::vector<uint8_t> encode_jpeg_gpu(const cv::Mat &bgr, int quality) {
  static thread_local turbo_ocr::encode::NvJpegEncoder enc;
  if (!enc.available()) return {};
  auto out = enc.encode(bgr, quality);
  static std::atomic<bool> logged{false};
  if (!out.empty() && !logged.exchange(true))
    std::cerr << "[NvJpegEnc] page-image JPEG encode running on GPU (nvJPEG)\n";
  return out;
}

void note_cpu_jpeg_fallback() {
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true))
    std::cerr << "[NvJpegEnc] page-image JPEG encode using CPU (libjpeg-turbo)\n";
}
} // namespace
#endif


PageImageFormat parse_page_image_format(const char *s) noexcept {
  if (!s) return PageImageFormat::Png;
  // Case-insensitive prefix comparison
  auto eq = [](const char *a, const char *b) noexcept {
    while (*a && *b) {
      char ca = static_cast<char>((*a >= 'A' && *a <= 'Z') ? *a + 32 : *a);
      char cb = static_cast<char>((*b >= 'A' && *b <= 'Z') ? *b + 32 : *b);
      if (ca != cb) return false;
      ++a; ++b;
    }
    return *a == '\0' && *b == '\0';
  };
  if (eq(s, "jpeg") || eq(s, "jpg")) return PageImageFormat::Jpeg;
  if (eq(s, "png"))                   return PageImageFormat::Png;
  if (eq(s, "webp"))                  return PageImageFormat::WebP;
  return PageImageFormat::Png;
}

bool is_valid_page_image_format(const char *s) noexcept {
  if (!s || !*s) return false;
  // Round-trip through the lenient parser: a value is valid iff parsing it
  // does not merely hit the Png fallback.
  const PageImageFormat f = parse_page_image_format(s);
  if (f != PageImageFormat::Png) return true;
  auto lower_eq = [](const char *a, const char *b) noexcept {
    while (*a && *b) {
      char ca = static_cast<char>((*a >= 'A' && *a <= 'Z') ? *a + 32 : *a);
      if (ca != *b) return false;
      ++a; ++b;
    }
    return *a == '\0' && *b == '\0';
  };
  return lower_eq(s, "png");
}

const char *page_image_format_name(PageImageFormat fmt) noexcept {
  switch (fmt) {
    case PageImageFormat::Jpeg: return "jpeg";
    case PageImageFormat::Png:  return "png";
    case PageImageFormat::WebP: return "webp";
  }
  return "jpeg";
}

const char *page_image_content_type(PageImageFormat fmt) noexcept {
  switch (fmt) {
    case PageImageFormat::Jpeg: return "image/jpeg";
    case PageImageFormat::Png:  return "image/png";
    case PageImageFormat::WebP: return "image/webp";
  }
  return "image/jpeg";
}

// Downscale `src` so its larger dimension is at most `max_side`.
// Returns a reference to src if no resize is needed (no copy).
static cv::Mat maybe_resize(const cv::Mat &src, int max_side, cv::Mat &tmp) {
  if (max_side <= 0) return src;
  int largest = std::max(src.cols, src.rows);
  if (largest <= max_side) return src;
  double scale = static_cast<double>(max_side) / largest;
  int nw = static_cast<int>(src.cols * scale);
  int nh = static_cast<int>(src.rows * scale);
  if (nw < 1) nw = 1;
  if (nh < 1) nh = 1;
  cv::resize(src, tmp, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
  return tmp;
}

std::vector<uint8_t> encode_page_image(const cv::Mat &bgr,
                                        const EncodeOptions &opts) {
  if (bgr.empty()) return {};

  cv::Mat resized_tmp;
  const cv::Mat &src = maybe_resize(bgr, opts.max_side, resized_tmp);

  if (opts.format == PageImageFormat::Jpeg) {
#ifndef USE_CPU_ONLY
    // GPU path: nvJPEG entropy-codes on the device, leaving the CPU core free
    // for PDF rasterization. Falls through to libjpeg-turbo on any failure.
    if (gpu_jpeg_enabled()) {
      std::vector<uint8_t> gpu_out = encode_jpeg_gpu(src, opts.quality);
      if (!gpu_out.empty()) return gpu_out;
    }
    note_cpu_jpeg_fallback();
#endif
    // libjpeg-turbo fast path: BGR → JPEG in one call, no intermediate copy.
    tjhandle tj = tjInitCompress();
    if (!tj) return {};

    unsigned char *out_buf = nullptr;
    unsigned long out_size = 0;
    int stride = static_cast<int>(src.step[0]);

    int rc = tjCompress2(
        tj,
        src.data,
        src.cols, stride, src.rows,
        TJPF_BGR,
        &out_buf, &out_size,
        TJSAMP_420,
        opts.quality,
        TJFLAG_FASTDCT);

    tjDestroy(tj);
    if (rc != 0 || out_buf == nullptr) {
      if (out_buf) tjFree(out_buf);
      return {};
    }

    std::vector<uint8_t> result(out_buf, out_buf + out_size);
    tjFree(out_buf);
    return result;
  }

  // PNG and WebP via OpenCV. (A SIMD PNG encoder like fpnge was evaluated and
  // rejected: it encodes faster but produces ~2× larger files, and since the
  // image is base64-embedded in the JSON response, payload size — not encode
  // CPU — dominates end-to-end. For small lossless output prefer WebP, which
  // is ~3× smaller than PNG here.)
  std::vector<int> params;
  std::string ext;
  if (opts.format == PageImageFormat::Png) {
    ext = ".png";
    // PNG compression level: 0=fastest/biggest, 9=slowest/smallest.
    // Default 3 = good speed-size sweet spot.
    int comp = std::clamp(opts.png_compression, 0, 9);
    params = {cv::IMWRITE_PNG_COMPRESSION, comp};
  } else {
    ext = ".webp";
    if (opts.lossless) {
      // OpenCV convention: WEBP_QUALITY=101 triggers libwebp lossless mode.
      // Bit-exact reconstruction; slower than lossy but pixel-perfect.
      params = {cv::IMWRITE_WEBP_QUALITY, 101};
    } else {
      params = {cv::IMWRITE_WEBP_QUALITY, opts.quality};
    }
  }

  std::vector<uint8_t> buf;
  if (!cv::imencode(ext, src, buf, params)) return {};
  return buf;
}

} // namespace turbo_ocr::pdf
