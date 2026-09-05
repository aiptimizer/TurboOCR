#include "turbo_ocr/pipeline/jpeg_infer.h"

#include <climits>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/decode/size_classify.h"
#include "turbo_ocr/pipeline/pool/gpu_pipeline_pool.h"

#include "turbo_ocr/decode/jpeg_codec.h"

#include <format>

#include <tuple>

namespace turbo_ocr::pipeline {

namespace {

// The host codec for bitstreams nvJPEG does not decode. Throws on failure.
cv::Mat host_codec_jpeg(const unsigned char *data, size_t len) {
  cv::Mat img;
  if (len <= static_cast<size_t>(INT_MAX)) {
    img = cv::imdecode(cv::Mat(1, static_cast<int>(len), CV_8UC1,
                               const_cast<unsigned char *>(data)),
                       cv::IMREAD_COLOR);
  }
  if (img.empty())
    throw turbo_ocr::ImageDecodeError("Failed to decode JPEG");
  return img;
}

} // namespace

namespace {

// Decode on the GPU into the pipeline's device buffer: hardware backend
// first; a bitstream it reports as unsupported goes to the replica's hybrid
// backend before anything touches the host codec. Returns the final status
// and fills gpu_img on Ok. Throws GpuDecodeError on a device fault.
decode::JpegDecodeStatus decode_to_device(GpuPipelineEntry &e, const unsigned char *data,
                                          size_t len, int w, int h,
                                          turbo_ocr::GpuImage &gpu_img) {
  auto [d_buf, pitch] = e.pipeline->ensure_gpu_buf(h, w);
  auto attempt = [&](decode::NvJpegDecoder &dec) {
    const auto st = dec.decode_to_gpu(data, len, d_buf, pitch, w, h, e.stream);
    if (st == decode::JpegDecodeStatus::Failed)
      throw turbo_ocr::GpuDecodeError(std::format(
          "nvJPEG decode failed (status {})", dec.last_nvjpeg_status()));
    return st;
  };
  auto st = attempt(e.get_nvjpeg());
  if (st == decode::JpegDecodeStatus::Unsupported && e.get_nvjpeg_hybrid().available()) {
    st = attempt(e.get_nvjpeg_hybrid());
    if (st == decode::JpegDecodeStatus::Ok)
      TOCR_LOG_INFO_RL("JPEG outside the hardware decoder's format support decoded on the hybrid GPU backend");
  }
  if (st == decode::JpegDecodeStatus::Ok)
    gpu_img = turbo_ocr::GpuImage{.data = d_buf, .step = pitch, .rows = h, .cols = w};
  return st;
}

} // namespace

OcrPipelineResult decode_jpeg_and_run(GpuPipelineEntry &e,
                                      const unsigned char *data, size_t len,
                                      const JpegRunOpts &o) {
  auto &nvjpeg = e.get_nvjpeg();
  if (nvjpeg.available()) {
    auto [w, h] = nvjpeg.get_dimensions(data, len);
    if (w <= 0 || h <= 0) {
      // The hardware handle could not parse the header; the hybrid one may.
      auto &hy = e.get_nvjpeg_hybrid();
      if (hy.available()) std::tie(w, h) = hy.get_dimensions(data, len);
    }
    // Decompression-bomb guard for JPEGs the caller's pre-decode header sniff
    // couldn't parse (progressive / unusual SOF markers): nvJPEG's own dims
    // are authoritative — reject before allocating GPU memory for them.
    decode::throw_if_image_too_large(w, h);
    if (w > 0 && h > 0) {
      turbo_ocr::GpuImage gpu_img{};
      if (decode_to_device(e, data, len, w, h, gpu_img) == decode::JpegDecodeStatus::Ok) {
        // An inference failure here is an inference failure; it propagates
        // like on every other path instead of re-running the request on a
        // CPU-decoded copy.
        OcrPipelineResult out =
            o.layout_only
                ? e.pipeline->run_layout_only(gpu_img, e.stream)
                : e.pipeline->run_with_layout(
                      gpu_img, e.stream, o.want_layout, o.want_reading_order,
                      o.routing, o.defer_external, o.want_tables,
                      o.want_formulas);
        out.image_cols = w;
        out.image_rows = h;
        return out;
      }
      // Unsupported by both GPU backends: host codec, by specification.
    }
  }
  cv::Mat img = host_codec_jpeg(data, len);
  // Host-codec bomb guard: re-check the decoded size, since
  // get_dimensions={0,0} and the 64KB sniff can both miss.
  decode::throw_if_image_too_large(img.cols, img.rows);
  OcrPipelineResult out =
      o.layout_only
          ? e.pipeline->run_layout_only(img, e.stream)
          : e.pipeline->run_with_layout(img, e.stream, o.want_layout,
                                        o.want_reading_order, o.routing,
                                        o.defer_external, o.want_tables,
                                        o.want_formulas);
  out.image_cols = img.cols;
  out.image_rows = img.rows;
  return out;
}

cv::Mat decode_jpeg_on_replica(GpuPipelineEntry &e, const unsigned char *data,
                               size_t len) {
  auto try_backend = [&](decode::NvJpegDecoder &dec, cv::Mat &out) {
    auto hd = dec.decode(data, len, e.stream);
    if (hd.status == decode::JpegDecodeStatus::Failed)
      throw turbo_ocr::GpuDecodeError(
          std::format("nvJPEG decode failed (status {})", hd.nvjpeg_status));
    if (hd.status == decode::JpegDecodeStatus::Ok) out = std::move(hd.image);
    return hd.status;
  };
  cv::Mat img;
  if (e.get_nvjpeg().available()) {
    auto st = try_backend(e.get_nvjpeg(), img);
    if (st == decode::JpegDecodeStatus::Unsupported && e.get_nvjpeg_hybrid().available()) {
      st = try_backend(e.get_nvjpeg_hybrid(), img);
      if (st == decode::JpegDecodeStatus::Ok)
        TOCR_LOG_INFO_RL("JPEG outside the hardware decoder's format support decoded on the hybrid GPU backend");
    }
    if (st == decode::JpegDecodeStatus::Ok) return img;
  }
  return host_codec_jpeg(data, len);
}

} // namespace turbo_ocr::pipeline
