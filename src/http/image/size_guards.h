#pragma once

#include <drogon/HttpRequest.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/size_classify.h"
#include "turbo_ocr/server/http_responses.h"

// Internal to src/http/image/: the decompression-bomb size guards shared by
// the image routes. Classification and message text live in
// decode/size_classify.h (one source for every transport); this is the
// HTTP-callback emission adapter.
namespace turbo_ocr::routes {

// Emit the 400 for an oversized image. Returns true if the request was
// rejected (caller should return).
[[nodiscard]] inline bool reject_image_size(int w, int h,
                                            server::DrogonCallback &cb) {
  auto v = decode::classify_image_size(w, h);
  if (v == decode::ImageSizeVerdict::kOk) return false;
  cb(server::error_response(drogon::k400BadRequest,
                            decode::image_size_error_code(v),
                            decode::image_size_error_message(v, w, h)));
  return true;
}

// Two-stage check for the image routes:
//   1. Pre-decode header sniff (PNG / JPEG): refuses oversized inputs
//      without ever calling the decoder, defending against decompression
//      bombs (a 1 KB PNG can claim 100k×100k → 30 GB decode buffer).
//   2. Post-decode check on the resulting cv::Mat: catches formats we
//      don't sniff (BMP, TIFF, WEBP).
// Returns true if the request was rejected (caller should return).
[[nodiscard]] inline bool reject_if_too_large_pre(
    const unsigned char *data, size_t len, server::DrogonCallback &cb) {
  if (auto d = decode::peek_image_dimensions(data, len))
    return reject_image_size(d->width, d->height, cb);
  return false;
}
[[nodiscard]] inline bool reject_if_too_large_post(
    const cv::Mat &img, server::DrogonCallback &cb) {
  return reject_image_size(img.cols, img.rows, cb);
}

} // namespace turbo_ocr::routes
