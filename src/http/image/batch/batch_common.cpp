
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/markdown/markdown_export.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/error_codes.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/validation/pixel_dims.h"


#include "batch_internal.h"

namespace turbo_ocr::routes::batchdetail {

void batch_decode_base64(const std::vector<std::string> &b64_strings,
                          std::vector<std::string> &raw_bytes,
                          std::vector<std::string> &errors) {
  size_t n = b64_strings.size();
  for (size_t i = 0; i < n; ++i) {
    const auto &b64 = b64_strings[i];
    if (b64.empty()) {
      errors[i] = "empty";
      continue;
    }
    raw_bytes[i] = base64_decode(b64);
    if (raw_bytes[i].empty()) errors[i] = "base64_decode_failed";
  }
}

// Stage 2: header-sniff (PNG/JPEG) every still-valid slot and reject
// oversized inputs before paying decode cost. Same MAX_IMAGE_DIM env as
// /ocr/raw and /ocr/pixels — but errors are per-slot, not whole-request 400s.
void batch_check_dims_pre(const std::vector<std::string> &raw_bytes,
                           int max_image_dim,
                           std::vector<std::string> &errors) {
  size_t n = raw_bytes.size();
  // Aggregate decoded-pixel budget: the per-image cap below bounds one slot,
  // but the route holds every decoded image alive at once, so a batch of
  // highly-compressible bomb images can still OOM the host. Tag sniffable
  // slots once the running sum would exceed the budget so they are never
  // decoded. Covers every compressible bomb vector (PNG/JPEG/WebP/TIFF/GIF);
  // residual uncompressed formats (BMP/PNM) are body-cap-bounded and fall to
  // the per-image post-decode cap.
  int64_t cumulative_pixels = 0;
  const int64_t batch_pixel_budget = decode::max_batch_pixels();
  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    const auto &raw = raw_bytes[i];
    if (auto d = turbo_ocr::decode::peek_image_dimensions(
            reinterpret_cast<const unsigned char *>(raw.data()), raw.size())) {
      // Shared verdict (decode/size_classify.h); the per-slot snake_case
      // error strings are the batch wire contract, distinct from the
      // whole-request 400 messages.
      const auto v = decode::classify_image_size(d->width, d->height);
      if (v == decode::ImageSizeVerdict::kDimTooLarge) {
        errors[i] = std::format("dimensions_too_large ({}x{} > {}x{})",
                                 d->width, d->height,
                                 max_image_dim, max_image_dim);
      } else if (v == decode::ImageSizeVerdict::kPixelsTooLarge) {
        errors[i] = std::format("pixels_too_large ({}x{} > {} px)",
                                 d->width, d->height, decode::max_image_pixels());
      } else {
        const int64_t pix = static_cast<int64_t>(d->width) * d->height;
        if (cumulative_pixels + pix > batch_pixel_budget) {
          errors[i] = std::format("batch_pixels_exceeded (batch sum > {} px)",
                                   batch_pixel_budget);
        } else {
          cumulative_pixels += pix;
        }
      }
    }
  }
}

// Stage 3: decode pixels. JPEGs go through nvJPEG batch decode (when
// available and we have ≥2 of them); everything else runs through the
void batch_check_dims_post(std::vector<cv::Mat> &imgs,
                            int max_image_dim,
                            std::vector<std::string> &errors) {
  size_t n = imgs.size();
  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    const auto v = decode::classify_image_size(imgs[i].cols, imgs[i].rows);
    if (v == decode::ImageSizeVerdict::kDimTooLarge) {
      errors[i] = std::format("dimensions_too_large ({}x{} > {}x{})",
                               imgs[i].cols, imgs[i].rows,
                               max_image_dim, max_image_dim);
      imgs[i].release();
    } else if (v == decode::ImageSizeVerdict::kPixelsTooLarge) {
      errors[i] = std::format("pixels_too_large ({}x{} > {} px)",
                               imgs[i].cols, imgs[i].rows,
                               decode::max_image_pixels());
      imgs[i].release();
    }
  }
}

// Stage 5: run the pipeline on every slot that survived decode. Chunks
// into kMaxBatch-sized batches through run_batch_with_layout — both with
// and without layout the batched path applies (batched det/rec), so
// ?layout=1 no longer falls back to serial single-image inference.
//
// C4: the submitted lambda is SELF-CONTAINED. It owns the decoded images
// (moved in) and writes results into the packaged_task's own return value,
// never into request-scoped state. So if the future is abandoned on timeout
// (TimeoutError), the still-running task can finish safely against memory it
// owns; the caller scatters the returned results back only after the future
// resolves in time. Results are chunk-local (indexed 0..valid count); the
std::string batch_emit_json(std::vector<BatchItem> &all_items,
                             const std::vector<std::string> &errors,
                             bool want_layout,
                             bool want_blocks) {
  size_t n = all_items.size();
  std::string json_str;
  json_str.reserve(n * 1024);
  json_str += "{\"batch_results\":[";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) json_str += ',';
    if (want_layout) {
      // Full per-page emitter: text + layout (+ tables/formulas when the
      // CUA router fired, byte-identical to the legacy shape otherwise).
      json_str += emit_pipeline_result_json(all_items[i].out, want_blocks);
    } else {
      json_str += results_to_json(all_items[i].out.results);
    }
  }
  json_str += "],\"errors\":[";
  for (size_t i = 0; i < n; ++i) {
    if (i > 0) json_str += ',';
    const auto &e = errors[i];
    if (e.empty()) {
      json_str += "null";
    } else {
      // Error strings can carry exception text — route them through the
      // shared JSON escaper (control chars, UTF-8 backstop), same as the CPU
      // batch route.
      json_str += '"';
      detail::append_escaped_string(json_str, e);
      json_str += '"';
    }
  }
  json_str += "]}";
  return json_str;
}



} // namespace turbo_ocr::routes::batchdetail
