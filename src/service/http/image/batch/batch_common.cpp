
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>
#include <optional>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/image/image_dims.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/size_classify.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/service/server/error_codes.h"
#include "turbo_ocr/service/validation/request_gate.h"
#include "turbo_ocr/service/validation/pixel_dims.h"


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
// `max_image_dim` is vestigial: every caller passes decode::max_image_dim(),
// and the shared classifier and formatter read that same function-static
// themselves, so a caller cannot hand this stage a cap it will not enforce.
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
      // Shared verdict AND shared text (decode/size_classify.h); the per-slot
      // snake_case error strings are the batch wire contract, distinct from the
      // whole-request 400 messages, so they have their own formatter there
      // rather than a std::format spelled out at each stage.
      const auto v = decode::classify_image_size(d->width, d->height);
      if (v != decode::ImageSizeVerdict::kOk) {
        errors[i] = decode::image_size_slot_error(v, d->width, d->height);
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

// Stage 4: post-decode safety net for residual formats we don't header-sniff
// (BMP/PNM). Releases the image so it doesn't get fed to the pipeline.
// `max_image_dim` is vestigial here for the same reason as the pre-decode stage.
void batch_check_dims_post(std::vector<cv::Mat> &imgs,
                            int max_image_dim,
                            std::vector<std::string> &errors) {
  size_t n = imgs.size();
  for (size_t i = 0; i < n; ++i) {
    if (!errors[i].empty()) continue;
    const auto v = decode::classify_image_size(imgs[i].cols, imgs[i].rows);
    if (v != decode::ImageSizeVerdict::kOk) {
      errors[i] = decode::image_size_slot_error(v, imgs[i].cols, imgs[i].rows);
      imgs[i].release();
    }
  }
}

// Stage 6: serialize {batch_results, errors} JSON. Mirrors the CPU contract
// — null in the errors array for successful slots, an error string otherwise.
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
    auto &out = all_items[i].out;
    // Full per-page emitter: text + layout (+ tables/formulas when the CUA
    // router fired, byte-identical to the legacy shape otherwise). It is also
    // the ONLY emitter that writes the additive *_degraded / *_warning keys,
    // so take it whenever a slot carries structure or a degradation signal —
    // not just when layout was asked for. Without that guard a page where
    // detection found N boxes and recognition produced nothing serialized as
    // {"results":[]}, byte-identical to a genuinely blank page, which is the
    // exact condition text_degraded exists to report. Same guard as
    // server::emit_infer_result_json (server/infer_result.h) and
    // pipeline::serialize_page_results (pdf/pdf_job_pages.cpp).
    //
    // Clean path unchanged: no layout, no structure, no degradation -> the
    // text-only results_to_json bytes exactly as before.
    if (want_layout || out.text_degraded || out.table_degraded ||
        out.formula_degraded || !out.tables.empty() || !out.formulas.empty()) {
      json_str += emit_pipeline_result_json(out, want_blocks);
    } else {
      json_str += results_to_json(out.results);
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
