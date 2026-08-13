#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>

#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::decode {

// Single source of truth for the MAX_IMAGE_DIM cap (px). Seeded from the env
// on first read; ServerConfig applies the resolved value (env OR the
// --max-image-dim CLI flag) through set_max_image_dim() at bootstrap. Same
// knob across every image-accepting route (HTTP /ocr, /ocr/raw, /ocr/batch,
// /ocr/pixels and gRPC Recognize/RecognizeBatch).
//
// The setter exists because this used to be an env-only function-static while
// the CLI flag only fed /capabilities — an operator hardening a deployment
// with `--max-image-dim 2048` got a capability document SAYING 2048 and an
// enforcement still at 16384. The knob the operator sets and the value the
// guard reads must be one number.
namespace detail {
[[nodiscard]] inline std::atomic<int> &max_image_dim_slot() {
  static std::atomic<int> v{env::env_int("MAX_IMAGE_DIM", 16384, 64, 65535)};
  return v;
}
} // namespace detail

[[nodiscard]] inline int max_image_dim() {
  return detail::max_image_dim_slot().load(std::memory_order_relaxed);
}

// Bootstrap-time override (ServerConfig). Clamped to the same range as the
// env read so a bad flag cannot zero the cap.
inline void set_max_image_dim(int v) {
  detail::max_image_dim_slot().store(std::clamp(v, 64, 65535),
                                     std::memory_order_relaxed);
}

// Pixel-AREA cap (total px = width*height). The per-side MAX_IMAGE_DIM above
// bounds one dimension, but a 16384x16384 image is still ~268 MP -> ~805 MB
// host cv::Mat (and the same in VRAM on the nvJPEG fast path): a tiny solid
// colour PNG advertising huge dimensions is a decompression bomb the per-side
// cap alone does not stop. This rejects such inputs, mirroring the PDF path's
// MAX_PDF_PAGE_PIXELS_MP. MAX_IMAGE_PIXELS_MP is in megapixels; default 128 MP
// — allows large/high-DPI document scans (the OmniDocBench corpus reaches ~62 MP
// and 600-DPI large-format pages go higher) while still rejecting the ~268 MP
// solid-colour decompression bomb; tighten via the env for stricter deployments —
// range [1, 4096]. Same knob across every image-accepting route.
[[nodiscard]] inline int64_t max_image_pixels() {
  static const int64_t v =
      static_cast<int64_t>(env::env_int("MAX_IMAGE_PIXELS_MP", 128, 1, 4096)) *
      1000000;
  return v;
}

// True when width*height exceeds the area cap. The int64 product cannot
// overflow for in-range dims (each <= 65535, so the product <= ~4.3e9).
[[nodiscard]] inline bool exceeds_pixel_cap(int64_t w, int64_t h) {
  return w * h > max_image_pixels();
}

// Aggregate decoded-pixel budget for ONE /ocr/batch request (sum of
// width*height across all decoded slots). The per-image cap above bounds a
// single slot, but /ocr/batch holds every decoded image alive at once, so the
// real peak is per-image-cap * max_batch_images (e.g. 1024 * 40 MP * 3 B
// ~= 122 GB host RAM). A solid-colour PNG/JPEG advertising huge dims is tens
// of KB encoded, so a body well under MAX_BODY_MB can still decode to >100 GB
// at once. This bounds the running sum: sniffable slots past the budget are
// tagged per-slot (never decoded), not whole-request rejected. In megapixels;
// default 2048 MP (~6 GB BGR), range [16, 1048576]. Under the default 100 MB
// body cap no legitimate (incompressible) batch can approach this — only
// highly-compressible bomb inputs can; operators who raise MAX_BODY_MB should
// raise this knob to match. Same knob across both server binaries.
[[nodiscard]] inline int64_t max_batch_pixels() {
  static const int64_t v =
      static_cast<int64_t>(
          env::env_int("MAX_BATCH_PIXELS_MP", 2048, 16, 1048576)) *
      1000000;
  return v;
}

} // namespace turbo_ocr::decode
