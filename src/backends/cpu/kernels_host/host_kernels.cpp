// HostKernels implementation — OpenCV on the host. Each op reproduces the
// existing CPU-path preprocessing/postprocessing so the ONE pipeline gets
// identical numbers whether it drives this backend or a device one.

#include "cpu/kernels_host/host_kernels.h"

#include <algorithm>
#include <cstring>

#include <opencv2/imgproc.hpp>

#include "cpu/support/host_common.h" // to_mat

#include "turbo_ocr/core/norm_params.h"        // SHARED norm factories
#include "turbo_ocr/image/cpu_image_decode.h"   // decode::decode_cpu_fallback
#include "turbo_ocr/analysis/detection/det_postprocess.h" // extract_boxes_from_bitmap

namespace turbo_ocr::cpu {

namespace {

// Output plane c reads which channel of a BGR cv::Mat, given the requested
// channel order. RGB: plane0=R(ch2), plane1=G(ch1), plane2=B(ch0). BGR: identity.
[[nodiscard]] int src_channel_for(backend::ChannelOrder order, int c) noexcept {
  return order == backend::ChannelOrder::RGB ? (2 - c) : c;
}

// Write a BGR8 (or float) `canvas` into a caller-owned CHW float tensor of size
// [3, th, tw] applying out = (pixel*inv_scale - mean[c]) * inv_std[c] per plane,
// folded into a single convertTo (alpha, beta). Columns beyond `content_w`
// (when the canvas is narrower than tw) are left as the caller initialized them.
void write_chw_normalized(const cv::Mat &canvas, float *dst_chw, int tw, int th,
                          int content_w, const backend::NormParams &params) {
  cv::Mat ch[3];
  cv::split(canvas, ch); // ch[0]=B, ch[1]=G, ch[2]=R
  const int copy_w = std::min(content_w, tw);
  cv::Mat planef;
  for (int c = 0; c < 3; ++c) {
    const double alpha = static_cast<double>(params.inv_scale) * params.inv_std[c];
    const double beta = -static_cast<double>(params.mean[c]) * params.inv_std[c];
    ch[src_channel_for(params.order, c)].convertTo(planef, CV_32F, alpha, beta);
    float *plane = dst_chw + static_cast<std::size_t>(c) * th * tw;
    const int rows = std::min(th, planef.rows);
    for (int r = 0; r < rows; ++r)
      std::memcpy(plane + static_cast<std::size_t>(r) * tw, planef.ptr<float>(r),
                  static_cast<std::size_t>(copy_w) * sizeof(float));
  }
}

// Normalization comes from the SHARED factories — never re-typed here. The
// local imagenet_bgr() this file used to define was one of eight copies of the
// same six floats (see turbo_ocr/core/norm_params.h for why that mattered).
using backend::norm::imagenet_bgr;

} // namespace

backend::KernelCaps HostKernels::caps() const {
  backend::KernelCaps c;
  c.device = backend::DeviceKind::Host;
  // The host runs everything natively — it is the fallback target itself.
  c.decode_image = true;
  c.resize_normalize = true;
  c.warp_crops = true;
  c.threshold = true;
  c.db_postprocess = true;
  c.argmax = true;
  c.preprocess_region = true;
  // PARAMETER CONTRACT: OpenCV honours every field, including letterbox (this
  // is the only backend that does — see the letterbox note in kernels.h; the
  // shared callers all pass false so the four backends stay in step).
  c.params.norm_mean_std = true;
  // Stated explicitly rather than left to the struct default: the two per-path
  // flags are the ones a backend with a baked-in kernel turns off, and OpenCV
  // takes real parameters on BOTH paths (write_chw_normalized folds mean/std/
  // inv_scale into one convertTo for the full-frame resize and the warp alike).
  c.params.norm_mean_std_full_frame = true;
  c.params.norm_mean_std_warp = true;
  c.params.norm_channel_order = true;
  c.params.norm_letterbox = true;
  c.params.db_oriented = true;
  // extract_boxes_from_bitmap is minAreaRect-based (rotated quads), but
  // oriented=false is the portable mode every backend must serve (the CUDA
  // path is AABB-only), so db_postprocess reduces each quad to its bounding
  // rect for that mode instead of refusing. Declaring false here made the
  // guard return an EMPTY list for oriented=false — indistinguishable from
  // "this page has no text" (caught by the db_postprocess parity test).
  c.params.db_axis_aligned = true;
  // No per-component expand clamp in the contour path (Clipper offsets by the
  // full unclip ratio); min/max_expand are CCL-path concepts.
  c.params.db_expand_limits = false;
  c.params.db_side_limits = true;
  // The contour path has no component budget — it processes every contour.
  c.params.db_max_components = false;
  return c;
}

backend::ImageView HostKernels::decode_image(const std::uint8_t *data,
                                             std::size_t len,
                                             backend::DeviceQueue & /*queue*/) {
  decoded_ = decode::decode_cpu_fallback(data, len); // owns the pixels
  if (decoded_.empty())
    return backend::ImageView{};
  return to_image_view(decoded_);
}

void HostKernels::resize_normalize(const backend::ImageView &src, float *dst_chw,
                                   int dst_w, int dst_h,
                                   const backend::NormParams &params,
                                   backend::DeviceQueue & /*queue*/) {
  // PARAMETER CONTRACT (kernels.h): honour every field or refuse loudly.
  // NormPath::FullFrame is what makes caps().params.norm_mean_std_full_frame
  // load-bearing — omit it and the guard silently checks only the generic
  // norm_mean_std, so the per-path claim above would never be read. The host
  // declares it true, so nothing new is refused here.
  if (!backend::require_norm_supported(params, caps().params,
                                       "HostKernels::resize_normalize",
                                       backend::NormPath::FullFrame))
    return;
  cv::Mat srcMat = to_mat(src);
  if (srcMat.empty() || dst_w <= 0 || dst_h <= 0)
    return;

  cv::Mat canvas;
  int content_w = dst_w;
  if (params.letterbox) {
    // Preserve aspect ratio, top-left place, pad remainder with 0 pixels (so a
    // padded pixel normalizes to -mean*inv_std, matching the det pad-then-norm).
    const double scale =
        std::min(static_cast<double>(dst_w) / srcMat.cols,
                 static_cast<double>(dst_h) / srcMat.rows);
    const int rw = std::max(1, static_cast<int>(srcMat.cols * scale));
    const int rh = std::max(1, static_cast<int>(srcMat.rows * scale));
    cv::Mat resized;
    cv::resize(srcMat, resized, cv::Size(rw, rh));
    cv::copyMakeBorder(resized, canvas, 0, dst_h - rh, 0, dst_w - rw,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  } else {
    cv::resize(srcMat, canvas, cv::Size(dst_w, dst_h));
  }
  write_chw_normalized(canvas, dst_chw, dst_w, dst_h, content_w, params);
}

void HostKernels::warp_crops(const backend::ImageView &src,
                             const float *d_M_invs, const int *d_crop_widths,
                             float *d_dst_batch, int batch_size, int dst_h,
                             int dst_w, const backend::NormParams &params,
                             backend::DeviceQueue & /*queue*/) {
  // Same contract, NormPath::Warp — see resize_normalize above.
  if (!backend::require_norm_supported(params, caps().params,
                                       "HostKernels::warp_crops",
                                       backend::NormPath::Warp))
    return;
  cv::Mat srcMat = to_mat(src);
  const std::size_t slot_elems =
      static_cast<std::size_t>(3) * dst_h * dst_w;
  for (int i = 0; i < batch_size; ++i) {
    float *slot = d_dst_batch + static_cast<std::size_t>(i) * slot_elems;
    // Padded columns (and any empty crop) stay zero (mid-gray in norm space).
    std::memset(slot, 0, slot_elems * sizeof(float));
    if (srcMat.empty())
      continue;

    const int content_w = std::min(d_crop_widths[i], dst_w);
    if (content_w <= 0)
      continue;

    const float *m = d_M_invs + static_cast<std::size_t>(i) * 9;
    const cv::Matx33f m_inv(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);

    // Single inverse warp straight to the recognizer input height — the exact
    // dst->src mapping OrtPaddleRec::preprocess_box / the GPU roi_warp evaluate.
    cv::Mat warped;
    cv::warpPerspective(srcMat, warped, m_inv, cv::Size(content_w, dst_h),
                        cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                        cv::BORDER_REPLICATE);
    write_chw_normalized(warped, slot, dst_w, dst_h, content_w, params);
  }
}

void HostKernels::threshold(const float *src, std::uint8_t *dst, int w, int h,
                            int batch_size, float thresh,
                            backend::DeviceQueue & /*queue*/) {
  const std::size_t per = static_cast<std::size_t>(w) * h;
  for (int b = 0; b < batch_size; ++b) {
    const cv::Mat in(h, w, CV_32F, const_cast<float *>(src + b * per));
    cv::Mat out(h, w, CV_8U, dst + b * per);
    // cv::compare, NOT cv::threshold.
    //
    // cv::threshold requires dst to have the SAME type as src. Given a CV_32F
    // source and a CV_8U dst wrapping the caller's buffer, it does not convert
    // and it does not error — it REALLOCATES dst as CV_32F and writes there, so
    // the caller's bitmap is silently left as it was. Detection then thresholds
    // to an all-zero bitmap, finds no connected components, and returns zero
    // boxes at full inference cost. cv::compare produces CV_8U (255/0) natively,
    // so the external buffer keeps its type and is written in place.
    //
    // `in > thresh -> 255` is exactly THRESH_BINARY's semantics, so the intended
    // behaviour is unchanged.
    //
    // This was latent: the CPU backend reaches DB post-processing through the
    // wrapped main-tree OrtPaddleDet and never calls this op, so the bug only
    // surfaced when the Intel backend became the first real consumer.
    cv::compare(in, static_cast<double>(thresh), out, cv::CMP_GT);
  }
}

std::vector<turbo_ocr::Box>
HostKernels::db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap,
                            int w, int h, const backend::DbPostParams &params,
                            backend::DeviceQueue & /*queue*/) {
  // PARAMETER CONTRACT: refuse loudly rather than substitute (kernels.h).
  if (!backend::require_db_supported(params, caps().params, "HostKernels::db_postprocess"))
    return {};
  const cv::Mat pred_map(h, w, CV_32F, const_cast<float *>(d_pred_map));
  // extract_boxes_from_bitmap takes a mutable bitmap; copy into scratch so the
  // caller's buffer is never mutated.
  const cv::Mat bmp_in(h, w, CV_8U, const_cast<std::uint8_t *>(d_bitmap));
  bmp_in.copyTo(bitmap_scratch_);

  // Boxes come back in the map's (resized) coordinate space — orig == resize
  // here; the caller rescales to original dims (matching the seam contract).
  std::vector<turbo_ocr::Box> boxes = detection::extract_boxes_from_bitmap(
      pred_map, bitmap_scratch_, /*orig_h=*/h, /*orig_w=*/w, /*resize_h=*/h,
      /*resize_w=*/w, params.box_thresh, params.unclip_ratio,
      params.min_box_side, params.min_unclipped_side, shifted_buf_,
      mask_buf_, contours_buf_, hierarchy_buf_);
  if (!params.oriented) {
    // Portable AABB mode: reduce each minAreaRect quad to its bounding rect so
    // this backend answers the same question the CUDA CCL path answers.
    for (auto &b : boxes) {
      const auto r = aabb(b);
      b.pts = {{{r[0], r[1]}, {r[2], r[1]}, {r[2], r[3]}, {r[0], r[3]}}};
    }
  }
  return boxes;
}

void HostKernels::argmax(const float *input_probs, int *output_indices,
                         float *output_scores, int batch_size, int seq_len,
                         int num_classes, backend::DeviceQueue & /*queue*/) {
  const std::size_t total = static_cast<std::size_t>(batch_size) * seq_len;
  for (std::size_t t = 0; t < total; ++t) {
    const float *row = input_probs + t * num_classes;
    int best = 0;
    float best_v = row[0];
    for (int k = 1; k < num_classes; ++k) {
      if (row[k] > best_v) {
        best_v = row[k];
        best = k;
      }
    }
    output_indices[t] = best;
    output_scores[t] = best_v;
  }
}

void HostKernels::preprocess_region(const backend::ImageView &src,
                                    const backend::Rect &rect,
                                    backend::PreprocKind kind, float *dst_chw,
                                    backend::DeviceQueue & /*queue*/) {
  // NOTE: the fused region preprocessors are provided for interface
  // completeness; the thin CPU stages delegate to the wrapped Cpu* classes,
  // which own their preprocessing, so this path is not on Deliverable-1's
  // critical route. The model-input dimensions are not carried by the seam
  // signature, so each kind uses its documented fixed target size.
  cv::Mat page = to_mat(src);
  if (page.empty())
    return;
  cv::Rect roi(std::max(0, rect.x), std::max(0, rect.y),
               std::min(rect.w, page.cols - std::max(0, rect.x)),
               std::min(rect.h, page.rows - std::max(0, rect.y)));
  if (roi.width <= 0 || roi.height <= 0)
    return;
  cv::Mat region = page(roi);

  switch (kind) {
  case backend::PreprocKind::LayoutSubRect: {
    // pixel/255, BGR CHW at the layout model input (800x800), stretch resize.
    constexpr int kSize = 800;
    cv::Mat canvas;
    cv::resize(region, canvas, cv::Size(kSize, kSize));
    write_chw_normalized(canvas, dst_chw, kSize, kSize, kSize,
                         backend::norm::layout_norm());
    break;
  }
  case backend::PreprocKind::TableCls: {
    // resize-short(256) -> center-crop(224) -> ImageNet -> BGR CHW.
    constexpr int kShort = 256, kCrop = 224;
    const double scale =
        static_cast<double>(kShort) / std::min(region.cols, region.rows);
    cv::Mat resized;
    cv::resize(region, resized, cv::Size(), scale, scale);
    const int x0 = std::max(0, (resized.cols - kCrop) / 2);
    const int y0 = std::max(0, (resized.rows - kCrop) / 2);
    cv::Rect crop(x0, y0, std::min(kCrop, resized.cols - x0),
                  std::min(kCrop, resized.rows - y0));
    cv::Mat canvas;
    cv::resize(resized(crop), canvas, cv::Size(kCrop, kCrop));
    write_chw_normalized(canvas, dst_chw, kCrop, kCrop, kCrop, imagenet_bgr());
    break;
  }
  case backend::PreprocKind::SlanextBGR:
  case backend::PreprocKind::SlanextRGB: {
    // ResizeByLong(488) preserve-AR -> ImageNet -> pad. BGR or RGB per kind.
    constexpr int kLong = 488;
    const double scale =
        static_cast<double>(kLong) / std::max(region.cols, region.rows);
    const int rw = std::max(1, static_cast<int>(region.cols * scale));
    const int rh = std::max(1, static_cast<int>(region.rows * scale));
    cv::Mat resized;
    cv::resize(region, resized, cv::Size(rw, rh));
    cv::Mat canvas;
    cv::copyMakeBorder(resized, canvas, 0, kLong - rh, 0, kLong - rw,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    const backend::NormParams p = (kind == backend::PreprocKind::SlanextRGB)
                                      ? backend::norm::imagenet_rgb()
                                      : backend::norm::imagenet_bgr();
    write_chw_normalized(canvas, dst_chw, kLong, kLong, kLong, p);
    break;
  }
  }
}

} // namespace turbo_ocr::cpu
