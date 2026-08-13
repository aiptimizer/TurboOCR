// SyclKernels — SYCL device pre/post kernels.
//
// TOOLCHAIN: `icpx -fsycl` (oneAPI/DPC++). The SYCL bodies are guarded by
// TURBO_OCR_HAS_SYCL and are NOT compilable on the dev Mac. The guarded-IN path
// is authoritative.
//
// SEMANTIC CONTRACT: every native kernel here must produce the same numbers as
// src/backends/cpu/kernels_host/host_kernels.cpp for the same inputs, because the CPU backend is
// the golden reference the Intel path is diffed against on hardware. The three
// conventions that actually bite are called out inline:
//   1. bilinear sample point  = (d + 0.5) * scale - 0.5   (OpenCV INTER_LINEAR)
//   2. letterbox content size = trunc(src * scale), NOT round  (cv::resize is
//      handed the truncated size in host_kernels; rounding drifts by one row or
//      column on about half of all inputs, which shows up in a golden diff)
//   3. plane order follows NormParams::order: RGB => plane0 = R. The CUDA warp
//      kernel bakes RGB in; host_kernels parameterises it; we follow the seam.
//
// The two host-fallback ops (db_postprocess, decode_image) need no SYCL and are
// always compiled, so they are exercisable off-hardware.

#include "intel/kernels_sycl/sycl_kernels.h"
#include "intel/memory/l0_allocator.h"
#include "intel/queue/l0_device_queue.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/core/norm_params.h"         // SHARED norm factories
#include "turbo_ocr/analysis/detection/det_postprocess.h" // SHARED extract_boxes_from_bitmap

#if defined(TURBO_OCR_HAS_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace turbo_ocr::intel {

struct SyclKernels::Impl {
  std::shared_ptr<L0Allocator> alloc;

  // Kernels-owned decode pool: one USM device image reused across
  // decode_image() calls (valid until the next decode, per the IKernels
  // contract). Grown only when a larger image arrives.
  backend::DeviceBuffer decode_dev;
  std::size_t decode_cap = 0;
  cv::Mat decode_host; // keeps the decoded pixels alive for the H2D

  // Host mirrors for the db_postprocess fallback, pre-sized by
  // reserve_host_fallback() so the hot path never allocates.
  std::vector<std::uint8_t> h_bitmap;
  std::vector<float> h_pred;
  cv::Mat bitmap_scratch;
  // extract_boxes_from_bitmap's caller-owned scratch (thread-safety by design).
  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  std::vector<std::vector<cv::Point>> contours_buf;
  std::vector<cv::Vec4i> hierarchy_buf;

  explicit Impl(std::shared_ptr<L0Allocator> a) : alloc(std::move(a)) {}
};

SyclKernels::SyclKernels(std::shared_ptr<L0Allocator> alloc)
    : impl_(std::make_unique<Impl>(std::move(alloc))) {}
SyclKernels::~SyclKernels() = default;

backend::KernelCaps SyclKernels::caps() const {
  backend::KernelCaps c;
  c.device = backend::DeviceKind::L0;
  c.decode_image = false;     // host OpenCV decode + H2D (no VAAPI/oneVPL yet)
  c.resize_normalize = true;  // native SYCL
  c.warp_crops = true;        // native SYCL
  c.threshold = true;         // native SYCL
  c.db_postprocess = false;   // DECLARED host fallback -> shared det post
  c.argmax = true;            // native SYCL
  c.preprocess_region = true; // native SYCL
#if !defined(TURBO_OCR_HAS_SYCL)
  // Built without DPC++: nothing is native. Report that truthfully rather than
  // claiming device execution this binary cannot perform.
  c.device = backend::DeviceKind::Host;
  c.resize_normalize = c.warp_crops = c.threshold = c.argmax =
      c.preprocess_region = false;
#endif
  // PARAMETER CONTRACT (kernels.h): declare exactly what this backend honours.
  c.params.norm_mean_std = true;
  c.params.norm_channel_order = true;   // sycl_kernels.cpp honours params.order
  // SYCL genuinely implements letterbox (resize + top-left pad). The shared
  // callers still pass false so all backends stay in step — see kernels.h.
  c.params.norm_letterbox = true;
  c.params.db_oriented = true;          // extract_boxes_from_bitmap = minAreaRect
  c.params.db_axis_aligned = false;     // ...and it CANNOT emit AABBs
  c.params.db_expand_limits = false;    // contour path has no expand clamp
  c.params.db_side_limits = true;
  c.params.db_max_components = false;   // contour path has no component budget
  return c;
}

void SyclKernels::reserve_host_fallback(std::size_t max_map_pixels) {
  auto &I = *impl_;
  I.h_bitmap.resize(max_map_pixels);
  I.h_pred.resize(max_map_pixels);
}

#if defined(TURBO_OCR_HAS_SYCL)

namespace {

[[nodiscard]] sycl::queue &q_of(backend::DeviceQueue &queue) {
  return *static_cast<sycl::queue *>(queue.native_handle());
}

// Tell the lane its new tail so DeviceQueue::record() stays accurate.
void note(backend::DeviceQueue &queue, sycl::event &e) {
  if (auto *lq = dynamic_cast<L0DeviceQueue *>(&queue))
    lq->note_submission(&e);
}

struct Bgr {
  float b, g, r;
};

// OpenCV INTER_LINEAR sample of a BGR8 image — mirror of bilinear_sample_bgr in
// src/backends/nvidia/kernels_cuda/preprocess_kernels.cu, and of what cv::resize computes in
// host_kernels.cpp.
inline Bgr sample_bgr(const std::uint8_t *src, int src_h, int src_w,
                      int src_step, float scale_x, float scale_y, int dx,
                      int dy) {
  const float sx = (dx + 0.5f) * scale_x - 0.5f;
  const float sy = (dy + 0.5f) * scale_y - 0.5f;
  int x0 = static_cast<int>(sycl::floor(sx));
  int y0 = static_cast<int>(sycl::floor(sy));
  const float fx = sx - x0, fy = sy - y0;
  int x1 = x0 + 1, y1 = y0 + 1;
  x0 = sycl::max(0, sycl::min(x0, src_w - 1));
  x1 = sycl::max(0, sycl::min(x1, src_w - 1));
  y0 = sycl::max(0, sycl::min(y0, src_h - 1));
  y1 = sycl::max(0, sycl::min(y1, src_h - 1));
  const std::uint8_t *r0 = src + static_cast<std::size_t>(y0) * src_step;
  const std::uint8_t *r1 = src + static_cast<std::size_t>(y1) * src_step;
  auto px = [](const std::uint8_t *row, int x, int c) {
    return static_cast<float>(row[x * 3 + c]);
  };
  const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
  const float w01 = (1 - fx) * fy, w11 = fx * fy;
  Bgr o;
  o.b = w00 * px(r0, x0, 0) + w10 * px(r0, x1, 0) + w01 * px(r1, x0, 0) + w11 * px(r1, x1, 0);
  o.g = w00 * px(r0, x0, 1) + w10 * px(r0, x1, 1) + w01 * px(r1, x0, 1) + w11 * px(r1, x1, 1);
  o.r = w00 * px(r0, x0, 2) + w10 * px(r0, x1, 2) + w01 * px(r1, x0, 2) + w11 * px(r1, x1, 2);
  return o;
}

} // namespace

void SyclKernels::resize_normalize(const backend::ImageView &src, float *dst_chw,
                                   int dst_w, int dst_h,
                                   const backend::NormParams &p,
                                   backend::DeviceQueue &queue) {
  // PARAMETER CONTRACT (kernels.h): honour or refuse loudly, never substitute.
  if (!backend::require_norm_supported(p, caps().params,
                                       "SyclKernels::resize_normalize"))
    return;
  if (src.empty() || dst_w <= 0 || dst_h <= 0)
    return;
  auto &q = q_of(queue);
  const auto *src_data = static_cast<const std::uint8_t *>(src.data);
  const int src_h = src.rows, src_w = src.cols, src_step = static_cast<int>(src.step);
  const bool rgb = (p.order == backend::ChannelOrder::RGB);
  const float m0 = p.mean[0], m1 = p.mean[1], m2 = p.mean[2];
  const float s0 = p.inv_std[0], s1 = p.inv_std[1], s2 = p.inv_std[2];
  const float inv = p.inv_scale;
  const bool letterbox = p.letterbox;

  // Letterbox: preserve aspect, content top-left, pad the remainder with pixel
  // value 0 (which normalizes to -mean*inv_std) — identical to host_kernels'
  // cv::resize + copyMakeBorder(0,0,0). NOTE the truncating cast: cv::resize is
  // handed trunc(src*scale) there, so rounding here would drift by a pixel.
  int content_w = dst_w, content_h = dst_h;
  if (letterbox && src_w > 0 && src_h > 0) {
    const double scale = std::min(static_cast<double>(dst_w) / src_w,
                                  static_cast<double>(dst_h) / src_h);
    content_w = std::max(1, static_cast<int>(src_w * scale));
    content_h = std::max(1, static_cast<int>(src_h * scale));
  }
  const float scale_x = static_cast<float>(src_w) / content_w;
  const float scale_y = static_cast<float>(src_h) / content_h;
  const int plane = dst_h * dst_w;

  sycl::event e = q.parallel_for(
      sycl::range<2>(static_cast<std::size_t>(dst_h), static_cast<std::size_t>(dst_w)),
      [=](sycl::id<2> it) {
        const int dy = static_cast<int>(it[0]), dx = static_cast<int>(it[1]);
        const int idx = dy * dst_w + dx;
        float c0, c1, c2;
        if (letterbox && (dx >= content_w || dy >= content_h)) {
          c0 = (0.0f - m0) * s0;
          c1 = (0.0f - m1) * s1;
          c2 = (0.0f - m2) * s2;
        } else {
          const Bgr v = sample_bgr(src_data, src_h, src_w, src_step, scale_x,
                                   scale_y, dx, dy);
          const float ch0 = rgb ? v.r : v.b;
          const float ch1 = v.g;
          const float ch2 = rgb ? v.b : v.r;
          c0 = (ch0 * inv - m0) * s0;
          c1 = (ch1 * inv - m1) * s1;
          c2 = (ch2 * inv - m2) * s2;
        }
        dst_chw[0 * plane + idx] = c0;
        dst_chw[1 * plane + idx] = c1;
        dst_chw[2 * plane + idx] = c2;
      });
  note(queue, e);
}

void SyclKernels::warp_crops(const backend::ImageView &src, const float *d_M_invs,
                             const int *d_crop_widths, float *d_dst_batch,
                             int batch_size, int dst_h, int dst_w,
                             const backend::NormParams &p,
                             backend::DeviceQueue &queue) {
  if (!backend::require_norm_supported(p, caps().params,
                                       "SyclKernels::warp_crops"))
    return;
  if (batch_size <= 0 || src.empty())
    return;
  auto &q = q_of(queue);
  const auto *src_data = static_cast<const std::uint8_t *>(src.data);
  const int src_h = src.rows, src_w = src.cols, src_step = static_cast<int>(src.step);
  const bool rgb = (p.order == backend::ChannelOrder::RGB);
  const float m0 = p.mean[0], m1 = p.mean[1], m2 = p.mean[2];
  const float s0 = p.inv_std[0], s1 = p.inv_std[1], s2 = p.inv_std[2];
  const float inv = p.inv_scale;
  const int plane = dst_h * dst_w;

  // One work-item per (crop, y, x) — mirror of batch_roi_warp_kernel.
  sycl::event e = q.parallel_for(
      sycl::range<3>(static_cast<std::size_t>(batch_size),
                     static_cast<std::size_t>(dst_h),
                     static_cast<std::size_t>(dst_w)),
      [=](sycl::id<3> it) {
        const int b = static_cast<int>(it[0]);
        const int y = static_cast<int>(it[1]);
        const int x = static_cast<int>(it[2]);
        const float *M = d_M_invs + static_cast<std::size_t>(b) * 9;
        const int crop_w = d_crop_widths[b];
        const int base = b * 3 * plane + y * dst_w + x;

        auto write_zero = [&]() {
          d_dst_batch[base + 0 * plane] = 0.0f;
          d_dst_batch[base + 1 * plane] = 0.0f;
          d_dst_batch[base + 2 * plane] = 0.0f;
        };
        // Columns beyond the crop's real width are zero-padded (CUDA and host
        // both write literal 0.0f there, NOT a normalized black).
        if (x >= crop_w) {
          write_zero();
          return;
        }
        // Sign-preserving denominator clamp: a bare +1e-7 would flip a small
        // negative denom positive and mirror the sampled pixel.
        float denom = M[6] * x + M[7] * y + M[8];
        denom = sycl::copysign(sycl::fmax(sycl::fabs(denom), 1e-7f), denom);
        const float inv_denom = 1.0f / denom;
        float sx = (M[0] * x + M[1] * y + M[2]) * inv_denom;
        float sy = (M[3] * x + M[4] * y + M[5]) * inv_denom;
        if (!sycl::isfinite(sx) || !sycl::isfinite(sy)) {
          write_zero();
          return;
        }
        sx = sycl::fmin(sycl::fmax(sx, -1.0f), static_cast<float>(src_w + 1));
        sy = sycl::fmin(sycl::fmax(sy, -1.0f), static_cast<float>(src_h + 1));
        int xl = static_cast<int>(sycl::floor(sx));
        int yl = static_cast<int>(sycl::floor(sy));
        int xh = xl + 1, yh = yl + 1;
        const float dxf = sx - xl, dyf = sy - yl;
        // BORDER_REPLICATE (cv::warpPerspective in host_kernels does the same).
        xl = sycl::min(sycl::max(xl, 0), src_w - 1);
        xh = sycl::min(sycl::max(xh, 0), src_w - 1);
        yl = sycl::min(sycl::max(yl, 0), src_h - 1);
        yh = sycl::min(sycl::max(yh, 0), src_h - 1);

        auto ch = [&](int px, int py, int c) { // c: 0=R 1=G 2=B, source is BGR
          const std::uint8_t *row = src_data + static_cast<std::size_t>(py) * src_step;
          const std::uint8_t *pp = row + px * 3;
          return static_cast<float>(c == 0 ? pp[2] : (c == 1 ? pp[1] : pp[0]));
        };
        const float w00 = (1 - dxf) * (1 - dyf), w10 = dxf * (1 - dyf);
        const float w01 = (1 - dxf) * dyf, w11 = dxf * dyf;
        auto interp = [&](int c) {
          return w00 * ch(xl, yl, c) + w10 * ch(xh, yl, c) + w01 * ch(xl, yh, c) +
                 w11 * ch(xh, yh, c);
        };
        const float R = interp(0), G = interp(1), B = interp(2);
        const float c0 = rgb ? R : B, c1 = G, c2 = rgb ? B : R;
        d_dst_batch[base + 0 * plane] = (c0 * inv - m0) * s0;
        d_dst_batch[base + 1 * plane] = (c1 * inv - m1) * s1;
        d_dst_batch[base + 2 * plane] = (c2 * inv - m2) * s2;
      });
  note(queue, e);
}

void SyclKernels::threshold(const float *src, std::uint8_t *dst, int w, int h,
                            int batch_size, float thresh,
                            backend::DeviceQueue &queue) {
  const std::size_t total = static_cast<std::size_t>(w) * h *
                            static_cast<std::size_t>(std::max(1, batch_size));
  if (total == 0)
    return;
  auto &q = q_of(queue);
  sycl::event e = q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> i) {
    dst[i] = (src[i] > thresh) ? std::uint8_t(255) : std::uint8_t(0);
  });
  note(queue, e);
}

void SyclKernels::argmax(const float *input_probs, int *output_indices,
                         float *output_scores, int batch_size, int seq_len,
                         int num_classes, backend::DeviceQueue &queue) {
  const std::size_t steps =
      static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(seq_len);
  if (steps == 0 || num_classes <= 0)
    return;
  auto &q = q_of(queue);
  // One work-item per timestep, ascending scan with a STRICT `>` so exact ties
  // keep the LOWEST class index — the same tie-break the CUDA reduction and the
  // host/AVX2 CTC reference use. A different tie-break silently changes text.
  sycl::event e = q.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> s) {
    const float *p = input_probs + s[0] * static_cast<std::size_t>(num_classes);
    float best = p[0];
    int bi = 0;
    for (int c = 1; c < num_classes; ++c) {
      if (p[c] > best) {
        best = p[c];
        bi = c;
      }
    }
    output_indices[s[0]] = bi;
    output_scores[s[0]] = best;
  });
  note(queue, e);
}

void SyclKernels::preprocess_region(const backend::ImageView &src,
                                    const backend::Rect &rect,
                                    backend::PreprocKind kind, float *dst_chw,
                                    backend::DeviceQueue &queue) {
  // Geometry per kind is taken from host_kernels::preprocess_region so the two
  // agree: LayoutSubRect stretches to 800 with pixel/255 BGR; TableCls does
  // resize-short(256) + center-crop(224) + ImageNet BGR; Slanext* does
  // ResizeByLong(488) preserving AR + bottom-right zero pad + ImageNet.
  if (src.empty())
    return;
  auto &q = q_of(queue);
  const auto *base = static_cast<const std::uint8_t *>(src.data);
  const int src_step = static_cast<int>(src.step);
  const int rx = std::max(0, rect.x), ry = std::max(0, rect.y);
  const int rw = std::max(1, std::min(rect.w, src.cols - rx));
  const int rh = std::max(1, std::min(rect.h, src.rows - ry));
  const std::uint8_t *sub =
      base + static_cast<std::size_t>(ry) * src_step + static_cast<std::size_t>(rx) * 3;

  int out_w = 0, out_h = 0;
  float m0 = 0, m1 = 0, m2 = 0, s0 = 1, s1 = 1, s2 = 1;
  const float inv = 1.0f / 255.0f;
  bool rgb = false;
  // Sub-window of the region actually sampled (TableCls center-crop) and the
  // content extent inside the output (Slanext letterbox).
  int win_x = 0, win_y = 0, win_w = rw, win_h = rh;
  int content_w = 0, content_h = 0; // 0 => the whole output is content

  // Normalization constants come from the SHARED factories, never retyped.
  const backend::NormParams k_layout = backend::norm::layout_norm();
  const backend::NormParams k_imagenet = backend::norm::imagenet_bgr();
  switch (kind) {
  case backend::PreprocKind::LayoutSubRect:
    out_w = out_h = 800;
    m0 = k_layout.mean[0]; m1 = k_layout.mean[1]; m2 = k_layout.mean[2];
    s0 = k_layout.inv_std[0]; s1 = k_layout.inv_std[1]; s2 = k_layout.inv_std[2];
    break;
  case backend::PreprocKind::TableCls: {
    out_w = out_h = 224;
    m0 = k_imagenet.mean[0]; m1 = k_imagenet.mean[1]; m2 = k_imagenet.mean[2];
    s0 = k_imagenet.inv_std[0]; s1 = k_imagenet.inv_std[1]; s2 = k_imagenet.inv_std[2];
    // resize-short(256) then center-crop(224) == sampling the centered
    // (224/256 * short_side) square of the region, expressed in source pixels.
    const double k = 224.0 / 256.0;
    const int short_side = std::min(rw, rh);
    win_w = std::max(1, static_cast<int>(short_side * k));
    win_h = win_w;
    win_x = std::max(0, (rw - win_w) / 2);
    win_y = std::max(0, (rh - win_h) / 2);
    break;
  }
  case backend::PreprocKind::SlanextBGR:
  case backend::PreprocKind::SlanextRGB: {
    out_w = out_h = 488;
    m0 = k_imagenet.mean[0]; m1 = k_imagenet.mean[1]; m2 = k_imagenet.mean[2];
    s0 = k_imagenet.inv_std[0]; s1 = k_imagenet.inv_std[1]; s2 = k_imagenet.inv_std[2];
    rgb = (kind == backend::PreprocKind::SlanextRGB);
    const double scale = 488.0 / std::max(rw, rh);
    content_w = std::max(1, static_cast<int>(rw * scale));
    content_h = std::max(1, static_cast<int>(rh * scale));
    break;
  }
  }

  const int cw = content_w ? content_w : out_w;
  const int chh = content_h ? content_h : out_h;
  const float scale_x = static_cast<float>(win_w) / cw;
  const float scale_y = static_cast<float>(win_h) / chh;
  const std::uint8_t *win =
      sub + static_cast<std::size_t>(win_y) * src_step + static_cast<std::size_t>(win_x) * 3;
  const int plane = out_w * out_h;
  const bool pad = (content_w != 0);

  sycl::event e = q.parallel_for(
      sycl::range<2>(static_cast<std::size_t>(out_h), static_cast<std::size_t>(out_w)),
      [=](sycl::id<2> it) {
        const int dy = static_cast<int>(it[0]), dx = static_cast<int>(it[1]);
        const int idx = dy * out_w + dx;
        float c0, c1, c2;
        if (pad && (dx >= cw || dy >= chh)) {
          c0 = (0.0f - m0) * s0;
          c1 = (0.0f - m1) * s1;
          c2 = (0.0f - m2) * s2;
        } else {
          const Bgr v = sample_bgr(win, win_h, win_w, src_step, scale_x, scale_y, dx, dy);
          const float a = rgb ? v.r : v.b, b = v.g, c = rgb ? v.b : v.r;
          c0 = (a * inv - m0) * s0;
          c1 = (b * inv - m1) * s1;
          c2 = (c * inv - m2) * s2;
        }
        dst_chw[0 * plane + idx] = c0;
        dst_chw[1 * plane + idx] = c1;
        dst_chw[2 * plane + idx] = c2;
      });
  note(queue, e);
}

#else // !TURBO_OCR_HAS_SYCL — no device; the native ops are unavailable.
      // caps() already reports them false so the pipeline never expects device
      // execution from this build. They are no-ops rather than silently wrong
      // math: a build without DPC++ is a compile-conformance build only.

void SyclKernels::resize_normalize(const backend::ImageView &, float *, int, int,
                                   const backend::NormParams &,
                                   backend::DeviceQueue &) {}
void SyclKernels::warp_crops(const backend::ImageView &, const float *,
                             const int *, float *, int, int, int,
                             const backend::NormParams &, backend::DeviceQueue &) {}
void SyclKernels::threshold(const float *, std::uint8_t *, int, int, int, float,
                            backend::DeviceQueue &) {}
void SyclKernels::argmax(const float *, int *, float *, int, int, int,
                         backend::DeviceQueue &) {}
void SyclKernels::preprocess_region(const backend::ImageView &,
                                    const backend::Rect &, backend::PreprocKind,
                                    float *, backend::DeviceQueue &) {}

#endif

// ---- Declared host-fallback ops (always compiled; need no SYCL) -------------

backend::ImageView SyclKernels::decode_image(const std::uint8_t *data,
                                             std::size_t len,
                                             backend::DeviceQueue &queue) {
  // Host OpenCV decode, then H2D into a kernels-owned USM buffer that is valid
  // until the next decode_image() (the seam's stated invalidation point).
  // TODO(on-hw): VAAPI / oneVPL hardware JPEG decode straight into USM — the
  // nvJPEG analogue. Until then caps().decode_image is false.
  auto &I = *impl_;
  backend::ImageView out;
  if (!data || len == 0)
    return out;

  const cv::Mat enc(1, static_cast<int>(len), CV_8UC1,
                    const_cast<std::uint8_t *>(data));
  I.decode_host = cv::imdecode(enc, cv::IMREAD_COLOR);
  if (I.decode_host.empty())
    return out;

  const std::size_t step = static_cast<std::size_t>(I.decode_host.step);
  const std::size_t bytes = step * static_cast<std::size_t>(I.decode_host.rows);
  if (bytes > I.decode_cap) { // grow-only; steady state does not reallocate
    I.decode_dev = I.alloc->allocate_buffer(bytes);
    I.decode_cap = I.decode_dev ? bytes : 0;
  }
  if (!I.decode_dev)
    return out;

  I.alloc->copy_h2d(I.decode_dev.data(), I.decode_host.data, bytes, queue);
  queue.synchronize(); // the image must be resident before the caller uses it

  out.data = I.decode_dev.data();
  out.step = step;
  out.rows = I.decode_host.rows;
  out.cols = I.decode_host.cols;
  out.kind = I.alloc->has_device() ? backend::DeviceKind::L0
                                   : backend::DeviceKind::Host;
  return out;
}

std::vector<turbo_ocr::Box>
SyclKernels::db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap,
                            int w, int h, const backend::DbPostParams &params,
                            backend::DeviceQueue &queue) {
  // DECLARED HOST FALLBACK (caps().db_postprocess == false).
  //
  // DEDUP: this does NOT reimplement DB post-processing. It copies the two small
  // maps down and calls the SHARED detection::extract_boxes_from_bitmap — the
  // exact function the CPU backend and the NVIDIA contour path use, with the
  // same min_box_side / min_unclipped_side constants as
  // src/backends/cpu/kernels_host/host_kernels.cpp. Any future fix to score / unclip / corner
  // ordering lands here for free, and an Intel-vs-CPU golden diff on this stage
  // is exact by construction rather than "close".
  // PARAMETER CONTRACT (kernels.h): honour or refuse loudly, never substitute.
  if (!backend::require_db_supported(params, caps().params,
                                     "SyclKernels::db_postprocess"))
    return {};
  auto &I = *impl_;
  const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  if (n == 0 || !d_pred_map || !d_bitmap)
    return {};

  if (I.h_bitmap.size() < n)
    I.h_bitmap.resize(n); // pre-sized by reserve_host_fallback() at warmup
  if (I.h_pred.size() < n)
    I.h_pred.resize(n);

  I.alloc->copy_d2h(I.h_bitmap.data(), d_bitmap, n * sizeof(std::uint8_t), queue);
  I.alloc->copy_d2h(I.h_pred.data(), d_pred_map, n * sizeof(float), queue);
  queue.synchronize();

  const cv::Mat pred_map(h, w, CV_32F, I.h_pred.data());
  const cv::Mat bmp_in(h, w, CV_8U, I.h_bitmap.data());
  // extract_boxes_from_bitmap mutates its bitmap; give it scratch.
  bmp_in.copyTo(I.bitmap_scratch);

  // Boxes come back in the MAP's coordinate space (orig == resize here); the
  // caller rescales to original dims, exactly as the seam documents and as the
  // CPU backend does.
  return detection::extract_boxes_from_bitmap(
      pred_map, I.bitmap_scratch, /*orig_h=*/h, /*orig_w=*/w, /*resize_h=*/h,
      /*resize_w=*/w, params.box_thresh, params.unclip_ratio,
      params.min_box_side, params.min_unclipped_side, I.shifted_buf,
      I.mask_buf, I.contours_buf, I.hierarchy_buf);
}

} // namespace turbo_ocr::intel
