#include "amd/kernels_hip/hip_kernels.h"

#include "amd/support/hip_check.h"
#include "amd/kernels_hip/kernels_hip.h" // amd::kernels::* (hipified op set)
#include "amd/memory/hip_allocator.h"
#include "amd/queue/hip_queue.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <hip/hip_runtime.h>
#include <opencv2/imgcodecs.hpp>

namespace turbo_ocr::amd {

// The device kernels bake their component budget as k::kMaxGpuComponents
// while the seam's refusal logic compares against the shared
// detection::kMaxDbComponents. A comment used to assert they match; this pins
// it (full qualification: the k alias is declared further down).
static_assert(::turbo_ocr::amd::kernels::kMaxGpuComponents ==
                  turbo_ocr::detection::kMaxDbComponents,
              "device component budget diverged from the shared DB budget");

namespace k = ::turbo_ocr::amd::kernels;

namespace {
k::HipImage to_hip_image(const ImageView &v) {
  return k::HipImage{v.data, v.step, v.rows, v.cols};
}
} // namespace

struct HipKernels::Impl {
  std::shared_ptr<HipAllocator> alloc;

  // Pooled decode buffer (device-resident BGR image), reused across
  // decode_image calls; valid until the next decode_image (per the IKernels
  // contract). Grows monotonically.
  void *decode_buf = nullptr;
  std::size_t decode_cap = 0;

  // Cached DB post-process scratch, sized to the largest w*h seen. Pre-allocated
  // once (no per-call alloc on the hot path), matching the kernels.h contract.
  std::size_t px_cap = 0; // capacity in pixels
  int *labels = nullptr;
  int *compact_ids = nullptr;
  int *id_counter = nullptr;
  k::GpuDetBox *bboxes = nullptr;   // [kMaxGpuComponents * 2]
  k::GpuDetBox *bboxes2 = nullptr;  // [kMaxGpuComponents]  (JFA extract target)
  int *num_boxes = nullptr;
  std::uint32_t *expanded = nullptr; // [w*h]
  std::uint32_t *seeds = nullptr;    // [w*h]
  int *perim = nullptr;              // [kMaxGpuComponents]
  float *expand = nullptr;           // [kMaxGpuComponents]
  unsigned long long *moments = nullptr; // [kMaxGpuComponents*6]
  float *orient = nullptr;               // [kMaxGpuComponents*6]
  k::GpuDetBox *h_boxes = nullptr;   // pinned host output

  explicit Impl(std::shared_ptr<HipAllocator> a) : alloc(std::move(a)) {}
  ~Impl() {
    auto f = [&](void *p) { if (p) alloc->free(p); };
    f(decode_buf); f(labels); f(compact_ids); f(id_counter); f(bboxes);
    f(bboxes2); f(num_boxes); f(expanded); f(seeds); f(perim); f(expand);
    f(moments); f(orient);
    if (h_boxes) alloc->free_host(h_boxes);
  }

  void ensure_scratch(int w, int h) {
    const std::size_t px = static_cast<std::size_t>(w) * h;
    const int C = k::kMaxGpuComponents;
    if (px <= px_cap && labels)
      return;
    auto f = [&](void *p) { if (p) alloc->free(p); };
    f(labels); f(compact_ids); f(expanded); f(seeds);
    labels = static_cast<int *>(alloc->allocate(px * sizeof(int)));
    compact_ids = static_cast<int *>(alloc->allocate(px * sizeof(int)));
    expanded = static_cast<std::uint32_t *>(alloc->allocate(px * sizeof(std::uint32_t)));
    seeds = static_cast<std::uint32_t *>(alloc->allocate(px * sizeof(std::uint32_t)));
    if (!id_counter) {
      id_counter = static_cast<int *>(alloc->allocate(sizeof(int)));
      num_boxes = static_cast<int *>(alloc->allocate(sizeof(int)));
      bboxes = static_cast<k::GpuDetBox *>(alloc->allocate(2 * C * sizeof(k::GpuDetBox)));
      bboxes2 = static_cast<k::GpuDetBox *>(alloc->allocate(C * sizeof(k::GpuDetBox)));
      perim = static_cast<int *>(alloc->allocate(C * sizeof(int)));
      expand = static_cast<float *>(alloc->allocate(C * sizeof(float)));
      moments = static_cast<unsigned long long *>(alloc->allocate((std::size_t)C * 6 * sizeof(unsigned long long)));
      orient = static_cast<float *>(alloc->allocate((std::size_t)C * 6 * sizeof(float)));
      h_boxes = static_cast<k::GpuDetBox *>(alloc->allocate_host(C * sizeof(k::GpuDetBox)));
    }
    px_cap = px;
  }
};

HipKernels::HipKernels(std::shared_ptr<HipAllocator> alloc)
    : p_(std::make_unique<Impl>(std::move(alloc))) {}
HipKernels::~HipKernels() = default;

KernelCaps HipKernels::caps() const {
  KernelCaps c;
  c.device = backend::DeviceKind::Hip;
  c.decode_image = false;      // host fallback (OpenCV imdecode + H2D)
  c.resize_normalize = true;
  c.warp_crops = true;
  c.threshold = true;
  c.db_postprocess = true;     // native GPU CCL + JFA unclip
  c.argmax = true;
  c.preprocess_region = true;  // table_kernels.hip (all four PreprocKinds)
  // PARAMETER CONTRACT (kernels.h). NormParams::order is now plumbed through
  // both HIP preprocessors (`rgb_out`); it used to be silently dropped, which
  // left AMD det on RGB planes while every other backend feeds it BGR —
  // R and B swapped on AMD only.
  c.params.norm_mean_std = true;
  c.params.norm_channel_order = true;
  c.params.norm_letterbox = false;   // see the letterbox note in kernels.h
  c.params.db_oriented = true;       // hip_jfa_extract_oriented
  c.params.db_axis_aligned = true;   // hip_jfa_extract_bboxes
  c.params.db_expand_limits = true;
  c.params.db_side_limits = true;    // extract loop gates on params.min_unclipped_side
  c.params.db_max_components = true;
  return c;
}

ImageView HipKernels::decode_image(const std::uint8_t *data, std::size_t len,
                                   DeviceQueue &queue) {
  // TODO(on-hardware): replace with rocJPEG / VAAPI device decode. For now,
  // host-decode with OpenCV then H2D into the pooled device buffer. Still yields
  // a device-resident ImageView the rest of the pipeline consumes.
  cv::Mat enc(1, static_cast<int>(len), CV_8UC1,
              const_cast<std::uint8_t *>(data));
  cv::Mat bgr = cv::imdecode(enc, cv::IMREAD_COLOR); // 8UC3 BGR
  if (bgr.empty())
    return ImageView{};

  const int rows = bgr.rows, cols = bgr.cols;
  const std::size_t step = static_cast<std::size_t>(cols) * 3; // tight BGR pitch
  const std::size_t bytes = step * rows;
  if (bytes > p_->decode_cap) {
    if (p_->decode_buf)
      p_->alloc->free(p_->decode_buf);
    p_->decode_buf = p_->alloc->allocate(bytes);
    p_->decode_cap = bytes;
  }
  hipStream_t s = hip_stream_of(queue);
  // hipMemcpy2D handles a non-continuous cv::Mat (row stride != cols*3).
  HIP_CHECK(hipMemcpy2DAsync(p_->decode_buf, step, bgr.data, bgr.step, step,
                             rows, hipMemcpyHostToDevice, s));
  HIP_CHECK(hipStreamSynchronize(s)); // buffer reused next call; make it safe
  return ImageView{p_->decode_buf, step, rows, cols, backend::DeviceKind::Hip};
}

void HipKernels::resize_normalize(const ImageView &src, float *dst_chw,
                                  int dst_w, int dst_h, const NormParams &params,
                                  DeviceQueue &queue) {
  // PARAMETER CONTRACT (kernels.h): honour or refuse loudly, never substitute.
  if (!backend::require_norm_supported(params, caps().params,
                                       "HipKernels::resize_normalize"))
    return;
  hipStream_t s = hip_stream_of(queue);
  k::HipImage img = to_hip_image(src);
  // Param-driven: pass the caller's NormParams straight through. The CUDA
  // adapter has to SNIFF params.mean to pick between two baked variants (see
  // src/backends/nvidia/kernels_cuda/cuda_kernels.cpp), which silently misroutes any third
  // normalization; the HIP kernel takes mean/std/inv_scale as arguments, so
  // there is nothing to guess. With the det constants this is bit-identical to
  // hip_fused_resize_normalize_det (same kernel, same operand order).
  //
  // NOTE (parity, deliberate): params.letterbox is NOT honoured here. Neither
  // is it on the CUDA path — cuda_fused_resize_normalize_layout STRETCHES to
  // the canvas, and the layout stage's coordinate rescale is derived from that
  // stretch. Honouring letterbox here alone would silently desynchronize the
  // AMD boxes from every other backend's. If letterboxed full-frame resize is
  // ever wanted it is a SHARED policy change (kernels.h contract + all
  // backends + the layout coordinate math), not an AMD-local one.
  k::hip_fused_resize_normalize(
      img, dst_chw, dst_w, dst_h, params.mean[0], params.mean[1], params.mean[2],
      params.inv_std[0], params.inv_std[1], params.inv_std[2], params.inv_scale,
      params.order == backend::ChannelOrder::RGB ? 1 : 0, s);
}

void HipKernels::warp_crops(const ImageView &src, const float *d_M_invs,
                            const int *d_crop_widths, float *d_dst_batch,
                            int batch_size, int dst_h, int dst_w,
                            const NormParams &params, DeviceQueue &queue) {
  if (!backend::require_norm_supported(params, caps().params,
                                       "HipKernels::warp_crops"))
    return;
  hipStream_t s = hip_stream_of(queue);
  k::HipImage img = to_hip_image(src);
  // mean/std passed through in RGB order (the warp kernel emits RGB), matching
  // the CUDA cuda_batch_roi_warp signature exactly.
  k::hip_batch_roi_warp(img, d_M_invs, d_crop_widths, d_dst_batch, batch_size,
                        dst_h, dst_w, s, params.mean[0], params.mean[1],
                        params.mean[2], params.inv_std[0], params.inv_std[1],
                        params.inv_std[2], params.inv_scale,
                        params.order == backend::ChannelOrder::RGB ? 1 : 0);
}

void HipKernels::threshold(const float *src, std::uint8_t *dst, int w, int h,
                           int batch_size, float thresh, DeviceQueue &queue) {
  hipStream_t s = hip_stream_of(queue);
  if (batch_size <= 1)
    k::hip_threshold_to_u8(src, dst, w, h, thresh, s);
  else
    k::hip_batch_threshold_to_u8(src, dst, w, h, batch_size, thresh, s);
}

std::vector<turbo_ocr::Box>
HipKernels::db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap,
                           int w, int h, const DbPostParams &params,
                           DeviceQueue &queue) {
  if (!backend::require_db_supported(params, caps().params,
                                     "HipKernels::db_postprocess"))
    return {};
  p_->ensure_scratch(w, h);
  hipStream_t s = hip_stream_of(queue);

  // Step 1: CCL — label components, extract pre-filter bboxes into p_->bboxes
  // [0..N), populate p_->compact_ids, and return the PRE-filter component total.
  int h_num_boxes = 0, h_num_total = 0;
  k::hip_gpu_ccl_detect(d_bitmap, d_pred_map, w, h, params.box_thresh,
                        p_->labels, p_->compact_ids, p_->id_counter, p_->bboxes,
                        p_->num_boxes, p_->h_boxes, &h_num_boxes, s,
                        &h_num_total);
  // h_num_total is the RAW atomicAdd counter, which keeps incrementing past the
  // component budget (the CCL kernel only clamps the ids it STORES, writing -1
  // beyond max_components). Every scratch array below — bboxes2, perim, expand,
  // moments, orient, h_boxes — is sized kMaxGpuComponents, and the D2H at the
  // end copies N entries. Using the raw counter as N is therefore an
  // out-of-bounds read/write on any dense page that exceeds the budget. Clamp
  // it, and honour the caller's (smaller) max_components budget too.
  int budget = k::kMaxGpuComponents;
  if (params.max_components > 0 && params.max_components < budget)
    budget = params.max_components;
  const int N = std::min(h_num_total, budget); // pre-filter compact-id count
  if (N <= 0)
    return {};

  // Step 2: per-component Euclidean unclip (device-resident, no pred_map D2H).
  k::hip_accumulate_crack_perimeter(p_->compact_ids, d_bitmap, w, h, N,
                                    p_->perim, s);
  k::hip_compute_expand_per_comp(p_->bboxes, p_->perim, N, params.unclip_ratio,
                                 params.min_expand,
                                 params.max_expand > 0.0f ? params.max_expand
                                                          : 1e30f,
                                 params.box_thresh, p_->expand, s);
  k::hip_jfa_expand_labels(d_bitmap, p_->compact_ids, p_->expand, p_->expanded,
                           w, h, params.max_expand, p_->seeds, nullptr, s);

  // Step 3: extract the unclipped boxes (oriented quads if requested).
  if (params.oriented)
    k::hip_jfa_extract_oriented(p_->expanded, w, h, p_->bboxes2, N, p_->moments,
                                p_->orient, s);
  else
    k::hip_jfa_extract_bboxes(p_->expanded, w, h, p_->bboxes2, N, s);

  // Single D2H of the small per-component box array + sync.
  HIP_CHECK(hipMemcpyAsync(p_->h_boxes, p_->bboxes2,
                           static_cast<std::size_t>(N) * sizeof(k::GpuDetBox),
                           hipMemcpyDeviceToHost, s));
  HIP_CHECK(hipStreamSynchronize(s));

  // Build host quads in RESIZED coords (caller rescales to original dims).
  // Side gate = params.min_unclipped_side — the CALLER'S values, which is what
  // flips caps().params.db_side_limits to true. These boxes are post-expand
  // (the JFA unclip ran on device), so the post-unclip limit is the one that
  // applies. This loop used to hardcode "< 3" under a comment claiming "same
  // min-side gates as the CUDA det path" — at the time there were FOUR
  // different gate sets across the arms and this was the loosest, emitting
  // 3px slivers the shared reference (5.0 post-unclip) drops, each of which
  // became a rec call producing garbage text.
  std::vector<turbo_ocr::Box> out;
  out.reserve(N);
  for (int i = 0; i < N; ++i) {
    const k::GpuDetBox &b = p_->h_boxes[i];
    if (b.pixel_count < 1)
      continue; // empty / filtered slot
    const int bw = b.xmax - b.xmin + 1;
    const int bh = b.ymax - b.ymin + 1;
    if (static_cast<float>(bw) < params.min_unclipped_side ||
        static_cast<float>(bh) < params.min_unclipped_side)
      continue;
    turbo_ocr::Box box{};
    if (params.oriented) {
      for (int k4 = 0; k4 < 4; ++k4) {
        box[k4][0] = static_cast<int>(std::lround(b.ox[k4]));
        box[k4][1] = static_cast<int>(std::lround(b.oy[k4]));
      }
    } else {
      // tl, tr, br, bl
      box[0] = {b.xmin, b.ymin};
      box[1] = {b.xmax, b.ymin};
      box[2] = {b.xmax, b.ymax};
      box[3] = {b.xmin, b.ymax};
    }
    out.push_back(box);
  }
  return out;
}

void HipKernels::argmax(const float *input_probs, int *output_indices,
                        float *output_scores, int batch_size, int seq_len,
                        int num_classes, DeviceQueue &queue) {
  k::hip_argmax(input_probs, output_indices, output_scores, batch_size, seq_len,
                num_classes, hip_stream_of(queue));
}

void HipKernels::preprocess_region(const ImageView &src, const Rect &rect,
                                   PreprocKind kind, float *dst_chw,
                                   DeviceQueue &queue) {
  hipStream_t s = hip_stream_of(queue);
  const k::HipImage img = to_hip_image(src);
  switch (kind) {
  case PreprocKind::LayoutSubRect:
    // Cell-detection canvas. 800x800 matches the CUDA adapter's hard-coded dst
    // dims; both read it from the same place (the cell-det model's input size),
    // so if that model ever changes size it must change in BOTH — the seam
    // carries no dst dims for this op.
    k::hip_fused_resize_normalize_layout_subrect(img, rect.x, rect.y, rect.w,
                                                 rect.h, dst_chw, /*dst_w=*/800,
                                                 /*dst_h=*/800, s);
    break;
  case PreprocKind::TableCls:
    k::hip_fused_table_cls_pre(img, rect.x, rect.y, rect.w, rect.h, dst_chw, s);
    break;
  case PreprocKind::SlanextBGR:
    k::hip_fused_slanext_pre(img, rect.x, rect.y, rect.w, rect.h, dst_chw, s);
    break;
  case PreprocKind::SlanextRGB:
    k::hip_fused_slanext_pre_rgb(img, rect.x, rect.y, rect.w, rect.h, dst_chw,
                                 s);
    break;
  }
}

} // namespace turbo_ocr::amd
