// CudaKernels implementation — forwards each IKernels op to the corresponding
// existing CUDA kernel in src/backends/nvidia/kernels_cuda/kernels_cuda.h. Output is
// bit-identical because it IS the same kernel; the adapter only translates the
// argument vocabulary (ImageView->GpuImage, DeviceQueue->stream, the caps enums
// -> the baked-in normalization variants).

#include "nvidia/kernels_cuda/cuda_kernels.h"

#include <algorithm>

#include <cuda_runtime.h>

#include "nvidia/support/cuda_common.h"
#include "nvidia/support/cuda_check.h"
#include "nvidia/support/cuda_ptr.h"
#include "nvidia/support/nvjpeg_decoder.h"
#include "turbo_ocr/core/norm_params.h" // SHARED norm factories
#include "nvidia/kernels_cuda/kernels_cuda.h"

namespace turbo_ocr::nvidia {

// The device kernels bake their component budget as kernels::kMaxGpuComponents
// while the seam's refusal logic compares against the shared
// detection::kMaxDbComponents. A comment used to assert they match; this pins
// it, so raising one without the other is a compile error instead of scratch
// buffers sized from one constant and indices clamped by the other.
static_assert(kernels::kMaxGpuComponents == turbo_ocr::detection::kMaxDbComponents,
              "device component budget diverged from the shared DB budget");

// ---- db_postprocess scratch ------------------------------------------------
// Mirrors PaddleDet's mode-2 (all-GPU JFA per-component Euclidean unclip)
// buffer set. Grown to fit w*h; RAII via CudaPtr/CudaHostPtr.
struct CudaKernels::DbScratch {
  int cap_pixels = 0;
  // Device buffers backing a persistent device-resident decode.
  turbo_ocr::CudaPtr<uint8_t> d_decode; // decode_image() pool
  int decode_cap = 0;

  // CCL + JFA scratch.
  turbo_ocr::CudaPtr<int> d_labels;
  turbo_ocr::CudaPtr<int> d_compact_ids;
  turbo_ocr::CudaPtr<int> d_id_counter;
  turbo_ocr::CudaPtr<kernels::GpuDetBox> d_bboxes;
  turbo_ocr::CudaPtr<int> d_num_boxes;
  turbo_ocr::CudaPtr<uint32_t> d_jfa_labels;
  turbo_ocr::CudaPtr<uint32_t> d_jfa_seeds;
  turbo_ocr::CudaPtr<uint32_t> d_jfa_seeds_alt;
  turbo_ocr::CudaPtr<float> d_expand;
  turbo_ocr::CudaPtr<int> d_perim;
  turbo_ocr::CudaHostPtr<kernels::GpuDetBox> h_boxes;
  int h_num_boxes = 0;
  int h_num_total = 0;
};

CudaKernels::CudaKernels() = default;
CudaKernels::~CudaKernels() = default;

backend::KernelCaps CudaKernels::caps() const {
  backend::KernelCaps c;
  c.device = backend::DeviceKind::Cuda;
  c.decode_image = true;
  c.resize_normalize = true;
  c.warp_crops = true;
  c.threshold = true;
  c.db_postprocess = true; // native CCL + JFA unclip on device
  c.argmax = true;
  c.preprocess_region = true;
  // PARAMETER CONTRACT (kernels.h). These four `false`s are the honest report
  // this adapter used to omit while answering "yes" to everything:
  //  * the full-frame preprocessor is TWO kernels with the det and layout
  //    constants baked into the .cu (main tree, not editable from here), so it
  //    serves exactly those two distributions. resize_normalize now MATCHES the
  //    caller's params against the shared factories and refuses anything else,
  //    instead of sniffing `mean[0] == 0` and silently misrouting a third.
  //  * cuda_batch_roi_warp always emits RGB planes: `order` cannot be honoured.
  //  * the JFA extract path emits AABBs only — a caller asking for rotated
  //    quads used to get axis-aligned boxes back with no indication, which
  //    changes the crop geometry and therefore the recognized text.
  //  * there is no per-call component budget; kMaxGpuComponents is compiled in.
  c.params.norm_mean_std = true;              // warp_crops takes real params
  c.params.norm_mean_std_full_frame = false;  // resize_normalize is baked
  c.params.norm_channel_order = false;        // warp emits RGB, always
  c.params.norm_letterbox = false;
  c.params.db_oriented = false;               // AABB only (see below)
  c.params.db_axis_aligned = true;
  c.params.db_expand_limits = true;
  c.params.db_side_limits = true;  // extract loop gates on params.min_unclipped_side
  c.params.db_max_components = false;
  return c;
}

bool CudaKernels::can_decode_image(const std::uint8_t *data,
                                   std::size_t len) const {
  return decode::NvJpegDecoder::is_jpeg(
      reinterpret_cast<const unsigned char *>(data), len);
}

backend::ImageView CudaKernels::decode_image(const std::uint8_t *data,
                                             std::size_t len,
                                             backend::DeviceQueue &queue) {
  if (!decoder_)
    decoder_ = std::make_unique<decode::NvJpegDecoder>();
  if (!db_)
    db_ = std::make_unique<DbScratch>();
  const cudaStream_t stream = cuda_stream(queue);

  auto [w, h] = decoder_->get_dimensions(data, len);
  if (w <= 0 || h <= 0)
    return {}; // not a JPEG / header parse failed -> caller falls back to host
  const int bytes = h * w * 3;
  if (bytes > db_->decode_cap) {
    db_->d_decode.reset(static_cast<size_t>(bytes));
    db_->decode_cap = bytes;
  }
  // Device-resident decode: BGRI straight into the pool buffer, async on the
  // queue. Caller syncs the queue before host-reading (it won't — it feeds the
  // ImageView straight into the detector, staying resident).
  if (!decoder_->decode_to_gpu(data, len, db_->d_decode.get(),
                               static_cast<size_t>(w) * 3, w, h, stream))
    return {};
  return backend::ImageView{.data = db_->d_decode.get(),
                            .step = static_cast<std::size_t>(w) * 3,
                            .rows = h,
                            .cols = w,
                            .kind = backend::DeviceKind::Cuda};
}

void CudaKernels::resize_normalize(const backend::ImageView &src, float *dst_chw,
                                   int dst_w, int dst_h,
                                   const backend::NormParams &params,
                                   backend::DeviceQueue &queue) {
  const auto g = to_gpu_image(src);
  const cudaStream_t stream = cuda_stream(queue);

  // PARAMETER CONTRACT (kernels.h): this backend's full-frame preprocessor is
  // two kernels with their constants BAKED IN, so it can serve exactly two
  // distributions. Match the caller's params EXACTLY against the shared
  // factories and refuse anything else.
  //
  // The deleted code sniffed `params.mean[0] == 0` to choose between them and
  // dropped order / inv_std / inv_scale / letterbox entirely: any third
  // normalization was silently rendered as det's, and a BGR-vs-RGB request had
  // no effect at all. Every other backend honours `order`
  // (host_kernels.cpp, shaders.metal, sycl_kernels.cpp, preprocess_kernels.hip).
  if (backend::norm_equal(params, backend::norm::layout_norm())) {
    kernels::cuda_fused_resize_normalize_layout(g, dst_chw, dst_w, dst_h, stream);
    return;
  }
  if (backend::norm_equal(params, backend::norm::imagenet_bgr())) {
    kernels::cuda_fused_resize_normalize_det(g, dst_chw, dst_w, dst_h, stream);
    return;
  }
  // ON-HARDWARE TODO: parameterize src/backends/nvidia/kernels_cuda/preprocess_kernels.cu the way
  // the HIP port already is (hip_fused_resize_normalize takes mean/std/
  // inv_scale/rgb_out) and delete this whitelist. Until then, refuse loudly —
  // never substitute.
  //
  // The refusal is the SHARED guard (kernels.h), driven by
  // caps().params.norm_mean_std_full_frame = false declared above; the two
  // norm_equal tests are only the positive fast-path match. Hand-rolling the
  // refusal here left that flag unread, so a backend that declared the same
  // limitation without copying this block would substitute silently.
  (void)backend::refuse_unbaked_norm(
      params, caps().params, "CudaKernels::resize_normalize",
      backend::NormPath::FullFrame,
      "NormParams other than norm::imagenet_bgr()/norm::layout_norm() "
      "(the only two distributions baked into the .cu)");
}

void CudaKernels::warp_crops(const backend::ImageView &src, const float *d_M_invs,
                             const int *d_crop_widths, float *d_dst_batch,
                             int batch_size, int dst_h, int dst_w,
                             const backend::NormParams &params,
                             backend::DeviceQueue &queue) {
  // BINDING: cuda_batch_roi_warp always emits RGB planes, so an `order` request
  // of BGR cannot be honoured — refuse rather than hand back swapped channels.
  // NormPath::Warp additionally consults caps().params.norm_mean_std_warp, which
  // stays true here (this warp DOES take real mean/inv_std/inv_scale), so the
  // added path argument refuses nothing new.
  if (!backend::require_norm_supported(params, caps().params,
                                       "CudaKernels::warp_crops",
                                       backend::NormPath::Warp))
    return;
  kernels::cuda_batch_roi_warp(to_gpu_image(src), d_M_invs, d_crop_widths,
                               d_dst_batch, batch_size, dst_h, dst_w,
                               cuda_stream(queue), params.mean[0], params.mean[1],
                               params.mean[2], params.inv_std[0],
                               params.inv_std[1], params.inv_std[2],
                               params.inv_scale);
}

void CudaKernels::threshold(const float *src, std::uint8_t *dst, int w, int h,
                            int batch_size, float thresh,
                            backend::DeviceQueue &queue) {
  const cudaStream_t stream = cuda_stream(queue);
  if (batch_size <= 1)
    kernels::cuda_threshold_to_u8(src, dst, w, h, thresh, stream);
  else
    kernels::cuda_batch_threshold_to_u8(src, dst, w, h, batch_size, thresh,
                                        stream);
}

void CudaKernels::ensure_db_scratch(int w, int h) {
  if (!db_)
    db_ = std::make_unique<DbScratch>();
  const int pixels = w * h;
  if (pixels <= db_->cap_pixels)
    return;
  const auto p = static_cast<size_t>(pixels);
  db_->d_labels.reset(p);
  db_->d_compact_ids.reset(p);
  db_->d_id_counter.reset(1);
  db_->d_bboxes.reset(static_cast<size_t>(kernels::kMaxGpuComponents) * 2);
  db_->d_num_boxes.reset(1);
  db_->d_jfa_labels.reset(p);
  db_->d_jfa_seeds.reset(p);
  db_->d_jfa_seeds_alt.reset(p);
  db_->d_expand.reset(static_cast<size_t>(kernels::kMaxGpuComponents));
  db_->d_perim.reset(static_cast<size_t>(kernels::kMaxGpuComponents));
  db_->h_boxes.reset(static_cast<size_t>(kernels::kMaxGpuComponents));
  db_->cap_pixels = pixels;
}

std::vector<turbo_ocr::Box>
CudaKernels::db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap,
                            int w, int h, const backend::DbPostParams &params,
                            backend::DeviceQueue &queue) {
  // Reference note: the AUTHORITATIVE, regression-gated DB post-process is
  // PaddleDet::run_gpu_ccl / run_gpu_ccl_fast (owned by NvDetector). This op
  // reproduces the mode-2 all-GPU JFA path (AXIS-ALIGNED ONLY) for the generic
  // pipeline / non-detector callers, and MUST be byte-diffed against PaddleDet
  // on hardware before it replaces any detector call site.
  //
  // BINDING (kernels.h): this path cannot emit rotated quads and has no
  // per-call component budget. It used to accept params.oriented == true and
  // return AABBs anyway — different crop geometry, therefore different
  // recognized text, on NVIDIA alone, with caps() reporting success. AMD's
  // hip_jfa_extract_oriented is the reference for implementing it properly.
  if (!backend::require_db_supported(params, caps().params,
                                     "CudaKernels::db_postprocess"))
    return {};
  ensure_db_scratch(w, h);
  const cudaStream_t stream = cuda_stream(queue);

  const int num = kernels::cuda_gpu_ccl_detect(
      d_bitmap, d_pred_map, w, h, params.box_thresh, db_->d_labels.get(),
      db_->d_compact_ids.get(), db_->d_id_counter.get(), db_->d_bboxes.get(),
      db_->d_num_boxes.get(), db_->h_boxes.get(), &db_->h_num_boxes, stream,
      &db_->h_num_total);
  (void)num;

  const int slots = std::min(db_->h_num_total, kernels::kMaxGpuComponents);
  kernels::cuda_accumulate_crack_perimeter(db_->d_compact_ids.get(), d_bitmap, w,
                                           h, slots, db_->d_perim.get(), stream);
  kernels::cuda_compute_expand_per_comp(
      db_->d_bboxes.get(), db_->d_perim.get(), slots, params.unclip_ratio,
      params.min_expand, params.max_expand, params.box_thresh,
      db_->d_expand.get(), stream);
  kernels::cuda_jfa_expand_labels(d_bitmap, db_->d_compact_ids.get(),
                                  db_->d_expand.get(), db_->d_jfa_labels.get(), w,
                                  h, params.max_expand, db_->d_jfa_seeds.get(),
                                  db_->d_jfa_seeds_alt.get(), stream);
  kernels::cuda_jfa_extract_bboxes(db_->d_jfa_labels.get(), w, h,
                                   db_->d_bboxes.get(), slots, stream);
  // One sync to bring the small post-expand bbox array back to the host.
  CUDA_CHECK(cudaMemcpyAsync(db_->h_boxes.get(), db_->d_bboxes.get(),
                             static_cast<size_t>(slots) *
                                 sizeof(kernels::GpuDetBox),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<turbo_ocr::Box> boxes;
  boxes.reserve(static_cast<size_t>(slots));
  for (int i = 0; i < slots; ++i) {
    const auto &b = db_->h_boxes.get()[i];
    if (b.pixel_count == 0)
      continue; // empty / filtered slot
    // Side gate = params.min_unclipped_side (post-expand boxes, so the
    // post-unclip limit applies) — the caller's values, which is what flips
    // caps().params.db_side_limits to true. This loop used to have NO side
    // gate at all: every 1-2px sliver the CCL emitted became a rec call
    // producing garbage text the shared reference would have dropped.
    const int bw = b.xmax - b.xmin + 1;
    const int bh = b.ymax - b.ymin + 1;
    if (static_cast<float>(bw) < params.min_unclipped_side ||
        static_cast<float>(bh) < params.min_unclipped_side)
      continue;
    // Boxes are in the (resized) map coordinate space; the caller rescales to
    // original dims. AABB quad [tl, tr, br, bl] (Box stores integer corners in
    // `pts`; DB scores are not carried on Box — the detector applies box_thresh
    // during CCL, so a returned box already passed the score filter).
    turbo_ocr::Box box{};
    box.pts = {{{b.xmin, b.ymin}, {b.xmax, b.ymin}, {b.xmax, b.ymax}, {b.xmin, b.ymax}}};
    boxes.push_back(box);
  }
  return boxes;
}

void CudaKernels::argmax(const float *input_probs, int *output_indices,
                         float *output_scores, int batch_size, int seq_len,
                         int num_classes, backend::DeviceQueue &queue) {
  kernels::cuda_argmax(input_probs, output_indices, output_scores, batch_size,
                       seq_len, num_classes, cuda_stream(queue));
}

void CudaKernels::preprocess_region(const backend::ImageView &src,
                                    const backend::Rect &rect,
                                    backend::PreprocKind kind, float *dst_chw,
                                    backend::DeviceQueue &queue) {
  const auto g = to_gpu_image(src);
  const cudaStream_t stream = cuda_stream(queue);
  switch (kind) {
  case backend::PreprocKind::LayoutSubRect: {
    // Size from the seam, not a literal: host/CUDA/SYCL all preprocess to the
    // same target and three private copies of 800 must never disagree.
    const int sz = backend::preproc_geometry(kind).target;
    kernels::cuda_fused_resize_normalize_layout(g, rect.x, rect.y, rect.w, rect.h,
                                                dst_chw, sz, sz, stream);
    break;
  }
  case backend::PreprocKind::TableCls:
    kernels::cuda_fused_table_cls_pre(g, rect.x, rect.y, rect.w, rect.h, dst_chw,
                                      stream);
    break;
  case backend::PreprocKind::SlanextBGR:
    kernels::cuda_fused_slanext_pre(g, rect.x, rect.y, rect.w, rect.h, dst_chw,
                                    stream);
    break;
  case backend::PreprocKind::SlanextRGB:
    kernels::cuda_fused_slanext_pre_rgb(g, rect.x, rect.y, rect.w, rect.h,
                                        dst_chw, stream);
    break;
  }
}

} // namespace turbo_ocr::nvidia
