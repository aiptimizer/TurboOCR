#pragma once

// kernels_hip.h — hipified public signatures of the CUDA pre/post kernel set.
//
// This is the AMD (HIP) mirror of src/backends/nvidia/kernels_cuda/kernels_cuda.h. The
// kernel *bodies* in the .hip translation units are byte-for-byte the same math
// as src/backends/nvidia/kernels_cuda/*.cu; only the device runtime types change (cudaStream_t ->
// hipStream_t, cuda_runtime.h -> hip/hip_runtime.h, the cuda* host API ->
// hip*). Everything is single-source under hipcc (__HIP_PLATFORM_AMD__).
//
// This header intentionally does NOT include the shared kernels.h — that pulls
// <cuda_runtime.h> and decode::GpuImage. Instead we mirror the two POD types the
// kernels need (HipImage, GpuDetBox) so the AMD kernel library has no CUDA
// include dependency. The device-agnostic IKernels wrapper (amd/kernels_hip/hip_kernels.h)
// translates backend::ImageView <-> HipImage at the seam.

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h> // hipStream_t (host-side signatures only)

namespace turbo_ocr::amd::kernels {

// Non-owning device image descriptor — the HIP mirror of decode::GpuImage /
// backend::ImageView. `data` is a HIP device pointer (hipMalloc). Interleaved
// 8-bit BGR, `step` = row pitch in bytes.
struct HipImage {
  const void *data = nullptr;
  std::size_t step = 0;
  int rows = 0;
  int cols = 0;
};

// One connected component result (D2H-transferred). Identical layout to
// kernels::GpuDetBox in the CUDA path so host-side DB decode is shared verbatim.
struct GpuDetBox {
  int xmin, ymin, xmax, ymax; // bbox in resize coords
  float score;                // mean pred within bbox
  int pixel_count;            // fg pixel count
  float ox[4], oy[4];         // oriented rect corners (mode-2 only)
};

// Sized to the PP-OCRv6 DB candidate budget, matching the CUDA constant.
static constexpr int kMaxGpuComponents = 3000;

// --- Preprocess -------------------------------------------------------------

// `rgb_out`: 1 => output planes are R,G,B (the rec/cls convention and the
// historical default); 0 => B,G,R. mean/inv_std are POSITIONAL either way.
// Added so HipKernels can HONOUR NormParams::order instead of dropping it — see
// the PARAMETER CONTRACT in include/turbo_ocr/backend/kernels.h.
void hip_batch_roi_warp(const HipImage &src, const float *d_M_invs,
                        const int *d_crop_widths, float *d_dst_batch,
                        int batch_size, int dst_h, int dst_w,
                        hipStream_t stream = nullptr,
                        float mean0 = 0.5f, float mean1 = 0.5f, float mean2 = 0.5f,
                        float inv_std0 = 2.0f, float inv_std1 = 2.0f,
                        float inv_std2 = 2.0f, float inv_scale = 1.0f / 255.0f,
                        int rgb_out = 1);

void hip_fused_resize_normalize_det(const HipImage &src, float *dst_chw,
                                    int dst_w, int dst_h,
                                    hipStream_t stream = nullptr);

void hip_fused_resize_normalize_layout(const HipImage &src, float *dst_chw,
                                       int dst_w, int dst_h,
                                       hipStream_t stream = nullptr);

// Param-driven full-frame resize+normalize+CHW. Same kernel as the two baked
// variants above (calling it with the det constants is bit-identical to
// hip_fused_resize_normalize_det); it exists so the IKernels wrapper can honour
// an arbitrary NormParams instead of guessing which baked variant was meant.
// `rgb_out`: 0 => output planes keep the source B,G,R order (the det/layout
// convention); 1 => R,G,B. mean/inv_std are POSITIONAL either way.
void hip_fused_resize_normalize(const HipImage &src, float *dst_chw, int dst_w,
                                int dst_h, float mean0, float mean1, float mean2,
                                float inv_std0, float inv_std1, float inv_std2,
                                float inv_scale, int rgb_out = 0,
                                hipStream_t stream = nullptr);

void hip_batch_fused_resize_normalize_det(
    const void *const *d_src_ptrs, const int *d_src_steps,
    const int *d_src_heights, const int *d_src_widths,
    const int *d_dst_heights, const int *d_dst_widths,
    float *dst_chw, int dst_w, int dst_h, int batch_size,
    hipStream_t stream = nullptr);

// --- Reduce -----------------------------------------------------------------

void hip_argmax(const float *input_probs, int *output_indices,
                float *output_scores, int batch_size, int seq_len,
                int num_classes, hipStream_t stream = nullptr);

void hip_threshold_to_u8(const float *src, std::uint8_t *dst, int w, int h,
                         float thresh, hipStream_t stream = nullptr);

void hip_batch_threshold_to_u8(const float *src, std::uint8_t *dst, int w, int h,
                               int batch_size, float thresh,
                               hipStream_t stream = nullptr);

// --- DB post: CCL ------------------------------------------------------------

int hip_gpu_ccl_detect(const std::uint8_t *d_bitmap, const float *d_pred_map,
                       int w, int h, float box_thresh, int *d_labels,
                       int *d_compact_ids, int *d_id_counter,
                       GpuDetBox *d_bboxes, int *d_num_boxes,
                       GpuDetBox *h_boxes, int *h_num_boxes, hipStream_t stream,
                       int *h_num_total = nullptr);

// --- DB post: bounded Euclidean unclip ("JFA" path) --------------------------

void hip_jfa_expand_labels(const std::uint8_t *d_bitmap,
                           const std::int32_t *d_compact_ids,
                           const float *d_expand_per_comp,
                           std::uint32_t *d_expanded_labels, int w, int h,
                           float max_expand, std::uint32_t *d_seeds,
                           std::uint32_t *d_seeds_alt, hipStream_t stream);

void hip_accumulate_crack_perimeter(const std::int32_t *d_compact_ids,
                                    const std::uint8_t *d_bitmap, int w, int h,
                                    int num_slots, int *d_perim_per_comp,
                                    hipStream_t stream);

void hip_compute_expand_per_comp(const GpuDetBox *d_bboxes,
                                 const int *d_perim_per_comp, int num_slots,
                                 float unclip_ratio, float min_expand,
                                 float max_expand, float box_thresh,
                                 float *d_expand_per_comp, hipStream_t stream);

void hip_jfa_extract_bboxes(const std::uint32_t *d_expanded_labels, int w, int h,
                            GpuDetBox *d_bboxes, int num_slots,
                            hipStream_t stream);

void hip_jfa_extract_oriented(const std::uint32_t *d_expanded_labels, int w,
                              int h, GpuDetBox *d_bboxes, int num_slots,
                              unsigned long long *d_moments, float *d_orient,
                              hipStream_t stream);

// --- Fused region preprocessors (table/layout) -------------------------------
// HIP ports of src/backends/nvidia/stages/table_kernels.cu; back IKernels::
// preprocess_region's four PreprocKind variants. Output tensors are the model
// input, so these are transcribed verbatim for fp32 parity (see
// table_kernels.hip header).

// resize-short(256) -> center-crop(224) -> ImageNet -> BGR CHW [3*224*224].
void hip_fused_table_cls_pre(const HipImage &src, int rect_x, int rect_y,
                             int rect_w, int rect_h, float *dst_chw,
                             hipStream_t stream = nullptr);

// ResizeByLong(488) preserve-AR -> ImageNet -> bottom-right pad -> BGR CHW.
void hip_fused_slanext_pre(const HipImage &src, int rect_x, int rect_y,
                           int rect_w, int rect_h, float *dst_chw,
                           hipStream_t stream = nullptr);

// 488 letterbox, RGB order + ImageNet + pad-0 in normalized space.
void hip_fused_slanext_pre_rgb(const HipImage &src, int rect_x, int rect_y,
                               int rect_w, int rect_h, float *dst_chw,
                               hipStream_t stream = nullptr);

// Sub-rect layout preprocess (cell-det): resize sub-rect, /255, BGR CHW.
void hip_fused_resize_normalize_layout_subrect(const HipImage &src, int rect_x,
                                               int rect_y, int rect_w,
                                               int rect_h, float *dst_chw,
                                               int dst_w, int dst_h,
                                               hipStream_t stream = nullptr);

} // namespace turbo_ocr::amd::kernels
