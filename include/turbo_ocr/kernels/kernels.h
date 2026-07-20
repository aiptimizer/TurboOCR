#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "turbo_ocr/common/geometry/perspective_math.h"
#include "turbo_ocr/decode/gpu_image.h"

namespace turbo_ocr::kernels {

using decode::GpuImage;

// Fused batched ROI warp (perspective) + resize + normalize for recognition
// Normalization defaults to REC's (pixel/255 - 0.5)/0.5. cls passes ImageNet
// mean/std after `stream`. res channels are RGB, so mean/std are in RGB order.
void cuda_batch_roi_warp(const GpuImage &src, const float *d_M_invs,
                         const int *d_crop_widths, float *d_dst_batch,
                         int batch_size, int dst_h, int dst_w,
                         cudaStream_t stream = 0,
                         float mean0 = 0.5f, float mean1 = 0.5f, float mean2 = 0.5f,
                         float inv_std0 = 2.0f, float inv_std1 = 2.0f, float inv_std2 = 2.0f,
                         float inv_scale = 1.0f / 255.0f);

// ArgMax for CTC decoding
void cuda_argmax(const float *input_probs, int *output_indices,
                 float *output_scores, int batch_size, int seq_len,
                 int num_classes, cudaStream_t stream = 0);

// Fused resize + normalize + CHW for detection (eliminates intermediate buffer)
void cuda_fused_resize_normalize_det(const GpuImage &src, float *dst_chw,
                                      int dst_w, int dst_h,
                                      cudaStream_t stream = 0);

// Fused resize + normalize + CHW for PP-DocLayoutV3. Identical kernel to the
// det variant, but applies `pixel / 255` normalization (mean=0, std=1) to
// match the model's inference.yml NormalizeImage(norm_type=none) step.
// Input is expected as BGR uint8; output is float CHW at dst_h x dst_w.
void cuda_fused_resize_normalize_layout(const GpuImage &src, float *dst_chw,
                                         int dst_w, int dst_h,
                                         cudaStream_t stream = 0);

// Sub-rect overload for cell-det. Resizes the sub-rect (rect_x, rect_y,
// rect_w, rect_h) of `src` to dst_w × dst_h with `pixel / 255` normalization
// (mean=0, std=1), BGR CHW. Implementation lives in src/table/kernels/.
void cuda_fused_resize_normalize_layout(const GpuImage &src,
                                        int rect_x, int rect_y,
                                        int rect_w, int rect_h,
                                        float *dst_chw,
                                        int dst_w, int dst_h,
                                        cudaStream_t stream = 0);

// TableCls preprocess: from sub-rect of src GpuImage, resize-short(256) →
// center-crop(224) → ImageNet normalize → BGR CHW into dst_chw[3*224*224].
// Implementation in src/table/kernels/table_kernels.cu.
void cuda_fused_table_cls_pre(const GpuImage &src,
                              int rect_x, int rect_y,
                              int rect_w, int rect_h,
                              float *dst_chw,
                              cudaStream_t stream = 0);

// SLANeXt preprocess: from sub-rect, ResizeByLong(488) preserve AR →
// ImageNet normalize → bottom-right pad with PAD_VALUE=(0-mean)/std →
// BGR CHW into dst_chw[3*488*488].
// Implementation in src/table/kernels/table_kernels.cu.
void cuda_fused_slanext_pre(const GpuImage &src,
                            int rect_x, int rect_y,
                            int rect_w, int rect_h,
                            float *dst_chw,
                            cudaStream_t stream = 0);

// SLANeXt encoder-split preprocess: 488 letterbox, RGB channel order + ImageNet
// norm + pad 0 in normalized space (matches PaddleOCR DecodeImage img_mode=RGB
// + PaddingTableImage). For the encoder-split table backend.
void cuda_fused_slanext_pre_rgb(const GpuImage &src,
                                int rect_x, int rect_y,
                                int rect_w, int rect_h,
                                float *dst_chw,
                                cudaStream_t stream = 0);

// Batched fused resize + normalize + CHW for detection
// Processes N images (each with different src dimensions) into a single
// batched CHW tensor [N, 3, dst_h, dst_w].
// d_src_ptrs[N], d_src_steps[N], d_src_heights[N], d_src_widths[N] are
// device arrays describing each source image.
// dst_heights/dst_widths: per-image letterbox content dims inside the
// (dst_h, dst_w) canvas; the remainder is padded with normalized black.
void cuda_batch_fused_resize_normalize_det(
    const void *const *d_src_ptrs, const int *d_src_steps,
    const int *d_src_heights, const int *d_src_widths,
    const int *d_dst_heights, const int *d_dst_widths,
    float *dst_chw, int dst_w, int dst_h, int batch_size,
    cudaStream_t stream = 0);

// Batched threshold + float->uint8 (processes batch_size * w * h elements)
void cuda_batch_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                                int batch_size, float thresh,
                                cudaStream_t stream = 0);

// Fused threshold + float->uint8
void cuda_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                          float thresh, cudaStream_t stream = 0);

// Compute inverse perspective transform: maps dst quad to src quad.
// Delegates to turbo_ocr::compute_perspective_inv in common/perspective_math.h
// (CUDA-free pure math). Kept here for backward compatibility.
inline void compute_perspective_inv(
    const float* dst_pts, const float* src_pts,
    float* M_inv) {
  turbo_ocr::compute_perspective_inv(dst_pts, src_pts, M_inv);
}

// --- GPU Connected Component Labeling + BBox Extraction ---

// Result struct for one connected component (transferred from GPU to CPU)
struct GpuDetBox {
  int xmin, ymin, xmax, ymax; // bounding box in resize coords
  float score;                // mean of pred_map within bbox
  int pixel_count;            // number of foreground pixels in component
  // Oriented (rotated) min-area-rect corners over the expanded region, in
  // resize coords. Filled ONLY by the mode-2 oriented extract path
  // (cuda_jfa_extract_oriented); the axis-aligned CCL path leaves them unused.
  float ox[4], oy[4];
};

// Maximum number of components we track on GPU. Sized to the PP-OCRv6 DB
// candidate budget (3000) so dense multi-column pages are not truncated.
static constexpr int kMaxGpuComponents = 3000;

// Run full GPU CCL pipeline: label components, extract bboxes, compute scores.
// Returns number of valid boxes written to h_boxes (host memory).
// All GPU work is on the given stream. This function synchronizes the stream
// exactly ONCE at the end to transfer the small result array to the host.
//
// Required GPU buffers (ALL pre-allocated by caller, no per-request alloc):
//   d_labels:       int[w*h]                      -- label map
//   d_compact_ids:  int[w*h]                      -- compact component IDs
//   d_id_counter:   int[1]                        -- atomic counter for compact IDs
//   d_bboxes:       GpuDetBox[kMaxGpuComponents*2] -- per-component bbox + filtered output
//   d_num_boxes:    int[1]                        -- output count
//
// h_boxes must point to at least kMaxGpuComponents GpuDetBox entries (pinned).
int cuda_gpu_ccl_detect(
    const uint8_t *d_bitmap,     // binary bitmap (255=fg, 0=bg)
    const float *d_pred_map,     // raw probability map
    int w, int h,
    float box_thresh,            // score threshold for filtering
    int *d_labels,               // [w*h] scratch
    int *d_compact_ids,          // [w*h] scratch
    int *d_id_counter,           // [1] scratch
    GpuDetBox *d_bboxes,         // [kMaxGpuComponents*2] scratch
    int *d_num_boxes,            // [1] scratch
    GpuDetBox *h_boxes,          // host output (pinned)
    int *h_num_boxes,            // host output count
    cudaStream_t stream,
    int *h_num_total = nullptr); // optional: pre-filter component total

// Exact bounded per-component Euclidean unclip (all-GPU, no merges, no
// pred_map download): matches Clipper's polygon-offset distance
// area*ratio/perimeter per component. Each foreground boundary pixel scatters
// its exact squared distance + label into every pixel within that component's
// expand radius; an atomicMin picks the exact nearest reaching component per
// pixel — no approximate global distance transform, exact at the boundaries
// between adjacent text.
//   d_compact_ids       = CCL compact label map (int32_t, -1=bg, 0..N-1)
//   d_expand_per_comp   = float[kMaxGpuComponents], per-component expand (px)
//   d_expanded_labels   = uint32_t output (1..N, 0=bg)
// d_seeds is repurposed as the per-pixel winner-key scratch (uint32[w*h]);
// d_seeds_alt and max_expand are retained for interface stability (each stamp
// is bounded by its own component radius, so no global bound is needed).
void cuda_jfa_expand_labels(const uint8_t *d_bitmap,
                            const int32_t *d_compact_ids,
                            const float *d_expand_per_comp,
                            uint32_t *d_expanded_labels,
                            int w, int h, float max_expand,
                            uint32_t *d_seeds, uint32_t *d_seeds_alt,
                            cudaStream_t stream);

// Accumulate each component's contour perimeter as its exposed 4-crack-edge
// count (cityblock/staircase boundary length ≈ cv2.arcLength). One thread per
// pixel, atomicAdd into a per-component int counter. Indexed by PRE-filter
// compact_id; the buffer is zeroed internally before accumulation. Must run
// BEFORE cuda_compute_expand_per_comp, which uses it as the expand divisor.
//   d_perim_per_comp = int[kMaxGpuComponents]
void cuda_accumulate_crack_perimeter(
    const int32_t *d_compact_ids, const uint8_t *d_bitmap,
    int w, int h, int num_slots, int *d_perim_per_comp, cudaStream_t stream);

// Compute per-component expand distance from PRE-filter CCL bboxes.
// Indexed by PRE-filter compact_id (matches what compact_ids[] stores) so JFA
// expand can look up expand_per_comp[compact_ids[seed]] directly. Empty /
// size-rejected / score-rejected slots get expand=0 → JFA treats as bg.
// d_perim_per_comp is the per-component contour perimeter from
// cuda_accumulate_crack_perimeter, used as the area*ratio/perimeter divisor.
void cuda_compute_expand_per_comp(
    const GpuDetBox *d_bboxes, const int *d_perim_per_comp, int num_slots,
    float unclip_ratio, float min_expand, float max_expand,
    float box_thresh, float *d_expand_per_comp, cudaStream_t stream);

// Image-wide bbox extraction over the expanded label map. One thread per
// pixel, atomicMin/Max/Add scatter into the per-component bbox slot.
// Empty / filtered-out slots end up with pixel_count == 0.
void cuda_jfa_extract_bboxes(const uint32_t *d_expanded_labels,
                             int w, int h,
                             GpuDetBox *d_bboxes, int num_slots,
                             cudaStream_t stream);

// Oriented (rotated) min-area-rect extraction over the expanded label map,
// fully on GPU. Superset of cuda_jfa_extract_bboxes: it fills the axis-aligned
// bbox + pixel_count + the PCA-oriented rect corners (GpuDetBox::ox/oy) so the
// mode-2 host path emits rotated quads matching the CPU minAreaRect geometry.
//   d_moments = uint64[num_slots*6] scratch (n,sx,sy,sxx,syy,sxy)
//   d_orient  = float[num_slots*6]  scratch (cos,sin,umin,umax,vmin,vmax)
// Both are zeroed/seeded internally. Only the small d_bboxes array is later
// copied to host (unchanged transfer contract).
void cuda_jfa_extract_oriented(const uint32_t *d_expanded_labels,
                               int w, int h,
                               GpuDetBox *d_bboxes, int num_slots,
                               unsigned long long *d_moments, float *d_orient,
                               cudaStream_t stream);

} // namespace turbo_ocr::kernels
