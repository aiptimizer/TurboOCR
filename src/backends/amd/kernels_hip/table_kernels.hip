// table_kernels.hip — HIP port of src/backends/nvidia/stages/table_kernels.cu.
//
// The four FUSED REGION preprocessors behind IKernels::preprocess_region:
//   TableCls      — resize-short(256) -> center-crop(224) -> ImageNet -> BGR CHW
//   SlanextBGR    — ResizeByLong(488) preserve-AR -> ImageNet -> bottom-right pad
//   SlanextRGB    — 488 letterbox, RGB order + ImageNet + pad-0 (encoder-split)
//   LayoutSubRect — sub-rect resize, pixel/255, mean=0 std=1, BGR CHW
//
// NUMERICAL PARITY IS LOAD-BEARING: these tensors ARE the model input, and the
// AMD outputs must match the CUDA/CPU reference to the last fp32 ulp or the
// table/formula stages drift. The kernel bodies below are therefore transcribed
// VERBATIM from the CUDA source — same half-pixel-center coordinate map, same
// index-clamp border rule (bit-identical to cv2's frac-zeroing; proof in the
// CUDA file header), same 4-weight bilinear form in the same operand order so
// the FFMA contraction matches, same per-byte __ldg loads (a 3-byte BGR pixel is
// not vector-aligned).
//
// ONE AMD-SPECIFIC RISK, stated honestly: fp-contraction. amdclang++ contracts
// `a*b + c` into v_fma_f32 by default, exactly as nvcc does, so the emitted
// arithmetic should match. But contraction is a COMPILER choice, not a language
// guarantee — if the per-stage golden diff (see README validation) shows a
// 1-ulp-class drift on this path, build this TU with -ffp-contract=fast-honor-
// pragmas (or =off on BOTH sides) rather than "fixing" the source. Do not
// reorder these expressions.
//
// WARP-SIZE: none of these kernels use wavefront primitives, shared memory, or
// reductions — they are pure per-output-pixel maps. The 32-vs-64 lane
// difference is irrelevant here; they port unchanged to CDNA and RDNA.

#include "kernels_hip.h"
#include "amd/support/hip_check.h"

#include <hip/hip_runtime.h>

namespace turbo_ocr::amd::kernels {

namespace {

// One BGR pixel via the read-only cache. Per-byte because a 3-byte pixel is not
// vector-aligned (a wide vector load would be misaligned/UB).
__device__ __forceinline__ uchar3 tk_ldg_bgr(const unsigned char *__restrict__ row,
                                             int px) {
  const unsigned char *p = row + px * 3;
  return make_uchar3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
}

// Bilinear sample within a sub-rect of a BGR uint8 image, normalize per channel.
// (sx_rel, sy_rel) are source-pixel coords RELATIVE to the sub-rect top-left,
// already in the half-pixel-center frame.
__device__ __forceinline__ void tk_bilinear_sample_norm(
    const unsigned char *__restrict__ src, int src_step, int rect_x, int rect_y,
    int rect_w, int rect_h, float sx_rel, float sy_rel, float mean0, float mean1,
    float mean2, float inv_std0, float inv_std1, float inv_std2, float inv_255,
    float &b_out, float &g_out, float &r_out) {
  int x0 = (int)floorf(sx_rel);
  int y0 = (int)floorf(sy_rel);
  float fx = sx_rel - x0;
  float fy = sy_rel - y0;
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  x0 = max(0, min(x0, rect_w - 1));
  x1 = max(0, min(x1, rect_w - 1));
  y0 = max(0, min(y0, rect_h - 1));
  y1 = max(0, min(y1, rect_h - 1));

  const unsigned char *row0 = src + (rect_y + y0) * src_step;
  const unsigned char *row1 = src + (rect_y + y1) * src_step;
  int ax0 = rect_x + x0;
  int ax1 = rect_x + x1;

  uchar3 p00 = tk_ldg_bgr(row0, ax0), p10 = tk_ldg_bgr(row0, ax1);
  uchar3 p01 = tk_ldg_bgr(row1, ax0), p11 = tk_ldg_bgr(row1, ax1);

  // Weights computed once, shared across B/G/R. Operand order preserved verbatim
  // for bit-exact parity with the CUDA export path.
  float w00 = (1.0f - fx) * (1.0f - fy);
  float w10 = fx * (1.0f - fy);
  float w01 = (1.0f - fx) * fy;
  float w11 = fx * fy;

  float b = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  float g = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  float r = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;

  b_out = (b * inv_255 - mean0) * inv_std0;
  g_out = (g * inv_255 - mean1) * inv_std1;
  r_out = (r * inv_255 - mean2) * inv_std2;
}

// ResizeByLong letterbox extents (shared by both SLANeXt host wrappers).
// cv2.resize samples PER AXIS from the ROUNDED output dim: scale = src/new. On
// the long axis new==dst so this equals 1/s, but the short axis rounds, so a
// single 1/s would drift up to ~1px there. Match cv2 exactly.
struct LetterboxLong {
  int new_w, new_h;
  float scale_x, scale_y; // src px per resized px, per axis
};

inline LetterboxLong resize_by_long(int rect_w, int rect_h, int dst) {
  float s = (float)dst / (float)(rect_w > rect_h ? rect_w : rect_h);
  int new_w = (int)lroundf((float)rect_w * s);
  int new_h = (int)lroundf((float)rect_h * s);
  new_w = new_w < 1 ? 1 : (new_w > dst ? dst : new_w);
  new_h = new_h < 1 ? 1 : (new_h > dst ? dst : new_h);
  return {new_w, new_h, (float)rect_w / (float)new_w,
          (float)rect_h / (float)new_h};
}

// 32-wide x (one 32-column run -> coalesced plane stores), 8 rows => 256
// threads/block, matching the det/layout kernels. NOTE: 32 here is a MEMORY
// COALESCING tile width, NOT a warp width — it is deliberately unchanged on
// AMD, where a 64-lane wavefront simply covers two rows of the tile. Nothing in
// these kernels communicates across lanes, so the tile shape is free to stay.
inline dim3 pre_block() { return dim3(32, 8); }
inline dim3 pre_grid(int w, int h) {
  dim3 b = pre_block();
  return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y);
}

} // namespace

// =================================================================
// TableCls: resize-short(256) + center-crop(224) + ImageNet + BGR CHW
// =================================================================
__global__ __launch_bounds__(256)
void table_cls_pre_kernel(const unsigned char *__restrict__ src, int src_step,
                          int rect_x, int rect_y, int rect_w, int rect_h,
                          float *__restrict__ dst_chw, int dst, float scale,
                          float crop_off_x_src, float crop_off_y_src,
                          float mean0, float mean1, float mean2, float inv_std0,
                          float inv_std1, float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;

  float sx_rel = (dx + 0.5f) * scale - 0.5f + crop_off_x_src;
  float sy_rel = (dy + 0.5f) * scale - 0.5f + crop_off_y_src;

  float b, g, r;
  tk_bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h, sx_rel,
                          sy_rel, mean0, mean1, mean2, inv_std0, inv_std1,
                          inv_std2, inv_255, b, g, r);

  int idx = dy * dst + dx;
  int plane = dst * dst;
  dst_chw[0 * plane + idx] = b;
  dst_chw[1 * plane + idx] = g;
  dst_chw[2 * plane + idx] = r;
}

void hip_fused_table_cls_pre(const HipImage &src, int rect_x, int rect_y,
                             int rect_w, int rect_h, float *dst_chw,
                             hipStream_t stream) {
  constexpr int DST = 224;
  constexpr int SHORT = 256;
  float scale = (float)(rect_w < rect_h ? rect_w : rect_h) / (float)SHORT;
  float crop_off_x_src = ((float)rect_w - DST * scale) * 0.5f;
  float crop_off_y_src = ((float)rect_h - DST * scale) * 0.5f;

  table_cls_pre_kernel<<<pre_grid(DST, DST), pre_block(), 0, stream>>>(
      (const unsigned char *)src.data, (int)src.step, rect_x, rect_y, rect_w,
      rect_h, dst_chw, DST, scale, crop_off_x_src, crop_off_y_src, 0.485f,
      0.456f, 0.406f, 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f,
      1.0f / 255.0f);
  HIP_CHECK(hipGetLastError());
}

// =================================================================
// SLANeXt (BGR): ResizeByLong(488) + ImageNet + bottom-right pad + CHW
// =================================================================
__global__ __launch_bounds__(256)
void slanext_pre_kernel(const unsigned char *__restrict__ src, int src_step,
                        int rect_x, int rect_y, int rect_w, int rect_h,
                        float *__restrict__ dst_chw, int dst, int new_w,
                        int new_h, float scale_x, float scale_y, float mean0,
                        float mean1, float mean2, float inv_std0, float inv_std1,
                        float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;

  int idx = dy * dst + dx;
  int plane = dst * dst;

  if (dx < new_w && dy < new_h) {
    float sx_rel = (dx + 0.5f) * scale_x - 0.5f;
    float sy_rel = (dy + 0.5f) * scale_y - 0.5f;
    float b, g, r;
    tk_bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                            sx_rel, sy_rel, mean0, mean1, mean2, inv_std0,
                            inv_std1, inv_std2, inv_255, b, g, r);
    dst_chw[0 * plane + idx] = b;
    dst_chw[1 * plane + idx] = g;
    dst_chw[2 * plane + idx] = r;
  } else {
    dst_chw[0 * plane + idx] = (-mean0) * inv_std0;
    dst_chw[1 * plane + idx] = (-mean1) * inv_std1;
    dst_chw[2 * plane + idx] = (-mean2) * inv_std2;
  }
}

void hip_fused_slanext_pre(const HipImage &src, int rect_x, int rect_y,
                           int rect_w, int rect_h, float *dst_chw,
                           hipStream_t stream) {
  constexpr int DST = 488;
  LetterboxLong lb = resize_by_long(rect_w, rect_h, DST);

  slanext_pre_kernel<<<pre_grid(DST, DST), pre_block(), 0, stream>>>(
      (const unsigned char *)src.data, (int)src.step, rect_x, rect_y, rect_w,
      rect_h, dst_chw, DST, lb.new_w, lb.new_h, lb.scale_x, lb.scale_y, 0.485f,
      0.456f, 0.406f, 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f,
      1.0f / 255.0f);
  HIP_CHECK(hipGetLastError());
}

// =================================================================
// SLANeXt encoder-split: 488 letterbox, RGB order, pad 0 in normalized space
// =================================================================
__global__ __launch_bounds__(256)
void slanext_pre_rgb_kernel(const unsigned char *__restrict__ src, int src_step,
                            int rect_x, int rect_y, int rect_w, int rect_h,
                            float *__restrict__ dst_chw, int dst, int new_w,
                            int new_h, float scale_x, float scale_y) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;
  int idx = dy * dst + dx;
  int plane = dst * dst;
  if (dx < new_w && dy < new_h) {
    float sx_rel = (dx + 0.5f) * scale_x - 0.5f;
    float sy_rel = (dy + 0.5f) * scale_y - 0.5f;
    float b, g, r;
    // Source is BGR; per-channel means are passed swapped so `b` comes back as
    // (Blue-0.406)/0.225 etc., then R,G,B are scattered to the planes.
    tk_bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                            sx_rel, sy_rel, 0.406f, 0.456f, 0.485f,
                            1.0f / 0.225f, 1.0f / 0.224f, 1.0f / 0.229f,
                            1.0f / 255.0f, b, g, r);
    dst_chw[0 * plane + idx] = r; // R
    dst_chw[1 * plane + idx] = g; // G
    dst_chw[2 * plane + idx] = b; // B
  } else {
    dst_chw[0 * plane + idx] = 0.0f; // PaddingTableImage pads normalized 0
    dst_chw[1 * plane + idx] = 0.0f;
    dst_chw[2 * plane + idx] = 0.0f;
  }
}

void hip_fused_slanext_pre_rgb(const HipImage &src, int rect_x, int rect_y,
                               int rect_w, int rect_h, float *dst_chw,
                               hipStream_t stream) {
  constexpr int DST = 488;
  LetterboxLong lb = resize_by_long(rect_w, rect_h, DST);

  slanext_pre_rgb_kernel<<<pre_grid(DST, DST), pre_block(), 0, stream>>>(
      (const unsigned char *)src.data, (int)src.step, rect_x, rect_y, rect_w,
      rect_h, dst_chw, DST, lb.new_w, lb.new_h, lb.scale_x, lb.scale_y);
  HIP_CHECK(hipGetLastError());
}

// =================================================================
// Sub-rect layout preprocess (cell-det): resize sub-rect, /255, BGR CHW
// =================================================================
__global__ __launch_bounds__(256)
void layout_subrect_pre_kernel(const unsigned char *__restrict__ src,
                               int src_step, int rect_x, int rect_y, int rect_w,
                               int rect_h, float *__restrict__ dst_chw,
                               int dst_h, int dst_w, float scale_x,
                               float scale_y, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h) return;

  float sx_rel = (dx + 0.5f) * scale_x - 0.5f;
  float sy_rel = (dy + 0.5f) * scale_y - 0.5f;
  float b, g, r;
  tk_bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h, sx_rel,
                          sy_rel, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, inv_255,
                          b, g, r);
  int idx = dy * dst_w + dx;
  int plane = dst_h * dst_w;
  dst_chw[0 * plane + idx] = b;
  dst_chw[1 * plane + idx] = g;
  dst_chw[2 * plane + idx] = r;
}

void hip_fused_resize_normalize_layout_subrect(const HipImage &src, int rect_x,
                                               int rect_y, int rect_w,
                                               int rect_h, float *dst_chw,
                                               int dst_w, int dst_h,
                                               hipStream_t stream) {
  float scale_x = (float)rect_w / (float)dst_w;
  float scale_y = (float)rect_h / (float)dst_h;
  layout_subrect_pre_kernel<<<pre_grid(dst_w, dst_h), pre_block(), 0, stream>>>(
      (const unsigned char *)src.data, (int)src.step, rect_x, rect_y, rect_w,
      rect_h, dst_chw, dst_h, dst_w, scale_x, scale_y, 1.0f / 255.0f);
  HIP_CHECK(hipGetLastError());
}

} // namespace turbo_ocr::amd::kernels
