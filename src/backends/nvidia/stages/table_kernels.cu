// NOTE — CUDA device code outside src/backends/nvidia/. This is deliberate and
// documented: see src/backends/nvidia/kernels_cuda/README.md for why the three CUDA sites in this
// tree have not been moved into the NVIDIA backend yet, and what gates it.
//
// Fused preprocess kernels for the table-detection / layout pipeline.
// Reads sub-rects of a single page GpuImage (BGR uint8), produces normalized
// CHW float tensors directly into TRT input buffers — no intermediate copies,
// no host round-trip. Every stage (resize, normalize, letterbox pad, channel
// swap, CHW scatter) is fused into one pass over the output tensor.
//
// Kernels (all declared in nvidia/kernels_cuda/kernels_cuda.h):
//   - cuda_fused_table_cls_pre     : resize-short(256) + center-crop(224)
//                                    + ImageNet normalize + BGR CHW
//   - cuda_fused_slanext_pre       : ResizeByLong(488) preserve AR + ImageNet
//                                    normalize + bottom-right pad + BGR CHW
//   - cuda_fused_slanext_pre_rgb   : ResizeByLong(488) + ImageNet normalize +
//                                    pad 0 (normalized space) + RGB CHW
//   - cuda_fused_resize_normalize_layout (sub-rect) : resize sub-rect, /255, CHW
//
// -------------------------------------------------------------------------
// NUMERICAL PARITY (load-bearing — these tensors ARE the model input):
//
// Resampling is cv2 INTER_LINEAR on uint8, evaluated in fp32. This matches the
// PaddleX export path (cv2.resize INTER_LINEAR + fp32 Normalize) bit-for-bit up
// to fp32 rounding, and is identical to the sibling det/layout kernels.
//
//   * Coordinate map: half-pixel-center, src = (dst + 0.5)*scale - 0.5.
//     (NOT align-corners; align-corners would shift every sample and regress
//     the models, which were exported against cv2's half-pixel convention.)
//
//   * Border handling: we clamp the integer sample indices (x0,x1,y0,y1) to
//     the valid range and DO NOT zero the fractional weights. This is provably
//     bit-identical to cv2's border rule (which instead sets frac=0 at the
//     edge): whenever a fractional weight would fall outside the image, the two
//     horizontal (or vertical) taps collapse onto the SAME clamped pixel, so
//     their weights sum to 1 on that pixel and the fractional value cancels.
//     x1 = x0+1 only clamps when x0 == W-1 (then x1->W-1 == x0); x0 only clamps
//     when x0 < 0 (then x1 = x0+1 <= 0 clamps to 0 == x0). Either way both taps
//     address one pixel => result == that pixel == cv2's frac=0 result. QED.
//     This is why the simple index-clamp is exact, not an approximation.
//
//   * Arithmetic: the 4-weight form shares the bilinear weights across all 3
//     channels (weights computed once), and the compiler contracts each
//     `w*p + w*p` into hardware FFMA (fmad on by default) — so the emitted code
//     is already fused-multiply-add and instruction-minimal. Rewriting it as a
//     nested-lerp would change the rounding order and shift the model input, so
//     the weight form is kept verbatim for parity.
//
// HARDWARE TEXTURE BILINEAR is deliberately NOT used: tex2D linear filtering
// quantizes the interpolation weights to 1/256 fixed-point and uses its own
// coordinate convention — it is NOT bit-equal to fp32 bilinear and would shift
// the input tensor. The read-only data cache (__ldg) already gives us the
// texture cache's bandwidth without the numerical drift.
//
// MEMORY: BGR pixels are 3 bytes and therefore not vector-aligned, so a
// reinterpret to uchar4/uint32 for a wide load would be a misaligned vector
// access (undefined / faulting on CUDA). We instead issue three per-byte __ldg
// loads per pixel through the read-only cache; neighbouring output threads read
// overlapping source pixels, so the spatial reuse is served from L1/tex cache.
// Output writes are contiguous within each NCHW plane and the 32-wide thread-x
// dimension keeps every plane store fully coalesced.
//
// Channel order: BGR source throughout (PaddleOCR img_mode). Pad value for the
// BGR SLANeXt path is (0 - mean)/std per PaddleX export; the RGB split path pads
// with 0 in normalized space (PaddingTableImage).

#include "nvidia/kernels_cuda/kernels_cuda.h"
#include "nvidia/support/cuda_check.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace turbo_ocr::kernels {

// One BGR pixel via the read-only cache. Per-byte because a 3-byte pixel is not
// vector-aligned (see file header: wide vector loads would be misaligned/UB).
__device__ __forceinline__ uchar3 ldg_bgr(const unsigned char* __restrict__ row,
                                          int px) {
  const unsigned char* p = row + px * 3;
  return make_uchar3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
}

// Bilinear sample within a sub-rect of a BGR uint8 image, normalize per channel.
// (sx_rel, sy_rel) are source-pixel coords RELATIVE to the sub-rect top-left,
// already in the half-pixel-center frame. cv2 INTER_LINEAR with index-clamp
// border handling (bit-identical to cv2's frac-zeroing — proof in file header).
__device__ __forceinline__ void bilinear_sample_norm(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float sx_rel, float sy_rel,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255,
    float& b_out, float& g_out, float& r_out) {
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

  const unsigned char* row0 = src + (rect_y + y0) * src_step;
  const unsigned char* row1 = src + (rect_y + y1) * src_step;
  int ax0 = rect_x + x0;
  int ax1 = rect_x + x1;

  uchar3 p00 = ldg_bgr(row0, ax0), p10 = ldg_bgr(row0, ax1);
  uchar3 p01 = ldg_bgr(row1, ax0), p11 = ldg_bgr(row1, ax1);

  // Weights computed once, shared across B/G/R; each dot product contracts to
  // FFMA. Order preserved verbatim for bit-exact parity with the export.
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
struct LetterboxLong {
  int new_w, new_h;
  // cv2.resize samples PER AXIS from the ROUNDED output dim: scale = src/new.
  // On the long axis new==dst so this equals 1/s, but the short axis rounds
  // (new_short = round(short*s) != short*s), so a single 1/s drifts up to ~1px
  // on the short axis. Match cv2 (and the layout sub-rect kernel) exactly.
  float scale_x, scale_y;  // src px per resized px, per axis
};
static inline LetterboxLong resize_by_long(int rect_w, int rect_h, int dst) {
  float s = (float)dst / (float)::max(rect_w, rect_h);
  int new_w = (int)lroundf((float)rect_w * s);
  int new_h = (int)lroundf((float)rect_h * s);
  new_w = ::max(1, ::min(new_w, dst));
  new_h = ::max(1, ::min(new_h, dst));
  return {new_w, new_h, (float)rect_w / (float)new_w, (float)rect_h / (float)new_h};
}

// Launch: 32-wide x (one warp -> 32 contiguous output columns => coalesced
// plane stores), 8 rows -> 256 threads/block, matching the det/layout kernels.
static inline dim3 pre_block() { return dim3(32, 8); }
static inline dim3 pre_grid(int w, int h) {
  dim3 b = pre_block();
  return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y);
}

// =================================================================
// TableCls: resize-short(256) + center-crop(224) + ImageNet + BGR CHW
// =================================================================
// Fused crop: crop offset is folded into the source coordinate so the resize
// and the center-crop are one sampling pass (mirrors PaddleX ResizeImage(256)
// + CropImage(224)). Per dst pixel:
//   sx_rel = (dx + 0.5)*scale - 0.5 + crop_off_x_src,  scale = min(w,h)/256
__global__ __launch_bounds__(256)
void table_cls_pre_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst,
    float scale,
    float crop_off_x_src, float crop_off_y_src,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;

  float sx_rel = (dx + 0.5f) * scale - 0.5f + crop_off_x_src;
  float sy_rel = (dy + 0.5f) * scale - 0.5f + crop_off_y_src;

  float b, g, r;
  bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                       sx_rel, sy_rel,
                       mean0, mean1, mean2,
                       inv_std0, inv_std1, inv_std2, inv_255,
                       b, g, r);

  int idx = dy * dst + dx;
  int plane = dst * dst;
  dst_chw[0 * plane + idx] = b;
  dst_chw[1 * plane + idx] = g;
  dst_chw[2 * plane + idx] = r;
}

void cuda_fused_table_cls_pre(const GpuImage& src,
                              int rect_x, int rect_y, int rect_w, int rect_h,
                              float* dst_chw, cudaStream_t stream) {
  constexpr int DST = 224;
  constexpr int SHORT = 256;
  float scale = (float)::min(rect_w, rect_h) / (float)SHORT;  // src px / resized px
  float crop_off_x_src = ((float)rect_w - DST * scale) * 0.5f;
  float crop_off_y_src = ((float)rect_h - DST * scale) * 0.5f;

  table_cls_pre_kernel<<<pre_grid(DST, DST), pre_block(), 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, DST,
      scale, crop_off_x_src, crop_off_y_src,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

// =================================================================
// SLANeXt (BGR): ResizeByLong(488) + ImageNet + bottom-right pad + CHW
// =================================================================
// Inside the resized region -> sample+normalize; outside -> PAD = (0-mean)/std.
__global__ __launch_bounds__(256)
void slanext_pre_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst,
    int new_w, int new_h, float scale_x, float scale_y,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;

  int idx = dy * dst + dx;
  int plane = dst * dst;

  if (dx < new_w && dy < new_h) {
    float sx_rel = (dx + 0.5f) * scale_x - 0.5f;
    float sy_rel = (dy + 0.5f) * scale_y - 0.5f;
    float b, g, r;
    bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                         sx_rel, sy_rel,
                         mean0, mean1, mean2,
                         inv_std0, inv_std1, inv_std2, inv_255,
                         b, g, r);
    dst_chw[0 * plane + idx] = b;
    dst_chw[1 * plane + idx] = g;
    dst_chw[2 * plane + idx] = r;
  } else {
    dst_chw[0 * plane + idx] = (-mean0) * inv_std0;
    dst_chw[1 * plane + idx] = (-mean1) * inv_std1;
    dst_chw[2 * plane + idx] = (-mean2) * inv_std2;
  }
}

void cuda_fused_slanext_pre(const GpuImage& src,
                            int rect_x, int rect_y, int rect_w, int rect_h,
                            float* dst_chw, cudaStream_t stream) {
  constexpr int DST = 488;
  LetterboxLong lb = resize_by_long(rect_w, rect_h, DST);

  slanext_pre_kernel<<<pre_grid(DST, DST), pre_block(), 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, DST, lb.new_w, lb.new_h, lb.scale_x, lb.scale_y,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

// SLANeXt encoder-split preprocess: same 488 letterbox but RGB channel order
// (PaddleOCR DecodeImage img_mode=RGB for table) + ImageNet norm + pad with 0
// in normalized space (PaddingTableImage pads the normalized image with 0).
// Source is BGR; we pass per-channel means so bilinear_sample_norm returns
// b=(Blue-0.406)/0.225, g=(Green-0.456)/0.224, r=(Red-0.485)/0.229, then scatter
// R,G,B to the planes.
__global__ __launch_bounds__(256)
void slanext_pre_rgb_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst,
    int new_w, int new_h, float scale_x, float scale_y) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst || dy >= dst) return;
  int idx = dy * dst + dx;
  int plane = dst * dst;
  if (dx < new_w && dy < new_h) {
    float sx_rel = (dx + 0.5f) * scale_x - 0.5f;
    float sy_rel = (dy + 0.5f) * scale_y - 0.5f;
    float b, g, r;
    bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                         sx_rel, sy_rel,
                         0.406f, 0.456f, 0.485f,
                         1.0f / 0.225f, 1.0f / 0.224f, 1.0f / 0.229f,
                         1.0f / 255.0f, b, g, r);
    dst_chw[0 * plane + idx] = r;  // R
    dst_chw[1 * plane + idx] = g;  // G
    dst_chw[2 * plane + idx] = b;  // B
  } else {
    dst_chw[0 * plane + idx] = 0.0f;  // PaddingTableImage pads normalized 0
    dst_chw[1 * plane + idx] = 0.0f;
    dst_chw[2 * plane + idx] = 0.0f;
  }
}

void cuda_fused_slanext_pre_rgb(const GpuImage& src,
                                int rect_x, int rect_y, int rect_w, int rect_h,
                                float* dst_chw, cudaStream_t stream) {
  constexpr int DST = 488;
  LetterboxLong lb = resize_by_long(rect_w, rect_h, DST);

  slanext_pre_rgb_kernel<<<pre_grid(DST, DST), pre_block(), 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, DST, lb.new_w, lb.new_h, lb.scale_x, lb.scale_y);
  CUDA_CHECK(cudaGetLastError());
}

// =================================================================
// Sub-rect overload of layout preprocess (used by cell-det).
// resize sub-rect to dst_w × dst_h, /255, mean=0, std=1, BGR CHW.
// =================================================================
__global__ __launch_bounds__(256)
void layout_subrect_pre_kernel(
    const unsigned char* __restrict__ src, int src_step,
    int rect_x, int rect_y, int rect_w, int rect_h,
    float* __restrict__ dst_chw, int dst_h, int dst_w,
    float scale_x, float scale_y, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h) return;

  float sx_rel = (dx + 0.5f) * scale_x - 0.5f;
  float sy_rel = (dy + 0.5f) * scale_y - 0.5f;
  float b, g, r;
  bilinear_sample_norm(src, src_step, rect_x, rect_y, rect_w, rect_h,
                       sx_rel, sy_rel,
                       0.0f, 0.0f, 0.0f,
                       1.0f, 1.0f, 1.0f, inv_255,
                       b, g, r);
  int idx = dy * dst_w + dx;
  int plane = dst_h * dst_w;
  dst_chw[0 * plane + idx] = b;
  dst_chw[1 * plane + idx] = g;
  dst_chw[2 * plane + idx] = r;
}

void cuda_fused_resize_normalize_layout(const GpuImage& src,
                                        int rect_x, int rect_y,
                                        int rect_w, int rect_h,
                                        float* dst_chw, int dst_w, int dst_h,
                                        cudaStream_t stream) {
  float scale_x = (float)rect_w / (float)dst_w;
  float scale_y = (float)rect_h / (float)dst_h;
  layout_subrect_pre_kernel<<<pre_grid(dst_w, dst_h), pre_block(), 0, stream>>>(
      (const unsigned char*)src.data, (int)src.step,
      rect_x, rect_y, rect_w, rect_h,
      dst_chw, dst_h, dst_w, scale_x, scale_y, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace turbo_ocr::kernels
