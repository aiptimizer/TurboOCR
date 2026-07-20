#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace turbo_ocr::kernels {

// __ldg routes reads through the read-only / texture cache. uchar3 has no
// __ldg overload, so load the 3 channels as individual bytes.
__device__ __forceinline__ uchar3 ldg_uchar3(const unsigned char *base,
                                             int px) {
  const unsigned char *p = base + px * 3;
  return make_uchar3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
}

// OpenCV INTER_LINEAR sample of a BGR u8 image at dst pixel (dx, dy) under
// (scale_x, scale_y). Returns interpolated BGR floats. Shared by the single
// and batched fused resize+normalize kernels (identical math in both).
__device__ __forceinline__ float3 bilinear_sample_bgr(
    const unsigned char *src, int src_h, int src_w, int src_step,
    float scale_x, float scale_y, int dx, int dy) {
  float sx = (dx + 0.5f) * scale_x - 0.5f;
  float sy = (dy + 0.5f) * scale_y - 0.5f;

  int x0 = (int)floorf(sx);
  int y0 = (int)floorf(sy);
  float fx = sx - x0;
  float fy = sy - y0;

  int x1 = x0 + 1;
  int y1 = y0 + 1;
  x0 = max(0, min(x0, src_w - 1));
  x1 = max(0, min(x1, src_w - 1));
  y0 = max(0, min(y0, src_h - 1));
  y1 = max(0, min(y1, src_h - 1));

  const unsigned char *row0 = src + y0 * src_step;
  const unsigned char *row1 = src + y1 * src_step;

  uchar3 p00 = ldg_uchar3(row0, x0), p10 = ldg_uchar3(row0, x1);
  uchar3 p01 = ldg_uchar3(row1, x0), p11 = ldg_uchar3(row1, x1);

  float w00 = (1.0f - fx) * (1.0f - fy);
  float w10 = fx * (1.0f - fy);
  float w01 = (1.0f - fx) * fy;
  float w11 = fx * fy;

  // Interpolate BGR and keep BGR order (PaddleOCR uses img_mode: BGR)
  float3 bgr;
  bgr.x = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  bgr.y = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  bgr.z = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;
  return bgr;
}

// --- Fused Resize + Normalize + CHW for Detection ---

__global__ __launch_bounds__(256)
void fused_resize_normalize_chw_kernel(
    const uchar3 * __restrict__ src, int src_h, int src_w, int src_step,
    float * __restrict__ dst_chw, int dst_h, int dst_w,
    float scale_x, float scale_y,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx >= dst_w || dy >= dst_h)
    return;

  float3 bgr = bilinear_sample_bgr((const unsigned char *)src, src_h, src_w,
                                   src_step, scale_x, scale_y, dx, dy);

  int idx = dy * dst_w + dx;
  int plane = dst_h * dst_w;
  dst_chw[0 * plane + idx] = (bgr.x * inv_255 - mean0) * inv_std0;
  dst_chw[1 * plane + idx] = (bgr.y * inv_255 - mean1) * inv_std1;
  dst_chw[2 * plane + idx] = (bgr.z * inv_255 - mean2) * inv_std2;
}

void cuda_fused_resize_normalize_det(const GpuImage &src, float *dst_chw,
                                      int dst_w, int dst_h,
                                      cudaStream_t stream) {
  float scale_x = (float)src.cols / dst_w;
  float scale_y = (float)src.rows / dst_h;

  dim3 block(32, 8);  // 256 threads for better occupancy on SM 120
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  // Precompute inv_std on host (multiplication is 5-10x faster than division on GPU)
  fused_resize_normalize_chw_kernel<<<grid, block, 0, stream>>>(
      (const uchar3 *)src.data, src.rows, src.cols, (int)src.step,
      dst_chw, dst_h, dst_w, scale_x, scale_y,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

// --- Fused Resize + Normalize for PP-DocLayoutV3 ---
// Same kernel as det, but with mean=[0,0,0] and std=[1,1,1]. The model's
// inference.yml specifies `NormalizeImage {mean: [0,0,0], std: [1,1,1],
// norm_type: none}` + `Permute`, which is exactly `pixel / 255` in CHW order.
// Input channel order follows the det fix on develop: BGR (matching OpenCV
// cv::Mat default). The smoke test confirmed correct predictions with this
// ordering.
void cuda_fused_resize_normalize_layout(const GpuImage &src, float *dst_chw,
                                         int dst_w, int dst_h,
                                         cudaStream_t stream) {
  float scale_x = (float)src.cols / dst_w;
  float scale_y = (float)src.rows / dst_h;

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  fused_resize_normalize_chw_kernel<<<grid, block, 0, stream>>>(
      (const uchar3 *)src.data, src.rows, src.cols, (int)src.step,
      dst_chw, dst_h, dst_w, scale_x, scale_y,
      0.0f, 0.0f, 0.0f,       // mean
      1.0f, 1.0f, 1.0f,       // inv_std (== 1)
      1.0f / 255.0f);         // inv_255 → pixel/255
  CUDA_CHECK(cudaGetLastError());
}

// --- Batched Fused Resize + Normalize + CHW for Detection ---
// Each image in the batch may have different source dimensions but all map
// to the same dst_h x dst_w output plane. The batch index is blockIdx.z.

__global__ __launch_bounds__(256)
void batch_fused_resize_normalize_chw_kernel(
    const void *const * __restrict__ src_ptrs,
    const int * __restrict__ src_steps,
    const int * __restrict__ src_heights,
    const int * __restrict__ src_widths,
    const int * __restrict__ dst_heights,
    const int * __restrict__ dst_widths,
    float * __restrict__ dst_chw, int dst_h, int dst_w,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;

  if (dx >= dst_w || dy >= dst_h)
    return;

  // Per-image source info
  const unsigned char *src = (const unsigned char *)src_ptrs[b];
  int src_h = src_heights[b];
  int src_w = src_widths[b];
  int src_step = src_steps[b];

  int plane = dst_h * dst_w;
  int batch_offset = b * 3 * plane;
  int idx = dy * dst_w + dx;

  // Letterbox: this image occupies only its own aspect-preserving
  // (content_h, content_w) in the canvas top-left. Outside it, write a
  // normalized black pixel so the DB head sees "no text", never stretched
  // glyphs from a foreign aspect ratio.
  int content_h = dst_heights[b];
  int content_w = dst_widths[b];
  if (dx >= content_w || dy >= content_h) {
    dst_chw[batch_offset + 0 * plane + idx] = (0.0f - mean0) * inv_std0;
    dst_chw[batch_offset + 1 * plane + idx] = (0.0f - mean1) * inv_std1;
    dst_chw[batch_offset + 2 * plane + idx] = (0.0f - mean2) * inv_std2;
    return;
  }

  float scale_x = (float)src_w / content_w;
  float scale_y = (float)src_h / content_h;

  float3 bgr = bilinear_sample_bgr(src, src_h, src_w, src_step,
                                   scale_x, scale_y, dx, dy);

  dst_chw[batch_offset + 0 * plane + idx] = (bgr.x * inv_255 - mean0) * inv_std0;
  dst_chw[batch_offset + 1 * plane + idx] = (bgr.y * inv_255 - mean1) * inv_std1;
  dst_chw[batch_offset + 2 * plane + idx] = (bgr.z * inv_255 - mean2) * inv_std2;
}

void cuda_batch_fused_resize_normalize_det(
    const void *const *d_src_ptrs, const int *d_src_steps,
    const int *d_src_heights, const int *d_src_widths,
    const int *d_dst_heights, const int *d_dst_widths,
    float *dst_chw, int dst_w, int dst_h, int batch_size,
    cudaStream_t stream) {
  if (batch_size == 0)
    return;

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y,
            batch_size);

  batch_fused_resize_normalize_chw_kernel<<<grid, block, 0, stream>>>(
      d_src_ptrs, d_src_steps, d_src_heights, d_src_widths,
      d_dst_heights, d_dst_widths,
      dst_chw, dst_h, dst_w,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f);
  CUDA_CHECK(cudaGetLastError());
}

// --- Fused Batched ROI Warp + Normalize ---
// block(32,8): 32 threads along x map to contiguous dst memory, so the CHW
// plane WRITES are fully coalesced. The source READS are not — the perspective
// map plus the 3-byte pixel stride makes them gathers by nature; they are
// served through __ldg / the read-only cache, whose 2D spatial locality (the
// warp's samples cluster in a small source region) is what actually absorbs
// the scatter, not coalescing.

__global__ __launch_bounds__(256)
void batch_roi_warp_kernel(
    const uchar3 * __restrict__ src_data, int src_h, int src_w, int src_step,
    const float * __restrict__ M_invs,    // [batch_size * 9] inverse perspective matrices
    const int * __restrict__ crop_widths, // [batch_size] per-crop actual width
    float * __restrict__ dst_batch,       // [batch_size, 3, dst_h, dst_w]
    int dst_h, int dst_w,
    float mean0, float mean1, float mean2,       // per-channel (RGB) mean
    float inv_std0, float inv_std1, float inv_std2, // per-channel 1/std
    float inv_scale) {                            // pixel scale (1/255)
  // Load M_invs and crop_width into shared memory (read once, used by all 256 threads)
  __shared__ float s_M[9];
  __shared__ int s_crop_w;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int b = blockIdx.z;
  if (tid < 9)
    s_M[tid] = M_invs[b * 9 + tid];
  if (tid == 0)
    s_crop_w = crop_widths[b];
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= dst_w || y >= dst_h)
    return;

  int plane_size = dst_h * dst_w;
  int batch_offset = b * 3 * plane_size;
  int pixel_idx = y * dst_w + x;

  // Zero-pad pixels beyond per-crop actual width
  if (x >= s_crop_w) {
    dst_batch[batch_offset + 0 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 1 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 2 * plane_size + pixel_idx] = 0.0f;
    return;
  }

  // Perspective transform using shared memory M. Clamp the projective
  // denominator away from zero while PRESERVING its sign — a bare `+1e-7`
  // would flip a small negative denom positive, mirroring the sampled pixel
  // (dead for the affine rec/cls transforms where denom==1, but correct now
  // for any genuine projective matrix).
  float denom = s_M[6] * x + s_M[7] * y + s_M[8];
  denom = copysignf(fmaxf(fabsf(denom), 1e-7f), denom);
  float inv_denom = 1.0f / denom;  // division -> multiplication
  float src_x = (s_M[0] * x + s_M[1] * y + s_M[2]) * inv_denom;
  float src_y = (s_M[3] * x + s_M[4] * y + s_M[5]) * inv_denom;

  // Guard against NaN/Inf
  if (!isfinite(src_x) || !isfinite(src_y)) {
    dst_batch[batch_offset + 0 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 1 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 2 * plane_size + pixel_idx] = 0.0f;
    return;
  }

  // Clamp source coordinates
  src_x = fminf(fmaxf(src_x, -1.0f), (float)(src_w + 1));
  src_y = fminf(fmaxf(src_y, -1.0f), (float)(src_h + 1));

  // Bilinear sampling with border replicate
  int x_low = (int)floorf(src_x);
  int y_low = (int)floorf(src_y);
  int x_high = x_low + 1;
  int y_high = y_low + 1;

  float dx = src_x - x_low;
  float dy = src_y - y_low;

  // Border replicate: clamp to [0, dim-1]
  x_low  = min(max(x_low,  0), src_w - 1);
  x_high = min(max(x_high, 0), src_w - 1);
  y_low  = min(max(y_low,  0), src_h - 1);
  y_high = min(max(y_high, 0), src_h - 1);

  // Use __ldg() to route reads through the read-only / texture cache.
  // The perspective transform creates a 2D random access pattern that
  // benefits significantly from the texture cache's 2D spatial locality.
  // uchar3 has no __ldg overload, so load individual bytes.
  auto get_pixel = [&](int px, int py) -> float3 {
    const unsigned char *row =
        (const unsigned char *)src_data + py * src_step;
    const unsigned char *p = row + px * 3;
    return make_float3((float)__ldg(p + 2), (float)__ldg(p + 1), (float)__ldg(p)); // RGB from BGR
  };

  float3 p00 = get_pixel(x_low, y_low);
  float3 p10 = get_pixel(x_high, y_low);
  float3 p01 = get_pixel(x_low, y_high);
  float3 p11 = get_pixel(x_high, y_high);

  // Precompute bilinear weights (reused for all 3 channels)
  float w00 = (1.0f - dx) * (1.0f - dy);
  float w10 = dx * (1.0f - dy);
  float w01 = (1.0f - dx) * dy;
  float w11 = dx * dy;

  float3 res;
  res.x = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  res.y = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  res.z = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;

  // Per-channel normalize: (pixel*inv_scale - mean) * inv_std. res is RGB.
  // Rec passes mean=0.5,inv_std=2,inv_scale=1/255 (== the old pixel/127.5-1);
  // cls passes ImageNet mean/std so PP-LCNet textline-ori gets in-distribution input.
  dst_batch[batch_offset + 0 * plane_size + pixel_idx] = (res.x * inv_scale - mean0) * inv_std0;
  dst_batch[batch_offset + 1 * plane_size + pixel_idx] = (res.y * inv_scale - mean1) * inv_std1;
  dst_batch[batch_offset + 2 * plane_size + pixel_idx] = (res.z * inv_scale - mean2) * inv_std2;
}

void cuda_batch_roi_warp(const GpuImage &src, const float *d_M_invs,
                         const int *d_crop_widths, float *d_dst_batch,
                         int batch_size, int dst_h, int dst_w,
                         cudaStream_t stream,
                         float mean0, float mean1, float mean2,
                         float inv_std0, float inv_std1, float inv_std2,
                         float inv_scale) {
  if (batch_size == 0)
    return;

  // (32,8) block: 32 threads along x for coalesced writes to dst (row-major),
  // 8 along y to keep occupancy. 256 threads total.
  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y,
            batch_size);

  batch_roi_warp_kernel<<<grid, block, 0, stream>>>(
      (const uchar3 *)src.data, src.rows, src.cols, (int)src.step, d_M_invs,
      d_crop_widths, d_dst_batch, dst_h, dst_w,
      mean0, mean1, mean2, inv_std0, inv_std1, inv_std2, inv_scale);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace turbo_ocr::kernels
