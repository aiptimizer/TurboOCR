// preprocess_kernels.hip — HIP port of src/backends/nvidia/kernels_cuda/preprocess_kernels.cu.
//
// Kernel bodies are IDENTICAL to the CUDA source (bit-for-bit math): fused
// resize+normalize+CHW (det/layout), batched letterboxed det resize, and the
// batched perspective ROI warp that is the residency win. Only the device
// runtime surface changed:
//   <cuda_runtime.h>            -> <hip/hip_runtime.h>
//   cudaStream_t                -> hipStream_t
//   cudaGetLastError            -> hipGetLastError
//   CUDA_CHECK                  -> HIP_CHECK
//   uchar3/float3/make_*        -> HIP provides the same vector types + ctors
//   __ldg / floorf / fminf ...  -> identical device math intrinsics in HIP
//
// WARP-SIZE: none of these kernels use warp/wavefront primitives or shared-memory
// reductions, so the 32-vs-64 lane difference is irrelevant here — they are pure
// per-output-pixel maps and port unchanged to CDNA/RDNA.

#include "kernels_hip.h"
#include "amd/support/hip_check.h"
#include <hip/hip_runtime.h>

namespace turbo_ocr::amd::kernels {

__device__ __forceinline__ uchar3 ldg_uchar3(const unsigned char *base, int px) {
  const unsigned char *p = base + px * 3;
  return make_uchar3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
}

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

  float3 bgr;
  bgr.x = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  bgr.y = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  bgr.z = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;
  return bgr;
}

__global__ __launch_bounds__(256)
void fused_resize_normalize_chw_kernel(
    const uchar3 *__restrict__ src, int src_h, int src_w, int src_step,
    float *__restrict__ dst_chw, int dst_h, int dst_w,
    float scale_x, float scale_y,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255,
    int rgb_out) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h)
    return;

  float3 bgr = bilinear_sample_bgr((const unsigned char *)src, src_h, src_w,
                                   src_step, scale_x, scale_y, dx, dy);

  // NormParams::order selects which SOURCE component feeds output plane 0.
  // mean/inv_std are POSITIONAL (indexed by output plane), so only the source
  // components swap. rgb_out == 0 keeps the source BGR order (the det/layout
  // convention); rgb_out == 1 emits R,G,B.
  float c0 = rgb_out ? bgr.z : bgr.x;
  float c2 = rgb_out ? bgr.x : bgr.z;

  int idx = dy * dst_w + dx;
  int plane = dst_h * dst_w;
  dst_chw[0 * plane + idx] = (c0 * inv_255 - mean0) * inv_std0;
  dst_chw[1 * plane + idx] = (bgr.y * inv_255 - mean1) * inv_std1;
  dst_chw[2 * plane + idx] = (c2 * inv_255 - mean2) * inv_std2;
}

void hip_fused_resize_normalize_det(const HipImage &src, float *dst_chw,
                                    int dst_w, int dst_h, hipStream_t stream) {
  float scale_x = (float)src.cols / dst_w;
  float scale_y = (float)src.rows / dst_h;

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  fused_resize_normalize_chw_kernel<<<grid, block, 0, stream>>>(
      (const uchar3 *)src.data, src.rows, src.cols, (int)src.step,
      dst_chw, dst_h, dst_w, scale_x, scale_y,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f,
      /*rgb_out=*/0); // BGR-positional — the shared det convention
  HIP_CHECK(hipGetLastError());
}

void hip_fused_resize_normalize_layout(const HipImage &src, float *dst_chw,
                                       int dst_w, int dst_h, hipStream_t stream) {
  float scale_x = (float)src.cols / dst_w;
  float scale_y = (float)src.rows / dst_h;

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  fused_resize_normalize_chw_kernel<<<grid, block, 0, stream>>>(
      (const uchar3 *)src.data, src.rows, src.cols, (int)src.step,
      dst_chw, dst_h, dst_w, scale_x, scale_y,
      0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f / 255.0f,
      /*rgb_out=*/0); // BGR-positional — the shared layout convention
  HIP_CHECK(hipGetLastError());
}

// Param-driven full-frame variant. Same kernel, same arithmetic — the det and
// layout wrappers above are just this one with their constants baked in, and
// calling it with the det constants is bit-identical to hip_fused_resize_
// normalize_det. It exists so the IKernels wrapper can honour an ARBITRARY
// NormParams instead of sniffing which of two baked variants the caller meant
// (the CUDA adapter still has to sniff — see the ON-HARDWARE TODO in
// src/backends/nvidia/kernels_cuda/cuda_kernels.cpp::resize_normalize; that heuristic silently
// misroutes any third normalization). `order` selects the channel order of the
// OUTPUT planes: BGR keeps the source order, RGB swaps plane 0 and 2 by
// swapping the mean/std triples and the store order in the caller.
void hip_fused_resize_normalize(const HipImage &src, float *dst_chw, int dst_w,
                                int dst_h, float mean0, float mean1, float mean2,
                                float inv_std0, float inv_std1, float inv_std2,
                                float inv_scale, int rgb_out,
                                hipStream_t stream) {
  float scale_x = (float)src.cols / dst_w;
  float scale_y = (float)src.rows / dst_h;

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

  fused_resize_normalize_chw_kernel<<<grid, block, 0, stream>>>(
      (const uchar3 *)src.data, src.rows, src.cols, (int)src.step, dst_chw,
      dst_h, dst_w, scale_x, scale_y, mean0, mean1, mean2, inv_std0, inv_std1,
      inv_std2, inv_scale, rgb_out);
  HIP_CHECK(hipGetLastError());
}

__global__ __launch_bounds__(256)
void batch_fused_resize_normalize_chw_kernel(
    const void *const *__restrict__ src_ptrs,
    const int *__restrict__ src_steps,
    const int *__restrict__ src_heights,
    const int *__restrict__ src_widths,
    const int *__restrict__ dst_heights,
    const int *__restrict__ dst_widths,
    float *__restrict__ dst_chw, int dst_h, int dst_w,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_255,
    int rgb_out) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (dx >= dst_w || dy >= dst_h)
    return;

  const unsigned char *src = (const unsigned char *)src_ptrs[b];
  int src_h = src_heights[b];
  int src_w = src_widths[b];
  int src_step = src_steps[b];

  int plane = dst_h * dst_w;
  int batch_offset = b * 3 * plane;
  int idx = dy * dst_w + dx;

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

  // Same positional-mean / source-swap rule as the single-image kernel.
  float c0 = rgb_out ? bgr.z : bgr.x;
  float c2 = rgb_out ? bgr.x : bgr.z;
  dst_chw[batch_offset + 0 * plane + idx] = (c0 * inv_255 - mean0) * inv_std0;
  dst_chw[batch_offset + 1 * plane + idx] = (bgr.y * inv_255 - mean1) * inv_std1;
  dst_chw[batch_offset + 2 * plane + idx] = (c2 * inv_255 - mean2) * inv_std2;
}

void hip_batch_fused_resize_normalize_det(
    const void *const *d_src_ptrs, const int *d_src_steps,
    const int *d_src_heights, const int *d_src_widths,
    const int *d_dst_heights, const int *d_dst_widths,
    float *dst_chw, int dst_w, int dst_h, int batch_size, hipStream_t stream) {
  if (batch_size == 0)
    return;

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y, batch_size);

  batch_fused_resize_normalize_chw_kernel<<<grid, block, 0, stream>>>(
      d_src_ptrs, d_src_steps, d_src_heights, d_src_widths,
      d_dst_heights, d_dst_widths, dst_chw, dst_h, dst_w,
      0.485f, 0.456f, 0.406f,
      1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f, 1.0f / 255.0f,
      /*rgb_out=*/0); // BGR-positional — the shared det convention
  HIP_CHECK(hipGetLastError());
}

__global__ __launch_bounds__(256)
void batch_roi_warp_kernel(
    const uchar3 *__restrict__ src_data, int src_h, int src_w, int src_step,
    const float *__restrict__ M_invs,
    const int *__restrict__ crop_widths,
    float *__restrict__ dst_batch, int dst_h, int dst_w,
    float mean0, float mean1, float mean2,
    float inv_std0, float inv_std1, float inv_std2, float inv_scale,
    int rgb_out) {
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

  if (x >= s_crop_w) {
    dst_batch[batch_offset + 0 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 1 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 2 * plane_size + pixel_idx] = 0.0f;
    return;
  }

  float denom = s_M[6] * x + s_M[7] * y + s_M[8];
  denom = copysignf(fmaxf(fabsf(denom), 1e-7f), denom);
  float inv_denom = 1.0f / denom;
  float src_x = (s_M[0] * x + s_M[1] * y + s_M[2]) * inv_denom;
  float src_y = (s_M[3] * x + s_M[4] * y + s_M[5]) * inv_denom;

  if (!isfinite(src_x) || !isfinite(src_y)) {
    dst_batch[batch_offset + 0 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 1 * plane_size + pixel_idx] = 0.0f;
    dst_batch[batch_offset + 2 * plane_size + pixel_idx] = 0.0f;
    return;
  }

  src_x = fminf(fmaxf(src_x, -1.0f), (float)(src_w + 1));
  src_y = fminf(fmaxf(src_y, -1.0f), (float)(src_h + 1));

  int x_low = (int)floorf(src_x);
  int y_low = (int)floorf(src_y);
  int x_high = x_low + 1;
  int y_high = y_low + 1;

  float dx = src_x - x_low;
  float dy = src_y - y_low;

  x_low = min(max(x_low, 0), src_w - 1);
  x_high = min(max(x_high, 0), src_w - 1);
  y_low = min(max(y_low, 0), src_h - 1);
  y_high = min(max(y_high, 0), src_h - 1);

  auto get_pixel = [&](int px, int py) -> float3 {
    const unsigned char *row = (const unsigned char *)src_data + py * src_step;
    const unsigned char *p = row + px * 3;
    return make_float3((float)__ldg(p + 2), (float)__ldg(p + 1),
                       (float)__ldg(p)); // RGB from BGR
  };

  float3 p00 = get_pixel(x_low, y_low);
  float3 p10 = get_pixel(x_high, y_low);
  float3 p01 = get_pixel(x_low, y_high);
  float3 p11 = get_pixel(x_high, y_high);

  float w00 = (1.0f - dx) * (1.0f - dy);
  float w10 = dx * (1.0f - dy);
  float w01 = (1.0f - dx) * dy;
  float w11 = dx * dy;

  float3 res;
  res.x = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  res.y = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  res.z = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;

  // `res` is sampled RGB (get_pixel reads B,G,R and returns R,G,B).
  // NormParams::order selects which SOURCE component feeds output plane 0;
  // mean/inv_std stay POSITIONAL. rgb_out == 1 keeps R,G,B (the rec/cls
  // convention); rgb_out == 0 emits B,G,R.
  float c0 = rgb_out ? res.x : res.z;
  float c2 = rgb_out ? res.z : res.x;
  dst_batch[batch_offset + 0 * plane_size + pixel_idx] = (c0 * inv_scale - mean0) * inv_std0;
  dst_batch[batch_offset + 1 * plane_size + pixel_idx] = (res.y * inv_scale - mean1) * inv_std1;
  dst_batch[batch_offset + 2 * plane_size + pixel_idx] = (c2 * inv_scale - mean2) * inv_std2;
}

void hip_batch_roi_warp(const HipImage &src, const float *d_M_invs,
                        const int *d_crop_widths, float *d_dst_batch,
                        int batch_size, int dst_h, int dst_w, hipStream_t stream,
                        float mean0, float mean1, float mean2,
                        float inv_std0, float inv_std1, float inv_std2,
                        float inv_scale, int rgb_out) {
  if (batch_size == 0)
    return;

  dim3 block(32, 8);
  dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y,
            batch_size);

  batch_roi_warp_kernel<<<grid, block, 0, stream>>>(
      (const uchar3 *)src.data, src.rows, src.cols, (int)src.step, d_M_invs,
      d_crop_widths, d_dst_batch, dst_h, dst_w,
      mean0, mean1, mean2, inv_std0, inv_std1, inv_std2, inv_scale, rgb_out);
  HIP_CHECK(hipGetLastError());
}

} // namespace turbo_ocr::amd::kernels
