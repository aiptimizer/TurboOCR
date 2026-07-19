#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/common/cuda_check.h"
#include <cfloat>
#include <climits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace turbo_ocr::kernels {

// Cache occupancy + SM count per (device, kernel, block_size). Uses
// cudaGetDevice() so multi-GPU hosts don't get sized for device 0.
template <typename Fn>
static int coop_grid_for(Fn kernel, int threads) {
  int dev = 0;
  cudaGetDevice(&dev);
  static thread_local int cached_dev = -1;
  static thread_local int cached_sms = 0;
  static thread_local const void *cached_fn = nullptr;
  static thread_local int cached_threads = 0;
  static thread_local int cached_per_sm = 0;
  if (dev != cached_dev) {
    cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount, dev);
    cached_dev = dev;
    cached_fn = nullptr;
  }
  if (cached_fn != (const void *)kernel || cached_threads != threads) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &cached_per_sm, kernel, threads, 0);
    cached_fn = (const void *)kernel;
    cached_threads = threads;
  }
  return cached_per_sm * cached_sms;
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

  // OpenCV INTER_LINEAR mapping
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

  // Use __ldg() to route reads through the read-only / texture cache.
  // uchar3 has no __ldg overload, so we load as uint via the byte offset
  // and extract the 3 channels manually.
  const unsigned char *row0 = (const unsigned char *)src + y0 * src_step;
  const unsigned char *row1 = (const unsigned char *)src + y1 * src_step;

  auto ldg_uchar3 = [](const unsigned char *base, int px) -> uchar3 {
    const unsigned char *p = base + px * 3;
    return make_uchar3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
  };

  uchar3 p00 = ldg_uchar3(row0, x0), p10 = ldg_uchar3(row0, x1);
  uchar3 p01 = ldg_uchar3(row1, x0), p11 = ldg_uchar3(row1, x1);

  float w00 = (1.0f - fx) * (1.0f - fy);
  float w10 = fx * (1.0f - fy);
  float w01 = (1.0f - fx) * fy;
  float w11 = fx * fy;

  // Interpolate BGR and keep BGR order (PaddleOCR uses img_mode: BGR)
  float b = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  float g = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  float r = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;

  int idx = dy * dst_w + dx;
  int plane = dst_h * dst_w;
  dst_chw[0 * plane + idx] = (b * inv_255 - mean0) * inv_std0;
  dst_chw[1 * plane + idx] = (g * inv_255 - mean1) * inv_std1;
  dst_chw[2 * plane + idx] = (r * inv_255 - mean2) * inv_std2;
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
  const uchar3 *src = (const uchar3 *)src_ptrs[b];
  int src_h = src_heights[b];
  int src_w = src_widths[b];
  int src_step = src_steps[b];

  // Letterbox: this image occupies only its own aspect-preserving
  // (content_h, content_w) in the canvas top-left. Outside it, write a
  // normalized black pixel so the DB head sees "no text", never stretched
  // glyphs from a foreign aspect ratio.
  int content_h = dst_heights[b];
  int content_w = dst_widths[b];
  if (dx >= content_w || dy >= content_h) {
    int plane = dst_h * dst_w;
    int batch_offset = b * 3 * plane;
    int idx = dy * dst_w + dx;
    dst_chw[batch_offset + 0 * plane + idx] = (0.0f - mean0) * inv_std0;
    dst_chw[batch_offset + 1 * plane + idx] = (0.0f - mean1) * inv_std1;
    dst_chw[batch_offset + 2 * plane + idx] = (0.0f - mean2) * inv_std2;
    return;
  }

  float scale_x = (float)src_w / content_w;
  float scale_y = (float)src_h / content_h;

  // OpenCV INTER_LINEAR mapping
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

  const unsigned char *row0 = (const unsigned char *)src + y0 * src_step;
  const unsigned char *row1 = (const unsigned char *)src + y1 * src_step;

  auto ldg_uchar3 = [](const unsigned char *base, int px) -> uchar3 {
    const unsigned char *p = base + px * 3;
    return make_uchar3(__ldg(p), __ldg(p + 1), __ldg(p + 2));
  };

  uchar3 p00 = ldg_uchar3(row0, x0), p10 = ldg_uchar3(row0, x1);
  uchar3 p01 = ldg_uchar3(row1, x0), p11 = ldg_uchar3(row1, x1);

  float w00 = (1.0f - fx) * (1.0f - fy);
  float w10 = fx * (1.0f - fy);
  float w01 = (1.0f - fx) * fy;
  float w11 = fx * fy;

  // Interpolate BGR and keep BGR order (PaddleOCR uses img_mode: BGR)
  float bb = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
  float g = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
  float r = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;

  int plane = dst_h * dst_w;
  int batch_offset = b * 3 * plane;
  int idx = dy * dst_w + dx;
  dst_chw[batch_offset + 0 * plane + idx] = (bb * inv_255 - mean0) * inv_std0;
  dst_chw[batch_offset + 1 * plane + idx] = (g * inv_255 - mean1) * inv_std1;
  dst_chw[batch_offset + 2 * plane + idx] = (r * inv_255 - mean2) * inv_std2;
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

  // Perspective transform using shared memory M
  float denom = s_M[6] * x + s_M[7] * y + s_M[8];
  denom += (fabsf(denom) < 1e-7f) ? 1e-7f : 0.0f;
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

// --- ArgMax ---

// Shared-memory parallel reduction argmax: one block per sequence position
// 256 threads cooperate to find max across the class dim (v6 width 18,710;
// tiny tier 6,906). Class-agnostic — the reduction loop strides num_classes.
__global__ __launch_bounds__(256)
void argmax_kernel(const float *input_probs, int *output_indices,
                              float *output_scores, int total_steps,
                              int num_classes) {
  int step_idx = blockIdx.x;
  if (step_idx >= total_steps)
    return;

  const float *probs_ptr = input_probs + step_idx * num_classes;
  int tid = threadIdx.x;
  int block_size = blockDim.x; // 256

  // Each thread finds local max over its stripe
  float local_max = -FLT_MAX;
  int local_idx = 0;
  for (int c = tid; c < num_classes; c += block_size) {
    float val = probs_ptr[c];
    if (val > local_max) {
      local_max = val;
      local_idx = c;
    }
  }

  // Shared memory reduction (strides > 32)
  __shared__ float s_vals[256];
  __shared__ int s_idxs[256];
  s_vals[tid] = local_max;
  s_idxs[tid] = local_idx;
  __syncthreads();

  for (int stride = block_size / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      if (s_vals[tid + stride] > s_vals[tid]) {
        s_vals[tid] = s_vals[tid + stride];
        s_idxs[tid] = s_idxs[tid + stride];
      }
    }
    __syncthreads();
  }

  // Warp-level reduction (strides <= 32, no syncthreads needed)
  if (tid < 32) {
    // Load from shared to registers
    float wval = s_vals[tid];
    int widx = s_idxs[tid];
    if (s_vals[tid + 32] > wval) { wval = s_vals[tid + 32]; widx = s_idxs[tid + 32]; }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_val = __shfl_down_sync(0xffffffff, wval, offset);
      int other_idx = __shfl_down_sync(0xffffffff, widx, offset);
      if (other_val > wval) { wval = other_val; widx = other_idx; }
    }

    if (tid == 0) {
      output_indices[step_idx] = widx;
      output_scores[step_idx] = wval;
    }
  }
}

void cuda_argmax(const float *input_probs, int *output_indices,
                 float *output_scores, int batch_size, int seq_len,
                 int num_classes, cudaStream_t stream) {
  int total_steps = batch_size * seq_len;
  // One block per step, 256 threads per block for parallel reduction
  argmax_kernel<<<total_steps, 256, 0, stream>>>(
      input_probs, output_indices, output_scores, total_steps, num_classes);
  CUDA_CHECK(cudaGetLastError());
}

// --- Fused Threshold + float->uint8 (vectorized float4 loads) ---

__global__ __launch_bounds__(256)
void threshold_to_u8_kernel(const float *src, uint8_t *dst, int w,
                                       int h, float thresh) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int total = w * h;

  if (idx + 3 < total) {
    // Vectorized path: load 4 floats at once via float4
    float4 vals = *reinterpret_cast<const float4*>(src + idx);
    uint8_t r0 = (vals.x > thresh) ? 255 : 0;
    uint8_t r1 = (vals.y > thresh) ? 255 : 0;
    uint8_t r2 = (vals.z > thresh) ? 255 : 0;
    uint8_t r3 = (vals.w > thresh) ? 255 : 0;
    // Pack 4 bytes into one uint32 store
    *reinterpret_cast<uint32_t*>(dst + idx) =
        (uint32_t)r0 | ((uint32_t)r1 << 8) | ((uint32_t)r2 << 16) | ((uint32_t)r3 << 24);
  } else {
    // Tail elements: scalar fallback
    for (int i = idx; i < min(idx + 4, total); i++)
      dst[i] = (src[i] > thresh) ? 255 : 0;
  }
}

void cuda_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                           float thresh, cudaStream_t stream) {
  int total = w * h;
  int threads = 256;
  // Each thread processes 4 elements, so divide by 4
  int elements_per_block = threads * 4;
  int blocks = (total + elements_per_block - 1) / elements_per_block;

  threshold_to_u8_kernel<<<blocks, threads, 0, stream>>>(src, dst, w, h,
                                                          thresh);
  CUDA_CHECK(cudaGetLastError());
}

void cuda_batch_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                                 int batch_size, float thresh,
                                 cudaStream_t stream) {
  // Treat as flat 1D array: pass total as (w_flat, 1) so kernel sees total = w_flat * 1.
  // Guard the size_t->int narrowing: w*h*batch as int would overflow once
  // per-side limits approach 16384 at batch 8, silently misindexing the
  // threshold writes. Fail loud instead if limits ever grow that far.
  const size_t total = static_cast<size_t>(w) * h * batch_size;
  if (total > static_cast<size_t>(INT_MAX))
    CUDA_CHECK(cudaErrorInvalidValue);
  const int total_pixels = static_cast<int>(total);
  int threads = 256;
  int elements_per_block = threads * 4;
  int blocks = (total_pixels + elements_per_block - 1) / elements_per_block;

  threshold_to_u8_kernel<<<blocks, threads, 0, stream>>>(src, dst, total_pixels, 1,
                                                          thresh);
  CUDA_CHECK(cudaGetLastError());
}

// ==========================================================================
// GPU Connected Component Labeling (CCL) for text detection
// ==========================================================================
//
// Block-based Union-Find (BUF) — the Allegretti-Bolelli-Grana GPU-CCL state of
// the art, an evolution of the Komura-Aharoni equivalence and Playne-Russell
// union-find lineage. The image is tiled into independent 2x2 blocks: under
// 8-connectivity every foreground pixel inside a 2x2 block is mutually adjacent,
// so the whole block collapses to a single node in the union-find forest. That
// quarters the node count and the atomic/union traffic versus per-pixel CCL.
//
// Labeling is ONE lock-free hooking pass (no iterate-to-convergence, no
// whole-grid barrier loop) followed by ONE path-compression pass — the property
// that makes union-find CCL near-optimal against older iterated label-
// equivalence propagation. Each block examines only its 4 backward neighbours
// {NW, N, NE, W}; the forward duals are covered when those blocks run, so every
// block adjacency is unioned exactly once with no redundant passes.
//
// Pipeline:
//   1. init:     one node per 2x2 block, self-labelled if it holds any fg pixel
//   2. merge:    hook adjacent fg blocks via atomic union-find (single pass)
//   3. compress: path-compress each block node to its component root
//   4. compact:  dense component ids over root blocks (cooperative barrier),
//                then scatter to the per-pixel compact_ids map (fg only)
//   5. extract:  fused bbox + pred_map score accumulation
//   6. filter:   score/size threshold -> output boxes
//
// A block node is addressed by its top-left pixel's raster index, so find/union
// run on the shared int[w*h] label buffer exactly as a per-pixel forest would.
// The resulting component partition is identical to what a correct per-pixel
// 8-connected labeller produces — only the compact-id ordering differs, and no
// downstream consumer depends on that ordering (bbox = min/max, score = sum).
//
// MEMORY SAFETY:
// - ALL buffers pre-allocated by caller (no cudaMallocAsync)
// - d_labels / d_compact_ids are int[w*h] scratch; only fg pixels of
//   compact_ids end up >= 0 (bg = -1), as the JFA expand path requires
// - d_bboxes must be allocated for kMaxGpuComponents * 2 GpuDetBox
//   (first half for extraction, second half for filtered output)
// - float4 loads in threshold kernel require 16-byte alignment
//   (guaranteed by cudaMalloc which returns 256-byte aligned ptrs)
// - Only ONE cudaStreamSynchronize at the very end; no mid-pipeline host reads
// ==========================================================================

// --- Device helpers ---

__device__ __forceinline__ int ccl_find_root(const int *labels, int idx) {
  while (labels[idx] != idx)
    idx = labels[idx];
  return idx;
}

// Find with path-halving (ECL-CC's in-find compression): every hop rewrites
// idx's parent to its grandparent, so concurrent and subsequent traversals of
// the same chain get exponentially shorter. Lock-free safe: the store only
// ever replaces a parent with one of idx's current ANCESTORS (gp was reached
// by following parent pointers, and union-find only merges, never splits), so
// root-reachability is preserved under any interleaving; racing halvers may
// store different ancestors, all of them valid. Roots are never written here
// (p == idx returns first), so this cannot race the hooking CAS in ccl_union,
// which only targets labels[b] while labels[b] == b.
__device__ __forceinline__ int ccl_find_root_halve(int *labels, int idx) {
  while (true) {
    int p = labels[idx];
    if (p == idx) return idx;
    int gp = labels[p];
    if (p == gp) return p;
    labels[idx] = gp;  // path-halving
    idx = gp;
  }
}

// Lock-free union: hook the higher-index root onto the lower one, retrying the
// find on a CAS race. Standard concurrent union-find (Komura / ECL-CC style).
// Single-pass correctness: this call only returns after its two endpoints
// share a root — the loop exits either via a successful hook (CAS observed
// labels[b] == b and linked it) or via a == b (already connected), and a
// failed CAS retries with the freshly hooked target (b = find(old)). Since
// union-find connectivity is monotone (merges are never undone), one pass over
// every edge yields the transitive closure. This invariant is enforced
// empirically by the pathological-shape partition test in
// tests/cpp/test_gpu_safety.cpp (spirals, staircases, dense fuzz vs OpenCV).
__device__ __forceinline__ void ccl_union(int *labels, int a, int b) {
  a = ccl_find_root_halve(labels, a);
  b = ccl_find_root_halve(labels, b);
  while (a != b) {
    if (a > b) { int t = a; a = b; b = t; }
    int old = atomicCAS(&labels[b], b, a);
    if (old == b) break;
    a = ccl_find_root_halve(labels, a);
    b = ccl_find_root_halve(labels, old);
  }
}

__device__ __forceinline__ bool ccl_fg(const uint8_t *bitmap, int w,
                                        int x, int y) {
  return bitmap[y * w + x] != 0;
}

// --- Block-based Union-Find (BUF) kernels ---
// A 2x2 block at grid coord (bx, by) is addressed by the raster index of its
// top-left pixel: (2*by)*w + 2*bx. That top-left pixel is always in bounds
// because nbx = ceil(w/2), nby = ceil(h/2).

// Step 1: init one union-find node per 2x2 block (self if any fg pixel, else bg)
__global__ __launch_bounds__(256)
void ccl_buf_init_kernel(const uint8_t * __restrict__ bitmap,
                         int * __restrict__ labels,
                         int w, int h, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  bool has_fg = ccl_fg(bitmap, w, x0, y0)
      || ((x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0))
      || ((y0 + 1 < h) && ccl_fg(bitmap, w, x0, y0 + 1))
      || ((x0 + 1 < w) && (y0 + 1 < h) && ccl_fg(bitmap, w, x0 + 1, y0 + 1));
  int rep = y0 * w + x0;
  labels[rep] = has_fg ? rep : -1;
}

// Step 2: single hooking pass. Union this fg block with its backward fg
// neighbours over the exact 8-connected block adjacencies. Pixels of a block:
//   P0 P1     at (x0,y0) (x0+1,y0)
//   P2 P3        (x0,y0+1) (x0+1,y0+1)
__global__ __launch_bounds__(256)
void ccl_buf_merge_kernel(const uint8_t * __restrict__ bitmap,
                          int * __restrict__ labels,
                          int w, int h, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  int rep = y0 * w + x0;
  if (labels[rep] < 0) return;  // background block

  bool p0 = ccl_fg(bitmap, w, x0, y0);
  bool p1 = (x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0);
  bool p2 = (y0 + 1 < h) && ccl_fg(bitmap, w, x0, y0 + 1);

  if (by > 0) {
    // N block: X top row (P0,P1) vs N bottom row — all four pairs 8-adjacent.
    bool n2 = ccl_fg(bitmap, w, x0, y0 - 1);
    bool n3 = (x0 + 1 < w) && ccl_fg(bitmap, w, x0 + 1, y0 - 1);
    if ((p0 || p1) && (n2 || n3))
      ccl_union(labels, rep, (y0 - 2) * w + x0);
    // NW block: only the X.P0 ~ NW.P3 corner touches.
    if (bx > 0 && p0 && ccl_fg(bitmap, w, x0 - 1, y0 - 1))
      ccl_union(labels, rep, (y0 - 2) * w + (x0 - 2));
    // NE block: only the X.P1 ~ NE.P2 corner touches.
    if (bx + 1 < nbx && p1 && (x0 + 2 < w) && ccl_fg(bitmap, w, x0 + 2, y0 - 1))
      ccl_union(labels, rep, (y0 - 2) * w + (x0 + 2));
  }
  if (bx > 0) {
    // W block: X left col (P0,P2) vs W right col — all four pairs 8-adjacent.
    bool w1 = ccl_fg(bitmap, w, x0 - 1, y0);
    bool w3 = (y0 + 1 < h) && ccl_fg(bitmap, w, x0 - 1, y0 + 1);
    if ((p0 || p2) && (w1 || w3))
      ccl_union(labels, rep, y0 * w + (x0 - 2));
  }
}

// Step 3: path-compress each fg block node to its component root.
__global__ __launch_bounds__(256)
void ccl_buf_compress_kernel(int * __restrict__ labels,
                             int w, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int rep = (2 * by) * w + (2 * bx);
  if (labels[rep] < 0) return;
  labels[rep] = ccl_find_root(labels, rep);
}

// Step 4a: dense component ids over root blocks, then propagate to non-root
// blocks. Cooperative grid.sync() keeps the two phases device-resident.
// Grid is limited to the cooperative maximum; a stride loop covers all blocks.
__global__ __launch_bounds__(256)
void ccl_buf_compact_assign_kernel(const int * __restrict__ labels,
                                   int * __restrict__ compact_ids,
                                   int * __restrict__ id_counter,
                                   int w, int nbx, int nby,
                                   int max_components) {
  auto grid = cg::this_grid();
  int nblocks = nbx * nby;
  int stride = gridDim.x * blockDim.x;

  for (int bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblocks;
       bi += stride) {
    int rep = (2 * (bi / nbx)) * w + 2 * (bi % nbx);
    if (labels[rep] == rep) {  // root of a foreground component
      int cid = atomicAdd(id_counter, 1);
      compact_ids[rep] = (cid < max_components) ? cid : -1;
    }
  }
  grid.sync();
  for (int bi = blockIdx.x * blockDim.x + threadIdx.x; bi < nblocks;
       bi += stride) {
    int rep = (2 * (bi / nbx)) * w + 2 * (bi % nbx);
    int root = labels[rep];
    if (root >= 0 && root != rep)
      compact_ids[rep] = compact_ids[root];
  }
}

// Step 4b: expand block-level ids to the per-pixel compact_ids map. Foreground
// pixels get their block's component id; every other pixel (incl. bg pixels
// inside a fg block) gets -1, so extract counts only true foreground.
__global__ __launch_bounds__(256)
void ccl_buf_scatter_kernel(const uint8_t * __restrict__ bitmap,
                            const int * __restrict__ labels,
                            int * __restrict__ compact_ids,
                            int w, int h, int nbx, int nby) {
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  if (bx >= nbx || by >= nby) return;
  int x0 = 2 * bx, y0 = 2 * by;
  int rep = y0 * w + x0;
  int cid = (labels[rep] >= 0) ? compact_ids[rep] : -1;  // read before overwrite
  compact_ids[rep] = ccl_fg(bitmap, w, x0, y0) ? cid : -1;
  if (x0 + 1 < w)
    compact_ids[rep + 1] = ccl_fg(bitmap, w, x0 + 1, y0) ? cid : -1;
  if (y0 + 1 < h)
    compact_ids[rep + w] = ccl_fg(bitmap, w, x0, y0 + 1) ? cid : -1;
  if (x0 + 1 < w && y0 + 1 < h)
    compact_ids[rep + w + 1] = ccl_fg(bitmap, w, x0 + 1, y0 + 1) ? cid : -1;
}

// Step 5: Init bboxes on GPU (kernel, not host memcpy -- no sync needed)
__global__ __launch_bounds__(256)
void ccl_init_bboxes_kernel(GpuDetBox * __restrict__ bboxes, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  bboxes[idx].xmin = INT_MAX;
  bboxes[idx].ymin = INT_MAX;
  bboxes[idx].xmax = 0;
  bboxes[idx].ymax = 0;
  bboxes[idx].pixel_count = 0;
  bboxes[idx].score = 0.0f;
}

// Step 6: Fused extract bboxes + accumulate score sum (single memory pass)
// score field accumulates raw pred_map sum; divided by pixel_count in filter.
//
// OPTIMIZED: shared-memory per-block accumulation with a fixed-size component
// hash table (32 slots). Each block accumulates local bbox/score in shared
// memory, then ONE thread per slot flushes to global with atomics.
// Reduces global atomics from N_pixels * 6 to N_blocks * 6 per component.
//
// Fallback: if a block sees >32 unique components (rare), excess pixels
// fall back to direct global atomics.

static constexpr int kExtractSlots = 32;       // shared-memory hash table size
static constexpr int kExtractSlotsMask = 31;    // kExtractSlots - 1

struct ExtractSlot {
  int cid;          // component ID (-1 = empty)
  int xmin, xmax;
  int ymin, ymax;
  int pixel_count;
  float score_sum;
};

__global__ __launch_bounds__(256)
void ccl_fused_extract_kernel(const int * __restrict__ compact_ids,
                               const float * __restrict__ pred_map,
                               GpuDetBox * __restrict__ bboxes,
                               int w, int h, int total) {
  __shared__ ExtractSlot slots[kExtractSlots];

  int tid = threadIdx.x;

  // Initialize shared memory slots
  if (tid < kExtractSlots) {
    slots[tid].cid = -1;
    slots[tid].xmin = INT_MAX;
    slots[tid].xmax = 0;
    slots[tid].ymin = INT_MAX;
    slots[tid].ymax = 0;
    slots[tid].pixel_count = 0;
    slots[tid].score_sum = 0.0f;
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int cid = compact_ids[idx];
    if (cid >= 0) {
      int x = idx % w;
      int y = idx / w;
      float score_val = pred_map[idx];

      // Open-addressing hash: linear probe to find or claim a slot
      int hash = (unsigned int)cid & kExtractSlotsMask;
      bool inserted = false;
      for (int probe = 0; probe < kExtractSlots; probe++) {
        int slot_idx = (hash + probe) & kExtractSlotsMask;

        // Try to claim an empty slot with atomicCAS
        int old = atomicCAS(&slots[slot_idx].cid, -1, cid);
        if (old == -1 || old == cid) {
          // We own this slot (either claimed it or it was already ours)
          atomicMin(&slots[slot_idx].xmin, x);
          atomicMax(&slots[slot_idx].xmax, x);
          atomicMin(&slots[slot_idx].ymin, y);
          atomicMax(&slots[slot_idx].ymax, y);
          atomicAdd(&slots[slot_idx].pixel_count, 1);
          atomicAdd(&slots[slot_idx].score_sum, score_val);
          inserted = true;
          break;
        }
        // Slot taken by different cid, continue probing
      }

      // Fallback: hash table full (>32 unique components in this block)
      if (!inserted) {
        atomicMin(&bboxes[cid].xmin, x);
        atomicMax(&bboxes[cid].xmax, x);
        atomicMin(&bboxes[cid].ymin, y);
        atomicMax(&bboxes[cid].ymax, y);
        atomicAdd(&bboxes[cid].pixel_count, 1);
        atomicAdd(&bboxes[cid].score, score_val);
      }
    }
  }
  __syncthreads();

  // Flush shared memory slots to global memory: one thread per slot
  if (tid < kExtractSlots) {
    int cid = slots[tid].cid;
    if (cid >= 0 && slots[tid].pixel_count > 0) {
      atomicMin(&bboxes[cid].xmin, slots[tid].xmin);
      atomicMax(&bboxes[cid].xmax, slots[tid].xmax);
      atomicMin(&bboxes[cid].ymin, slots[tid].ymin);
      atomicMax(&bboxes[cid].ymax, slots[tid].ymax);
      atomicAdd(&bboxes[cid].pixel_count, slots[tid].pixel_count);
      atomicAdd(&bboxes[cid].score, slots[tid].score_sum);
    }
  }
}

// Step 7: Filter (launched with kMaxGpuComponents threads -- no CPU sync needed)
// Unused slots have pixel_count=0 from init, so they early-return.
__global__ __launch_bounds__(256)
void ccl_filter_kernel(const GpuDetBox * __restrict__ bboxes,
                        float box_thresh,
                        GpuDetBox * __restrict__ out_bboxes,
                        int * __restrict__ out_count,
                        int max_components, int max_out) {
  int cid = blockIdx.x * blockDim.x + threadIdx.x;
  if (cid >= max_components) return;

  GpuDetBox box = bboxes[cid];
  if (box.pixel_count < 3) return;

  int bw = box.xmax - box.xmin + 1;
  int bh = box.ymax - box.ymin + 1;
  if (bw < 3 || bh < 3) return;

  float score = box.score / (float)box.pixel_count;
  if (score < box_thresh) return;

  box.score = score;
  int out_idx = atomicAdd(out_count, 1);
  if (out_idx < max_out)
    out_bboxes[out_idx] = box;
}

// ==========================================================================
// Host wrapper: full GPU CCL pipeline
// ALL buffers pre-allocated. No cudaMallocAsync/cudaFreeAsync.
// ONE cudaStreamSynchronize at the end.
// ==========================================================================

int cuda_gpu_ccl_detect(
    const uint8_t *d_bitmap,
    const float *d_pred_map,
    int w, int h,
    float box_thresh,
    int *d_labels,
    int *d_compact_ids,
    int *d_id_counter,
    GpuDetBox *d_bboxes,
    int *d_num_boxes,
    GpuDetBox *h_boxes,
    int *h_num_boxes,
    cudaStream_t stream,
    int *h_num_total) {

  int total = w * h;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  // 2x2-block grid for the BUF core (nbx * nby = ceil(w/2) * ceil(h/2) nodes).
  int nbx = (w + 1) / 2;
  int nby = (h + 1) / 2;
  dim3 bblock(32, 8);
  dim3 bgrid((nbx + bblock.x - 1) / bblock.x,
             (nby + bblock.y - 1) / bblock.y);

  // Step 1: init one union-find node per 2x2 block.
  ccl_buf_init_kernel<<<bgrid, bblock, 0, stream>>>(
      d_bitmap, d_labels, w, h, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 2: single lock-free hooking pass over the 8-connected block
  // adjacencies. No iterate-to-convergence, no whole-grid barrier loop.
  ccl_buf_merge_kernel<<<bgrid, bblock, 0, stream>>>(
      d_bitmap, d_labels, w, h, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 3: path-compress block nodes to their component roots.
  ccl_buf_compress_kernel<<<bgrid, bblock, 0, stream>>>(d_labels, w, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 4: dense component ids over root blocks (cooperative grid.sync between
  // assign and propagate), then scatter block ids to the per-pixel map.
  CUDA_CHECK(cudaMemsetAsync(d_id_counter, 0, sizeof(int), stream));
  {
    int nblocks = nbx * nby;
    int need = (nblocks + threads - 1) / threads;
    int coop_grid = coop_grid_for(ccl_buf_compact_assign_kernel, threads);
    if (coop_grid > need) coop_grid = need;
    if (coop_grid < 1) coop_grid = 1;

    int max_comp = kMaxGpuComponents;
    void *args[] = { (void*)&d_labels, (void*)&d_compact_ids,
                     (void*)&d_id_counter, (void*)&w, (void*)&nbx,
                     (void*)&nby, (void*)&max_comp };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)ccl_buf_compact_assign_kernel, dim3(coop_grid), dim3(threads),
        args, 0, stream));
    CUDA_CHECK(cudaGetLastError());
  }
  ccl_buf_scatter_kernel<<<bgrid, bblock, 0, stream>>>(
      d_bitmap, d_labels, d_compact_ids, w, h, nbx, nby);
  CUDA_CHECK(cudaGetLastError());

  // Step 5: Init bboxes via GPU kernel (not host memcpy -- no contention)
  {
    int bbox_blocks = (kMaxGpuComponents + threads - 1) / threads;
    ccl_init_bboxes_kernel<<<bbox_blocks, threads, 0, stream>>>(
        d_bboxes, kMaxGpuComponents);
    CUDA_CHECK(cudaGetLastError());
  }

  // Step 6: Fused extract bboxes + accumulate score
  ccl_fused_extract_kernel<<<blocks, threads, 0, stream>>>(
      d_compact_ids, d_pred_map, d_bboxes, w, h, total);
  CUDA_CHECK(cudaGetLastError());

  // Step 7: Filter (launch kMaxGpuComponents threads -- no CPU read needed)
  CUDA_CHECK(cudaMemsetAsync(d_num_boxes, 0, sizeof(int), stream));
  GpuDetBox *d_out_bboxes = d_bboxes + kMaxGpuComponents;
  {
    int filter_blocks = (kMaxGpuComponents + threads - 1) / threads;
    ccl_filter_kernel<<<filter_blocks, threads, 0, stream>>>(
        d_bboxes, box_thresh, d_out_bboxes, d_num_boxes,
        kMaxGpuComponents, kMaxGpuComponents);
    CUDA_CHECK(cudaGetLastError());
  }

  // === SINGLE SYNC: copy count + boxes in one batch, one sync ===
  CUDA_CHECK(cudaMemcpyAsync(h_num_boxes, d_num_boxes, sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
  // Optional: pre-filter component total for callers that index by pre-filter
  // compact_id (e.g. the JFA per-component expand path).
  if (h_num_total != nullptr) {
    CUDA_CHECK(cudaMemcpyAsync(h_num_total, d_id_counter, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
  }
  CUDA_CHECK(cudaMemcpyAsync(h_boxes, d_out_bboxes,
                              kMaxGpuComponents * sizeof(GpuDetBox),
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int h_count = std::min(*h_num_boxes, (int)kMaxGpuComponents);

  *h_num_boxes = h_count;
  return h_count;
}

} // namespace turbo_ocr::kernels
