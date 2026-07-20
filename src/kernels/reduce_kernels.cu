#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include <cfloat>
#include <climits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace turbo_ocr::kernels {

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

  // Every reduction breaks an exact-value tie toward the LOWER class index
  // (not the lower thread id), so the GPU argmax is deterministic and matches
  // the CPU/AVX2 CTC reference (ctc_decode.cpp), which keeps the lowest index.
  for (int stride = block_size / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      if (s_vals[tid + stride] > s_vals[tid] ||
          (s_vals[tid + stride] == s_vals[tid] && s_idxs[tid + stride] < s_idxs[tid])) {
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
    if (s_vals[tid + 32] > wval ||
        (s_vals[tid + 32] == wval && s_idxs[tid + 32] < widx)) {
      wval = s_vals[tid + 32]; widx = s_idxs[tid + 32];
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other_val = __shfl_down_sync(0xffffffff, wval, offset);
      int other_idx = __shfl_down_sync(0xffffffff, widx, offset);
      if (other_val > wval || (other_val == wval && other_idx < widx)) {
        wval = other_val; widx = other_idx;
      }
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

} // namespace turbo_ocr::kernels
