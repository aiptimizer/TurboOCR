// reduce_kernels.hip — HIP port of src/backends/nvidia/kernels_cuda/reduce_kernels.cu.
//
// argmax (CTC) + threshold->u8. Bodies match the CUDA math bit-for-bit; the
// argmax reduction is where the CUDA warp==32 assumption meets the AMD
// wavefront==64 reality — see the WARP-SIZE TODO on argmax_kernel.

#include "kernels_hip.h"
#include "amd/support/hip_check.h"
#include <cfloat>
#include <climits>
#include <hip/hip_runtime.h>

namespace turbo_ocr::amd::kernels {

// ============================================================================
// WARP-SIZE / WAVEFRONT — HANDLED, not deferred. Read this before touching the
// reduction below.
// ----------------------------------------------------------------------------
// The CUDA original (src/backends/nvidia/kernels_cuda/reduce_kernels.cu) reduces in shared
// memory down to stride==64 (every loop level closed by __syncthreads(), so
// the follow-up read of s_vals[tid+32] IS synchronized — the CUDA kernel is
// CORRECT; do not "fix" it), then finishes with a warp-shuffle tail:
// __shfl_down_sync(0xffffffff, v, offset) for offset in {16,8,4,2,1},
// hard-coded to NVIDIA's warpSize == 32.
//
// On AMD, warpSize is 64 on CDNA (gfx90a/gfx942) and configurable 32-or-64 on
// RDNA (gfx10/11 run wave32 by default under HIP, but wave64 is selectable and
// is what the compiler picks for some kernels). The porting hazard is the
// SHUFFLE tail:
//
//   __shfl_down_sync's `mask` is a no-op in HIP: all active lanes of the
//   wavefront participate and the width defaults to warpSize. With
//   warpSize==64 and offsets {16,8,4,2,1}, lanes 32..63 shuffle their own
//   (never-combined) partials into lanes 0..31 — a silent wrong answer, not
//   a crash.
//
// FIX APPLIED: the hand-rolled tail is GONE. The reduction is now a pure
// shared-memory tree all the way down to stride==1, with a __syncthreads() at
// every level. That is correct for ANY wavefront width (32, 64, or a future
// value) and on any target, with no warpSize query, no shuffle intrinsic, and
// no lockstep assumption. The cost is 5 extra __syncthreads() per timestep,
// which is noise next to the num_classes-wide (up to ~18,710) strided load loop
// that dominates this kernel — and correctness on unknown hardware is worth far
// more than a barrier we cannot measure.
//
// TIE-BREAKING IS PART OF THE CONTRACT: on equal values the LOWER class index
// wins, matching the scalar and AVX2 CPU reference in
// src/analysis/recognition/ctc_decode.cpp. The comparison below is written so this holds
// at every tree level regardless of the reduction ORDER — `>` for the value and
// `<` on the index for the tie — so a different wavefront width cannot change
// the result. This is the property to assert in the per-stage golden diff.
// ============================================================================

__global__ __launch_bounds__(256)
void argmax_kernel(const float *input_probs, int *output_indices,
                   float *output_scores, int total_steps, int num_classes) {
  int step_idx = blockIdx.x;
  if (step_idx >= total_steps)
    return;

  const float *probs_ptr = input_probs + step_idx * num_classes;
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  float local_max = -FLT_MAX;
  int local_idx = 0;
  for (int c = tid; c < num_classes; c += block_size) {
    float val = probs_ptr[c];
    if (val > local_max) {
      local_max = val;
      local_idx = c;
    }
  }

  __shared__ float s_vals[256];
  __shared__ int s_idxs[256];
  s_vals[tid] = local_max;
  s_idxs[tid] = local_idx;
  __syncthreads();

  // Wavefront-width-agnostic tree reduction. Ties break toward the LOWER class
  // index to match the CPU CTC reference (ctc_decode.cpp).
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float ov = s_vals[tid + stride];
      const int oi = s_idxs[tid + stride];
      if (ov > s_vals[tid] || (ov == s_vals[tid] && oi < s_idxs[tid])) {
        s_vals[tid] = ov;
        s_idxs[tid] = oi;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    output_indices[step_idx] = s_idxs[0];
    output_scores[step_idx] = s_vals[0];
  }
}

void hip_argmax(const float *input_probs, int *output_indices,
                float *output_scores, int batch_size, int seq_len,
                int num_classes, hipStream_t stream) {
  int total_steps = batch_size * seq_len;
  argmax_kernel<<<total_steps, 256, 0, stream>>>(
      input_probs, output_indices, output_scores, total_steps, num_classes);
  HIP_CHECK(hipGetLastError());
}

// --- Threshold + float->uint8 (vectorized float4). No warp ops; ports 1:1. ---

__global__ __launch_bounds__(256)
void threshold_to_u8_kernel(const float *src, uint8_t *dst, int w, int h,
                            float thresh) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  int total = w * h;

  if (idx + 3 < total) {
    float4 vals = *reinterpret_cast<const float4 *>(src + idx);
    uint8_t r0 = (vals.x > thresh) ? 255 : 0;
    uint8_t r1 = (vals.y > thresh) ? 255 : 0;
    uint8_t r2 = (vals.z > thresh) ? 255 : 0;
    uint8_t r3 = (vals.w > thresh) ? 255 : 0;
    *reinterpret_cast<uint32_t *>(dst + idx) =
        (uint32_t)r0 | ((uint32_t)r1 << 8) | ((uint32_t)r2 << 16) |
        ((uint32_t)r3 << 24);
  } else {
    for (int i = idx; i < min(idx + 4, total); i++)
      dst[i] = (src[i] > thresh) ? 255 : 0;
  }
}

void hip_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                         float thresh, hipStream_t stream) {
  int total = w * h;
  int threads = 256;
  int elements_per_block = threads * 4;
  int blocks = (total + elements_per_block - 1) / elements_per_block;

  threshold_to_u8_kernel<<<blocks, threads, 0, stream>>>(src, dst, w, h, thresh);
  HIP_CHECK(hipGetLastError());
}

void hip_batch_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                               int batch_size, float thresh, hipStream_t stream) {
  const size_t total = static_cast<size_t>(w) * h * batch_size;
  if (total > static_cast<size_t>(INT_MAX))
    HIP_CHECK(hipErrorInvalidValue);
  const int total_pixels = static_cast<int>(total);
  int threads = 256;
  int elements_per_block = threads * 4;
  int blocks = (total_pixels + elements_per_block - 1) / elements_per_block;

  threshold_to_u8_kernel<<<blocks, threads, 0, stream>>>(src, dst, total_pixels,
                                                         1, thresh);
  HIP_CHECK(hipGetLastError());
}

} // namespace turbo_ocr::amd::kernels
