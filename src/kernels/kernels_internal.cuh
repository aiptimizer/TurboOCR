#pragma once

#include <cuda_runtime.h>

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
    if (cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount,
                               dev) != cudaSuccess || cached_sms <= 0)
      cached_sms = 1;  // conservative fallback; the kernel's stride loop is
                       // still correct on a single-block grid, just slower
    cached_dev = dev;
    cached_fn = nullptr;
  }
  if (cached_fn != (const void *)kernel || cached_threads != threads) {
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &cached_per_sm, kernel, threads, 0) != cudaSuccess ||
        cached_per_sm <= 0)
      cached_per_sm = 1;
    cached_fn = (const void *)kernel;
    cached_threads = threads;
  }
  return cached_per_sm * cached_sms;
}

} // namespace turbo_ocr::kernels
