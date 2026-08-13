#pragma once

#include <hip/hip_runtime.h>

namespace turbo_ocr::amd::kernels {

// Cache occupancy + CU count per (device, kernel, block_size). HIP mirror of
// kernels_internal.cuh's coop_grid_for. Uses hipGetDevice so multi-GPU hosts
// don't get sized for device 0.
//
// TODO(on-hardware): cooperative launch is only valid when the gfx target and
// driver support it — gate the *caller* on
//   hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, dev)
// and fall back to the non-cooperative two-kernel compact path (see
// ccl_kernels_hip.hip TODO) when coop == 0. This helper only SIZES the grid; it
// does not verify support.
template <typename Fn>
static int coop_grid_for(Fn kernel, int threads) {
  int dev = 0;
  hipGetDevice(&dev);
  static thread_local int cached_dev = -1;
  static thread_local int cached_cus = 0;
  static thread_local const void *cached_fn = nullptr;
  static thread_local int cached_threads = 0;
  static thread_local int cached_per_cu = 0;
  if (dev != cached_dev) {
    // hipDeviceAttributeMultiprocessorCount == number of compute units (CUs).
    if (hipDeviceGetAttribute(&cached_cus,
                              hipDeviceAttributeMultiprocessorCount,
                              dev) != hipSuccess ||
        cached_cus <= 0)
      cached_cus = 1; // conservative fallback; the stride loop is still correct
    cached_dev = dev;
    cached_fn = nullptr;
  }
  if (cached_fn != (const void *)kernel || cached_threads != threads) {
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &cached_per_cu, kernel, threads, 0) != hipSuccess ||
        cached_per_cu <= 0)
      cached_per_cu = 1;
    cached_fn = (const void *)kernel;
    cached_threads = threads;
  }
  return cached_per_cu * cached_cus;
}

} // namespace turbo_ocr::amd::kernels
