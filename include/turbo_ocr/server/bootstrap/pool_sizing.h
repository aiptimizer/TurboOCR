#pragma once

#include <algorithm>
#include <cstddef>

#include "turbo_ocr/common/log/logger.h"

namespace turbo_ocr::server {

// Pipeline pool auto-sizing policy. Throughput is GPU-compute-bound on this
// stack: a clean pool sweep (RTX 5090, FUNSD) plateaus by ~5 pipelines
// (5→278, 8→263, 10→258 img/s) because the det/rec kernels already saturate
// the GPU and extra pipelines only add scheduling/cache pressure. So the
// ladder caps at 5 deliberately — raising it costs VRAM for negative
// throughput. An explicit --pool-size / PIPELINE_POOL_SIZE override bypasses
// this function entirely.
//
// Pure policy over the cudaMemGetInfo numbers (no CUDA dependency) so it is
// unit-testable on the CPU build.
[[nodiscard]] inline int compute_pipeline_pool_size(size_t free_mem,
                                                    size_t total_mem) {
  int pool_size;
  int vram_gb = static_cast<int>(total_mem >> 30);
  if (vram_gb >= 14) pool_size = 5;
  else if (vram_gb >= 12) pool_size = 3;
  else if (vram_gb >= 8)  pool_size = 2;
  else                     pool_size = 1;

  // Footprint-based safety floor: the tier above keys off TOTAL VRAM, but a
  // card that *reports* 16 GB while another process already holds most of
  // it would OOM during warmup. Estimate each pipeline's resident footprint
  // (engines + activation/workspace buffers, generous to stay conservative)
  // and reduce the tier so it fits in the FREE VRAM measured right now.
  // This only ever LOWERS the count — never raises it above the tier cap —
  // and never below 1, so a healthy 32 GB card with the tiny model still
  // picks 5 (a few hundred MB per pipeline against ~31 GB free).
  constexpr size_t kPerPipelineFootprintBytes = size_t{2} << 30;  // 2 GiB
  // Leave a 1 GiB headroom so the renderer daemons / CUDA context / OS
  // don't get squeezed to the byte.
  constexpr size_t kVramHeadroomBytes = size_t{1} << 30;
  size_t budget = free_mem > kVramHeadroomBytes ? free_mem - kVramHeadroomBytes : 0;
  int fits = static_cast<int>(budget / kPerPipelineFootprintBytes);
  if (fits < 1) fits = 1;
  if (fits < pool_size) {
    TOCR_LOG_WARN("Reducing pipeline pool to fit available VRAM",
                  "tier_pool_size", pool_size, "footprint_capped", fits,
                  "free_mem_gb", static_cast<int>(free_mem >> 30));
    pool_size = fits;
  }
  TOCR_LOG_INFO("Auto-detected pipeline pool size", "pool_size", pool_size, "vram_gb", vram_gb);
  return pool_size;
}

// Work-pool threads in front of the replica pool: they decode, parse and
// serialise while the replicas infer, so a few per replica keep the GPU fed
// and more only add idle threads. Measured flat from 20 to 48 threads on an
// RTX 5090 (pool 5); the previous max(pool*32, 128) put 128-160 threads on a
// 20-core box. Every extra thread is one more per-thread scratch set and one
// more allocator arena holding its own high-water mark of freed request
// buffers, which is host RSS that never comes back. HTTP_THREADS overrides.
[[nodiscard]] constexpr int compute_work_threads(int pool_size) noexcept {
  return std::clamp(pool_size * 4, 16, 64);
}

} // namespace turbo_ocr::server
