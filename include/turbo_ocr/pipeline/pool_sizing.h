#pragma once

// Pipeline-pool auto-sizing policy.
//
// It lives under pipeline/ because that is what it sizes, and because its ONE
// consumer is a vendor backend answering BackendCaps::recommended_pool_size. It
// used to sit in service/server/bootstrap/, which made it the single edge from
// src/backends/ up into the transport layer — a device-memory policy that a
// backend had to reach through the HTTP server to find. Nothing about it is
// transport, and nothing about it is CUDA either: it is arithmetic over two
// memory numbers, which is why the test below it runs on the CPU build.

#include <cstddef>
#include <string_view>

#include "turbo_ocr/base/log/logger.h"

namespace turbo_ocr::pipeline {

// Pipeline pool auto-sizing policy. The ladder caps at 5 because a pool sweep
// (RTX 5090, FUNSD) plateaued there — 5→278, 8→263, 10→258 img/s — the det/rec
// kernels saturating the GPU while extra pipelines only add scheduling and
// cache pressure.
//
// THAT CEILING IS TIER-DEPENDENT, and 5 is not always the peak. Re-measured on
// the tiny tier with rec CUDA graphs live (2026-08-01, same card, concurrency
// 16): 5→579, 6→633, 8→647 img/s, still climbing where the sweep above had
// already turned over. A cheaper model leaves GPU headroom a single replica
// cannot fill, so it wants a deeper pool. The cap stays at 5 because raising it
// spends VRAM on every card to help one tier, and the footprint floor below
// would let a 16 GB card jump to 7 replicas — deployments that know their tier
// set PIPELINE_POOL_SIZE (or --pool-size), which bypasses this function.
//
// Pure policy over the free/total device-memory numbers (no CUDA dependency) so
// it is unit-testable on the CPU build. PURE also means SILENT: it used to log
// its result at INFO, and load_stages() calls it once per replica, so a
// 5-replica boot printed the same "Auto-detected pipeline pool size" line seven
// times. The decision is logged once by whoever makes it (backend_stages.cpp).
// The VRAM-shortfall warning below stays, because that one fires rarely and
// says something the caller cannot reconstruct.
// Measured resident VRAM per pipeline replica, by model tier (RTX 5090,
// 2026-08-01, layout on, rec CUDA graphs live). These are the numbers the
// footprint floor below needs and did not have:
//
//   tiny   14.6 GB / 5 replicas ~= 2.9 GB each
//   small  22.5 GB / 5 replicas ~= 4.5 GB each
//
// The old floor assumed a flat 2 GiB. It is not a conservative estimate — it is
// under the lightest tier and less than HALF the small tier, so its job (stop a
// card from OOMing during warmup) was one it could not do. `small` at pool 8
// needs ~36 GB and duly died with "CUDA Error ... out of memory" at startup,
// which is the failure this floor exists to prevent.
//
// 4.5 GiB is the measured worst case across the tiers that ship. Using it costs
// nothing on a large card (the tier ladder caps at 5 either way) and correctly
// reduces the pool on a small one instead of letting it crash.
inline constexpr size_t kPerPipelineFootprintBytes =
    (size_t{9} << 30) / 2;  // 4.5 GiB

// EXTRA per-replica device scratch of the routed FORMULA engine, added on top
// of kPerPipelineFootprintBytes by the vendor caps() call sites. The base
// footprint's 2026-08-01 measurements were taken with the default plus-S
// engine loaded, so plus-S is already inside the base. plus-M is not: each
// replica allocates ~3.3 GiB of decode buffers (4x 1056-window static-KV at
// ~415 MiB, 4x 384-window bucket at ~151 MiB, 128-crop cross-KV + all-crop
// encoder memory) plus its ORT sessions; `auto` loads plus-M NEXT TO plus-S.
// Without this surcharge the sizer picked the tier pool of 5 for
// FORMULA_BACKEND=auto on a 32 GiB card and boot OOM'd inside the plus-M
// encoder load (2026-08-05); 4 GiB lands it on the pool that actually fits
// (the same count that booted and served when set by hand).
[[nodiscard]] inline size_t formula_engine_scratch_bytes(std::string_view engine) {
  if (engine == "ppformulanet_plus_m" || engine == "auto")
    return size_t{4} << 30;  // 4 GiB: plus-M replica buffers + sessions
  return 0;  // plus-S / vlm / none — inside (or irrelevant to) the base
}

// `per_pipeline_bytes` defaults to the measured WORST case across shipping
// tiers, because an auto-sizer that guesses low crashes at warmup while one that
// guesses high only leaves throughput on the table — and PIPELINE_POOL_SIZE
// overrides it either way. A caller that knows its tier should pass that tier's
// number instead of inheriting the worst case.
[[nodiscard]] inline int
compute_pipeline_pool_size(size_t free_mem, size_t total_mem,
                           size_t per_pipeline_bytes =
                               kPerPipelineFootprintBytes) {
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
  // and never below 1. Footprint constant and its measurements: above.
  // Leave a 1 GiB headroom so the renderer daemons / CUDA context / OS
  // don't get squeezed to the byte.
  constexpr size_t kVramHeadroomBytes = size_t{1} << 30;
  size_t budget = free_mem > kVramHeadroomBytes ? free_mem - kVramHeadroomBytes : 0;
  int fits = static_cast<int>(budget / per_pipeline_bytes);
  if (fits < 1) fits = 1;
  if (fits < pool_size) {
    TOCR_LOG_WARN("Reducing pipeline pool to fit available VRAM",
                  "tier_pool_size", pool_size, "footprint_capped", fits,
                  "free_mem_gb", static_cast<int>(free_mem >> 30));
    pool_size = fits;
  }
  return pool_size;
}

} // namespace turbo_ocr::pipeline
