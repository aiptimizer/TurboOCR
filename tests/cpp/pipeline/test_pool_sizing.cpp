#include <catch_amalgamated.hpp>

#include "turbo_ocr/pipeline/pool_sizing.h"

using turbo_ocr::pipeline::compute_pipeline_pool_size;

namespace {
constexpr size_t GiB = size_t{1} << 30;
// The measured per-replica footprints the policy is calibrated against
// (pool_sizing.h carries the measurements).
constexpr size_t kTiny = (size_t{29} << 30) / 10;  // 2.9 GiB
constexpr size_t kSmall = (size_t{9} << 30) / 2;   // 4.5 GiB — the default
}

TEST_CASE("VRAM tiers pick the sweep-derived ladder", "[pool_sizing]") {
  // A card with room for the tier: the ladder decides, the floor does not bind.
  CHECK(compute_pipeline_pool_size(31 * GiB, 32 * GiB) == 5);
  CHECK(compute_pipeline_pool_size(11 * GiB, 12 * GiB, kTiny) == 3);
  CHECK(compute_pipeline_pool_size(7 * GiB, 8 * GiB, kTiny) == 2);
  CHECK(compute_pipeline_pool_size(5 * GiB, 6 * GiB, kTiny) == 1);
}

TEST_CASE("the floor binds where the tier ladder alone would OOM",
          "[pool_sizing]") {
  // THE BUG THIS ENCODES. The floor used to assume 2 GiB per replica, so a
  // 16 GB card was told it could host the full tier of 5. Five replicas of the
  // `small` tier need ~22.5 GB (measured), so that card died at warmup with
  // "CUDA Error ... out of memory" — the exact failure the floor exists to stop.
  // At the measured 4.5 GiB it answers 3, which fits.
  CHECK(compute_pipeline_pool_size(15 * GiB, 16 * GiB, kSmall) == 3);
  CHECK(compute_pipeline_pool_size(13 * GiB, 14 * GiB, kSmall) == 2);
  // kSmall IS the default, so the same answer arrives without naming it.
  CHECK(compute_pipeline_pool_size(15 * GiB, 16 * GiB) == 3);
  // A caller that knows it is running the light tier gets more out of the same
  // card — which is why the footprint is a parameter and not a baked constant.
  CHECK(compute_pipeline_pool_size(15 * GiB, 16 * GiB, kTiny) == 4);
}

TEST_CASE("footprint floor only lowers the tier, never raises", "[pool_sizing]") {
  // 16 GB card with most VRAM held by another process: (5 GiB - 1 headroom)
  // / 2.9 GiB footprint = 1 pipeline despite the 5-tier.
  CHECK(compute_pipeline_pool_size(5 * GiB, 16 * GiB, kTiny) == 1);
  CHECK(compute_pipeline_pool_size(9 * GiB, 16 * GiB, kTiny) == 2);
  // Nothing free: clamps to 1, never 0.
  CHECK(compute_pipeline_pool_size(0, 16 * GiB) == 1);
  CHECK(compute_pipeline_pool_size(512 << 20, 16 * GiB) == 1);
  // Tiny tier with abundant free memory stays at the tier cap.
  CHECK(compute_pipeline_pool_size(7 * GiB, 8 * GiB, kTiny) == 2);
}

TEST_CASE("formula engine scratch surcharge sizes the auto/plus-M pool to fit",
          "[pool_sizing]") {
  using turbo_ocr::pipeline::formula_engine_scratch_bytes;
  using turbo_ocr::pipeline::kPerPipelineFootprintBytes;
  // plus-S (the default) is inside the measured base footprint; plus-M and the
  // auto ladder add their per-replica decode buffers on top.
  CHECK(formula_engine_scratch_bytes("ppformulanet_s") == 0);
  CHECK(formula_engine_scratch_bytes("") == 0);
  CHECK(formula_engine_scratch_bytes("vlm") == 0);
  CHECK(formula_engine_scratch_bytes("ppformulanet_plus_m") == 4 * GiB);
  CHECK(formula_engine_scratch_bytes("auto") == 4 * GiB);

  // The 2026-08-05 OOM, re-run through the policy: 32 GiB card, ~21 GiB free
  // at sizing time, FORMULA_BACKEND=auto. Base footprint alone picked 4 and
  // boot died inside the plus-M encoder load; with the surcharge the sizer
  // lands on 2 — the pool that booted and served when set by hand.
  const size_t per = kPerPipelineFootprintBytes + formula_engine_scratch_bytes("auto");
  CHECK(compute_pipeline_pool_size(21 * GiB, 32 * GiB, per) == 2);
  // Plenty of VRAM: the tier cap still rules — the surcharge never raises it.
  CHECK(compute_pipeline_pool_size(60 * GiB, 64 * GiB, per) == 5);
}
