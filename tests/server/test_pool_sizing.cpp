#include <catch_amalgamated.hpp>

#include "turbo_ocr/server/bootstrap/pool_sizing.h"

using turbo_ocr::server::compute_pipeline_pool_size;

namespace {
constexpr size_t GiB = size_t{1} << 30;
}

TEST_CASE("VRAM tiers pick the sweep-derived ladder", "[pool_sizing]") {
  // Healthy cards: free ≈ total, tier decides.
  CHECK(compute_pipeline_pool_size(31 * GiB, 32 * GiB) == 5);
  CHECK(compute_pipeline_pool_size(15 * GiB, 16 * GiB) == 5);
  CHECK(compute_pipeline_pool_size(13 * GiB, 14 * GiB) == 5);
  CHECK(compute_pipeline_pool_size(11 * GiB, 12 * GiB) == 3);
  CHECK(compute_pipeline_pool_size(7 * GiB, 8 * GiB) == 2);
  CHECK(compute_pipeline_pool_size(5 * GiB, 6 * GiB) == 1);
}

TEST_CASE("footprint floor only lowers the tier, never raises", "[pool_sizing]") {
  // 16 GB card with most VRAM held by another process: (5 GiB - 1 headroom)
  // / 2 GiB footprint = 2 pipelines despite the 5-tier.
  CHECK(compute_pipeline_pool_size(5 * GiB, 16 * GiB) == 2);
  // Nothing free: clamps to 1, never 0.
  CHECK(compute_pipeline_pool_size(0, 16 * GiB) == 1);
  CHECK(compute_pipeline_pool_size(512 << 20, 16 * GiB) == 1);
  // Tiny tier with abundant free memory stays at the tier cap.
  CHECK(compute_pipeline_pool_size(7 * GiB, 8 * GiB) == 2);
}
