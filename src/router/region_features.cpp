#include "turbo_ocr/router/cua_router.h"

#include <algorithm>
#include <cmath>

namespace turbo_ocr::router {

namespace {


} // namespace

float region_symbol_density_hint(
    const std::array<int, 4> &layout_aabb,
    int contained_det_count) noexcept {
  // Cheap proxy: small layout cells with several det boxes are
  // formula-shaped — sub/superscripts, fraction bars, integral
  // symbols all surface as separate det boxes within a compact
  // region. log1p smooths over the wide area range without needing
  // a per-page normalization pass.
  const float w = static_cast<float>(std::max(0, layout_aabb[2] - layout_aabb[0]));
  const float h = static_cast<float>(std::max(0, layout_aabb[3] - layout_aabb[1]));
  const float area = std::max(1.0f, w * h);
  const float density = static_cast<float>(contained_det_count) /
                        std::log1p(area);
  return std::clamp(density, 0.0f, 1.0f);
}

} // namespace turbo_ocr::router
