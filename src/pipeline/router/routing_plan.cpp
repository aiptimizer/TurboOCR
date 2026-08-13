#include "turbo_ocr/pipeline/router/routing_plan.h"

namespace turbo_ocr::router {

void RoutingPlan::clear() noexcept {
  // `clear()` keeps capacity — the owning CuaRouter is thread-local so
  // the vectors grow once to the page's working set and stay sized.
  text_indices.clear();
  table_layout_ids.clear();
  formula_layout_ids.clear();
  skip_layout_ids.clear();
  text_to_layout_id.clear();
  rec_suppress.clear();
}

} // namespace turbo_ocr::router
