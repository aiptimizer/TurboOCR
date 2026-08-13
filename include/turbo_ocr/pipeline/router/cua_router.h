#pragma once

#include <cstdint>
#include <vector>

#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pipeline/router/router_destination.h"
#include "turbo_ocr/core/router_types.h"
#include "turbo_ocr/pipeline/router/routing_plan.h"

namespace turbo_ocr::router {

// Device-free routing stage: a PURE CLASSIFIER. It emits a RoutingPlan and
// executes nothing — UnifiedOcrPipeline::dispatch_router_ is what acts on the
// plan. (This banner used to promise "dispatch entrypoints" that are "stubs
// until later tasks", naming TablePipeline / FormulaPipeline; neither type has
// ever existed in this tree, and the split of concerns turned out to be the
// right one anyway.) One instance per pipeline-pool slot, so the internal
// scratch buffers never get cross-thread access.
class CuaRouter {
 public:
  CuaRouter() noexcept;
  ~CuaRouter() noexcept;

  CuaRouter(const CuaRouter &) = delete;
  CuaRouter &operator=(const CuaRouter &) = delete;


  // NOTE (removed): set_config()/config() had zero callers, so cfg_ was
  // permanently RouterConfig::defaults() and none of its tunables could be
  // changed at runtime — a configuration surface that looked live and was not.
  // Restore them together with a caller if per-deployment thresholds are
  // wanted; an accessor pair with no reader is worse than neither.

  // Main entry: emit RoutingPlan from already-sorted det boxes + layout
  // boxes. CPU only — no GPU calls. Bounded ≤2 ms on a 30-layout / 200-
  // det-box page (text-only path drops to <0.5 ms via the OverlapStats
  // lazy gate).
  void classify(const std::vector<turbo_ocr::Box> &det_boxes,
                const std::vector<turbo_ocr::layout::LayoutBox> &layout,
                RoutingPlan &plan) const;

  // Per-page decisions for inspection / debugging / bench. Lives on the
  // router so it survives across classify() calls (cleared each call).
  [[nodiscard]] const std::vector<RouterDecision> &
  last_decisions() const noexcept { return decisions_; }

 private:
  RouterConfig cfg_ = RouterConfig::defaults();

  // Reusable scratch — kept on the router instance so classify() does
  // no allocations on the hot path once the per-pipeline-slot warm-up
  // has run. Caller is single-threaded per CuaRouter instance.
  mutable std::vector<RouterDecision> decisions_;
  mutable OverlapStats overlap_;
  mutable PageStats page_stats_;
  mutable std::vector<std::array<int, 4>> layout_aabbs_;
};

// --- pure free functions: the 9 unit-testable seams from plan 05 §12 ---

[[nodiscard]] Destination default_destination(int class_id) noexcept;

[[nodiscard]] ConfidenceTier tier_for(int class_id, float score,
                                      const RouterConfig &cfg) noexcept;

[[nodiscard]] FormulaWrap formula_wrap_for(int class_id) noexcept;

// Single CPU pass over det boxes — caller fills `out` with size ==
// layout.size(). Skips work entirely when the gate is closed.
void build_overlap_stats(const std::vector<turbo_ocr::layout::LayoutBox> &layout,
                         const std::vector<turbo_ocr::Box> &det_boxes,
                         const std::vector<std::array<int, 4>> &layout_aabbs,
                         OverlapStats &out);

[[nodiscard]] bool
is_formula_salvage_candidate(const turbo_ocr::layout::LayoutBox &lb,
                             int layout_idx,
                             const OverlapStats &ov,
                             const PageStats &page,
                             const RouterConfig &cfg) noexcept;

RouterDecision route(int layout_idx,
                     const turbo_ocr::layout::LayoutBox &lb,
                     const OverlapStats &ov,
                     const PageStats &page,
                     const RouterConfig &cfg) noexcept;

void resolve_tie_breakers(
    std::vector<RouterDecision> &decisions,
    const std::vector<turbo_ocr::layout::LayoutBox> &layout,
    const std::vector<std::array<int, 4>> &layout_aabbs) noexcept;

// Table verification: turbostruct-table-cls gate. Should we fire it?
struct VerifyTablePath {
  [[nodiscard]] static bool
  should_invoke(const RouterDecision &d) noexcept;
};

// Verdict from the (sub-1 ms) PP-LCNet table-cls model. `max_score`
// is the larger of P(wired)/P(wireless); `is_table` is true iff
// max_score ≥ 0.55.
struct TableClsVerdict {
  bool is_table;
  TableClass cls;
  float max_score;
};

RouterDecision apply_verify_result(RouterDecision d,
                                   TableClsVerdict v) noexcept;

// --- region_features.cpp helpers (used by build_overlap_stats) ---

// Heuristic for "how many small symbols per unit area" — proxies for
// formula density. Currently: clamped det-count / log-scaled
// layout-area. Stays in [0, 1].
[[nodiscard]] float region_symbol_density_hint(
    const std::array<int, 4> &layout_aabb,
    int contained_det_count) noexcept;

} // namespace turbo_ocr::router
