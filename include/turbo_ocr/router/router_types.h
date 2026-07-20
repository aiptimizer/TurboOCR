#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/router/router_destination.h"

namespace turbo_ocr::router {

inline constexpr int kNumLayoutClasses = 25;

enum class FormulaWrap : uint8_t {
  None = 0,
  Inline = 1,    // class_id 15 (inline_formula) + salvage path
  Display = 2,   // class_id 5  (display_formula)
};

enum class ConfidenceTier : uint8_t {
  Trust = 0,
  Verify = 1,
  Fallback = 2,
};

// Wired/wireless table verdict produced by turbostruct-table-cls.
// Forwarded as a hint to the table pipeline so it doesn't re-classify.
enum class TableClass : uint8_t {
  Wired = 0,
  Wireless = 1,
};

enum class RouterReason : uint8_t {
  ClassDefault = 0,
  FormulaSalvage,
  TableVerifyDemoted,
  FormulaVerifyDemoted,
  ContainmentLoser,
  IoUOverlapLoser,
  TableFallback,
  FormulaFallback,
  SkipWithDetPassthrough,
  SupplementaryRegion,
};

struct RouterConfig {
  // Per-class confidence cutoffs. Indices 0..24 are class_id; out-of-range
  // class_ids (e.g. the SupplementaryRegion sentinel -1) fall back to the
  // generic text-bound thresholds applied directly inside tier_for().
  std::array<float, kNumLayoutClasses> tau_trust{};
  std::array<float, kNumLayoutClasses> tau_verify{};

  bool enable_formula_salvage = true;
  bool enable_table_verification = true;
  bool image_text_fallback = true;

  // Per plan 05 §3:
  //   table (21)                              : 0.60 / 0.35
  //   display_formula (5), inline_formula (15): 0.55 / 0.30
  //   image (14), chart (3), seal (20)        : 0.50 / 0.30
  //   all other text-bound                    : 0.40 / 0.20
  static constexpr RouterConfig defaults() noexcept {
    RouterConfig c{};
    for (int i = 0; i < kNumLayoutClasses; ++i) {
      c.tau_trust[i]  = 0.40f;
      c.tau_verify[i] = 0.20f;
    }
    c.tau_trust[21]  = 0.60f; c.tau_verify[21]  = 0.25f; // table (verify<0.30 layout floor: route every detected table to SLANeXt — it's cheap and demoting a real table to text is the worst error; recovers low-confidence tables the layout emits in [0.30,0.35))
    c.tau_trust[5]   = 0.55f; c.tau_verify[5]   = 0.30f; // display_formula
    c.tau_trust[15]  = 0.55f; c.tau_verify[15]  = 0.30f; // inline_formula
    c.tau_trust[14]  = 0.50f; c.tau_verify[14]  = 0.30f; // image
    c.tau_trust[3]   = 0.50f; c.tau_verify[3]   = 0.30f; // chart
    c.tau_trust[20]  = 0.50f; c.tau_verify[20]  = 0.30f; // seal
    return c;
  }
};

// Per-layout-cell stats from a single linear pass over the det boxes.
// All four vectors are sized to layout.size() and indexed by layout_idx.
// Built lazily — only when ≥1 layout box ∈ {table, display_formula,
// inline_formula, image, chart}; pure-text pages skip the pass entirely.
struct OverlapStats {
  std::vector<int>   det_count;
  std::vector<float> det_coverage;        // sum(det_aabb_area) / layout_area, capped at 1.0
  std::vector<float> mean_aspect_ratio;   // mean(w/h) over contained det boxes
  std::vector<float> symbol_density_hint; // proxy for character density (see region_features.cpp)

  void clear() noexcept {
    det_count.clear();
    det_coverage.clear();
    mean_aspect_ratio.clear();
    symbol_density_hint.clear();
  }

  void resize(std::size_t n) {
    det_count.assign(n, 0);
    det_coverage.assign(n, 0.0f);
    mean_aspect_ratio.assign(n, 0.0f);
    symbol_density_hint.assign(n, 0.0f);
  }

  [[nodiscard]] bool empty() const noexcept { return det_count.empty(); }
};

struct PageStats {
  bool  has_confident_formula = false;
  int   median_text_line_height = 0;
  float median_text_line_area = 0.0f;
};

// Per-region decoded outputs produced by the table / formula stages.
// Surfaced on OcrPipelineResult and serialized as "tables" / "formulas"
// JSON arrays — see common/serialization/serialization.h.
struct TableResult {
  int         layout_id = -1;
  std::string html;
  float       score = 0.0f;
  turbo_ocr::Box box{};
};

struct FormulaResult {
  int         layout_id = -1;
  std::string latex;
  float       score = 0.0f;
  turbo_ocr::Box box{};
};

struct RouterDecision {
  int                       layout_idx = -1;
  Destination               dest = Destination::Text;
  FormulaWrap               wrap = FormulaWrap::None;
  ConfidenceTier            tier = ConfidenceTier::Fallback;
  RouterReason              reason = RouterReason::ClassDefault;
  std::optional<TableClass> wired_hint;
  // Dual-routing flag for formula salvage: when true the layout cell
  // is dispatched to BOTH the formula stream and the text path; the
  // merge step keeps whichever fires confidently.
  bool                      also_text = false;
};

} // namespace turbo_ocr::router
