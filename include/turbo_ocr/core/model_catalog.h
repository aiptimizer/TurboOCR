#pragma once

#include <array>
#include <string_view>

#include "turbo_ocr/analysis/detection/det_config.h"

namespace turbo_ocr::server {

using turbo_ocr::detection::DbParams;
using turbo_ocr::detection::DetResizeParams;

// Origin of a registry model. v6 entries are the script-agnostic PP-OCRv6
// tiers (Latin + Chinese + Japanese); V5Lang entries are the retained legacy
// recognizers for scripts v6 does not cover.
enum class ModelFamily { V6, V5Lang };

// Per-model detection inference config: the official PaddleOCR values for this
// model's detector (resize policy + DB post-processing). Env vars still override
// each field at read time (read_det_resize/read_db_params).
struct DetInferConfig {
  DetResizeParams resize{};
  DbParams db{};
};

// Official PaddleOCR det config for the PP-OCRv6 tiers whose box_thresh is 0.45
// (medium, small) — and reused by the V5Lang rows, which serve through the
// shared default v6 detector (models/det.onnx). resize policy is the official
// PaddleOCR "min"/64 (native resolution); max_side_limit 1280 is the pooled-
// server practical cap (official 4000 OOMs the pre-allocated pool — see
// det_config.h kDetResizeDefault). DB is thresh 0.2 / box 0.45 / unclip 1.4.
// Env-overridable per field at read time.
inline constexpr DetInferConfig kV6DetConfig{
    DetResizeParams{"min", 64, 1280}, DbParams{0.2f, 0.45f, 1.4f}};

// Tiny tier differs only in box_thresh (0.40, per its inference.yml).
inline constexpr DetInferConfig kV6DetConfigTiny{
    DetResizeParams{"min", 64, 1280}, DbParams{0.2f, 0.40f, 1.4f}};

// One selectable OCR model. Adding a model is a single row.
//
// `det_path` empty => the default detector (`models/det.onnx`, v6 medium-tier).
// Each v6 tier ships its own detector (det.onnx / det_small.onnx /
// det_tiny.onnx); the V5Lang rec recognizers carry no detector of their own and
// fall back to the default, reusing the v6 det config.
struct ModelEntry {
  std::string_view name;
  std::string_view rec_path;
  std::string_view dict_path;
  std::string_view det_path;  // empty => kDefaultDet
  ModelFamily family = ModelFamily::V6;
  DetInferConfig det_cfg{};
};

inline constexpr std::string_view kDefaultDet = "models/det.onnx";
inline constexpr std::string_view kDefaultModel = "tiny";

// The model registry. v6 tiers share the medium/small dictionary except tiny,
// which ships its own (6,904 chars vs 18,708). Legacy scripts keep their v5
// recognizer + dict under models/rec/<name>/ and the shared v6 detector + its
// det config. Each model carries its official det inference values.
// A row is NOT required to pair a tier's detector with the same tier's
// recognizer. "tiny-bigdet" below is deliberately asymmetric, and the reason is
// measured: on degraded scans the DETECTOR is what fails first. Holding a fax
// degradation constant and shrinking glyph height (em px = pt/72*dpi), swapping
// ONLY the detector to the full det.onnx while keeping rec_tiny recovers far
// more than swapping only the recognizer:
//
//   em px   tiny    +big det   +big rec   medium
//   16.2    96.17%   96.17%     98.62%    98.62%   <- rec still carries it here
//   13.9    91.26%   94.79%     97.70%    98.16%
//   12.2    81.29%   87.27%     92.48%    97.55%
//   11.1    20.86%   74.54%     31.29%    93.25%   <- detector dominates: +53.7
//                                                     vs +10.4 for the rec
// At 11 px tiny's detector fragments 16 lines into 19 boxes and loses whole
// line-starts; no recognizer can read text that was never detected.
//
// BUT MEASURE BEFORE REACHING FOR THIS ROW: it is Pareto-dominated by "small".
// The full detector's forward pass (50.4 ms at 1280x1280, against det_small's
// 13.3 ms) costs more than small's whole pipeline, so this row lands SLOWER and
// LESS ACCURATE than small — 128-141 ms and 74.54% at 11 px, versus small's
// 102-115 ms and 81.44%. Its one advantage is footprint: 150 MB of det+rec
// against small's 207 MB, because the tiny recognizer's bucket ladder is 66 MB
// and small's is 173 MB. It earns its place only when disk or memory is the
// binding constraint, NOT as a cheap route to small's robustness.
//
// THE ROW EXISTS BECAUSE THE ENV ROUTE CANNOT EXPRESS THIS SAFELY. Setting
// DET_MODEL=models/det.onnx with OCR_MODEL=tiny makes resolve_model treat the
// detector as "overridden", which discards the entry's det_cfg and falls back
// to {kDetResizeDefault, kDbDefaults}. For THIS pairing those defaults happen
// to equal kV6DetConfig, so the env route is accidentally right — but the same
// mechanism silently runs det_tiny at box_thresh 0.45 instead of its own 0.40
// whenever the override goes the other way. A registry row carries det_path AND
// det_cfg together, so the pairing is correct by construction.
inline constexpr std::array<ModelEntry, 9> kModelCatalog{{
    {"medium", "models/rec.onnx",       "models/keys.txt",      "models/det.onnx",       ModelFamily::V6,     kV6DetConfig},
    {"small",  "models/rec_small.onnx", "models/keys.txt",      "models/det_small.onnx", ModelFamily::V6,     kV6DetConfig},
    {"tiny",   "models/rec_tiny.onnx",  "models/keys_tiny.txt", "models/det_tiny.onnx",  ModelFamily::V6,     kV6DetConfigTiny},
    // Full-size detector + tiny recognizer. det_cfg is kV6DetConfig (box_thresh
    // 0.45), NOT kV6DetConfigTiny — the DB parameters belong to the detector
    // that runs, and det.onnx is the medium tier's detector, whose inference.yml
    // specifies 0.45. Pairing it with tiny's 0.40 would mis-threshold a detector
    // this row never uses. The dict MUST stay keys_tiny.txt: rec_tiny.onnx has
    // 6,904 classes and keys.txt has 18,708, and a mismatched dict decodes every
    // crop against the wrong table, producing confident garbage rather than an
    // error. Not a tier — the recognizer is still tiny's, so it inherits tiny's
    // ceiling on script (no Japanese kana) and on very small glyphs.
    {"tiny-bigdet", "models/rec_tiny.onnx", "models/keys_tiny.txt", "models/det.onnx", ModelFamily::V6, kV6DetConfig},
    {"arabic", "models/rec/arabic/rec.onnx", "models/rec/arabic/dict.txt", "", ModelFamily::V5Lang, kV6DetConfig},
    {"eslav",  "models/rec/eslav/rec.onnx",  "models/rec/eslav/dict.txt",  "", ModelFamily::V5Lang, kV6DetConfig},
    {"korean", "models/rec/korean/rec.onnx", "models/rec/korean/dict.txt", "", ModelFamily::V5Lang, kV6DetConfig},
    {"thai",   "models/rec/thai/rec.onnx",   "models/rec/thai/dict.txt",   "", ModelFamily::V5Lang, kV6DetConfig},
    {"greek",  "models/rec/greek/rec.onnx",  "models/rec/greek/dict.txt",  "", ModelFamily::V5Lang, kV6DetConfig},
}};

// Look up a model by name. Returns nullptr if no entry matches.
[[nodiscard]] inline constexpr const ModelEntry *find_model(
    std::string_view name) noexcept {
  for (const auto &e : kModelCatalog)
    if (e.name == name) return &e;
  return nullptr;
}

// The default model must exist in the catalog: resolve_model and
// lang_alias_entry dereference find_model(kDefaultModel) unconditionally.
static_assert(find_model(kDefaultModel) != nullptr,
              "kDefaultModel is not present in kModelCatalog");

// Resolve an entry's detector: its own det_path, or the default detector when
// empty. Always a valid non-empty path.
[[nodiscard]] inline constexpr std::string_view model_det_path(
    const ModelEntry &e) noexcept {
  return e.det_path.empty() ? kDefaultDet : e.det_path;
}

} // namespace turbo_ocr::server
