#pragma once

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/server/model_catalog.h"

namespace turbo_ocr::server {

struct RecPaths {
  std::string rec;
  std::string dict;
};

// Fully resolved model selection: the recognizer + dictionary + detector
// paths and the registry name they came from. `name` is ALWAYS the selected
// registry entry's name (never empty) — an explicit per-stage env override
// (REC_ONNX/DET_ONNX/REC_DICT) replaces that stage's path but leaves `name`
// pointing at the registry entry so logging/telemetry stays truthful.
//
// `det_cfg` is the per-model OFFICIAL detection inference base (resize policy +
// DB params). It is a BASE: env per-field overrides are applied at detector
// construction via read_det_resize(det_cfg.resize) / read_db_params(det_cfg.db),
// so env always wins over the per-model value. When an explicit DET_ONNX
// override bypasses the registry detector, det_cfg holds the official defaults
// (kDetResizeDefault / kDbDefaults).
struct ResolvedModel {
  std::string name;
  std::string rec;
  std::string dict;
  std::string det;
  DetInferConfig det_cfg{};
};

namespace detail {

// Build the comma-separated list of valid OCR_MODEL names for diagnostics.
inline std::string model_names_csv() {
  std::string s;
  for (const auto& e : kModelCatalog) {
    if (!s.empty()) s += ", ";
    s += std::string(e.name);
  }
  return s;
}

// Map a deprecated OCR_LANG value onto a registry name. Non-Latin scripts
// with a dedicated entry select that entry; everything else (latin, "",
// chinese, japanese, en, …) maps onto the default v6 model (kDefaultModel).
inline const ModelEntry& lang_alias_entry(const std::string& lang) {
  if (const ModelEntry* e = find_model(lang); e && e->family == ModelFamily::V5Lang) return *e;
  return *find_model(kDefaultModel);
}

}  // namespace detail

// Resolve the OCR model: recognizer, dictionary, AND detector paths plus the
// selected registry name.
//
// Precedence (highest first):
//   1. Explicit per-variable env overrides always win, independently:
//      `rec_env` (REC_ONNX/REC_MODEL), `dict_env` (REC_DICT), `det_env`
//      (DET_ONNX/DET_MODEL). API-unchanged behavior.
//   2. `OCR_MODEL=<name>` — registry lookup. An unknown name is fatal: it
//      pushes an error listing the valid names (when `errors` is provided).
//   3. `OCR_LANG=<lang>` — DEPRECATED alias (warns when `warnings` is
//      provided): {arabic,eslav,korean,thai,greek} select that entry;
//      anything else maps to the default. OCR_MODEL takes precedence.
//   4. Default — kDefaultModel ("tiny").
//
// Each non-override field falls back to the resolved registry entry, so an
// override on one stage never disturbs the others (e.g. DET_ONNX alone still
// wins while rec/dict come from the selected model). The detector default is
// the entry's det_path (per-tier for v6) or `models/det.onnx` when empty (the
// V5Lang rows, which carry no detector of their own).
//
// `det_cfg` follows the detector path: the entry's per-model official config,
// or the official defaults ({kDetResizeDefault, kDbDefaults}) when an explicit
// DET_ONNX/DET_MODEL override supplies a detector outside the registry.
//
// `errors`/`warnings` are optional; pass nullptr from callers that resolve
// outside the ServerConfig strict-validation path (CLI, Python binding),
// which then silently fall back to the default on an unknown OCR_MODEL.
[[nodiscard]] inline ResolvedModel resolve_model(const char* rec_env, const char* det_env,
                                                 const char* dict_env = "REC_DICT",
                                                 std::vector<std::string>* errors = nullptr,
                                                 std::vector<std::string>* warnings = nullptr) {
  // Always resolves to a valid entry (default on unknown OCR_MODEL), so
  // `entry->name` is the single source of truth for the selected name.
  const ModelEntry* entry = find_model(kDefaultModel);

  if (const char* model_c = std::getenv("OCR_MODEL"); model_c && *model_c) {
    if (const ModelEntry* e = find_model(model_c))
      entry = e;
    else if (errors)
      errors->push_back(std::string("OCR_MODEL=\"") + model_c +
                        "\" is not a known model (valid: " + detail::model_names_csv() + ")");
  } else if (const char* lang_c = std::getenv("OCR_LANG"); lang_c && *lang_c) {
    entry = &detail::lang_alias_entry(lang_c);
    if (warnings)
      warnings->push_back(std::string("OCR_LANG is deprecated; use OCR_MODEL. OCR_LANG=\"") +
                          lang_c + "\" resolved to model \"" + std::string(entry->name) + "\"");
  }

  // An explicit DET override supplies a detector outside the registry, so its
  // per-model config no longer applies — fall back to the official defaults.
  const bool det_overridden = env::env_present(det_env);

  return ResolvedModel{
      .name = std::string(entry->name),
      .rec = env::env_or(rec_env, entry->rec_path),
      .dict = env::env_or(dict_env, entry->dict_path),
      .det = env::env_or(det_env, model_det_path(*entry)),
      .det_cfg = det_overridden
                     ? DetInferConfig{detection::kDetResizeDefault, detection::kDbDefaults}
                     : entry->det_cfg,
  };
}

// Back-compat shim for call sites that only need rec+dict and resolve the
// detector themselves. Delegates to resolve_model with the det override key
// matching the rec key's profile (REC_ONNX->DET_ONNX, REC_MODEL->DET_MODEL).
[[nodiscard]] inline RecPaths resolve_rec_paths(const char* rec_env,
                                                const char* dict_env = "REC_DICT") {
  const bool is_gpu = std::string_view(rec_env) == "REC_ONNX";
  ResolvedModel m = resolve_model(rec_env, is_gpu ? "DET_ONNX" : "DET_MODEL", dict_env);
  return RecPaths{.rec = std::move(m.rec), .dict = std::move(m.dict)};
}

} // namespace turbo_ocr::server
