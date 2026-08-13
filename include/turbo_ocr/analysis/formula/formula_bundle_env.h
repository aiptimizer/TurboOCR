#pragma once

#include <string>

#include "turbo_ocr/base/env_utils.h"

// Resolve the LOCAL formula model bundle from the environment — the ONE policy
// for every server flavour.
//
// The rule (unchanged since v3.x): FORMULA_ONNX / FORMULA_TOKENIZER win when
// set; otherwise `FORMULA_BACKEND=ppformulanet_s|ppformulanet_plus_s|
// ppformulanet_plus_m|auto` alone resolves the bundle the release ships under
// models/formula/, so the documented one-knob setup works out of the box.
//
// SHARED on purpose. This resolution briefly lived only in the GPU pool
// (gpu_pipeline_pool.h), so `FORMULA_BACKEND=ppformulanet_s` loaded formulas on
// the GPU server but left the unified server advertising formulas unavailable
// and rejecting ?formulas=1 with FORMULA_BACKEND_DISABLED — two binaries
// disagreeing on the same env. Bundle-path policy is generic; keep it here.
namespace turbo_ocr::formula {

struct FormulaBundlePaths {
  std::string model_dir;  // empty => no local bundle configured
  std::string tokenizer;  // empty => no tokenizer configured
};

[[nodiscard]] inline FormulaBundlePaths resolve_formula_bundle_env() {
  const auto read = [](const char *k) { return env::env_or(k, ""); };
  FormulaBundlePaths out{read("FORMULA_ONNX"), read("FORMULA_TOKENIZER")};
  if (out.model_dir.empty()) {
    const std::string fb = read("FORMULA_BACKEND");
    if (fb == "ppformulanet_s" || fb == "ppformulanet_plus_m")
      out.model_dir = "models/formula/" + fb;
    else if (fb == "ppformulanet_plus_s" || fb == "auto")
      // "ppformulanet_plus_s" is the accurate alias for the shipped fast
      // bundle: models/formula/ppformulanet_s keeps its historical dir name
      // but its weights ARE PP-FormulaNet_plus-S (byte-identical rebuild from
      // paddle's official plus-S download, 2026-08-05). `auto` starts from the
      // same bundle; AutoCjkFormula resolves the plus-M sibling itself.
      out.model_dir = "models/formula/ppformulanet_s";
  }
  if (out.tokenizer.empty() && !out.model_dir.empty())
    out.tokenizer = out.model_dir + "/tokenizer.json";
  return out;
}

} // namespace turbo_ocr::formula
