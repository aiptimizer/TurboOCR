// UnifiedOcrPipeline — optional-stage bootstrap.
//
// Builds the router + table/formula registries from the routing config, using
// the BACKEND's device-appropriate recognizer factories (the true dedup: no
// free-function factory per backend). No-silent-failure load semantics: a
// stage that was asked for and could not be loaded says so, rather than leaving
// a null the request path rediscovers as an empty result.
//
// Split out of unified_ocr_pipeline.cpp, which had reached 1000 lines — over the
// 900-line ceiling tools/checks/architecture.sh enforces. The seams were already
// named by the banner comments in that file; this is those seams made physical,
// not a new decomposition. All four TUs define members of the SAME class, so the
// header is unchanged and nothing outside this directory can tell the difference.

#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/base/log/logger.h"               // TOCR_LOG_*
#include "turbo_ocr/analysis/formula/formula_bundle_env.h"      // resolve_formula_bundle_env
#include "turbo_ocr/base/geometry/box.h"             // Box, sorted_boxes, is_vertical_box
#include "turbo_ocr/core/types.h"                    // OCRResultItem, kDropScore

namespace turbo_ocr::pipeline {


using ::turbo_ocr::Box;
using ::turbo_ocr::OCRResultItem;

bool UnifiedOcrPipeline::load_router_models() {
  if (!router_) router_ = std::make_unique<router::CuaRouter>();
  return load_formula_model();
}

bool UnifiedOcrPipeline::load_formula_model() {
  const backend_routing::RoutingTable routing =
      backend_routing::load_routing_config();
  const backend_routing::BackendSpec *fspec =
      backend_routing::resolve(routing, "formula");
  if (!fspec) return true;  // formula unrouted -> not a failure
  const bool is_remote =
      fspec->kind == backend_routing::Kind::Openai || fspec->engine == "vlm";

  // Local bundle resolution: the spec's model_path wins, else the ONE shared
  // env policy (formula_bundle_env.h) — FORMULA_ONNX/FORMULA_TOKENIZER, with
  // `FORMULA_BACKEND=ppformulanet_s|plus_m|auto` alone resolving the baked
  // models/formula/ bundle exactly as the GPU pool and the v3.5.0 CPU server
  // did. A LOCAL engine with no bundle configured or a bundle not on disk is a
  // TOLERATED skip — announced, never silent, never fatal. Only a
  // configured-and-present local bundle that fails to LOAD aborts boot.
  const auto bundle = formula::resolve_formula_bundle_env();
  std::string model_dir = fspec->model_path;
  if (model_dir.empty()) model_dir = bundle.model_dir;
  const std::string tokenizer = bundle.tokenizer;
  if (!is_remote) {
    const bool bundle_ok = !model_dir.empty() && !tokenizer.empty() &&
                           std::filesystem::exists(model_dir) &&
                           std::filesystem::exists(tokenizer);
    if (!bundle_ok) {
      if (!model_dir.empty())
        std::cout << "[Unified] Formula engine skipped (path missing): "
                  << model_dir << '\n';
      else
        std::cout << "[Unified] Formula engine skipped (backend '" << fspec->name
                  << "' routed but no local model bundle configured)\n";
      return true;
    }
  }
  auto eng = backend_.make_formula_recognizer(*fspec);
  if (!eng) {
    std::cerr << "[Unified] FATAL: unknown formula backend '" << fspec->engine
              << "'\n";
    return false;
  }
  const bool loaded = eng->load_model_dir(model_dir) &&
                      eng->load_tokenizer(tokenizer);
  if (!loaded && !is_remote) {
    std::cerr << "[Unified] FATAL: local formula backend '" << fspec->engine
              << "' failed to load — refusing to start with formulas silently "
                 "disabled\n";
    return false;
  }
  auto &slot =
      formula_registry_.insert_or_assign(fspec->name, std::move(eng)).first->second;
  formula_ = slot.get();
  if (!loaded)
    std::cerr << "[Unified] ERROR: remote formula backend '" << fspec->name
              << "' not reachable at boot; registered but requests error until "
                 "it recovers\n";
  return true;
}

bool UnifiedOcrPipeline::load_table_backend() {
  const backend_routing::RoutingTable routing =
      backend_routing::load_routing_config();
  const backend_routing::BackendSpec *tspec =
      backend_routing::resolve(routing, "table");
  if (!tspec) return true;  // tables unrouted -> geometric fallback
  const bool is_remote =
      tspec->kind == backend_routing::Kind::Openai || tspec->engine == "vlm";
  auto t = backend_.make_table_recognizer(*tspec);
  if (!t) {
    std::cerr << "[Unified] FATAL: unknown table backend '" << tspec->engine
              << "'\n";
    return false;
  }
  const bool loaded = t->load();
  if (!loaded && !is_remote) {
    std::cerr << "[Unified] FATAL: local table backend '" << tspec->engine
              << "' failed to load — refusing to start with tables silently "
                 "disabled\n";
    return false;
  }
  // Per-cell crop OCR for grid cells the page detector under-segmented.
  if (rec_) t->set_cell_recognizer(rec_.get());
  auto &slot =
      table_registry_.insert_or_assign(tspec->name, std::move(t)).first->second;
  table_recognizer_ = slot.get();
  if (!loaded)
    std::cerr << "[Unified] ERROR: remote table backend '" << tspec->name
              << "' not reachable at boot; registered but requests error until "
                 "it recovers\n";
  return true;
}

void UnifiedOcrPipeline::warmup() {
  cv::Mat dummy(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(dummy, cv::Point(10, 30), cv::Point(90, 70), cv::Scalar(0, 0, 0),
                2);
  (void)run(dummy);
}


} // namespace turbo_ocr::pipeline
