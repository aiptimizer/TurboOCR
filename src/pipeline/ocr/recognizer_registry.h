#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "turbo_ocr/backend_routing/routing_config.h"

namespace turbo_ocr::table   { class ITableRecognizer; }
namespace turbo_ocr::formula { class IFormulaRecognizer; }

namespace turbo_ocr::pipeline {

// Per-modality recognizer registries owned by OcrPipeline: backend NAME ->
// recognizer. Aliased here so the construction helpers below and the pipeline
// agree on the exact map type.
using TableRegistry =
    std::map<std::string, std::unique_ptr<table::ITableRecognizer>>;
using FormulaRegistry =
    std::map<std::string, std::unique_ptr<formula::IFormulaRecognizer>>;

// CUDA stream/event lifetime stays in OcrPipeline (it owns and destroys them);
// these construction helpers only need to *ensure* the relevant stream/event
// exists before they register a backend that will run on it. The pipeline
// passes a callback that creates them lazily on first use.
using EnsureStreamFn = std::function<void()>;

// Build the route-default formula recognizer (the "formula half" of
// OcrPipeline::load_router_models) and register it under its config NAME.
// On success, `default_out` is set and `ensure_formula_stream` is invoked.
// Returns FALSE only when an explicitly-configured LOCAL backend fails to load
// (out-of-memory / bad model) — the caller MUST abort boot rather than serve a
// silently formula-less pipeline. Returns true when nothing is configured, when
// the local backend loads, or when a remote backend is registered-but-unreachable
// (logged loudly; it may recover). Never silently disables a configured backend.
[[nodiscard]] bool
load_formula_into_registry(FormulaRegistry &registry,
                           formula::IFormulaRecognizer *&default_out,
                           const backend_routing::RoutingTable &routing,
                           const std::string &formula_onnx,
                           const std::string &formula_tokenizer_json,
                           const EnsureStreamFn &ensure_formula_stream);

// Build the route-default table recognizer (body of
// OcrPipeline::load_table_backend) and register it under its config NAME.
// Returns FALSE only when an explicitly-configured LOCAL backend fails to load
// (caller MUST abort boot). Returns true when tables are unrouted (geometric
// fallback), the local backend loads, or a remote backend is registered-but-
// unreachable (logged loudly). On success `default_out` is set and
// `ensure_table_stream` is invoked.
[[nodiscard]] bool
load_table_into_registry(TableRegistry &registry,
                         table::ITableRecognizer *&default_out,
                         const backend_routing::RoutingTable &routing,
                         const EnsureStreamFn &ensure_table_stream);

// Prewarm every kind:openai backend (for `modality` == "formula"|"table") that
// isn't already the route default, so per-request overrides can target them
// with no first-request build latency. Mirrors the previous
// OcrPipeline::prewarm_openai_registry_ exactly (registers regardless of the
// health-check result so the override name is always addressable).
void prewarm_openai_into_registry(const std::string &modality,
                                  const backend_routing::RoutingTable &tbl,
                                  TableRegistry &table_registry,
                                  FormulaRegistry &formula_registry,
                                  const EnsureStreamFn &ensure_table_stream,
                                  const EnsureStreamFn &ensure_formula_stream);

} // namespace turbo_ocr::pipeline
