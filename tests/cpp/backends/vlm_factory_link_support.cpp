// Link-support for the FUNSD proof-gate binary ONLY.
//
// The Backend seam declares two shared free factories:
//   turbo_ocr::backend::make_table_recognizer(const backend_routing::BackendSpec&)
//   turbo_ocr::backend::make_formula_recognizer(const backend_routing::BackendSpec&)
// CpuBackend::make_table/formula_recognizer call these for the REMOTE (Openai /
// VLM) branch only — local SLANeXt/PP-FormulaNet specs are served directly by
// rebuild/cpu's CpuTableRecognizer / CpuFormulaRecognizerAdapter and never reach
// here.
//
// The real definition of these two overloads lives behind the OpenAI-compatible
// VLM endpoint (turbo_ocr/analysis/vlm/openai_endpoint.*), which pulls in the drogon HTTP
// client stack. That stack is deliberately EXCLUDED from this self-contained
// FUNSD accuracy binary (it exercises det/rec/cls only — no table/formula/router
// routing is ever requested). To give the linker closure for the unused remote
// path WITHOUT dragging drogon into the proof-gate, we provide the overloads here
// as "remote backend unavailable in this binary" (return nullptr).
//
// This affects NO recognition/detection/classification correctness — those are
// the real rebuild stages. It only declines to construct a network VLM backend
// that this offline benchmark never asks for.
//
// NOTE for maintainers: this is NOT the production definition. In the merged
// server_main these two overloads come from the shared VLM factory TU compiled
// with drogon. See REPORT.

#include <memory>

#include "turbo_ocr/backend/formula_recognizer.h"
#include "turbo_ocr/backend/table_recognizer.h"

namespace turbo_ocr::backend_routing { struct BackendSpec; }

namespace turbo_ocr::backend {
std::unique_ptr<ITableRecognizer>
make_table_recognizer(const backend_routing::BackendSpec & /*spec*/) {
  return nullptr; // remote/VLM table backend not linked into the FUNSD proof binary
}
} // namespace turbo_ocr::backend

namespace turbo_ocr::backend {
std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(const backend_routing::BackendSpec & /*spec*/) {
  return nullptr; // remote/VLM formula backend not linked into the FUNSD proof binary
}
} // namespace turbo_ocr::backend
