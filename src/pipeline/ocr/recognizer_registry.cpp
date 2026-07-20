#include "recognizer_registry.h"

#include <filesystem>
#include <iostream>
#include <string_view>
#include <utility>

#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/table/table_recognizer.h"

namespace turbo_ocr::pipeline {

bool load_formula_into_registry(FormulaRegistry &registry,
                                formula::IFormulaRecognizer *&default_out,
                                const backend_routing::RoutingTable &routing,
                                const std::string &formula_onnx,
                                const std::string &formula_tokenizer_json,
                                const EnsureStreamFn &ensure_formula_stream) {
  const backend_routing::BackendSpec *fspec = backend_routing::resolve(routing, "formula");
  // "remote" = needs no local ONNX bundle: an explicit openai endpoint OR the
  // env `vlm` engine (kind:Local engine="vlm" -> the VLMFormula HTTP client).
  const bool formula_is_remote =
      fspec && (fspec->kind == backend_routing::Kind::Openai || fspec->engine == "vlm");
  // Non-owning view into the routing table (outlives this call) or a literal;
  // used only for diagnostics, so no std::string copy is warranted.
  const std::string_view formula_backend =
      !fspec ? std::string_view{}
             : (formula_is_remote ? std::string_view{"openai"}
                                   : std::string_view{fspec->engine});
  const bool want_formula =
      formula_is_remote ||
      (!formula_onnx.empty() && !formula_tokenizer_json.empty() &&
       std::filesystem::exists(formula_onnx) &&
       std::filesystem::exists(formula_tokenizer_json));
  if (!want_formula) {
    // A LOCAL bundle that isn't on disk yet is a tolerated skip (the server
    // still boots while upstream re-exports the ONNX), but it must never be
    // SILENT: announce every skip of a routed backend so an operator can never
    // mistake a missing bundle for a working formula stage.
    if (!formula_onnx.empty())
      std::cout << "[Pipeline] Formula engine skipped (path missing): "
                << formula_onnx << '\n';
    else if (fspec)
      std::cout << "[Pipeline] Formula engine skipped (backend '" << fspec->name
                << "' routed but no local model bundle configured)\n";
    return true;  // nothing loadable — tolerated, but announced above
  }
  auto eng = fspec ? formula::make_formula_recognizer(*fspec) : nullptr;
  if (!eng) {
    std::cerr << "[Pipeline] FATAL: unknown formula backend '" << formula_backend
              << "'\n";
    return false;  // explicitly configured but unbuildable -> abort boot
  }
  const bool loaded = eng->load_model_dir(formula_onnx) &&
                      eng->load_tokenizer(formula_tokenizer_json);
  if (!loaded && !formula_is_remote) {
    // A configured LOCAL engine that won't load (out-of-memory, missing/bad
    // model) must NEVER silently disable formulas — fail the boot loudly so the
    // operator sees it instead of getting a server that returns no formulas.
    std::cerr << "[Pipeline] FATAL: local formula backend '" << formula_backend
              << "' failed to load (out-of-memory or bad model) — refusing to "
                 "start with formulas silently disabled\n";
    return false;
  }
  // Registry OWNS the recognizer (unique_ptr moved in); default_out is a
  // non-owning pointer to this route-default entry, keyed by the config backend
  // NAME so a per-request override can address it; the hot path uses default_out.
  // insert_or_assign hands back the slot iterator in one lookup, so ownership
  // transfer and the pointer we hand out are read from the same owned object.
  auto &slot = registry.insert_or_assign(fspec->name, std::move(eng)).first->second;
  default_out = slot.get();
  const std::string_view bname = slot->backend_name();
  ensure_formula_stream();
  if (loaded) {
    std::cout << "[Pipeline] Formula stage enabled (backend=" << bname
              << ", name=" << fspec->name << ")\n";
  } else {
    // Remote endpoint unreachable at boot: keep it addressable (it may recover),
    // but make the degradation LOUD — requests will error (never a silent no-op).
    std::cerr << "[Pipeline] ERROR: remote formula backend '" << fspec->name
              << "' is NOT reachable at boot; registered but requests error until "
                 "it recovers (no silent fallback to a default model)\n";
  }
  return true;
}

bool load_table_into_registry(TableRegistry &registry,
                              table::ITableRecognizer *&default_out,
                              const backend_routing::RoutingTable &routing,
                              const EnsureStreamFn &ensure_table_stream) {
  const backend_routing::BackendSpec *tspec = backend_routing::resolve(routing, "table");
  if (!tspec) return true;  // tables unrouted -> geometric fallback, not a failure
  const bool is_remote =
      tspec->kind == backend_routing::Kind::Openai || tspec->engine == "vlm";
  auto t = table::make_table_recognizer(*tspec);
  if (!t) {
    std::cerr << "[Pipeline] FATAL: unknown table backend '" << tspec->engine << "'\n";
    return false;  // configured but unbuildable -> abort boot
  }
  const bool loaded = t->load();
  if (!loaded && !is_remote) {
    // Configured LOCAL table backend that won't load (OOM, bad model) must never
    // silently disable tables — fail the boot loudly.
    std::cerr << "[Pipeline] FATAL: local table backend '" << tspec->engine
              << "' failed to load (out-of-memory or bad model) — refusing to "
                 "start with tables silently disabled\n";
    return false;
  }
  // Registry OWNS (unique_ptr moved in); default_out is a non-owning pointer to
  // the route default, keyed by config NAME so a per-request override can address
  // it. Single-lookup insert_or_assign: the pointer we hand out comes straight
  // from the owned slot.
  auto &slot = registry.insert_or_assign(tspec->name, std::move(t)).first->second;
  default_out = slot.get();
  ensure_table_stream();
  if (!loaded)
    std::cerr << "[Pipeline] ERROR: remote table backend '" << tspec->name
              << "' is NOT reachable at boot; registered but requests error until "
                 "it recovers (no silent fallback)\n";
  return true;
}

void prewarm_openai_into_registry(const std::string &modality,
                                  const backend_routing::RoutingTable &tbl,
                                  TableRegistry &table_registry,
                                  FormulaRegistry &formula_registry,
                                  const EnsureStreamFn &ensure_table_stream,
                                  const EnsureStreamFn &ensure_formula_stream) {
  for (const auto &[name, spec] : tbl.backends) {
    if (spec.kind != backend_routing::Kind::Openai) continue;
    if (modality == "formula") {
      if (formula_registry.count(name)) continue;        // already the default
      auto eng = formula::make_formula_recognizer(spec);
      if (!eng) continue;
      // health_check() runs inside load_model_dir; register regardless of its
      // result so the override name is always addressable (a transiently-down
      // endpoint reports per-region failures rather than silently routing to
      // the default model, which would surprise the operator). load_tokenizer
      // is a no-op for the remote backend.
      const bool ready = eng->load_model_dir("") && eng->load_tokenizer("");
      formula_registry[name] = std::move(eng);
      ensure_formula_stream();
      std::cout << "[Pipeline] Formula override backend registered (name=" << name
                << ", ready=" << (ready ? "yes" : "no") << ")\n";
    } else {  // table
      if (table_registry.count(name)) continue;
      auto t = table::make_table_recognizer(spec);
      if (!t) continue;
      const bool ready = t->load();
      table_registry[name] = std::move(t);
      ensure_table_stream();
      std::cout << "[Pipeline] Table override backend registered (name=" << name
                << ", ready=" << (ready ? "yes" : "no") << ")\n";
    }
  }
}

} // namespace turbo_ocr::pipeline
