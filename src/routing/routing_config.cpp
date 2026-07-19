#include "turbo_ocr/routing/routing_config.h"

#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/logger.h"

#include <cstdlib>
#include <fstream>
#include <array>
#include <iostream>

#include <nlohmann/json.hpp>

namespace turbo_ocr::routing {

namespace {

using nlohmann::json;

using turbo_ocr::env::env_or;

// Valid local engine keys per modality (mirrors the factories; kept as a
// static list so this module stays free of the GPU recognizer libs).
// "formulanet" is retained as a recognized config name for back-compat (so an
// old routing file still parses), but the factory no longer constructs it —
// resolving FORMULA_BACKEND=formulanet yields a null recognizer (unavailable).
// ppformulanet_s     = small 2-layer model, in-process C++ FAST host-loop, ~10 pg/s,
//                      English/Latin only (the default — fastest).
// ppformulanet_plus_m = 6-layer MBart model, in-process C++ FAST host-loop, the only
//                      backend that reads CHINESE formulas (CJK-F1 0.73); slower.
// auto                = composite: -S over all crops, plus-M re-run on the crops
//                      whose -S output shows CJK it mangled. EN crops keep -S
//                      speed; CJK crops get plus-M accuracy. GPU build only.
// Swap with FORMULA_BACKEND + point FORMULA_ONNX/FORMULA_TOKENIZER at the chosen model.
const std::array<const char *, 5> kFormulaEngines{"formulanet", "ppformulanet_s",
                                                  "ppformulanet_plus_m", "vlm",
                                                  "auto"};
const std::array<const char *, 2> kTableEngines{"slanext", "vlm"};

bool engine_valid_for(const std::string &modality, const std::string &engine) {
  if (modality == "formula") {
    for (const char *e : kFormulaEngines)
      if (engine == e) return true;
    return false;
  }
  for (const char *e : kTableEngines)
    if (engine == e) return true;
  return false;
}

// api_key is accepted ONLY as "env:NAME"; resolve NAME from the environment.
// Throws ROUTING_MISSING_SECRET when the named var is unset.
std::string resolve_api_key(const std::string &name, const std::string &raw) {
  if (raw.empty()) return "";
  if (raw.rfind("env:", 0) != 0)
    throw RoutingConfigError("ROUTING_MISSING_SECRET: backend '" + name +
                             "' api_key must use env-indirection ('env:NAME'), "
                             "not a literal secret");
  const std::string var = raw.substr(4);
  // Secret indirection reads the NAMED variable's raw value — env_or's
  // default-substitution semantics would mask an unset secret.
  const std::string v = env_or(var.c_str(), "");
  if (v.empty())
    throw RoutingConfigError("ROUTING_MISSING_SECRET: backend '" + name +
                             "' api_key env '" + var + "' is unset");
  return v;
}

// Zero-config: synthesize the table from env so behavior == today.
RoutingTable synth_from_env() {
  RoutingTable t;

  // --- formula ---
  // env-only `vlm` resolves to kind:Local engine="vlm" so the string factory
  // builds the TESTED VLMFormula client (byte-identical to before this module).
  // The generic OpenAIEndpoint is reached ONLY via an explicit
  // TURBO_ROUTING_CONFIG kind:openai entry — opt-in, never the env default.
  // Default to the WORKING ORT-sidecar backend. The legacy "formulanet" backend
  // (single fused-Loop TRT engine, inert on TRT 10.15) has been removed: its
  // decode never ran and it would have silently disabled formula out of the box
  // (violates the no-silent-failure contract). The factory no longer constructs
  // it, so an explicit FORMULA_BACKEND=formulanet now resolves to a null
  // recognizer (formula unavailable) instead of serving silent-empty.
  // Default = the fast English/Latin -S model. FORMULA_BACKEND=ppformulanet_plus_m
  // swaps in the slower Chinese-capable model (point FORMULA_ONNX/FORMULA_TOKENIZER at it).
  std::string fb = env_or("FORMULA_BACKEND", "ppformulanet_s");
  if (!engine_valid_for("formula", fb))
    // A typo'd FORMULA_BACKEND must not silently disable formula (the env path
    // previously skipped this validation the config-file path enforces).
    TOCR_LOG_ERROR("FORMULA_BACKEND is not a recognized formula engine, formula DISABLED",
                   "value", fb,
                   "expected", "ppformulanet_s|ppformulanet_plus_m|auto|vlm");
  {
    BackendSpec b; b.name = "formula-env"; b.kind = Kind::Local; b.engine = fb;
    t.backends["formula-env"] = b;
    t.routes["formula"] = "formula-env";
  }

  // --- table (mirror resolve_table_backend_env's inference) ---
  // Route to an ML table backend ONLY when one is actually configured; otherwise
  // leave table on "default" (the geometric line-art fallback). An unconfigured
  // slanext default would make load_table_backend attempt a model that isn't there
  // and — with the fail-loud load contract — abort boot; a config that never asked
  // for ML tables must NOT fatal (and the geometric fallback still produces tables).
  std::string tb = env_or("TABLE_BACKEND", "");
  if (tb.empty()) {
    if (!env_or("TABLE_SLANEXT_ENCODER_ONNX", "").empty()) tb = "slanext";
    else if (!env_or("VLLM_TABLE_BASE_URL", "").empty())   tb = "vlm";
  }
  // env-only `vlm` → kind:Local engine="vlm" (tested VLMTable via the string
  // factory); OpenAIEndpoint only via explicit kind:openai config (see formula).
  if (!tb.empty() && !engine_valid_for("table", tb))
    TOCR_LOG_ERROR("TABLE_BACKEND is not a recognized table engine, geometric fallback",
                   "value", tb, "expected", "slanext|vlm");
  if (!tb.empty()) {
    BackendSpec b; b.name = "table-env"; b.kind = Kind::Local; b.engine = tb;
    t.backends["table-env"] = b;
    t.routes["table"] = "table-env";
  } else {
    t.routes["table"] = "default";  // geometric line-art fallback; no ML backend loaded
  }

  // --- text: built-in det/cls/rec path ---
  t.routes["text"] = "default";
  return t;
}

BackendSpec parse_backend(const std::string &name, const json &j) {
  BackendSpec b;
  b.name = name;
  // Every nlohmann accessor below (value<T>/get<T>) throws json::type_error on a
  // wrong-typed field (e.g. {"kind":123} or {"max_tokens":"x"}). Map those to a
  // RoutingConfigError so a malformed /infer body is a clean 400, never a crash.
  // RoutingConfigError derives from std::runtime_error (not json::exception), so
  // the explicit ROUTING_* throws inside propagate undisturbed.
  try {
    const std::string kind = j.value("kind", "");
    if (kind == "local")        b.kind = Kind::Local;
    else if (kind == "openai")  b.kind = Kind::Openai;
    else
      throw RoutingConfigError("ROUTING_BAD_KIND: backend '" + name +
                               "' has unknown kind '" + kind +
                               "' (expected local|openai)");

    if (b.kind == Kind::Local) {
      b.engine     = j.value("engine", "");
      b.model_path = j.value("model_path", "");
      if (b.engine.empty())
        throw RoutingConfigError("ROUTING_UNKNOWN_ENGINE: backend '" + name +
                                 "' kind=local requires an 'engine' key");
    } else { // Openai
      b.base_url = j.value("base_url", "");
      b.model    = j.value("model", "");
      b.prompt   = j.value("prompt", "");
      if (b.base_url.empty())
        throw RoutingConfigError("ROUTING_UNKNOWN_ENGINE: backend '" + name +
                                 "' kind=openai requires 'base_url'");
      if (b.model.empty())
        throw RoutingConfigError("ROUTING_UNKNOWN_ENGINE: backend '" + name +
                                 "' kind=openai requires 'model'");
      if (b.prompt.empty())
        throw RoutingConfigError("ROUTING_EMPTY_PROMPT: backend '" + name +
                                 "' kind=openai requires a non-empty 'prompt' "
                                 "(e.g. 'Table Recognition:' / 'Formula Recognition:')");
      const std::string ps = j.value("parser", "raw");
      if (!parse_parser(ps, b.parser))
        throw RoutingConfigError("ROUTING_BAD_KIND: backend '" + name +
                                 "' parser '" + ps +
                                 "' (expected otsl|latex|text|raw)");
      b.api_key    = resolve_api_key(name, j.value("api_key", ""));
      b.max_tokens = j.value("max_tokens", 1024);
      b.timeout_s  = j.value("timeout_s", 30);
      b.batch      = j.value("batch", 8);
    }
    if (j.contains("fallback") && j["fallback"].is_string())
      b.fallback = j["fallback"].get<std::string>();
  } catch (const nlohmann::json::exception &e) {
    throw RoutingConfigError(std::string("ROUTING_BAD_KIND: invalid backend field in '") +
                             name + "': " + e.what());
  }
  return b;
}

RoutingTable parse_file(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw RoutingConfigError("ROUTING_BAD_KIND: cannot open TURBO_ROUTING_CONFIG '" +
                             path + "'");
  json root;
  try {
    f >> root;
  } catch (const std::exception &e) {
    throw RoutingConfigError(std::string("ROUTING_BAD_KIND: invalid JSON in '") +
                             path + "': " + e.what());
  }

  RoutingTable t;
  if (root.contains("backends") && root["backends"].is_object())
    for (auto it = root["backends"].begin(); it != root["backends"].end(); ++it)
      t.backends[it.key()] = parse_backend(it.key(), it.value());

  if (root.contains("routes") && root["routes"].is_object())
    for (auto it = root["routes"].begin(); it != root["routes"].end(); ++it)
      if (it.value().is_string())
        t.routes[it.key()] = it.value().get<std::string>();

  // --- validation ---
  for (const auto &[modality, bname] : t.routes) {
    if (bname == "default") continue;
    if (modality == "text")
      throw RoutingConfigError("ROUTING_TEXT_NOT_ROUTABLE: route 'text' -> '" + bname +
                               "' but text uses the built-in det/cls/rec path; "
                               "only 'default' is valid for 'text'");
    auto bit = t.backends.find(bname);
    if (bit == t.backends.end())
      throw RoutingConfigError("ROUTING_DANGLING_REF: route '" + modality +
                               "' -> '" + bname + "' which is not in backends{}");
    const BackendSpec &b = bit->second;
    if (b.kind == Kind::Local && (modality == "formula" || modality == "table") &&
        !engine_valid_for(modality, b.engine))
      throw RoutingConfigError("ROUTING_MODALITY_MISMATCH: backend '" + bname +
                               "' engine '" + b.engine + "' is not a " + modality +
                               " backend");
  }

  // Fill any modality the file omitted from env (so a partial config still
  // yields a complete, today-compatible table).
  RoutingTable env = synth_from_env();
  for (const char *m : {"text", "table", "formula"}) {
    if (t.routes.find(m) == t.routes.end()) {
      t.routes[m] = env.routes[m];
      // pull the env-synth backend in if referenced
      auto er = env.routes[m];
      if (er != "default" && t.backends.find(er) == t.backends.end())
        t.backends[er] = env.backends[er];
    }
  }
  return t;
}

} // namespace

BackendSpec parse_inline_backend(const std::string &modality,
                                 const std::string &json_text) {
  json j;
  try {
    j = json::parse(json_text);
  } catch (const std::exception &e) {
    throw RoutingConfigError(std::string("ROUTING_BAD_KIND: invalid inline backend JSON: ") +
                             e.what());
  }
  // parse_backend is now internally exception-safe; the extra guard here keeps
  // any future nlohmann throw from escaping the route's RoutingConfigError catch.
  BackendSpec b;
  try {
    b = parse_backend("inline", j);
  } catch (const nlohmann::json::exception &e) {
    throw RoutingConfigError(std::string("ROUTING_BAD_KIND: invalid inline backend field: ") +
                             e.what());
  }
  if (b.kind == Kind::Local && (modality == "formula" || modality == "table") &&
      !engine_valid_for(modality, b.engine))
    throw RoutingConfigError("ROUTING_MODALITY_MISMATCH: inline backend engine '" +
                             b.engine + "' is not a " + modality + " backend");
  return b;
}

std::set<std::string> routable_backend_names(const RoutingTable &tbl,
                                             const std::string &modality) {
  std::set<std::string> names;
  for (const auto &[name, spec] : tbl.backends)
    if (spec.kind == Kind::Openai) names.insert(name);
  auto r = tbl.routes.find(modality);
  if (r != tbl.routes.end() && r->second != "default")
    names.insert(r->second);
  return names;
}

const char *kind_name(Kind k) noexcept {
  switch (k) { case Kind::Local: return "local"; case Kind::Openai: return "openai"; }
  return "?";
}
const char *parser_name(Parser p) noexcept {
  switch (p) {
    case Parser::Otsl: return "otsl"; case Parser::Latex: return "latex";
    case Parser::Text: return "text"; case Parser::Raw:   return "raw";
  }
  return "?";
}
bool parse_parser(const std::string &s, Parser &out) noexcept {
  if (s == "otsl")  { out = Parser::Otsl;  return true; }
  if (s == "latex") { out = Parser::Latex; return true; }
  if (s == "text")  { out = Parser::Text;  return true; }
  if (s == "raw" || s.empty()) { out = Parser::Raw; return true; }
  return false;
}

RoutingTable load_routing_config() {
  const std::string cfg = env_or("TURBO_ROUTING_CONFIG", "");
  if (!cfg.empty()) return parse_file(cfg);
  return synth_from_env();
}

const BackendSpec *resolve(const RoutingTable &tbl, const std::string &modality,
                           const std::string &override_name) {
  // precedence: per-request override > routes[modality] > nullptr(default)
  if (!override_name.empty()) {
    auto it = tbl.backends.find(override_name);
    if (it == tbl.backends.end())
      throw RoutingConfigError("ROUTING_UNKNOWN_OVERRIDE: per-request route override '" +
                               override_name + "' names no configured backend");
    return &it->second;
  }
  auto r = tbl.routes.find(modality);
  if (r == tbl.routes.end() || r->second == "default") return nullptr;
  auto b = tbl.backends.find(r->second);
  return (b != tbl.backends.end()) ? &b->second : nullptr;
}

} // namespace turbo_ocr::routing
