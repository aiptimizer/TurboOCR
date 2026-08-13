#pragma once

#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

// Config-driven backend routing (design: "Config-driven routing/plugin system").
//
// A declarative table of named backend specs + a modality->backend routing map.
// Each modality (text|table|formula) routes to a named backend that is either
// LOCAL (a factory engine key: formulanet|ppformulanet_s|slanext|vlm)
// or an external OPENAI-compatible endpoint (base_url + model + prompt).
//
// Loaded once at boot from TURBO_ROUTING_CONFIG (a JSON file path). When that
// env var is unset, the table is SYNTHESIZED from the existing env knobs
// (FORMULA_BACKEND / TABLE_BACKEND / VLLM_* ...) so zero-config behavior is
// byte-identical to before this module existed.
//
// This module lives in turbo_ocr_common and has NO dependency on the GPU
// recognizer libs: it only parses/validates the declarative table and reads
// env. The factories (make_*_recognizer) consume a resolved BackendSpec.
//
// Security/open-source posture: api_key is ONLY accepted as an env-indirection
// ("env:NAME"); a literal secret in the file/request is rejected. Pointing an
// openai backend at a proprietary/account-gated endpoint is the operator's
// explicit choice.

namespace turbo_ocr::backend_routing {

enum class Kind { Local, Openai };

// Response decode for an openai endpoint (the only per-modality difference).
enum class Parser { Otsl, Latex, Text, Raw };

struct BackendSpec {
  std::string name;
  Kind        kind = Kind::Local;

  // kind == Local
  std::string engine;      // factory key, validated per modality at load
  std::string model_path;  // optional; empty -> env/default

  // kind == Openai
  std::string base_url;    // http://host:port  (/v1/... appended by the client)
  std::string model;       // server's served-model-name (auto-resolved if empty)
  std::string prompt;      // modality instruction
  Parser      parser = Parser::Raw;
  std::string api_key;     // resolved from env at load; empty == no auth header
  int         max_tokens = 1024;
  int         timeout_s  = 30;
  int         batch      = 8;

  // Parsed for forward compatibility; FAILOVER IS NOT IMPLEMENTED and the
  // loader REFUSES a config that sets it (ROUTING_FALLBACK_UNIMPLEMENTED)
  // rather than silently ignoring an operator's failover plan.
  std::optional<std::string> fallback;
};

struct RoutingTable {
  std::map<std::string, BackendSpec> backends;        // name -> spec
  std::map<std::string, std::string> routes;          // "text"|"table"|"formula" -> backend name | "default"
};

// Per-request routing override: a backend NAME per modality (empty == use the
// configured route default). Threaded from /ocr (JSON `routing{}`) and
// /ocr/raw (`?route_table=`/`?route_formula=`) into run_with_layout. Default-
// constructed (.empty()==true) means "no override", so non-override callers
// (gRPC, PDF, batch) are byte-identical to today.
struct RequestRouting {
  std::string table;
  std::string formula;
  [[nodiscard]] bool empty() const noexcept { return table.empty() && formula.empty(); }
};

// Thrown on invalid config; .what() carries a ROUTING_* coded message.
struct RoutingConfigError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Load + validate. Reads TURBO_ROUTING_CONFIG (JSON file) when set; else
// synthesizes from env (zero-config == today). Throws RoutingConfigError on
// invalid config (fail-fast, like ServerConfig).
RoutingTable load_routing_config();

// Resolve the backend for a modality, honoring an optional per-request
// override NAME (precedence: override > routes[modality] > nullptr).
// Returns nullptr when the modality is unrouted / routed to "default" (caller
// uses its built-in path) or the override names an unknown backend.
const BackendSpec *resolve(const RoutingTable &tbl, const std::string &modality,
                           const std::string &override_name = "");

// Backend names a per-request override may legitimately target for `modality`:
// the modality's configured default (when it isn't "default") plus every
// kind:openai backend (an OpenAI-compatible endpoint can serve any modality via
// its own prompt/parser). This is EXACTLY the set the pipeline instantiates into
// its per-modality recognizer registry, so route-layer validation (reject
// unknown names with a 400) and the registry can never drift apart.
std::set<std::string> routable_backend_names(const RoutingTable &tbl,
                                             const std::string &modality);

// Parse + fully validate a single INLINE backend spec (JSON text) for Tier-B
// /infer. Applies the same checks as a config-file backend (kind/engine/
// base_url/model/prompt/parser/api_key-env-indirection) PLUS modality-engine
// match for kind:local. Throws RoutingConfigError (ROUTING_* codes) on any
// violation; the /infer route maps that to a 400. The SSRF gate for inline
// kind:openai (TURBO_ALLOW_ADHOC_BACKENDS) is enforced by the route, not here.
BackendSpec parse_inline_backend(const std::string &modality,
                                 const std::string &json_text);

const char *kind_name(Kind k) noexcept;
const char *parser_name(Parser p) noexcept;
bool        parse_parser(const std::string &s, Parser &out) noexcept;

} // namespace turbo_ocr::backend_routing
