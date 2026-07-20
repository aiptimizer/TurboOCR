#pragma once

#include <array>
#include <set>
#include <string>
#include <string_view>

#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/backend_routing/routing_config.h"

// The ONE request-parameter validation core, shared by every HTTP endpoint on
// both builds (and reusable from gRPC for the routing piece). Deliberately
// drogon-free so it unit-tests without a web framework and stays usable from
// any transport.
//
// Why this exists: each endpoint used to hand-assemble its own validation —
// its own allow-list literal, its own strict-mode helper copy, its own
// routing block. Ten handlers times two builds meant any new endpoint (or the
// CPU twin of a GPU endpoint) could silently drop a step, which is exactly
// how routing overrides went missing on /ocr/pixels and strict-param handling
// on /ocr/markdown. Here an endpoint DECLARES what it supports (EndpointSpec)
// and the core derives everything else, so every parameter lands in exactly
// one of three structural categories:
//
//   1. supported by this endpoint        -> parsed
//   2. known API param, unsupported here -> ALWAYS a loud 400 (never ignored)
//   3. unknown                           -> 400 in strict mode, ignored in
//                                           lenient mode (compat)
//
// A capability an endpoint forgets to declare is therefore rejected loudly at
// the first request — a visible bug filed by the caller, never a silent gap.
namespace turbo_ocr::server {

// L3 strict-query-params (opt-in). Default OFF preserves the historical
// lenient behavior for UNKNOWN parameters (category 3); known-but-unsupported
// parameters (category 2) are rejected regardless of this flag. Read once;
// cached for the process lifetime. The single definition — every endpoint on
// both transports reads the same switch.
[[nodiscard]] inline bool strict_query_params_enabled() noexcept {
  static const bool v = env::env_enabled("TURBO_OCR_STRICT_QUERY_PARAMS");
  return v;
}

// Empty message == success. `code` is the HTTP error-envelope / gRPC
// x-error-code identifier.
struct ValidationError {
  std::string message;
  std::string code;
  [[nodiscard]] bool ok() const noexcept { return message.empty(); }
};

// Whether the endpoint honors per-request routing overrides
// (route_table/route_formula). kUnsupported endpoints REJECT a non-empty
// override with `routing_unsupported_reason` — validating-and-ignoring or
// silently ignoring are both silent failures.
enum class RoutingSupport { kSupported, kUnsupported };

// The default routing policy for image endpoints on this build: the CPU
// pipeline has no routing plumbing, so overrides are structurally
// GPU-only. Shared-TU registrars (compiled into both builds) use this so the
// policy can never diverge between the builds by hand-editing one of them.
#ifdef USE_CPU_ONLY
inline constexpr RoutingSupport kBuildRoutingSupport = RoutingSupport::kUnsupported;
#else
inline constexpr RoutingSupport kBuildRoutingSupport = RoutingSupport::kSupported;
#endif

inline constexpr const char *kRoutingUnsupportedCpu =
    "per-request routing overrides are not supported on the CPU build (the "
    "configured backend is always used)";
inline constexpr const char *kRoutingUnsupportedPdf =
    "per-request routing overrides are not supported on PDF endpoints (the "
    "configured backends are always used)";
inline constexpr const char *kRoutingUnsupportedEndpoint =
    "per-request routing overrides are not supported on this endpoint";

// Declarative endpoint capability surface. Each flag both ALLOWS the
// corresponding parameter group and (for routing) selects the enforcement
// policy. Anything not declared here is category 2 or 3 above.
struct EndpointSpec {
  // layout / reading_order / as_blocks / tables / formulas / text
  bool ocr_options = true;
  // text=0 (layout-only) permitted? The batched det/rec path has no
  // layout-only equivalent, so /ocr/batch declares false and the caller gets
  // an explicit redirect instead of full OCR against their stated intent.
  bool layout_only_allowed = true;
  RoutingSupport routing = RoutingSupport::kUnsupported;
  const char *routing_unsupported_reason = kRoutingUnsupportedCpu;
  // width / height / channels (raw-pixel input endpoints)
  bool pixel_dims = false;
  // dpi / mode / markdown / as_pages / images / format / lossless /
  // png_compression / quality / max_side / autorotate (PDF endpoints)
  bool pdf_options = false;
  // embed (/ocr/markdown)
  bool markdown_embed = false;
};

namespace detail {
inline constexpr std::array<std::string_view, 6> kOcrOptionParams = {
    "layout", "reading_order", "as_blocks", "tables", "formulas", "text"};
inline constexpr std::array<std::string_view, 2> kRoutingParams = {
    "route_table", "route_formula"};
inline constexpr std::array<std::string_view, 3> kPixelDimParams = {
    "width", "height", "channels"};
inline constexpr std::array<std::string_view, 11> kPdfParams = {
    "dpi",     "mode",            "markdown", "as_pages",
    "images",  "format",          "lossless", "png_compression",
    "quality", "max_side",        "autorotate"};
inline constexpr std::array<std::string_view, 1> kMarkdownParams = {"embed"};

template <class Group>
[[nodiscard]] constexpr bool in_group(const Group &g, std::string_view name) {
  for (auto p : g)
    if (p == name) return true;
  return false;
}
} // namespace detail

// Category test: is `name` a parameter ANY endpoint understands? Derived from
// the same group tables the allow test uses, so the known-parameter universe
// can never drift from the per-endpoint surfaces.
[[nodiscard]] constexpr bool is_known_param(std::string_view name) {
  using namespace detail;
  return in_group(kOcrOptionParams, name) || in_group(kRoutingParams, name) ||
         in_group(kPixelDimParams, name) || in_group(kPdfParams, name) ||
         in_group(kMarkdownParams, name);
}

// Category test: does `spec` support `name`? Routing params are excluded —
// they are policy-checked (with a specific reason) in validate_params, never
// generically.
[[nodiscard]] constexpr bool spec_allows(const EndpointSpec &spec,
                                         std::string_view name) {
  using namespace detail;
  if (spec.ocr_options && in_group(kOcrOptionParams, name)) return true;
  if (spec.pixel_dims && in_group(kPixelDimParams, name)) return true;
  if (spec.pdf_options && in_group(kPdfParams, name)) return true;
  if (spec.markdown_embed && in_group(kMarkdownParams, name)) return true;
  return false;
}

// Routing-override enforcement — the single implementation behind the HTTP
// query params, the /ocr JSON `routing{}` field, and the gRPC request fields.
// kSupported: unknown backend name -> ROUTING_UNKNOWN_OVERRIDE; valid name ->
// populated into *out. kUnsupported: any non-empty override -> the spec's
// reason (INVALID_PARAMETER). Empty names are always fine (no override).
[[nodiscard]] inline ValidationError apply_routing_override(
    const std::string &table, const std::string &formula,
    const EndpointSpec &spec, const std::set<std::string> &valid_table,
    const std::set<std::string> &valid_formula, backend_routing::RequestRouting *out) {
  if (table.empty() && formula.empty()) return {};
  if (spec.routing == RoutingSupport::kUnsupported)
    return {spec.routing_unsupported_reason, "INVALID_PARAMETER"};
  if (!table.empty()) {
    if (valid_table.find(table) == valid_table.end())
      return {"route_table override '" + table +
                  "' names no configured table backend (see /capabilities)",
              "ROUTING_UNKNOWN_OVERRIDE"};
    out->table = table;
  }
  if (!formula.empty()) {
    if (valid_formula.find(formula) == valid_formula.end())
      return {"route_formula override '" + formula +
                  "' names no configured formula backend (see /capabilities)",
              "ROUTING_UNKNOWN_OVERRIDE"};
    out->formula = formula;
  }
  return {};
}

// The core pass over a request's query parameters. `params` is any map-like
// range of (name, value) pairs (drogon's parameter map, or a std::map in
// tests). Every present parameter is classified per the three categories;
// routing params are policy-checked and, when supported, validated against
// the registered backend-name sets and written into *routing_out.
template <class ParamMap>
[[nodiscard]] ValidationError validate_params(
    const ParamMap &params, const EndpointSpec &spec,
    const std::set<std::string> &valid_route_table,
    const std::set<std::string> &valid_route_formula, bool strict,
    backend_routing::RequestRouting *routing_out,
    std::vector<std::string> *ignored_out = nullptr) {
  std::string route_table, route_formula;
  for (const auto &kv : params) {
    const std::string &name = kv.first;
    if (detail::in_group(detail::kRoutingParams, name)) {
      (name == "route_table" ? route_table : route_formula) = kv.second;
      continue;  // policy-checked below with a specific message
    }
    if (spec_allows(spec, name)) continue;
    if (is_known_param(name)) {
      // Known API parameter this endpoint does not support. Dropping it does
      // NOT falsify the response (the endpoint never acts on it), so it is
      // tolerated in lenient mode for v3.4 compatibility and loud only under
      // TURBO_OCR_STRICT_QUERY_PARAMS=1. Parameters whose silent dropping
      // WOULD falsify the response never reach this branch: routing
      // overrides are policy-checked above with their own reason, and
      // text=0 / embed=0 are enforced at their endpoint gates.
      if (strict)
        return {"'" + name + "' is not supported on this endpoint",
                "INVALID_PARAMETER"};
      // DEPRECATED tolerance, removed in v4 (always 400 there). Reported so
      // the caller (gate) can surface it via response header + log.
      if (ignored_out) ignored_out->push_back(name);
      continue;
    }
    if (strict)
      return {"Unknown query parameter '" + name +
                  "' (TURBO_OCR_STRICT_QUERY_PARAMS=1)",
              "INVALID_PARAMETER"};
  }
  return apply_routing_override(route_table, route_formula, spec,
                                valid_route_table, valid_route_formula,
                                routing_out);
}

} // namespace turbo_ocr::server
