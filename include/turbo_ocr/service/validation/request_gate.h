#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/service/validation/request_validation.h"
#include "turbo_ocr/service/server/server_types.h"

// The Drogon-side request gate: the ONE call every HTTP inference handler
// makes before doing any work. It sequences the shared validation steps in a
// fixed order over the transport-agnostic core (request_validation.h):
//
//   1. parse_query_options      — shared OCR option semantics, INCLUDING the
//                                 requested-vs-loaded capability gate
//   2. validate_params          — three-way parameter classification +
//                                 the endpoint's routing policy
//   3. layout-only gate         — endpoints without a layout-only path
//
// (check_structure_backends is NOT a step here: step 1 already applies the
// identical gate; it survives only for callers that mutate InferOptions after
// parsing, like /ocr/pdf's markdown defaults.)
//
// A handler therefore cannot accidentally skip a validation step; the only
// place a capability can be mis-declared is its EndpointSpec, where it is one
// visible line reviewed next to the handler.
namespace turbo_ocr::server {

// Parameters from the URL query string ONLY. drogon's getParameters() also
// parses the POST body when the client sends (or curl defaults to)
// application/x-www-form-urlencoded — for binary uploads that turns image
// bytes into garbage "parameters", which strict mode would then reject.
[[nodiscard]] inline std::map<std::string, std::string>
query_only_params(const drogon::HttpRequestPtr &req) {
  std::map<std::string, std::string> out;
  const std::string &q = req->query();
  std::size_t pos = 0;
  while (pos < q.size()) {
    std::size_t amp = q.find('&', pos);
    if (amp == std::string::npos) amp = q.size();
    const std::string_view kv(q.data() + pos, amp - pos);
    const std::size_t eq = kv.find('=');
    std::string k(eq == std::string_view::npos ? kv : kv.substr(0, eq));
    if (!k.empty()) {
      std::string v(eq == std::string_view::npos ? std::string_view{}
                                                 : kv.substr(eq + 1));
      out[drogon::utils::urlDecode(k)] = drogon::utils::urlDecode(v);
    }
    pos = amp + 1;
  }
  return out;
}

// The Tier-A routing-override name sets, derived ONCE from the routing config
// the pipeline loaded.
//
// These are what apply_routing_override validates ?route_table= / ?route_formula=
// against on a kSupported endpoint. Passing EMPTY sets there is not a harmless
// no-op: it rejects every override with ROUTING_UNKNOWN_OVERRIDE and the message
// "names no configured table backend (see /capabilities)" — which is false when
// /capabilities lists the backend. So a registrar must either declare routing
// kUnsupported (and get a specific, honest reason) or pass these sets.
//
// Call at REGISTRATION time and capture the result by value into the handler:
// load_routing_config() reads the routing table from disk/env.
struct RoutingNameSets {
  std::set<std::string> table;
  std::set<std::string> formula;
};
[[nodiscard]] inline RoutingNameSets routing_name_sets() {
  const auto rtbl = backend_routing::load_routing_config();
  return RoutingNameSets{backend_routing::routable_backend_names(rtbl, "table"),
                         backend_routing::routable_backend_names(rtbl, "formula")};
}

// `loaded`: the capabilities this server actually brought up (see
// capability/capability.h). ONE argument replaces the layout/table/formula bool
// triple that used to be threaded through every registrar — a distinct type
// cannot be transposed with its neighbours, which is how the gRPC registrar
// came to take the same three flags in a DIFFERENT ORDER without anything
// noticing.
//
// `body`: the parsed JSON body when the endpoint has one, so flags sent as
// `{"layout": true}` are honoured identically to `?layout=1`. Pass nullptr for
// endpoints that take no JSON body.
//
// `post_parse` (optional): runs BETWEEN the capability parse and the parameter
// classification, and may MUTATE *opts. It exists for the one caller whose
// options are not fully determined by the request — /ocr/pdf's ?markdown=1
// defaults tables/formulas from what the server loaded, and parses the
// document-output params (output/min_confidence/editable/...) that only it
// honours. Return a non-empty ValidationError to reject.
//
// When a hook runs, the gate re-applies check_structure_backends afterwards:
// the hook can add to opts->requested AFTER parse_query_options already gated
// it, so without the second check a mutated request could reach the pipeline
// asking for a capability the server never loaded. That re-check is exactly why
// /ocr/pdf hand-sequenced these steps instead of calling this function, and
// folding it in here is what let it stop.
using PostParseHook = std::function<ValidationError(InferOptions *)>;

[[nodiscard]] inline bool validate_request(
    const drogon::HttpRequestPtr &req, const EndpointSpec &spec,
    const capability::CapabilityMask &loaded,
    const std::set<std::string> &valid_route_table,
    const std::set<std::string> &valid_route_formula, InferOptions *opts,
    DrogonCallback &callback, bool allow_image_only = false,
    const Json::Value *body = nullptr,
    const PostParseHook &post_parse = nullptr) {
  const auto fail = [&](const std::string &code, const std::string &msg) {
    callback(error_response(code, msg));
    return false;
  };
  if (spec.ocr_options) {
    if (auto r = parse_query_options(req, loaded, opts, allow_image_only, body,
                                     spec.acts_on);
        !r.error.empty())
      return fail(r.error_code, r.error);
  }
  if (post_parse) {
    if (auto e = post_parse(opts); !e.ok()) return fail(e.code, e.message);
  }
  std::vector<std::string> ignored;
  if (auto e = validate_params(query_only_params(req), spec, valid_route_table,
                               valid_route_formula,
                               strict_query_params_enabled(),
                               &opts->routing_override, &ignored);
      !e.ok())
    return fail(e.code, e.message);
  if (!ignored.empty()) {
    // Deprecated v3.4-compat tolerance: surfaced to the client via the
    // observability middleware (X-Ignored-Params response header) and the
    // server log. v4 rejects these with 400.
    std::string csv;
    for (const auto &n : ignored) {
      if (!csv.empty()) csv += ',';
      csv += n;
    }
    // Relay through request ATTRIBUTES, not a request header: a header slot is
    // client-writable, so a client sending its own X-Ignored-Params would have
    // it reflected verbatim into the response by the middleware.
    req->getAttributes()->insert("turbo.ignored_params", csv);
    TOCR_LOG_WARN_RL("Ignoring unsupported query parameter(s) — deprecated "
                     "tolerance, v4 rejects with 400",
                     "params", csv, "path", req->path());
  }
  // The requested-vs-loaded gate, re-applied ONLY when a hook ran. Without a
  // hook it is redundant — parse_query_options already applied the identical
  // check and nothing between there and here touches opts->requested. With one,
  // it is load-bearing: the hook may have requested a capability the server
  // never loaded, and this is the last point before the pipeline.
  if (post_parse) {
    if (auto r = check_structure_backends(*opts, loaded); !r.error.empty())
      return fail(r.error_code, r.error);
  }
  if (!opts->want_text && !spec.layout_only_allowed)
    return fail("INVALID_PARAMETER",
                "text=0 (layout-only) is not supported on this endpoint; "
                "send single images to /ocr/raw?text=0 instead");
  return true;
}

} // namespace turbo_ocr::server
