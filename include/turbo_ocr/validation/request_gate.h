#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/validation/request_validation.h"
#include "turbo_ocr/server/server_types.h"

// The Drogon-side request gate: the ONE call every HTTP inference handler
// makes before doing any work. It sequences the shared validation steps in a
// fixed order over the transport-agnostic core (request_validation.h):
//
//   1. parse_query_options      — shared OCR option semantics
//   2. validate_params          — three-way parameter classification +
//                                 the endpoint's routing policy
//   3. layout-only gate         — endpoints without a layout-only path
//   4. check_structure_backends — fail loud on unavailable stages
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

[[nodiscard]] inline bool validate_request(
    const drogon::HttpRequestPtr &req, const EndpointSpec &spec,
    bool layout_available, bool table_available, bool formula_available,
    const std::set<std::string> &valid_route_table,
    const std::set<std::string> &valid_route_formula, InferOptions *opts,
    DrogonCallback &callback, bool allow_image_only = false) {
  const auto fail = [&](const std::string &code, const std::string &msg) {
    callback(error_response(drogon::k400BadRequest, code.c_str(), msg));
    return false;
  };
  if (spec.ocr_options) {
    if (auto r = parse_query_options(req, layout_available, opts,
                                     allow_image_only);
        !r.error.empty())
      return fail(r.error_code, r.error);
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
    req->addHeader("X-Ignored-Params", csv);  // relay to post-handler
    TOCR_LOG_WARN_RL("Ignoring unsupported query parameter(s) — deprecated "
                     "tolerance, v4 rejects with 400",
                     "params", csv, "path", req->path());
  }
  if (!opts->want_text && !spec.layout_only_allowed)
    return fail("INVALID_PARAMETER",
                "text=0 (layout-only) is not supported on this endpoint; "
                "send single images to /ocr/raw?text=0 instead");
  if (spec.ocr_options) {
    if (auto r = check_structure_backends(*opts, table_available,
                                          formula_available);
        !r.error.empty())
      return fail(r.error_code, r.error);
  }
  return true;
}

} // namespace turbo_ocr::server
