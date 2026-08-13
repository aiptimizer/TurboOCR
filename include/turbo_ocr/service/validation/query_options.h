#pragma once

#include <cctype>
#include <format>
#include <string>

#include <drogon/HttpRequest.h>
#include <json/json.h>

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/core/capability.h"
// The parsed VALUE (struct InferOptions) is a transport-free leaf so the
// pipeline can name it without dragging Drogon/JsonCpp in; only the PARSERS
// below need a request object. Included here — not merely forward-declared —
// so every historical includer of query_options.h still gets the struct and
// needs no edit.
#include "turbo_ocr/core/infer_options.h"
// The transport-free core this file adapts to Drogon, plus ParseOptionsResult
// and check_structure_backends (both moved there so gRPC can use them without
// dragging Drogon in). Included, not forward-declared, so every historical
// includer of query_options.h keeps compiling unchanged.
#include "turbo_ocr/service/validation/options_core.h"

namespace turbo_ocr::server {

// Interpret one boolean flag value. Shared by the query-string and JSON-body
// forms so the two can never disagree about what "on" means.
[[nodiscard]] inline std::string parse_bool_value(std::string_view raw,
                                                  const char *key, bool *out) {
  std::string s(raw);
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "1" || s == "true" || s == "on" || s == "yes")  { *out = true;  return {}; }
  if (s == "0" || s == "false" || s == "off" || s == "no") { *out = false; return {}; }
  return std::format("Invalid {} param: '{}' "
                     "(expected 0/1, true/false, on/off, or yes/no)",
                     key, s);
}

// Parse a generic boolean query param ("1"/"true"/"on"/"yes" etc.).
// Returns empty string on success and writes to *out; otherwise returns an
// error message. When the parameter is absent, *out is set to false and an
// empty string is returned.
[[nodiscard]] inline std::string parse_bool_query(const drogon::HttpRequestPtr &req,
                                                   const char *key,
                                                   bool *out) {
  *out = false;
  auto v = req->getParameter(key);
  if (v.empty()) return {};
  return parse_bool_value(v, key, out);
}

// Read one flag from the query string OR the JSON body, whichever carries it.
//
// WHY BOTH: /ocr takes its image in a JSON body, and clients reasonably send
// their flags there too — but every endpoint used to read flags ONLY via
// getParameter(), so `{"image":..., "layout":true}` was accepted with HTTP 200
// and the layout silently dropped. The client could not tell that from a page
// that genuinely had no layout. Resolving both forms in one function is what
// makes "the body is ignored" unrepresentable.
//
// The query string WINS when both are present: it is the more explicit,
// more visible form (it shows up in access logs and proxies), so an operator
// debugging a URL sees the flag that actually took effect.
[[nodiscard]] inline std::string
parse_bool_flag(const drogon::HttpRequestPtr &req, const Json::Value *body,
                std::string_view name, bool *out) {
  *out = false;
  const std::string key(name);
  if (auto v = req->getParameter(key); !v.empty())
    return parse_bool_value(v, key.c_str(), out);
  if (body && body->isObject() && body->isMember(key)) {
    const Json::Value &v = (*body)[key];
    if (v.isBool())   { *out = v.asBool(); return {}; }
    // Integers accept exactly 0/1 — the query form rejects "2", and the two
    // forms must share one boolean grammar or clients learn the wrong one.
    if (v.isIntegral()) {
      const auto n = v.asInt64();
      if (n == 0 || n == 1) { *out = n == 1; return {}; }
      return std::format("Invalid {} field: expected a boolean (0/1)", key);
    }
    if (v.isString())  return parse_bool_value(v.asString(), key.c_str(), out);
    return std::format("Invalid {} field: expected a boolean", key);
  }
  return {};
}

// Parse the full set of opt-in query parameters for inference routes.
// `layout` and `reading_order` both default to 0; either being set to 1
// without the underlying layout model causes a 400 with a descriptive
// error code (LAYOUT_DISABLED).
//
// This is the DROGON ADAPTER over parse_options_core (options_core.h): it
// supplies a flag reader that resolves each name against the query string and
// the JSON body, and the core does everything else. gRPC supplies a
// proto-reflection reader to the same core, which is what stops the two
// transports' gates from drifting — see the header comment in options_core.h
// for the three times they already had.
// `acts_on` (EndpointSpec::acts_on): the capabilities THIS endpoint can run.
// Flags outside it are not parsed here — they fall through to validate_params'
// classification like any other unsupported parameter (lenient: ignored +
// X-Ignored-Params; strict: 400), exactly as at v3.5.0. Parsing them here
// instead would either 400 on availability for a stage the endpoint could
// never run, or accept the flag and silently not act on it.
[[nodiscard]] inline ParseOptionsResult
parse_query_options(const drogon::HttpRequestPtr &req,
                    const capability::CapabilityMask &loaded,
                    InferOptions *out,
                    bool allow_image_only = false,
                    const Json::Value *body = nullptr,
                    capability::CapabilityMask acts_on =
                        capability::CapabilityMask::all().set(
                            capability::CapabilityId::DocOrientation, false)) {
  // The ONLY Drogon-aware part: resolve one flag name against the query string
  // and the JSON body. Everything downstream — the capability sweep, the
  // implications, the availability gate, the bool projection and the text=0
  // rules — is parse_options_core, shared verbatim with gRPC.
  const auto read_flag = [&](std::string_view name, bool *value,
                             bool *present) -> std::string {
    const std::string key(name);
    *present = !req->getParameter(key).empty() ||
               (body && body->isObject() && body->isMember(key));
    return parse_bool_flag(req, body, name, value);
  };
  return parse_options_core(read_flag, loaded, out, allow_image_only, acts_on);
}

} // namespace turbo_ocr::server
