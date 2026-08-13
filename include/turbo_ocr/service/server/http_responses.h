#pragma once

#include <functional>
#include <string>
#include <string_view>

#include <drogon/HttpResponse.h>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/service/server/error_codes.h"
#include "turbo_ocr/service/server/error_classify.h"

namespace turbo_ocr::server {

/// Drogon callback alias.
using DrogonCallback = std::function<void(const drogon::HttpResponsePtr &)>;

// ── Response helpers ────────────────────────────────────────────────────

namespace detail {
[[nodiscard]] inline drogon::HttpResponsePtr
error_response_impl(int status, std::string_view code,
                    const std::string &message) {
  std::string body;
  body.reserve(64 + code.size() + message.size());
  body += R"({"error":{"code":")";
  body += code;
  body += R"(","message":")";
  // FULL RFC 8259 string escape, not just quote/backslash: several messages
  // splice request-controlled text (unknown parameter names, override names),
  // and a raw control byte — a newline, a NUL, a \x01 — inside a JSON string
  // is invalid JSON that a strict client parser rejects, turning a clean 4xx
  // into an unparseable body.
  static constexpr char kHex[] = "0123456789abcdef";
  for (char c : message) {
    const auto u = static_cast<unsigned char>(c);
    switch (c) {
    case '"': body += "\\\""; break;
    case '\\': body += "\\\\"; break;
    case '\n': body += "\\n"; break;
    case '\r': body += "\\r"; break;
    case '\t': body += "\\t"; break;
    default:
      if (u < 0x20) {
        body += "\\u00";
        body += kHex[u >> 4];
        body += kHex[u & 0xF];
      } else {
        body += c;
      }
    }
  }
  body += R"("}})";
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(static_cast<drogon::HttpStatusCode>(status));
  resp->setBody(std::move(body));
  resp->setContentTypeString("application/json");
  return resp;
}
} // namespace detail

/// Structured JSON error response: {"error":{"code":"...","message":"..."}}
///
/// There is NO status parameter, and that is the point: a caller that could
/// pass one is a caller that can answer a different status than another route
/// for the same code. The status comes from error_codes.h, which is also where
/// gRPC reads its StatusCode — one row changes both surfaces.
[[nodiscard]] inline drogon::HttpResponsePtr
error_response(ErrorCode code, const std::string &message) {
  return detail::error_response_impl(error_http_status(code),
                                     error_code_str(code), message);
}

/// Overload for codes that exist only as a runtime string: a ValidationError
/// from validation/, a capability_table.def code, or the ROUTING_* prefix
/// parsed out of a RoutingConfigError. Same table, looked up by wire name.
[[nodiscard]] inline drogon::HttpResponsePtr
error_response(std::string_view code, const std::string &message) {
  return detail::error_response_impl(error_http_status(code), code, message);
}

/// Plain-text response (for /health and non-error uses).
[[nodiscard]] inline drogon::HttpResponsePtr make_response(
    drogon::HttpStatusCode code, std::string body) {
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(code);
  resp->setBody(std::move(body));
  return resp;
}

/// JSON success response.
[[nodiscard]] inline drogon::HttpResponsePtr json_response(std::string json_str) {
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(drogon::k200OK);
  resp->setBody(std::move(json_str));
  resp->setContentTypeString("application/json");
  return resp;
}

// ── Error handling wrapper ──────────────────────────────────────────────

template <typename F>
void run_with_error_handling(DrogonCallback &cb, const char *route, F &&fn) {
  // EVERY HTTP inference route funnels its exceptions here. The type→code
  // decision lives in classify_current_exception() (error_classify.h), shared
  // with the gRPC wrapper so the two transports cannot drift; this side only
  // renders the result and logs the unknown case.
  try {
    fn();
  } catch (...) {
    ExceptionClass ec = classify_current_exception();
    if (ec.log) {
      std::string detail;
      try { throw; } catch (const std::exception &e) { detail = e.what(); }
                     catch (...) { detail = "unknown exception"; }
      TOCR_LOG_ERROR_RL("Inference error", "route", std::string_view(route),
                        "error", std::string_view(detail));
    }
    cb(error_response(ec.code, ec.message));
  }
}

} // namespace turbo_ocr::server
