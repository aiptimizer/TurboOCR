#pragma once

#include <cstring>
#include <functional>
#include <string>

#include <drogon/HttpResponse.h>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/logger.h"

namespace turbo_ocr::server {

/// Drogon callback alias.
using DrogonCallback = std::function<void(const drogon::HttpResponsePtr &)>;

// ── Response helpers ────────────────────────────────────────────────────

/// Structured JSON error response: {"error":{"code":"...","message":"..."}}
[[nodiscard]] inline drogon::HttpResponsePtr error_response(
    drogon::HttpStatusCode status, const char *code, const std::string &message) {
  std::string body;
  body.reserve(64 + std::strlen(code) + message.size());
  body += R"({"error":{"code":")";
  body += code;
  body += R"(","message":")";
  // Escape quotes in message
  for (char c : message) {
    if (c == '"') body += "\\\"";
    else if (c == '\\') body += "\\\\";
    else body += c;
  }
  body += R"("}})";
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(status);
  resp->setBody(std::move(body));
  resp->setContentTypeString("application/json");
  return resp;
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
  try {
    fn();
  } catch (const turbo_ocr::TimeoutError &e) {
    // C4: a per-request deadline overrun. Must map to 504 INFERENCE_TIMEOUT —
    // same as the GPU image routes — not the generic 500 below. /ocr (base64)
    // is the one inference route still on this shared handler in the GPU build.
    cb(error_response(drogon::k504GatewayTimeout, "INFERENCE_TIMEOUT", e.what()));
  } catch (const turbo_ocr::PoolExhaustedError &e) {
    cb(error_response(drogon::k503ServiceUnavailable, "SERVER_BUSY", e.what()));
  } catch (const turbo_ocr::ImageTooLargeError &e) {
    cb(error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE", e.what()));
  } catch (const turbo_ocr::GpuDecodeError &e) {
    // Server fault, retryable: the GPU decoder failed or none was free.
    TOCR_LOG_ERROR_RL("GPU decode failed", "error", e.what());
    cb(error_response(drogon::k503ServiceUnavailable, "GPU_DECODE_FAILED", e.what()));
  } catch (const turbo_ocr::ImageDecodeError &e) {
    cb(error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", e.what()));
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("Inference error", "route", std::string_view(route), "error", std::string_view(e.what()));
    cb(error_response(drogon::k500InternalServerError, "INFERENCE_ERROR", "Inference error"));
  } catch (...) {
    TOCR_LOG_ERROR_RL("Inference error: unknown exception", "route", std::string_view(route));
    cb(error_response(drogon::k500InternalServerError, "INFERENCE_ERROR", "Inference error"));
  }
}

} // namespace turbo_ocr::server
