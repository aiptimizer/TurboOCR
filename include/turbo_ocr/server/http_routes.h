#pragma once

#include <climits>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/fast_png_decoder.h"
#include "crow/crow_all.h"

namespace turbo_ocr::server {

// ---------------------------------------------------------------------------
// Image decoder type: (raw_bytes_ptr, length) -> cv::Mat
// ---------------------------------------------------------------------------
using ImageDecoder = std::function<cv::Mat(const unsigned char *data, size_t len)>;

/// Default CPU-only image decoder: JPEG (OpenCV) and PNG (Wuffs) only.
/// Returns empty cv::Mat for unsupported formats.
[[nodiscard]] inline cv::Mat cpu_decode_image(const unsigned char *data, size_t len) {
  // JPEG: magic bytes 0xFF 0xD8
  if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
    if (len > static_cast<size_t>(INT_MAX)) return {};
    return cv::imdecode(
        cv::Mat(1, static_cast<int>(len), CV_8UC1,
                const_cast<unsigned char *>(data)),
        cv::IMREAD_COLOR);
  }
  // PNG: Wuffs fast decoder
  if (decode::FastPngDecoder::is_png(data, len)) {
    return decode::FastPngDecoder::decode(data, len);
  }
  // Unsupported format
  return {};
}

// ---------------------------------------------------------------------------
// InferFunc: given a cv::Mat, run OCR and return results.
// This is the bridge between pool-type-specific acquire/release and the
// generic route handlers. Each server constructs its own InferFunc that
// acquires from its specific pool type.
// ---------------------------------------------------------------------------
using InferFunc = std::function<std::vector<OCRResultItem>(const cv::Mat &)>;

// ---------------------------------------------------------------------------
// Shared route: POST /ocr  (base64 JSON input)
// ---------------------------------------------------------------------------
inline void register_ocr_base64_route(crow::SimpleApp &app,
                                       const InferFunc &infer,
                                       const ImageDecoder &decode) {
  CROW_ROUTE(app, "/ocr")
      .methods(crow::HTTPMethod::Post)(
          [&infer, &decode](const crow::request &req) {
    auto x = crow::json::load(req.body);
    if (!x)
      return crow::response(400, "Invalid JSON");

    if (!x.has("image") || x["image"].t() != crow::json::type::String
        || x["image"].s().size() == 0)
      return crow::response(400, "Empty or missing image field");

    std::string decoded = base64_decode(x["image"].s());
    if (decoded.empty())
      return crow::response(400, "Failed to decode base64");

    cv::Mat img = decode(
        reinterpret_cast<const unsigned char *>(decoded.data()),
        decoded.size());

    if (img.empty())
      return crow::response(400, "Failed to decode image");

    try {
      auto results = infer(img);
      auto json_str = results_to_json(results);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return crow::response(503, "Service overloaded, try again later");
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });
}

// ---------------------------------------------------------------------------
// Shared route: POST /ocr/raw  (raw image bytes)
// ---------------------------------------------------------------------------
inline void register_ocr_raw_route(crow::SimpleApp &app,
                                    const InferFunc &infer,
                                    const ImageDecoder &decode) {
  CROW_ROUTE(app, "/ocr/raw")
      .methods(crow::HTTPMethod::Post)(
          [&infer, &decode](const crow::request &req) {
    if (req.body.empty())
      return crow::response(400, "Empty body");

    cv::Mat img = decode(
        reinterpret_cast<const unsigned char *>(req.body.data()),
        req.body.size());

    if (img.empty())
      return crow::response(400, "Failed to decode image");

    try {
      auto results = infer(img);
      auto json_str = results_to_json(results);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return crow::response(503, "Service overloaded, try again later");
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr/raw] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr/raw] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });
}

// ---------------------------------------------------------------------------
// Shared route: GET /health
// ---------------------------------------------------------------------------
inline void register_health_route(crow::SimpleApp &app) {
  CROW_ROUTE(app, "/health")
      .methods(crow::HTTPMethod::Get)([](const crow::request &) {
    return crow::response(200, "ok");
  });
}

// ---------------------------------------------------------------------------
// Convenience: register all shared routes at once
// ---------------------------------------------------------------------------
inline void register_common_routes(crow::SimpleApp &app,
                                    const InferFunc &infer,
                                    const ImageDecoder &decode) {
  register_health_route(app);
  register_ocr_base64_route(app, infer, decode);
  register_ocr_raw_route(app, infer, decode);
}

} // namespace turbo_ocr::server
