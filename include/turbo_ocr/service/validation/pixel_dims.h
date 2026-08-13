#pragma once

#include <charconv>
#include <cstddef>
#include <exception>
#include <format>
#include <optional>
#include <string>

#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/size_classify.h"

namespace turbo_ocr::server {

// Resolved /ocr/pixels dimensions, or a 400 to emit. `ok()` is false when
// error_code is set; otherwise width/height/channels are valid.
struct PixelDims {
  int width = 0;
  int height = 0;
  int channels = 3;
  bool used_legacy_header = false;  // a request carried an X-* dim header
  std::string error_code;           // empty => ok
  std::string error;                // human-readable message
  [[nodiscard]] bool ok() const { return error_code.empty(); }
};

// Resolve width/height/channels from query params (preferred, consistent with
// the rest of the API) with the legacy X-Width/X-Height/X-Channels headers as a
// v2.3-compat fallback. A value supplied in BOTH that disagrees -> 400
// DIMENSION_CONFLICT (fail loud, never silently pick one). Shared verbatim by
// the GPU and CPU /ocr/pixels handlers so the contract can't drift between them.
[[nodiscard]] inline PixelDims
resolve_pixel_dims(const drogon::HttpRequestPtr &req) {
  PixelDims d;
  std::string parse_err, conflict_err;
  bool legacy = false;
  auto resolve = [&](const char *qname, const char *hname,
                     std::optional<int> &out) {
    const std::string q = req->getParameter(qname);
    const std::string h = req->getHeader(hname);
    std::optional<int> qv, hv;
    auto parse = [&](const std::string &s, std::optional<int> &o) -> bool {
      if (s.empty()) return true;
      int v = 0;
      auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
      if (ec != std::errc{} || ptr != s.data() + s.size())
        return false;  // reject overflow and trailing junk ("12abc")
      o = v;
      return true;
    };
    if (!parse(q, qv) || !parse(h, hv)) {
      if (parse_err.empty())  // keep the first malformed param's message
        parse_err = std::string("Invalid ") + qname + " value";
      return;
    }
    if (qv && hv && *qv != *hv)
      conflict_err = std::format(
          "Conflicting {0}: query ?{0}={1} disagrees with {2} header {3}",
          qname, q, hname, h);
    if (hv) legacy = true;
    out = qv ? qv : hv;
  };
  std::optional<int> ow, oh, oc;
  resolve("width",    "X-Width",    ow);
  resolve("height",   "X-Height",   oh);
  resolve("channels", "X-Channels", oc);
  d.used_legacy_header = legacy;
  if (!parse_err.empty()) {
    d.error_code = "INVALID_DIMENSIONS";
    d.error = parse_err;
    return d;
  }
  if (!conflict_err.empty()) {
    d.error_code = "DIMENSION_CONFLICT";
    d.error = conflict_err;
    return d;
  }
  if (!ow || !oh) {
    d.error_code = "MISSING_DIMENSIONS";
    d.error = "Missing width/height (use ?width=&height= query params or the "
              "X-Width/X-Height headers)";
    return d;
  }
  d.width = *ow;
  d.height = *oh;
  d.channels = oc.value_or(3);
  if (d.width <= 0 || d.height <= 0 || (d.channels != 1 && d.channels != 3)) {
    d.error_code = "INVALID_DIMENSIONS";
    d.error = "Invalid dimensions or channels";
    return d;
  }
  return d;
}

// RFC 8594 deprecation signal for clients still passing the legacy X-* dim
// headers. Stamped on the success response so the migration to query params is
// discoverable without breaking the v2.3 contract. A custom advisory header
// carries the actionable hint (RFC 8594 leaves the body/link to the operator).
inline void stamp_pixel_dim_deprecation(const drogon::HttpResponsePtr &resp) {
  resp->addHeader("Deprecation", "true");
  resp->addHeader("X-Deprecation-Notice",
                  "X-Width/X-Height/X-Channels headers are deprecated; pass "
                  "?width=&height=&channels= query params instead");
}

// Full /ocr/pixels payload validation, shared verbatim by the GPU and CPU
// handlers so the caps can never drift between the builds: dimension
// resolution (query params + legacy X-* headers), per-side cap
// (MAX_IMAGE_DIM), pixel-AREA cap, and the exact body-size check. On failure
// the returned PixelDims carries the error code/message (`ok()` false).
[[nodiscard]] inline PixelDims
validate_pixel_payload(const drogon::HttpRequestPtr &req) {
  PixelDims d = resolve_pixel_dims(req);
  if (!d.ok()) return d;
  // Shared bomb verdict + message (decode/size_classify.h) — same code AND
  // text as every other image endpoint, HTTP and gRPC. The pixel-AREA cap
  // matters because the body-size check below bounds RAM only while
  // MAX_BODY_MB stays small; without it, raising the body cap would let a
  // huge body allocate proportionally on the decode side.
  if (auto v = decode::classify_image_size(d.width, d.height);
      v != decode::ImageSizeVerdict::kOk) {
    d.error_code = decode::image_size_error_code(v);
    d.error = decode::image_size_error_message(v, d.width, d.height);
    return d;
  }
  const size_t expected = static_cast<size_t>(d.width) * d.height * d.channels;
  if (req->body().size() != expected) {
    d.error_code = "BODY_SIZE_MISMATCH";
    d.error = std::format(
        "Body size mismatch: expected {} bytes ({}x{}x{}), got {}", expected,
        d.width, d.height, d.channels, req->body().size());
  }
  return d;
}

// Wrap the raw pixel body as a BGR cv::Mat. The pipeline is BGR-only: a
// 1-channel Mat trips the degenerate-input guards downstream, so grayscale is
// expanded up front (allocating). A 3-channel input stays a NON-OWNING view
// into req->body() — the caller must keep `req` alive for the Mat's lifetime.
[[nodiscard]] inline cv::Mat
pixel_body_to_bgr(const drogon::HttpRequestPtr &req, const PixelDims &d) {
  cv::Mat img(d.height, d.width, d.channels == 3 ? CV_8UC3 : CV_8UC1,
              const_cast<char *>(req->body().data()));
  if (d.channels == 1)
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
  return img;
}

}  // namespace turbo_ocr::server
