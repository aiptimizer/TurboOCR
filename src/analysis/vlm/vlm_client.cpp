#include "turbo_ocr/analysis/vlm/vlm_client.h"

#include <string>
#include <vector>

#include <curl/curl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "simdutf.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"

namespace turbo_ocr::vlm {

namespace {

// Initialize libcurl exactly once.
struct CurlGlobalInit {
  CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobalInit() { curl_global_cleanup(); }
};
CurlGlobalInit g_curl_init;

} // namespace

void ensure_curl_init() { (void)&g_curl_init; }

namespace {

size_t curl_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
  auto *buf = static_cast<std::string *>(userdata);
  // Abort (short count -> CURLE_WRITE_ERROR) past the cap: a wedged or
  // hostile endpoint must not grow the buffer without bound for the whole
  // timeout window.
  constexpr size_t kMaxResponseBytes = 32u << 20;
  const size_t n = size * nmemb;
  if (buf->size() + n > kMaxResponseBytes) return 0;
  buf->append(ptr, n);
  return n;
}

// A stalled TCP/TLS handshake must not silently consume the whole request
// budget; cap it independently of the total timeout (mirrors openai_endpoint).
constexpr long kConnectTimeoutS = 5;

// RAII wrappers so the easy handle + header list are always released on every
// return/throw path — no manual cleanup to forget on a new early-return.
struct CurlEasy {
  CURL *h = curl_easy_init();
  CurlEasy() = default;
  CurlEasy(const CurlEasy &) = delete;
  CurlEasy &operator=(const CurlEasy &) = delete;
  ~CurlEasy() { if (h) curl_easy_cleanup(h); }
  explicit operator bool() const noexcept { return h != nullptr; }
  operator CURL *() const noexcept { return h; }
};

struct CurlHeaders {
  curl_slist *h = nullptr;
  CurlHeaders() = default;
  CurlHeaders(const CurlHeaders &) = delete;
  CurlHeaders &operator=(const CurlHeaders &) = delete;
  ~CurlHeaders() { if (h) curl_slist_free_all(h); }
  void append(const char *s) { h = curl_slist_append(h, s); }
};

} // namespace

bool use_pool_backend() {
  return env::env_or("VLM_BACKEND", "pool").front() != 'l';
}


HttpResp http_post_json(const std::string &url, const std::string &json_body,
                        int timeout_s, const char *log_tag) {
  HttpResp r;
  CurlEasy curl;
  if (!curl) return r;
  CurlHeaders hdrs;
  hdrs.append("Content-Type: application/json");
  hdrs.append("Accept: application/json");
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);      // thread-safe timeout path
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)json_body.size());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs.h);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)timeout_s);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutS);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r.body);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 3L);  // bound redirect chains
  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r.status);
    r.ok = (r.status >= 200 && r.status < 300);
  } else {
    TOCR_LOG_ERROR_RL(log_tag, "error", curl_easy_strerror(rc));
  }
  return r;
}

HttpResp http_get(const std::string &url, int timeout_s,
                  const std::string &bearer) {
  HttpResp r;
  CurlEasy curl;
  if (!curl) return r;
  struct curl_slist *headers = nullptr;
  if (!bearer.empty()) {
    const std::string auth = "Authorization: Bearer " + bearer;
    headers = curl_slist_append(headers, auth.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  }
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)timeout_s);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutS);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r.body);
  CURLcode rc = curl_easy_perform(curl);
  if (headers) curl_slist_free_all(headers);
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r.status);
    r.ok = (r.status >= 200 && r.status < 300);
  }
  return r;
}

// PNG-encode a BGR crop. Compression 0 = fastest; crops are tiny, net is loopback.
std::vector<uint8_t> encode_png_bgr(const uint8_t *data, int w, int h, int stride) {
  cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t *>(data), (size_t)stride);
  std::vector<uint8_t> out;
  std::vector<int> params{cv::IMWRITE_PNG_COMPRESSION, 0};
  if (!cv::imencode(".png", src, out, params)) return {};
  return out;
}

nlohmann::json make_image_block(const std::string &b64_png) {
  return {
      {"type", "image_url"},
      {"image_url", {{"url", std::string("data:image/png;base64,") + b64_png}}},
  };
}

} // namespace turbo_ocr::vlm
