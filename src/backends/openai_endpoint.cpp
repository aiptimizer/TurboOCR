#include "turbo_ocr/backends/openai_endpoint.h"
#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/formula/latex_extract.h"
#include "turbo_ocr/table/html_reconstruct.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <future>
#include <iostream>
#include <regex>
#include <utility>

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/table/vlm_table.h" // table::otsl_to_html
#include "turbo_ocr/vlm/crop_pool.h"

namespace turbo_ocr::backends {
namespace {

size_t write_cb(char *p, size_t s, size_t n, void *ud) {
  static_cast<std::string *>(ud)->append(p, s * n);
  return s * n;
}

// Minimal libcurl GET; true on 2xx. (Mirrors the file-local http_get already
// duplicated in vlm_formula.cpp — kept local here to stay purely additive.)
// Sends 'Authorization: Bearer <api_key>' when api_key is non-empty so an
// auth-gated endpoint isn't silently disabled at boot despite a valid key.
bool http_get_ok(const std::string &url, int timeout_s, std::string &body,
                 const std::string &api_key = "") {
  CURL *c = curl_easy_init();
  if (!c) return false;
  struct curl_slist *headers = nullptr;
  if (!api_key.empty()) {
    const std::string auth = "Authorization: Bearer " + api_key;
    headers = curl_slist_append(headers, auth.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
  }
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &body);
  curl_easy_setopt(c, CURLOPT_TIMEOUT, static_cast<long>(timeout_s));
  curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);       // SIGALRM path isn't thread-safe
  curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 5L); // a stalled handshake mustn't eat the budget
  CURLcode rc = curl_easy_perform(c);
  long code = 0;
  curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
  if (headers) curl_slist_free_all(headers);
  curl_easy_cleanup(c);
  return rc == CURLE_OK && code >= 200 && code < 300;
}

// PNG-encode a BGR crop (compression 0; crops tiny, transfer is loopback).
std::vector<uint8_t> encode_png_bgr(const uint8_t *data, int w, int h,
                                    int stride) {
  cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t *>(data),
              static_cast<size_t>(stride));
  std::vector<uint8_t> out;
  std::vector<int> params{cv::IMWRITE_PNG_COMPRESSION, 0};
  if (!cv::imencode(".png", src, out, params)) return {};
  return out;
}

std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}


// Mirror vlm_table.cpp::otsl_or_html: a model that already returns HTML
// (`<table…`) is passed through untouched; only OTSL is converted. Without
// this, otsl_to_html would mangle a valid-HTML response from a kind:openai model.
std::string otsl_or_html(const std::string &raw) {
  std::string t = trim(raw);
  // Sanitize model-emitted HTML passthrough (adversarial-image XSS vector).
  if (t.rfind("<table", 0) == 0) return table::sanitize_table_html(raw);
  return raw.empty() ? "" : table::otsl_to_html(raw);
}

// Stateless dispatch keyed on the parser enum only — no recognizer state — so
// both parse_one and the self-contained async_result_parser() snapshot share it.
std::string parse_with(routing::Parser parser, const std::string &raw) {
  switch (parser) {
  case routing::Parser::Otsl:
    return otsl_or_html(raw);
  case routing::Parser::Latex:
    return formula::extract_latex(raw);
  case routing::Parser::Text:
    return trim(raw);
  case routing::Parser::Raw:
  default:
    return raw;
  }
}

} // namespace

OpenAIEndpoint::OpenAIEndpoint(routing::BackendSpec spec)
    : spec_(std::move(spec)) {}

bool OpenAIEndpoint::health_check() {
  while (!spec_.base_url.empty() && spec_.base_url.back() == '/')
    spec_.base_url.pop_back();
  std::string body;
  if (!http_get_ok(spec_.base_url + "/v1/models", 5, body, spec_.api_key)) {
    TOCR_LOG_ERROR("OpenAIEndpoint /v1/models unreachable, backend disabled",
                   "base_url", spec_.base_url, "backend", spec_.name);
    ready_ = false;
    return false;
  }
  if (spec_.model.empty()) {
    try {
      auto j = nlohmann::json::parse(body);
      if (j.contains("data") && j["data"].is_array() && !j["data"].empty())
        spec_.model = j["data"][0].value("id", "");
    } catch (const std::exception &e) {
      TOCR_LOG_WARN("OpenAIEndpoint /v1/models parse warning", "error", e.what());
    }
  }
  ready_ = !spec_.model.empty();
  TOCR_LOG_INFO("OpenAIEndpoint ready", "base_url", spec_.base_url,
                "model", spec_.model,
                "parser", routing::parser_name(spec_.parser));
  return ready_;
}

std::string OpenAIEndpoint::parse_one(const std::string &raw) const {
  return parse_with(spec_.parser, raw);
}

std::function<std::string(const std::string &)>
OpenAIEndpoint::async_result_parser() const {
  // Capture the parser enum by value — the returned callable holds no pointer
  // back into this endpoint, so it stays valid after a pipeline recycle.
  const routing::Parser parser = spec_.parser;
  return [parser](const std::string &raw) { return parse_with(parser, raw); };
}

std::vector<std::future<std::string>>
OpenAIEndpoint::submit_crops_async(const GpuImage &page,
                                   const std::vector<Box> &boxes,
                                   cudaStream_t stream) {
  const int n = static_cast<int>(boxes.size());
  std::vector<std::future<std::string>> futs;
  if (n == 0 || !ready_ || page.empty()) return futs;

  // D2H the page once on the GPU worker; PNG bytes are copied into the pool at
  // submit, so the futures outlive the page buffer — safe to free after return.
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  std::vector<uint8_t> host(need);
  if (cudaSuccess != cudaMemcpyAsync(host.data(), page.data, need,
                                     cudaMemcpyDeviceToHost, stream))
    return futs;
  // A sync failure leaves `host` partially/garbage-filled; PNG-encoding and
  // shipping crops built from undefined memory is worse than failing loud.
  if (cudaSuccess != cudaStreamSynchronize(stream)) return futs;

  auto &pool = vlm::VLMCropPool::instance();
  futs.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto r = aabb(boxes[i]);
    int x0 = std::clamp(r[0], 0, std::max(0, page.cols - 1));
    int y0 = std::clamp(r[1], 0, std::max(0, page.rows - 1));
    int x1 = std::clamp(r[2], x0, page.cols);
    int y1 = std::clamp(r[3], y0, page.rows);
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);
    const uint8_t *src =
        host.data() + static_cast<size_t>(y0) * page.step + static_cast<size_t>(x0) * 3;
    auto png = encode_png_bgr(src, w, h, static_cast<int>(page.step));
    futs.push_back(pool.submit(std::move(png), spec_.prompt, spec_.model,
                               spec_.max_tokens, spec_.timeout_s, spec_.base_url,
                               spec_.api_key));
  }
  return futs;
}

std::vector<std::future<std::string>>
OpenAIEndpoint::submit_async(const GpuImage &page, const std::vector<Box> &boxes,
                             cudaStream_t stream) {
  return submit_crops_async(page, boxes, stream);
}

std::vector<OpenAIEndpoint::CropOut>
OpenAIEndpoint::infer_crops(const GpuImage &page, const std::vector<Box> &boxes,
                            cudaStream_t stream) {
  const int n = static_cast<int>(boxes.size());
  std::vector<CropOut> out(boxes.size());
  if (n == 0 || !ready_ || page.empty()) return out;

  // D2H the page once; PNG bytes are copied into the pool at submit, so the
  // futures outlive the host buffer — safe to free after the submit loop.
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  std::vector<uint8_t> host(need);
  if (cudaSuccess != cudaMemcpyAsync(host.data(), page.data, need,
                                     cudaMemcpyDeviceToHost, stream))
    return out;
  if (cudaSuccess != cudaStreamSynchronize(stream)) return out;

  auto &pool = vlm::VLMCropPool::instance();
  std::vector<std::future<vlm::CropOutcome>> futs;
  futs.reserve(n);
  for (int i = 0; i < n; ++i) {
    auto r = aabb(boxes[i]);
    int x0 = std::clamp(r[0], 0, std::max(0, page.cols - 1));
    int y0 = std::clamp(r[1], 0, std::max(0, page.rows - 1));
    int x1 = std::clamp(r[2], x0, page.cols);
    int y1 = std::clamp(r[3], y0, page.rows);
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);
    const uint8_t *src =
        host.data() + static_cast<size_t>(y0) * page.step + static_cast<size_t>(x0) * 3;
    auto png = encode_png_bgr(src, w, h, static_cast<int>(page.step));
    futs.push_back(pool.submit_with_status(std::move(png), spec_.prompt,
                                           spec_.model, spec_.max_tokens,
                                           spec_.timeout_s, spec_.base_url,
                                           spec_.api_key));
  }

  for (size_t i = 0; i < futs.size() && i < out.size(); ++i) {
    vlm::CropOutcome o = futs[i].get();
    // Don't parse a failed transport — the empty string is meaningless and
    // would otherwise look like a clean (confident) empty result downstream.
    out[i].ok   = o.ok;
    out[i].text = o.ok ? parse_one(o.text) : std::string{};
  }
  return out;
}

std::vector<formula::FormulaEngineResult>
OpenAIEndpoint::run(const GpuImage &page, const std::vector<Box> &boxes,
                    cudaStream_t stream) {
  auto crops = infer_crops(page, boxes, stream);
  std::vector<formula::FormulaEngineResult> out;
  out.reserve(crops.size());
  for (auto &c : crops)
    // hit_eos reflects a real completion: false on a failed/empty endpoint
    // call so a non-confident result isn't stamped as a clean stop.
    out.push_back(formula::FormulaEngineResult{std::move(c.text), 0, c.ok});
  return out;
}

std::vector<router::TableResult>
OpenAIEndpoint::run(const GpuImage &page, const std::vector<Box> &regions,
                    const std::vector<OCRResultItem> & /*page_ocr*/,
                    cudaStream_t stream) {
  auto crops = infer_crops(page, regions, stream);
  std::vector<router::TableResult> out;
  out.reserve(crops.size());
  for (size_t i = 0; i < crops.size(); ++i) {
    router::TableResult t;
    t.layout_id = -1;
    t.html = std::move(crops[i].text);
    // 0.0 on a failed/empty endpoint call so downstream can tell a failure
    // from a genuine empty table (default-confident is 1.0).
    t.score = crops[i].ok ? 1.0f : 0.0f;
    if (i < regions.size()) t.box = regions[i];
    out.push_back(std::move(t));
  }
  return out;
}

} // namespace turbo_ocr::backends
