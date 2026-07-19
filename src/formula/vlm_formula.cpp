#include "turbo_ocr/formula/vlm_formula.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "simdutf.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/vlm/crop_pool.h"

namespace turbo_ocr::formula {

namespace {

using env::env_int;
using env::env_or;

bool use_pool_backend() {
  const char *v = std::getenv("VLM_BACKEND");
  // Default to pool; legacy only if explicitly set.
  return !(v && v[0] == 'l');  // "legacy" starts with 'l'
}

std::string to_base64(const std::vector<uint8_t> &bin) {
  size_t out_len = simdutf::base64_length_from_binary(bin.size());
  std::string out(out_len, '\0');
  simdutf::binary_to_base64(reinterpret_cast<const char *>(bin.data()),
                             bin.size(), out.data());
  return out;
}

// Initialize libcurl exactly once.
struct CurlGlobalInit {
  CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobalInit() { curl_global_cleanup(); }
};
CurlGlobalInit g_curl_init;

size_t curl_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
  auto *buf = static_cast<std::string *>(userdata);
  buf->append(ptr, size * nmemb);
  return size * nmemb;
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

struct HttpResp {
  bool        ok      = false;
  long        status  = 0;
  std::string body;
};

HttpResp http_post_json(const std::string &url, const std::string &json_body,
                        int timeout_s) {
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
  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r.status);
    r.ok = (r.status >= 200 && r.status < 300);
  } else {
    TOCR_LOG_ERROR_RL("VLMFormula curl error", "error", curl_easy_strerror(rc));
  }
  return r;
}

HttpResp http_get(const std::string &url, int timeout_s) {
  HttpResp r;
  CurlEasy curl;
  if (!curl) return r;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, (long)timeout_s);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, kConnectTimeoutS);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &r.body);
  CURLcode rc = curl_easy_perform(curl);
  if (rc == CURLE_OK) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &r.status);
    r.ok = (r.status >= 200 && r.status < 300);
  }
  return r;
}

// Pull LaTeX out of the assistant message.
std::string extract_latex(const std::string &msg) {
  static const std::regex re_fence(R"(```(?:latex|tex|math)?\s*\n?([\s\S]*?)```)",
                                    std::regex::ECMAScript);
  std::smatch m;
  if (std::regex_search(msg, m, re_fence) && m.size() >= 2) {
    std::string s = m[1].str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
      s.pop_back();
    return s;
  }
  static const std::regex re_disp(R"(\$\$([\s\S]*?)\$\$)");
  if (std::regex_search(msg, m, re_disp) && m.size() >= 2) return m[1].str();
  static const std::regex re_brk(R"(\\\[([\s\S]*?)\\\])");
  if (std::regex_search(msg, m, re_brk) && m.size() >= 2) return m[1].str();
  static const std::regex re_inline(R"(\$([^\$\n]+)\$)");
  if (std::regex_search(msg, m, re_inline) && m.size() >= 2) return m[1].str();
  std::string s = msg;
  while (!s.empty() && (s.front() == ' ' || s.front() == '\n' || s.front() == '\r'))
    s.erase(s.begin());
  while (!s.empty() && (s.back() == ' ' || s.back() == '\n' || s.back() == '\r'))
    s.pop_back();
  for (auto pre : {"LaTeX:", "Latex:", "latex:", "Answer:", "answer:"}) {
    if (s.rfind(pre, 0) == 0) { s.erase(0, std::strlen(pre)); break; }
  }
  while (!s.empty() && (s.front() == ' ' || s.front() == '\n')) s.erase(s.begin());
  return s;
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

} // namespace

VLMFormula::VLMFormula() = default;
VLMFormula::~VLMFormula() noexcept = default;

bool VLMFormula::load_model_dir(const std::string &/*model_dir*/) {
  base_url_   = env_or("VLLM_BASE_URL", "http://localhost:8000");
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
  // Default to PaddleOCR-VL-1.6 (96.33% OmniDocBench, architecturally
  // identical to 1.5 → drop-in). Its formula head is trained on the exact
  // "Formula Recognition:" prompt; a free-form instruction degrades it. Point
  // VLLM_MODEL/VLLM_FORMULA_PROMPT at a MiniCPM endpoint to override.
  model_      = env_or("VLLM_MODEL", "PaddleOCR-VL-1.6-0.9B");
  prompt_     = env_or("VLLM_FORMULA_PROMPT", "Formula Recognition:");
  batch_      = env_int("VLLM_FORMULA_BATCH", 8, 1, 1024);
  timeout_s_  = env_int("VLLM_FORMULA_TIMEOUT_S", 30, 1, 3600);
  max_tokens_ = env_int("VLLM_FORMULA_MAX_TOKENS", 512, 16, 65535);
  png_threads_ = env_int("VLM_PNG_THREADS", 4, 1, 256);

  HttpResp r = http_get(base_url_ + "/v1/models", 5);
  if (!r.ok) {
    TOCR_LOG_ERROR("VLMFormula /v1/models unreachable, formulas disabled", "base_url", base_url_, "status", r.status);
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
      std::string first = j["data"][0].value("id", "");
      if (!first.empty()) {
        if (!std::getenv("VLLM_MODEL") || std::getenv("VLLM_MODEL")[0] == '\0') {
          model_ = first;
        }
        TOCR_LOG_INFO("VLMFormula endpoint probed", "base_url", base_url_, "server_model", first, "using_model", model_);
      }
    }
  } catch (const std::exception &e) {
    TOCR_LOG_WARN("VLMFormula /v1/models parse warning", "error", e.what());
  }

  ready_ = true;
  const char *backend = use_pool_backend() ? "pool" : "legacy";
  TOCR_LOG_INFO("VLMFormula ready", "base_url", base_url_, "model", model_, "batch", batch_, "timeout_s", timeout_s_, "max_tokens", max_tokens_, "backend", backend, "png_threads", png_threads_);
  return true;
}

bool VLMFormula::load_tokenizer(const std::string &/*path*/) { return true; }

bool VLMFormula::single_request(const std::vector<uint8_t> &crop_png,
                                 std::string &out_latex) {
  std::string b64 = to_base64(crop_png);
  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_},
      {"temperature", 0.0},
      {"messages", nlohmann::json::array({
          {{"role", "user"},
           {"content", nlohmann::json::array({
               make_image_block(b64),
               {{"type", "text"}, {"text", prompt_}},
           })}},
      })},
  };
  HttpResp r = http_post_json(base_url_ + "/v1/chat/completions",
                              body.dump(), timeout_s_);
  if (!r.ok) {
    TOCR_LOG_ERROR_RL("VLMFormula chat error", "status", r.status, "body", r.body.substr(0, std::min<size_t>(r.body.size(), 200)));
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    std::string msg = j.at("choices").at(0).at("message").at("content").get<std::string>();
    out_latex = extract_latex(msg);
    return true;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("VLMFormula response parse failed", "error", e.what(), "body", r.body.substr(0, std::min<size_t>(r.body.size(), 200)));
    return false;
  }
}

bool VLMFormula::batched_request(
    const std::vector<std::vector<uint8_t>> &crops_png,
    std::vector<std::string> &out_latex) {
  out_latex.clear();
  if (crops_png.empty()) return true;

  nlohmann::json content = nlohmann::json::array();
  for (size_t i = 0; i < crops_png.size(); ++i) {
    content.push_back(make_image_block(to_base64(crops_png[i])));
  }
  std::string multi_prompt =
      "You will see " + std::to_string(crops_png.size()) +
      " formula images in order. Extract the LaTeX of each. Output ONE LaTeX "
      "expression per image, each on its own line wrapped in ``` fences, in "
      "the same order as the images. No commentary, no numbering.";
  content.push_back({{"type", "text"}, {"text", multi_prompt}});

  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_ * (int)crops_png.size()},
      {"temperature", 0.0},
      {"messages", nlohmann::json::array({
          {{"role", "user"}, {"content", content}},
      })},
  };
  HttpResp r = http_post_json(base_url_ + "/v1/chat/completions",
                              body.dump(), timeout_s_ * std::max(1, (int)crops_png.size() / 4));
  if (!r.ok) {
    TOCR_LOG_ERROR_RL("VLMFormula batched chat error, falling back to per-crop", "status", r.status, "body", r.body.substr(0, std::min<size_t>(r.body.size(), 200)));
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    std::string msg = j.at("choices").at(0).at("message").at("content").get<std::string>();
    static const std::regex re_fence(R"(```(?:latex|tex|math)?\s*\n?([\s\S]*?)```)");
    auto begin = std::sregex_iterator(msg.begin(), msg.end(), re_fence);
    auto end   = std::sregex_iterator();
    std::vector<std::string> hits;
    for (auto it = begin; it != end; ++it) {
      std::string s = (*it)[1].str();
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
      hits.push_back(std::move(s));
    }
    if (hits.size() == crops_png.size()) {
      out_latex = std::move(hits);
      return true;
    }
    TOCR_LOG_WARN_RL("VLMFormula batched fence-count mismatch, falling back to per-crop", "got", hits.size(), "expected", crops_png.size());
    return false;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("VLMFormula batched parse failed", "error", e.what());
    return false;
  }
}

// ---------------------------------------------------------------------------
// run() — pool backend
// ---------------------------------------------------------------------------

std::vector<FormulaEngineResult>
VLMFormula::run_pool(const std::vector<uint8_t> &host_page,
                     const GpuImage &page,
                     const std::vector<Box> &boxes) {
  std::vector<FormulaEngineResult> out(boxes.size());
  const int n = static_cast<int>(boxes.size());
  const int workers = std::min(png_threads_, n);

  // Parallel PNG encode.
  std::vector<std::vector<uint8_t>> crops_png(n);
  if (workers <= 1) {
    for (int i = 0; i < n; ++i) {
      auto cr = clamped_crop_rect(boxes[i], page.cols, page.rows);
      const uint8_t *src = host_page.data() + (size_t)cr[1] * page.step + (size_t)cr[0] * 3;
      crops_png[i] = encode_png_bgr(src, cr[2], cr[3], (int)page.step);
    }
  } else {
    std::vector<std::thread> enc_threads;
    enc_threads.reserve(workers);
    std::atomic<int> next{0};
    for (int t = 0; t < workers; ++t) {
      enc_threads.emplace_back([&, t] {
        for (int i = next.fetch_add(1, std::memory_order_relaxed); i < n;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
          auto cr = clamped_crop_rect(boxes[i], page.cols, page.rows);
          const uint8_t *src = host_page.data() + (size_t)cr[1] * page.step + (size_t)cr[0] * 3;
          crops_png[i] = encode_png_bgr(src, cr[2], cr[3], (int)page.step);
        }
      });
    }
    for (auto &t : enc_threads) t.join();
  }

  // Submit all crops to the global pool; collect futures.
  auto &pool = turbo_ocr::vlm::VLMCropPool::instance();
  std::vector<std::future<std::string>> futs;
  futs.reserve(n);
  for (int i = 0; i < n; ++i) {
    futs.push_back(pool.submit(
        std::move(crops_png[i]),
        prompt_, model_, max_tokens_, timeout_s_, base_url_, std::string()));
  }

  // Collect results.
  for (int i = 0; i < n; ++i) {
    std::string raw = futs[i].get();
    FormulaEngineResult res;
    res.latex = extract_latex(raw);
    res.token_count = res.latex.size();
    res.hit_eos = !res.latex.empty();
    // A transport failure resolves the pool future to "" -> empty latex; mark
    // the region failed so the sync pipeline flags degradation rather than
    // treating it as a formula-free region.
    res.ok = !res.latex.empty();
    out[i] = std::move(res);
  }
  return out;
}

// ---------------------------------------------------------------------------
// submit_async() — submit crops to pool, return futures immediately
// ---------------------------------------------------------------------------

std::vector<std::future<std::string>>
VLMFormula::submit_async(const std::vector<uint8_t> &host_page,
                         const GpuImage &page,
                         const std::vector<Box> &boxes) {
  const int n = static_cast<int>(boxes.size());
  std::vector<std::future<std::string>> futs;
  futs.reserve(n);
  if (n == 0) return futs;

  const int workers = std::min(png_threads_, n);
  std::vector<std::vector<uint8_t>> crops_png(n);

  // Parallel PNG encode (same as run_pool).
  if (workers <= 1) {
    for (int i = 0; i < n; ++i) {
      auto cr = clamped_crop_rect(boxes[i], page.cols, page.rows);
      const uint8_t *src = host_page.data() + (size_t)cr[1] * page.step + (size_t)cr[0] * 3;
      crops_png[i] = encode_png_bgr(src, cr[2], cr[3], (int)page.step);
    }
  } else {
    std::vector<std::thread> enc_threads;
    enc_threads.reserve(workers);
    std::atomic<int> next{0};
    for (int t = 0; t < workers; ++t) {
      enc_threads.emplace_back([&, t] {
        for (int i = next.fetch_add(1, std::memory_order_relaxed); i < n;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
          auto cr = clamped_crop_rect(boxes[i], page.cols, page.rows);
          const uint8_t *src = host_page.data() + (size_t)cr[1] * page.step + (size_t)cr[0] * 3;
          crops_png[i] = encode_png_bgr(src, cr[2], cr[3], (int)page.step);
        }
      });
    }
    for (auto &t : enc_threads) t.join();
  }

  // Submit to global pool without blocking.
  auto &pool = turbo_ocr::vlm::VLMCropPool::instance();
  for (int i = 0; i < n; ++i) {
    futs.push_back(pool.submit(
        std::move(crops_png[i]),
        prompt_, model_, max_tokens_, timeout_s_, base_url_, std::string()));
  }
  return futs;
}

// ---------------------------------------------------------------------------
// IFormulaRecognizer async decouple — D2H on the GPU worker, then non-blocking
// submit. The expensive HTTP await happens later, off the GPU worker, when the
// caller resolves the futures + calls parse_async_result.
// ---------------------------------------------------------------------------

bool VLMFormula::supports_async() const noexcept {
  return ready_ && use_pool_backend();
}

std::vector<std::future<std::string>>
VLMFormula::submit_async(const GpuImage &page, const std::vector<Box> &boxes,
                         cudaStream_t stream) {
  std::vector<std::future<std::string>> futs;
  if (boxes.empty() || page.empty() || !ready_) return futs;

  // D2H the page once (same as run()); the PNG-encode + pool submit then
  // reference only this host copy + pool-owned bytes — gpu_img can be freed
  // by the caller immediately after this returns.
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  std::vector<uint8_t> host_page(need);
  if (cudaSuccess != cudaMemcpyAsync(host_page.data(), page.data, need,
                                     cudaMemcpyDeviceToHost, stream)) {
    TOCR_LOG_ERROR_RL("VLMFormula async page D2H failed");
    return futs;
  }
  if (cudaError_t serr = cudaStreamSynchronize(stream); serr != cudaSuccess) {
    TOCR_LOG_ERROR_RL("VLMFormula page D2H sync failed", "cuda", cudaGetErrorString(serr));
    return futs;
    return futs;
  }
  return submit_async(host_page, page, boxes);
}

std::string VLMFormula::parse_async_result(const std::string &raw) const {
  return extract_latex(raw);
}

std::function<std::string(const std::string &)>
VLMFormula::async_result_parser() const {
  // extract_latex is a free function (no recognizer state), so the snapshot
  // holds no pointer back into this object.
  return [](const std::string &raw) { return extract_latex(raw); };
}

// ---------------------------------------------------------------------------
// run() — legacy backend (original code, no call_mu_ on outer scope)
// ---------------------------------------------------------------------------

std::vector<FormulaEngineResult>
VLMFormula::run_legacy(const std::vector<uint8_t> &host_page,
                       const GpuImage &page,
                       const std::vector<Box> &boxes) {
  std::vector<FormulaEngineResult> out(boxes.size());

  // PNG-encode every crop up front.
  std::vector<std::vector<uint8_t>> crops_png;
  crops_png.reserve(boxes.size());
  for (const auto &b : boxes) {
    auto cr = clamped_crop_rect(b, page.cols, page.rows);
    const uint8_t *src = host_page.data() + (size_t)cr[1] * page.step + (size_t)cr[0] * 3;
    crops_png.push_back(encode_png_bgr(src, cr[2], cr[3], (int)page.step));
  }

  struct Job {
    size_t off = 0;
    std::vector<std::vector<uint8_t>> chunk;
    std::vector<std::string> results;
  };
  std::vector<Job> jobs;
  for (size_t off = 0; off < crops_png.size(); off += (size_t)batch_) {
    size_t end = std::min(crops_png.size(), off + (size_t)batch_);
    jobs.push_back({off, {crops_png.begin() + off, crops_png.begin() + end}, {}});
  }
  std::vector<std::thread> workers;
  workers.reserve(jobs.size());
  for (auto &job : jobs) {
    workers.emplace_back([this, &job] {
      if (!batched_request(job.chunk, job.results)) {
        job.results.assign(job.chunk.size(), std::string{});
        std::vector<std::thread> inner;
        inner.reserve(job.chunk.size());
        for (size_t i = 0; i < job.chunk.size(); ++i) {
          inner.emplace_back([this, &job, i] {
            single_request(job.chunk[i], job.results[i]);
          });
        }
        for (auto &t : inner) t.join();
      }
    });
  }
  for (auto &t : workers) t.join();
  for (auto &job : jobs) {
    for (size_t i = 0; i < job.chunk.size(); ++i) {
      FormulaEngineResult r;
      r.latex = std::move(job.results[i]);
      r.token_count = r.latex.size();
      r.hit_eos = !r.latex.empty();
      r.ok = !r.latex.empty();  // empty == request failed for this crop
      out[job.off + i] = std::move(r);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// run() — dispatch
// ---------------------------------------------------------------------------

std::vector<FormulaEngineResult>
VLMFormula::run(const GpuImage &page, const std::vector<Box> &boxes,
                cudaStream_t stream) {
  std::vector<FormulaEngineResult> out;
  if (boxes.empty()) return out;
  if (page.empty()) {
    TOCR_LOG_WARN_RL("VLMFormula empty page");
    return out;
  }
  if (!ready_) {
    TOCR_LOG_WARN_RL("VLMFormula not ready");
    return out;
  }

  // D2H copy into a local buffer — no class-level lock needed.
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  std::vector<uint8_t> host_page(need);
  if (cudaSuccess !=
      cudaMemcpyAsync(host_page.data(), page.data, need,
                      cudaMemcpyDeviceToHost, stream)) {
    TOCR_LOG_ERROR_RL("VLMFormula page D2H failed");
    return out;
  }
  if (cudaError_t serr = cudaStreamSynchronize(stream); serr != cudaSuccess) {
    TOCR_LOG_ERROR_RL("VLMFormula page D2H sync failed", "cuda", cudaGetErrorString(serr));
    return out;
  }

  if (use_pool_backend()) {
    return run_pool(host_page, page, boxes);
  } else {
    return run_legacy(host_page, page, boxes);
  }
}

} // namespace turbo_ocr::formula
