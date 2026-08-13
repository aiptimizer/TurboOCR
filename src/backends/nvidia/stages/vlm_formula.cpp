#include "nvidia/stages/vlm_formula.h"

#include "turbo_ocr/analysis/formula/latex_extract.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "nvidia/support/cuda_check.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/analysis/vlm/crop_pool.h"
#include "turbo_ocr/analysis/vlm/vlm_client.h"

namespace turbo_ocr::formula {

namespace {

using env::env_int;
using env::env_or;

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

  vlm::HttpResp r = vlm::http_get(base_url_ + "/v1/models", 5);
  if (!r.ok) {
    TOCR_LOG_ERROR("VLMFormula /v1/models unreachable, formulas disabled", "base_url", base_url_, "status", r.status);
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
      std::string first = j["data"][0].value("id", "");
      if (!first.empty()) {
        if (!env::env_present("VLLM_MODEL")) {
          model_ = first;
        }
        TOCR_LOG_INFO("VLMFormula endpoint probed", "base_url", base_url_, "server_model", first, "using_model", model_);
      }
    }
  } catch (const std::exception &e) {
    TOCR_LOG_WARN("VLMFormula /v1/models parse warning", "error", e.what());
  }

  ready_ = true;
  const char *backend = vlm::use_pool_backend() ? "pool" : "legacy";
  TOCR_LOG_INFO("VLMFormula ready", "base_url", base_url_, "model", model_, "batch", batch_, "timeout_s", timeout_s_, "max_tokens", max_tokens_, "backend", backend, "png_threads", png_threads_);
  return true;
}

bool VLMFormula::load_tokenizer(const std::string &/*path*/) { return true; }

bool VLMFormula::single_request(const std::vector<uint8_t> &crop_png,
                                 std::string &out_latex) {
  std::string b64 = base64_encode(crop_png.data(), crop_png.size());
  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_},
      {"temperature", 0.0},
      {"messages", nlohmann::json::array({
          {{"role", "user"},
           {"content", nlohmann::json::array({
               vlm::make_image_block(b64),
               {{"type", "text"}, {"text", prompt_}},
           })}},
      })},
  };
  vlm::HttpResp r = vlm::http_post_json(base_url_ + "/v1/chat/completions",
                              body.dump(), timeout_s_,
                              "VLMFormula curl error");
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
    content.push_back(vlm::make_image_block(base64_encode(crops_png[i].data(), crops_png[i].size())));
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
  vlm::HttpResp r = vlm::http_post_json(base_url_ + "/v1/chat/completions",
                              body.dump(), timeout_s_ * std::max(1, (int)crops_png.size() / 4),
                              "VLMFormula curl error");
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
      crops_png[i] = vlm::encode_png_bgr(src, cr[2], cr[3], (int)page.step);
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
          crops_png[i] = vlm::encode_png_bgr(src, cr[2], cr[3], (int)page.step);
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
      crops_png[i] = vlm::encode_png_bgr(src, cr[2], cr[3], (int)page.step);
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
          crops_png[i] = vlm::encode_png_bgr(src, cr[2], cr[3], (int)page.step);
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
  return ready_ && vlm::use_pool_backend();
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
    crops_png.push_back(vlm::encode_png_bgr(src, cr[2], cr[3], (int)page.step));
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

  if (vlm::use_pool_backend()) {
    return run_pool(host_page, page, boxes);
  } else {
    return run_legacy(host_page, page, boxes);
  }
}

} // namespace turbo_ocr::formula
