#include "turbo_ocr/base/string_utils.h"
#include "nvidia/stages/vlm_table.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "nvidia/support/cuda_check.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/analysis/table/html_reconstruct.h"
#include "turbo_ocr/analysis/vlm/crop_pool.h"
#include "turbo_ocr/analysis/vlm/openai_policy.h" // the ONE otsl_or_html
#include "turbo_ocr/analysis/vlm/vlm_client.h"

namespace turbo_ocr::table {

namespace {

using env::env_int;
using env::env_or;

// Trim whitespace from both ends.

} // namespace

// ---------------------------------------------------------------------------
// VLMTable
// ---------------------------------------------------------------------------

VLMTable::VLMTable() = default;
VLMTable::~VLMTable() noexcept = default;

bool VLMTable::init() {
  // Prefer table-specific env, then fall back to the shared VLLM_* envs so
  // a single endpoint can drive both modalities without duplicating config.
  base_url_ = env_or("VLLM_TABLE_BASE_URL",
                     env_or("VLLM_BASE_URL", "http://localhost:8000"));
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();

  model_ = env_or("VLLM_TABLE_MODEL",
                  env_or("VLLM_MODEL", "PaddleOCR-VL-1.6-0.9B"));

  prompt_     = env_or("VLLM_TABLE_PROMPT", "Table Recognition:");
  batch_      = env_int("VLLM_TABLE_BATCH", 8, 1, 1024);
  timeout_s_  = env_int("VLLM_TABLE_TIMEOUT_S", 60, 1, 3600);
  max_tokens_ = env_int("VLLM_TABLE_MAX_TOKENS", 4096, 64, 65535);

  vlm::HttpResp r = vlm::http_get(base_url_ + "/v1/models", 5);
  if (!r.ok) {
    TOCR_LOG_ERROR("VLMTable /v1/models unreachable, tables disabled", "base_url", base_url_, "status", r.status);
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
      // If VLLM_MODEL was unset and we defaulted to the canonical name, but
      // the server registered a different id, adopt the first registered id.
      bool model_present = false;
      for (const auto &m : j["data"]) {
        if (m.value("id", "") == model_) { model_present = true; break; }
      }
      if (!model_present) {
        std::string first = j["data"][0].value("id", "");
        if (!first.empty()) {
          TOCR_LOG_INFO("VLMTable requested model not on server, adopting registered id",
                        "requested", model_, "using", first);
          model_ = first;
        }
      }
    }
  } catch (const std::exception &e) {
    TOCR_LOG_WARN("VLMTable /v1/models parse warning", "error", e.what());
  }

  ready_ = true;
  TOCR_LOG_INFO("VLMTable ready", "base_url", base_url_, "model", model_, "batch", batch_, "timeout_s", timeout_s_, "max_tokens", max_tokens_);
  return true;
}

bool VLMTable::single_request(const std::vector<uint8_t> &crop_png,
                               std::string &out_otsl) {
  std::string b64 = base64_encode(crop_png.data(), crop_png.size());
  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_},
      {"temperature", 0.0},
      {"top_p", 1.0},
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
                              "VLMTable curl error");
  if (!r.ok) {
    TOCR_LOG_ERROR_RL("VLMTable chat error", "status", r.status, "body", r.body.substr(0, std::min<size_t>(r.body.size(), 200)));
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    out_otsl = j.at("choices").at(0).at("message").at("content").get<std::string>();
    return true;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("VLMTable response parse failed", "error", e.what(), "body", r.body.substr(0, std::min<size_t>(r.body.size(), 200)));
    return false;
  }
}

// Convert raw VLM response to HTML. ONE definition, in the shared endpoint
// policy: this used to be a second, hand-maintained copy of
// openai_policy::otsl_or_html, each documented as "mirrors the other". The
// `<table` branch is a SECURITY control (model-emitted HTML is semi-trusted —
// an adversarial page can steer it — so the passthrough is sanitized), and two
// copies of a security control is one hardening away from leaving the other
// exploitable.
static std::string otsl_or_html(const std::string &raw) {
  return vlm::openai_policy::otsl_or_html(raw);
}

std::vector<std::string>
VLMTable::run(const GpuImage &page, const std::vector<Box> &regions,
              cudaStream_t stream) {
  std::vector<std::string> out;
  if (regions.empty()) return out;
  if (page.empty()) return out;
  if (!ready_) return out;

  // D2H into a local buffer — no class-level lock.
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  std::vector<uint8_t> host_page(need);
  if (cudaSuccess != cudaMemcpyAsync(host_page.data(), page.data, need,
                                      cudaMemcpyDeviceToHost, stream)) {
    TOCR_LOG_ERROR_RL("VLMTable page D2H failed");
    return out;
  }
  if (cudaError_t serr = cudaStreamSynchronize(stream); serr != cudaSuccess) {
    TOCR_LOG_ERROR_RL("VLMTable page D2H stream sync failed", "cuda", cudaGetErrorString(serr));
    return out;
  }

  const int n = static_cast<int>(regions.size());
  const int png_threads = env_int("VLM_PNG_THREADS", 4, 1, 256);

  // PNG-encode all crops in parallel.
  std::vector<std::vector<uint8_t>> crops_png(n);
  {
    std::atomic<int> next{0};
    const int workers = std::min(png_threads, n);
    std::vector<std::thread> enc_threads;
    enc_threads.reserve(workers);
    for (int t = 0; t < workers; ++t) {
      enc_threads.emplace_back([&] {
        for (int i = next.fetch_add(1, std::memory_order_relaxed); i < n;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
          auto cr = clamped_crop_rect(regions[i], page.cols, page.rows);
          const uint8_t *src =
              host_page.data() + (size_t)cr[1] * page.step + (size_t)cr[0] * 3;
          crops_png[i] = vlm::encode_png_bgr(src, cr[2], cr[3], (int)page.step);
        }
      });
    }
    for (auto &t : enc_threads) t.join();
  }

  out.resize(n);

  if (vlm::use_pool_backend()) {
    // Submit all crops to the global pool; collect futures.
    auto &pool = turbo_ocr::vlm::VLMCropPool::instance();
    std::vector<std::future<std::string>> futs;
    futs.reserve(n);
    for (int i = 0; i < n; ++i) {
      futs.push_back(pool.submit(
          std::move(crops_png[i]),
          prompt_, model_, max_tokens_, timeout_s_, base_url_, std::string()));
    }
    for (int i = 0; i < n; ++i) {
      out[i] = otsl_or_html(futs[i].get());
    }
  } else {
    // Legacy: capped parallel threads per-page.
    const int max_inflight = std::min<int>(batch_, n);
    for (int off = 0; off < n; off += max_inflight) {
      int end = std::min(n, off + max_inflight);
      std::vector<std::thread> workers;
      workers.reserve(end - off);
      for (int i = off; i < end; ++i) {
        workers.emplace_back([this, i, &crops_png, &out] {
          std::string otsl;
          if (single_request(crops_png[i], otsl)) {
            out[i] = otsl_or_html(otsl);
          } else {
            out[i] = "";
          }
        });
      }
      for (auto &t : workers) t.join();
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// submit_async() — D2H + PNG-encode + submit futures, no blocking
// ---------------------------------------------------------------------------

std::vector<std::future<std::string>>
VLMTable::submit_async(const GpuImage &page,
                       const std::vector<Box> &regions,
                       cudaStream_t stream) {
  std::vector<std::future<std::string>> futs;
  const int n = static_cast<int>(regions.size());
  futs.reserve(n);
  if (n == 0 || !ready_ || page.empty()) return futs;

  // D2H copy (same as run()).
  const size_t need = static_cast<size_t>(page.rows) * page.step;
  std::vector<uint8_t> host_page(need);
  if (cudaSuccess != cudaMemcpyAsync(host_page.data(), page.data, need,
                                      cudaMemcpyDeviceToHost, stream)) {
    return futs;
  }
  if (cudaError_t serr = cudaStreamSynchronize(stream); serr != cudaSuccess) {
    TOCR_LOG_ERROR_RL("VLMTable submit_async page D2H stream sync failed", "cuda", cudaGetErrorString(serr));
    return futs;
  }

  const int png_threads = env_int("VLM_PNG_THREADS", 4, 1, 256);
  const int workers = std::min(png_threads, n);
  std::vector<std::vector<uint8_t>> crops_png(n);
  {
    std::atomic<int> next{0};
    std::vector<std::thread> enc_threads;
    enc_threads.reserve(workers);
    for (int t = 0; t < workers; ++t) {
      enc_threads.emplace_back([&] {
        for (int i = next.fetch_add(1, std::memory_order_relaxed); i < n;
             i = next.fetch_add(1, std::memory_order_relaxed)) {
          auto cr = clamped_crop_rect(regions[i], page.cols, page.rows);
          const uint8_t *src =
              host_page.data() + (size_t)cr[1] * page.step + (size_t)cr[0] * 3;
          crops_png[i] = vlm::encode_png_bgr(src, cr[2], cr[3], (int)page.step);
        }
      });
    }
    for (auto &t : enc_threads) t.join();
  }

  auto &pool = turbo_ocr::vlm::VLMCropPool::instance();
  for (int i = 0; i < n; ++i) {
    futs.push_back(pool.submit(
        std::move(crops_png[i]),
        prompt_, model_, max_tokens_, timeout_s_, base_url_, std::string()));
  }
  return futs;
}

} // namespace turbo_ocr::table
