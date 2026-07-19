#include "turbo_ocr/table/vlm_table.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
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
#include "turbo_ocr/vlm/crop_pool.h"

namespace turbo_ocr::table {

namespace {

using env::env_int;
using env::env_or;

bool use_pool_backend() {
  const char *v = std::getenv("VLM_BACKEND");
  return !(v && v[0] == 'l');
}

std::string to_base64(const std::vector<uint8_t> &bin) {
  size_t out_len = simdutf::base64_length_from_binary(bin.size());
  std::string out(out_len, '\0');
  simdutf::binary_to_base64(reinterpret_cast<const char *>(bin.data()),
                             bin.size(), out.data());
  return out;
}

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
    std::cerr << "[VLMTable] curl error: " << curl_easy_strerror(rc) << '\n';
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

// HTML escape for cell text.
std::string html_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&':  out += "&amp;";  break;
      case '<':  out += "&lt;";   break;
      case '>':  out += "&gt;";   break;
      case '"':  out += "&quot;"; break;
      case '\'': out += "&#39;";  break;
      default:   out += c;
    }
  }
  return out;
}

// Trim whitespace from both ends.
std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

} // namespace

// ---------------------------------------------------------------------------
// OTSL → HTML
// Port of paddlex's convert_otsl_to_html (uilts.py). Token set:
//   <fcel>  full cell  (followed by text until next token)
//   <ecel>  empty cell
//   <nl>    new row
//   <lcel>  left-merge  (extends previous cell to the right)
//   <ucel>  up-merge    (extends cell above downward)
//   <xcel>  corner merge (both left and up)
// ---------------------------------------------------------------------------

namespace {

constexpr std::string_view kFcel = "<fcel>";
constexpr std::string_view kEcel = "<ecel>";
constexpr std::string_view kNl   = "<nl>";
constexpr std::string_view kLcel = "<lcel>";
constexpr std::string_view kUcel = "<ucel>";
constexpr std::string_view kXcel = "<xcel>";

enum class OtslTok { Fcel, Ecel, Nl, Lcel, Ucel, Xcel };

struct OtslElement {
  OtslTok     tok = OtslTok::Ecel;
  std::string text;  // only meaningful for Fcel
};

// Parse OTSL into a flat element list (Fcel/Ecel/Lcel/Ucel/Xcel/Nl).
// Each Fcel absorbs any text up to the next tag.
std::vector<OtslElement> parse_otsl(const std::string &otsl) {
  std::vector<OtslElement> out;
  static const std::regex re(R"((<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>))");
  auto begin = std::sregex_iterator(otsl.begin(), otsl.end(), re);
  auto end   = std::sregex_iterator();
  std::vector<std::pair<size_t, std::string>> matches;
  for (auto it = begin; it != end; ++it) {
    matches.emplace_back(static_cast<size_t>(it->position(0)),
                         it->str(0));
  }
  for (size_t i = 0; i < matches.size(); ++i) {
    const std::string &tag = matches[i].second;
    OtslTok t;
    if (tag == kFcel) t = OtslTok::Fcel;
    else if (tag == kEcel) t = OtslTok::Ecel;
    else if (tag == kNl)   t = OtslTok::Nl;
    else if (tag == kLcel) t = OtslTok::Lcel;
    else if (tag == kUcel) t = OtslTok::Ucel;
    else                    t = OtslTok::Xcel;
    OtslElement e{t, {}};
    if (t == OtslTok::Fcel) {
      size_t text_start = matches[i].first + tag.size();
      size_t text_end   = (i + 1 < matches.size()) ? matches[i + 1].first
                                                   : otsl.size();
      if (text_end > text_start) {
        e.text = trim(otsl.substr(text_start, text_end - text_start));
      }
    }
    out.push_back(std::move(e));
  }
  return out;
}

// Pad each row to the dominant width by appending <ecel>. Mirrors
// otsl_pad_to_sqr_v2 in spirit but keeps the implementation simple: pick
// the modal row length (ties broken by the longest row).
struct Row {
  std::vector<OtslElement> cells;  // never contains Nl
};

std::vector<Row> split_rows(const std::vector<OtslElement> &elems) {
  std::vector<Row> rows;
  Row cur;
  for (const auto &e : elems) {
    if (e.tok == OtslTok::Nl) {
      if (!cur.cells.empty()) rows.push_back(std::move(cur));
      cur = {};
    } else {
      cur.cells.push_back(e);
    }
  }
  if (!cur.cells.empty()) rows.push_back(std::move(cur));
  return rows;
}

void pad_rows(std::vector<Row> &rows) {
  if (rows.empty()) return;
  size_t max_w = 0;
  for (const auto &r : rows) max_w = std::max(max_w, r.cells.size());
  for (auto &r : rows) {
    if (r.cells.empty()) {
      r.cells.resize(max_w, OtslElement{OtslTok::Ecel, ""});
      continue;
    }
    // A short row usually means the decoder truncated the trailing colspan
    // tokens of the row's LAST cell, so pad with <lcel> to left-merge the
    // missing columns into it (appending <ecel> would fabricate phantom columns
    // that shift every <ucel> rowspan below). EXCEPTION: a pure <ucel>
    // rowspan-continuation cannot root a horizontal merge, so an <lcel> chained
    // off it is malformed — in that one case pad with standalone <ecel> (same
    // column count, valid geometry). Decide once from the last *real* cell.
    const OtslTok pad_tok =
        (r.cells.back().tok == OtslTok::Ucel) ? OtslTok::Ecel : OtslTok::Lcel;
    while (r.cells.size() < max_w) {
      r.cells.push_back(OtslElement{pad_tok, ""});
    }
  }
}

bool is_l(OtslTok t) { return t == OtslTok::Lcel || t == OtslTok::Xcel; }
bool is_u(OtslTok t) { return t == OtslTok::Ucel || t == OtslTok::Xcel; }

} // namespace

std::string otsl_to_html(const std::string &otsl_in) {
  std::string otsl = trim(otsl_in);
  if (otsl.empty()) return "";
  // If no <nl> at all, treat the entire string as a single-row table.
  if (otsl.find(kNl) == std::string::npos) otsl += std::string(kNl);

  auto elems = parse_otsl(otsl);
  if (elems.empty()) return "";

  auto rows = split_rows(elems);
  if (rows.empty()) return "";

  // Cap the padded grid BEFORE pad_rows materializes it: model output fully
  // controls row count and max row width, and a sparse OTSL (one wide row +
  // many 1-cell rows) otherwise expands quadratically during padding.
  constexpr size_t kMaxTableCells = 1u << 16;
  size_t max_w = 0;
  for (const auto &r : rows) max_w = std::max(max_w, r.cells.size());
  if (max_w == 0 || rows.size() > kMaxTableCells / max_w) return "";

  pad_rows(rows);

  const size_t nrows = rows.size();
  const size_t ncols = rows.front().cells.size();
  if (ncols == 0) return "";

  // Build a 2D grid keyed by (row, col) -> origin (root cell idx + spans).
  // For each Fcel/Ecel root, compute col_span by counting Lcel/Xcel to the
  // right and row_span by counting Ucel/Xcel below in same column.
  struct CellInfo {
    bool        is_root  = false;
    bool        empty    = false;
    int         row_span = 1;
    int         col_span = 1;
    std::string text;
  };
  std::vector<std::vector<CellInfo>> grid(nrows,
      std::vector<CellInfo>(ncols, CellInfo{}));

  for (size_t r = 0; r < nrows; ++r) {
    for (size_t c = 0; c < ncols; ++c) {
      const auto &e = rows[r].cells[c];
      if (e.tok == OtslTok::Fcel || e.tok == OtslTok::Ecel) {
        CellInfo info;
        info.is_root = true;
        info.empty   = (e.tok == OtslTok::Ecel);
        info.text    = e.text;
        // Count Lcel/Xcel to the right on this row.
        size_t cc = c + 1;
        while (cc < ncols && is_l(rows[r].cells[cc].tok)) {
          info.col_span += 1;
          ++cc;
        }
        // Count Ucel/Xcel below in column c.
        size_t rr = r + 1;
        while (rr < nrows && is_u(rows[rr].cells[c].tok)) {
          info.row_span += 1;
          ++rr;
        }
        grid[r][c] = info;
      }
    }
  }

  // Emit HTML.
  std::string out = "<table>";
  for (size_t r = 0; r < nrows; ++r) {
    out += "<tr>";
    for (size_t c = 0; c < ncols; ++c) {
      const auto &g = grid[r][c];
      if (!g.is_root) continue;
      std::string tag = "<td";
      if (g.row_span > 1) tag += " rowspan=\"" + std::to_string(g.row_span) + "\"";
      if (g.col_span > 1) tag += " colspan=\"" + std::to_string(g.col_span) + "\"";
      tag += ">";
      out += tag;
      out += html_escape(g.text);
      out += "</td>";
    }
    out += "</tr>";
  }
  out += "</table>";
  return out;
}

// ---------------------------------------------------------------------------
// VLMTable
// ---------------------------------------------------------------------------

VLMTable::VLMTable() = default;
VLMTable::~VLMTable() noexcept = default;

bool VLMTable::init() {
  // Prefer table-specific env, then fall back to the shared VLLM_* envs so
  // a single endpoint can drive both modalities without duplicating config.
  const char *url_env = std::getenv("VLLM_TABLE_BASE_URL");
  base_url_ = (url_env && url_env[0])
                  ? std::string(url_env)
                  : env_or("VLLM_BASE_URL", "http://localhost:8000");
  while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();

  const char *model_env = std::getenv("VLLM_TABLE_MODEL");
  model_ = (model_env && model_env[0])
               ? std::string(model_env)
               : env_or("VLLM_MODEL", "PaddleOCR-VL-1.6-0.9B");

  prompt_     = env_or("VLLM_TABLE_PROMPT", "Table Recognition:");
  batch_      = env_int("VLLM_TABLE_BATCH", 8, 1, 1024);
  timeout_s_  = env_int("VLLM_TABLE_TIMEOUT_S", 60, 1, 3600);
  max_tokens_ = env_int("VLLM_TABLE_MAX_TOKENS", 4096, 64, 65535);

  HttpResp r = http_get(base_url_ + "/v1/models", 5);
  if (!r.ok) {
    std::cerr << "[VLMTable] /v1/models unreachable at " << base_url_
              << " (status=" << r.status << ") — tables disabled\n";
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
          std::cout << "[VLMTable] requested model='" << model_
                    << "' not found on server; using '" << first << "'\n";
          model_ = first;
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[VLMTable] /v1/models parse warning: " << e.what() << '\n';
  }

  ready_ = true;
  std::cout << "[VLMTable] ready: " << base_url_ << " model=" << model_
            << " batch=" << batch_ << " timeout=" << timeout_s_ << "s"
            << " max_tokens=" << max_tokens_ << '\n';
  return true;
}

bool VLMTable::single_request(const std::vector<uint8_t> &crop_png,
                               std::string &out_otsl) {
  std::string b64 = to_base64(crop_png);
  nlohmann::json body = {
      {"model", model_},
      {"max_tokens", max_tokens_},
      {"temperature", 0.0},
      {"top_p", 1.0},
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
    std::cerr << "[VLMTable] chat status=" << r.status
              << " body=" << r.body.substr(0, std::min<size_t>(r.body.size(), 200))
              << '\n';
    return false;
  }
  try {
    auto j = nlohmann::json::parse(r.body);
    out_otsl = j.at("choices").at(0).at("message").at("content").get<std::string>();
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[VLMTable] response parse failed: " << e.what()
              << " body=" << r.body.substr(0, std::min<size_t>(r.body.size(), 200))
              << '\n';
    return false;
  }
}

// Helper: convert raw VLM response to HTML. A model that already emits HTML
// (`<table…`) passes through untouched; only OTSL is converted. Trim first so a
// whitespace/newline-prefixed HTML response isn't mangled by otsl_to_html
// (matches backends/openai_endpoint.cpp::otsl_or_html).
static std::string otsl_or_html(const std::string &raw) {
  if (trim(raw).rfind("<table", 0) == 0) return raw;
  return raw.empty() ? "" : otsl_to_html(raw);
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
    std::cerr << "[VLMTable] page D2H failed\n";
    return out;
  }
  if (cudaError_t serr = cudaStreamSynchronize(stream); serr != cudaSuccess) {
    std::cerr << "[VLMTable] page D2H stream sync failed: "
              << cudaGetErrorString(serr) << '\n';
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
          crops_png[i] = encode_png_bgr(src, cr[2], cr[3], (int)page.step);
        }
      });
    }
    for (auto &t : enc_threads) t.join();
  }

  out.resize(n);

  if (use_pool_backend()) {
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
    std::cerr << "[VLMTable] submit_async page D2H stream sync failed: "
              << cudaGetErrorString(serr) << '\n';
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
          crops_png[i] = encode_png_bgr(src, cr[2], cr[3], (int)page.step);
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
