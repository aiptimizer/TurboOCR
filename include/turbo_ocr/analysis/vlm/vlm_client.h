#pragma once

// Shared transport + encode plumbing for the VLM table and formula
// clients: one copy of the curl HTTP helpers, base64, PNG crop encode and
// the chat-completions image block. Runtime-opt-in like its two users.

#include "turbo_ocr/base/encoding.h"
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace turbo_ocr::vlm {

// Default to pool; "legacy" (prefix 'l') selects the per-request curl path.
[[nodiscard]] bool use_pool_backend();

// Reference libcurl's process-wide init (RAII global in vlm_client.cpp).
// Call from any TU that uses curl directly so the linker cannot dead-strip
// the initializer object out of the static archive.
void ensure_curl_init();

struct HttpResp {
  bool        ok      = false;
  long        status  = 0;
  std::string body;
};

// POST json_body to url. `log_tag` names the calling client in the
// rate-limited curl error log (keeps the pre-dedup per-client messages).
[[nodiscard]] HttpResp http_post_json(const std::string &url,
                                      const std::string &json_body,
                                      int timeout_s, const char *log_tag);

// Optional `bearer` adds 'Authorization: Bearer <token>' (auth-gated
// endpoints must not be silently disabled at boot despite a valid key).
[[nodiscard]] HttpResp http_get(const std::string &url, int timeout_s,
                                const std::string &bearer = "");


// PNG-encode a BGR crop. Compression 0 = fastest; crops are tiny, net is
// loopback.
[[nodiscard]] std::vector<uint8_t> encode_png_bgr(const uint8_t *data,
                                                  int w, int h, int stride);

[[nodiscard]] nlohmann::json make_image_block(const std::string &b64_png);

} // namespace turbo_ocr::vlm
