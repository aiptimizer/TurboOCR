#pragma once

// Internals shared by the VLM crop-pool TUs (crop_pool.cpp,
// crop_pool_transport.cpp). Not a public API.

#include <chrono>

namespace turbo_ocr::vlm {

inline long steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace turbo_ocr::vlm
