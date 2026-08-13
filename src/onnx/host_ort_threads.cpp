// host_ort_threads.cpp — see host_ort_threads.h.

#include "turbo_ocr/onnx/host_ort_threads.h"

#include <cstdlib>

namespace turbo_ocr {
namespace {
// Set during load_stages(), read when each stage session is constructed. Both
// happen on the bootstrap thread before the server accepts traffic, so a plain
// int is the honest representation — an atomic here would imply a concurrency
// story that does not exist.
int g_backend_hint = 0;
} // namespace

void set_host_ort_intra_op_threads(int n) noexcept { g_backend_hint = n; }

int host_ort_intra_op_threads(int stage_default) noexcept {
  // 1. The operator's existing override wins everywhere, unchanged.
  if (const char *env = std::getenv("ORT_NUM_THREADS")) { // pre-commit-allow-getenv
    const int n = std::atoi(env);
    if (n > 0) return n;
  }
  // 2. The backend's "my host is idle" hint.
  if (g_backend_hint > 0) return g_backend_hint;
  // 3. Whatever this stage did before.
  return stage_default;
}

} // namespace turbo_ocr
