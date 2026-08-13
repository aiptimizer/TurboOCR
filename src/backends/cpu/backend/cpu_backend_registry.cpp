// cpu_backend_registry.cpp — registers the CPU (host) backend into the ONE
// shared link-time registry (src/backend/backend_registry.cpp).
//
// This file used to DEFINE backend::make_backend / available_backends, which is
// why only one vendor registry could ever be linked into a binary. It is now
// pure registration: a single namespace-scope BackendRegistrar whose constructor
// runs at static init. Link this TU together with apple/nvidia/... registration
// TUs and the resulting binary holds every one of them, selectable at runtime
// via --backend / TURBO_BACKEND.
//
// The host backend registers at priority 0 — always constructible, so it is the
// auto-detect fallback of last resort and must never shadow a real accelerator.

#include "cpu/backend/cpu_backend.h"
#include "turbo_ocr/backend/backend_registry.h"

namespace turbo_ocr::cpu {
namespace {

std::unique_ptr<backend::Backend> make_cpu_backend() {
  return std::make_unique<CpuBackend>();
}

const backend::BackendRegistrar g_cpu_registrar{
    "cpu", {"host"}, backend::kBackendPriorityCpu, &make_cpu_backend};

} // namespace
} // namespace turbo_ocr::cpu
