// amd_backend_registry.cpp — registers the AMD (HIP + MIGraphX) backend into the
// ONE shared link-time registry (src/backend/backend_registry.cpp).
//
// Pure registration, mirroring src/backends/cpu/backend/cpu_backend_registry.cpp. The TU no
// longer defines backend::make_backend / available_backends.
//
// AUTO-DETECT: make_rocm_backend() probes for a HIP device (hipGetDeviceCount)
// and returns nullptr when none is present. The shared registry treats a null
// factory result as "compiled in but unusable here" and walks down to the next
// priority — ultimately the host backend — so an AMD-enabled binary on a
// ROCm-less box still boots.

#include "amd/backend/rocm_backend.h"
#include "turbo_ocr/backend/backend_registry.h"

namespace turbo_ocr::amd {
namespace {

std::unique_ptr<backend::Backend> make_amd_backend_entry() {
  return make_rocm_backend(); // nullptr when no HIP device
}

const backend::BackendRegistrar g_amd_registrar{
    "amd", {"rocm", "hip"}, backend::kBackendPriorityAmd, &make_amd_backend_entry};

} // namespace
} // namespace turbo_ocr::amd
