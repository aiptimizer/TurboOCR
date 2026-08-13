// nv_backend_registry.cpp — registers the NVIDIA (CUDA/TensorRT) backend into the
// ONE shared link-time registry (src/backend/backend_registry.cpp).
//
// Pure registration now: this TU no longer defines backend::make_backend /
// available_backends (that per-vendor definition is exactly what limited a binary
// to a single selectable backend). Link it alongside any other vendor's
// registration TU and one binary holds both.
//
// The FACTORY is deliberately unconditional — CudaBackend is still usable on a
// CUDA-less machine (load_stages() falls back to the vendor ONNX stages), so
// an explicit `--backend nvidia` must keep constructing. But AUTO-DETECT must
// not be won by that degraded configuration: this registrar sits at the
// highest priority, so a never-declining factory would lock auto-detect onto
// "nvidia" (silently running on the CPU) on a box whose REAL accelerator is a
// lower-priority vendor — an AMD or Intel GPU would never even be tried. The
// old comment here claimed the fallback "only changes which name is
// reported"; that is true only when no other accelerator exists. Hence the
// auto_usable probe: auto-detect skips nvidia when no CUDA device answers,
// named selection is untouched.

#include "nvidia/backend/cuda_backend.h"
#include "turbo_ocr/backend/backend_registry.h"

#include <cuda_runtime.h>

namespace turbo_ocr::nvidia {
namespace {

std::unique_ptr<backend::Backend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}

bool cuda_device_present() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

const backend::BackendRegistrar g_nvidia_registrar{
    "nvidia", {"cuda"}, backend::kBackendPriorityNvidia, &make_cuda_backend,
    &cuda_device_present};

} // namespace
} // namespace turbo_ocr::nvidia
