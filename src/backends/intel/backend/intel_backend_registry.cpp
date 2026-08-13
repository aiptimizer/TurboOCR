// intel_backend_registry.cpp — registers the Intel (OpenVINO / Level-Zero)
// backend into the ONE shared link-time registry
// (src/backend/backend_registry.cpp).
//
// Pure registration, mirroring src/backends/cpu/backend/cpu_backend_registry.cpp; this TU no
// longer defines backend::make_backend / available_backends.
//
// AUTO-DETECT semantics are preserved by the factory itself, and they matter: an
// Intel binary on a box with no Intel GPU must NOT claim the auto slot, or the
// server silently boots onto a backend whose engine will fail to load. The
// factory therefore returns nullptr unless a Level-Zero device is really present
// (or the operator pinned OV_DEVICE, an explicit choice) — the shared registry
// then walks down to the next-priority backend. An EXPLICIT `--backend intel`
// still goes through the same factory; when no device is present it returns null
// and the server reports the normal "backend not available" startup error.
//
// DELIBERATE SMALL CHANGE vs the old per-vendor selector: OV_DEVICE=CPU now also
// claims the AUTO slot. device_from_env(GPU) can only return CPU when the
// operator pinned OV_DEVICE=CPU, i.e. explicitly asked for OpenVINO-on-CPU; the
// old code refused that on the auto path so the ORT CpuBackend won instead.
// UNTESTED (no Intel hardware here) — flag for the Intel owner.

#include <cstdlib>
#include <memory>

#include "intel/backend/intel_backend.h"
#include "intel/memory/l0_allocator.h"
#include "turbo_ocr/backend/backend_registry.h"

namespace turbo_ocr::intel {
namespace {

std::unique_ptr<backend::Backend> make_intel_backend_entry() {
  const auto dev = OpenVINOEngine::device_from_env(OpenVINOEngine::DeviceType::GPU);
  if (dev == OpenVINOEngine::DeviceType::CPU)
    return make_intel_backend(dev); // operator pinned OV_DEVICE=CPU explicitly

  // Availability = "does the OpenVINO runtime enumerate this device", NOT "do we
  // have a Level-Zero USM context".
  //
  // This gate used to be `L0Allocator::has_device()`, which is false whenever
  // the backend is built without SYCL — so on a machine with a perfectly working
  // Intel iGPU (measured: Core Ultra 7 265T under WSL2, where OpenVINO reports
  // ['CPU','GPU'] and the GPU plugin beats the CPU on detection and on every
  // wide rec bucket) `--backend intel` returned null and the operator was told
  // "not compiled in / no device". Losing L0 costs zero-copy, not the device;
  // caps() already downgrades io_space to Host in that case.
  if (!OpenVINOEngine::device_available(dev)) return nullptr;
  return make_intel_backend(dev);
}

// AUTO-DETECT PRIORITY: deliberately BELOW cpu until the engine is async.
//
// BackendFactory takes no arguments, so the factory cannot tell an auto-detect
// call from an explicit `--backend intel` — returning nullptr to dodge the auto
// slot would also break explicit selection. Priority is the mechanism the
// registry actually provides for auto-ordering ("highest priority whose factory
// yields a backend wins"), so it is the right knob.
//
// WHY below cpu: measured on Core Ultra 7 265T, intel/OpenVINO runs 4.3 img/s
// where the ORT CpuBackend runs 8.8, because OpenVINOEngine::run() is
// synchronous and cannot use the multi-stream parallelism OpenVINO's throughput
// figures depend on (476 crops/s at streams=1 vs 2144 with the throughput hint).
// Fixing the GPU availability gate made this backend eligible for the auto slot
// on every Intel box; at kBackendPriorityIntel it would then WIN by default and
// silently halve throughput. `--backend intel` still selects it explicitly.
//
// RAISE THIS BACK to kBackendPriorityIntel once the async engine lands and is
// measured to beat ORT — bring-up item 1 in src/backends/intel/SETUP.md, worth ~4.5x.
constexpr int kIntelAutoPriority = backend::kBackendPriorityCpu - 1;

const backend::BackendRegistrar g_intel_registrar{
    "intel", {"openvino"}, kIntelAutoPriority, &make_intel_backend_entry};

} // namespace
} // namespace turbo_ocr::intel
