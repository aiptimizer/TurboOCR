#pragma once

// backend_registry.h — the ONE link-time-collecting backend registry.
//
// WHY: `make_backend()` / `available_backends()` (declared in backend.h) used to
// be DEFINED once per vendor, in that vendor's `*_backend_registry.cpp`. Because
// all three TUs defined the same two symbols, exactly ONE of them could be linked
// into a binary — i.e. a build could select among exactly one backend, which made
// `--backend` and the "available backends" list vacuous, and made every vendor
// re-write the same selection/alias/auto-detect logic (the classic per-backend
// duplication this rebuild exists to remove).
//
// NOW: the two functions are defined ONCE, in the shared layer
// (src/backend/backend_registry.cpp). Each vendor ships a tiny registration
// TU that declares one namespace-scope `BackendRegistrar`, whose constructor runs
// at static-init time and adds {canonical name, aliases, priority, factory} to the
// registry. Linking N vendor registration TUs therefore yields ONE binary that
// holds N backends: `available_backends()` lists them all and `--backend <name>`
// picks at runtime.
//
// PULL-IN CONTRACT: a registration TU defines no symbol anybody references, so it
// must reach the linker as an explicit object file (or via -force_load /
// --whole-archive) — never as an unreferenced member of a plain static archive.
// The CMake backend targets whole-archive these registration TUs.
//
// THREADING: registration happens during static init (single-threaded) but the
// registry is mutex-guarded anyway, so a backend registered from a dlopen'd
// plugin is safe too.

#include <initializer_list>
#include <memory>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/backend.h" // Backend, make_backend, available_backends

namespace turbo_ocr::backend {

// A vendor's construction hook. Returning nullptr (or throwing) means "this
// vendor is compiled in but not usable on this machine" — auto-detect then falls
// through to the next-highest-priority backend.
using BackendFactory = std::unique_ptr<Backend> (*)();

// Auto-detect ordering: highest priority whose factory actually yields a backend
// wins when make_backend("") is called. Discrete accelerators outrank integrated
// ones; the host backend is last because it is always constructible and would
// otherwise shadow everything.
inline constexpr int kBackendPriorityNvidia = 100;
inline constexpr int kBackendPriorityAmd = 90;
inline constexpr int kBackendPriorityApple = 80;
inline constexpr int kBackendPriorityIntel = 70;
inline constexpr int kBackendPriorityCpu = 0;

// Add a vendor. `name` is the canonical id reported by available_backends();
// `aliases` are extra spellings accepted by make_backend() ("metal", "host", …).
// Re-registering an existing canonical name is a no-op (idempotent), so a TU that
// is accidentally linked twice cannot duplicate an entry.
// Optional AUTO-DETECT-ONLY usability probe. nullptr => auto-detect constructs
// the backend to find out (the factory declines by returning null/throwing).
// A vendor whose factory NEVER declines — CudaBackend degrades to its ONNX
// stages internally, so constructing it always "succeeds" — supplies this so
// auto-detect can fall through to a lower-priority vendor whose real device IS
// present, while an explicit --backend still gets the degraded-but-working
// configuration. The probe must be cheap and must not throw.
using AutoUsableProbe = bool (*)();

void register_backend(std::string_view name,
                      std::initializer_list<std::string_view> aliases,
                      int priority, BackendFactory factory,
                      AutoUsableProbe auto_usable = nullptr);

// Static-init self-registration handle. Declare ONE of these at namespace scope
// in the vendor's registration TU:
//
//   const backend::BackendRegistrar g_cpu{"cpu", {"host"},
//                                         backend::kBackendPriorityCpu,
//                                         &make_cpu_backend};
struct BackendRegistrar {
  BackendRegistrar(std::string_view name,
                   std::initializer_list<std::string_view> aliases,
                   int priority, BackendFactory factory,
                   AutoUsableProbe auto_usable = nullptr) {
    register_backend(name, aliases, priority, factory, auto_usable);
  }
};

} // namespace turbo_ocr::backend
