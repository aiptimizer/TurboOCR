// apple_backend_registry.cpp — registers the Apple (Metal/MPSGraph/ANE) backend
// into the ONE shared link-time registry (src/backend/backend_registry.cpp).
//
// Pure registration, exactly like src/backends/cpu/backend/cpu_backend_registry.cpp: this TU
// no longer defines backend::make_backend / available_backends (that was the
// reason only ONE vendor registry could be linked per binary). Linking this TU
// alongside the CPU one produces a single server that lists BOTH backends and
// selects between them at runtime.
//
// Kept as .cpp (not .mm) on purpose. It dates from a standalone build script
// that archived every *.mm into one library, where this TU would have been
// swallowed — but the reason outlived the script (which is gone; see
// src/backends/apple/README.md), because it was never really about the script.
// A registration TU defines a registrar that NOTHING references, so a static
// archive is entitled to drop the whole object and the backend silently
// vanishes; it has to reach the linker as an explicit object. That is what
// turbo_link_backends() force-links, and what tests/cpp/backends/README.md
// step 6 checks for. It is compiled with -ObjC++ by the server build because
// apple_backend.h is reachable from Objective-C++ headers.

#include "apple/backend/apple_backend.h"
#include "turbo_ocr/backend/backend_registry.h"

namespace turbo_ocr::apple {
namespace {

std::unique_ptr<backend::Backend> make_apple_backend_entry() {
  return make_apple_backend();
}

const backend::BackendRegistrar g_apple_registrar{
    "apple", {"metal"}, backend::kBackendPriorityApple, &make_apple_backend_entry};

} // namespace
} // namespace turbo_ocr::apple
