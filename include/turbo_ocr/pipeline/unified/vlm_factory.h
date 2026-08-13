#pragma once

// vlm_factory — the SHARED, device-agnostic definition of the two free
// factories the Backend seam declares for the REMOTE (kind:openai) branch:
//
//   turbo_ocr::backend::make_table_recognizer(const backend_routing::BackendSpec&)
//   turbo_ocr::backend::make_formula_recognizer(const backend_routing::BackendSpec&)
//
// (declared in include/turbo_ocr/backend/{table,formula}_recognizer.h).
//
// WHY IT IS SHARED, NOT PER-BACKEND: an OpenAI-compatible VLM endpoint is pure
// host work — D2H the page, crop, PNG-encode, POST, parse. Nothing about it is
// device-specific, so every backend routing a kind:openai spec must reach the
// SAME implementation. Each Backend::make_table/formula_recognizer() forwards
// its remote branch here and keeps only its LOCAL device recognizer.
//
// The only device-touching step is reading the page pixels back to the host.
// That is expressed through the ONE seam abstraction that can do it portably —
// backend::IDeviceAllocator::copy_d2h on a backend::DeviceQueue — registered at
// startup by server_main via register_device_readback() below. Without it, Host
// pages still work (zero-copy) and host-coherent (unified-memory) pages work;
// discrete-VRAM pages decline cleanly instead of dereferencing device memory.
//
// KEYED BY DEVICE, NOT PROCESS-GLOBAL. This used to be a single global slot
// (set_device_readback), which was safe only while a binary could hold exactly
// one backend. Now that the shared registry lets ONE binary hold cpu+apple (see
// backend/backend_registry.h), a single slot would be a global with two owners:
// last writer wins, and a Metal page would be read back through the host
// allocator or vice versa. The registration is therefore keyed on the
// backend::DeviceKind it belongs to, and host_pixels() looks up the entry for
// the PAGE's device — so N backends can coexist and each page is always read
// back through its own backend's allocator.
//
// (Today server_main still constructs exactly one Backend per process, so this
// is defence in depth rather than a live bug fix. The properly-scoped fix is to
// inject the allocator into the endpoint at construction — see the REPORT: that
// requires changing the seam's free-factory signature and therefore every
// vendor's Backend::make_table/formula_recognizer, which are owned elsewhere.)

#include <cstddef>
#include <functional>
#include <memory>

#include "turbo_ocr/backend/image_view.h" // backend::DeviceKind

namespace turbo_ocr::backend {
class DeviceQueue;
class IDeviceAllocator;
} // namespace turbo_ocr::backend

namespace turbo_ocr::pipeline {

// Copy `bytes` from a device pointer `src` (valid in the owning backend's
// space) into host memory `dst`, ordered on `queue`, and BLOCK until the copy
// has landed. Return false on failure.
using DeviceReadbackFn = std::function<bool(void *dst, const void *src,
                                            std::size_t bytes,
                                            backend::DeviceQueue &queue)>;

// What one backend tells the remote-VLM endpoint about reaching its pages.
struct DeviceReadback {
  DeviceReadbackFn copy;      // empty => this device cannot be read back
  bool host_coherent = false; // pointer is directly host-addressable after sync
};

// Register `rb` as the way to reach pages living in `kind`'s space. Call once
// per constructed Backend at startup. Re-registering the same kind replaces the
// entry (a second backend on the SAME device is by definition equivalent);
// passing a default-constructed DeviceReadback clears it.
void register_device_readback(backend::DeviceKind kind, DeviceReadback rb);

// Convenience: derive the registration from a backend allocator (the normal
// case). The returned closure CO-OWNS `alloc` — the readback table is a
// process-lifetime static and outlives the BackendRuntime that owns the Backend,
// so a raw pointer would dangle from teardown until static destruction. A null
// allocator registers nothing.
//   pipeline::register_device_readback(caps.device,
//       pipeline::make_allocator_readback(backend.allocator()));
[[nodiscard]] DeviceReadback
make_allocator_readback(std::shared_ptr<backend::IDeviceAllocator> alloc);

} // namespace turbo_ocr::pipeline
