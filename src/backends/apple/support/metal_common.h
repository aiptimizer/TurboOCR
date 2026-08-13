#pragma once

// Apple backend — shared Metal plumbing.
//
// Two concerns live here that every Apple TU needs:
//   1. A process-wide MTLDevice + shader library (the compiled shaders.metal,
//      loaded once) and MPSGraphDevice — the M3 Max has one GPU; a singleton
//      matches the pipeline's one-device assumption.
//   2. The void*<->MTLBuffer REGISTRY. The Backend seam speaks bare `void*`
//      device pointers (ImageView::data, DeviceTensor::data, IKernels raw args)
//      that must be "valid in Metal space" (image_view.h:18). On Apple silicon a
//      MTLResourceStorageModeShared buffer is unified memory: its `.contents`
//      pointer is valid on BOTH the CPU and the GPU. So the allocator hands out
//      `buffer.contents` as the portable void*, and this registry recovers the
//      owning id<MTLBuffer> + byte offset when an op must bind it to an
//      MPSGraphTensorData or a Metal encoder (both take a buffer, not a pointer).
//
// This is the one place C++ interface types meet ObjC Metal handles.

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#endif

namespace turbo_ocr::apple {

#ifdef __OBJC__
// Process-wide device + resources (created on first use, never freed).
id<MTLDevice>          mtl_device();
// A PROCESS-GLOBAL queue for one-off ops. NOTHING ON THE PER-PAGE PATH MAY USE
// IT: its only former caller (ensure_texture's page pack) submitted one command
// buffer per page here and blocked in waitUntilCompleted, which measured 1.0 ms
// of blocked time per page at K=8 and 11.9 ms at K=24 — a serialization that got
// worse with concurrency. Encode onto the caller's DeviceQueue instead.
id<MTLCommandQueue>    mtl_command_queue();
id<MTLLibrary>         mtl_library();         // compiled shaders.metal
MPSGraphDevice*        mps_graph_device();

// Fetch (compile+cache) a compute pipeline state for a shader function name.
id<MTLComputePipelineState> mtl_pipeline(const char *fn_name);

// Registry: record a Shared buffer so its .contents pointer can later be
// resolved back to (buffer, offset). Called by the allocator on allocate() and
// by MetalImage; unregister on free.
void  register_buffer(id<MTLBuffer> buf);
void  unregister_buffer(void *contents);

// Resolve a portable device pointer (a .contents pointer, possibly offset into a
// registered buffer) to its owning buffer + byte offset. Returns nil if `p` is
// not inside any registered buffer.
id<MTLBuffer> resolve_buffer(const void *p, std::size_t *offset_out);
#endif

// Path to the compiled shader library (`turbo_apple.metallib`). Resolution order:
//   $TURBO_APPLE_METALLIB, then alongside the executable, then the CWD.
const char *metallib_path();

} // namespace turbo_ocr::apple
