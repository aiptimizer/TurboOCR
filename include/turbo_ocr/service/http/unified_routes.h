#pragma once

// unified_routes.h — the route registrations the unified server needs that the
// main tree only ships in vendor-typed form.
//
// The main tree had TWO /ocr/batch registrars, each typed on its backend's
// concrete pool: routes::register_ocr_batch_route_cpu(CpuPipelinePool&) and
// routes::register_ocr_batch_route_gpu(PipelineDispatcher&). Registering either
// from src/service/server/unified/server_main.cpp would re-fork the server on the device
// axis — the exact duplication this rebuild removes. So the ONE batch route
// below is typed on pipeline::UnifiedPipelinePool and drives
// UnifiedOcrPipeline::run_batch_with_layout, i.e. it serves EVERY backend.
//
// Both vendor-typed registrars are gone: register_ocr_batch_route_unified is
// the only /ocr/batch registrar in the tree. The CPU one went with its file
// (the old src/http/image/batch/batch_route_cpu.cpp) and the GPU one with the
// CUDA route layer.
//
// Everything device-neutral is reused, not re-derived: the per-slot stages
// (base64 decode, pre/post-decode dimension + pixel-budget caps) and the
// response emitter come from src/service/http/image/batch/batch_common.cpp via
// batch_internal.h, so the wire contract cannot drift from the CPU route it
// replaces.

#include <memory>

#include "turbo_ocr/backend/backend.h" // BackendCaps
#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/core/service_fns.h"
#include "turbo_ocr/service/server/work_pool.h"

#include "turbo_ocr/pipeline/unified/make_infer_func.h" // pipeline::UnifiedPipelinePool

namespace turbo_ocr::routes {

// POST /ocr/batch — request/response contract IDENTICAL to the deleted
// register_ocr_batch_route_cpu, baseline
// `git show HEAD:src/http/image/batch/batch_route_cpu.cpp` (same validation
// gate, same error codes, same per-slot error strings, same
// {batch_results, errors} body). Only the inference
// stage differs: instead of one image at a time on a CpuOcrPipeline, each worker
// leases ONE UnifiedOcrPipeline and pushes chunks of up to 8 images through
// run_batch_with_layout (the batched det/rec path the GPU route already used),
// falling back to per-image run_with_layout when a chunk throws so one
// degenerate image can never blank its neighbours.
void register_ocr_batch_route_unified(
    server::WorkPool &work_pool,
    std::shared_ptr<pipeline::UnifiedPipelinePool> pool, int pool_size,
    const server::ImageDecoder &decode, const capability::CapabilityMask &loaded, int max_batch_images);

// GET /capabilities/backend — the active backend + everything compiled into this
// binary. It reports `available_backends` — every vendor registrar linked into
// this binary, not just the active one — which /capabilities does not.
// `backend`/`device` are ALSO in /capabilities since the 2026-07-23 merge
// (routes::CapabilitiesInfo carries backend_name/device_name; the header that
// owns it was frozen at the time this endpoint was written and no longer is);
// they are duplicated here for clients that already consume this endpoint.
//
// DO NOT DELETE. It is a live endpoint — see src/service/server/unified/server_main.cpp where it
// is registered, with the same rationale.
void register_backend_capabilities_route(const backend::BackendCaps &caps,
                                         int pool_size);

} // namespace turbo_ocr::routes
