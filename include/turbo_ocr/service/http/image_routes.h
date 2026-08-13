#pragma once

#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/core/service_fns.h"
#include "turbo_ocr/service/server/work_pool.h"

// InferFunc-based image endpoints (device-neutral): /ocr/pixels (raw-BGR input)
// and GET /profile. There is no GPU counterpart — the forked image_routes_gpu
// went with the duplicate HTTP layer, which is why these are typed on InferFunc
// rather than on any device type. The batch endpoint lives in unified_routes.h,
// typed on pipeline::UnifiedPipelinePool.
namespace turbo_ocr::routes {

void register_ocr_pixels_route(server::WorkPool &work_pool,
                               const server::InferFunc &infer,
                               const capability::CapabilityMask &loaded);

// GET /profile — read-and-reset per-stage timing (PROFILE_STAGES=1).
void register_profile_route();

} // namespace turbo_ocr::routes
