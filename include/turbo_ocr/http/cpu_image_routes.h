#pragma once

#include "turbo_ocr/pipeline/pool/cpu_pipeline_pool.h"
#include "turbo_ocr/server/service_fns.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/server/work_pool.h"

// CPU-build image endpoints that need more than the generic InferFunc:
// /ocr/pixels (raw-BGR input) and /ocr/batch (bounded jthread fan-out over
// the pipeline pool). Their GPU counterparts live in image_routes.cpp; the
// request contract (validation, caps, response shape) is shared through
// request_gate.h / pixel_dims.h so the two builds cannot drift.
namespace turbo_ocr::routes {

void register_ocr_pixels_route_cpu(server::WorkPool &work_pool,
                                   const server::InferFunc &infer,
                                   bool layout_available,
                                   bool table_available,
                                   bool formula_available);

void register_ocr_batch_route_cpu(server::WorkPool &work_pool,
                                  pipeline::CpuPipelinePool &pool,
                                  int pool_size,
                                  const server::ImageDecoder &decode,
                                  bool layout_available,
                                  bool table_available,
                                  bool formula_available,
                                  int max_batch_images);

// GET /profile — read-and-reset per-stage timing (PROFILE_STAGES=1).
void register_profile_route();

} // namespace turbo_ocr::routes
