#pragma once

#include <string>

#include "turbo_ocr/pipeline/pool/cpu_pipeline_pool.h"
#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/server/service_fns.h"

// CPU-build stage loading and pipeline adapters, extracted from cpu_main so
// main() stays pure orchestration. Everything here mirrors the GPU build's
// startup semantics: optional stages soft-disable when their model is simply
// absent, but a stage the operator explicitly CONFIGURED that fails to load
// aborts startup — a server must never come up silently structure-less.
namespace turbo_ocr::server {

// What actually loaded into every pipeline of the pool — the single source of
// truth for the per-request fail-loud gates and /capabilities.
struct CpuStageAvailability {
  bool layout = false;
  bool table = false;
  bool formula = false;
  bool doc_ori = false;
};

// Load layout, formula/table (env-gated), and doc-orientation models into
// every pipeline in the pool. Throws std::runtime_error when an explicitly
// configured backend fails to load (fatal, exit via main's catch); missing
// OPTIONAL models just leave their flag false.
[[nodiscard]] CpuStageAvailability
load_cpu_stages(pipeline::CpuPipelinePool &pool, int pool_size,
                const ServerConfig &cfg);

// The InferFunc every InferFunc-driven route (and gRPC) runs through:
// acquire a pipeline lease, run with the requested options, map the result
// (including all degradation signals — dropping them would let a failed
// stage serve a clean 200).
[[nodiscard]] InferFunc make_cpu_infer_func(pipeline::CpuPipelinePool &pool);

// Document-orientation probe for /ocr/pdf?autorotate=1. Empty when the
// doc-ori model is unavailable.
[[nodiscard]] OrientFunc make_cpu_orient_func(pipeline::CpuPipelinePool &pool,
                                              bool doc_ori_available);

} // namespace turbo_ocr::server
