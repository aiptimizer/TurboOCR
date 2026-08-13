#pragma once

// stages.h — the ONE device-neutral stage bootstrap, replacing the forked
// include/nvidia/support/stages_gpu.h + stages_cpu.h.
//
// Those two headers each carried a private copy of "turn a ServerConfig into a
// loaded pipeline pool + an availability struct + an InferFunc + an OrientFunc",
// typed on their backend's concrete pool (PipelineDispatcher / CpuPipelinePool).
// Here it is written once against the Backend seam:
//
//   ServerConfig -> backend::BackendConfig -> Backend::load_stages()
//                -> N x UnifiedOcrPipeline -> pipeline::UnifiedPipelinePool
//
// The InferFunc is NOT built here — it is pipeline::make_infer_func(pool), the
// single shared builder. Only the two genuinely per-vendor service hooks
// (ImageDecoder, OrientFunc) still come from the Backend.

#include <memory>
#include <string>
#include <string_view>

#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/service/server/bootstrap/server_config.h"

#include "turbo_ocr/pipeline/unified/make_infer_func.h" // pipeline::UnifiedPipelinePool

namespace turbo_ocr::server {

// The whole device side of the server, owned in one place.
//
// DECLARATION ORDER IS LOAD-BEARING: every UnifiedOcrPipeline in `pool` holds a
// reference to `backend`, so `pool` must be destroyed first. Members are
// destroyed in reverse declaration order, hence backend first, pool second.
struct BackendRuntime {
  std::unique_ptr<backend::Backend> backend;
  std::shared_ptr<pipeline::UnifiedPipelinePool> pool;
  backend::BackendCaps caps;
  // LOADED (capability/capability.h) — the single source of truth for the
  // per-request fail-loud gates and /capabilities, handed to every route
  // registrar as ONE value. This replaces a four-bool struct that each
  // registrar unpacked into its own positional argument list.
  capability::CapabilityMask available;
  int pool_size = 1;
};

// Select the backend (`name` empty => auto-detect among the compiled-in
// vendors), translate the ServerConfig into a BackendConfig, load N pipeline
// entries, and bootstrap the optional router/table/formula stages on each.
//
// Throws std::runtime_error when the backend is unknown, a REQUIRED stage
// (detector/recognizer) fails, or an explicitly-configured local table/formula
// backend fails to load — a server must never come up silently structure-less.
[[nodiscard]] BackendRuntime
build_backend_runtime(std::string_view backend_name, const ServerConfig &cfg);

} // namespace turbo_ocr::server
