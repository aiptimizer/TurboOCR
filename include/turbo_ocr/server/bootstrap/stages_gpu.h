#pragma once

#include <memory>
#include <string>

#include "turbo_ocr/decode/host_image_pool.h"
#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/server/service_fns.h"

// GPU-build stage loading and pipeline adapters, extracted from gpu_main so
// main() stays pure orchestration — the mirror of stages_cpu.h. GPU target
// only (TRT/CUDA/nvJPEG behind the TU).
namespace turbo_ocr::pipeline {
class PipelineDispatcher;
}

namespace turbo_ocr::server {

// TRT engine paths resolved by load_gpu_stages, plus the auto-sized pool.
struct GpuStages {
  std::string det_model;
  std::string rec_model;
  std::string rec_dict;
  std::string cls_model;
  std::string layout_model;
  std::string doc_ori_model;
  int pool_size = 4;
};

// Fail-fast model-path validation. MUST run before the PdfRenderer fork()s
// its daemon pool: a missing model then exits without orphaning daemons, and
// no CUDA call happens before the forks.
void validate_gpu_models(const ServerConfig &cfg);

// Sweep orphan engine temps and build/resolve every TRT engine (det/rec/cls/
// layout/doc_ori), enforcing the explicit-CLS fail-loud rule; resolves the
// pipeline pool size (explicit override, else VRAM-tier heuristic). Touches
// the CUDA context — call only AFTER the PdfRenderer is constructed. Calls
// std::exit(2) when an explicitly configured CLS engine cannot be loaded.
[[nodiscard]] GpuStages load_gpu_stages(const ServerConfig &cfg);

// Pinned host memory (cudaHostAlloc, portable) for the host image pool.
[[nodiscard]] decode::BlockMemory cuda_pinned_block_memory() noexcept;

// Give the device's stream-ordered memory pool (cudaMallocAsync, used for the
// decoders' scratch) an explicit release threshold, so device memory kept
// between requests is a fixed number rather than "whatever the pool grew to".
void configure_device_memory_pool();

// Probe nvJPEG once at startup (constructs and discards a decoder; logs the
// outcome). JPEG is decoded by each replica's own decoder, created lazily on
// the replica thread (GpuPipelineEntry::get_nvjpeg): one per replica, so the
// decoder footprint is bounded by the pool size and nothing is shared or
// leased. The v3.5.1 work-pool decoders and the v3.5.2 shared pool are gone.
[[nodiscard]] bool probe_nvjpeg();

// Host image decoder for the non-JPEG formats: PNG via Wuffs, every other
// format via cv::imdecode (decode::decode_cpu_fallback). Every route decodes
// JPEG on the replica; with a GPU decoder present this decoder refuses JPEG
// loudly rather than decoding it on the CPU with different pixels, so a route
// that bypasses the replica path fails in tests instead of degrading quietly.
// Without nvJPEG (nvjpeg_available=false) JPEG is a host format like the rest.
[[nodiscard]] ImageDecoder make_gpu_image_decoder(bool nvjpeg_available);

// JPEG-bytes inference for the transport-neutral routes (/ocr base64):
// GPU-direct decode + inference on the replica, identical to /ocr/raw.
[[nodiscard]] JpegInferFunc
make_gpu_jpeg_infer_func(pipeline::PipelineDispatcher &dispatcher);

// The InferFunc for shared routes (/ocr base64): submit to the dispatcher
// with by-value captures (timeout-safe), forward every degradation signal.
[[nodiscard]] InferFunc
make_gpu_infer_func(pipeline::PipelineDispatcher &dispatcher);

} // namespace turbo_ocr::server
