#pragma once

#include <memory>
#include <string>

#include "turbo_ocr/decode/nvjpeg_decoder_pool_fwd.h"
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

// Open the shared nvJPEG decoder pool (`capacity` decoders, leased per
// decode by the work-pool routes) and probe availability on the calling
// thread, logging the outcome. nullptr when nvJPEG is unavailable — every
// consumer then decodes on the CPU. Decoders are pooled, never per-thread:
// each holds ~190 MB of VRAM for the life of the process (GitHub #33).
[[nodiscard]] std::shared_ptr<decode::NvJpegDecoderPool>
open_nvjpeg_decoders(int capacity);

// Image decoder: JPEG via a leased nvJPEG decoder when the pool is non-null
// (CPU fallback when every decoder is busy past kNvJpegLeaseWait), PNG via
// Wuffs, every other format via cv::imdecode (decode::decode_cpu_fallback).
[[nodiscard]] ImageDecoder
make_gpu_image_decoder(std::shared_ptr<decode::NvJpegDecoderPool> nvjpeg);

// The InferFunc for shared routes (/ocr base64): submit to the dispatcher
// with by-value captures (timeout-safe), forward every degradation signal.
[[nodiscard]] InferFunc
make_gpu_infer_func(pipeline::PipelineDispatcher &dispatcher);

} // namespace turbo_ocr::server
