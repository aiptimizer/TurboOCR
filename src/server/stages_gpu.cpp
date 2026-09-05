#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/bootstrap/stages_gpu.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/decode/cpu_image_decode.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/common/errors.h"
#include <format>
#include "turbo_ocr/pipeline/jpeg_infer.h"
#include "turbo_ocr/decode/jpeg_codec.h"
#include "turbo_ocr/engine/trt/onnx_to_trt.h"
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#include "turbo_ocr/server/bootstrap/pool_sizing.h"
#include "turbo_ocr/server/bootstrap/server_bootstrap.h"

namespace turbo_ocr::server {

void validate_gpu_models(const ServerConfig &cfg) {
  // Validate model paths up front so a missing models/ tree fails fast with
  // a clear error rather than tripping a confusing CUDA/TRT error deep in
  // pipeline construction. ensure_trt_engine() returns "" on missing ONNX,
  // which the dispatcher only notices much later.
  auto require_model = [](const std::string &path, const char *purpose) {
    bootstrap::require_model(path, purpose, "model", "_ONNX");
  };
  require_model(cfg.det_onnx, "DET");
  require_model(cfg.rec_paths.rec, "REC");
  require_model(cfg.cls_onnx, "CLS");
}

GpuStages load_gpu_stages(const ServerConfig &cfg) {
  GpuStages s;
  const auto &rec_paths = cfg.rec_paths;
  s.rec_dict = rec_paths.dict;

  // Auto-build TRT engines from ONNX (cached by TRT version + model hash)
  // Sweep orphan .trt.tmp.* files left by previous crashed processes; safe
  // because in-progress builds by sibling replicas are protected by the
  // 60-second min-age window inside the sweeper.
  turbo_ocr::engine::sweep_orphan_engine_temps();
  s.det_model = turbo_ocr::engine::ensure_trt_engine(cfg.det_onnx, "det");
  s.rec_model = turbo_ocr::engine::ensure_trt_engine(rec_paths.rec, "rec");
  s.cls_model = turbo_ocr::engine::ensure_trt_engine(cfg.cls_onnx, "cls");
  // An explicitly configured CLS_ONNX that fails to resolve must not silently
  // disable orientation handling (only the unset default may soft-disable).
  if (s.cls_model.empty() && cfg.cls_explicit && !cfg.disable_angle_cls) {
    TOCR_LOG_ERROR("CLS_ONNX could not be loaded (file missing or engine "
                   "build failed); refusing to start with a silently disabled "
                   "angle classifier — unset CLS_ONNX or set "
                   "DISABLE_ANGLE_CLS=1 to run without it",
                   "cls_onnx", cfg.cls_onnx);
    std::exit(2);
  }
  if (cfg.disable_angle_cls) {
    s.cls_model.clear();
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Optional PP-DocLayoutV3 stage. ON by default — users can disable with
  // DISABLE_LAYOUT=1 to save ~300-500 MB VRAM.
  if (!cfg.layout_disabled) {
    if (cfg.layout_trt && !cfg.layout_trt->empty()) {
      s.layout_model = *cfg.layout_trt;
      TOCR_LOG_INFO("Layout detection enabled", "engine", std::string_view(s.layout_model));
    } else {
      // Layout ONNX is optional — soft-disable with a warning if missing
      // rather than hard-failing, so installs without layout still serve.
      s.layout_model = turbo_ocr::engine::ensure_trt_engine(cfg.layout_onnx, "layout");
      if (s.layout_model.empty()) {
        TOCR_LOG_WARN("Layout model (layout.onnx) not found; layout stage will be disabled");
      } else {
        TOCR_LOG_INFO("Layout detection enabled");
      }
    }
  } else {
    TOCR_LOG_INFO("Layout detection disabled (set DISABLE_LAYOUT=0 to enable)");
  }

  // Document-orientation engine (optional) — powers /ocr/pdf?autorotate=1.
  // Soft-disable if the model is absent: ensure_trt_engine returns "" and
  // autorotate requests are then rejected with AUTOROTATE_DISABLED.
  s.doc_ori_model =
      turbo_ocr::engine::ensure_trt_engine(cfg.doc_ori_onnx, "doc_ori");
  if (s.doc_ori_model.empty())
    TOCR_LOG_WARN("Doc-orientation model (doc_ori.onnx) not found; autorotate disabled");
  else
    TOCR_LOG_INFO("Doc-orientation (autorotate) enabled");

  // Pipeline pool size: explicit override wins, else the VRAM-tier policy.
  s.pool_size = 4;
  if (cfg.pipeline_pool_size) {
    s.pool_size = *cfg.pipeline_pool_size;
  } else {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
      s.pool_size = compute_pipeline_pool_size(free_mem, total_mem);
  }
  return s;
}

namespace {
void *pinned_alloc(size_t bytes) noexcept {
  void *p = nullptr;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) == cudaSuccess) return p;
  (void)cudaGetLastError();
  return nullptr;
}
void pinned_free(void *p) noexcept {
  if (cudaFreeHost(p) != cudaSuccess) (void)cudaGetLastError();
}
} // namespace

decode::BlockMemory cuda_pinned_block_memory() noexcept {
  return {&pinned_alloc, &pinned_free, "cuda_pinned"};
}

void configure_device_memory_pool() {
  int dev = 0;
  cudaMemPool_t pool = nullptr;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetDefaultMemPool(&pool, dev) != cudaSuccess) {
    (void)cudaGetLastError();
    return;
  }
  // 256 MiB: room for a few in-flight decode scratches (a 40 MP page is
  // 120 MB) without holding every peak forever. Above it, freed blocks go
  // back to the driver at the next synchronisation.
  unsigned long long threshold = 256ULL << 20;  // cuuint64_t lives in cuda.h; the runtime attribute takes the value by pointer
  if (cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) != cudaSuccess) {
    (void)cudaGetLastError();
    TOCR_LOG_WARN("Device memory pool release threshold not set");
    return;
  }
  TOCR_LOG_INFO("Device memory pool release threshold set", "mb", 256);
}

bool probe_nvjpeg() {
  const bool available = decode::NvJpegDecoder{}.available();
  if (available)
    TOCR_LOG_INFO("nvJPEG GPU-accelerated JPEG decode enabled (one decoder per replica)");
  else
    TOCR_LOG_WARN("nvJPEG not available, JPEG decodes on the CPU");
  return available;
}

ImageDecoder make_gpu_image_decoder(bool nvjpeg_available) {
  return [nvjpeg_available](const unsigned char *data, size_t len) -> cv::Mat {
    if (nvjpeg_available && decode::looks_like_jpeg(data, len)) {
      TOCR_LOG_ERROR_RL("JPEG reached the host image decoder; every route must decode JPEG on the replica",
                        "bytes", len);
      return {};
    }
    return decode::decode_cpu_fallback(data, len);
  };
}

JpegInferFunc make_gpu_jpeg_infer_func(pipeline::PipelineDispatcher &dispatcher) {
  return [&dispatcher](std::shared_ptr<const std::string> jpeg,
                       const InferOptions &opts) -> InferResult {
    // Same by-value ownership rules as make_gpu_infer_func: the task may be
    // abandoned on timeout, so it owns the bytes (shared) and plain flags.
    pipeline::JpegRunOpts run_opts{
        .want_layout = opts.want_layout,
        .want_reading_order = opts.want_reading_order,
        .want_tables = opts.want_tables,
        .want_formulas = opts.want_formulas,
        .routing = opts.routing_override,
        .defer_external = false,
        .layout_only = !opts.want_text,
    };
    auto out = dispatcher.submit_for_default([jpeg, run_opts](auto &e) {
      return pipeline::decode_jpeg_and_run(
          e, reinterpret_cast<const unsigned char *>(jpeg->data()),
          jpeg->size(), run_opts);
    });
    return from_pipeline_result(std::move(out));
  };
}

InferFunc make_gpu_infer_func(pipeline::PipelineDispatcher &dispatcher) {
  return [&dispatcher](const cv::Mat &img,
                       const InferOptions &opts) -> InferResult {
    // submit_for_default may ABANDON the task on timeout, so the lambda owns
    // its inputs BY VALUE: img is a cheap cv::Mat refcount bump, the flags are
    // bools. The dispatcher itself is long-lived and captured by reference.
    // On deadline this throws turbo_ocr::TimeoutError; the route's
    // run_with_error_handling maps it to HTTP 504 (INFERENCE_TIMEOUT).
    const bool want_text = opts.want_text;
    const bool want_layout = opts.want_layout;
    const bool want_reading_order = opts.want_reading_order;
    const bool want_tables = opts.want_tables;
    const bool want_formulas = opts.want_formulas;
    const auto routing_override = opts.routing_override;  // by-value (timeout-safe)
    auto out = dispatcher.submit_for_default(
        [img, want_text, want_layout, want_reading_order, want_tables,
         want_formulas, routing_override](auto &e) {
          if (!want_text)
            return e.pipeline->run_layout_only(img, e.stream);
          return e.pipeline->run_with_layout(img, e.stream, want_layout,
                                             want_reading_order, routing_override,
                                             /*defer_external=*/false,
                                             want_tables, want_formulas);
        });
    // dispatch_router_ ran synchronously (defer_external defaults false on this
    // path), so out carries any table/formula structure + degradation flags.
    return from_pipeline_result(std::move(out));
  };
}

} // namespace turbo_ocr::server
