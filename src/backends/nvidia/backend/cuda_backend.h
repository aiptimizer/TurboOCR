#pragma once

// CudaBackend — the NVIDIA implementation of the ONE device seam
// (backend/backend.h). It is the single object the merged server_main
// constructs for an NVIDIA build; from it flow the device factories the ONE
// OcrPipeline uses (queue / allocator / kernels / engine / stages) and the
// service-boundary functions the HTTP/gRPC routes consume (InferFunc /
// ImageDecoder / OrientFunc). It collapses stages_gpu.h + gpu_server_main.cpp
// (load_gpu_stages / make_gpu_infer_func / make_gpu_image_decoder / probe_nvjpeg)
// into methods on this class — WRAPPING those helpers, never rewriting them.
//
// Everything device-specific is behind the returned interface objects; this
// header pulls no CUDA/TRT type into its own signatures.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/analysis/classification/ort_doc_orientation.h"

#include "nvidia/stages/nv_doc_orientation.h"

namespace turbo_ocr::nvidia {

class CudaAllocator;

class CudaBackend final : public backend::Backend {
public:
  CudaBackend();
  ~CudaBackend() override;

  [[nodiscard]] backend::BackendCaps caps() const override;

  // --- Low-level device factories -------------------------------------------
  [[nodiscard]] std::unique_ptr<backend::DeviceQueue> make_queue() override;
  [[nodiscard]] std::shared_ptr<backend::IDeviceAllocator> allocator() override;
  [[nodiscard]] std::unique_ptr<backend::IKernels> make_kernels() override;
  // Not a seam override any more (see backend.h): this arm's own engine
  // factory, used inside load_stages().
  [[nodiscard]] std::unique_ptr<backend::IEngine> make_engine();

  [[nodiscard]] std::unique_ptr<turbo_ocr::backend::ITableRecognizer>
  make_table_recognizer(const backend_routing::BackendSpec &spec) override;
  [[nodiscard]] std::unique_ptr<turbo_ocr::backend::IFormulaRecognizer>
  make_formula_recognizer(const backend_routing::BackendSpec &spec) override;

  // --- Stage bootstrap ------------------------------------------------------
  [[nodiscard]] backend::StageSet load_stages(const backend::BackendConfig &cfg) override;

  // --- High-level service-boundary functions --------------------------------
  // NOTE (dedup): no make_infer_func() and no attach_dispatcher(). The ONE
  // server::InferFunc is built above the seam by pipeline::make_infer_func()
  // over a pool of UnifiedOcrPipeline entries constructed from the StageSet
  // load_stages() returns — the same code path every other backend uses. Only
  // the genuinely device-specific decode (nvJPEG) and page-orientation hooks
  // remain per-vendor.
  [[nodiscard]] server::ImageDecoder make_image_decoder() override;
  [[nodiscard]] server::OrientFunc make_orient_func() override;

  // --- Device queries asked through the seam --------------------------------
  // can_device_decode: whether nvJPEG will take THIS buffer, so the caller can
  // route without knowing what nvJPEG is (it decodes JPEG only).
  // device_memory: what metrics.h reports as VRAM. Injected as a callback
  // rather than called directly, which is what keeps cudaMemGetInfo — and the
  // CUDA link dependency — out of the service layer.
  [[nodiscard]] bool can_device_decode(const std::uint8_t *data,
                                       std::size_t len) const override;
  [[nodiscard]] bool device_memory(std::size_t &used,
                                   std::size_t &total) const override;

private:
  std::shared_ptr<CudaAllocator> allocator_;
  std::unique_ptr<NvDocOrientation> doc_ori_;
  // Which path load_stages() settled on (backend/engine_mode.h). Native =
  // TensorRT engines (built once per GPU/driver, fastest steady state); Onnx =
  // the .onnx through the CUDA execution provider — no engine build, so a cold
  // box serves in seconds. In onnx mode the device factories below return the
  // HOST implementations, because the ONNX path IS the shared host stage set.
  backend::EngineMode mode_ = backend::EngineMode::Native;
  std::shared_ptr<backend::IDeviceAllocator> host_allocator_;
  std::unique_ptr<turbo_ocr::classification::OrtDocOrientation> onnx_doc_ori_;
  [[nodiscard]] bool native_device_() const noexcept {
    return mode_ == backend::EngineMode::Native;
  }
  bool nvjpeg_available_ = false;
  // 0 = not operator-set: caps() then sizes by VRAM tier (pool_sizing.h),
  // matching the v3.5.0 GPU main. load_stages(cfg.pool_size>0) overrides.
  int pool_size_ = 0;
};

} // namespace turbo_ocr::nvidia
