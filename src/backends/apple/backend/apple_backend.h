#pragma once

// AppleBackend — the ONE Apple device seam (backend/backend.h). This is the
// object a unified server_main constructs when the selected device is Apple
// (Metal). From it flow the low-level device factories the ONE OcrPipeline uses
// to keep data resident (queue / allocator / kernels / engine / constructed
// stages) and the high-level service-boundary functions the HTTP/gRPC routes
// consume (server::InferFunc / ImageDecoder / OrientFunc).
//
// What runs on this M3 Max today: detection (MPSGraph DBNet + host DB post) and
// the proven fused recognition (Metal warp -> MPSGraph rec -> GPU argmax -> host
// CTC). Classifier is structural; layout/table/formula are TODOs (see README).

#include <memory>
#include <string>

#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/analysis/classification/ort_doc_orientation.h"

namespace turbo_ocr::apple {

class MetalAllocator;

class AppleBackend final : public backend::Backend {
public:
  AppleBackend();
  ~AppleBackend() override;

  [[nodiscard]] backend::BackendCaps caps() const override;

  // --- Low-level device factories -------------------------------------------
  [[nodiscard]] std::unique_ptr<backend::DeviceQueue> make_queue() override;
  [[nodiscard]] std::shared_ptr<backend::IDeviceAllocator> allocator() override;
  [[nodiscard]] std::unique_ptr<backend::IKernels> make_kernels() override;
  // Not a seam override any more (see backend.h): this arm's own engine
  // factory, used inside load_stages().
  [[nodiscard]] std::unique_ptr<backend::IEngine> make_engine();

  [[nodiscard]] std::unique_ptr<backend::ITableRecognizer>
  make_table_recognizer(const backend_routing::BackendSpec &spec) override;
  [[nodiscard]] std::unique_ptr<backend::IFormulaRecognizer>
  make_formula_recognizer(const backend_routing::BackendSpec &spec) override;

  // --- Stage bootstrap + service-boundary functions -------------------------
  [[nodiscard]] backend::StageSet load_stages(const backend::BackendConfig &cfg) override;

private:
  // The two paths (backend/engine_mode.h). Native = MPSGraph over an exported
  // graph.json/weights.bin ("ultra"); Onnx = the .onnx through the CoreML
  // execution provider, assembled by the SHARED cpu::make_onnx_stages ("fast",
  // no graph build, fp16 on ANE/GPU). load_stages() picks between them.
  [[nodiscard]] backend::StageSet load_native_stages_(const backend::BackendConfig &cfg);
  [[nodiscard]] backend::StageSet load_onnx_stages_(const backend::BackendConfig &cfg);
  backend::EngineMode mode_ = backend::EngineMode::Native;
  // True only in native mode; gates every device factory (queue/allocator/
  // kernels/engine) onto Metal instead of the host implementations.
  [[nodiscard]] bool native_device_() const noexcept;
  std::shared_ptr<backend::IDeviceAllocator> host_allocator_;
  // Doc-orientation model owned by the ONNX path (the native path has none).
  std::unique_ptr<classification::OrtDocOrientation> onnx_doc_ori_;

public:

  // NOTE (dedup): no make_infer_func() — the ONE orchestration lives in
  // the unified pipeline (UnifiedOcrPipeline + pipeline::make_infer_func), built
  // over the StageSet load_stages() returns. Only decode/orient stay per-vendor.
  [[nodiscard]] server::ImageDecoder make_image_decoder() override;
  [[nodiscard]] server::OrientFunc make_orient_func() override;

private:
  backend::BackendConfig cfg_; // captured by load_stages() (model paths/dict)
  bool configured_ = false;
};

// Apple backend factory. The global backend::make_backend() registry (backend.h)
// is wired in server_main among the compiled-in vendors; this per-vendor factory
// keeps the Apple lib self-contained and avoids an ODR clash with the other
// backends' registrations.
std::unique_ptr<backend::Backend> make_apple_backend();

} // namespace turbo_ocr::apple
