#pragma once

// RocmBackend — the AMD implementation of backend::Backend. One object the
// merged server_main constructs when the "amd" backend is selected; from it flow
// the device factories the ONE OcrPipeline uses (queue / allocator / kernels /
// engine / stages) and the service-boundary functions the routes consume.
//
// Device identity is DeviceKind::Hip throughout; every pointer it hands out lives
// in HIP device memory (hipMalloc), validated against the single shared
// HipAllocator. This mirrors NvidiaBackend exactly, swapping CUDA/TRT for
// HIP/MIGraphX.

#include <memory>
#include <string_view>

#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/analysis/classification/ort_doc_orientation.h"

namespace turbo_ocr::amd {

class HipAllocator;
class HipKernels;

class RocmBackend final : public backend::Backend {
public:
  explicit RocmBackend(int device_id = 0);
  ~RocmBackend() override;

  [[nodiscard]] backend::BackendCaps caps() const override;

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

  [[nodiscard]] backend::StageSet load_stages(const backend::BackendConfig &cfg) override;

  // NOTE (dedup): no make_infer_func() — the shared pipeline builds it.
  [[nodiscard]] server::ImageDecoder make_image_decoder() override;
  [[nodiscard]] server::OrientFunc make_orient_func() override;

private:
  // Which path load_stages() settled on (backend/engine_mode.h). Native =
  // MIGraphX programs (compiled per gfx target); Onnx = the .onnx through the
  // MIGraphX EXECUTION PROVIDER — no program compile. In onnx mode the device
  // factories return the HOST implementations, since the ONNX path IS the
  // shared host stage set. UNVERIFIED: this backend has no CMake target yet
  // (needs ROCm hardware), so nothing here has been compiled.
  backend::EngineMode mode_ = backend::EngineMode::Native;
  std::shared_ptr<backend::IDeviceAllocator> host_allocator_;
  std::unique_ptr<turbo_ocr::classification::OrtDocOrientation> onnx_doc_ori_;
  [[nodiscard]] bool native_device_() const noexcept {
    return mode_ == backend::EngineMode::Native;
  }
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// Vendor entry point. The shared backend::make_backend("amd") dispatcher calls
// this when the AMD library is compiled in. Returns nullptr if no HIP device is
// present.
[[nodiscard]] std::unique_ptr<backend::Backend> make_rocm_backend(int device_id = 0);

} // namespace turbo_ocr::amd
