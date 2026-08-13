#pragma once

// IntelBackend — the Intel vendor's single Backend implementation (backend.h),
// at the stages_* seam altitude. It hands the ONE shared UnifiedOcrPipeline the
// Intel device factories (L0 queue / USM allocator / SYCL kernels / OpenVINO
// engine) plus the constructed stages, and the two genuinely per-vendor
// service-boundary functions (image decode, page orientation).
//
// DEDUP (deliberate absences, mirroring src/backends/cpu/backend/cpu_backend.h):
//   * There is NO make_infer_func(). The det->cls->rec->layout->router->
//     table/formula orchestration is written exactly once, above the seam, in
//     turbo_ocr::pipeline::make_infer_func over a UnifiedOcrPipeline pool. Every
//     backend that used to override it carried a private copy of that loop.
//   * There is NO Intel-private table/formula recognizer. Table/formula specs
//     go to the SHARED registries; only a genuinely device-resident local
//     structure encoder would justify a vendor class here, and that is a
//     follow-up (see README), not a fork of the dispatch.
//
// Device identity: DeviceKind::L0 for the GPU/NPU plugins, Host for the
// OpenVINO CPU plugin. Selected at startup from OV_DEVICE.

#include <memory>
#include <string>
#include <string_view>

#include "intel/engine/openvino_engine.h"
#include "intel/memory/l0_allocator.h"
#include "turbo_ocr/backend/backend.h"

namespace turbo_ocr::intel {

class IntelBackend final : public backend::Backend {
public:
  explicit IntelBackend(OpenVINOEngine::DeviceType device);
  ~IntelBackend() override;

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

  [[nodiscard]] server::ImageDecoder make_image_decoder() override;
  [[nodiscard]] server::OrientFunc make_orient_func() override;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

// Intel vendor entry point, called by the make_backend() dispatch. `device`
// defaults to OV_DEVICE (GPU when unset).
std::unique_ptr<backend::Backend>
make_intel_backend(OpenVINOEngine::DeviceType device =
                       OpenVINOEngine::device_from_env(OpenVINOEngine::DeviceType::GPU));

} // namespace turbo_ocr::intel
