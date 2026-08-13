#pragma once

// CpuBackend — the CPU implementation of the ONE device seam (backend/backend.h).
// It is the single object the merged server_main constructs for a CPU build (or
// the auto-detect fallback when no accelerator is present); from it flow the
// device factories the ONE OcrPipeline uses (queue / allocator / kernels /
// engine / stages) and the service-boundary functions the routes consume.
//
// It is the CPU peer of nvidia/backend/cuda_backend.h, but far simpler: the Host address
// space IS host RAM, so the queue is a synchronous no-op, the allocator is plain
// malloc, and the stages wrap the existing CpuPaddle* classes directly (no
// device transfer). This header pulls in no ORT/OpenCV type in its signatures.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/backend.h"

#include "turbo_ocr/analysis/classification/ort_doc_orientation.h"

namespace turbo_ocr::cpu {

class HostAllocator;

class CpuBackend final : public backend::Backend {
public:
  CpuBackend();
  ~CpuBackend() override;

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
  [[nodiscard]] backend::StageSet
  load_stages(const backend::BackendConfig &cfg) override;

  // --- High-level service-boundary functions --------------------------------
  // NOTE (dedup): no make_infer_func() — see backend.h. The ONE InferFunc is
  // pipeline::make_infer_func() over a UnifiedOcrPipeline pool.
  [[nodiscard]] server::ImageDecoder make_image_decoder() override;
  [[nodiscard]] server::OrientFunc make_orient_func() override;

private:
  std::shared_ptr<HostAllocator> allocator_;
  std::unique_ptr<classification::OrtDocOrientation> doc_ori_;
  bool doc_ori_ready_ = false;
  int pool_size_ = 0; // 0 => picked from hardware_concurrency in caps()
};

} // namespace turbo_ocr::cpu
