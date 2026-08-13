#pragma once

// CpuEngineAdapter — the CpuBackend IEngine (backend/engine.h) over the existing
// engine::OrtEngine (ORT-CPU/XNNPACK/CoreML/DNNL). It is the host analogue of
// nvidia/engine/trt_engine_adapter.h, but reports the OrtEngine's honest native
// contract via caps(): HOST io_space, SYNCHRONOUS (async=false), single-IO
// float32, and OUTPUTS COPIED OUT (caller_owns_outputs=false) — results come
// back as an OutputLease into an engine-owned buffer, valid until the next run().
//
// No profiles, no CUDA graph: profiles()/graph() stay nullptr.

#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/backend/engine.h"

#include "turbo_ocr/onnx/ort_engine.h"

namespace turbo_ocr::cpu {

class CpuEngineAdapter final : public backend::IEngine {
public:
  CpuEngineAdapter() = default;
  ~CpuEngineAdapter() override = default;

  [[nodiscard]] bool load(const std::string &model_path) override;
  [[nodiscard]] backend::EngineCaps caps() const override;

  [[nodiscard]] const std::vector<std::string> &input_names() const override {
    return input_names_;
  }
  [[nodiscard]] const std::vector<std::string> &output_names() const override {
    return output_names_;
  }

  [[nodiscard]] bool run(const std::vector<backend::DeviceTensor> &inputs,
                         const std::vector<backend::DeviceTensor> &outputs,
                         std::vector<backend::OutputLease> &leases,
                         backend::DeviceQueue &queue) override;

private:
  std::unique_ptr<engine::OrtEngine> engine_;
  // Single-IO names (OrtEngine is index-0 only); ORT resolves the real tensor
  // names internally, so these are stable placeholders for the seam.
  std::vector<std::string> input_names_{"input"};
  std::vector<std::string> output_names_{"output"};
  // Holds the most recent output so an OutputLease can point into it until the
  // next run() (OrtEngine::infer returns an owned copy).
  engine::OrtEngine::InferResult last_result_;
};

} // namespace turbo_ocr::cpu
