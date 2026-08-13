#pragma once

// TrtEngineAdapter — the NVIDIA IEngine (backend/engine.h) over the existing
// engine::TrtEngine (src/backends/nvidia/engine/trt_engine.h). A pure
// forwarder: it owns a TrtEngine and re-expresses its native contract through
// the caps()-gated interface, keeping every TRT-specific behaviour EXACTLY as
// documented in wf_engine.txt (trt_contract):
//
//   caps().io_space               = Cuda      (caller-owned DEVICE pointers)
//   caps().async                  = true      (enqueueV3; caller syncs)
//   caps().caller_owns_outputs    = true      (TRT writes into caller buffers)
//   caps().multi_io               = true      (name-based; layout is 3-in/3-out)
//   caps().dynamic_shapes         = true      (setInputShape per call)
//   caps().graph                  = true      (bake_graph/launch_baked)
//   caps().has_profiles           = true      (select_profile/num_profiles)
//   caps().thread_safe_concurrent = false     (ONE engine+context per thread)
//
// The two leak-prone optimizers are quarantined behind the optional query
// interfaces: profiles() exposes TRT's multi-profile machinery, graph() exposes
// the EXPLICIT-SLOT bake/launch model (mode()==ExplicitSlot). A backend with no
// profiles/graph returns nullptr from those; this one always returns non-null.
//
// NOTE ON SCOPE: the production det/rec/cls/layout stages keep driving their OWN
// TrtEngine directly (they own ~30 CudaPtr buffers + baked graphs the stage
// logic depends on — see nv_detector/nv_recognizer). This adapter is the
// interface-conformant IEngine for the ONE OcrPipeline's generic engine slot
// and for any future stage rewritten against IEngine. Wrapping here is
// non-regression by construction: it forwards, it does not re-derive.

#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/backend/engine.h"
#include "nvidia/engine/trt_engine.h"

namespace turbo_ocr::nvidia {

class TrtEngineAdapter final : public backend::IEngine,
                              public backend::IProfiles,
                              public backend::IGraphCapture {
public:
  TrtEngineAdapter() = default;
  ~TrtEngineAdapter() override = default;

  // --- IEngine --------------------------------------------------------------
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

  [[nodiscard]] backend::IProfiles *profiles() override { return this; }
  [[nodiscard]] backend::IGraphCapture *graph() override { return this; }

  // --- IProfiles ------------------------------------------------------------
  [[nodiscard]] int num_profiles() const override;
  void select_profile(int idx, backend::DeviceQueue &queue) override;
  [[nodiscard]] bool
  set_input_shape(const std::string &name,
                  const std::vector<int64_t> &dims) override;

  // --- IGraphCapture (ExplicitSlot) -----------------------------------------
  [[nodiscard]] Mode mode() const override { return Mode::ExplicitSlot; }
  [[nodiscard]] int
  begin_capture(const std::vector<backend::DeviceTensor> &inputs,
                const std::vector<backend::DeviceTensor> &outputs,
                backend::DeviceQueue &queue) override;
  [[nodiscard]] bool launch(int slot, backend::DeviceQueue &queue) override;
  void reset() override;
  [[nodiscard]] std::vector<int64_t> output_shape(int slot) const override;

  // Escape hatch for the stage adapters that still own the raw engine (unused
  // by portable callers).
  [[nodiscard]] engine::TrtEngine *native() noexcept { return engine_.get(); }

private:
  std::unique_ptr<engine::TrtEngine> engine_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  int current_profile_ = 0;
};

} // namespace turbo_ocr::nvidia
