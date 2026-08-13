#pragma once

// OrtCudaEngine — the second NVIDIA IEngine, wrapping formula::OrtSession
// (ORT + CUDA-13 EP, include/turbo_ocr/analysis/formula/ppformulanet/ort_session.h).
// PP-FormulaNet already runs ORT-CUDA beside the 4 TRT stages in-tree, so two
// IEngine implementations coexisting on the same backend is proven — this
// adapter just gives that ORT session the common face.
//
// Contract (wf_engine.txt ort_cuda_contract) preserved verbatim:
//   caps().io_space               = Cuda   (load(); load_cpu() would be Host)
//   caps().async                  = true   (runs on caller's stream, no sync)
//   caps().caller_owns_outputs    = true   (IoBinding over caller buffers)
//   caps().multi_io               = true
//   caps().graph                  = true   (TRANSPARENT capture on first run)
//   caps().has_profiles           = false
//   caps().thread_safe_concurrent = true   (shared session from worker pool)
//
// graph() reports mode()==Transparent: begin_capture() only fixes the
// persistent binding; the FIRST subsequent run() captures and later ones replay
// automatically (run_graph()); launch() is a no-op returning sentinel slot 0.
// This mirrors OrtSession::run_graph()/reset_graph() exactly.
//
// SCOPE: the production formula recognizer (PPFormulaNetOrt) drives OrtSession
// directly through a bespoke host AR loop with on-GPU argmax + KV ping-pong —
// that stays wrapped as nv_formula_recognizer. This engine adapter is the
// interface-conformant IEngine for generic single-shot ORT-CUDA inference.

#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/backend/engine.h"
#include "turbo_ocr/analysis/formula/ppformulanet/ort_session.h"

namespace turbo_ocr::nvidia {

class OrtCudaEngine final : public backend::IEngine,
                            public backend::IGraphCapture {
public:
  // device_id + the caller's cudaStream_t (as void*) are read here rather than
  // unioned into IEngine::load's signature (there is no common constructor —
  // wf_engine.txt). do_copy_default_stream defaults false (host-loop sessions);
  // set true only for a fused CUDA-Loop graph.
  OrtCudaEngine(int device_id, void *cuda_stream,
                bool do_copy_default_stream = false,
                bool enable_cuda_graph = false)
      : device_id_(device_id), cuda_stream_(cuda_stream),
        do_copy_default_stream_(do_copy_default_stream),
        enable_cuda_graph_(enable_cuda_graph) {}
  ~OrtCudaEngine() override = default;

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

  [[nodiscard]] backend::IGraphCapture *graph() override { return this; }

  // --- IGraphCapture (Transparent) ------------------------------------------
  [[nodiscard]] Mode mode() const override { return Mode::Transparent; }
  [[nodiscard]] int
  begin_capture(const std::vector<backend::DeviceTensor> &inputs,
                const std::vector<backend::DeviceTensor> &outputs,
                backend::DeviceQueue &queue) override;
  [[nodiscard]] bool launch(int /*slot*/, backend::DeviceQueue &) override {
    return true; // transparent replay happens inside run()/run_graph()
  }
  void reset() override;
  [[nodiscard]] std::vector<int64_t> output_shape(int) const override {
    return {};
  }

  [[nodiscard]] formula::OrtSession *native() noexcept { return &session_; }

private:
  formula::OrtSession session_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  int device_id_ = 0;
  void *cuda_stream_ = nullptr;
  bool do_copy_default_stream_ = false;
  bool enable_cuda_graph_ = false;
  bool graph_mode_ = false; // set by begin_capture(); routes run() -> run_graph()
};

} // namespace turbo_ocr::nvidia
