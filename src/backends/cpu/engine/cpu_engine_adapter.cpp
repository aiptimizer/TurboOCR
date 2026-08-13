// CpuEngineAdapter implementation — thin forwarding to engine::OrtEngine, with
// the single-IO float32 / copy-out contract surfaced through OutputLease.

#include "cpu/engine/cpu_engine_adapter.h"

namespace turbo_ocr::cpu {

bool CpuEngineAdapter::load(const std::string &model_path) {
  engine_ = std::make_unique<engine::OrtEngine>(model_path);
  return engine_->load();
}

backend::EngineCaps CpuEngineAdapter::caps() const {
  backend::EngineCaps c;
  c.io_space = backend::DeviceKind::Host;
  c.async = false;              // synchronous; results ready on return
  c.caller_owns_outputs = false; // results come back via OutputLease
  c.multi_io = false;           // single index-0 IO
  c.dynamic_shapes = true;      // per-call input shape
  c.graph = false;
  c.has_profiles = false;
  c.thread_safe_concurrent = false; // holds per-run output state; one per thread
  c.dtypes = {backend::DType::F32};
  return c;
}

bool CpuEngineAdapter::run(const std::vector<backend::DeviceTensor> &inputs,
                           const std::vector<backend::DeviceTensor> & /*outputs*/,
                           std::vector<backend::OutputLease> &leases,
                           backend::DeviceQueue & /*queue*/) {
  if (!engine_ || inputs.empty() || inputs[0].data == nullptr)
    return false;
  // Single-IO float32 host input, bound in place (zero-copy into OrtEngine).
  const auto *in = static_cast<const float *>(inputs[0].data);
  last_result_ = engine_->infer(in, inputs[0].shape);
  if (last_result_.empty())
    return false;

  leases.clear();
  backend::OutputLease lease;
  lease.name = output_names_.front();
  lease.data = last_result_.data.data(); // valid until the next run()
  lease.space = backend::DeviceKind::Host;
  lease.dtype = backend::DType::F32;
  lease.shape = last_result_.shape;
  leases.push_back(std::move(lease));
  return true;
}

} // namespace turbo_ocr::cpu
