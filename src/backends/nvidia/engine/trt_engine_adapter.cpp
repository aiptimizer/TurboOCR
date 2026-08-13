// TrtEngineAdapter implementation — forwards backend::IEngine calls to
// engine::TrtEngine's native name-based TRT-10 API. No new device logic.

#include "nvidia/engine/trt_engine_adapter.h"

#include "nvidia/support/cuda_common.h" // to_trt_dims/from_trt_dims, cuda_stream, is_cuda_space

namespace turbo_ocr::nvidia {

bool TrtEngineAdapter::load(const std::string &model_path) {
  engine_ = std::make_unique<engine::TrtEngine>(model_path);
  if (!engine_->load())
    return false;
  input_names_ = engine_->input_names();
  output_names_ = engine_->output_names();
  return true;
}

backend::EngineCaps TrtEngineAdapter::caps() const {
  backend::EngineCaps c;
  c.io_space = backend::DeviceKind::Cuda;
  c.async = true;
  c.caller_owns_outputs = true; // TRT writes into the caller's output buffer
  c.multi_io = true;
  c.dynamic_shapes = true;
  c.graph = engine::TrtEngine::graphs_enabled();
  c.has_profiles = true;
  c.thread_safe_concurrent = false; // one engine+context per worker thread
  c.dtypes = {backend::DType::F32, backend::DType::F16, backend::DType::I32,
              backend::DType::I64};
  return c;
}

bool TrtEngineAdapter::run(const std::vector<backend::DeviceTensor> &inputs,
                           const std::vector<backend::DeviceTensor> &outputs,
                           std::vector<backend::OutputLease> &leases,
                           backend::DeviceQueue &queue) {
  leases.clear(); // caller owns outputs; no lease for TRT
  if (!engine_ || inputs.empty() || outputs.empty())
    return false;
  for (const auto &t : inputs)
    if (!is_cuda_space(t))
      return false; // TRT requires DEVICE pointers (wf_engine.txt)
  for (const auto &t : outputs)
    if (!is_cuda_space(t))
      return false;

  const cudaStream_t stream = cuda_stream(queue);

  // Single-IO legacy fast path (det/rec/cls): bind_io + infer_dynamic, which
  // caches last_input_dims_ and skips setInputShape when unchanged.
  if (inputs.size() == 1 && outputs.size() == 1) {
    engine_->bind_io(inputs[0].data, outputs[0].data);
    return engine_->infer_dynamic(to_trt_dims(inputs[0].shape), stream);
  }

  // Multi-IO path (PP-DocLayoutV3 and table-cell models): set every address +
  // input shape by name, then execute().
  for (const auto &t : inputs) {
    engine_->set_tensor_address(t.name, t.data);
    if (!engine_->set_input_shape(t.name, to_trt_dims(t.shape)))
      return false;
  }
  for (const auto &t : outputs)
    engine_->set_tensor_address(t.name, t.data);
  return engine_->execute(stream);
}

// ---- IProfiles -------------------------------------------------------------

int TrtEngineAdapter::num_profiles() const {
  return engine_ ? engine_->num_profiles() : 0;
}

void TrtEngineAdapter::select_profile(int idx, backend::DeviceQueue &queue) {
  if (engine_) {
    engine_->select_profile(idx, cuda_stream(queue));
    current_profile_ = idx;
  }
}

bool TrtEngineAdapter::set_input_shape(const std::string &name,
                                       const std::vector<int64_t> &dims) {
  return engine_ && engine_->set_input_shape(name, to_trt_dims(dims));
}

// ---- IGraphCapture (ExplicitSlot) ------------------------------------------

int TrtEngineAdapter::begin_capture(
    const std::vector<backend::DeviceTensor> &inputs,
    const std::vector<backend::DeviceTensor> &outputs,
    backend::DeviceQueue &queue) {
  // TRT bakes a graph on the CURRENT profile for the given fixed shape +
  // fixed I/O addresses; -1 => caller falls back to plain run(). Must be
  // called at warmup, never during traffic (Myelin corruption).
  if (!engine_ || inputs.size() != 1 || outputs.size() != 1)
    return -1;
  return engine_->bake_graph(current_profile_, to_trt_dims(inputs[0].shape),
                             inputs[0].data, outputs[0].data,
                             cuda_stream(queue));
}

bool TrtEngineAdapter::launch(int slot, backend::DeviceQueue &queue) {
  return engine_ && slot >= 0 && engine_->launch_baked(slot, cuda_stream(queue));
}

void TrtEngineAdapter::reset() {
  // Honours the contract instead of ignoring it. This was a documented no-op on
  // the grounds that NVIDIA never rebinds buffers after warmup — true of today's
  // callers, but reset() exists precisely for the caller that does, and a baked
  // graph replayed against rebound buffers reads the addresses it recorded.
  // Silently doing nothing here would surface as memory corruption, not as an
  // unimplemented feature.
  if (engine_)
    engine_->destroy_graphs();
}

std::vector<int64_t> TrtEngineAdapter::output_shape(int slot) const {
  if (!engine_ || slot < 0)
    return {};
  return from_trt_dims(engine_->baked_output_dims(slot));
}

} // namespace turbo_ocr::nvidia
