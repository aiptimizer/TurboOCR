// OrtCudaEngine implementation — maps backend::DeviceTensor <-> OrtTensor and
// forwards to formula::OrtSession. The session's IoBinding + user-stream +
// transparent-graph semantics are unchanged.

#include "nvidia/engine/ort_cuda_engine.h"

#include "nvidia/support/cuda_common.h" // is_cuda_space, is_i64

namespace turbo_ocr::nvidia {

namespace {
// backend::DeviceTensor -> formula::OrtTensor. Names must outlive the run()
// call; DeviceTensor owns its std::string name, so `.c_str()` is valid for the
// duration of the vector we build from `src`.
std::vector<formula::OrtTensor>
to_ort(const std::vector<backend::DeviceTensor> &src) {
  std::vector<formula::OrtTensor> out;
  out.reserve(src.size());
  for (const auto &t : src)
    out.push_back(formula::OrtTensor{.name = t.name.c_str(),
                                     .data = t.data,
                                     .shape = t.shape,
                                     .i64 = is_i64(t.dtype)});
  return out;
}
} // namespace

bool OrtCudaEngine::load(const std::string &model_path) {
#if defined(TURBO_CPU_ONLY)
  (void)model_path;
  return false; // the CUDA EP is compiled out in the CPU-only build
#else
  if (!session_.load(model_path, device_id_, cuda_stream_,
                     do_copy_default_stream_, enable_cuda_graph_))
    return false;
  input_names_ = session_.input_names();
  output_names_ = session_.output_names();
  return true;
#endif
}

backend::EngineCaps OrtCudaEngine::caps() const {
  backend::EngineCaps c;
  c.io_space = backend::DeviceKind::Cuda;
  c.async = true;
  c.caller_owns_outputs = true;
  c.multi_io = true;
  c.dynamic_shapes = true;
  c.graph = enable_cuda_graph_;
  c.has_profiles = false;
  c.thread_safe_concurrent = true; // shared session, worker-pool driven
  c.dtypes = {backend::DType::F32, backend::DType::I64};
  return c;
}

bool OrtCudaEngine::run(const std::vector<backend::DeviceTensor> &inputs,
                        const std::vector<backend::DeviceTensor> &outputs,
                        std::vector<backend::OutputLease> &leases,
                        backend::DeviceQueue & /*queue*/) {
  // ORT runs on the stream bound at load() (user_compute_stream), not a
  // per-call one — matching OrtSession's contract; `queue` is honored via that
  // load-time binding. leases stays empty: caller owns the bound buffers.
  leases.clear();
  for (const auto &t : inputs)
    if (!is_cuda_space(t))
      return false;
  for (const auto &t : outputs)
    if (!is_cuda_space(t))
      return false;

  const auto in = to_ort(inputs);
  const auto out = to_ort(outputs);
  // Transparent graph: run_graph() reuses the cached binding and captures on
  // first call; plain run() rebuilds the binding each call.
  return graph_mode_ ? session_.run_graph(in, out) : session_.run(in, out);
}

int OrtCudaEngine::begin_capture(
    const std::vector<backend::DeviceTensor> & /*inputs*/,
    const std::vector<backend::DeviceTensor> & /*outputs*/,
    backend::DeviceQueue & /*queue*/) {
  // Transparent capture: just switch run() to the persistent-binding path; the
  // first subsequent run() captures the graph, later ones replay it. The fixed
  // device buffers are the ones the caller passes to run().
  graph_mode_ = true;
  return 0; // sentinel slot
}

void OrtCudaEngine::reset() {
  session_.reset_graph();
  graph_mode_ = false;
}

} // namespace turbo_ocr::nvidia
