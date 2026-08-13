#include "turbo_ocr/analysis/formula/ppformulanet/ort_session.h"
#include "turbo_ocr/onnx/host_ort_threads.h" // host_ort_intra_op_threads
#include "turbo_ocr/base/log/logger.h"

#include <onnxruntime_cxx_api.h>

#include <cstdlib>
#include <iostream>

#include "turbo_ocr/onnx/ort_path.h"  // ORTCHAR_T path (wchar_t on Windows)

namespace turbo_ocr::formula {

namespace {
// Element count of a bind shape (all dims are concrete here).
inline size_t numel(const std::vector<int64_t> &s) {
  size_t n = 1;
  for (int64_t d : s) n *= static_cast<size_t>(d);
  return n;
}
}  // namespace

struct OrtSession::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "ppfns_ort"};
  std::unique_ptr<Ort::Session> sess;
  Ort::MemoryInfo mem{nullptr};       // I/O device: "Cuda" (GPU) or "Cpu" (CPU EP)
  Ort::MemoryInfo cpu_mem{nullptr};   // host target for run_tokens' dynamic output
  int device_id = 0;
  bool ready = false;
  // Persistent binding for run_graph() (CUDA-graph replay): built once, reused.
  std::unique_ptr<Ort::IoBinding> binding;
  std::vector<Ort::Value> gvals;  // own the Value wrappers across replays

  // Build one Ort::Value view over a caller-owned buffer bound by OrtTensor.
  Ort::Value make_value(const OrtTensor &t) const {
    const size_t n = numel(t.shape);
    return t.i64
        ? Ort::Value::CreateTensor<int64_t>(mem, static_cast<int64_t *>(t.data), n,
                                            t.shape.data(), t.shape.size())
        : Ort::Value::CreateTensor<float>(mem, static_cast<float *>(t.data), n,
                                          t.shape.data(), t.shape.size());
  }
};

OrtSession::OrtSession() : p_(std::make_unique<Impl>()) {}
OrtSession::~OrtSession() = default;

#if !defined(TURBO_CPU_ONLY)
bool OrtSession::load(const std::string &onnx_path, int device_id, void *cuda_stream,
                      bool do_copy_default_stream, bool enable_cuda_graph) {
  try {
    p_->device_id = device_id;
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // V2 CUDA provider options so we can optionally hand ORT our cudaStream_t.
    // do_copy_in_default_stream=1 is REQUIRED by ORT's CUDA Loop op (the fused
    // PP-FormulaNet-S AR decoder is a Loop) — a dedicated copy stream is rejected.
    // enable_cuda_graph=1 captures the (static-shape, fixed-address) step graph on the
    // first run_graph() and replays it after — used only for the plus-M decoder step.
    const OrtApi &api = Ort::GetApi();
    OrtCUDAProviderOptionsV2 *cuda = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
    // RAII: release the provider-options handle on every exit path (ORT copies the
    // options into the session on Append, so releasing after is correct) — a throw
    // from any Update*/Append call below would otherwise leak the handle.
    const std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2 *)>
        cuda_guard(cuda, [](OrtCUDAProviderOptionsV2 *c) {
          Ort::GetApi().ReleaseCUDAProviderOptions(c);
        });
    std::string dev = std::to_string(device_id);
    const char *keys[] = {"device_id", "do_copy_in_default_stream", "enable_cuda_graph"};
    const char *vals[] = {dev.c_str(), do_copy_default_stream ? "1" : "0",
                          enable_cuda_graph ? "1" : "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, keys, vals, 3));
    if (cuda_stream)
      Ort::ThrowOnError(
          api.UpdateCUDAProviderOptionsWithValue(cuda, "user_compute_stream", cuda_stream));
    opts.AppendExecutionProvider_CUDA_V2(*cuda);

    p_->sess = std::make_unique<Ort::Session>(p_->env, turbo_ocr::onnx::ort_path(onnx_path).c_str(), opts);
    p_->mem = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device_id, OrtMemTypeDefault);
    p_->cpu_mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    p_->ready = true;
    return true;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR("OrtSession FATAL load failed", "onnx", onnx_path, "error", e.what());
    return false;
  }
}
#endif  // !TURBO_CPU_ONLY

bool OrtSession::load_cpu(const std::string &onnx_path) {
  try {
    p_->device_id = -1;
    Ort::SessionOptions opts;
    // Formula fused graph is validated at ALL (unlike v6 rec, which regresses under
    // SimplifiedLayerNormFusion) — this drives the measured CDM and must not change.
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Bound the per-session intra-op pool: the recognizer runs one shared session
    // concurrently from the worker pool, so the ORT default (= all physical cores per
    // session) oversubscribes under load. Mirror the engine CPU path (intra 4, inter 1,
    // sequential); result is bit-identical (MLAS partitions output tiles with a fixed
    // accumulation order, deterministic across thread count). ORT_NUM_THREADS overrides.
    // The 4 remains this stage's default; ORT_NUM_THREADS still overrides it,
    // and a backend whose det/rec are on an accelerator can raise it without
    // this file (and the two other host ORT stages) drifting apart — see
    // common/host_ort_threads.h.
    opts.SetIntraOpNumThreads(turbo_ocr::host_ort_intra_op_threads(4));
    opts.SetInterOpNumThreads(1);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    opts.EnableCpuMemArena();
    // No execution provider appended => ORT's default CPUExecutionProvider.
    p_->sess = std::make_unique<Ort::Session>(p_->env, turbo_ocr::onnx::ort_path(onnx_path).c_str(), opts);
    p_->mem = Ort::MemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
    p_->cpu_mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    p_->ready = true;
    return true;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR("OrtSession FATAL CPU load failed", "onnx", onnx_path, "error", e.what());
    return false;
  }
}

bool OrtSession::ready() const { return p_->ready; }

std::vector<std::string> OrtSession::input_names() const {
  std::vector<std::string> names;
  if (!p_->sess) return names;
  Ort::AllocatorWithDefaultOptions alloc;
  size_t n = p_->sess->GetInputCount();
  names.reserve(n);
  for (size_t i = 0; i < n; ++i)
    names.emplace_back(p_->sess->GetInputNameAllocated(i, alloc).get());
  return names;
}

std::vector<std::string> OrtSession::output_names() const {
  std::vector<std::string> names;
  if (!p_->sess) return names;
  Ort::AllocatorWithDefaultOptions alloc;
  size_t n = p_->sess->GetOutputCount();
  names.reserve(n);
  for (size_t i = 0; i < n; ++i)
    names.emplace_back(p_->sess->GetOutputNameAllocated(i, alloc).get());
  return names;
}

bool OrtSession::run(const std::vector<OrtTensor> &inputs,
                     const std::vector<OrtTensor> &outputs) {
  try {
    Ort::IoBinding binding(*p_->sess);
    std::vector<Ort::Value> vals;  // own the Value wrappers for the Run() lifetime
    vals.reserve(inputs.size() + outputs.size());
    for (const auto &t : inputs) {
      vals.push_back(p_->make_value(t));
      binding.BindInput(t.name, vals.back());
    }
    for (const auto &t : outputs) {
      vals.push_back(p_->make_value(t));
      binding.BindOutput(t.name, vals.back());
    }
    p_->sess->Run(Ort::RunOptions{nullptr}, binding);
    return true;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("OrtSession run failed", "error", e.what());
    return false;
  }
}

bool OrtSession::run_graph(const std::vector<OrtTensor> &inputs,
                           const std::vector<OrtTensor> &outputs) {
  try {
    if (!p_->binding) {  // first call: build + cache the binding (fixed buffers)
      p_->binding = std::make_unique<Ort::IoBinding>(*p_->sess);
      p_->gvals.clear();
      p_->gvals.reserve(inputs.size() + outputs.size());
      for (const auto &t : inputs) {
        p_->gvals.push_back(p_->make_value(t));
        p_->binding->BindInput(t.name, p_->gvals.back());
      }
      for (const auto &t : outputs) {
        p_->gvals.push_back(p_->make_value(t));
        p_->binding->BindOutput(t.name, p_->gvals.back());
      }
    }
    p_->sess->Run(Ort::RunOptions{nullptr}, *p_->binding);  // captures on 1st call, replays after
    return true;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("OrtSession run_graph failed", "error", e.what());
    return false;
  }
}

void OrtSession::reset_graph() {
  p_->binding.reset();
  p_->gvals.clear();
}

bool OrtSession::run_tokens(const char *in_name, const char *out_name, const float *d_x,
                            int64_t B, std::vector<int64_t> &tokens, int64_t &L,
                            int64_t &rows) {
  rows = 0;
  try {
    Ort::IoBinding binding(*p_->sess);
    int64_t shp[4] = {B, 1, 384, 384};
    Ort::Value xv = Ort::Value::CreateTensor<float>(
        p_->mem, const_cast<float *>(d_x), (size_t)B * 1 * 384 * 384, shp, 4);
    binding.BindInput(in_name, xv);
    // Let ORT allocate the dynamic [B,L] token output directly in host memory
    // (cached MemoryInfo — avoids re-interning "Cpu" on every crop).
    binding.BindOutput(out_name, p_->cpu_mem);
    binding.SynchronizeInputs();
    p_->sess->Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
    std::vector<Ort::Value> outs = binding.GetOutputValues();
    auto shape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    L = shape.size() >= 2 ? shape[1] : 0;
    // Copy exactly what the model returned: an output batch smaller than the
    // requested B would over-read ORT's buffer if we trusted the input B.
    // `rows` goes OUT so the caller can apply the same discipline to `tokens`
    // — it used to be discarded here, which just moved the over-read from
    // ORT's buffer into `tokens`.
    rows = shape.empty() ? 0 : shape[0];
    const size_t emitted = static_cast<size_t>(rows) * static_cast<size_t>(L);
    const int64_t *d = outs[0].GetTensorData<int64_t>();
    tokens.assign(d, d + emitted);
    return true;
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("OrtSession run_tokens failed", "error", e.what());
    return false;
  }
}

}  // namespace turbo_ocr::formula
