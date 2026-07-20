#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace turbo_ocr::formula {

// One tensor bound by pointer (device pointer for a CUDA session, host pointer for
// a CPU session — matches the session's MemoryInfo). i64=true for int64 inputs
// (tokens/pos), false for float32.
struct OrtTensor {
  const char *name = nullptr;
  void *data = nullptr;       // device (CUDA session) or host (CPU session) pointer
  std::vector<int64_t> shape;
  bool i64 = false;
};

// Thin in-process ONNX Runtime CUDA-13 session: runs a model with all I/O bound
// to GPU device pointers (IoBinding), so the formula encoder, cross-KV prep, and
// static-KV step all execute in ORT — matching the fused ORT baseline bit-for-bit
// — without ever leaving the process (no Python sidecar, no RPC, no dropped
// formulas). pImpl keeps the ORT C++ headers out of nvcc translation units.
class OrtSession {
public:
  OrtSession();
  ~OrtSession();

  // cuda_stream (a cudaStream_t as void*) makes ORT run on the caller's stream so
  // encoder, prep, and the whole step loop stay async on ONE stream — no per-op
  // host sync; the caller syncs only when it needs results (deferred EOS check).
  // do_copy_default_stream MUST be true for models with a CUDA Loop op (the fused
  // graph), but for the host-loop sessions it must be FALSE so output copies land on
  // cuda_stream (else they race the caller's kernels on that stream).
  //
  // Compiled out of the CPU-only build (TURBO_CPU_ONLY): it appends the CUDA
  // execution provider, so the CUDA-free CPU library never references it — only
  // load_cpu() below is available there.
#if !defined(TURBO_CPU_ONLY)
  [[nodiscard]] bool load(const std::string &onnx_path, int device_id,
                          void *cuda_stream = nullptr, bool do_copy_default_stream = true,
                          bool enable_cuda_graph = false);
#endif

  // Load on ORT's CPUExecutionProvider (no CUDA provider, no user stream). run()/
  // run_tokens() then bind HOST pointers (MemoryInfo "Cpu"). Pure C++, no Python.
  [[nodiscard]] bool load_cpu(const std::string &onnx_path);

  [[nodiscard]] bool ready() const;

  // Graph I/O names (resolved from the loaded session). Empty before load.
  [[nodiscard]] std::vector<std::string> input_names() const;
  [[nodiscard]] std::vector<std::string> output_names() const;

  // Bind inputs/outputs (device pointers) and enqueue on the session's stream.
  // Does NOT block the host or sync the stream — the caller orders/sync's.
  [[nodiscard]] bool run(const std::vector<OrtTensor> &inputs,
                         const std::vector<OrtTensor> &outputs);

  // Persistent-binding run for a STATIC-shape, FIXED-ADDRESS step (the plus-M decoder
  // step). Builds the IoBinding once from the first call's tensors and reuses it every
  // call — so a session loaded with enable_cuda_graph captures the CUDA graph on the
  // first call and replays it after, eliminating per-step kernel-launch + binding
  // overhead. The caller MUST pass the SAME device buffers on every call (only their
  // contents change); call reset_graph() before changing any buffer or shape.
  [[nodiscard]] bool run_graph(const std::vector<OrtTensor> &inputs,
                               const std::vector<OrtTensor> &outputs);
  void reset_graph();  // drop the cached binding/captured graph

  // Run a single-input (device float) model whose ONE output is a dynamic int64
  // token tensor [B, L] (the fused PP-FormulaNet-S graph, batched). ORT allocates
  // the [B,L] output on the host; tokens is filled row-major, L set to the length.
  [[nodiscard]] bool run_tokens(const char *in_name, const char *out_name,
                                const float *d_x, int64_t B,
                                std::vector<int64_t> &tokens, int64_t &L);

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace turbo_ocr::formula
