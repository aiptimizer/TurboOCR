#pragma once

#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace turbo_ocr::engine {

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cerr << std::format("[TRT] {}", msg) << '\n';
    }
  }
};

/// TensorRT inference engine wrapper (name-based API, TRT 10+).
class TrtEngine {
public:
  /// Construct with path to a serialized TensorRT engine file (.trt).
  explicit TrtEngine(const std::string &model_path);
  ~TrtEngine() noexcept;

  /// Deserialize and load the engine. Must be called before any inference.
  [[nodiscard]] bool load();

  void bind_io(void *input, void *output);

  // Infer with explicit full input dims (for dynamic shapes).
  // Skips setInputShape if dims haven't changed.
  [[nodiscard]] bool infer_dynamic(const nvinfer1::Dims &input_dims, cudaStream_t stream = 0);

  // Switch optimization profile. Calls setOptimizationProfileAsync if the
  // requested profile differs from the current one. Multi-profile engines
  // (e.g. rec with small/large profiles) use this for better kernel selection.
  void select_profile(int profile_idx, cudaStream_t stream = 0);

  // Returns the number of optimization profiles in the engine.
  [[nodiscard]] int num_profiles() const noexcept;

  [[nodiscard]] nvinfer1::Dims get_output_dims() const noexcept;

  void probe_output_dims(const nvinfer1::Dims &input_dims,
                         int &out_seq_len, int &out_num_classes);

  // ── Multi-IO API (for models with more than one input/output) ──────────
  //
  // PaddleLayout needs this: PP-DocLayoutV3 has 3 inputs (image, im_shape,
  // scale_factor) and 3 outputs. Existing det/rec/cls models are single-IO
  // and keep using bind_io() / infer_dynamic() above — nothing changes for
  // them.

  [[nodiscard]] const std::vector<std::string> &input_names() const noexcept {
    return input_names_;
  }
  [[nodiscard]] const std::vector<std::string> &output_names() const noexcept {
    return output_names_;
  }

  // Bind a single tensor address by name. Call once per tensor after load(),
  // or again after select_profile() to restore addresses TRT cleared.
  void set_tensor_address(const std::string &name, void *ptr);

  // Set the shape of one dynamic-shape input. Idempotent: the first call
  // caches the dims and subsequent calls with the same dims are no-ops.
  // Use this instead of infer_dynamic() for multi-input models.
  [[nodiscard]] bool set_input_shape(const std::string &name,
                                     const nvinfer1::Dims &dims);

  // Execute the context on the given stream. All inputs must have addresses
  // and shapes set. No-op if context is null.
  [[nodiscard]] bool execute(cudaStream_t stream = 0);

  // Query a tensor's current shape (reflects last set_input_shape() calls).
  [[nodiscard]] nvinfer1::Dims tensor_shape(const std::string &name) const;

  // ── Baked CUDA graphs ───────────────────────────────────────────────────
  //
  // A graph is recorded ONCE at warmup on a dedicated execution context that
  // is frozen forever after: fixed profile, fixed shape, fixed I/O addresses,
  // private exactly-sized activation arena. Serving traffic then replays the
  // graph with one cudaGraphLaunch instead of ~100-300 kernel launches.
  // Never capture during traffic: a capture interleaved with other enqueues
  // of the same engine corrupts Myelin state (observed empirically), and
  // lazily allocated arenas make VRAM unbounded.

  // Default ON (rec is launch-bound; see the A/B in trt_engine.cpp).
  // Opt out with TURBO_OCR_CUDA_GRAPHS=0 on VRAM-constrained cards.
  [[nodiscard]] static bool graphs_enabled();

  // Record a graph for `dims` on optimization profile `profile_idx` with the
  // given I/O addresses. Returns a slot id for launch_baked(), or -1 (engine
  // falls back to plain enqueue). Runs two real executions before capture.
  [[nodiscard]] int bake_graph(int profile_idx, const nvinfer1::Dims &dims,
                               void *input, void *output, cudaStream_t stream);

  [[nodiscard]] bool launch_baked(int slot, cudaStream_t stream);

  [[nodiscard]] const nvinfer1::Dims &baked_output_dims(int slot) const {
    return baked_[static_cast<size_t>(slot)].out_dims;
  }

private:
  struct BakedGraph {
    cudaGraphExec_t exec = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx;
    void *arena = nullptr;
    nvinfer1::Dims out_dims{};
  };
  void destroy_graphs() noexcept;

  std::string model_path_;
  // Private engine, deserialized for this instance (see engine_loader.cpp — never
  // shared across pool workers). The shared_ptr keeps the engine (and the TRT
  // runtime captured in its deleter) alive for as long as this instance — and its
  // per-instance contexts/baked graphs — exist.
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;

  std::vector<BakedGraph> baked_;

  // Single-IO legacy (points at input_names_[0] / output_names_[0] when those
  // vectors are non-empty — kept for backward compat with det/rec/cls).
  std::string input_name_;
  std::string output_name_;

  // Multi-IO discovery — populated for all models.
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  nvinfer1::Dims last_input_dims_{};
  int current_profile_ = 0;
  void *bound_input_ = nullptr;
  void *bound_output_ = nullptr;
  // Every tensor address set via set_tensor_address(), so ALL of them can be
  // restored after a profile switch (TRT clears every address, not just the
  // primary IO). Without this, a multi-input model's extra inputs read a stale
  // pointer after select_profile() -> garbage / illegal address.
  std::unordered_map<std::string, void *> extra_addrs_;
};

} // namespace turbo_ocr::engine
