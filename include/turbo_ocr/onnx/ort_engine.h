#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include "turbo_ocr/backend/engine_mode.h" // backend::EpConfig, Fp16Support

// ONNX Runtime inference engine.
// Loads ONNX models and runs them with all optimizations enabled, on the
// default CPU provider or on any vendor execution provider (CUDA / CoreML /
// OpenVINO / MIGraphX / DirectML / XNNPACK / oneDNN). This is the engine
// behind EVERY backend's FAST path (backend/engine_mode.h): the .onnx as-is,
// no graph build.

namespace turbo_ocr::engine {

/// ONNX Runtime inference engine (drop-in replacement for TrtEngine).
class OrtEngine {
public:
  /// The RESOLVED execution provider this engine runs on ("cpu", "cuda",
  /// "migraphx", "dml", ...) — env-derived or explicit. Lets stage code make
  /// device-vs-host decisions (e.g. recognition batching defaults) without
  /// re-reading process env.
  [[nodiscard]] const std::string &provider() const { return ep_.provider; }

  /// Construct with path to an ONNX model file (.onnx). The execution provider
  /// comes from the ORT_EP env var — the historical behaviour, kept for every
  /// existing call site.
  explicit OrtEngine(const std::string &model_path);

  /// Construct with an EXPLICIT provider config (backend/engine_mode.h).
  ///
  /// Preferred over the env form for anything the backend seam drives: one
  /// process can hold two backends on two providers, which a process-global
  /// ORT_EP cannot express. `ep.fp16` is honoured per the provider's real
  /// Fp16Support class — a provider knob for OpenVINO, already-native for
  /// CoreML, and an fp16 MODEL (a sibling `*.fp16.onnx`, when present) for
  /// CUDA/DirectML/MIGraphX. It never triggers a graph/engine build.
  OrtEngine(const std::string &model_path, const backend::EpConfig &ep);
  ~OrtEngine() noexcept = default;

  /// Load the ONNX model and create an inference session.
  [[nodiscard]] bool load();

  // Run inference. Input is a flat float buffer with given shape.
  // Returns output as a flat float vector + output shape.
  struct InferResult {
    std::vector<float> data;
    std::vector<int64_t> shape;

    [[nodiscard]] bool empty() const noexcept { return data.empty(); }
  };

  [[nodiscard]] InferResult infer(const float *input_data,
                                  const std::vector<int64_t> &input_shape);

  // Batched inference. input_shape is {B,3,H,W}; the returned tensor is the
  // full {B,seq,classes} output so the caller can slice each row. Identical
  // mechanics to infer() — kept separate for call-site clarity.
  [[nodiscard]] InferResult infer_batch(const float *input_data,
                                        const std::vector<int64_t> &input_shape);

  // Zero-copy batched inference. Same input contract as infer_batch, but
  // returns a view into an engine-owned output buffer instead of copying the
  // tensor out. The view (and its data pointer) is valid until the next
  // infer*/probe call on THIS engine. Used by the rec hot path, where the
  // {B,seq,classes} tensor is large and the caller fully consumes one batch
  // before issuing the next Run.
  struct InferView {
    const float *data = nullptr;
    std::vector<int64_t> shape;

    [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  };

  [[nodiscard]] InferView
  infer_batch_view(const float *input_data,
                   const std::vector<int64_t> &input_shape);

  // Probe output dims for a given input shape (runs a dummy inference)
  void probe_output_dims(const std::vector<int64_t> &input_shape,
                         int &out_dim1, int &out_dim2);

private:
  std::string model_path_;
  // When ORT_SHARED_POOL=1, sessions draw threads from one process-wide pool
  // (see process_env()) instead of each owning its own intra-op threadpool.
  bool use_shared_pool_ = false;
  // Alternate execution provider from env ORT_EP; empty/"cpu" = default MLAS.
  // Portable Tier-B backend switch — one host pipeline, one EP string per vendor:
  //   "xnnpack" / "dnnl"  — CPU accel (ARM/x86)
  //   "coreml"            — Apple ANE/GPU (auto on macOS; see load() ctor)
  //   "openvino"          — Intel CPU/iGPU/Arc/NPU (OPENVINO_DEVICE=AUTO|CPU|GPU|NPU)
  //   "migraphx" / "rocm" — AMD Instinct/Radeon on Linux+ROCm (ROCM_DEVICE_ID)
  //   "dml"               — DirectML, vendor-agnostic Windows consumer GPUs
  // Each is applied in load(); a provider missing from the linked onnxruntime
  // build fails cleanly (clear load() error) rather than at link time.
  std::string ort_ep_;
  // Explicit provider config when constructed through the EpConfig ctor; when
  // the env ctor was used this mirrors ORT_EP with fp16 forced OFF (an env-only
  // caller expressed no fp16 opinion) so apply_execution_provider() has ONE
  // code path for both.
  backend::EpConfig ep_{};
  // True only for the EpConfig ctor. Gates the behaviour changes that must NOT
  // reach existing env-driven call sites — chiefly macOS CoreML, which every
  // OrtEngine has attached by default since before providers were selectable.
  bool ep_explicit_ = false;
  // Whether configure_session_() ACTUALLY attached the CoreML provider — the
  // outcome, not the intent. False when DISABLE_COREML=1 overrode the request
  // or the append returned a non-OK status. apply_execution_provider() reads it
  // so an explicit `coreml` request cannot silently run on plain MLAS while the
  // backend keeps reporting the apple/onnx path. Always false off macOS.
  bool coreml_attached_ = false;
  std::unique_ptr<Ort::Session> session_;
  Ort::SessionOptions session_options_;

  // Shared tail of both constructors (threads, opt level, arena, CoreML).
  void configure_session_();
  // Swap in a sibling `<stem>.fp16.onnx` when fp16 was asked for on a provider
  // whose fp16 is a property of the MODEL, not of the session. No-op otherwise.
  void resolve_fp16_model_();

  // Configure session_options_ with the ORT_EP provider (throws Ort::Exception
  // if the requested provider isn't available in this onnxruntime build).
  void apply_execution_provider();

  // The single process-wide ORT Env, lazily created on first use. ORT fixes an
  // Env's threadpool config at first creation, so we must NOT eagerly construct
  // a plain Env per engine: when ORT_SHARED_POOL=1 this builds the Env with
  // global threadpools (thread count from ORT_GLOBAL_THREADS); otherwise a
  // plain Env with per-session threadpools (today's behavior).
  static Ort::Env &process_env();

  std::string input_name_;
  std::string output_name_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // Owns the output tensor from the most recent infer_batch_view call so its
  // buffer outlives the returned view (until the next inference call).
  Ort::Value last_output_{nullptr};
};

} // namespace turbo_ocr::engine
