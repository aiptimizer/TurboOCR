#include "nvidia/engine/trt_engine.h"

#include "turbo_ocr/base/env_utils.h" // env::env_or — records the read
#include "nvidia/support/cuda_check.h"
#include "turbo_ocr/base/errors.h"
#include "nvidia/engine/engine_loader.h"

#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

using namespace turbo_ocr::engine;

TrtEngine::TrtEngine(const std::string &model_path) : model_path_(model_path) {}

TrtEngine::~TrtEngine() noexcept { destroy_graphs(); }

bool TrtEngine::graphs_enabled() {
  // Default ON: a clean free-GPU A/B (FUNSD, tiny tier, 2026-07-01) measured baked
  // graphs at +14% throughput (515.7 -> 586.4 img/s) and lower p50 (13 -> 11 ms),
  // F1 unchanged (85.38 -> 85.37) — the many small rec crops are launch-bound, so
  // replaying a graph beats re-issuing ~100-300 kernels per crop. Cost is
  // ~0.5 GiB/pipeline of VRAM; opt OUT with TURBO_OCR_CUDA_GRAPHS=0 on
  // VRAM-constrained cards. (The earlier "graphs neutral" note measured under GPU
  // contention / on the compute-bound det path — not the rec launch-bound reality.)
  static const bool v = [] {
    const std::string env = turbo_ocr::env::env_or("TURBO_OCR_CUDA_GRAPHS", "");
    return !(!env.empty() && std::atoi(env.c_str()) == 0); // on unless explicitly disabled
  }();
  return v;
}

void TrtEngine::destroy_graphs() noexcept {
  for (auto &g : baked_) {
    if (g.exec)
      cudaGraphExecDestroy(g.exec);
    g.ctx.reset();
    if (g.arena)
      cudaFree(g.arena);
  }
  baked_.clear();
}

int TrtEngine::bake_graph(int profile_idx, const nvinfer1::Dims &dims,
                          void *input, void *output, cudaStream_t stream) {
  if (!engine_ || !graphs_enabled())
    return -1;
  // Pipelines warm up concurrently. bake_mu serializes CAPTURES against each
  // other only — it does not (and must not) stop other pipelines' warmup
  // enqueues/allocs, which is why the capture runs in ThreadLocal mode:
  // Global mode would make those legitimate concurrent cudaMallocs fail
  // spuriously. The safety invariant is weaker but sufficient: every pipeline
  // owns a private engine/context/stream, so foreign work can never be
  // recorded into this capture; the only cross-thread hazard is an implicit
  // legacy-stream dependency invalidating the capture, which surfaces as a
  // DETECTED EndCapture error below and falls back to plain enqueue.
  static std::mutex bake_mu;
  std::lock_guard<std::mutex> lock(bake_mu);

  auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (!ctx)
    return -1;
  if (profile_idx != 0) {
    // A failed profile switch must not bake a graph bound to the wrong
    // profile — that would silently serve wrong-shape inference forever.
    if (!ctx->setOptimizationProfileAsync(profile_idx, stream) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      (void)cudaGetLastError();
      return -1;
    }
  }
  ctx->setTensorAddress(input_name_.c_str(), input);
  ctx->setTensorAddress(output_name_.c_str(), output);
  if (!ctx->setInputShape(input_name_.c_str(), dims) ||
      !ctx->allInputDimensionsSpecified())
    return -1;

  const int64_t arena_size =
      engine_->getDeviceMemorySizeForProfileV2(profile_idx);
  void *arena = nullptr;
  if (arena_size <= 0 ||
      cudaMalloc(&arena, static_cast<size_t>(arena_size)) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  ctx->setDeviceMemoryV2(arena, arena_size);

  auto fail = [&] {
    cudaGetLastError();
    ctx.reset(); // before the arena it points into
    cudaFree(arena);
    return -1;
  };

  // Two real executions then a synced capture. TRT's documented contract is
  // "at least one enqueue before capturing" (the first phase flushes deferred
  // shape/profile updates and is not capturable); trtexec does exactly one,
  // two is a strict superset kept for Myelin state finalization headroom.
  for (int k = 0; k < 2; ++k)
    if (!ctx->enqueueV3(stream))
      return fail();
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    return fail();

  cudaGraph_t graph = nullptr;
  if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) !=
      cudaSuccess)
    return fail();
  const bool enq_ok = ctx->enqueueV3(stream);
  const cudaError_t cap = cudaStreamEndCapture(stream, &graph);
  if (!enq_ok || cap != cudaSuccess || !graph) {
    if (graph)
      cudaGraphDestroy(graph);
    std::cerr << std::format(
        "[TRT] graph capture failed for {} profile {} — plain enqueue\n",
        model_path_, profile_idx);
    return fail();
  }
  cudaGraphExec_t exec = nullptr;
  if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    return fail();
  }
  cudaGraphDestroy(graph);

  // Smoke-replay once so a broken graph surfaces at warmup, not in traffic.
  if (cudaGraphLaunch(exec, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaGraphExecDestroy(exec);
    return fail();
  }

  BakedGraph g;
  g.exec = exec;
  g.ctx = std::move(ctx);
  g.arena = arena;
  g.out_dims = g.ctx->getTensorShape(output_name_.c_str());
  baked_.push_back(std::move(g));
  return static_cast<int>(baked_.size()) - 1;
}

bool TrtEngine::launch_baked(int slot, cudaStream_t stream) {
  return slot >= 0 && static_cast<size_t>(slot) < baked_.size() &&
         cudaGraphLaunch(baked_[static_cast<size_t>(slot)].exec, stream) ==
             cudaSuccess;
}

bool TrtEngine::load() {
  // load_engine() deserializes a PRIVATE engine for this instance; the
  // IExecutionContext below is per-instance/per-thread. Engines are never shared
  // across workers — N contexts on one shared multi-profile engine corrupt rec
  // output under load (see engine_loader.cpp).
  engine_ = engine::load_engine(model_path_);
  if (!engine_) [[unlikely]] {
    // load_engine already logged the specific read/deserialize failure.
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) [[unlikely]] {
    std::cerr << std::format("[TRT] Failed to create execution context: {}", model_path_) << '\n';
    return false;
  }

  auto nbIO = engine_->getNbIOTensors();
  for (int i = 0; i < nbIO; ++i) {
    const char *name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      input_names_.emplace_back(name);
    else
      output_names_.emplace_back(name);
  }
  // Single-IO convenience pointers (first of each). Multi-IO callers use
  // input_names() / output_names() directly.
  if (!input_names_.empty())  input_name_  = input_names_[0];
  if (!output_names_.empty()) output_name_ = output_names_[0];

  return true;
}

void TrtEngine::bind_io(void *input, void *output) {
  if (!context_) [[unlikely]]
    return;
  bound_input_ = input;
  bound_output_ = output;
  context_->setTensorAddress(input_name_.c_str(), input);
  context_->setTensorAddress(output_name_.c_str(), output);
}

static constexpr bool dims_equal(const nvinfer1::Dims &a, const nvinfer1::Dims &b) {
  if (a.nbDims != b.nbDims) return false;
  for (int i = 0; i < a.nbDims; ++i)
    if (a.d[i] != b.d[i]) return false;
  return true;
}

bool TrtEngine::infer_dynamic(const nvinfer1::Dims &input_dims,
                              cudaStream_t stream) {
  if (!context_) [[unlikely]]
    return false;
  if (!dims_equal(input_dims, last_input_dims_)) {
    // TRT 10.14: all profiles share the same base tensor names.
    // setInputShape applies to whichever profile is currently active.
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) [[unlikely]] {
      std::cerr << std::format("[TRT] setInputShape FAILED for input=({},{},{},{}) profile={} on {}",
                               input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3],
                               current_profile_, model_path_) << '\n';
      return false;
    }
    last_input_dims_ = input_dims;
  }
  // PP-DocLayoutV3 (and any future multi-input model) has more than one
  // dynamic-shape input. If a caller forgets to set one, enqueueV3 silently
  // produces garbage output with no error, so guard the dispatch.
  if (!context_->allInputDimensionsSpecified()) [[unlikely]]
    throw turbo_ocr::InferenceError(
        "input dimensions not all set before enqueueV3 (" + model_path_ + ")");
  if (!context_->enqueueV3(stream)) [[unlikely]] {
    // A sticky fault here (illegal address, ECC, launch failure, …) has
    // poisoned the context for every pipeline in the process — fail-fast
    // rather than serve garbage. Recoverable faults fall through to false.
    turbo_ocr::abort_on_sticky_cuda_fault("TrtEngine::infer_dynamic enqueueV3");
    return false;
  }
  return true;
}

void TrtEngine::select_profile(int profile_idx, cudaStream_t stream) {
  if (profile_idx == current_profile_)
    return;

  if (profile_idx < 0 || profile_idx >= engine_->getNbOptimizationProfiles()) {
    std::cerr << std::format("[TRT] Invalid profile index {} (engine has {})",
                             profile_idx, engine_->getNbOptimizationProfiles())
              << '\n';
    return;
  }

  context_->setOptimizationProfileAsync(profile_idx, stream);
  current_profile_ = profile_idx;
  // Invalidate cached dims — new profile requires setInputShape again
  last_input_dims_ = {};

  // Conservatively re-bind ALL tensor addresses after a profile switch. NVIDIA
  // does not document whether addresses survive one (shapes are profile-scoped
  // and must be re-set; addresses appear context-level), and a stale address on
  // a multi-input model would mean garbage reads or faults with no error from
  // allInputDimensionsSpecified (which guards SHAPES, not addresses). Re-binding
  // is a few setTensorAddress calls — cheap insurance against undocumented
  // behavior changing across TRT versions.
  if (bound_input_) context_->setTensorAddress(input_name_.c_str(), bound_input_);
  if (bound_output_) context_->setTensorAddress(output_name_.c_str(), bound_output_);
  for (const auto &[name, ptr] : extra_addrs_)
    if (ptr) context_->setTensorAddress(name.c_str(), ptr);
}

int TrtEngine::num_profiles() const noexcept {
  return engine_ ? engine_->getNbOptimizationProfiles() : 1;
}

nvinfer1::Dims TrtEngine::get_output_dims() const noexcept {
  if (!context_) [[unlikely]]
    return {};
  return context_->getTensorShape(output_name_.c_str());
}

void TrtEngine::set_tensor_address(const std::string &name, void *ptr) {
  if (!context_) [[unlikely]]
    return;
  context_->setTensorAddress(name.c_str(), ptr);
  extra_addrs_[name] = ptr; // remembered so select_profile() can restore it
}

bool TrtEngine::set_input_shape(const std::string &name,
                                const nvinfer1::Dims &dims) {
  if (!context_) [[unlikely]]
    return false;
  if (!context_->setInputShape(name.c_str(), dims)) [[unlikely]] {
    std::cerr << std::format("[TRT] setInputShape FAILED for input={} on {}",
                             name, model_path_) << '\n';
    return false;
  }
  return true;
}

bool TrtEngine::execute(cudaStream_t stream) {
  if (!context_) [[unlikely]]
    return false;
  // PP-DocLayoutV3 (and any future multi-input model) has more than one
  // dynamic-shape input. If a caller forgets to set one, enqueueV3 silently
  // produces garbage output with no error, so guard the dispatch.
  if (!context_->allInputDimensionsSpecified()) [[unlikely]]
    throw turbo_ocr::InferenceError(
        "input dimensions not all set before enqueueV3 (" + model_path_ + ")");
  if (!context_->enqueueV3(stream)) [[unlikely]] {
    // A sticky fault here (illegal address, ECC, launch failure, …) has
    // poisoned the context for every pipeline in the process — fail-fast
    // rather than serve garbage. Recoverable faults fall through to false.
    turbo_ocr::abort_on_sticky_cuda_fault("TrtEngine::execute enqueueV3");
    return false;
  }
  return true;
}

nvinfer1::Dims TrtEngine::tensor_shape(const std::string &name) const {
  if (!context_) [[unlikely]]
    return {};
  return context_->getTensorShape(name.c_str());
}

void TrtEngine::probe_output_dims(const nvinfer1::Dims &input_dims,
                                   int &out_seq_len, int &out_num_classes) {
  if (!context_) [[unlikely]]
    return;
  if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
    std::cerr << "[TRT] probe_output_dims: setInputShape failed\n";
    return;
  }
  nvinfer1::Dims od = context_->getTensorShape(output_name_.c_str());
  if (od.nbDims >= 3) {
    out_seq_len = od.d[1];
    out_num_classes = od.d[2];
  }
  // Invalidate cached dims so the next infer_dynamic() call will re-set the
  // actual inference shape (probe may have used a different shape).
  last_input_dims_ = {};
}
