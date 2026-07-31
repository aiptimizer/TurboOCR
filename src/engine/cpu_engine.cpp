#include "turbo_ocr/engine/cpu_engine.h"

#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

// CoreML support on macOS (Apple Neural Engine + GPU acceleration)
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

using namespace turbo_ocr::engine;

Ort::Env &CpuEngine::process_env() {
  // ORT keeps one Env per process and fixes its threadpool config at first
  // creation. We therefore create exactly one Env here, lazily, and decide
  // global-vs-per-session threadpools up front from ORT_SHARED_POOL. (An
  // eagerly constructed plain per-engine Env would win the singleton, and a
  // later DisablePerSessionThreads() session would then abort with "env must be
  // created with CreateEnvWithGlobalThreadPools".)
  static Ort::Env env = [] {
    bool shared = false;
    if (const char *e = std::getenv("ORT_SHARED_POOL"))
      shared = (std::strcmp(e, "1") == 0);
    if (!shared)
      return Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CpuEngine");

    // Shared global intra-op threadpool sized to ORT_GLOBAL_THREADS (default
    // hardware_concurrency). Avoids the oversubscription of N sessions × M
    // concurrent engines each spinning up their own pool on a fixed core count.
    int n = static_cast<int>(std::thread::hardware_concurrency());
    if (const char *e = std::getenv("ORT_GLOBAL_THREADS")) {
      int v = std::atoi(e);
      if (v > 0)
        n = v;
    }
    if (n <= 0)
      n = 1;
    Ort::ThreadingOptions topt;
    topt.SetGlobalIntraOpNumThreads(n);
    topt.SetGlobalInterOpNumThreads(1);
    topt.SetGlobalDenormalAsZero();
    std::cout << std::format("[CpuEngine] Shared ORT threadpool: {} threads", n)
              << '\n';
    return Ort::Env(topt, ORT_LOGGING_LEVEL_WARNING, "CpuEngine");
  }();
  return env;
}

CpuEngine::CpuEngine(const std::string &model_path) : model_path_(model_path) {
  if (const char *env = std::getenv("ORT_EP"))
    ort_ep_ = env;

  if (const char *env = std::getenv("ORT_SHARED_POOL"))
    use_shared_pool_ = (std::strcmp(env, "1") == 0);
  // XNNPACK and OpenVINO run ops on their own intra-op threadpools; the shared
  // global pool / DisablePerSessionThreads path conflicts with them, so force it
  // off for those EPs.
  if (ort_ep_ == "xnnpack" || ort_ep_ == "openvino" || ort_ep_ == "openvino_cpu" ||
      ort_ep_ == "openvino_gpu")
    use_shared_pool_ = false;

  // CPU optimizations — balance threads per inference vs concurrency
  if (const char *env = std::getenv("ORT_NUM_THREADS"))
    session_options_.SetIntraOpNumThreads(std::atoi(env));
  else
    session_options_.SetIntraOpNumThreads(4);
  session_options_.SetInterOpNumThreads(1);
  // v6 rec FP16 produces wrong output under ORT_ENABLE_ALL on ORT 1.26
  // (SimplifiedLayerNormFusion); cap rec at EXTENDED. ORT_REC_OPT_CAP=0 lets a
  // fixed ORT restore ALL without a recompile. GPU/TRT path is unaffected.
  bool rec_opt_cap = true;
  if (const char* env = std::getenv("ORT_REC_OPT_CAP")) rec_opt_cap = (std::strcmp(env, "0") != 0);
  // Classify by filename prefix, not a whole-path substring: a path like
  // /correct/det.onnx or /models/recent/det.onnx must NOT be mistaken for rec.
  const std::string base = model_path_.substr(model_path_.find_last_of('/') + 1);
  const bool is_rec = base.rfind("rec", 0) == 0;
  session_options_.SetGraphOptimizationLevel((rec_opt_cap && is_rec)
                                                 ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED
                                                 : GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.EnableCpuMemArena();
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  // Flush denormal activations to zero — correctness-neutral, avoids the
  // microcode penalty if any near-zero values appear mid-graph.
  session_options_.AddConfigEntry("session.set_denormal_as_zero", "1");

  // Draw threads from the shared global pool instead of a per-session one.
  // Requires an Env built with global threadpools (see process_env()).
  if (use_shared_pool_)
    session_options_.DisablePerSessionThreads();

  // On macOS: use CoreML for Neural Engine + GPU acceleration
  // Supported ops run on ANE/GPU, unsupported fall back to CPU automatically
#ifdef __APPLE__
  bool use_coreml = true;
  if (const char *env = std::getenv("DISABLE_COREML"))
    use_coreml = (std::strcmp(env, "1") != 0);
  if (use_coreml) {
    // COREML_FLAG_USE_CPU_AND_GPU = 0x020 routes to GPU + Neural Engine
    // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004 requires ANE
    uint32_t coreml_flags = 0;
    if (const char *env = std::getenv("COREML_FLAGS"))
      coreml_flags = std::strtoul(env, nullptr, 0);
    else
      coreml_flags = 0x020; // CPU + GPU (includes Neural Engine when available)
    OrtSessionOptionsAppendExecutionProvider_CoreML(session_options_, coreml_flags);
    std::cout << "[CpuEngine] CoreML enabled (flags=0x" << std::hex
              << coreml_flags << std::dec << ")\n";
  }
#endif
}

void CpuEngine::apply_execution_provider() {
  // Applied to every model (det/rec/cls). Unset/"cpu" keeps the default MLAS CPU
  // EP; for other EPs unsupported ops still fall back to the CPU EP. All calls
  // route through the OrtApi (GetApi()) so they link regardless of which
  // providers a given onnxruntime build ships; an unavailable provider throws
  // here and is turned into a clean load() failure by the caller.
  if (ort_ep_.empty() || ort_ep_ == "cpu")
    return;

  if (ort_ep_ == "xnnpack") {
    // XNNPACK manages its own threads; size from ORT_NUM_THREADS or all cores.
    int n = static_cast<int>(std::thread::hardware_concurrency());
    if (const char *env = std::getenv("ORT_NUM_THREADS")) {
      int v = std::atoi(env);
      if (v > 0)
        n = v;
    }
    if (n <= 0)
      n = 1;
    session_options_.AppendExecutionProvider(
        "XNNPACK", {{"intra_op_num_threads", std::to_string(n)}});
    std::cout << std::format("[CpuEngine] EP=XNNPACK (intra_op_num_threads={})", n)
              << '\n';
  } else if (ort_ep_ == "dnnl") {
    // OrtDnnlProviderOptions is opaque in this build, so build it through the
    // API (defaults include use_arena=1) rather than the legacy free function,
    // which isn't an exported symbol here.
    OrtDnnlProviderOptions *dnnl_opts = nullptr;
    Ort::ThrowOnError(Ort::GetApi().CreateDnnlProviderOptions(&dnnl_opts));
    try {
      session_options_.AppendExecutionProvider_Dnnl(*dnnl_opts);
    } catch (...) {
      Ort::GetApi().ReleaseDnnlProviderOptions(dnnl_opts);
      throw;
    }
    Ort::GetApi().ReleaseDnnlProviderOptions(dnnl_opts);
    std::cout << "[CpuEngine] EP=oneDNN\n";
  } else if (ort_ep_ == "openvino" || ort_ep_ == "openvino_cpu" ||
             ort_ep_ == "openvino_gpu") {
    // OpenVINO EP: CPU inference through the OpenVINO runtime. For machines
    // with Intel integrated graphics, set ORT_EP=openvino_gpu to use the GPU
    // device instead. The default num_of_threads is 0 (use all cores); we honor
    // ORT_NUM_THREADS when set to keep behavior consistent with other EPs.
    int n = 0; // 0 = OpenVINO picks all cores
    if (const char *env = std::getenv("ORT_NUM_THREADS")) {
      int v = std::atoi(env);
      if (v > 0)
        n = v;
    }
    const std::string device_type = (ort_ep_ == "openvino_gpu") ? "GPU" : "CPU";
    const std::string device_id = "0";
    session_options_.AppendExecutionProvider(
        "OpenVINO",
        {{"device_type", device_type},
         {"device_id", device_id},
         {"num_of_threads", std::to_string(n)}});
    std::cout << std::format("[CpuEngine] EP=OpenVINO (device={}, threads={})",
                             device_type, n)
              << '\n';
  } else {
    throw std::runtime_error(
        std::format("[CpuEngine] Unknown ORT_EP='{}'", ort_ep_));
  }
}

bool CpuEngine::load() {
  try {
    apply_execution_provider();
    session_ = std::make_unique<Ort::Session>(
        process_env(), model_path_.c_str(), session_options_);
  } catch (const Ort::Exception &e) {
    std::cerr << std::format("[CpuEngine] Failed to load ONNX model: {} - {}", model_path_, e.what()) << '\n';
    if (!ort_ep_.empty() && ort_ep_ != "cpu")
      std::cerr << std::format("[CpuEngine] ORT_EP='{}' likely unavailable in this "
                               "onnxruntime build (provider not compiled in / "
                               "provider shared library missing)", ort_ep_) << '\n';
    return false;
  } catch (const std::exception &e) {
    std::cerr << std::format("[CpuEngine] Failed to load ONNX model: {} - {}", model_path_, e.what()) << '\n';
    return false;
  }

  // Get input/output names
  auto input_name = session_->GetInputNameAllocated(0, allocator_);
  input_name_ = input_name.get();

  auto output_name = session_->GetOutputNameAllocated(0, allocator_);
  output_name_ = output_name.get();

  std::cout << std::format("[CpuEngine] Loaded: {} (input={}, output={})",
                          model_path_, input_name_, output_name_) << '\n';
  return true;
}

CpuEngine::InferResult
CpuEngine::infer(const float *input_data,
                 const std::vector<int64_t> &input_shape) {
  InferResult result;
  if (!session_)
    return result;

  int64_t input_count =
      std::accumulate(input_shape.begin(), input_shape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  // Cached MemoryInfo — avoid recreating every call
  static const auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(input_data), input_count,
      input_shape.data(), input_shape.size());

  const char *input_names[] = {input_name_.c_str()};
  const char *output_names[] = {output_name_.c_str()};

  auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &input_tensor, 1, output_names, 1);

  auto &output_tensor = output_tensors.front();
  auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
  result.shape = type_info.GetShape();

  int64_t output_count = type_info.GetElementCount();
  const float *output_data = output_tensor.GetTensorData<float>();
  result.data.assign(output_data, output_data + output_count);

  return result;
}

CpuEngine::InferResult
CpuEngine::infer_batch(const float *input_data,
                       const std::vector<int64_t> &input_shape) {
  // ORT runs an N-row {B,3,H,W} input in a single Run; the output is the full
  // {B,seq,classes} tensor. Mechanically identical to infer().
  return infer(input_data, input_shape);
}

CpuEngine::InferView
CpuEngine::infer_batch_view(const float *input_data,
                           const std::vector<int64_t> &input_shape) {
  InferView view;
  if (!session_)
    return view;

  int64_t input_count =
      std::accumulate(input_shape.begin(), input_shape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  static const auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(input_data), input_count,
      input_shape.data(), input_shape.size());

  const char *input_names[] = {input_name_.c_str()};
  const char *output_names[] = {output_name_.c_str()};

  auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &input_tensor, 1, output_names, 1);

  // Retain the OrtValue in the engine so its buffer stays alive; the returned
  // view points directly into it (no copy out).
  last_output_ = std::move(output_tensors.front());
  auto type_info = last_output_.GetTensorTypeAndShapeInfo();
  view.shape = type_info.GetShape();
  view.data = last_output_.GetTensorData<float>();
  return view;
}

void CpuEngine::probe_output_dims(const std::vector<int64_t> &input_shape,
                                   int &out_dim1, int &out_dim2) {
  if (!session_)
    return;

  int64_t input_count =
      std::accumulate(input_shape.begin(), input_shape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  std::vector<float> dummy(input_count, 0.0f);
  auto result = infer(dummy.data(), input_shape);

  if (result.shape.size() >= 3) {
    out_dim1 = static_cast<int>(result.shape[1]);
    out_dim2 = static_cast<int>(result.shape[2]);
  }
}
