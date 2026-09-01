#include "turbo_ocr/onnx/ort_engine.h"

#include "ep_options.h"                        // the pure EP-option policy
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/onnx/coreml_ep.h"        // the shared CoreML env policy
#include "turbo_ocr/onnx/host_ort_threads.h" // host_ort_intra_op_threads
#include "turbo_ocr/onnx/ort_path.h"         // ORTCHAR_T path (wchar_t on Windows)

#include <onnxruntime_session_options_config_keys.h> // kOrtSessionOptionsDisableCPUEPFallback

#include <algorithm>   // std::find over GetAvailableProviders()
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

// CoreML support on macOS (Apple Neural Engine + GPU acceleration)
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

// Vendor execution-provider factory headers are present only in the matching
// ORT build (a ROCm onnxruntime ships MIGraphX/ROCm; a Windows build ships DML).
// __has_include keeps the CPU/CoreML build (which has none of them) compiling —
// the ORT_EP branch then fails cleanly at load() instead of at link time.
//
// MIGraphX is the exception: ORT >= 1.27 removed its public factory header
// (only an INTERNAL one remains, uncompilable outside the ORT tree) and
// declares the C shim in onnxruntime_c_api.h — but only a ROCm build EXPORTS
// it, so presence is decided at the LINK level. CMake probes the resolved
// library for the symbol and passes TURBO_HAVE_MIGRAPHX; the header probe
// below remains for pre-1.27 ROCm ORTs, whose c_api.h lacks the declaration.
#if !defined(TURBO_HAVE_MIGRAPHX) && __has_include(<migraphx_provider_factory.h>)
#include <migraphx_provider_factory.h>
#define TURBO_HAVE_MIGRAPHX 1
#endif
#if __has_include(<rocm_provider_factory.h>)
#include <rocm_provider_factory.h>
#define TURBO_HAVE_ROCM 1
#endif
#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define TURBO_HAVE_DML 1
#endif
// Intel OpenVINO is appended through the header-free generic string API
// (session_options_.AppendExecutionProvider("OpenVINO", ...)) so it needs no
// provider header and compiles on every build.

using namespace turbo_ocr::engine;

// The steps that are genuinely common to more than one provider — the thread
// count, the device ordinal, the option-string merge — plus the whole OpenVINO
// option map now live in ep_options.h, next to this file. They are pure
// functions of their arguments and the environment, so a test TU can hold them;
// keeping them in this file's anonymous namespace made the decomposition
// unreachable from ctest, which is the one place this policy can be pinned
// without an Intel/AMD/NVIDIA box. Per-provider policy that needs a session
// stays in that provider's appender below, where its rationale is.
using turbo_ocr::engine::ep_options::device_id_for;
using turbo_ocr::engine::ep_options::env_thread_count;

// DISABLE_COREML=1 — the operator's off switch for the whole CoreML path — and
// COREML_FLAGS now live in turbo_ocr/onnx/coreml_ep.h, because this file was
// not the only one reading them: src/analysis/layout/ort_paddle_layout.cpp and
// src/analysis/forms/field_model.cpp had grown their own copies of both. The
// argument that put the predicate in one place here (configure_session_() asks
// whether to append the provider; the coreml arm of apply_execution_provider()
// asks again to say WHY nothing is attached, and two copies could disagree
// about which happened) applies across files exactly as it did within one.
using turbo_ocr::engine::coreml_disabled_by_env;
using turbo_ocr::engine::coreml_flags;

Ort::Env &OrtEngine::process_env() {
  // ORT keeps one Env per process and fixes its threadpool config at first
  // creation. We therefore create exactly one Env here, lazily, and decide
  // global-vs-per-session threadpools up front from ORT_SHARED_POOL. (An
  // eagerly constructed plain per-engine Env would win the singleton, and a
  // later DisablePerSessionThreads() session would then abort with "env must be
  // created with CreateEnvWithGlobalThreadPools".)
  // ORT_LOGGING_LEVEL_ERROR, not WARNING: at WARNING ONNX Runtime narrates
  // every session onto the host application's stderr — "CoreMLExecutionProvider
  // ::GetCapability, number of partitions supported by CoreML: ..." and
  // "VerifyEachNodeIsAssignedToAnEp". Both describe the NORMAL outcome (a
  // partially-offloaded graph) and neither is actionable, but a library cannot
  // hand that to its caller unasked. Genuine load failures are reported by this
  // file explicitly. Matches ort_paddle_layout / slanext / ppformulanet.
  static Ort::Env env = [] {
    if (!turbo_ocr::env::env_enabled("ORT_SHARED_POOL"))
      return Ort::Env(ORT_LOGGING_LEVEL_ERROR, "OrtEngine");

    // Shared global intra-op threadpool sized to ORT_GLOBAL_THREADS (default
    // hardware_concurrency). Avoids the oversubscription of N sessions × M
    // concurrent engines each spinning up their own pool on a fixed core count.
    const int n = env_thread_count("ORT_GLOBAL_THREADS");
    Ort::ThreadingOptions topt;
    topt.SetGlobalIntraOpNumThreads(n);
    topt.SetGlobalInterOpNumThreads(1);
    topt.SetGlobalDenormalAsZero();
    std::cout << std::format("[OrtEngine] Shared ORT threadpool: {} threads", n)
              << '\n';
    return Ort::Env(topt, ORT_LOGGING_LEVEL_ERROR, "OrtEngine");
  }();
  return env;
}

OrtEngine::OrtEngine(const std::string &model_path) : model_path_(model_path) {
  if (turbo_ocr::env::env_present("ORT_EP"))
    ort_ep_ = turbo_ocr::env::env_or("ORT_EP", "");
  // Mirror the env choice into ep_ so apply_execution_provider() has exactly
  // ONE code path. device/device_id stay UNSET here ("" / -1): the env ctor
  // keeps reading OPENVINO_DEVICE / CUDA_DEVICE_ID / ... per provider,
  // unchanged.
  ep_.provider = ort_ep_;
  // fp16 too. EpConfig defaults it to TRUE because a caller that builds one has
  // opted into the fast path; an env-only caller has expressed no opinion, and
  // honouring an unexpressed "yes" made the same field mean two things: this
  // ctor deliberately does NOT call resolve_fp16_model_(), so ep_.fp16 was
  // inert for CUDA/DML/MIGraphX (no *.fp16.onnx lookup) yet still stamped
  // precision=FP16 into the OpenVINO option map on a GPU/NPU device.
  //
  // OpenVINO is therefore the ONE provider where this OFF is observable, and
  // the exact shape of it is: ORT_EP=openvino with a GPU*/NPU* device and no
  // OPENVINO_PRECISION leaves `precision` UNSET, because
  // ep_options::openvino_options() only derives that key from ep.fp16. UNSET
  // means "whatever the plugin picks", and the ASSUMPTION here — untested on
  // this tree, no Intel GPU/NPU to test it on — is that the OpenVINO GPU plugin
  // already infers in f16 unless told otherwise, so precision=FP16 would have
  // been a no-op. NPU is not covered by even that assumption. Needs a
  // measurement on real Intel silicon (device=GPU and device=NPU, with and
  // without OPENVINO_PRECISION=FP16); until then this stays as it is, since
  // guessing "yes" here is what conflated the two meanings in the first place.
  // An operator who wants it explicit already has OPENVINO_PRECISION, which
  // openvino_options() passes through untouched.
  ep_.fp16 = false;
  configure_session_();
}

OrtEngine::OrtEngine(const std::string &model_path, const backend::EpConfig &ep)
    : model_path_(model_path), ep_(ep), ep_explicit_(true) {
  // An EMPTY provider means "caller has no opinion", NOT "force the default
  // CPU provider" — so ORT_EP still decides, exactly as it always has. Getting
  // this backwards silently disabled ORT_EP for the whole cpu backend the
  // moment it started passing an EpConfig, which would have broken every
  // existing ORT_EP=cuda / =openvino / =dml deployment.
  if (ep.provider.empty()) {
    if (turbo_ocr::env::env_present("ORT_EP")) ort_ep_ = turbo_ocr::env::env_or("ORT_EP", "");
    ep_.provider = ort_ep_;
  } else if (!ep.is_default_cpu()) {
    ort_ep_ = ep.provider;
  }
  // fp16-by-model must be decided BEFORE the session is configured, because it
  // changes which file we are about to open.
  resolve_fp16_model_();
  configure_session_();
}

void OrtEngine::configure_session_() {
  // Force the process-wide Ort::Env into existence BEFORE any provider is
  // appended. ORT's CoreML factory calls HasNeuralEngine(), which reaches
  // LoggingManager::DefaultLogger() — and that THROWS when no Env has been
  // created yet. The throw lands in this constructor, which no caller can
  // catch (load() is where the try lives), so the process aborts with a bare
  // "uncaught OnnxRuntimeException". Creating the Env here is the fix and
  // costs nothing: process_env() is a function-local static that every
  // subsequent session shares.
  (void)process_env();

  use_shared_pool_ = turbo_ocr::env::env_enabled("ORT_SHARED_POOL");
  // XNNPACK runs ops on its own intra-op threadpool; the shared global pool /
  // DisablePerSessionThreads path conflicts with it, so force it off for xnnpack.
  if (ort_ep_ == "xnnpack")
    use_shared_pool_ = false;

  // CPU optimizations — balance threads per inference vs concurrency.
  // The 4 is this stage's historical cap and still applies unless the operator
  // (ORT_NUM_THREADS) or the backend (a host-idle hint) says otherwise; see
  // common/host_ort_threads.h for why that decision is shared rather than
  // re-made here.
  session_options_.SetIntraOpNumThreads(host_ort_intra_op_threads(4));
  session_options_.SetInterOpNumThreads(1);
  // v6 rec FP16 produces wrong output under ORT_ENABLE_ALL on ORT 1.26
  // (SimplifiedLayerNormFusion); cap rec at EXTENDED. ORT_REC_OPT_CAP=0 lets a
  // fixed ORT restore ALL without a recompile. GPU/TRT path is unaffected.
  const bool rec_opt_cap = turbo_ocr::env::env_or("ORT_REC_OPT_CAP", "1") != "0";
  // Classify by filename prefix, not a whole-path substring: a path like
  // /correct/det.onnx or /models/recent/det.onnx must NOT be mistaken for rec.
  const std::string base = model_path_.substr(model_path_.find_last_of('/') + 1);
  const bool is_rec = base.rfind("rec", 0) == 0;
  // OpenVINO EP: ORT's OWN fusions must be OFF.
  //
  // ORT rewrites the graph before it hands subgraphs to an EP, and the
  // OpenVINO EP then re-optimizes internally. Running both is not merely
  // redundant — the fused ops partition differently per OpenVINO device, and
  // on the iGPU that MEASURABLY destroys accuracy: FUNSD-50 F1 collapsed to
  // 66.7% (recall 58.6%, i.e. detection losing boxes) against 85.8% on the
  // CPU device with the identical model and pipeline. Disabling ORT's pass
  // (the vendor's documented recommendation for this EP) is what makes the
  // GPU device agree with the CPU one.
  const bool ov_ep = (ort_ep_ == "openvino");
  session_options_.SetGraphOptimizationLevel(
      ov_ep ? GraphOptimizationLevel::ORT_DISABLE_ALL
            : ((rec_opt_cap && is_rec) ? GraphOptimizationLevel::ORT_ENABLE_EXTENDED
                                       : GraphOptimizationLevel::ORT_ENABLE_ALL));
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
  // Historically ON by default for every OrtEngine on macOS unless
  // DISABLE_COREML=1. That default is preserved for the env ctor. With an
  // EXPLICIT EpConfig the caller decides: CoreML is attached only when it
  // actually asked for provider "coreml", so `--backend cpu` on a Mac cannot
  // silently become a CoreML run just because the seam picked the cpu backend.
  bool use_coreml = ep_explicit_ ? (ep_.provider == "coreml") : true;
  if (coreml_disabled_by_env())
    use_coreml = false;
  if (use_coreml) {
    // COREML_FLAG_USE_CPU_AND_GPU = 0x020 routes to GPU + Neural Engine
    // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004 requires ANE
    const uint32_t flags = coreml_flags();
    // The append RETURNS a status (ORT_API_STATUS) and it is not decorative:
    // discarding it printed "CoreML enabled" for a session that had no CoreML
    // on it, and leaked the OrtStatus. It must not THROW from here, though —
    // this runs from a constructor, and the process_env() comment above
    // documents exactly that hazard (load() is where the try lives). So: own
    // the status, say what happened, and record the OUTCOME rather than the
    // intent. apply_execution_provider() reads coreml_attached_ and refuses to
    // let an explicit coreml request pass for a CoreML run when it is false —
    // naming THIS failure, as distinct from a DISABLE_COREML=1 suppression,
    // because only one of the two is a provider problem worth chasing.
    if (OrtStatus *st = OrtSessionOptionsAppendExecutionProvider_CoreML(
            session_options_, flags)) {
      const Ort::Status owned{st}; // takes ownership; releases on scope exit
      std::cerr << "[OrtEngine] CoreML unavailable: " << owned.GetErrorMessage()
                << " — continuing on the default CPU provider\n";
    } else {
      coreml_attached_ = true;
      std::cout << "[OrtEngine] CoreML enabled (flags=0x" << std::hex
                << flags << std::dec << ")\n";
    }
  }
#endif
}

void OrtEngine::resolve_fp16_model_() {
  // Only providers whose fp16 lives in the MODEL (CUDA / DirectML / MIGraphX)
  // need a different file. OpenVINO has a precision knob and CoreML is already
  // fp16 on ANE/GPU, so for those the fp32 .onnx on disk is the right input.
  if (!ep_.fp16) return;
  if (backend::fp16_support_for(ep_.provider) != backend::Fp16Support::Model)
    return;

  // A sibling `<stem>.fp16.onnx`, produced offline by
  // scripts/models/onnx/make_fp16_models.py. Deliberately NOT generated here: converting
  // weights at load time would make the first request pay for it and would
  // write into the models tree from a server process. Absent => stay fp32 and
  // SAY SO, because a silent fp32 run is exactly how "we enabled fp16" becomes
  // a measurement that never moved.
  const auto dot = model_path_.find_last_of('.');
  if (dot == std::string::npos) return;
  const std::string cand = model_path_.substr(0, dot) + ".fp16" + model_path_.substr(dot);
  if (std::filesystem::exists(cand)) {
    std::cout << std::format("[OrtEngine] fp16 model: {}", cand) << '\n';
    model_path_ = cand;
  } else {
    std::cout << std::format(
                     "[OrtEngine] fp16 requested for EP='{}' but no {} — running "
                     "fp32 (generate one with scripts/models/onnx/make_fp16_models.py)",
                     ep_.provider, cand)
              << '\n';
  }
}

namespace {

// ---------------------------------------------------------------------------
// Per-provider appenders
//
// ONE function per execution provider, each owning its own env parsing and its
// own hard-won rationale, dispatched from OrtEngine::apply_execution_provider()
// below. The ladder these came from was a single ~200-line function in which
// every added vendor made the previous ones harder to audit — and this file has
// the regression history to show for it (the OpenVINO device default, the
// device-scoped FP16 knob, the CoreML early return). Keeping each vendor's
// decisions inside its own function is what makes a change to one provider
// unable to perturb another.
//
// They take the session options by reference plus, where relevant, the resolved
// EpConfig; none of them read OrtEngine state directly (require_coreml_attached
// is handed the attach OUTCOME, plus the DISABLE_COREML predicate that tells it
// which of the two ways to be un-attached happened). Values, env var names, log
// lines and throw texts are those of the ladder this replaced, except where a
// comment below says otherwise and says why.
// ---------------------------------------------------------------------------

// CoreML (Apple ANE/GPU) — a CHECK, not an append, hence the name: every other
// entry in this family appends a provider, and calling this one apply_*_ep()
// made the dispatch ladder below read as eight appenders when it is seven.
//
// CoreML is the ONE provider whose session option must be set before anything
// else touches session_options_, so it is attached in configure_session_() and
// NOT in the provider ladder. Reaching this function means the caller asked for
// coreml explicitly (EpConfig or ORT_EP), so this is where that request is
// reconciled with what actually happened — succeeding here is also what keeps
// the request from falling through to the "Unknown ORT_EP" throw.
void require_coreml_attached([[maybe_unused]] bool attached,
                             [[maybe_unused]] bool disabled_by_env) {
#ifndef __APPLE__
  throw std::runtime_error(
      "[OrtEngine] ORT_EP=coreml is macOS-only (this build is not Apple)");
#else
  // Explicitly requesting coreml AND setting DISABLE_COREML=1 is a
  // contradiction, and the silent resolution — run plain MLAS while the backend
  // keeps reporting the apple/onnx path — is the one outcome that must not be
  // available (cpu_stages.cpp says the same thing about TURBO_EP_PROVIDER:
  // "rather than silently downgrading, which would report Intel acceleration
  // while running MLAS").
  //
  // WARN, not throw: DISABLE_COREML=1 is exported for the WHOLE ctest suite
  // (CMakeLists.txt, "CoreML EP is non-deterministic across macOS versions —
  // fatal for an exact-match accuracy gate"), so a throw here would turn a
  // deliberate harness setting into a boot failure for any apple/onnx run under
  // ctest. The narrower knob (DISABLE_COREML) therefore still wins over the
  // broader one (the provider request) — it just no longer wins in silence.
  //
  // TWO messages, not one "either/or". A suppression is the operator getting
  // exactly what the narrower knob promised; a failed append is a provider
  // fault that ORT has already explained on its own line. Reporting both as
  // "DISABLE_COREML=1 is set, or the append failed" made a real CoreML
  // regression indistinguishable from the ctest harness doing its job, and sent
  // whoever read it to check an env var that was never set.
  if (attached) return;
  if (disabled_by_env)
    std::cerr << "[OrtEngine] WARNING: provider 'coreml' was requested "
                 "(ORT_EP/EpConfig) but DISABLE_COREML=1 SUPPRESSED it. This "
                 "session runs plain MLAS; any 'coreml' in the backend/"
                 "engine-mode banner is not what is executing.\n";
  else
    std::cerr << "[OrtEngine] WARNING: provider 'coreml' was requested "
                 "(ORT_EP/EpConfig) and DISABLE_COREML is not set, so the "
                 "CoreML append FAILED — see the '[OrtEngine] CoreML "
                 "unavailable' line above for ORT's own error. This session "
                 "runs plain MLAS; any 'coreml' in the backend/engine-mode "
                 "banner is not what is executing.\n";
#endif
}

void apply_xnnpack_ep(Ort::SessionOptions &so) {
  // XNNPACK manages its own threads; size from ORT_NUM_THREADS or all cores.
  const int n = env_thread_count("ORT_NUM_THREADS");
  so.AppendExecutionProvider("XNNPACK",
                             {{"intra_op_num_threads", std::to_string(n)}});
  std::cout << std::format("[OrtEngine] EP=XNNPACK (intra_op_num_threads={})", n)
            << '\n';
}

void apply_dnnl_ep(Ort::SessionOptions &so) {
  // OrtDnnlProviderOptions is opaque in this build, so build it through the
  // API (defaults include use_arena=1) rather than the legacy free function,
  // which isn't an exported symbol here.
  OrtDnnlProviderOptions *dnnl_opts = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateDnnlProviderOptions(&dnnl_opts));
  try {
    so.AppendExecutionProvider_Dnnl(*dnnl_opts);
  } catch (...) {
    Ort::GetApi().ReleaseDnnlProviderOptions(dnnl_opts);
    throw;
  }
  Ort::GetApi().ReleaseDnnlProviderOptions(dnnl_opts);
  std::cout << "[OrtEngine] EP=oneDNN\n";
}

void apply_openvino_ep(Ort::SessionOptions &so,
                       const turbo_ocr::backend::EpConfig &ep) {
  // Intel CPU / integrated GPU / Arc dGPU / NPU via the OpenVINO EP. Uses the
  // header-free generic provider API, so it compiles everywhere and fails
  // cleanly at load() if this ORT build lacks OpenVINO (use the
  // onnxruntime-openvino build). OPENVINO_DEVICE selects the target
  // (AUTO|CPU|GPU|NPU|GPU.0|...); OPENVINO_PRECISION (FP16/FP32) and
  // OPENVINO_CACHE_DIR (persist the compiled blob) are optional.
  //
  // The option map itself — device default, the device-scoped FP16 rule, and
  // the OPENVINO_EP_OPTS merge — is ep_options::openvino_options(), where it is
  // pure and testable. Both diagnostics below read the FINAL map, so the
  // warning and the log line can no longer disagree about which device was
  // selected (they could when the escape hatch was merged after them).
  auto ov = ep_options::openvino_options(ep);
  const std::string &dev = ov["device_type"];

  if (dev.rfind("CPU", 0) != 0) {
    std::cerr << std::format(
                     "[OrtEngine] WARNING: OpenVINO device '{}' — text "
                     "DETECTION accuracy is known to degrade badly on the "
                     "OpenVINO GPU plugin with these models (measured F1 "
                     "62-67% vs 85.8% on CPU; upstream openvino#29364/#28897, "
                     "unresolved). Use device CPU unless you have verified "
                     "accuracy on YOUR models.",
                     dev)
              << '\n';
  }
  // An operator-SET precision (OPENVINO_PRECISION / OPENVINO_EP_OPTS) is passed
  // through as asked — but FP16 on a device that cannot take it does not
  // degrade quietly, it fails EP load outright ("CPU only supports FP32,
  // ACCURACY") and surfaces as "openvino unavailable", which reads like a
  // missing provider rather than a bad option. Name it instead.
  if (const auto it = ov.find("precision");
      it != ov.end() && it->second == "FP16" &&
      !ep_options::device_takes_fp16(dev)) {
    std::cerr << std::format(
                     "[OrtEngine] WARNING: OpenVINO precision=FP16 requested on "
                     "device '{}' — only GPU*/NPU* accept FP16; the EP will "
                     "most likely fail to load and be reported as 'openvino "
                     "unavailable'. Drop OPENVINO_PRECISION (or the precision "
                     "key in OPENVINO_EP_OPTS) for this device.",
                     dev)
              << '\n';
  }

  so.AppendExecutionProvider("OpenVINO", ov);
  std::cout << std::format("[OrtEngine] EP=OpenVINO (device={})", dev) << '\n';
}

void apply_migraphx_ep([[maybe_unused]] Ort::SessionOptions &so,
                       [[maybe_unused]] const turbo_ocr::backend::EpConfig &ep) {
  // AMD Instinct (CDNA) + supported Radeon (RDNA) on Linux/ROCm. MIGraphX is
  // AMD's go-forward EP; the legacy ROCm EP was removed at ORT 1.23. Needs a
  // ROCm onnxruntime build (compiled per gfx target). ROCM_DEVICE_ID picks
  // the HIP device.
#ifdef TURBO_HAVE_MIGRAPHX
  const int dev = device_id_for(ep, "ROCM_DEVICE_ID");
  Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_MIGraphX(so, dev));
  std::cout << std::format("[OrtEngine] EP=MIGraphX (device={})", dev) << '\n';
#else
  throw std::runtime_error(
      "[OrtEngine] ORT_EP=migraphx but this onnxruntime has no MIGraphX EP "
      "(build/link a ROCm onnxruntime; MIGraphX programs are per-gfx)");
#endif
}

void apply_rocm_ep([[maybe_unused]] Ort::SessionOptions &so,
                   [[maybe_unused]] const turbo_ocr::backend::EpConfig &ep) {
  // Legacy AMD ROCm EP (removed upstream at ORT 1.23 in favor of MIGraphX);
  // kept for older ROCm ORT builds that still ship it.
#ifdef TURBO_HAVE_ROCM
  const int dev = device_id_for(ep, "ROCM_DEVICE_ID");
  Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_ROCM(so, dev));
  std::cout << std::format("[OrtEngine] EP=ROCm (device={})", dev) << '\n';
#else
  throw std::runtime_error(
      "[OrtEngine] ORT_EP=rocm but this onnxruntime has no ROCm EP "
      "(removed at ORT 1.23 — use ORT_EP=migraphx on a ROCm build)");
#endif
}

void apply_dml_ep([[maybe_unused]] Ort::SessionOptions &so,
                  [[maybe_unused]] const turbo_ocr::backend::EpConfig &ep) {
  // DirectML: vendor-agnostic D3D12 path (AMD/Intel/NVIDIA consumer on
  // Windows) — the realistic consumer fallback where ROCm/OpenVINO GPU is
  // absent. DML requires memory-pattern off and sequential execution.
#ifdef TURBO_HAVE_DML
  so.DisableMemPattern();
  so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  const int dev = device_id_for(ep, "DML_DEVICE_ID");
  Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_DML(so, dev));
  std::cout << std::format("[OrtEngine] EP=DirectML (device={})", dev) << '\n';
#else
  throw std::runtime_error(
      "[OrtEngine] ORT_EP=dml but this onnxruntime has no DirectML EP "
      "(Windows-only onnxruntime-directml build)");
#endif
}

void apply_cuda_ep(Ort::SessionOptions &so,
                   const turbo_ocr::backend::EpConfig &ep) {
  // NVIDIA CUDA EP: runs the ONNX graph on the GPU directly, with NO
  // TensorRT engine build (the fast-setup NVIDIA default). CUDA_DEVICE_ID
  // picks the device. Peak throughput via the native TensorRT path is a
  // separate Tier-A pipeline, not this EP.
  //
  // No #ifdef guard: AppendExecutionProvider_CUDA goes through the core OrtApi
  // vtable (unlike the migraphx/rocm/dml free functions that need vendor
  // headers), so it always links. On an onnxruntime WITHOUT CUDA it throws
  // Ort::Exception here, which OrtEngine::load() catches and turns into a
  // clean load() failure — same as the OpenVINO branch. Runtime provider
  // availability (Ort::GetAvailableProviders, surfaced to Python via
  // build_info) is the single source of truth; there is no separate compile
  // flag to drift from it. The Python layer pre-rejects on a non-CUDA build,
  // so this is normally only reached on a real CUDA onnxruntime.
  OrtCUDAProviderOptions cuda_opts{};
  cuda_opts.device_id = device_id_for(ep, "CUDA_DEVICE_ID");
  so.AppendExecutionProvider_CUDA(cuda_opts);
  std::cout << std::format("[OrtEngine] EP=CUDA (device={})", cuda_opts.device_id)
            << '\n';
}

} // namespace

void OrtEngine::apply_execution_provider() {
  // Applied to every model (det/rec/cls). Unset/"cpu" keeps the default MLAS CPU
  // EP; for other EPs unsupported ops still fall back to the CPU EP.
  //
  // LINKAGE IS NOT UNIFORM, despite what this comment used to claim (it was
  // true when the ladder held only xnnpack and dnnl). xnnpack/dnnl/openvino/
  // cuda link unconditionally — they go through the generic string API or the
  // core OrtApi vtable, so they are present in every build and an unavailable
  // provider merely throws at append time. migraphx/rocm/dml are free functions
  // that need vendor factory headers, so they are compiled in only when
  // __has_include found those headers (the TURBO_HAVE_* guards at the top of
  // this file); otherwise their appender throws a build-specific error saying
  // which onnxruntime you need. coreml is a vendor free function too, from a
  // header this file includes only under __APPLE__ — so its arm is a macOS-only
  // CHECK that throws outright off Apple, and on Apple it appends nothing here
  // (configure_session_() already did, before anything else touched
  // session_options_). Either way the throw lands in load()'s own try (below)
  // and becomes a clean load() failure — apply_execution_provider() is not
  // called from anywhere else.
  //
  // DISPATCH ONLY. Every provider's options, env vars and rationale live in its
  // appender above; the string this switches on (ort_ep_) is already the single
  // resolved provider name for BOTH constructors — the env ctor mirrors ORT_EP
  // into ep_, and the EpConfig ctor falls back to ORT_EP when the caller gave no
  // opinion — so there is exactly one selection point, here.
  if (ort_ep_.empty() || ort_ep_ == "cpu")
    return;

  // A GPU request must never be served by the CPU.
  //
  // THE GUARANTEE is ORT's own: kOrtSessionOptionsDisableCPUEPFallback makes
  // node assignment to the CPU EP an ERROR instead of a silent degradation, so
  // a provider that appends cleanly and then cannot claim the graph fails at
  // session construction. That is the case no external check can catch —
  // measured: an onnxruntime-gpu wheel built for CUDA 12 on a CUDA 13 host
  // reports CUDAExecutionProvider as available, accepts the append, and then
  // runs every node on the CPU. load() used to return true for that.
  //
  // Set BEFORE the provider is appended: it is a session-options config entry,
  // read when the session is built.
  //
  // WHICH PROVIDERS THIS IS SAFE FOR — an allowlist, not a denylist.
  //
  // The flag fails session construction if ANY node lands on the CPU EP. Whether
  // that is the right bar depends entirely on how a provider partitions, and the
  // providers here divide three ways:
  //
  //   cuda            VERIFIED on hardware (RTX 5090 / CUDA 13.3, and the ORT
  //                   CUDA EP on Windows). Claims these graphs whole, and the
  //                   flag is what turns a CUDA-12 wheel on a CUDA-13 host from
  //                   a silent all-CPU run into a load failure.
  //   coreml          PROVEN WRONG. CoreML partitions by design — ORT logs
  //                   "partitions supported by CoreML: 30 ... nodes supported:
  //                   46" of 76 as NORMAL — so the flag failed every session and
  //                   the server would not start on any machine without exported
  //                   MPSGraph artefacts, which is every fresh clone.
  //   openvino,       Partition the same way CoreML does, and NONE of them has
  //   tensorrt,       been run with this flag on real hardware from this tree.
  //   rocm, migraphx, Defaulting it on for them would be shipping the CoreML
  //   dml             breakage again to Intel and AMD users, sight unseen.
  //   xnnpack, dnnl   CPU-side providers; "must not touch the CPU EP" is a
  //                   contradiction in terms.
  //
  // So: on where it is proven, off where it is not, and never silent about which
  // — an operator who has tested their own combination turns it on with
  // TURBO_STRICT_EP=1, and one hitting an unexpected refusal turns it off with 0.
  const std::string strict = env::env_or("TURBO_STRICT_EP", "auto");
  const bool verified_ep = (ort_ep_ == "cuda");
  const bool strict_on = (strict == "1")   ? true
                         : (strict == "0") ? false
                                           : verified_ep;
  if (strict_on) {
    session_options_.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  } else if (strict == "auto" && !verified_ep) {
    // Say it once per session rather than leaving the weaker guarantee implicit:
    // the whole point of the flag is that silent CPU execution is the thing we
    // refuse to ship, so silently NOT enforcing it would be its own version of
    // the same problem.
    std::cout << "[OrtEngine] provider '" << ort_ep_
              << "' partitions its graph, so the CPU-fallback guard is OFF "
                 "(untested for this provider). Set TURBO_STRICT_EP=1 to "
                 "require a full-graph claim.\n";
  }

  // The availability test below is NOT the guarantee — it is error quality.
  // Without it, a provider missing from the build surfaces as ORT's generic
  // append failure; with it, the operator is told which provider was needed,
  // which ones this build has, and what to do about it.
  {
    static const std::unordered_map<std::string, std::string> kProviderName{
        {"cuda", "CUDAExecutionProvider"},
        {"tensorrt", "TensorrtExecutionProvider"},
        {"openvino", "OpenVINOExecutionProvider"},
        {"dml", "DmlExecutionProvider"},
        {"migraphx", "MIGraphXExecutionProvider"},
        {"rocm", "ROCMExecutionProvider"},
        {"coreml", "CoreMLExecutionProvider"},
        {"xnnpack", "XnnpackExecutionProvider"},
        {"dnnl", "DnnlExecutionProvider"},
    };
    if (auto it = kProviderName.find(ort_ep_); it != kProviderName.end()) {
      const auto have = Ort::GetAvailableProviders();
      if (std::find(have.begin(), have.end(), it->second) == have.end()) {
        std::string list;
        for (const auto &p : have) list += (list.empty() ? "" : ", ") + p;
        throw std::runtime_error(std::format(
            "[OrtEngine] ORT_EP='{}' requires {}, which this onnxruntime does "
            "not provide (available: {}). Refusing rather than running on the "
            "CPU under a GPU request — install the matching onnxruntime build, "
            "or set ORT_EP=cpu to ask for CPU explicitly.",
            ort_ep_, it->second, list));
      }
    }
    // "Did the provider actually claim the nodes" is answered by
    // DisableCPUEPFallback above — not by anything this code can inspect after
    // the fact, and not by the availability list, which reports a provider as
    // present in exactly the case that silently degrades.
  }

  if (ort_ep_ == "coreml")
    // Attached (or not) in configure_session_(); the second argument is what
    // lets the warning name WHICH kind of "not".
    require_coreml_attached(coreml_attached_, coreml_disabled_by_env());
  else if (ort_ep_ == "xnnpack")
    apply_xnnpack_ep(session_options_);
  else if (ort_ep_ == "dnnl")
    apply_dnnl_ep(session_options_);
  else if (ort_ep_ == "openvino")
    apply_openvino_ep(session_options_, ep_);
  else if (ort_ep_ == "migraphx")
    apply_migraphx_ep(session_options_, ep_);
  else if (ort_ep_ == "rocm")
    apply_rocm_ep(session_options_, ep_);
  else if (ort_ep_ == "dml")
    apply_dml_ep(session_options_, ep_);
  else if (ort_ep_ == "cuda")
    apply_cuda_ep(session_options_, ep_);
  else
    throw std::runtime_error(
        std::format("[OrtEngine] Unknown ORT_EP='{}'", ort_ep_));
}

bool OrtEngine::load() {
  try {
    apply_execution_provider();
    session_ = std::make_unique<Ort::Session>(
        process_env(), turbo_ocr::onnx::ort_path(model_path_).c_str(),
        session_options_);
  } catch (const Ort::Exception &e) {
    std::cerr << std::format("[OrtEngine] Failed to load ONNX model: {} - {}", model_path_, e.what()) << '\n';
    if (!ort_ep_.empty() && ort_ep_ != "cpu")
      std::cerr << std::format("[OrtEngine] ORT_EP='{}' likely unavailable in this "
                               "onnxruntime build (provider not compiled in / "
                               "provider shared library missing)", ort_ep_) << '\n';
    return false;
  } catch (const std::exception &e) {
    std::cerr << std::format("[OrtEngine] Failed to load ONNX model: {} - {}", model_path_, e.what()) << '\n';
    return false;
  }

  // Get input/output names
  auto input_name = session_->GetInputNameAllocated(0, allocator_);
  input_name_ = input_name.get();

  auto output_name = session_->GetOutputNameAllocated(0, allocator_);
  output_name_ = output_name.get();

  std::cout << std::format("[OrtEngine] Loaded: {} (input={}, output={})",
                          model_path_, input_name_, output_name_) << '\n';
  return true;
}

OrtEngine::InferResult
OrtEngine::infer(const float *input_data,
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

OrtEngine::InferResult
OrtEngine::infer_batch(const float *input_data,
                       const std::vector<int64_t> &input_shape) {
  // ORT runs an N-row {B,3,H,W} input in a single Run; the output is the full
  // {B,seq,classes} tensor. Mechanically identical to infer().
  return infer(input_data, input_shape);
}

OrtEngine::InferView
OrtEngine::infer_batch_view(const float *input_data,
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

void OrtEngine::probe_output_dims(const std::vector<int64_t> &input_shape,
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
