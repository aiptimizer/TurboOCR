#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/analysis/layout/ort_paddle_layout.h"

#include "turbo_ocr/analysis/layout/picodet_decode.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/onnx/ort_path.h"  // ORTCHAR_T path (wchar_t on Windows)

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/onnx/coreml_ep.h"        // the shared CoreML env policy
#include "turbo_ocr/onnx/host_ort_threads.h" // host_ort_intra_op_threads
#include "turbo_ocr/analysis/layout/layout_postfilter.h"

namespace turbo_ocr::layout {

// One OrtPaddleLayout drives one ONNX session and, like its GPU sibling
// PaddleLayout, owns per-instance staging buffers reused across run() calls —
// so a single detector instance is single-threaded by construction (the
// pipeline pool gives each worker its own detector). Everything that does not
// change between calls — the input/output name tables, the input-name→role
// mapping, the CPU MemoryInfo, and the ORT input tensors themselves (thin views
// over the member staging buffers) — is resolved once at load time. run() then
// only refreshes buffer *contents* and calls Session::Run, with zero per-call
// heap churn for names, tensor objects, or the CHW image buffer.
struct OrtPaddleLayout::Impl {
  enum class InputRole { kImage, kImShape, kScaleFactor };

  // ERROR, not WARNING: at WARNING the CoreML EP narrates every session with
  // "GetCapability, number of partitions supported by CoreML: ..." and
  // "VerifyEachNodeIsAssignedToAnEp", straight to the host application's
  // stderr. Both are informational — a partially-CoreML graph is the NORMAL
  // outcome — and neither is actionable. Real load failures are already
  // reported by this file explicitly. (slanext/ppformulanet already use
  // ERROR here for the same reason.)
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "CpuLayout"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  // Model IO, resolved once in resolve_io().
  std::vector<std::string> input_name_storage;
  std::vector<std::string> output_name_storage;
  std::vector<const char *> input_names;   // c_str() views into *_storage
  std::vector<const char *> output_names;
  std::vector<InputRole> input_roles;      // parallel to input_names
  bool inputs_resolved = false;

  // Staging buffers (reused every run; never reallocated after load).
  cv::Mat resized;
  cv::Mat bgr[3];
  std::vector<float> input_chw =
      std::vector<float>(3 * static_cast<size_t>(kInputSize) * kInputSize);
  std::array<float, 2> im_shape{static_cast<float>(kInputSize),
                                static_cast<float>(kInputSize)};
  std::array<float, 2> scale_factor{1.0f, 1.0f};

  // ORT input Values built once as persistent views over the buffers above.
  // Session::Run reads them read-only, so they stay valid across calls as long
  // as the backing buffers don't move (they never do).
  std::vector<Ort::Value> input_tensors;

  void resolve_io() {
    const size_t ni = session->GetInputCount();
    const size_t no = session->GetOutputCount();

    input_name_storage.reserve(ni);
    for (size_t i = 0; i < ni; ++i)
      input_name_storage.emplace_back(
          session->GetInputNameAllocated(i, allocator).get());
    output_name_storage.reserve(no);
    for (size_t i = 0; i < no; ++i)
      output_name_storage.emplace_back(
          session->GetOutputNameAllocated(i, allocator).get());

    input_names.reserve(ni);
    for (const auto &s : input_name_storage) input_names.push_back(s.c_str());
    output_names.reserve(no);
    for (const auto &s : output_name_storage) output_names.push_back(s.c_str());

    // Map each input slot to a role (same rule the per-call loop used before:
    // any name containing "image" is the image tensor; exact "im_shape" /
    // "scale_factor" for the two metadata tensors). An unclassifiable slot
    // leaves inputs_resolved false so run() reports the mismatch, matching the
    // prior "Input name mismatch" behaviour.
    input_roles.reserve(ni);
    for (const auto &name : input_name_storage) {
      if (name.find("image") != std::string::npos)
        input_roles.push_back(InputRole::kImage);
      else if (name == "im_shape")
        input_roles.push_back(InputRole::kImShape);
      else if (name == "scale_factor")
        input_roles.push_back(InputRole::kScaleFactor);
      else
        return; // inputs_resolved stays false
    }

    build_input_tensors();
    inputs_resolved = true;
  }

  void build_input_tensors() {
    const std::array<int64_t, 4> img_shape{1, 3, kInputSize, kInputSize};
    const std::array<int64_t, 2> vec_shape{1, 2};
    input_tensors.clear();
    input_tensors.reserve(input_roles.size());
    for (const auto role : input_roles) {
      switch (role) {
        case InputRole::kImage:
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, input_chw.data(), input_chw.size(),
              img_shape.data(), img_shape.size()));
          break;
        case InputRole::kImShape:
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, im_shape.data(), im_shape.size(),
              vec_shape.data(), vec_shape.size()));
          break;
        case InputRole::kScaleFactor:
          input_tensors.push_back(Ort::Value::CreateTensor<float>(
              memory_info, scale_factor.data(), scale_factor.size(),
              vec_shape.data(), vec_shape.size()));
          break;
      }
    }
  }
};

OrtPaddleLayout::OrtPaddleLayout() = default;
OrtPaddleLayout::~OrtPaddleLayout() noexcept = default;


#ifdef __APPLE__
namespace {
// Pool-homogeneity latch. Replicas each build their own layout session, so a
// contention blip during replica 3 of 4 would otherwise leave a pool that
// answers the same page differently depending on which replica served it.
// Once ANY layout session has had to drop CoreML, every later one in this
// process skips it, making the pool uniform by construction.
//
// Process-scoped and one-way on purpose: an engine is built once and served
// many times, and silently re-acquiring CoreML mid-pool is the very
// inhomogeneity this exists to prevent. A long-lived process that wants the
// accelerator back after a contention storm must construct a new process —
// noted in docs/reference/python.md.
std::atomic<bool> g_coreml_layout_wedged{false};
}  // namespace

bool coreml_layout_wedged() {
  return g_coreml_layout_wedged.load(std::memory_order_relaxed);
}
void set_coreml_layout_wedged() {
  g_coreml_layout_wedged.store(true, std::memory_order_relaxed);
}
#else
bool coreml_layout_wedged() { return false; }
void set_coreml_layout_wedged() {}
#endif

bool OrtPaddleLayout::load_model(const std::string &onnx_path) {
  impl_ = std::make_unique<Impl>();

  Ort::SessionOptions opts;
  // 2 remains this stage's default; the operator (ORT_NUM_THREADS) or a
  // backend whose det/rec are on an accelerator can raise it. One policy for
  // all three host ORT stages — see common/host_ort_threads.h.
  opts.SetIntraOpNumThreads(host_ort_intra_op_threads(2));
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
  // Set when CoreML is actually attached below, holding an EXACT copy of the
  // options taken BEFORE the append. SessionOptions cannot un-append an EP, so
  // a fallback build needs its own object; Clone() makes it exact by
  // construction rather than by a second builder that has to be kept in sync
  // with the first (threads, graph-opt level, every config entry).
  std::optional<Ort::SessionOptions> without_coreml;

  // Opt-in only (see set_use_coreml). DISABLE_COREML=1 forces it back off, so
  // a provider regression can be ruled out in the field without a rebuild.
  // coreml_layout_wedged() is the pool-homogeneity latch: see below.
  if (use_coreml_) {
    if (!engine::coreml_disabled_by_env() && !coreml_layout_wedged()) {
      const uint32_t coreml_flags = engine::coreml_flags();
      without_coreml = opts.Clone();
      // The append RETURNS a status and it is not decorative: discarding it
      // printed "CoreML enabled" for a session that had no CoreML on it, and
      // leaked the OrtStatus. Handled exactly as
      // OrtEngine::configure_session_() does — own the status (Ort::Status
      // releases it), report what ACTUALLY happened, and keep going on the
      // default CPU provider, because CoreML here is an accelerator for the
      // layout stage and not a precondition for it.
      if (OrtStatus *st = OrtSessionOptionsAppendExecutionProvider_CoreML(
              opts, coreml_flags)) {
        const Ort::Status owned{st}; // takes ownership; releases on scope exit
        without_coreml.reset();  // nothing attached: `opts` is already CPU-only
        TOCR_LOG_WARN("layout: CoreML unavailable — continuing on the "
                      "default CPU provider",
                      "error", owned.GetErrorMessage());
      } else {
        TOCR_LOG_INFO("layout: CoreML enabled", "flags", (long)coreml_flags);
      }
    }
  }
#endif

  const auto ort_model = turbo_ocr::onnx::ort_path(onnx_path);
  try {
    impl_->session = std::make_unique<Ort::Session>(
        impl_->env, ort_model.c_str(), opts);
  } catch (const std::bad_alloc &) {
    // NEVER retried. An allocation failure is not transient contention, and
    // the CPU-provider build below needs MORE host memory than the accelerated
    // one — retrying would turn an honest load failure into a success that
    // dies later, somewhere less debuggable.
    TOCR_LOG_ERROR("layout: out of memory loading the model", "model", onnx_path);
    impl_.reset();
    return false;
  } catch (const std::exception &e) {
    // std::exception, not Ort::Exception: this site caught the narrower type
    // while every sibling catches more, so a non-Ort throw escaped load_model()
    // entirely instead of becoming a clean "layout unavailable".
    const std::string first_error = e.what();
#ifdef __APPLE__
    // The accelerator is OPTIONAL for this stage — the append-failure branch
    // above already says so and keeps going on the CPU provider. Until now the
    // SESSION-BUILD failure path did not honour the same contract: it gave up,
    // so a transient CoreML compile failure (contention on the GPU/ANE)
    // permanently disabled layout on a live engine even though the identical
    // session builds fine without the EP.
    //
    // Rebuild once from the pre-append clone. Report the FIRST error whatever
    // happens — the fallback's error, if any, is a symptom of this one.
    if (without_coreml) {
      // Latch BEFORE the retry so every later stage in this process skips
      // CoreML too. Replicas each load their own layout session, and a pool
      // where some entries run CoreML and others the CPU provider would answer
      // the same page differently depending on which replica served it —
      // src/service/server/unified/backend_stages.cpp documents the opposite
      // invariant ("All entries load identically, so the last wins").
      set_coreml_layout_wedged();
      try {
        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, ort_model.c_str(), *without_coreml);
        coreml_dropped_ = true;
        TOCR_LOG_WARN(
            "layout: CoreML session build failed — rebuilt on the CPU "
            "provider, and every later layout load in this process will skip "
            "CoreML so the replica pool stays homogeneous. Layout is SLOWER "
            "but available; the alternative was no layout at all",
            "model", onnx_path, "error", first_error);
        impl_->resolve_io();
        if (!impl_->inputs_resolved)
          TOCR_LOG_WARN("layout: unexpected input names; run() will report a "
                        "mismatch", "model", onnx_path);
        TOCR_LOG_INFO("layout: loaded (CPU provider)", "model", onnx_path);
        return true;
      } catch (const std::exception &) {
        // Fall through and report the ORIGINAL failure, not this one.
      }
    }
#endif
    TOCR_LOG_ERROR("layout: failed to load", "model", onnx_path,
                   "error", first_error);
    impl_.reset();
    return false;
  }

  impl_->resolve_io();
  if (!impl_->inputs_resolved)
    TOCR_LOG_WARN("layout: unexpected input names; run() will report a "
                  "mismatch", "model", onnx_path);
  TOCR_LOG_INFO("layout: loaded", "model", onnx_path);
  return true;
}

std::vector<LayoutBox> OrtPaddleLayout::run(const cv::Mat &img,
                                             float score_threshold) {
  std::vector<LayoutBox> out;
  if (!impl_ || !impl_->session) return out;
  Impl &im = *impl_;
  if (!im.inputs_resolved) {
    std::cerr << "[cpu_layout] Input name mismatch\n";
    return out;
  }

  const int orig_h = img.rows;
  const int orig_w = img.cols;

  // 1. Preprocess: resize to 800x800, normalize to [0,1], CHW. Writes straight
  //    into the persistent CHW staging buffer that the cached image tensor
  //    already views — no reallocation, no extra copy.
  cv::resize(img, im.resized, cv::Size(kInputSize, kInputSize));

  constexpr int plane = kInputSize * kInputSize;
  cv::split(im.resized, im.bgr);
  cv::Mat p_b(kInputSize, kInputSize, CV_32F, im.input_chw.data());
  cv::Mat p_g(kInputSize, kInputSize, CV_32F, im.input_chw.data() + plane);
  cv::Mat p_r(kInputSize, kInputSize, CV_32F, im.input_chw.data() + 2 * plane);
  im.bgr[0].convertTo(p_b, CV_32F, 1.0 / 255.0); // B
  im.bgr[1].convertTo(p_g, CV_32F, 1.0 / 255.0); // G
  im.bgr[2].convertTo(p_r, CV_32F, 1.0 / 255.0); // R

  // 2. Refresh metadata buffers in place: im_shape=[800,800],
  //    scale_factor=[800/h, 800/w]. The cached tensors view these arrays.
  im.im_shape = {static_cast<float>(kInputSize),
                 static_cast<float>(kInputSize)};
  im.scale_factor = {
      static_cast<float>(kInputSize) / static_cast<float>(orig_h),
      static_cast<float>(kInputSize) / static_cast<float>(orig_w)};

  // 3. Run inference on the pre-built input tensors / name tables.
  std::vector<Ort::Value> outputs;
  try {
    outputs = im.session->Run(
        Ort::RunOptions{nullptr},
        im.input_names.data(), im.input_tensors.data(), im.input_tensors.size(),
        im.output_names.data(), im.output_names.size());
  } catch (const Ort::Exception &e) {
    std::cerr << "[cpu_layout] Inference failed: " << e.what() << '\n';
    return out;
  }

  // 4. Find the (N, 7) detection output.
  const float *dets = nullptr;
  int n_rows = 0;
  for (auto &output : outputs) {
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 2 && shape[1] == 7) {
      dets = output.GetTensorData<float>();
      n_rows = static_cast<int>(shape[0]);
      break;
    }
  }
  if (!dets) {
    // No (N,7) tensor among the outputs = unexpected model output, not a
    // genuinely empty page. Fail loud rather than return an empty layout that
    // reads as "clean" downstream (no-silent-failure).
    std::cerr << "[cpu_layout] model produced no (N,7) detection output; "
                 "returning empty layout — check the layout model\n";
    return out;
  }
  if (n_rows <= 0) return out; // genuinely zero detections — fine
  n_rows = std::min(n_rows, kMaxDetections);

  // TURBO_LAYOUT_DEBUG=1: dump what the model actually produced. Layout going
  // silently empty looks identical to a clean page, so there is no way to tell
  // a broken execution provider from a blank scan without this.
  if (env::env_enabled("TURBO_LAYOUT_DEBUG")) {
    float best = -1.0f;
    for (int i = 0; i < n_rows; ++i) best = std::max(best, dets[i * 7 + 1]);
    std::cerr << "[cpu_layout] rows=" << n_rows << " thr=" << score_threshold
              << " best_score=" << best << " first3=";
    for (int i = 0; i < std::min(n_rows, 3); ++i)
      std::cerr << "{cls=" << dets[i * 7] << ",s=" << dets[i * 7 + 1] << ",box=("
                << dets[i * 7 + 2] << "," << dets[i * 7 + 3] << ","
                << dets[i * 7 + 4] << "," << dets[i * 7 + 5] << ")} ";
    std::cerr << '\n';
  }

  // 5. Decode detections through THE shared PicoDet row decoder
  //    (layout/picodet_decode.h) — including its fail-loud non-finite guard
  //    (the CoreML-EP NaN failure). This loop was briefly a private copy here;
  //    policy identical, now shared with Intel/AMD/TRT.
  out = decode_picodet_rows(dets, n_rows, /*stride=*/7, /*count=*/nullptr,
                            score_threshold, orig_h, orig_w);

  // 6. Shared NMS + oversized-image drop + LAYOUT_MERGE_MODE reconciliation.
  return postfilter_layout_boxes(std::move(out), orig_h, orig_w);
}

} // namespace turbo_ocr::layout
