// FFDetr runner — RF-DETR over a page raster, decoded to FormField proposals.
//
// One session, one set of staging buffers, everything that does not change
// between calls resolved once at load: the same shape as the layout stage, for
// the same reason (a pipeline worker owns its instance and calls run() in a
// loop, so per-call heap churn is pure overhead).
//
// The decode is DETR-with-focal-loss, not softmax: every (query, class) pair
// is scored independently through a sigmoid, so one query can legitimately
// survive under two classes and there is no background class to skip. The
// reference top-ks over the flattened (query x class) product and then applies
// a class-agnostic NMS; thresholding first yields the same survivors without
// materialising the top-k, since the threshold is far above where 300 queries
// would ever compete for 300 slots.

#include "turbo_ocr/analysis/forms/field_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>

#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/onnx/ort_path.h"  // ORTCHAR_T path (wchar_t on Windows)

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/onnx/coreml_ep.h" // the shared CoreML env policy
#include "turbo_ocr/onnx/host_ort_threads.h"

namespace turbo_ocr::forms {

namespace {

// ImageNet statistics, as RF-DETR's own preprocessing uses them
// (rfdetr/detr.py: means=[0.485,0.456,0.406], stds=[0.229,0.224,0.225]).
constexpr std::array<float, 3> kMean{0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kStd{0.229f, 0.224f, 0.225f};

// The head has 91 output slots; CommonForms trained only the first three and
// FFDetrDetector maps exactly {0: TextBox, 1: ChoiceButton, 2: Signature}.
// The remaining 88 are untrained leftovers of the COCO-shaped checkpoint and
// must never be read — load() asserts they stay silent on a real page.
constexpr int kFormClasses = 3;

[[nodiscard]] FieldType class_to_type(int cls) noexcept {
  switch (cls) {
  case 1: return FieldType::Checkbox;
  case 2: return FieldType::Signature;
  default: return FieldType::Text;
  }
}

[[nodiscard]] inline float sigmoid(float x) noexcept {
  return 1.0f / (1.0f + std::exp(-x));
}

struct Candidate {
  float score;
  int cls;
  float x0, y0, x1, y1;
};

// Class-agnostic greedy NMS, matching the reference's
// with_nms(threshold, class_agnostic=True). Agnostic matters: a checkbox and a
// text input predicted over the same rectangle are one widget seen two ways,
// and emitting both would put two overlapping fields in the PDF.
void nms_inplace(std::vector<Candidate> &cand, float iou_thr) {
  std::sort(cand.begin(), cand.end(),
            [](const Candidate &a, const Candidate &b) {
              if (a.score != b.score) return a.score > b.score;
              // Deterministic tiebreak: identical scores must not depend on
              // the order the (query, class) loop happened to emit them.
              if (a.y0 != b.y0) return a.y0 < b.y0;
              return a.x0 < b.x0;
            });

  std::vector<Candidate> kept;
  kept.reserve(cand.size());
  for (const Candidate &c : cand) {
    const float area_c = (c.x1 - c.x0) * (c.y1 - c.y0);
    bool suppressed = false;
    for (const Candidate &k : kept) {
      const float ix = std::min(c.x1, k.x1) - std::max(c.x0, k.x0);
      const float iy = std::min(c.y1, k.y1) - std::max(c.y0, k.y0);
      if (ix <= 0.0f || iy <= 0.0f) continue;
      const float inter = ix * iy;
      const float area_k = (k.x1 - k.x0) * (k.y1 - k.y0);
      const float uni = area_c + area_k - inter;
      if (uni > 0.0f && inter / uni > iou_thr) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push_back(c);
  }
  cand.swap(kept);
}

} // namespace

struct FieldModel::Impl {
  FieldModelOptions opt;

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "FFDetr"};
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo memory_info{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  std::string input_name_storage;
  std::vector<std::string> output_name_storage;
  const char *input_names[1]{};
  std::vector<const char *> output_names;

  // Which output slot is which, resolved BY SHAPE rather than by name: the
  // boxes tensor is the [1, Q, 4] one and the logits the [1, Q, C] one. Names
  // ("dets"/"labels") are an artefact of the exporter and a re-export under a
  // newer rfdetr could rename them; the shapes are the contract.
  int dets_idx = -1;
  int logits_idx = -1;
  int64_t queries = 0;
  int64_t num_classes = 0;

  // Staging, allocated once. `resized` stays CV_8UC3 on purpose — see
  // preprocess(): the /255, the mean and the std all fold into one multiply
  // and add per sample, so there is never a float copy of the image.
  cv::Mat rgb, resized;
  std::vector<float> input_chw;
  std::vector<Ort::Value> input_tensors;

  [[nodiscard]] bool resolve_io() {
    if (session->GetInputCount() != 1) {
      std::cerr << "[ffdetr] expected 1 input, got " << session->GetInputCount()
                << '\n';
      return false;
    }
    input_name_storage = session->GetInputNameAllocated(0, allocator).get();
    input_names[0] = input_name_storage.c_str();

    const auto in_shape = session->GetInputTypeInfo(0)
                              .GetTensorTypeAndShapeInfo()
                              .GetShape();
    if (in_shape.size() != 4 || in_shape[1] != 3) {
      std::cerr << "[ffdetr] unexpected input rank/channels\n";
      return false;
    }
    // The graph is exported with the resolution baked in; trust the graph over
    // the option, so a re-export at another size cannot silently feed the
    // model a wrongly-scaled page.
    if (in_shape[2] > 0 && in_shape[3] > 0) {
      const int s = static_cast<int>(in_shape[2]);
      if (s != static_cast<int>(in_shape[3])) {
        std::cerr << "[ffdetr] non-square input " << in_shape[2] << "x"
                  << in_shape[3] << " is not supported\n";
        return false;
      }
      if (s != opt.image_size) {
        std::cout << "[ffdetr] graph is " << s << "px; using that (option said "
                  << opt.image_size << ")\n";
        opt.image_size = s;
      }
    }

    const size_t no = session->GetOutputCount();
    output_name_storage.reserve(no);
    for (size_t i = 0; i < no; ++i)
      output_name_storage.emplace_back(
          session->GetOutputNameAllocated(i, allocator).get());
    output_names.reserve(no);
    for (const auto &s : output_name_storage) output_names.push_back(s.c_str());

    for (size_t i = 0; i < no; ++i) {
      const auto shape = session->GetOutputTypeInfo(i)
                             .GetTensorTypeAndShapeInfo()
                             .GetShape();
      if (shape.size() != 3) continue;
      if (shape[2] == 4 && dets_idx < 0) {
        dets_idx = static_cast<int>(i);
        queries = shape[1];
      } else if (shape[2] > 4 && logits_idx < 0) {
        logits_idx = static_cast<int>(i);
        num_classes = shape[2];
      }
    }
    if (dets_idx < 0 || logits_idx < 0) {
      std::cerr << "[ffdetr] could not find [1,Q,4] boxes and [1,Q,C] logits "
                   "among the outputs\n";
      return false;
    }
    if (num_classes < kFormClasses) {
      std::cerr << "[ffdetr] head has " << num_classes << " classes, need >= "
                << kFormClasses << '\n';
      return false;
    }
    // The two heads must agree on the query count: `queries` comes from the
    // boxes head and later indexes logits[q*C + c] for q in [0, Q). A
    // re-exported graph whose heads disagree (different NMS settings, a
    // truncated head) would read past the ORT-owned logits buffer on every
    // inference — refuse at load, where the error is cheap and actionable.
    if (const auto lshape = session->GetOutputTypeInfo(static_cast<size_t>(logits_idx))
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
        lshape.size() == 3 && lshape[1] != queries) {
      std::cerr << "[ffdetr] output heads disagree on query count: boxes="
                << queries << " logits=" << lshape[1] << '\n';
      return false;
    }

    const int s = opt.image_size;
    input_chw.assign(static_cast<size_t>(3) * s * s, 0.0f);
    const std::array<int64_t, 4> shape{1, 3, s, s};
    input_tensors.clear();
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, input_chw.data(), input_chw.size(), shape.data(),
        shape.size()));
    return true;
  }

  // Resize to the square the graph expects, normalise, and write planar CHW
  // straight into the persistent buffer the cached input tensor views.
  //
  // The square resize does NOT preserve aspect ratio — that is the reference's
  // behaviour, not an oversight, and it is what lets normalised output coords
  // map back onto the page with a plain multiply instead of letterbox math.
  void preprocess(const cv::Mat &page) {
    const int s = opt.image_size;
    if (page.channels() == 1)
      cv::cvtColor(page, rgb, cv::COLOR_GRAY2RGB);
    else if (page.channels() == 4)
      cv::cvtColor(page, rgb, cv::COLOR_BGRA2RGB);
    else
      cv::cvtColor(page, rgb, cv::COLOR_BGR2RGB);

    // INTER_AREA is the antialiasing downscale; a page raster is essentially
    // always larger than 1024, so this is the branch that runs. Plain bilinear
    // here would alias exactly the thin rules and small checkbox edges the
    // model needs to see.
    cv::resize(rgb, resized, cv::Size(s, s), 0, 0,
               (rgb.cols > s || rgb.rows > s) ? cv::INTER_AREA
                                              : cv::INTER_LINEAR);

    // Interleaved uint8 -> planar normalised float in ONE pass, writing
    // straight into the buffer the cached input tensor views.
    //
    // The obvious spelling — convertTo(CV_32F, 1/255), split(), then
    // (plane - mean) / std — costs a 12 MB float image, three more plane
    // allocations and four passes over 3M samples, EVERY call. Folding /255,
    // the mean and the std into one scale+shift removes all of it: the only
    // float memory that exists is input_chw, allocated once at load.
    const size_t plane = static_cast<size_t>(s) * s;
    float scale[3], shift[3];
    for (int c = 0; c < 3; ++c) {
      scale[c] = 1.0f / (255.0f * kStd[c]);
      shift[c] = -kMean[c] / kStd[c];
    }
    const uint8_t *src = resized.ptr<uint8_t>();
    float *r = input_chw.data();
    float *g = r + plane;
    float *b = g + plane;
    for (size_t i = 0; i < plane; ++i) {
      r[i] = static_cast<float>(src[3 * i + 0]) * scale[0] + shift[0];
      g[i] = static_cast<float>(src[3 * i + 1]) * scale[1] + shift[1];
      b[i] = static_cast<float>(src[3 * i + 2]) * scale[2] + shift[2];
    }
  }
};

FieldModel::FieldModel() : impl_(std::make_unique<Impl>()) {}
FieldModel::~FieldModel() = default;

const FieldModelOptions &FieldModel::options() const noexcept {
  return impl_->opt;
}

std::unique_ptr<FieldModel> FieldModel::load(const std::string &onnx_path,
                                             const FieldModelOptions &opt) {
  std::unique_ptr<FieldModel> self(new FieldModel());
  self->impl_->opt = opt;

  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(host_ort_intra_op_threads(opt.stage_default_threads));
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
  // Opt-in, and off by default because it is MEASURED SLOWER, not out of
  // caution. Same page, same machine, median of 7:
  //
  //     fp32  CPU 600 ms   CoreML  952 ms
  //     fp16  CPU 718 ms   CoreML  718 ms   (identical: it all fell back)
  //
  // ORT will not give CoreML the graph — "CoreML does not support shapes with
  // dimension values of 0, Input:/transformer/Slice_6_output_0 {1,0,4}" — so
  // the model is cut into partitions, most of it runs on CPU anyway, and the
  // handoffs cost more than the accelerator saves. Running the graph through
  // onnx-simplifier first (1792 -> 918 nodes) did not change this: 1095 ms.
  //
  // Output is correct when it is enabled — no repeat of the layout.onnx NaN on
  // ORT 1.24 — so the flag stays available for a future ORT or a graph without
  // the zero-length slice. Verify BOTH latency and field counts before
  // trusting it: a detector that silently returns nothing is indistinguishable
  // from a page that genuinely has no fields.
  if (env::env_enabled("FFDETR_COREML")) {
    const uint32_t flags = engine::coreml_flags();
    // Say so when the provider is REFUSED rather than dropping the status on
    // the floor: silently falling back to CPU here looks identical to CoreML
    // working, and the two differ by an order of magnitude in latency.
    if (OrtStatus *st =
            OrtSessionOptionsAppendExecutionProvider_CoreML(so, flags)) {
      std::cerr << "[ffdetr] CoreML rejected: "
                << Ort::GetApi().GetErrorMessage(st) << " — using CPU\n";
      Ort::GetApi().ReleaseStatus(st);
    } else {
      std::cout << "[ffdetr] CoreML enabled (flags=0x" << std::hex << flags
                << std::dec << ")\n";
    }
  }
#endif

  try {
    self->impl_->session =
        std::make_unique<Ort::Session>(self->impl_->env, turbo_ocr::onnx::ort_path(onnx_path).c_str(), so);
  } catch (const Ort::Exception &e) {
    // Absence is a supported state: the caller degrades to geometry-only.
    std::cerr << "[ffdetr] not loaded (" << onnx_path << "): " << e.what()
              << '\n';
    return nullptr;
  }

  if (!self->impl_->resolve_io()) return nullptr;

  std::cout << "[ffdetr] Loaded " << onnx_path << " (" << self->impl_->queries
            << " queries, " << self->impl_->num_classes << " head slots, "
            << self->impl_->opt.image_size << "px)\n";
  return self;
}

std::vector<FormField> FieldModel::run(const cv::Mat &page) {
  std::vector<FormField> out;
  if (!impl_ || !impl_->session || page.empty()) return out;
  Impl &im = *impl_;

  im.preprocess(page);

  std::vector<Ort::Value> outputs;
  try {
    outputs = im.session->Run(Ort::RunOptions{nullptr}, im.input_names,
                              im.input_tensors.data(), 1,
                              im.output_names.data(), im.output_names.size());
  } catch (const Ort::Exception &e) {
    std::cerr << "[ffdetr] inference failed: " << e.what() << '\n';
    return out;
  }

  const float *dets = outputs[im.dets_idx].GetTensorData<float>();
  const float *logits = outputs[im.logits_idx].GetTensorData<float>();
  const auto Q = static_cast<size_t>(im.queries);
  const auto C = static_cast<size_t>(im.num_classes);
  const float pw = static_cast<float>(page.cols);
  const float ph = static_cast<float>(page.rows);

  std::vector<Candidate> cand;
  for (size_t q = 0; q < Q; ++q) {
    for (int c = 0; c < kFormClasses; ++c) {
      const float s = sigmoid(logits[q * C + static_cast<size_t>(c)]);
      if (s < im.opt.confidence) continue;

      // Boxes come back as cxcywh normalised to the resized square; because
      // that resize was a plain per-axis scale, the same normalised numbers
      // apply to the original page.
      const float cx = dets[q * 4 + 0], cy = dets[q * 4 + 1];
      const float bw = dets[q * 4 + 2], bh = dets[q * 4 + 3];
      Candidate k;
      k.score = s;
      k.cls = c;
      k.x0 = std::clamp((cx - bw * 0.5f) * pw, 0.0f, pw);
      k.y0 = std::clamp((cy - bh * 0.5f) * ph, 0.0f, ph);
      k.x1 = std::clamp((cx + bw * 0.5f) * pw, 0.0f, pw);
      k.y1 = std::clamp((cy + bh * 0.5f) * ph, 0.0f, ph);
      if (k.x1 - k.x0 < 1.0f || k.y1 - k.y0 < 1.0f) continue;
      cand.push_back(k);
    }
  }

  nms_inplace(cand, im.opt.nms_iou);

  out.reserve(cand.size());
  for (const Candidate &c : cand) {
    FormField f;
    f.type = class_to_type(c.cls);
    f.confidence = c.score;
    f.source = "ffdetr";
    const int x0 = static_cast<int>(std::lround(c.x0));
    const int y0 = static_cast<int>(std::lround(c.y0));
    const int x1 = static_cast<int>(std::lround(c.x1));
    const int y1 = static_cast<int>(std::lround(c.y1));
    f.box = Box{{{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}}};
    out.push_back(std::move(f));
  }
  return out;
}

} // namespace turbo_ocr::forms
