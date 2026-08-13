// CpuDetector / CpuRecognizer / CpuClassifier / CpuLayout implementation. Every
// method converts a Host ImageView back to a non-owning cv::Mat, drops the
// no-op Host DeviceQueue, and forwards to the wrapped CPU class. No stage logic
// is re-implemented.

#include <cstdlib>
#include <stdexcept>
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "cpu/stages/cpu_stages.h"

#include "cpu/support/host_common.h" // to_mat

namespace turbo_ocr::cpu {

// ---- CpuDetector -----------------------------------------------------------

bool CpuDetector::load(const std::string &model_path) {
  // Default resize/db config + env overrides (same as CpuOcrPipeline::init).
  ready_ = det_.load_model(model_path);
  return ready_;
}

std::vector<turbo_ocr::Box> CpuDetector::run(const backend::ImageView &img,
                                             int /*orig_h*/, int /*orig_w*/,
                                             backend::DeviceQueue & /*queue*/) {
  // OrtPaddleDet reads the original dims from the cv::Mat itself, so orig_h/w
  // are redundant on the host path.
  return det_.run(to_mat(img));
}

// ---- CpuRecognizer ---------------------------------------------------------

bool CpuRecognizer::load(const std::string &model_path) {
  ready_ = rec_.load_model(model_path);
  return ready_;
}

bool CpuRecognizer::load_dict(const std::string &dict_path) {
  return rec_.load_dict(dict_path);
}

std::vector<backend::RecResult>
CpuRecognizer::run(const backend::ImageView &img,
                   const std::vector<turbo_ocr::Box> &boxes,
                   backend::DeviceQueue & /*queue*/) {
  // OrtPaddleRec::run already returns vector<pair<string,float>> == RecResult.
  return rec_.run(to_mat(img), boxes);
}

// ---- CpuClassifier ---------------------------------------------------------

bool CpuClassifier::load(const std::string &model_path) {
  ready_ = cls_.load_model(model_path);
  return ready_;
}

void CpuClassifier::run(const backend::ImageView &img,
                        std::vector<turbo_ocr::Box> &boxes,
                        backend::DeviceQueue & /*queue*/) {
  cls_.run(to_mat(img), boxes);
}

// ---- CpuLayout -------------------------------------------------------------

bool CpuLayout::load(const std::string &model_path) {
  ready_ = layout_.load_model(model_path);
  return ready_;
}

std::vector<turbo_ocr::layout::LayoutBox>
CpuLayout::run(const backend::ImageView &img, int /*orig_h*/, int /*orig_w*/,
               float score_threshold, backend::DeviceQueue & /*queue*/) {
  return layout_.run(to_mat(img), score_threshold);
}



// ---------------------------------------------------------------------------
// make_onnx_stages — THE shared "fast" stage set (see cpu_stages.h).
//
// Lifted verbatim from CpuBackend::load_stages, which is now one caller of it
// (with the default CPU provider). The ONLY vendor-specific input is `ep`.
// ---------------------------------------------------------------------------
OnnxStageSet make_onnx_stages(const backend::BackendConfig &cfg,
                              const backend::EpConfig &ep) {
  OnnxStageSet out;
  backend::StageSet &set = out.stages;

  // Detection + recognition are required.
  auto det = std::make_unique<CpuDetector>();
  det->set_ep_config(ep);
  set.available.detector = det->load(cfg.det_model);
  set.detector = std::move(det);

  auto rec = std::make_unique<CpuRecognizer>();
  rec->set_ep_config(ep);
  bool rec_ok = rec->load(cfg.rec_model);
  if (rec_ok && !cfg.rec_dict.empty())
    rec_ok = rec->load_dict(cfg.rec_dict);
  set.available.recognizer = rec_ok;
  set.recognizer = std::move(rec);

  // Optional stages: null member + false flag when the path is empty or the
  // load fails.
  //
  // A CONFIGURED model that fails to load is REPORTED, not silently skipped.
  // This is the SHARED fast-path builder — cpu/apple/intel/nvidia/amd all reach
  // it — so one log covers every vendor. It matters most for the classifier:
  // there is no CapabilityId for text-line angle cls, so nothing else in the
  // system mentions it at all (/capabilities, the "Stages loaded" line, the
  // response), and a failed load simply stops 180-degree-rotated lines from
  // being corrected, forever, in silence. Layout and doc-orientation at least
  // get a request-time LAYOUT_DISABLED / AUTOROTATE_DISABLED, but an operator
  // who configured them still saw nothing at BOOT explaining why.
  //
  // An ABSENT path stays the tolerated soft-disable it has always been (the
  // `!empty()` guards); only "you asked for this file and it would not load" is
  // reported.
  if (!cfg.cls_model.empty()) {
    auto cls = std::make_unique<CpuClassifier>();
    cls->set_ep_config(ep);
    if (cls->load(cfg.cls_model)) {
      set.available.classifier = true;
      set.classifier = std::move(cls);
    } else {
      TOCR_LOG_ERROR("configured text-line angle classifier failed to load; "
                     "angle classification is DISABLED (rotated lines will "
                     "decode as garbage)",
                     "model", cfg.cls_model);
    }
  }
  if (cfg.want_layout && !cfg.layout_model.empty()) {
    auto lay = std::make_unique<CpuLayout>();
    if (lay->load(cfg.layout_model)) {
      set.available.optional.set(capability::CapabilityId::Layout, true);
      set.layout = std::move(lay);
    } else {
      TOCR_LOG_ERROR("configured layout model failed to load; layout is "
                     "DISABLED (requests will be rejected LAYOUT_DISABLED)",
                     "model", cfg.layout_model);
    }
  }
  if (!cfg.doc_orient_model.empty()) {
    auto ori = std::make_unique<classification::OrtDocOrientation>();
    ori->set_ep_config(ep);
    if (ori->load_model(cfg.doc_orient_model)) {
      set.available.optional.set(capability::CapabilityId::DocOrientation, true);
      out.doc_ori = std::move(ori);
    } else {
      TOCR_LOG_ERROR("configured doc-orientation model failed to load; "
                     "autorotate is DISABLED (requests will be rejected "
                     "AUTOROTATE_DISABLED)",
                     "model", cfg.doc_orient_model);
    }
  }

  set.available.optional.set(capability::CapabilityId::Table, cfg.want_tables);
  set.available.optional.set(capability::CapabilityId::Formula, cfg.want_formulas);
  return out;
}

// ---------------------------------------------------------------------------
// Vendor-neutral mode policy (see cpu_stages.h).
// ---------------------------------------------------------------------------
backend::EngineMode resolve_engine_mode(std::string_view vendor,
                                        const backend::BackendConfig &cfg,
                                        bool native_available) {
  switch (cfg.mode) {
  case backend::EngineMode::Onnx:
    return backend::EngineMode::Onnx;

  case backend::EngineMode::Native:
    // EXPLICIT native with nothing to load is a configuration error. Silently
    // serving the slower path would make "my ultra engine is not being used"
    // indistinguishable from "my ultra engine is slow".
    if (!native_available)
      throw std::runtime_error(
          "backend '" + std::string(vendor) +
          "' was asked for engine mode 'native' but its graph engine has no "
          "artefact to load for these models — build/export one, or use "
          "engine mode 'onnx' (the .onnx on the vendor provider) or 'auto'.");
    return backend::EngineMode::Native;

  case backend::EngineMode::Auto:
  default:
    if (native_available) return backend::EngineMode::Native;
    // LOUD: a fallback nobody can see is a silent performance cliff.
    TOCR_LOG_INFO("engine mode: no native artefact — using the ONNX fast path",
                  "backend", vendor, "provider",
                  backend::onnx_provider_for(vendor), "det",
                  std::string_view(cfg.det_model));
    return backend::EngineMode::Onnx;
  }
}

OnnxStageSet make_vendor_onnx_stages(std::string_view vendor,
                                     const backend::BackendConfig &cfg) {
  backend::EpConfig ep = cfg.ep;
  if (ep.provider.empty()) {
    // TURBO_EP_PROVIDER overrides the vendor default. This exists because a
    // vendor's provider can be ABSENT FROM THE ONNX RUNTIME even on that
    // vendor's own hardware: the stock onnxruntime-linux-x64 build ships only
    // the CPU provider, so `--backend intel` in onnx mode asks for "openvino"
    // and gets a clean "provider not compiled in" failure on a perfectly good
    // Intel box. Rather than silently downgrading (which would report Intel
    // acceleration while running MLAS), the operator names the provider they
    // actually have. "cpu" / "" selects the default provider explicitly.
    ep.provider = env::env_or("TURBO_EP_PROVIDER",
                              backend::onnx_provider_for(vendor));
  }
  if (ep.provider == "cpu") ep.provider.clear(); // "" == default provider
  return make_onnx_stages(cfg, ep);
}

} // namespace turbo_ocr::cpu
