// Per-model-type TRT optimization profiles for the ONNX->TRT builder.

#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/engine/trt/rec_graph_profiles.h"
#include "turbo_ocr/engine/trt/trt_engine.h"
#include "engine_internal.h"

#include <NvInfer.h>

#include <algorithm>
#include <iostream>
#include <string>

namespace turbo_ocr::engine::detail {

bool add_optimization_profiles(nvinfer1::IBuilder &builder,
                               nvinfer1::INetworkDefinition &network,
                               nvinfer1::IBuilderConfig &config,
                               const std::string &type) {
  auto profile = builder.createOptimizationProfile();
  auto input = network.getInput(0);
  bool profile_added_in_branch = false;

  if (type == "det") {
    // MAX is the effective det max-side (per-model max_side_limit, default
    // 1280, overridden by DET_MAX_SIDE) — it must be >= the largest resize
    // output the detector can feed in, else TRT rejects native-resolution
    // inputs at execute time. MIN is the floor (32). OPT is the sweet-spot for
    // typical inputs, capped by MAX so a small effective max doesn't violate
    // MIN<=OPT<=MAX.
    int det_max = read_det_effective_max_side();
    int det_opt = std::min(640, det_max);
    // Batch range 1..8: OPT stays at batch 1 so single-image latency (the common
    // /ocr/raw case) keeps its tuned tactics, while MAX=8 lets the dynamic-batch
    // path (PaddleDet::run_batch, dynamic-batching coalescer) submit batch>1 for
    // throughput — det at batch 8 is ~40% cheaper per image (measured).
    const int det_opt_batch = read_det_opt_batch();  // DET_OPT_BATCH, default 4
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, detection::kDetMaxSideMin, detection::kDetMaxSideMin});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{det_opt_batch, 3, det_opt, det_opt});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{8, 3, det_max, det_max});
  } else if (type == "rec") {
    // Profile 0: dynamic profile, covers all shapes (the default serving path).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 48, 48});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 48, 320});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{32, 3, 48, 4000});
    // Static (batch,width) profiles + maxAuxStreams=0 only when CUDA graphs
    // are opted into — they exist solely to give the baked-graph path frozen,
    // shape-specialized contexts. Keeping them out of the default build means
    // the default rec engine (and its cache key) is unchanged from baseline.
    if (TrtEngine::graphs_enabled()) {
      config.addOptimizationProfile(profile);
      for (const auto &gp : kRecGraphProfiles) {
        auto *p = builder.createOptimizationProfile();
        const nvinfer1::Dims4 d{gp.batch, 3, 48, gp.width};
        p->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, d);
        p->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, d);
        p->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, d);
        config.addOptimizationProfile(p);
      }
      config.setMaxAuxStreams(0);
      profile_added_in_branch = true;
    }
  } else if (type == "layout") {
    // PP-DocLayoutV3 has 3 inputs: image [B,3,800,800], im_shape [B,2],
    // scale_factor [B,2]. paddle2onnx does not guarantee input ordering, so
    // dispatch by name. Batch profile: min=1, opt=4, max=8 (measured sweet
    // spot on RTX 5090: 1.18 ms / image at B=4).
    for (int i = 0; i < network.getNbInputs(); ++i) {
      auto *in = network.getInput(i);
      std::string name = in->getName();
      if (name == "image") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{4, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{8, 3, 800, 800});
      } else if (name == "im_shape" || name == "scale_factor") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims2{4, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims2{8, 2});
      } else {
        std::cerr << "[TRT] Unexpected input for layout model: " << name << '\n';
        return false;
      }
    }
  } else if (type == "table_cls") {
    // PP-LCNet_x1_0_table_cls (224×224, 2 classes wired/wireless). Shape
    // profile from turbostruct-rs/crates/turbostruct-engine/src/builder/
    // profiles.rs::populate_table_cls (lines 166-183).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{8, 3, 224, 224});
  } else if (type == "table_cell_wired" || type == "table_cell_wireless") {
    // RT-DETR-L_{wired,wireless}_table_cell_det — same triple-input contract
    // as PP-DocLayoutV3 but at 640×640. Profile from profiles.rs::
    // populate_table_cell (lines 185-222).
    for (int i = 0; i < network.getNbInputs(); ++i) {
      auto *in = network.getInput(i);
      std::string name = in->getName();
      if (name == "image") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{1, 3, 640, 640});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{4, 3, 640, 640});
      } else if (name == "im_shape" || name == "scale_factor") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims2{4, 2});
      } else {
        std::cerr << "[TRT] Unexpected input for " << type << ": " << name
                  << '\n';
        return false;
      }
    }
  } else if (type == "slanext_wired" || type == "slanext_wireless") {
    // SLANeXt {wired,wireless} encoder, 488×488. Decoder Loop body is part
    // of the same engine — internal dynamic dim handled by TRT. Profile from
    // profiles.rs::populate_slanext (lines 224-243).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 488, 488});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 488, 488});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{4, 3, 488, 488});
  } else if (type == "formula") {
    // PP-FormulaNet-S — single grayscale 384×384 input, MTP-K=3 decoder
    // Loop op. Profile from profiles.rs::populate_formula (lines 245-276).
    // HGNetV2-B4 encoder carries Q/DQ ops baked into the ONNX; TRT honors
    // those under FP16 weakly-typed networks. Do NOT setFlag(kINT8) with a
    // fresh calibrator here — the decoder Loop body must stay FP16 (plan
    // 07 §5).
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 1, 384, 384});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 1, 384, 384});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{8, 1, 384, 384});
  } else if (type == "doc_ori") {
    // PP-LCNet_x1_0_doc_ori document-orientation classifier, 224x224. Opt at
    // batch 1 (one page per autorotate detect); max 8 to match the model's
    // own dynamic profile. Must match kDocOriSize in
    // classification/doc_orientation_common.h.
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{8, 3, 224, 224});
  } else if (type == "slanext_encoder") {
    // SLANeXt CNN encoder (PP-LCNet), fixed 488×488 letterbox (ResizeByLong488 +
    // pad). Batch pinned to 1 (one table region per encode). Output feature
    // transpose_0.tmp_0.0 = [1, T_enc=256, C=96] feeds the host GRU decoder.
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 488, 488});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, 488, 488});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{1, 3, 488, 488});
  } else {
    // cls: PP-OCRv5 textline orientation classifier (PP-LCNet_x0_25), input
    // 80x160. Must match kClsImageH/kClsImageW in classification/paddle_cls.h.
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{128, 3, 80, 160});
  }
  if (!profile_added_in_branch)
    config.addOptimizationProfile(profile);
  return true;
}

} // namespace turbo_ocr::engine::detail
