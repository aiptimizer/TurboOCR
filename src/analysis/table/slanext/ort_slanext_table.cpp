#include "turbo_ocr/analysis/table/slanext/ort_slanext_table.h"

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/analysis/table/slanext/slanext_paths.h"
#include "turbo_ocr/analysis/table/slanext/slanext_postprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "turbo_ocr/onnx/ort_path.h"  // ORTCHAR_T path (wchar_t on Windows)

#include "turbo_ocr/backend/stages.h"  // backend::IRecognizer (cell fill)

namespace turbo_ocr::table {

namespace {

constexpr int DST = SlanextDecoderWeights::kInputSize;  // 488
constexpr int kTenc = SlanextDecoderWeights::kTenc;     // 256
constexpr int kCtx = SlanextDecoderWeights::kCtx;       // 96

// CPU port of slanext_pre_rgb_kernel (table_kernels.cu): ResizeByLong-488
// (preserve AR, bilinear) + per-channel ImageNet normalize + bottom-right pad
// with 0 in normalized space, written R,G,B plane order. Source crop is BGR8.
// Bit-matched to the GPU kernel: same (dx+0.5)*scale-0.5 sampling, same
// sub-rect clamp, same means/stds. Produces [3,488,488] CHW float.
void slanext_preprocess(const cv::Mat &crop, std::vector<float> &dst_chw) {
  const int rect_w = crop.cols, rect_h = crop.rows;
  dst_chw.assign(static_cast<std::size_t>(3) * DST * DST, 0.0f);
  if (rect_w <= 0 || rect_h <= 0) return;

  const float long_side = static_cast<float>(std::max(rect_w, rect_h));
  const float s = static_cast<float>(DST) / long_side;
  int new_w = static_cast<int>(std::lround(rect_w * s));
  int new_h = static_cast<int>(std::lround(rect_h * s));
  new_w = std::clamp(new_w, 1, DST);
  new_h = std::clamp(new_h, 1, DST);
  const float scale = 1.0f / s;

  // BGR source channel means/inv-stds (kernel passes them so that, after the
  // R,G,B store, Red gets 0.485/0.229, Green 0.456/0.224, Blue 0.406/0.225).
  const float mean_b = 0.406f, mean_g = 0.456f, mean_r = 0.485f;
  const float istd_b = 1.0f / 0.225f, istd_g = 1.0f / 0.224f, istd_r = 1.0f / 0.229f;
  const float inv255 = 1.0f / 255.0f;
  const int plane = DST * DST;
  const auto *src = crop.ptr<uint8_t>();
  const int step = static_cast<int>(crop.step);

  auto px = [&](int x, int y, int c) -> float {
    return static_cast<float>(src[static_cast<std::size_t>(y) * step + x * 3 + c]);
  };

  for (int dy = 0; dy < new_h; ++dy) {
    const float sy_rel = (dy + 0.5f) * scale - 0.5f;
    int y0 = static_cast<int>(std::floor(sy_rel));
    const float fy = sy_rel - y0;
    int y1 = y0 + 1;
    y0 = std::clamp(y0, 0, rect_h - 1);
    y1 = std::clamp(y1, 0, rect_h - 1);
    for (int dx = 0; dx < new_w; ++dx) {
      const float sx_rel = (dx + 0.5f) * scale - 0.5f;
      int x0 = static_cast<int>(std::floor(sx_rel));
      const float fx = sx_rel - x0;
      int x1 = x0 + 1;
      x0 = std::clamp(x0, 0, rect_w - 1);
      x1 = std::clamp(x1, 0, rect_w - 1);
      const float w00 = (1.0f - fx) * (1.0f - fy), w10 = fx * (1.0f - fy);
      const float w01 = (1.0f - fx) * fy, w11 = fx * fy;
      const int idx = dy * DST + dx;
      // channel 0=Blue, 1=Green, 2=Red in source
      const float b = w00 * px(x0, y0, 0) + w10 * px(x1, y0, 0) +
                      w01 * px(x0, y1, 0) + w11 * px(x1, y1, 0);
      const float g = w00 * px(x0, y0, 1) + w10 * px(x1, y0, 1) +
                      w01 * px(x0, y1, 1) + w11 * px(x1, y1, 1);
      const float r = w00 * px(x0, y0, 2) + w10 * px(x1, y0, 2) +
                      w01 * px(x0, y1, 2) + w11 * px(x1, y1, 2);
      dst_chw[0 * plane + idx] = (r * inv255 - mean_r) * istd_r;  // R
      dst_chw[1 * plane + idx] = (g * inv255 - mean_g) * istd_g;  // G
      dst_chw[2 * plane + idx] = (b * inv255 - mean_b) * istd_b;  // B
    }
  }
}




}  // namespace

struct OrtSlanextEncoder::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "slanext_cpu"};
  std::unique_ptr<Ort::Session> sess;
  Ort::MemoryInfo mem{nullptr};
  std::string in_name, out_name;
};

OrtSlanextEncoder::~OrtSlanextEncoder() = default;

bool OrtSlanextEncoder::load(const std::string &encoder_onnx,
                             const std::string &decoder_bin,
                             const std::string &dict_path) {
  p_ = std::make_unique<Impl>();
  try {
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    p_->sess = std::make_unique<Ort::Session>(p_->env, turbo_ocr::onnx::ort_path(encoder_onnx).c_str(), opts);
    p_->mem = Ort::MemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions alloc;
    p_->in_name = p_->sess->GetInputNameAllocated(0, alloc).get();
    p_->out_name = p_->sess->GetOutputNameAllocated(0, alloc).get();
  } catch (const std::exception &e) {
    std::cerr << "[slanext-cpu] encoder load failed (" << encoder_onnx << "): " << e.what() << '\n';
    return false;
  }
  if (!weights_.load(decoder_bin)) return false;
  try {
    dict_ = dict_path.empty() ? CharDict::from_text(DEFAULT_DICT_TEXT)
                              : CharDict::from_path(dict_path);
  } catch (const std::exception &e) {
    // Match the GPU path: a missing/unreadable CONFIGURED dict falls back to the
    // bundled DEFAULT_DICT_TEXT (loudly) instead of hard-failing table recog.
    std::cerr << "[slanext-cpu] WARNING: dict load failed (" << e.what()
              << ") — falling back to bundled default dict\n";
    try {
      dict_ = CharDict::from_text(DEFAULT_DICT_TEXT);
    } catch (const std::exception &e2) {
      std::cerr << "[slanext-cpu] bundled default dict load failed: " << e2.what() << '\n';
      return false;
    }
  }
  return true;
}

StructureResult OrtSlanextEncoder::infer(const cv::Mat &page, const Box &region) {
  StructureResult empty{};
  if (!p_ || !p_->sess || page.empty()) return empty;

  auto bb = aabb(region);
  const int rx = std::clamp(bb[0], 0, page.cols - 1);
  const int ry = std::clamp(bb[1], 0, page.rows - 1);
  const int rw = std::clamp(bb[2] - rx, 1, page.cols - rx);
  const int rh = std::clamp(bb[3] - ry, 1, page.rows - ry);
  if (rw <= 0 || rh <= 0) return empty;

  // Zero-copy ROI view: slanext_preprocess samples through crop.step, so it
  // honours the parent's row stride directly and needs no contiguous clone.
  // Reading the shared page data yields byte-identical samples to a cloned copy
  // (same pixels, same order) — the clone was pure defensiveness, so dropping it
  // removes a per-region allocation + copy without changing the encoder input.
  const cv::Mat crop = page(cv::Rect(rx, ry, rw, rh));

  // Per-thread scratch reused across regions: both the preprocess input and the
  // encoder output are fixed-shape, so after the first infer() each already
  // holds its capacity and steady-state inference allocates nothing (kills the
  // per-region ~2.9 MB in_chw + feat heap churn plain locals incurred). Values
  // written are identical to fresh locals — slanext_preprocess unconditionally
  // fills all 3*DST*DST elements (content + zero pad) and the encoder Run fully
  // overwrites feat — so the encoder I/O, and thus the decode, is bit-unchanged.
  // thread_local (not a member) keeps a pipeline pool race-free without locking.
  thread_local std::vector<float> in_chw;
  thread_local std::vector<float> feat;
  slanext_preprocess(crop, in_chw);
  feat.resize(static_cast<std::size_t>(kTenc) * kCtx);

  try {
    const std::array<int64_t, 4> in_shape{1, 3, DST, DST};
    Ort::Value xv = Ort::Value::CreateTensor<float>(
        p_->mem, in_chw.data(), in_chw.size(), in_shape.data(), in_shape.size());
    const std::array<int64_t, 3> out_shape{1, kTenc, kCtx};
    Ort::Value ov = Ort::Value::CreateTensor<float>(
        p_->mem, feat.data(), feat.size(), out_shape.data(), out_shape.size());
    const char *ins[] = {p_->in_name.c_str()};
    const char *outs[] = {p_->out_name.c_str()};
    p_->sess->Run(Ort::RunOptions{nullptr}, ins, &xv, 1, outs, &ov, 1);
  } catch (const std::exception &e) {
    std::cerr << "[slanext-cpu] encoder run failed for region [" << rx << ',' << ry
              << ',' << rw << 'x' << rh << "] — DROPPED: " << e.what() << '\n';
    return empty;
  }

  return slanext_host_decode(feat.data(), weights_, dict_, rw, rh);
}

bool OrtSlanextTableRecognizer::load() {
  // Shared encoder resolution (env override, else the baked release path) —
  // see resolve_slanext_encoder for why this must not fork per backend.
  const std::string enc = table::resolve_slanext_encoder(
      env::env_or("TABLE_SLANEXT_ENCODER_ONNX", ""));
  if (enc.empty()) return false;
  if (!std::filesystem::exists(enc)) {
    std::cerr << "[slanext-cpu] encoder ONNX missing: " << enc << '\n';
    return false;
  }
  const std::string dec = env::env_or("TABLE_SLANEXT_DECODER_BIN", slanext_default_decoder_bin(enc));
  const std::string dict = env::env_or("TABLE_SLANEXT_DICT", slanext_default_dict(enc));
  wired_ = std::make_unique<OrtSlanextEncoder>();
  if (!wired_->load(enc, dec, dict)) {
    wired_.reset();
    return false;
  }

  if (env::env_present("TABLE_SLANEXT_WIRELESS_ENCODER_ONNX"))
    std::cerr << "[slanext-cpu] wired/wireless routing removed — ignoring "
                 "TABLE_SLANEXT_WIRELESS_ENCODER_ONNX\n";

  std::cout << "[Pipeline] Table backend=slanext-cpu (ORT-CPU encoder + host decode, wired)\n";
  return true;
}

std::vector<router::TableResult>
OrtSlanextTableRecognizer::run(const cv::Mat &page,
                               const backend::ImageView &page_view,
                               const std::vector<Box> &regions,
                               const std::vector<OCRResultItem> &page_ocr,
                               backend::DeviceQueue &queue) {
  std::vector<router::TableResult> out;
  out.reserve(regions.size());

  for (std::size_t ti = 0; ti < regions.size(); ++ti) {
   try {
    const Box &region = regions[ti];
    const auto sr = wired_->infer(page, region);

    // THE shared host postprocess (slanext_postprocess.h) — see the GPU twin.
    // The cell hook goes through backend::IRecognizer, so whichever recognizer
    // the active backend installed (ORT-CPU, MPSGraph, OpenVINO) fills the empty
    // cells; IRecognizer::run already returns vector<RecResult>, which IS
    // SlanextCellRecFn's vector<pair<string,float>>.
    table::SlanextCellRecFn cell_fn;
    if (cell_rec_)
      cell_fn = [&](const std::vector<Box> &empty_cells) {
        return cell_rec_->run(page_view, empty_cells, queue);
      };
    router::TableResult tr =
        table::slanext_postprocess_region(sr, page_ocr, region, cell_fn);
    out.push_back(std::move(tr));
   } catch (const std::exception &e) {
    // Per-region degrade, mirroring the GPU SlanextTableRecognizer: an ORT /
    // decode fault on ONE table region must not abort the whole page (every
    // other table lost) — also mirrors infer()'s graceful "region DROPPED"
    // path. One TableResult per region is pushed unconditionally so the
    // caller's layout_id stamping stays aligned; the empty html is counted as
    // degraded upstream.
    std::cerr << "[slanext-cpu] table region " << ti << " FAILED (" << e.what()
              << ") — region DROPPED, continuing\n";
    router::TableResult tr;
    tr.layout_id = -1;
    tr.box = regions[ti];  // html left empty -> degraded accounting
    out.push_back(std::move(tr));
   }
  }
  return out;
}

}  // namespace turbo_ocr::table
