#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/analysis/formula/ppformulanet/ort_formula_recognizer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "turbo_ocr/analysis/formula/ppformulanet/ppformulanet_preprocess.h"
#include "turbo_ocr/base/env_utils.h"

namespace fs = std::filesystem;

namespace turbo_ocr::formula {

namespace {
constexpr int S = kFormulaInputSize;  // 384

// Extract the clean content tokens from one fused-graph output row [L]: trim at
// the learned EOS, drop BOS(0)/pad(1). `seq` is cleared and refilled (reused
// across crops to avoid per-crop reallocation). Single source of truth so the
// single-crop and batched paths decode the IDENTICAL token sequence.
inline void extract_content_seq(const int64_t *row, int64_t L, int64_t eos,
                                std::vector<int64_t> &seq) {
  seq.clear();
  for (int64_t j = 0; j < L; ++j) {
    const int64_t t = row[j];
    if (t == eos) break;
    if (t != 0 && t != 1) seq.push_back(t);
  }
}
}  // namespace

bool OrtFormulaRecognizer::load(const std::string &model_path,
                                const std::string &tokenizer_json) {
  fs::path mp(model_path);
  fs::path onnx = fs::is_directory(mp) ? (mp / "inference_trt.onnx") : mp;
  if (!fs::exists(onnx)) {
    std::cerr << "[CpuFormula] model not found: " << onnx << '\n';
    return false;
  }
  if (!fused_.load_cpu(onnx.string())) {
    std::cerr << "[CpuFormula] fused graph load_cpu failed: " << onnx << '\n';
    return false;
  }
  tok_ = FormulaTokenizer::load(tokenizer_json);
  if (!tok_) {
    TOCR_LOG_ERROR("formula tokenizer load failed", "path", tokenizer_json);
    return false;
  }
  ready_ = true;
  // Success is INFO: a healthy stage announcing itself on stderr is
  // noise in a library, and this one printed on every construction.
  TOCR_LOG_INFO("formula CPU decode path ready (fused graph, ORT CPU EP)");
  return true;
}

std::string OrtFormulaRecognizer::recognize(const cv::Mat &bgr_crop) {
  if (!ready_ || bgr_crop.empty()) return {};

  // Reused per-thread scratch (mirrors the preprocess/GPU siblings): the encoder
  // input and the content-token buffer persist across calls, so a hot single-crop
  // loop does no per-call heap allocation. thread_local keeps concurrent callers
  // race-free (each thread also owns the preprocess's thread_local scratch).
  thread_local std::vector<float> in;
  thread_local std::vector<int64_t> seq;
  in.resize((size_t)S * S);

  const cv::Mat cont = bgr_crop.isContinuous() ? bgr_crop : bgr_crop.clone();
  formula_preprocess_one(cont.ptr<uint8_t>(), cont.cols, cont.rows, in.data());

  std::vector<int64_t> flat;
  int64_t L = 0, rows = 0;
  if (!fused_.run_tokens("x", "fetch_name_0", in.data(), 1, flat, L, rows) ||
      rows < 1)
    return {};

  extract_content_seq(flat.data(), L, tok_->eos_id(), seq);
  std::string latex = tok_->decode(seq, /*post_process=*/false);
  static const bool drop_collapse = env::env_present("PPFNS_DROP_COLLAPSE");
  if (drop_collapse && formula_is_mode_collapsed(seq, latex)) return {};
  return latex;
}

std::vector<std::string>
OrtFormulaRecognizer::recognize_regions(const cv::Mat &page,
                                        const std::vector<Box> &boxes) {
  std::vector<std::string> out;
  if (!ready_ || page.empty() || boxes.empty()) return out;
  out.resize(boxes.size());

  // Chunked batched decode of the fused graph (matches the GPU cpu_ branch /
  // PPFNS_CHUNK semantics): smaller batches keep the encoder's batched conv
  // bit-matched to the per-crop reference while amortizing the AR Loop.
  const int N = (int)boxes.size();
  static const int chunk = env::env_int("PPFNS_CHUNK", 8, 1, 32);
  static const bool drop_collapse = env::env_present("PPFNS_DROP_COLLAPSE");
  const int64_t EOS = tok_->eos_id();

  // Per-thread scratch reused across chunks AND across calls (no per-chunk/per-crop
  // heap churn): the encoder-input batch, one contiguous crop buffer, and the
  // decode/token buffers. formula_preprocess_one always writes the full [S,S]
  // output per crop (image + PAD_VAL letterbox), so no zero-fill of host_in is
  // needed — every element read by ORT is overwritten first.
  thread_local std::vector<float> host_in;   // [B,1,S,S]
  thread_local std::vector<uint8_t> crop_buf;
  thread_local std::vector<int64_t> flat;
  thread_local std::vector<int64_t> seq;

  const uint8_t *const page_data = page.data;
  const size_t page_step = page.step;

  for (int s0 = 0; s0 < N; s0 += chunk) {
    const int B = std::min(chunk, N - s0);
    const size_t need = (size_t)B * S * S;
    if (host_in.size() < need) host_in.resize(need);
    float *const in_base = host_in.data();

    for (int i = 0; i < B; ++i) {
      const auto cr = clamped_crop_rect(boxes[s0 + i], page.cols, page.rows);
      // NOTE: the boxes arrive PRE-PADDED by the shared dispatch
      // (dispatch_formulas_ expands each layout box by FORMULA_CROP_PAD) —
      // padding here as well would double it and pull neighbouring glyphs in.
      const int x0 = cr[0], y0 = cr[1], w = cr[2], h = cr[3];
      // Contiguous BGR8 sub-image via row memcpy (byte-identical to page(roi).clone())
      // into the reused buffer, avoiding a per-crop cv::Mat allocation.
      crop_buf.resize((size_t)w * h * 3);
      const uint8_t *sp = page_data + (size_t)y0 * page_step + (size_t)x0 * 3;
      for (int r = 0; r < h; ++r)
        std::memcpy(crop_buf.data() + (size_t)r * w * 3, sp + (size_t)r * page_step,
                    (size_t)w * 3);
      formula_preprocess_one(crop_buf.data(), w, h, in_base + (size_t)i * S * S);
    }

    int64_t L = 0, rows = 0;
    if (!fused_.run_tokens("x", "fetch_name_0", in_base, B, flat, L, rows)) {
      std::cerr << "[CpuFormula] fused decode failed (chunk s0=" << s0 << ")\n";
      continue;  // leave these slots empty
    }
    // Decode the rows the graph RETURNED. `flat` is sized rows*L; iterating
    // to B when rows < B scanned for EOS in heap memory past the vector's end
    // and emitted plausible-looking garbage LaTeX for real formula regions.
    const int n_dec = (int)std::min<int64_t>(B, rows);
    if (n_dec < B)
      std::cerr << "[CpuFormula] graph returned " << rows << " rows for a "
                << B << "-crop chunk (s0=" << s0
                << "); the shorted slots stay empty\n";
    for (int i = 0; i < n_dec; ++i) {
      extract_content_seq(flat.data() + (size_t)i * L, L, EOS, seq);
      std::string latex = tok_->decode(seq, /*post_process=*/false);
      out[s0 + i] = (drop_collapse && formula_is_mode_collapsed(seq, latex))
                        ? std::string()
                        : latex;
    }
  }
  return out;
}

}  // namespace turbo_ocr::formula
