#include "turbo_ocr/formula/ppformulanet/cpu_formula_recognizer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "turbo_ocr/formula/ppformulanet/ppformulanet_preprocess.h"

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

bool CpuFormulaRecognizer::load(const std::string &model_path,
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
    std::cerr << "[CpuFormula] tokenizer load failed: " << tokenizer_json << '\n';
    return false;
  }
  ready_ = true;
  std::cerr << "[CpuFormula] CPU decode path ready (fused graph, ORT CPUExecutionProvider)\n";
  return true;
}

std::string CpuFormulaRecognizer::recognize(const cv::Mat &bgr_crop) {
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
  int64_t L = 0;
  if (!fused_.run_tokens("x", "fetch_name_0", in.data(), 1, flat, L)) return {};

  extract_content_seq(flat.data(), L, tok_->eos_id(), seq);
  std::string latex = tok_->decode(seq, /*post_process=*/false);
  static const bool drop_collapse = std::getenv("PPFNS_DROP_COLLAPSE") != nullptr;
  if (drop_collapse && formula_is_mode_collapsed(seq, latex)) return {};
  return latex;
}

std::vector<std::string>
CpuFormulaRecognizer::recognize_regions(const cv::Mat &page,
                                        const std::vector<Box> &boxes) {
  std::vector<std::string> out;
  if (!ready_ || page.empty() || boxes.empty()) return out;
  out.resize(boxes.size());

  // Chunked batched decode of the fused graph (matches the GPU cpu_ branch /
  // PPFNS_CHUNK semantics): smaller batches keep the encoder's batched conv
  // bit-matched to the per-crop reference while amortizing the AR Loop.
  const int N = (int)boxes.size();
  static const int chunk = []{
    const char *e = std::getenv("PPFNS_CHUNK");
    int c = e ? std::atoi(e) : 8;
    return c < 1 ? 1 : (c > 32 ? 32 : c);
  }();
  static const bool drop_collapse = std::getenv("PPFNS_DROP_COLLAPSE") != nullptr;
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

    int64_t L = 0;
    if (!fused_.run_tokens("x", "fetch_name_0", in_base, B, flat, L)) {
      std::cerr << "[CpuFormula] fused decode failed (chunk s0=" << s0 << ")\n";
      continue;  // leave these slots empty
    }
    for (int i = 0; i < B; ++i) {
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
