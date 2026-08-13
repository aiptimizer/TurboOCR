// Dev/CI gate harness for the PP-FormulaNet FAST backends (NOT used by the
// server): decodes the 30 gate crops and dumps tokens/latex JSON + crops/s.
// Driven only by tools/checks/plusm_selftest.cpp.

#include "nvidia/stages/ppformulanet_ort.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "nvidia/stages/ppformulanet_internal.cuh"

namespace fs = std::filesystem;

namespace turbo_ocr::formula {

bool PPFormulaNetOrt::gate_bench(const std::string &gate_dir) {
  if (!fast_ || !step_.ready() || !tok_) {
    std::cerr << "[gate_bench] requires a loaded FAST backend + tokenizer\n";
    return false;
  }
  const fs::path eval(gate_dir);
  const int B = 30;
  std::vector<float> crops((size_t)B * S * S);
  std::ifstream fe((eval / "en15_crops.bin").string(), std::ios::binary);
  std::ifstream fz((eval / "zh15_crops.bin").string(), std::ios::binary);
  if (!fe || !fz) {
    std::cerr << "[gate_bench] gate crops not found under " << eval.string() << '\n';
    return false;
  }
  fe.read(reinterpret_cast<char *>(crops.data()), (size_t)15 * S * S * sizeof(float));
  fz.read(reinterpret_cast<char *>(crops.data() + (size_t)15 * S * S), (size_t)15 * S * S * sizeof(float));
  cudaMemcpyAsync(d_x_, crops.data(), crops.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
  const int64_t Bi = B, LY = plusm_ ? PM_LAYERS : 2, DH = plusm_ ? PM_Dh : Dh;
  auto enc_prep_decode = [&](std::vector<std::vector<int64_t>> &seqs) -> bool {
    OrtTensor pin{"memory", d_mem_, {Bi, CTX, 2048}, false};
    OrtTensor pck{"ck", d_ck_, {LY, Bi, H, CTX, DH}, false};
    OrtTensor pcv{"cv", d_cv_, {LY, Bi, H, CTX, DH}, false};
    if (!encode_crops(B, d_mem_) || !prep_.run({pin}, {pck, pcv})) return false;
    return plusm_ ? decode_chunk_plusm(B, seqs) : decode_chunk(B, seqs);
  };
  auto esc = [](const std::string &s) {
    std::string o = "\"";
    for (char c : s) {
      if (c == '\\' || c == '"') { o += '\\'; o += c; }
      else if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t"; else if (c == '\r') {}
      else o += c;
    }
    o += '"'; return o;
  };
  auto dump_tokens = [&](const std::string &path, const std::vector<std::vector<int64_t>> &s) {
    std::ofstream o(path); o << '[';
    for (int b = 0; b < (int)s.size(); ++b) { o << (b ? ",[" : "[");
      for (size_t j = 0; j < s[b].size(); ++j) o << (j ? "," : "") << s[b][j]; o << ']'; }
    o << "]\n";
  };
  std::vector<std::vector<int64_t>> seqs;
  if (!enc_prep_decode(seqs)) { std::cerr << "[gate_bench] enc/prep/decode failed\n"; return false; }
  const std::string base = "/tmp/cpp_" + label_;
  dump_tokens(base + "_tokens.json", seqs);
  { std::ofstream o(base + "_latex.json"); o << '[';
    for (int b = 0; b < B; ++b) o << (b ? "," : "") << esc(tok_->decode(seqs[b], /*post_process=*/false));
    o << "]\n"; }
  // plus-M: also validate the continuous (production) decode path WITH evict/refill (Bslots<B).
  if (plusm_) {
    cudaMemcpyAsync(d_mem_all_, d_mem_, (size_t)B * CTX * 2048 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);
    OrtTensor pin{"memory", d_mem_all_, {Bi, CTX, 2048}, false};
    OrtTensor pck{"ck", ck_all_, {PM_LAYERS, Bi, H, CTX, PM_Dh}, false};
    OrtTensor pcv{"cv", cv_all_, {PM_LAYERS, Bi, H, CTX, PM_Dh}, false};
    // Validate the continuous decode (WITH evict/refill, Bslots<B) in the production bucket
    // (384 step_short_ when present, else 1056).
    const bool hs = step_short_.ready();
    std::vector<std::vector<int64_t>> cseqs(B);
    std::vector<int> q(B), ovf; for (int i = 0; i < B; ++i) q[i] = i;
    if (prep_.run({pin}, {pck, pcv}) &&
        decode_continuous_plusm(hs ? step_short_ : step_, hs ? kA384_ : kA_, hs ? kB384_ : kB_,
                                hs ? vA384_ : vA_, hs ? vB384_ : vB_, hs ? PM_MAXLEN_S : MAXLEN,
                                16, B, q, cseqs, ovf, /*final_bucket=*/!hs))
      dump_tokens(base + "_cont_tokens.json", cseqs);
    // continuous throughput on an N=90 queue, full Bslots — 1056 window vs the 384 bucket.
    const int N = 3 * B;
    for (int r = 0; r < 3; ++r)
      cudaMemcpyAsync(d_mem_all_ + (size_t)r * B * CTX * 2048, d_mem_,
                      (size_t)B * CTX * 2048 * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    OrtTensor pin2{"memory", d_mem_all_, {(int64_t)N, CTX, 2048}, false};
    OrtTensor pck2{"ck", ck_all_, {PM_LAYERS, (int64_t)N, H, CTX, PM_Dh}, false};
    OrtTensor pcv2{"cv", cv_all_, {PM_LAYERS, (int64_t)N, H, CTX, PM_Dh}, false};
    if (prep_.run({pin2}, {pck2, pcv2})) {
      std::vector<int> qn(N); for (int i = 0; i < N; ++i) qn[i] = i;
      auto bench = [&](OrtSession &st, float *ka, float *kb, float *va, float *vb, int ml,
                       const char *tag, bool fin) {
        std::vector<std::vector<int64_t>> sc(N); std::vector<int> ov;
        cudaStreamSynchronize(stream_);
        auto c0 = std::chrono::steady_clock::now();
        const int CR = 5;
        for (int r = 0; r < CR; ++r) { for (auto &s : sc) s.clear(); ov.clear();
          decode_continuous_plusm(st, ka, kb, va, vb, ml, PM_MAX_BATCH, N, qn, sc, ov, fin); }
        cudaStreamSynchronize(stream_);
        double cs = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
        std::cerr << "[gate_bench] " << label_ << " continuous N=" << N << " " << tag << ": "
                  << (CR * N / cs) << " crops/s\n";
      };
      bench(step_, kA_, kB_, vA_, vB_, MAXLEN, "[1056 window]", true);
      if (hs) bench(step_short_, kA384_, kB384_, vA384_, vB384_, PM_MAXLEN_S, "[384 bucket]", false);
    }
  }
  std::cerr << "[gate_bench] " << label_ << " wrote " << base << "_*.json (" << B << " crops)\n";
  const int REPS = 10;
  cudaStreamSynchronize(stream_);
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < REPS; ++r) { std::vector<std::vector<int64_t>> s2; enc_prep_decode(s2); }
  cudaStreamSynchronize(stream_);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cerr << "[gate_bench] " << label_ << " lockstep B=30: " << (REPS * B / sec) << " crops/s\n";
  return true;
}

}  // namespace turbo_ocr::formula
