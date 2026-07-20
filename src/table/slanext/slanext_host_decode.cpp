#include "turbo_ocr/table/slanext/slanext_host_decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include "turbo_ocr/table/slanext/slanext_postprocess.h"

// SLANeXt table-structure host decoder: additive-attention GRU AR loop over one
// TRT/ORT encoder feature map [kTenc, kCtx], emitting the HTML-structure token
// sequence + per-cell quad regression. The encoder runs on GPU/TRT by design;
// this small autoregressive loop is host work (sequential data dependence, tiny
// GEMVs) and stays on the CPU.
//
// This TU is the ONE decode implementation for BOTH builds: the CPU backend
// and the GPU SlanextEncSplit call slanext_host_decode() (the GPU target
// compiles this file with the same -ffast-math + -fopenmp-simd source
// properties). Do NOT reorder any floating-point reduction here: reassociating
// a sum can flip a near-tie argmax and shift the structure sequence, which
// table-structure TEDS holds only because the stream is deterministic.

namespace turbo_ocr::table {

namespace {

// out[j] = sum_i x[i] * W[i*OUT + j] (+ b[j]); W is [IN, OUT] row-major.
// Rank-1 accumulate over rows: wr[j]/out[j] are contiguous so the inner j-loop
// auto-vectorizes, and the i-order accumulation per output is preserved.
inline void mv_io(const float *__restrict__ x, const float *__restrict__ W,
                  int IN, int OUT, const float *__restrict__ b,
                  float *__restrict__ out) {
  for (int j = 0; j < OUT; ++j) out[j] = b ? b[j] : 0.0f;
  for (int i = 0; i < IN; ++i) {
    const float xi = x[i];
    const float *wr = W + static_cast<std::size_t>(i) * OUT;
#pragma omp simd
    for (int j = 0; j < OUT; ++j) out[j] += xi * wr[j];
  }
}

// out[k] = sum_m x[m] * W[k*IN + m] (+ b[k]); W is [OUT, IN] row-major.
// Contiguous dot-product reduction; omp simd + -ffast-math fold it into AVX2 FMA
// accumulators (the GRU hidden hot path).
inline void mv_oi(const float *__restrict__ x, const float *__restrict__ W,
                  int OUT, int IN, const float *__restrict__ b,
                  float *__restrict__ out) {
  for (int k = 0; k < OUT; ++k) {
    const float *wr = W + static_cast<std::size_t>(k) * IN;
    float s = b ? b[k] : 0.0f;
#pragma omp simd reduction(+ : s)
    for (int m = 0; m < IN; ++m) s += x[m] * wr[m];
    out[k] = s;
  }
}

inline float sigmoidf(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// Fixed-shape decode workspaces. The recognizer calls slanext_host_decode once
// per table region (from a pipeline pool), so a per-thread instance is reused
// across regions: after the first call every buffer already holds its capacity
// and the steady-state decode allocates nothing. All shapes are compile-time
// constants, so `prepare()` is a set of no-op resizes plus the two state resets
// (h -> 0, growable outputs cleared) that each call requires.
struct DecodeScratch {
  using W = SlanextDecoderWeights;

  std::vector<float> bHp;  // [kTenc, kHidden] = feat @ lin0 (attention keys)
  std::vector<float> h, hp, a, ctx, gin, ghn, s1, st, l1, l8;
  std::vector<float> logits;  // [steps, kVocab] RAW structure logits
  std::vector<float> locs;    // [steps, kLoc]
  std::vector<float> probs;   // [steps, kVocab] softmax of logits

  void prepare() {
    bHp.resize(static_cast<std::size_t>(W::kTenc) * W::kHidden);
    h.assign(W::kHidden, 0.0f);  // GRU hidden state starts at zero each call
    hp.resize(W::kHidden);
    a.resize(W::kTenc);
    ctx.resize(W::kCtx);
    gin.resize(3 * W::kHidden);
    ghn.resize(3 * W::kHidden);
    s1.resize(W::kHidden);
    st.resize(W::kVocab);
    l1.resize(W::kHidden);
    l8.resize(W::kLoc);
    logits.clear();  // keeps capacity; reserve below is a no-op after call 1
    locs.clear();
    logits.reserve(static_cast<std::size_t>(W::kMaxTokens) * W::kVocab);
    locs.reserve(static_cast<std::size_t>(W::kMaxTokens) * W::kLoc);
  }
};

}  // namespace

bool SlanextDecoderWeights::load(const std::string &bin_path) {
  std::ifstream f(bin_path, std::ios::binary);
  if (!f) {
    std::cerr << "[slanext-cpu] cannot open decoder blob: " << bin_path << '\n';
    return false;
  }
  auto rd = [&](std::vector<float> &v, std::size_t n) -> bool {
    v.resize(n);
    f.read(reinterpret_cast<char *>(v.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<bool>(f);
  };
  constexpr int kGruGate = 3 * kHidden;   // 768 (reset|update|candidate)
  constexpr int kGruIn = kCtx + kVocab;   // 146 (ctx + onehot)
  const bool ok = rd(lin0_, kCtx * kHidden) && rd(lin1w_, kHidden * kHidden) &&
                  rd(lin1b_, kHidden) && rd(lin2_, kHidden * 1) &&
                  rd(lin3w_, kHidden * kHidden) && rd(lin3b_, kHidden) &&
                  rd(lin4w_, kHidden * kVocab) && rd(lin4b_, kVocab) &&
                  rd(lin5w_, kHidden * kHidden) && rd(lin5b_, kHidden) &&
                  rd(lin6w_, kHidden * kLoc) && rd(lin6b_, kLoc) &&
                  rd(gw0_, kGruGate * kGruIn) && rd(gw1_, kGruGate * kHidden) &&
                  rd(gb0_, kGruGate) && rd(gb1_, kGruGate);
  if (!ok) {
    std::cerr << "[slanext-cpu] decoder blob truncated: " << bin_path << '\n';
    return false;
  }
  if (f.peek() != std::ifstream::traits_type::eof()) {
    std::cerr << "[slanext-cpu] decoder blob has trailing bytes (wrong file?): "
              << bin_path << '\n';
    return false;
  }
  return true;
}

StructureResult slanext_host_decode(const float *feat,
                                    const SlanextDecoderWeights &W,
                                    const CharDict &dict, int ori_w, int ori_h) {
  using w = SlanextDecoderWeights;
  const int eos = static_cast<int>(dict.eos_idx());

  thread_local DecodeScratch scr;
  scr.prepare();
  auto &bHp = scr.bHp;
  auto &h = scr.h;
  auto &hp = scr.hp;
  auto &a = scr.a;
  auto &ctx = scr.ctx;
  auto &gin = scr.gin;
  auto &ghn = scr.ghn;
  auto &s1 = scr.s1;
  auto &st = scr.st;
  auto &l1 = scr.l1;
  auto &l8 = scr.l8;
  auto &logits = scr.logits;
  auto &locs = scr.locs;

  // bHp[i] = feat[i] @ lin0 (i2h, 96->256, no bias): attention keys, hoisted
  // out of the token loop since the encoder feature is fixed for the sample.
  for (int i = 0; i < w::kTenc; ++i)
    mv_io(feat + static_cast<std::size_t>(i) * w::kCtx, W.lin0_.data(), w::kCtx,
          w::kHidden, nullptr,
          bHp.data() + static_cast<std::size_t>(i) * w::kHidden);

  constexpr int kGruIn = w::kCtx + w::kVocab;  // GRU input width (ctx + onehot)
  // Degenerate-decode guard: cap back-to-back repeats of one structure token.
  // Legit runs (a wide row of <td></td>) top out near max table width, so 96
  // never fires on real tables but truncates a runaway that would otherwise spin
  // to the 501 cap on pathological input.
  constexpr int kMaxRunRepeat = 96;
  int prev = 0;  // sos
  int steps = 0;
  int run_tok = -1, run_len = 0;
  for (int t = 0; t < w::kMaxTokens; ++t) {
    mv_io(h.data(), W.lin1w_.data(), w::kHidden, w::kHidden, W.lin1b_.data(),
          hp.data());
    // additive attention energy a[i] = score(tanh(bHp[i] + hp)); softmax over T
    for (int i = 0; i < w::kTenc; ++i) {
      const float *br = bHp.data() + static_cast<std::size_t>(i) * w::kHidden;
      float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
      for (int j = 0; j < w::kHidden; ++j) acc += std::tanh(br[j] + hp[j]) * W.lin2_[j];
      a[i] = acc;
    }
    float emax = a[0];
    for (int i = 1; i < w::kTenc; ++i) emax = std::max(emax, a[i]);
    float esum = 0.0f;
    for (int i = 0; i < w::kTenc; ++i) { a[i] = std::exp(a[i] - emax); esum += a[i]; }
    const float inv = 1.0f / esum;
    // ctx = sum_i a[i] * feat[i]
    std::fill(ctx.begin(), ctx.end(), 0.0f);
    for (int i = 0; i < w::kTenc; ++i) {
      const float ai = a[i] * inv;
      const float *fr = feat + static_cast<std::size_t>(i) * w::kCtx;
#pragma omp simd
      for (int c = 0; c < w::kCtx; ++c) ctx[c] += ai * fr[c];
    }
    // GRU cell, gate order (reset, update, candidate). The 146-wide input is
    // ctx(96, dense) ++ onehot(prev): fold the onehot to a single column add
    // instead of a 50-wide multiply-by-zeros.
    for (int k = 0; k < 3 * w::kHidden; ++k) {
      const float *wr = W.gw0_.data() + static_cast<std::size_t>(k) * kGruIn;
      float s = W.gb0_[k];
#pragma omp simd reduction(+ : s)
      for (int m = 0; m < w::kCtx; ++m) s += ctx[m] * wr[m];
      gin[k] = s + wr[w::kCtx + prev];  // onehot: the single nonzero input dim
    }
    mv_oi(h.data(), W.gw1_.data(), 3 * w::kHidden, w::kHidden, W.gb1_.data(),
          ghn.data());
    for (int j = 0; j < w::kHidden; ++j) {
      const float r = sigmoidf(gin[j] + ghn[j]);
      const float z = sigmoidf(gin[w::kHidden + j] + ghn[w::kHidden + j]);
      const float n =
          std::tanh(gin[2 * w::kHidden + j] + r * ghn[2 * w::kHidden + j]);
      h[j] = (1.0f - z) * n + z * h[j];
    }
    // structure head (no activation between the two linears) -> raw logits
    mv_io(h.data(), W.lin3w_.data(), w::kHidden, w::kHidden, W.lin3b_.data(),
          s1.data());
    mv_io(s1.data(), W.lin4w_.data(), w::kHidden, w::kVocab, W.lin4b_.data(),
          st.data());
    // argmax on RAW logits (softmax is monotone -> identical argmax, same
    // first-max tie-break); the softmax is deferred out of the critical path.
    int best = 0; float bv = st[0];
    for (int v = 1; v < w::kVocab; ++v) if (st[v] > bv) { bv = st[v]; best = v; }
    if (best == eos) break;  // EOS: stop before storing/loc-head (discarded)
    // Runaway guard: truncate to the clean prefix already emitted.
    if (best == run_tok) {
      if (++run_len >= kMaxRunRepeat) break;
    } else {
      run_tok = best;
      run_len = 1;
    }
    logits.insert(logits.end(), st.begin(), st.end());
    // loc head
    mv_io(h.data(), W.lin5w_.data(), w::kHidden, w::kHidden, W.lin5b_.data(),
          l1.data());
    mv_io(l1.data(), W.lin6w_.data(), w::kHidden, w::kLoc, W.lin6b_.data(),
          l8.data());
    for (int k = 0; k < w::kLoc; ++k) locs.push_back(sigmoidf(l8[k]));
    ++steps;
    prev = best;
  }

  // Softmax the stored logits once (off the per-step critical path) so
  // decode_structure keeps an identical per-token confidence score.
  auto &probs = scr.probs;
  probs.resize(logits.size());
  for (int t = 0; t < steps; ++t) {
    const float *lr = logits.data() + static_cast<std::size_t>(t) * w::kVocab;
    float *pr = probs.data() + static_cast<std::size_t>(t) * w::kVocab;
    float mx = lr[0];
    for (int v = 1; v < w::kVocab; ++v) mx = std::max(mx, lr[v]);
    float sm = 0.0f;
    for (int v = 0; v < w::kVocab; ++v) { pr[v] = std::exp(lr[v] - mx); sm += pr[v]; }
    const float iv = 1.0f / sm;
    for (int v = 0; v < w::kVocab; ++v) pr[v] *= iv;
  }

  return decode_structure(probs.data(), locs.data(),
                          static_cast<std::size_t>(steps),
                          static_cast<std::size_t>(w::kVocab), dict,
                          w::kInputSize, w::kInputSize, ori_w, ori_h);
}

}  // namespace turbo_ocr::table
