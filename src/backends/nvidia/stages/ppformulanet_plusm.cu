// PP-FormulaNet_plus-M continuous (iteration-level) batched decode: the
// production whole-page path (decode_plusm_page) and its slot-based decode
// loop over a pre-encoded crop queue (decode_continuous_plusm).

#include "nvidia/stages/ppformulanet_ort.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "turbo_ocr/base/geometry/box.h"
#include "nvidia/stages/ppformulanet_internal.cuh"

namespace turbo_ocr::formula {

namespace {
// Accumulate the just-written self-KV slot at per-row position pos[b] from src (kb_out)
// into dst (kb), for every (layer,row,head,dim). Replaces the full-buffer ping-pong swap
// so the step's KV input/output addresses stay FIXED across iterations — the precondition
// for ORT CUDA-graph capture/replay. Only LY*B*H*Dh elements move (one position), so it is
// far cheaper than copying the whole 1056-wide buffer.
__global__ void accum_kv_slot(float *dst, const float *src, const int64_t *pos,
                              int LY, int B, int Hh, int MAXL, int DHh) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = LY * B * Hh * DHh;
  if (idx >= total) return;
  int d = idx % DHh; int t = idx / DHh;
  int h = t % Hh; t /= Hh;
  int b = t % B; t /= B;
  int l = t;
  const int64_t p = pos[b];
  const size_t off = ((((size_t)l * B + b) * Hh + h) * (size_t)MAXL + (size_t)p) * DHh + d;
  dst[off] = src[off];
}
}  // namespace

// plus-M continuous (iteration-level) batched decode over a chosen KV bucket. `step` +
// kA/kB/vA/vB + `maxlen` select the window (the 384 step_short_ for the common case, or the
// 1056 step_ for the long tail). `queue` is the crop indices (into ck_all_, of which n_total
// were prepped) to decode; out[crop] receives each finished sequence; a crop that reaches
// `maxlen` tokens without EOS is appended to `overflow` (its partial out[] cleared) for the
// caller to re-decode in a larger bucket. Fixed KV addresses (no ping-pong; accum_kv_slot
// writes back only the new pos slot). out is NOT resized here — the caller pre-sizes it.
bool PPFormulaNetOrt::decode_continuous_plusm(OrtSession &step, float *kA, float *kB,
                                              float *vA, float *vB, int maxlen, int Bslots,
                                              int n_total, const std::vector<int> &queue,
                                              std::vector<std::vector<int64_t>> &out,
                                              std::vector<int> &overflow, bool final_bucket) {
  const int64_t EOS = tok_->eos_id();
  const int LY = PM_LAYERS, DH = PM_Dh;
  if (Bslots > PM_MAX_BATCH) Bslots = PM_MAX_BATCH;  // ORT step throws at MAX_B=32
  const int QN = (int)queue.size();
  const int64_t Bi = Bslots;
  if (QN <= 0 || Bslots <= 0) return true;

  const size_t blk = (size_t)H * CTX * DH;  // one (layer,crop) cross-KV block
  auto load_cross = [&](int crop, int b) {  // ck_all_/cv_all_[*,crop] -> slot column b
    cudaMemcpy2DAsync(d_ck_ + (size_t)b * blk, (size_t)Bslots * blk * sizeof(float),
                      ck_all_ + (size_t)crop * blk, (size_t)n_total * blk * sizeof(float),
                      blk * sizeof(float), LY, cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpy2DAsync(d_cv_ + (size_t)b * blk, (size_t)Bslots * blk * sizeof(float),
                      cv_all_ + (size_t)crop * blk, (size_t)n_total * blk * sizeof(float),
                      blk * sizeof(float), LY, cudaMemcpyDeviceToDevice, stream_);
  };

  std::vector<int> slot_crop(Bslots, -1);
  std::vector<int64_t> hpos(Bslots, 0), htok(Bslots, PM_START), hnext(Bslots);
  int next_q = 0, active = 0;
  for (int b = 0; b < Bslots && next_q < QN; ++b) {
    slot_crop[b] = queue[next_q]; load_cross(queue[next_q], b); hpos[b] = 0; htok[b] = PM_START;
    ++next_q; ++active;
  }

  // Fixed KV addresses (no ping-pong): the step reads kb=kA/vb=vA and writes the full KV to
  // kb_out=kB/vb_out=vB; accum_kv_slot copies ONLY the new pos slot back into kA/vA.
  //
  // LOAD-BEARING INVARIANT (slot reuse): when a finished crop is evicted and a new one
  // starts in the same slot (hpos=0), the slot's self-KV is NOT cleared — positions
  // beyond the new crop's hpos still hold the PREVIOUS crop's K/V. This is correct only
  // because the exported step masks self-attention causally to [0, pos]: freshly written
  // slots shadow the stale ones before they can ever be attended. If a future export
  // attends the full maxlen window, slot refill MUST memset the slot's KV first.
  step.reset_graph();  // re-bind: Bslots + maxlen (shapes) differ across calls/buckets
  std::vector<OrtTensor> ins = {
      {"tokens", d_tok_, {Bi, 1}, true}, {"pos", d_pos_, {Bi}, true},
      {"kb", kA, {LY, Bi, H, maxlen, DH}, false}, {"vb", vA, {LY, Bi, H, maxlen, DH}, false},
      {"ck", d_ck_, {LY, Bi, H, CTX, DH}, false}, {"cv", d_cv_, {LY, Bi, H, CTX, DH}, false}};
  std::vector<OrtTensor> outs = {
      {"logits", d_log_, {Bi, 1, VOCAB}, false}, {"next_token", d_next_, {Bi, 1}, true},
      {"kb_out", kB, {LY, Bi, H, maxlen, DH}, false}, {"vb_out", vB, {LY, Bi, H, maxlen, DH}, false}};

  bool ok = true;
  const int kvt = LY * Bslots * H * DH;
  long guard = 0, guard_max = (long)QN * maxlen + maxlen;
  while (active > 0 && guard++ < guard_max) {
    cudaMemcpyAsync(d_pos_, hpos.data(), (size_t)Bslots * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_tok_, htok.data(), (size_t)Bslots * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
    if (!step.run_graph(ins, outs)) {
      std::cerr << "[plusm-cont] step.run_graph FAILED Bslots=" << Bslots << " maxlen=" << maxlen
                << " cuda=" << cudaGetErrorString(cudaGetLastError()) << '\n';
      ok = false; break;
    }
    accum_kv_slot<<<(kvt + 255) / 256, 256, 0, stream_>>>(kA, kB, d_pos_, LY, Bslots, H, maxlen, DH);
    accum_kv_slot<<<(kvt + 255) / 256, 256, 0, stream_>>>(vA, vB, d_pos_, LY, Bslots, H, maxlen, DH);
    cudaMemcpyAsync(hnext.data(), d_next_, (size_t)Bslots * sizeof(int64_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    for (int b = 0; b < Bslots; ++b) {
      if (slot_crop[b] < 0) { htok[b] = PM_START; continue; }
      const int crop = slot_crop[b];
      const int64_t t = hnext[b];
      const bool eos = (t == EOS);
      const bool over = !eos && (hpos[b] + 1 >= maxlen);  // hit the bucket cap without EOS
      // keep the token unless this is a non-final bucket that will re-decode the whole crop.
      if (!eos && t != 0 && t != 1 && (!over || final_bucket)) out[crop].push_back(t);
      if (eos || over) {
        if (over && !final_bucket) { out[crop].clear(); overflow.push_back(crop); }  // re-decode bigger
        --active;
        if (next_q < QN) {
          slot_crop[b] = queue[next_q]; load_cross(queue[next_q], b); hpos[b] = 0; htok[b] = PM_START;
          ++next_q; ++active;
        } else { slot_crop[b] = -1; hpos[b] = 0; htok[b] = PM_START; }
      } else { htok[b] = t; hpos[b] += 1; }
    }
  }
  cudaStreamSynchronize(stream_);
  if (cudaPeekAtLastError() != cudaSuccess) ok = false;
  if (active > 0) {
    // Guard tripped with crops still decoding: their outputs are truncated.
    // Never report that as success (no-silent-failure).
    std::cerr << "[plusm-cont] iteration guard tripped with " << active
              << " crop(s) still active — output truncated\n";
    ok = false;
  }
  return ok;
}

void PPFormulaNetOrt::decode_plusm_page(int N, const std::vector<Box> &boxes, const GpuImage &page,
                                        bool drop_collapse, std::vector<FormulaEngineResult> &out) {
  for (int g0 = 0; g0 < N; g0 += PM_MAX_N) {
    const int GN = std::min(PM_MAX_N, N - g0);
    bool eok = true;
    // Preprocess + encode the group's GN crops into d_mem_all_, in MAX_B-sized batches
    // (host_in_/d_x_ hold at most MAX_B crops). Encoder is per-crop (batch=1) to match ref.
    for (int sb = 0; sb < GN && eok; sb += MAX_B) {
      const int SB = std::min(MAX_B, GN - sb);
#pragma omp parallel for schedule(dynamic)
      for (int i = 0; i < SB; ++i)
        preprocess_crop(boxes[g0 + sb + i], page, host_in_.data() + (size_t)i * S * S);
      cudaMemcpyAsync(d_x_, host_in_.data(), (size_t)SB * S * S * sizeof(float),
                      cudaMemcpyHostToDevice, stream_);
      eok = encode_crops(SB, d_mem_all_ + (size_t)sb * CTX * 2048);
      cudaStreamSynchronize(stream_);  // H2D + encode complete before host_in_ is reused
    }
    const int64_t GNi = GN;
    OrtTensor pin{"memory", d_mem_all_, {GNi, CTX, 2048}, false};
    OrtTensor pck{"ck", ck_all_, {PM_LAYERS, GNi, H, CTX, PM_Dh}, false};
    OrtTensor pcv{"cv", cv_all_, {PM_LAYERS, GNi, H, CTX, PM_Dh}, false};
    std::vector<std::vector<int64_t>> seqs(GN);
    std::vector<int> queue(GN), overflow, ovf2;
    for (int i = 0; i < GN; ++i) queue[i] = i;
    bool okg = eok && prep_.run({pin}, {pck, pcv});
    if (okg) {
      // Common case: decode in the 384-KV bucket; formulas that exceed 384 tokens overflow
      // and re-decode in the full 1056 bucket (the final cap). If the 384 graph is absent,
      // everything runs in the 1056 bucket directly.
      const bool have_short = step_short_.ready();
      okg = decode_continuous_plusm(have_short ? step_short_ : step_,
                                    have_short ? kA384_ : kA_, have_short ? kB384_ : kB_,
                                    have_short ? vA384_ : vA_, have_short ? vB384_ : vB_,
                                    have_short ? PM_MAXLEN_S : MAXLEN,
                                    std::min(GN, PM_MAX_BATCH), GN, queue, seqs, overflow,
                                    /*final_bucket=*/!have_short);
      if (okg && !overflow.empty())
        okg = decode_continuous_plusm(step_, kA_, kB_, vA_, vB_, MAXLEN,
                                      std::min((int)overflow.size(), PM_MAX_BATCH), GN, overflow,
                                      seqs, ovf2, /*final_bucket=*/true);
    }
    if (!okg) std::cerr << "[PPFormulaNetOrt] plus-M page decode failed at group " << g0 << '\n';
    for (int i = 0; i < GN; ++i) {
      if (!okg) { out[g0 + i].ok = false; continue; }
      std::string latex = tok_->decode(seqs[i], /*post_process=*/false);
      out[g0 + i].latex =
          (drop_collapse && formula_is_mode_collapsed(seqs[i], latex)) ? std::string() : latex;
      out[g0 + i].token_count = seqs[i].size();
      out[g0 + i].hit_eos = !seqs[i].empty();
      out[g0 + i].ok = true;
    }
  }
}

}  // namespace turbo_ocr::formula
