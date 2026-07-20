// PP-FormulaNet FAST host-loop lockstep decode paths (-S MTP 3-token step and
// the plus-M single-token step), plus their on-device argmax/EOS kernels.

#include "turbo_ocr/formula/ppformulanet/ppformulanet_ort.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "ppformulanet_internal.cuh"

namespace turbo_ocr::formula {

namespace {
// Reduce (val,idx) across a warp keeping the LOWEST lane index on exact-value ties
// (strict-greater replaces, so the lower lane survives). __shfl_down_sync keeps the
// whole reduction in registers — no shared traffic, no per-level __syncthreads.
__device__ __forceinline__ void warp_argmax(float &v, int &i) {
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) {
    float ov = __shfl_down_sync(0xffffffffu, v, s);
    int oi = __shfl_down_sync(0xffffffffu, i, s);
    if (ov > v) { v = ov; i = oi; }
  }
}
// argmax over VOCAB per FP32 logit row. Coalesced strided scan (consecutive lanes read
// consecutive logits) -> per-lane max, then a two-level warp-shuffle reduction (intra-warp,
// then across the <=32 warp winners in warp 0). For every input with a STRICT row max
// (i.e. real model logits) this returns exactly that max's index. Tie-break caveat: on
// an EXACT FP32 tie the survivor is the tied index with the lowest THREAD id (lowest
// i % 256), NOT the lowest vocab index a serial numpy/paddle argmax returns — e.g. a
// tie between logits[100] and logits[257] emits 257 (thread 1), not 100. Exact cross-
// stripe FP32 ties do not occur in real logits, so decoded output is unchanged in
// practice (empirically bit-identical formula CDM) — but "bit-exact to reference
// argmax" is guaranteed only for strict maxima.
__global__ void argmax_kernel(const float *lg, int64_t *out, int rows, int V) {
  int row = blockIdx.x; if (row >= rows) return;
  const float *p = lg + static_cast<size_t>(row) * V;
  float m = -1e30f; int mi = 0;
  for (int i = threadIdx.x; i < V; i += blockDim.x) { float v = p[i]; if (v > m) { m = v; mi = i; } }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  warp_argmax(m, mi);  // lane 0 of each warp now holds that warp's winner
  __shared__ float sm[32]; __shared__ int si[32];
  if (lane == 0) { sm[warp] = m; si[warp] = mi; }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    m = lane < nwarps ? sm[lane] : -1e30f;
    mi = lane < nwarps ? si[lane] : 0;
    warp_argmax(m, mi);
    if (lane == 0) out[row] = mi;
  }
}
// On-device EOS termination: scan the newly-decoded steps [s0,s1) of the flat token
// history (all[step*B*Ks + b*Ks + j]) and set a STICKY per-row done flag when any of the
// row's Ks slots equals EOS. The host then D2Hs only the B flags (not the growing token
// history) and does no per-token rescan — it just reads the flags to decide termination.
__global__ void scan_eos_range(const int64_t *all, unsigned char *done, int B, int Ks,
                               int64_t EOS, int s0, int s1) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B || done[b]) return;
  for (int s = s0; s < s1; ++s) {
    const int64_t *row = all + ((size_t)s * B + b) * Ks;
    for (int j = 0; j < Ks; ++j)
      if (row[j] == EOS) { done[b] = 1; return; }
  }
}
__global__ void fill_val(int64_t *p, int64_t v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}
}  // namespace

// FAST path: lockstep-batched static-KV decode. Build the step bindings once and only
// retarget the ping-pong KV + pos pointers each iteration; all async on stream_ with a
// deferred EOS check every CHECK steps. Fills clean content token rows (BOS/pad/EOS
// stripped) per crop. Returns false if a step/argmax/CUDA error truncated the decode
// (the partial rows are still returned, but the caller must mark the crops failed).
bool PPFormulaNetOrt::decode_chunk(int B, std::vector<std::vector<int64_t>> &out) {
  const int64_t EOS = tok_->eos_id(), Bi = B;
  size_t kvB = (size_t)2 * B * H * MAXLEN * Dh * sizeof(float);
  cudaMemsetAsync(kA_, 0, kvB, stream_); cudaMemsetAsync(vA_, 0, kvB, stream_);
  cudaMemsetAsync(d_tok_, 0, (size_t)B * 3 * sizeof(int64_t), stream_);
  float *kin = kA_, *kout = kB_, *vin = vA_, *vout = vB_;
  std::vector<OrtTensor> ins = {
      {"tokens", d_tok_, {Bi, 3}, true}, {"pos", d_pos_, {1}, true},
      {"kb", kin, {2, Bi, H, MAXLEN, Dh}, false}, {"vb", vin, {2, Bi, H, MAXLEN, Dh}, false},
      {"ck", d_ck_, {2, Bi, H, CTX, Dh}, false}, {"cv", d_cv_, {2, Bi, H, CTX, Dh}, false}};
  std::vector<OrtTensor> outs = {
      {"logits", d_log_, {Bi, 3, VOCAB}, false}, {"kb_out", kout, {2, Bi, H, MAXLEN, Dh}, false},
      {"vb_out", vout, {2, Bi, H, MAXLEN, Dh}, false}};
  // d_all_ is never zeroed, so `last` MUST bound the steps actually written this
  // call. On either error break only steps 0..it-1 are valid -> set last=it; the
  // clean all-done break sets last=it+1. ok=false on any error so the caller can
  // mark the chunk's crops failed instead of decoding stale tokens as success.
  cudaMemsetAsync(d_done_, 0, (size_t)B, stream_);
  std::vector<char> done(B, 0); std::vector<int64_t> hall; int last = MAXIT; bool ok = true;
  int checked = 0;  // steps already scanned for EOS on device (sticky d_done_)
  for (int it = 0; it < MAXIT; ++it) {
    ins[1].data = d_pos_ + it; ins[2].data = kin; ins[3].data = vin;
    outs[1].data = kout; outs[2].data = vout;
    if (!step_.run(ins, outs)) {
      std::cerr << "[PPFormulaNetOrt] step run failed it=" << it << '\n';
      last = it; ok = false; break;
    }
    argmax_kernel<<<B * 3, 256, 0, stream_>>>(d_log_, d_next_, B * 3, VOCAB);
    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
      std::cerr << "[PPFormulaNetOrt] argmax launch failed it=" << it << ": " << cudaGetErrorString(e) << '\n';
      last = it; ok = false; break;
    }
    cudaMemcpyAsync(d_all_ + (size_t)it * B * 3, d_next_, (size_t)B * 3 * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(d_tok_, d_next_, (size_t)B * 3 * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    std::swap(kin, kout); std::swap(vin, vout);
    if ((it + 1) % CHECK == 0) {
      // On-device EOS scan of the new steps sets sticky per-row flags; only B bytes cross
      // the bus (vs the whole growing token history) and the host does no per-token rescan.
      scan_eos_range<<<(B + 63) / 64, 64, 0, stream_>>>(d_all_, d_done_, B, 3, EOS, checked, it + 1);
      cudaStreamSynchronize(stream_);
      if (cudaError_t e = cudaMemcpy(done.data(), d_done_, (size_t)B, cudaMemcpyDeviceToHost);
          e != cudaSuccess) {
        std::cerr << "[PPFormulaNetOrt] periodic done D2H failed it=" << it
                  << ": " << cudaGetErrorString(e) << '\n';
        last = it + 1; ok = false; break;  // don't early-stop on stale tokens
      }
      checked = it + 1;
      int nd = 0; for (int b = 0; b < B; ++b) nd += done[b] ? 1 : 0;
      if (nd == B) { last = it + 1; break; }
    }
  }
  cudaStreamSynchronize(stream_);
  // The async memset/memcpy launches in the AR loop don't check their returns;
  // surface any error the sync just exposed (e.g. an illegal access) rather than
  // decoding from a corrupt d_all_.
  if (cudaError_t e = cudaPeekAtLastError(); e != cudaSuccess) {
    std::cerr << "[PPFormulaNetOrt] decode_chunk CUDA error: " << cudaGetErrorString(e) << '\n';
    ok = false;
  }
  size_t n = (size_t)last * B * 3; hall.resize(n);
  if (cudaError_t e = cudaMemcpy(hall.data(), d_all_, n * sizeof(int64_t),
                                 cudaMemcpyDeviceToHost); e != cudaSuccess) {
    std::cerr << "[PPFormulaNetOrt] final D2H copy failed: " << cudaGetErrorString(e)
              << " — failing chunk instead of decoding stale tokens\n";
    out.assign(B, {});  // empty seqs + return false -> caller marks crops failed
    return false;
  }
  out.assign(B, {});
  for (int b = 0; b < B; ++b) {
    bool stop = false;
    for (int s = 0; s < last && !stop; ++s)
      for (int j = 0; j < 3; ++j) {
        int64_t t = hall[(size_t)s * B * 3 + (size_t)b * 3 + j];
        if (t == EOS) { stop = true; break; }
        if (t != 0 && t != 1) out[b].push_back(t);
      }
  }
  return ok;
}

// plus-M FAST path: lockstep static-KV decode on the 6-layer MBart step graph.
// decoder_start = </s> (PM_START=2); decoder_step.onnx emits next_token (greedy argmax)
// in-graph, so we thread next_token->tokens device-to-device and only D2H the tiny [B]
// token row every CHECK steps for the EOS-eviction check. pos is shared per step
// (lockstep -> all rows at pos=it). Returns false on a truncating step/CUDA error.
bool PPFormulaNetOrt::decode_chunk_plusm(int B, std::vector<std::vector<int64_t>> &out) {
  const int64_t EOS = tok_->eos_id(), Bi = B;
  const int LY = PM_LAYERS, DH = PM_Dh;
  size_t kvB = (size_t)LY * B * H * MAXLEN * DH * sizeof(float);
  cudaMemsetAsync(kA_, 0, kvB, stream_);
  cudaMemsetAsync(vA_, 0, kvB, stream_);
  fill_val<<<(B + 63) / 64, 64, 0, stream_>>>(d_tok_, PM_START, B);  // decoder start = </s>
  float *kin = kA_, *kout = kB_, *vin = vA_, *vout = vB_;
  std::vector<OrtTensor> ins = {
      {"tokens", d_tok_, {Bi, 1}, true}, {"pos", d_pos_, {Bi}, true},
      {"kb", kin, {LY, Bi, H, MAXLEN, DH}, false}, {"vb", vin, {LY, Bi, H, MAXLEN, DH}, false},
      {"ck", d_ck_, {LY, Bi, H, CTX, DH}, false}, {"cv", d_cv_, {LY, Bi, H, CTX, DH}, false}};
  std::vector<OrtTensor> outs = {
      {"logits", d_log_, {Bi, 1, VOCAB}, false}, {"next_token", d_next_, {Bi, 1}, true},
      {"kb_out", kout, {LY, Bi, H, MAXLEN, DH}, false}, {"vb_out", vout, {LY, Bi, H, MAXLEN, DH}, false}};
  cudaMemsetAsync(d_done_, 0, (size_t)B, stream_);
  std::vector<char> done(B, 0); std::vector<int64_t> hall; int last = PM_MAXIT; bool ok = true;
  int checked = 0;  // steps already scanned for EOS on device (sticky d_done_)
  for (int it = 0; it < PM_MAXIT; ++it) {
    fill_val<<<(B + 63) / 64, 64, 0, stream_>>>(d_pos_, it, B);  // lockstep: all rows at pos=it
    ins[2].data = kin; ins[3].data = vin; outs[2].data = kout; outs[3].data = vout;
    if (!step_.run(ins, outs)) {
      std::cerr << "[PPFormulaNetOrt] plus-M step run failed it=" << it << '\n';
      last = it; ok = false; break;
    }
    cudaMemcpyAsync(d_all_ + (size_t)it * B, d_next_, (size_t)B * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(d_tok_, d_next_, (size_t)B * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream_);
    std::swap(kin, kout); std::swap(vin, vout);
    if ((it + 1) % CHECK == 0) {
      // On-device EOS scan (see decode_chunk): D2H only the B sticky flags, no host rescan.
      scan_eos_range<<<(B + 63) / 64, 64, 0, stream_>>>(d_all_, d_done_, B, 1, EOS, checked, it + 1);
      cudaStreamSynchronize(stream_);
      if (cudaMemcpy(done.data(), d_done_, (size_t)B, cudaMemcpyDeviceToHost) != cudaSuccess) {
        last = it + 1; ok = false; break;
      }
      checked = it + 1;
      int nd = 0; for (int b = 0; b < B; ++b) nd += done[b] ? 1 : 0;
      if (nd == B) { last = it + 1; break; }
    }
  }
  cudaStreamSynchronize(stream_);
  if (cudaPeekAtLastError() != cudaSuccess) ok = false;
  size_t n = (size_t)last * B; hall.resize(n);
  if (cudaMemcpy(hall.data(), d_all_, n * sizeof(int64_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
    out.assign(B, {});
    return false;
  }
  out.assign(B, {});
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < last; ++s) {
      int64_t t = hall[(size_t)s * B + b];
      if (t == EOS) break;
      if (t != 0 && t != 1) out[b].push_back(t);  // drop BOS(0)/pad(1)
    }
  return ok;
}

}  // namespace turbo_ocr::formula
