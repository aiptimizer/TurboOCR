// NOTE — CUDA device code outside src/backends/nvidia/. This is deliberate and
// documented: see src/backends/nvidia/kernels_cuda/README.md for why the three CUDA sites in this
// tree have not been moved into the NVIDIA backend yet, and what gates it.

#include "nvidia/stages/ppformulanet_ort.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/geometry/box.h"
#include "nvidia/stages/ppformulanet_internal.cuh"

namespace fs = std::filesystem;

namespace turbo_ocr::formula {

namespace {
__global__ void fill_pos(int64_t *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = static_cast<int64_t>(i) * 3;
}
}  // namespace

PPFormulaNetOrt::PPFormulaNetOrt(std::string backend_label)
    : label_(std::move(backend_label)) {}
PPFormulaNetOrt::~PPFormulaNetOrt() noexcept {
  free_buffers();
  if (stream_) cudaStreamDestroy(stream_);
}

bool PPFormulaNetOrt::alloc_buffers() {
  host_in_.assign((size_t)MAX_B * S * S, 0.0f);
  auto m = [](void **p, size_t n) { return cudaMalloc(p, n) == cudaSuccess; };
  if (!m((void **)&d_x_, (size_t)MAX_B * S * S * sizeof(float))) return false;
  if (fast_) {  // static-KV host-loop scratch
    // plus-M: 6 layers / Dh=32 / 1 KV slot per step / pos[B].  PP-FormulaNet-S:
    // 2 layers / Dh=24 / 3 slots per step (MTP) / pos[MAXIT] precomputed (it*3).
    const int LY = plusm_ ? PM_LAYERS : 2, DH = plusm_ ? PM_Dh : Dh;
    const int MIT = plusm_ ? PM_MAXIT : MAXIT, KS = plusm_ ? 1 : 3;
    size_t kv = (size_t)LY * MAX_B * H * MAXLEN * DH, cr = (size_t)LY * MAX_B * H * CTX * DH;
    bool ok = m((void **)&d_mem_, (size_t)MAX_B * CTX * 2048 * sizeof(float))
        && m((void **)&d_ck_, cr * sizeof(float)) && m((void **)&d_cv_, cr * sizeof(float))
        && m((void **)&d_log_, (size_t)MAX_B * 3 * VOCAB * sizeof(float))
        && m((void **)&kA_, kv * sizeof(float)) && m((void **)&kB_, kv * sizeof(float))
        && m((void **)&vA_, kv * sizeof(float)) && m((void **)&vB_, kv * sizeof(float))
        && m((void **)&d_tok_, (size_t)MAX_B * 3 * sizeof(int64_t))
        && m((void **)&d_next_, (size_t)MAX_B * 3 * sizeof(int64_t))
        && m((void **)&d_pos_, (size_t)std::max(MAXIT, MAX_B) * sizeof(int64_t))
        && m((void **)&d_all_, (size_t)MIT * MAX_B * KS * sizeof(int64_t))
        && m((void **)&d_done_, (size_t)MAX_B * sizeof(unsigned char));
    if (!ok) return false;
    if (plusm_) {  // continuous-batch: all-crop encoder memory + pre-computed cross-KV
      size_t cra = (size_t)LY * PM_MAX_N * H * CTX * DH;
      size_t kvs = (size_t)LY * MAX_B * H * PM_MAXLEN_S * DH;  // 384-window self-KV bucket
      if (!m((void **)&d_mem_all_, (size_t)PM_MAX_N * CTX * 2048 * sizeof(float))
          || !m((void **)&ck_all_, cra * sizeof(float))
          || !m((void **)&cv_all_, cra * sizeof(float))
          || !m((void **)&kA384_, kvs * sizeof(float)) || !m((void **)&kB384_, kvs * sizeof(float))
          || !m((void **)&vA384_, kvs * sizeof(float)) || !m((void **)&vB384_, kvs * sizeof(float)))
        return false;
    } else {  // -S precomputes pos=it*3; plus-M fills pos=it per step in-loop
      fill_pos<<<(MAXIT + 63) / 64, 64, 0, stream_>>>(d_pos_, MAXIT);
      if (cudaGetLastError() != cudaSuccess) return false;
    }
  }
  // Sync OUR stream only. cudaDeviceSynchronize would be illegal while any
  // other pipeline's CUDA-graph capture is in flight (device-wide sync against
  // an active capture fails regardless of capture mode) — and pipelines warm up
  // concurrently, so a formula load racing another pipeline's rec-graph bake
  // would spuriously fail. Launching fill_pos on stream_ (not the legacy
  // stream) likewise avoids invalidating a concurrent capture through an
  // implicit legacy-stream dependency.
  return cudaStreamSynchronize(stream_) == cudaSuccess;
}

void PPFormulaNetOrt::free_buffers() noexcept {
  for (void *p : {(void *)d_x_, (void *)d_mem_, (void *)d_ck_, (void *)d_cv_, (void *)d_log_,
                  (void *)kA_, (void *)kB_, (void *)vA_, (void *)vB_,
                  (void *)d_tok_, (void *)d_next_, (void *)d_pos_, (void *)d_all_, (void *)d_done_,
                  (void *)d_mem_all_, (void *)ck_all_, (void *)cv_all_,
                  (void *)kA384_, (void *)kB384_, (void *)vA384_, (void *)vB384_})
    if (p) cudaFree(p);
}

bool PPFormulaNetOrt::load_model_dir(const std::string &model_dir) {
  fs::path mp(model_dir);
  fs::path base = fs::is_directory(mp) ? mp : mp.parent_path();
  fs::path fast = base / "fast";
  // GPU FAST is the only path: the host-loop matches the fused CDM exactly (0.811)
  // at ~8x speed. There is no fused EXACT mode and no FORMULA_DEVICE=cpu here — CPU
  // formula runs through OrtFormulaRecognizer instead.
  fast_ = true;
  plusm_ = (label_ == "ppformulanet_plus_m");  // 6-layer MBart fast host-loop
  fast_dir_ = fast.string();
  if (cudaStreamCreate(&stream_) != cudaSuccess) {
    std::cerr << "[PPFormulaNetOrt] FATAL: CUDA stream create failed\n";
    return false;
  }
  if (!alloc_buffers()) {
    std::cerr << "[PPFormulaNetOrt] FATAL: buffer alloc failed\n";
    return false;
  }
  // FAST: encoder + cross-KV prep + static-KV step, all ORT-CUDA-13 on our stream,
  // driven by a host AR loop. The encoder is bit-exact to the fused in-graph encoder.
  // plus-M ships its split graphs in the model dir (encoder/prep/decoder_step.onnx);
  // PP-FormulaNet-S keeps them in a fast/ subdir (encoder/prep/step_batched.onnx).
  const fs::path enc_p  = plusm_ ? base / "encoder.onnx"      : fast / "encoder.onnx";
  const fs::path prep_p = plusm_ ? base / "prep.onnx"         : fast / "prep.onnx";
  const fs::path step_p = plusm_ ? base / "decoder_step.onnx" : fast / "step_batched.onnx";
  const bool have_all =
      fs::exists(enc_p) && fs::exists(prep_p) && fs::exists(step_p);
  const bool fast_loaded =
      have_all && enc_.load(enc_p.string(), 0, stream_, false)
      && prep_.load(prep_p.string(), 0, stream_, false)
      // NOTE: ORT CUDA-graph capture was tried for the plus-M step (enable_cuda_graph)
      // and does NOT work — it freezes the step's pos-dependent KV scatter at the first
      // step's position, so every replay writes to the wrong slot (0/30 correct), and it
      // gave no speedup anyway (the step is compute/memory-bound, not launch-bound). We
      // keep the persistent-binding run_graph() path (correct + slightly faster) but do
      // NOT enable cuda-graph. The real per-step lever is length-bucketing the KV window.
      && step_.load(step_p.string(), 0, stream_, false, /*enable_cuda_graph=*/false);
  if (!fast_loaded) {
    // The FAST split graphs are REQUIRED — there is no fused fallback. Missing/unloadable
    // graphs mean an incomplete deploy (or FORMULA_ONNX pointing at the wrong model dir);
    // fail loudly rather than silently serving the wrong/slower model (no-silent-failure).
    std::cerr << "[PPFormulaNetOrt] FATAL: FAST graphs missing/unloadable under "
              << (plusm_ ? base.string() : fast.string())
              << " (need encoder.onnx + prep.onnx + "
              << (plusm_ ? "decoder_step.onnx" : "step_batched.onnx")
              << "). The fast/ bundle must ship with the model.\n";
    return false;
  }
  // plus-M length-bucket: load the optional 384-KV-window step. Absent (e.g. fresh
  // deploy) -> decode_continuous_plusm transparently uses the 1056 window for all crops.
  if (plusm_) {
    const fs::path short_p = base / "decoder_step_384.onnx";
    if (fs::exists(short_p) && step_short_.load(short_p.string(), 0, stream_, false))
      std::cerr << "[PPFormulaNetOrt] plus-M length-bucket: 384-KV step loaded\n";
    else
      std::cerr << "[PPFormulaNetOrt] plus-M length-bucket: decoder_step_384.onnx absent — "
                   "1056 window for all crops\n";
  }
  std::cerr << "[PPFormulaNetOrt] FAST decode path ("
            << (plusm_ ? "plus-M 6-layer MBart" : "PP-FormulaNet_plus-S")
            << " encoder+prep+step host-loop)\n";
  ready_ = static_cast<bool>(tok_);  // ready once both model+tokenizer loaded
  return true;
}

bool PPFormulaNetOrt::load_tokenizer(const std::string &path) {
  tok_ = FormulaTokenizer::load(path);
  if (!tok_) { std::cerr << "[PPFormulaNetOrt] tokenizer load failed: " << path << '\n'; return false; }
  if (step_.ready()) ready_ = true;
  return true;
}

void PPFormulaNetOrt::preprocess_crop(const Box &box, const GpuImage &page,
                                      float *dst) const {
  auto cr = clamped_crop_rect(box, page.cols, page.rows);
  const int x0 = cr[0], y0 = cr[1], w = cr[2], h = cr[3];
  std::vector<uint8_t> tmp((size_t)std::max(1, w) * std::max(1, h) * 3);
  const uint8_t *sp = host_page_.data() + (size_t)y0 * page.step + (size_t)x0 * 3;
  for (int r = 0; r < h; ++r)
    std::memcpy(tmp.data() + (size_t)r * w * 3, sp + (size_t)r * page.step, (size_t)w * 3);
  formula_preprocess_one(tmp.data(), w, h, dst);
}

bool PPFormulaNetOrt::encode_crops(int n, float *mem0) {
  bool eok = true;
  for (int b = 0; b < n && eok; ++b) {
    OrtTensor ex{"x", d_x_ + (size_t)b * S * S, {1, 1, S, S}, false};
    OrtTensor em{"p2o.pd_op.transpose.0.0", mem0 + (size_t)b * CTX * 2048,
                 {1, CTX, 2048}, false};
    eok = enc_.run({ex}, {em});
  }
  return eok;
}

std::vector<FormulaEngineResult>
PPFormulaNetOrt::run(const GpuImage &page, const std::vector<Box> &boxes, cudaStream_t stream) {
  std::vector<FormulaEngineResult> out;
  if (boxes.empty()) return out;
  out.resize(boxes.size());
  if (!ready_ || page.empty()) { for (auto &r : out) r.ok = false; return out; }

  size_t need = (size_t)page.rows * page.step;
  if (host_page_.size() < need) host_page_.resize(need);
  cudaMemcpyAsync(host_page_.data(), page.data, need, cudaMemcpyDeviceToHost, stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) { for (auto &r : out) r.ok = false; return out; }

  const int N = (int)boxes.size();
  static const bool drop_collapse = env::env_present("PPFNS_DROP_COLLAPSE");
  // Decode chunk size. The fused encoder's batched cuDNN conv drifts slightly from
  // single-sample at large batches (flips a few near-tie tokens); small batches match
  // the per-crop reference while still amortizing the AR Loop. Tunable for testing.
  static const int chunk = []{ std::string e = env::env_or("PPFNS_CHUNK", "");
    int c = e.empty() ? 8 : std::atoi(e.c_str()); return c < 1 ? 1 : (c > MAX_B ? MAX_B : c); }();
  // plus-M decodes the whole page with continuous (iteration-level) batching — short crops
  // never stall behind long ones, and the batch stays full across the page's whole queue.
  // (-S / fused keep the simpler per-chunk lockstep path below.)
  if (fast_ && plusm_) {
    decode_plusm_page(N, boxes, page, drop_collapse, out);
    return out;
  }
  for (int s0 = 0; s0 < N; s0 += chunk) {
    const int B = std::min(chunk, N - s0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < B; ++i)
      preprocess_crop(boxes[s0 + i], page, host_in_.data() + (size_t)i * S * S);
    cudaMemcpyAsync(d_x_, host_in_.data(), (size_t)B * S * S * sizeof(float), cudaMemcpyHostToDevice, stream_);
    std::vector<std::vector<int64_t>> seqs(B);   // clean content tokens per crop
    bool chunk_ok = true;                        // false -> mark this chunk's crops failed
    if (fast_) {
      // PP-FormulaNet-S: encode per crop (batch=1 matches the reference) -> d_mem_, cross-KV
      // prep -> ck/cv, then the static-KV host AR loop. (plus-M took the whole-page continuous
      // path above and never reaches here.)
      // All on stream_, deferred sync inside decode_chunk.
      const int64_t Bi = B;
      OrtTensor pin{"memory", d_mem_, {Bi, CTX, 2048}, false};
      OrtTensor pck{"ck", d_ck_, {2, Bi, H, CTX, Dh}, false};
      OrtTensor pcv{"cv", d_cv_, {2, Bi, H, CTX, Dh}, false};
      if (!encode_crops(B, d_mem_) || !prep_.run({pin}, {pck, pcv})) {
        std::cerr << "[PPFormulaNetOrt] fast encode/prep failed\n";
        for (int i = 0; i < B; ++i) out[s0 + i].ok = false; continue;
      }
      chunk_ok = decode_chunk(B, seqs);
    }
    for (int i = 0; i < B; ++i) {
      std::string latex = tok_->decode(seqs[i], /*post_process=*/false);
      // Dropping collapsed formulas to empty measured WORSE (it also drops legit long
      // matrices). Emit-everything is better; keep the detector behind an opt-in env.
      out[s0 + i].latex = (drop_collapse && formula_is_mode_collapsed(seqs[i], latex)) ? std::string() : latex;
      out[s0 + i].token_count = seqs[i].size();
      out[s0 + i].hit_eos = !seqs[i].empty();  // EOS-stripped seq -> non-empty == normal stop
      out[s0 + i].ok = chunk_ok;               // false -> decode_chunk truncated on error
    }
  }
  return out;
}

}  // namespace turbo_ocr::formula
