#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "nvidia/stages/formula_recognizer.h"
#include "turbo_ocr/analysis/formula/ppformulanet/formula_tokenizer.h"
#include "turbo_ocr/analysis/formula/ppformulanet/ort_session.h"

namespace turbo_ocr::formula {

// Pure in-process PP-FormulaNet recognizer running ONNX Runtime (no TensorRT
// engines, no Python sidecar, no socket/GIL). It loads .onnx models only.
// The default "ppformulanet_s" bundle's weights ARE PP-FormulaNet_plus-S
// (verified 2026-08-05: byte-identical rebuild from paddle's plus-S download —
// see scripts/models/onnx/export_ppformulanet_plus_s_fast.py); -S and plus-S
// share one architecture, so this class serves both names.
//
// GPU FAST is the ONLY path: ORT-CUDA-13 encoder.onnx -> cross-KV prep.onnx -> a
// static-KV single-step decoder (step_batched.onnx, 1056-token KV buffer) driven
// by a host AR loop with on-GPU argmax + KV ping-pong. Matches the fused reference
// EXACTLY (CDM 0.811) and is ~8x faster than the fused Loop. The slow fused graph
// (inference_trt.onnx) is no longer used here — there is no EXACT mode and no fused
// fallback; on CPU use OrtFormulaRecognizer instead.
//
// The FAST split graphs live in <model_parent>/fast/ (encoder.onnx, prep.onnx,
// step_batched.onnx); plus-M ships them in the model dir itself. They are REQUIRED:
// if any is missing/unloadable, load fails LOUDLY (no fused fallback) — the fast/
// bundle must ship with the model. See docs/models/formula.md.
class PPFormulaNetOrt final : public IFormulaRecognizer {
public:
  // backend_label names the engine in logs/routing. PP-FormulaNet_plus-M reuses this
  // class via ("ppformulanet_plus_m"): it ships its OWN split graphs (encoder.onnx +
  // prep.onnx + decoder_step.onnx in the model dir) and runs the plus-M 6-layer MBart
  // FAST host-loop (decode_chunk_plusm / decode_continuous_plusm).
  explicit PPFormulaNetOrt(std::string backend_label = "ppformulanet_s");
  ~PPFormulaNetOrt() noexcept override;

  [[nodiscard]] bool load_model_dir(const std::string &model_dir) override;
  [[nodiscard]] bool load_tokenizer(const std::string &path) override;

  [[nodiscard]] std::vector<FormulaEngineResult>
  run(const GpuImage &page, const std::vector<Box> &boxes,
      cudaStream_t stream) override;

  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return label_;
  }

  // Dev/CI harness (NOT used by the server): decode the 30 gate crops in <gate_dir>
  // (en15_crops.bin + zh15_crops.bin, preprocessed [15,1,384,384] each) through this
  // backend's FAST host loop and write /tmp/cpp_<label>_tokens.json (lockstep), the
  // continuous tokens (plus-M), and /tmp/cpp_<label>_latex.json, plus print lockstep
  // crops/s. Driven only by tools/checks/plusm_selftest.cpp so no bench code lives in the
  // production load/run path. Returns false if the gate crops or model are unavailable.
  [[nodiscard]] bool gate_bench(const std::string &gate_dir);

private:
  bool alloc_buffers();
  void free_buffers() noexcept;
  // Crop the box out of the D2H'd page (host_page_) and run the PP-FormulaNet
  // preprocess into dst ([1,384,384] floats). Shared by run() and
  // decode_plusm_page (identical per-crop hot loop).
  void preprocess_crop(const Box &box, const GpuImage &page, float *dst) const;
  // Encode n preprocessed crops (already H2D'd to d_x_) one-by-one (batch=1
  // matches the reference) into mem0 + b*CTX*2048. Shared by run(),
  // decode_plusm_page and gate_bench.
  bool encode_crops(int n, float *mem0);
  // GPU FAST path: lockstep-batched static-KV step host-loop (encoder+prep+step).
  // Returns false when a step/argmax/CUDA error truncated the decode.
  bool decode_chunk(int B, std::vector<std::vector<int64_t>> &out);
  // PP-FormulaNet_plus-M FAST path: same host-loop shape as decode_chunk, but the
  // plus-M MBart decoder is 6-layer / Dh=32 / single-token (pos[B] per-seq) and the
  // step graph emits next_token in-graph (no argmax kernel). Reads encoder.onnx +
  // prep.onnx + decoder_step.onnx from the model dir (not a fast/ subdir).
  bool decode_chunk_plusm(int B, std::vector<std::vector<int64_t>> &out);
  // PP-FormulaNet_plus-M continuous (iteration-level) batched decode: Bslots slots
  // process a queue of N crops whose cross-KV is pre-computed in ck_all_/cv_all_. A
  // slot that emits EOS is evicted (its sequence saved to out[crop]) and refilled with
  // the next queued crop (cross-KV copied into the slot, pos->0, token->START); the
  // static-KV step overwrites each KV slot before reading it, so no per-slot KV reset
  // is needed. Keeps the batch full -> removes the lockstep long-tail stall.
  // `step`/`kA..vB`/`maxlen` select the KV bucket (the small 384-window step_short_ for the
  // common case, or the 1056 step_ for the long tail). `queue` is the crop indices (into
  // ck_all_, of which n_total were prepped) to decode; out[crop] gets each finished sequence,
  // and a crop that hits `maxlen` tokens without EOS is appended to `overflow` (its partial
  // out[] cleared) for the caller to re-decode in a larger bucket.
  bool decode_continuous_plusm(OrtSession &step, float *kA, float *kB, float *vA, float *vB,
                               int maxlen, int Bslots, int n_total,
                               const std::vector<int> &queue,
                               std::vector<std::vector<int64_t>> &out,
                               std::vector<int> &overflow, bool final_bucket);
  // Production plus-M path: preprocess+encode ALL N formula crops of a page into the
  // all-crop buffers (in MAX_B-sized preprocessing batches, then batched cross-KV prep),
  // and decode them with the continuous host loop (Bslots = min(N, PM_MAX_BATCH)) — the
  // 384-KV bucket first, the long tail re-decoded in the 1056 bucket. Replaces the per-chunk
  // lockstep path so short crops never stall behind long ones. host_page_ must already hold
  // the D2H'd page.
  void decode_plusm_page(int N, const std::vector<Box> &boxes, const GpuImage &page,
                         bool drop_collapse, std::vector<FormulaEngineResult> &out);

  OrtSession enc_, prep_, step_;     // GPU FAST path
  OrtSession step_short_;            // plus-M length-bucket: 384-KV-window step (common case)
  bool fast_ = false;                // GPU host-loop (always true once loaded)
  bool plusm_ = false;               // PP-FormulaNet_plus-M (6-layer MBart fast host-loop)
  std::string label_ = "ppformulanet_s";   // backend_name() (engine label)
  std::optional<FormulaTokenizer> tok_;
  bool ready_ = false;
  std::string fast_dir_;
  cudaStream_t stream_ = nullptr;    // GPU FAST decode stream

  float *d_x_ = nullptr;             // [MAX_B,1,384,384] device crops (GPU paths only)
  // FAST-path device buffers (allocated only when fast_).
  float *d_mem_ = nullptr, *d_ck_ = nullptr, *d_cv_ = nullptr, *d_log_ = nullptr;
  float *kA_ = nullptr, *kB_ = nullptr, *vA_ = nullptr, *vB_ = nullptr;
  // plus-M length-bucket: 384-KV-window self-attention buffers (the common-case fast path).
  float *kA384_ = nullptr, *kB384_ = nullptr, *vA384_ = nullptr, *vB384_ = nullptr;
  int64_t *d_tok_ = nullptr, *d_next_ = nullptr, *d_pos_ = nullptr, *d_all_ = nullptr;
  unsigned char *d_done_ = nullptr;  // sticky per-row EOS flags (on-device termination)
  // plus-M continuous-batch scratch: all-crop encoder memory + pre-computed cross-KV.
  float *d_mem_all_ = nullptr, *ck_all_ = nullptr, *cv_all_ = nullptr;
  std::vector<uint8_t> host_page_;   // page D2H scratch
  std::vector<float> host_in_;       // [MAX_B,1,384,384] preprocessed crops
};

}  // namespace turbo_ocr::formula
