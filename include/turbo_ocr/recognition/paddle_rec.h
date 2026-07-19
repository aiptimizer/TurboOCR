#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/rec_graph_profiles.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/recognition/rec_geometry.h"

namespace turbo_ocr::recognition {

/// GPU text recognizer using TensorRT (CRNN + CTC decoding).
class PaddleRec {
public:
  PaddleRec();
  ~PaddleRec() noexcept;

  /// Set the maximum batch size BEFORE load_model (default 32).
  void set_batch_num(int n) { rec_batch_num_ = n; }

  /// Load a TensorRT recognition engine and probe output dimensions.
  [[nodiscard]] bool load_model(const std::string &model_path);
  /// Load the character dictionary for CTC decoding.
  [[nodiscard]] bool load_dict(const std::string &dict_path);

  [[nodiscard]] std::vector<std::pair<std::string, float>>
  run(const GpuImage &img, const std::vector<Box> &boxes,
      cudaStream_t stream = 0);

  // Batched recognition across multiple images.
  // Each element is (gpu_image, boxes_for_that_image).
  // Returns one result vector per image, matching the input order.
  struct ImageCrops {
    GpuImage img;
    std::vector<Box> boxes;
  };
  [[nodiscard]] std::vector<std::vector<std::pair<std::string, float>>>
  run_multi(const std::vector<ImageCrops> &image_crops,
            cudaStream_t stream = 0);

  // Eagerly allocate all GPU/pinned buffers (called during warmup)
  void allocate_buffers();

  // Record one CUDA graph per static (batch, width) engine profile. Call
  // once from pipeline warmup, after allocate_buffers(). No-op when the
  // cached engine predates the static profiles or graphs are disabled.
  void bake_graphs(cudaStream_t stream);

  // Crops skipped by the most recent run()/run_multi() because an engine
  // output exceeded the preallocated decode buffers. The pipeline reads this
  // to flag text_degraded — a partial drop must never look like a page that
  // simply had less text. last_dropped_per_image() attributes run_multi()
  // drops to their source image (empty after run()).
  [[nodiscard]] int last_dropped_crops() const { return dropped_crops_; }
  [[nodiscard]] const std::vector<int> &last_dropped_per_image() const {
    return dropped_per_image_;
  }

private:
  std::vector<std::string> label_list_;
  int rec_batch_num_ = 32;
  int rec_image_h_ = 48;
  int rec_image_w_ = 320;
  int dropped_crops_ = 0;
  std::vector<int> dropped_per_image_;

  std::unique_ptr<engine::TrtEngine> engine_;
  // Crop-width constants live in rec_geometry.h (shared with the ORT
  // recognizer and the engine profiles).

  struct BakedSlot {
    int slot = -1;
    int seq_len = 0;
  };
  // Indexed by position in engine::kRecGraphProfiles.
  std::vector<BakedSlot> baked_slots_;
  bool graphs_baked_ = false;

  // Actual model dims (probed after load via probe_output_dims). The
  // initializers are placeholders only: actual_num_classes_ must stay >= the
  // widest tier (medium/small CTC width 18,710) so the over-dim skip guard in
  // run() never fires before the probe overwrites it with the true width.
  int actual_seq_len_ = 600;
  int actual_num_classes_ = 20000;

  // GPU buffers (RAII -- no manual cudaFree needed)
  CudaPtr<float> d_batch_input_;
  CudaPtr<float> d_output_;
  CudaPtr<float> d_M_invs_;
  CudaPtr<int> d_crop_widths_;

  // Multi-slot output buffers: one per batch iteration, allowing all GPU work
  // to be queued without inter-batch sync. Each slot has its own d_indices/d_scores
  // on GPU, h_indices/h_scores on host (pinned), and h_M_invs/h_crop_widths on
  // host (pinned) so the CPU can fill transforms for slot N+1 while the DMA
  // for slot N is still in flight. After all batches are queued, a single
  // cudaStreamSynchronize retrieves all results at once.
  static constexpr int kMaxSlots = 20; // enough for 640 boxes / 32 batch
  struct OutputSlot {
    CudaPtr<int> d_indices;
    CudaPtr<float> d_scores;
    CudaHostPtr<int> h_indices;
    CudaHostPtr<float> h_scores;
    CudaHostPtr<float> h_M_invs;    // per-slot to avoid race with DMA
    CudaHostPtr<int> h_crop_widths; // per-slot to avoid race with DMA
  };
  OutputSlot output_slots_[kMaxSlots];

  bool buffers_allocated_ = false;

  // Pre-allocated per-request buffers (avoid heap alloc in hot path)
  struct CropInfo {
    int orig_idx;
    int bucket_w;
  };
  std::vector<CropInfo> crops_buf_;

  // Common init after engine load (called by both load_model overloads)
  [[nodiscard]] bool probe_and_init();

  // Run one (cur_batch, imgW) bucket: baked graph when available, dynamic
  // fallback otherwise. Fills seq_len/num_classes for decoding.
  void infer_bucket(int cur_batch, int imgW, cudaStream_t stream,
                    int &seq_len, int &num_classes);
};

} // namespace turbo_ocr::recognition
