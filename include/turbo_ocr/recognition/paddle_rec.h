#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/cuda_ptr.h"

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

private:
  std::vector<std::string> label_list_;
  int rec_batch_num_ = 32;
  int rec_image_h_ = 48;
  int rec_image_w_ = 320;

  std::unique_ptr<engine::TrtEngine> engine_;

  static constexpr int kMaxRecWidth = 4000;
  static constexpr std::array kWidthBuckets = {320, 480, 800, 1200, 1600, 2000, 2500, 3200, 4000};

  // Actual model dims (probed after load)
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
};

} // namespace turbo_ocr::recognition
