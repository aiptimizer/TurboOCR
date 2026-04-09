#include "turbo_ocr/pipeline/ocr_pipeline.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/timing.h"
#include "turbo_ocr/decode/gpu_image.h"

#include <algorithm>
#include <cstring>
#include <format>

#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::pipeline;
using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::GpuImage;
using turbo_ocr::PipelineTimer;
using turbo_ocr::is_vertical_box;
using turbo_ocr::sorted_boxes;
using turbo_ocr::detection::PaddleDet;
using turbo_ocr::classification::PaddleCls;
using turbo_ocr::recognition::PaddleRec;

OcrPipeline::OcrPipeline() {
  det_ = std::make_unique<PaddleDet>();
  rec_ = std::make_unique<PaddleRec>();
}

OcrPipeline::~OcrPipeline() noexcept {
  if (rec_stream_)
    cudaStreamDestroy(rec_stream_);
  if (rec_event_)
    cudaEventDestroy(rec_event_);
  if (det_event_)
    cudaEventDestroy(det_event_);
  for (auto &buf : img_bufs_) {
    if (buf.d_buf)
      cudaFree(buf.d_buf);
  }
  cudaFreeHost(h_pinned_buf_);
  for (auto &buf : batch_img_bufs_) {
    if (buf.d_buf)
      cudaFree(buf.d_buf);
  }
}

bool OcrPipeline::init(const std::string &det_model,
                       const std::string &rec_model,
                       const std::string &rec_dict,
                       const std::string &cls_model) {
  if (!det_->load_model(det_model))
    return false;
  if (!rec_->load_model(rec_model))
    return false;
  if (!rec_->load_dict(rec_dict))
    return false;

  if (!cls_model.empty()) {
    cls_ = std::make_unique<PaddleCls>();
    if (!cls_->load_model(cls_model)) {
      std::cerr << std::format("[Pipeline] Failed to load GPU cls model: {}", cls_model) << '\n';
      return false;
    }
    use_cls_ = true;
    std::cout << "[Pipeline] Angle classifier enabled" << '\n';
  }

  // Pre-allocate double-buffered GPU upload buffers for a typical image
  // (1920x1080). Grow-only: reused for smaller images, reallocated only if a
  // larger image arrives. Two buffers allow recognition on rec_stream_ to read
  // the previous image while the next image is uploaded on the caller's stream.
  constexpr int kDefaultRows = 1080;
  constexpr int kDefaultCols = 1920;
  for (auto &buf : img_bufs_) {
    CUDA_CHECK(
        cudaMallocPitch(&buf.d_buf, &buf.pitch, kDefaultCols * 3, kDefaultRows));
    buf.cap_rows = kDefaultRows;
    buf.cap_cols = kDefaultCols;
  }

  // Eagerly allocate rec/cls buffers (avoids first-request latency)
  rec_->allocate_buffers();
  if (use_cls_)
    cls_->allocate_buffers();

  // Dedicated recognition stream — allows det on the caller's stream to
  // overlap with rec on rec_stream_ across consecutive pipeline invocations.
  CUDA_CHECK(cudaStreamCreateWithFlags(&rec_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&rec_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&det_event_, cudaEventDisableTiming));

  return true;
}

void OcrPipeline::warmup_gpu(cudaStream_t stream) {
  // Run full pipeline with a dummy image to trigger TRT JIT and lazy GPU allocations
  cv::Mat dummy(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(dummy, cv::Point(10, 30), cv::Point(90, 70), cv::Scalar(0, 0, 0), 2);
  (void)run(dummy, stream);
  cudaStreamSynchronize(stream);

  // Warm all 5 rec width buckets to eliminate TRT JIT latency on first use.
  // The initial run() above only hits one bucket; each unseen bucket pays ~5-50ms
  // on first inference. We call rec_->run() directly with fake boxes sized to
  // produce crops at each bucket width.
  static constexpr int warm_widths[] = {320, 480, 800, 1600, 4000};
  auto &buf = img_bufs_[0]; // use first buffer for warmup
  for (int w : warm_widths) {
    // Create a white dummy image wide enough for this bucket
    cv::Mat dummy_wide(48, w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Upload to GPU (reuses the grow-only buffer)
    if (dummy_wide.rows > buf.cap_rows || dummy_wide.cols > buf.cap_cols) {
      cudaFree(buf.d_buf);
      buf.d_buf = nullptr;
      CUDA_CHECK(cudaMallocPitch(&buf.d_buf, &buf.pitch,
                                  dummy_wide.cols * 3, dummy_wide.rows));
      buf.cap_rows = dummy_wide.rows;
      buf.cap_cols = dummy_wide.cols;
    }
    auto needed = static_cast<size_t>(dummy_wide.rows) * dummy_wide.step;
    if (needed > h_pinned_size_) {
      cudaFreeHost(h_pinned_buf_);
      h_pinned_buf_ = nullptr;
      CUDA_CHECK(cudaMallocHost(&h_pinned_buf_, needed));
      h_pinned_size_ = needed;
    }
    std::memcpy(h_pinned_buf_, dummy_wide.data, needed);
    CUDA_CHECK(cudaMemcpy2DAsync(buf.d_buf, buf.pitch, h_pinned_buf_,
                                  dummy_wide.step, dummy_wide.cols * 3,
                                  dummy_wide.rows, cudaMemcpyHostToDevice, stream));

    GpuImage gpu_img{buf.d_buf, buf.pitch, dummy_wide.rows, dummy_wide.cols};

    // A single box spanning the full image -> crop width = w
    Box box{};
    box[0] = {0, 0};                      // top-left
    box[1] = {w - 1, 0};                  // top-right
    box[2] = {w - 1, dummy_wide.rows - 1}; // bottom-right
    box[3] = {0, dummy_wide.rows - 1};     // bottom-left
    std::vector<Box> boxes = {box};

    auto rec_res = rec_->run(gpu_img, boxes, stream);
    cudaStreamSynchronize(stream);
    (void)rec_res;
  }
}

std::vector<OCRResultItem> OcrPipeline::run(const cv::Mat &img,
                                            cudaStream_t stream) {
  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  // --- Pipeline overlap: wait for previous recognition to finish -------------
  // rec_event_ was recorded on rec_stream_ at the end of the previous run().
  // This only blocks if the previous recognition hasn't finished yet — when it
  // HAS finished (common case), cudaEventSynchronize returns immediately.
  // Meanwhile, the previous rec was reading from img_bufs_[prev], so we must
  // wait before we can safely reuse that buffer (which is now cur_img_buf_
  // after the toggle below).
  CUDA_CHECK(cudaEventSynchronize(rec_event_));

  // Toggle to the other image buffer so recognition on rec_stream_ (previous
  // call) can safely finish reading the old buffer while we write the new
  // image to the current buffer.
  cur_img_buf_ ^= 1;
  auto &buf = img_bufs_[cur_img_buf_];

  // Grow-only realloc (typically allocates once for fixed camera resolution)
  if (img.rows > buf.cap_rows || img.cols > buf.cap_cols) [[unlikely]] {
    cudaFree(buf.d_buf);
    buf.d_buf = nullptr;
    CUDA_CHECK(cudaMallocPitch(&buf.d_buf, &buf.pitch, img.cols * 3,
                               img.rows));
    buf.cap_rows = img.rows;
    buf.cap_cols = img.cols;
  }

  // Grow-only pinned staging buffer for truly async upload
  timer.gpu_start("image_upload");
  auto needed = static_cast<size_t>(img.rows) * img.step;
  if (needed > h_pinned_size_) [[unlikely]] {
    cudaFreeHost(h_pinned_buf_);
    h_pinned_buf_ = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned_buf_, needed));
    h_pinned_size_ = needed;
  }

  // Copy pageable cv::Mat data into pinned buffer, then async upload to GPU
  std::memcpy(h_pinned_buf_, img.data, needed);
  CUDA_CHECK(cudaMemcpy2DAsync(buf.d_buf, buf.pitch, h_pinned_buf_, img.step,
                                img.cols * 3, img.rows,
                                cudaMemcpyHostToDevice, stream));
  timer.gpu_stop();

  GpuImage gpu_img{buf.d_buf, buf.pitch, img.rows, img.cols};

  // Detection
  timer.gpu_start("detection_inference");
  std::vector<Box> boxes = det_->run(gpu_img, img.rows, img.cols, stream);
  timer.gpu_stop();

  // Sort boxes top-to-bottom, left-to-right (in-place)
  timer.cpu_start("box_postprocessing");
  sorted_boxes(boxes);
  timer.cpu_stop();

  // Optional angle classification — only classify boxes that look vertical.
  // Saves time by not classifying horizontal text (majority of boxes).
  if (use_cls_) {
    // Collect indices of vertical-looking boxes (h >= w*1.5)
    vertical_box_indices_.clear();
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
      if (is_vertical_box(boxes[i]))
        vertical_box_indices_.push_back(i);
    }
    if (!vertical_box_indices_.empty()) {
      // Build subset of vertical boxes for classification
      vertical_boxes_buf_.clear();
      vertical_boxes_buf_.reserve(vertical_box_indices_.size());
      for (int idx : vertical_box_indices_)
        vertical_boxes_buf_.push_back(boxes[idx]);

      timer.gpu_start("angle_classification");
      cls_->run(gpu_img, vertical_boxes_buf_, stream);
      timer.gpu_stop();

      // Write classified boxes back
      for (size_t j = 0; j < vertical_box_indices_.size(); ++j)
        boxes[vertical_box_indices_[j]] = vertical_boxes_buf_[j];
    }
  }

  // Recognition — launch on dedicated rec_stream_ so the caller's stream is
  // free for the next image's upload+detection (pipeline parallelism).
  // Record det_event_ on the caller's stream after det+cls, then make
  // rec_stream_ wait on it before launching recognition.
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));

  timer.gpu_start("recognition_inference");
  auto rec_results = rec_->run(gpu_img, boxes, rec_stream_);
  timer.gpu_stop();

  // Record rec_event_ so the NEXT run() can wait for this recognition to
  // finish before reusing the image buffer. Note: rec_->run() syncs
  // rec_stream_ internally (for D2H + CTC decode), so by the time we get
  // here rec_stream_ is idle and this event is immediately "done". The event
  // is still useful as a correctness guard and for future async recognition.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  // Combine (filter by drop_score, matching Python's behavior)
  constexpr float kDropScore = turbo_ocr::kDropScore;
  std::vector<OCRResultItem> final_results;
  final_results.reserve(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (i < rec_results.size()) {
      if (rec_results[i].second < kDropScore)
        continue;
      if (rec_results[i].first.empty())
        continue;
      final_results.push_back({
        .text = std::move(rec_results[i].first),
        .confidence = rec_results[i].second,
        .box = boxes[i],
      });
    }
  }

  timer.print_total();

  return final_results;
}

std::pair<void *, size_t> OcrPipeline::ensure_gpu_buf(int rows, int cols) {
  auto &buf = img_bufs_[cur_img_buf_];
  if (rows > buf.cap_rows || cols > buf.cap_cols) {
    cudaFree(buf.d_buf);
    buf.d_buf = nullptr;
    CUDA_CHECK(cudaMallocPitch(&buf.d_buf, &buf.pitch, cols * 3, rows));
    buf.cap_rows = rows;
    buf.cap_cols = cols;
  }
  return {buf.d_buf, buf.pitch};
}

std::vector<OCRResultItem> OcrPipeline::run(GpuImage gpu_img,
                                            cudaStream_t stream) {
  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  // No image_upload stage — the image is already on the GPU.
  // Wait for any previous recognition that might still be reading its source
  // image. For caller-owned GpuImage this is a correctness guard only.
  CUDA_CHECK(cudaEventSynchronize(rec_event_));

  // Detection
  timer.gpu_start("detection_inference");
  std::vector<Box> boxes = det_->run(gpu_img, gpu_img.rows, gpu_img.cols, stream);
  timer.gpu_stop();

  // Sort boxes top-to-bottom, left-to-right (in-place)
  timer.cpu_start("box_postprocessing");
  sorted_boxes(boxes);
  timer.cpu_stop();

  // Optional angle classification — only classify boxes that look vertical.
  if (use_cls_) {
    vertical_box_indices_.clear();
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
      if (is_vertical_box(boxes[i]))
        vertical_box_indices_.push_back(i);
    }
    if (!vertical_box_indices_.empty()) {
      vertical_boxes_buf_.clear();
      vertical_boxes_buf_.reserve(vertical_box_indices_.size());
      for (int idx : vertical_box_indices_)
        vertical_boxes_buf_.push_back(boxes[idx]);

      timer.gpu_start("angle_classification");
      cls_->run(gpu_img, vertical_boxes_buf_, stream);
      timer.gpu_stop();

      for (size_t j = 0; j < vertical_box_indices_.size(); ++j)
        boxes[vertical_box_indices_[j]] = vertical_boxes_buf_[j];
    }
  }

  // Recognition — use det_event_ for det→rec stream handoff.
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));

  timer.gpu_start("recognition_inference");
  auto rec_results = rec_->run(gpu_img, boxes, rec_stream_);
  timer.gpu_stop();

  // Record rec_event_ for the next run() to wait on.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  // Combine (filter by drop_score)
  constexpr float kDropScore = turbo_ocr::kDropScore;
  std::vector<OCRResultItem> final_results;
  final_results.reserve(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (i < rec_results.size()) {
      if (rec_results[i].second < kDropScore)
        continue;
      if (rec_results[i].first.empty())
        continue;
      final_results.push_back({
        .text = std::move(rec_results[i].first),
        .confidence = rec_results[i].second,
        .box = boxes[i],
      });
    }
  }

  timer.print_total();

  return final_results;
}

std::vector<std::vector<OCRResultItem>> OcrPipeline::run_batch(
    const std::vector<cv::Mat> &imgs, cudaStream_t stream) {
  if (imgs.empty())
    return {};

  // If only one image, just use single-image path
  if (imgs.size() == 1)
    return {run(imgs[0], stream)};

  const int n = static_cast<int>(imgs.size());

  // Guard against exceeding pre-allocated batch buffer capacity.
  // Callers should chunk at kMaxBatchImages before calling this method.
  if (n > kMaxBatchImages) [[unlikely]] {
    std::cerr << std::format("[Pipeline] run_batch called with {} images, max is {}. "
                             "Processing first {} only.\n", n, kMaxBatchImages, kMaxBatchImages);
  }
  const int batch_n = std::min(n, kMaxBatchImages);

  // --- Phase 1: Upload all images to GPU, run batched detection + cls ---
  // We need all images alive on GPU simultaneously for batched recognition.
  struct PerImage {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int rows = 0, cols = 0;
    std::vector<Box> boxes;
  };
  std::vector<PerImage> per_img(batch_n);

  // Upload all images to GPU first
  for (int i = 0; i < batch_n; i++) {
    const auto &img = imgs[i];
    auto &pi = per_img[i];
    pi.rows = img.rows;
    pi.cols = img.cols;

    // Use pre-allocated GPU buffer (grow-only, avoids cudaMalloc per batch)
    auto &bbuf = batch_img_bufs_[i];
    if (img.rows > bbuf.cap_rows || img.cols > bbuf.cap_cols) [[unlikely]] {
      if (bbuf.d_buf) cudaFree(bbuf.d_buf);
      CUDA_CHECK(cudaMallocPitch(&bbuf.d_buf, &bbuf.pitch, img.cols * 3, img.rows));
      bbuf.cap_rows = img.rows;
      bbuf.cap_cols = img.cols;
    }
    pi.d_buf = bbuf.d_buf;
    pi.pitch = bbuf.pitch;

    // Upload via the shared pinned staging buffer
    auto needed = static_cast<size_t>(img.rows) * img.step;
    if (needed > h_pinned_size_) [[unlikely]] {
      if (h_pinned_buf_) {
        cudaFreeHost(h_pinned_buf_);
        h_pinned_buf_ = nullptr;
      }
      CUDA_CHECK(cudaMallocHost(&h_pinned_buf_, needed));
      h_pinned_size_ = needed;
    }
    std::memcpy(h_pinned_buf_, img.data, needed);
    CUDA_CHECK(cudaMemcpy2DAsync(pi.d_buf, pi.pitch, h_pinned_buf_, img.step,
                                  img.cols * 3, img.rows,
                                  cudaMemcpyHostToDevice, stream));
    // Sync before next iteration: h_pinned_buf_ is shared and will be
    // overwritten, so the async copy must complete first.
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Per-image detection (det engine uses batch=1 for optimal single-image speed)
  std::vector<std::vector<Box>> all_det_boxes(batch_n);
  for (int i = 0; i < batch_n; i++) {
    GpuImage gi{per_img[i].d_buf, per_img[i].pitch,
                per_img[i].rows, per_img[i].cols};
    all_det_boxes[i] = det_->run(gi, per_img[i].rows, per_img[i].cols, stream);
  }

  // Assign detection results and run angle classification per-image
  for (int i = 0; i < batch_n; i++) {
    per_img[i].boxes = std::move(all_det_boxes[i]);
    sorted_boxes(per_img[i].boxes);

    // Optional angle classification -- only classify vertical-looking boxes
    if (use_cls_) {
      vertical_box_indices_.clear();
      for (int vi = 0; vi < static_cast<int>(per_img[i].boxes.size()); ++vi) {
        if (is_vertical_box(per_img[i].boxes[vi]))
          vertical_box_indices_.push_back(vi);
      }
      if (!vertical_box_indices_.empty()) {
        vertical_boxes_buf_.clear();
        vertical_boxes_buf_.reserve(vertical_box_indices_.size());
        for (int idx : vertical_box_indices_)
          vertical_boxes_buf_.push_back(per_img[i].boxes[idx]);

        GpuImage gpu_img{per_img[i].d_buf, per_img[i].pitch,
                          per_img[i].rows, per_img[i].cols};
        cls_->run(gpu_img, vertical_boxes_buf_, stream);

        for (size_t j = 0; j < vertical_box_indices_.size(); ++j)
          per_img[i].boxes[vertical_box_indices_[j]] = vertical_boxes_buf_[j];
      }
    }
  }

  // --- Phase 2: Batched recognition across ALL images ---
  std::vector<PaddleRec::ImageCrops> image_crops(batch_n);
  for (int i = 0; i < batch_n; i++) {
    image_crops[i].img = GpuImage{per_img[i].d_buf, per_img[i].pitch,
                                  per_img[i].rows, per_img[i].cols};
    image_crops[i].boxes = std::move(per_img[i].boxes);
  }

  // Launch batched recognition on rec_stream_ (pipeline parallelism)
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));
  auto all_rec_results = rec_->run_multi(image_crops, rec_stream_);
  // Note: rec_->run_multi() syncs rec_stream_ internally for D2H + CTC decode,
  // so no additional cudaStreamSynchronize needed here.

  // --- Phase 3: Combine results and filter by drop_score ---
  constexpr float kDropScore = turbo_ocr::kDropScore;
  std::vector<std::vector<OCRResultItem>> all_results(batch_n);

  for (int i = 0; i < batch_n; i++) {
    const auto &boxes = image_crops[i].boxes;
    auto &rec_results = all_rec_results[i];
    auto &final_results = all_results[i];
    final_results.reserve(boxes.size());

    for (size_t j = 0; j < boxes.size(); ++j) {
      if (j < rec_results.size()) {
        if (rec_results[j].second < kDropScore)
          continue;
        if (rec_results[j].first.empty())
          continue;
        final_results.push_back({
          .text = std::move(rec_results[j].first),
          .confidence = rec_results[j].second,
          .box = boxes[j],
        });
      }
    }
  }

  // No cleanup needed — batch_img_bufs_ are pre-allocated and reused

  return all_results;
}
