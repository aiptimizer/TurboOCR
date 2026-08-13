// NvDetector / NvRecognizer / NvClassifier / NvLayout implementation. Every
// method converts the interface vocabulary and forwards to the wrapped class.
// No device logic is re-implemented.

#include "nvidia/stages/nv_stages.h"

#include "nvidia/support/cuda_common.h" // to_gpu_image, cuda_stream

namespace turbo_ocr::nvidia {

// ---- NvDetector ------------------------------------------------------------

bool NvDetector::load(const std::string &model_path) {
  ready_ = det_.load_model(model_path); // default resize/db config + env overrides
  return ready_;
}

std::vector<turbo_ocr::Box> NvDetector::run(const backend::ImageView &img,
                                            int orig_h, int orig_w,
                                            backend::DeviceQueue &queue) {
  return det_.run(to_gpu_image(img), orig_h, orig_w, cuda_stream(queue));
}

NvDetector::~NvDetector() {
  // Best-effort: a destructor cannot report, and by teardown the process is
  // either exiting or discarding this replica.
  if (upload_done_) cudaEventDestroy(upload_done_);
  if (det_stream_) cudaStreamDestroy(det_stream_);
}

backend::BoxesFuture NvDetector::enqueue(const backend::ImageView &img,
                                         int orig_h, int orig_w,
                                         backend::DeviceQueue &queue) {
  const cudaStream_t caller = cuda_stream(queue);
  // Lazily created so a replica that never takes the async path pays nothing.
  // cudaStreamNonBlocking: this lane must not implicitly serialize against the
  // legacy default stream, which would defeat the overlap it exists for.
  if (!det_stream_ &&
      cudaStreamCreateWithFlags(&det_stream_, cudaStreamNonBlocking) != cudaSuccess)
    return backend::BoxesFuture::ready(det_.run(to_gpu_image(img), orig_h, orig_w,
                                                caller));
  if (!upload_done_ &&
      cudaEventCreateWithFlags(&upload_done_, cudaEventDisableTiming) != cudaSuccess)
    return backend::BoxesFuture::ready(det_.run(to_gpu_image(img), orig_h, orig_w,
                                                caller));

  // ORDER THE LANE AFTER THE CALLER'S. The page was staged H2D on `caller`;
  // recording here and waiting there is what stops the forward pass from
  // reading the buffer before the copy lands. Recorded at the CURRENT point of
  // the caller's lane, so it covers exactly the work already submitted.
  CUDA_CHECK(cudaEventRecord(upload_done_, caller));
  CUDA_CHECK(cudaStreamWaitEvent(det_stream_, upload_done_, 0));

  det_.submit_forward(to_gpu_image(img), orig_h, orig_w, det_stream_);

  // `img` must stay alive until collect() runs — the seam contract puts that on
  // the caller, and run_pipelined()'s staging ring is what honours it.
  return backend::BoxesFuture([this, orig_h, orig_w] {
    return det_.collect_boxes(orig_h, orig_w, det_stream_);
  });
}

std::vector<std::vector<turbo_ocr::Box>>
NvDetector::run_batch(const std::vector<backend::ImageView> &imgs,
                      const std::vector<std::pair<int, int>> &orig_dims,
                      backend::DeviceQueue &queue) {
  std::vector<decode::GpuImage> gpu_imgs;
  gpu_imgs.reserve(imgs.size());
  for (const auto &v : imgs)
    gpu_imgs.push_back(to_gpu_image(v));
  return det_.run_batch(gpu_imgs, orig_dims, cuda_stream(queue));
}

// ---- NvRecognizer ----------------------------------------------------------

bool NvRecognizer::load(const std::string &model_path) {
  ready_ = rec_.load_model(model_path);
  return ready_;
}

bool NvRecognizer::load_dict(const std::string &dict_path) {
  return rec_.load_dict(dict_path);
}

std::vector<backend::RecResult>
NvRecognizer::run(const backend::ImageView &img,
                  const std::vector<turbo_ocr::Box> &boxes,
                  backend::DeviceQueue &queue) {
  // PaddleRec already returns vector<pair<string,float>> == backend::RecResult.
  return rec_.run(to_gpu_image(img), boxes, cuda_stream(queue));
}

std::vector<std::vector<backend::RecResult>>
NvRecognizer::run_multi(const std::vector<backend::ImageCrops> &items,
                        backend::DeviceQueue &queue) {
  std::vector<recognition::PaddleRec::ImageCrops> native_items;
  native_items.reserve(items.size());
  for (const auto &it : items)
    native_items.push_back(
        recognition::PaddleRec::ImageCrops{to_gpu_image(it.img), it.boxes});
  return rec_.run_multi(native_items, cuda_stream(queue));
}

void NvRecognizer::warmup(backend::DeviceQueue &queue) {
  rec_.bake_graphs(cuda_stream(queue));
}

// ---- NvClassifier ----------------------------------------------------------

bool NvClassifier::load(const std::string &model_path) {
  ready_ = cls_.load_model(model_path);
  return ready_;
}

void NvClassifier::run(const backend::ImageView &img,
                       std::vector<turbo_ocr::Box> &boxes,
                       backend::DeviceQueue &queue) {
  cls_.run(to_gpu_image(img), boxes, cuda_stream(queue));
}

// ---- NvLayout --------------------------------------------------------------

bool NvLayout::load(const std::string &model_path) {
  ready_ = layout_.load_model(model_path);
  return ready_;
}

std::vector<turbo_ocr::layout::LayoutBox>
NvLayout::run(const backend::ImageView &img, int orig_h, int orig_w,
              float score_threshold, backend::DeviceQueue &queue) {
  // Synchronous convenience: enqueue then immediately collect.
  if (!layout_.enqueue(to_gpu_image(img), orig_h, orig_w, cuda_stream(queue)))
    return {};
  return layout_.collect(score_threshold);
}

backend::LayoutFuture NvLayout::enqueue(const backend::ImageView &img,
                                        int orig_h, int orig_w,
                                        float score_threshold,
                                        backend::DeviceQueue &queue) {
  // An invalid future on a failed submission, NOT an empty-boxes one: the
  // pipeline reads validity as "a collect is owed" and falls back to the
  // blocking run() otherwise, so a page whose enqueue bounced still gets a
  // layout instead of silently coming back region-less.
  if (!layout_.enqueue(to_gpu_image(img), orig_h, orig_w, cuda_stream(queue)))
    return {};
  // Capturing `this` is safe: the pipeline owns the stage and the future, and
  // the single-slot contract keeps the future from outliving the submission it
  // names.
  return backend::LayoutFuture(
      [this, score_threshold] { return layout_.collect(score_threshold); });
}

} // namespace turbo_ocr::nvidia
