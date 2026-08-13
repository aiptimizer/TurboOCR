#include "nvidia/stages/doc_orientation.h"

#include <algorithm>
#include <iostream>

#include "turbo_ocr/analysis/classification/doc_orientation_common.h"
#include "nvidia/support/cuda_check.h"

using namespace turbo_ocr::classification;
using turbo_ocr::engine::TrtEngine;

bool DocOrientation::load_model(const std::string &model_path) {
  engine_ = std::make_unique<TrtEngine>(model_path);
  return engine_->load();
}

void DocOrientation::allocate_buffers() {
  if (buffers_allocated_) return;
  const size_t in_elems =
      static_cast<size_t>(3) * kDocOriSize * kDocOriSize;
  d_input_ = CudaPtr<float>(in_elems);
  d_output_ = CudaPtr<float>(4);
  h_input_ = CudaHostPtr<float>(in_elems);
  h_output_ = CudaHostPtr<float>(4);
  engine_->bind_io(d_input_.get(), d_output_.get());
  buffers_allocated_ = true;
}

int DocOrientation::detect(const cv::Mat &bgr, cudaStream_t stream) {
  if (!engine_ || bgr.empty() || bgr.type() != CV_8UC3) return 0;
  allocate_buffers();

  std::vector<float> input = doc_ori_preprocess(bgr);
  std::copy(input.begin(), input.end(), h_input_.get());
  const size_t bytes = input.size() * sizeof(float);
  CUDA_CHECK(cudaMemcpyAsync(d_input_.get(), h_input_.get(), bytes,
                              cudaMemcpyHostToDevice, stream));

  nvinfer1::Dims dims;
  dims.nbDims = 4;
  dims.d[0] = 1;
  dims.d[1] = 3;
  dims.d[2] = kDocOriSize;
  dims.d[3] = kDocOriSize;
  if (!engine_->infer_dynamic(dims, stream)) {
    // Fail loud: a real inference failure must not be indistinguishable from a
    // confidently-upright page (no-silent-failure).
    std::cerr << "[doc-ori] orientation inference failed; page left un-rotated\n";
    return 0; // best-effort: no rotate
  }

  CUDA_CHECK(cudaMemcpyAsync(h_output_.get(), d_output_.get(),
                              4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return doc_ori_label(h_output_.get());
}
