#include "turbo_ocr/classification/paddle_cls.h"
#include "turbo_ocr/kernels/kernels.h"

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/geometry/perspective.h"

#include <algorithm>
#include <cmath>

using namespace turbo_ocr::classification;
using turbo_ocr::engine::TrtEngine;
using turbo_ocr::Box;
using turbo_ocr::GpuImage;

bool PaddleCls::load_model(const std::string &model_path) {
  engine_ = std::make_unique<TrtEngine>(model_path);
  return engine_->load();
}

void PaddleCls::allocate_buffers() {
  if (buffers_allocated_)
    return;

  int bs = kClsBatchNum;

  size_t input_elems = static_cast<size_t>(bs) * 3 * kClsImageH * kClsImageW;
  d_batch_input_ = CudaPtr<float>(input_elems);

  size_t output_elems = static_cast<size_t>(bs) * 2;
  d_output_ = CudaPtr<float>(output_elems);

  d_M_invs_ = CudaPtr<float>(bs * 9);
  d_crop_widths_ = CudaPtr<int>(bs);

  h_output_ = CudaHostPtr<float>(output_elems);
  h_M_invs_ = CudaHostPtr<float>(bs * 9);
  h_crop_widths_ = CudaHostPtr<int>(bs);

  engine_->bind_io(d_batch_input_.get(), d_output_.get());

  buffers_allocated_ = true;
}

void PaddleCls::run(const GpuImage &img, std::vector<Box> &boxes,
                    cudaStream_t stream) {
  if (boxes.empty()) [[unlikely]]
    return;

  allocate_buffers();

  int total_boxes = static_cast<int>(boxes.size());

  for (int beg = 0; beg < total_boxes; beg += kClsBatchNum) {
    int cur_batch = std::min(kClsBatchNum, total_boxes - beg);

    for (int j = 0; j < cur_batch; ++j) {
      auto ct = turbo_ocr::compute_crop_transform(boxes[beg + j], kClsImageH, kClsImageW);
      h_crop_widths_.get()[j] = ct.crop_width;
      std::copy_n(ct.M_inv, 9, h_M_invs_.get() + j * 9);
    }

    CUDA_CHECK(cudaMemcpyAsync(d_M_invs_.get(), h_M_invs_.get(), cur_batch * 9 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_crop_widths_.get(), h_crop_widths_.get(), cur_batch * sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    // MEASURED: the shipped textline-ori ONNX scores higher with rec's
    // (x/255-0.5)/0.5 (85.37% FUNSD) than with ImageNet mean/std (85.30%), so the
    // "should be ImageNet" review claim does NOT hold for this exported model.
    // Uses the default (rec) normalization. (Warp kernel is now parameterized for
    // per-stage norm if a future model needs it.)
    turbo_ocr::kernels::cuda_batch_roi_warp(img, d_M_invs_.get(), d_crop_widths_.get(),
                                     d_batch_input_.get(), cur_batch, kClsImageH,
                                     kClsImageW, stream);

    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = cur_batch;
    input_dims.d[1] = 3;
    input_dims.d[2] = kClsImageH;
    input_dims.d[3] = kClsImageW;

    // I/O already bound — just set dims and infer
    if (!engine_->infer_dynamic(input_dims, stream)) {
      throw turbo_ocr::InferenceError("Classification TRT inference failed");
    }

    CUDA_CHECK(cudaMemcpyAsync(h_output_.get(), d_output_.get(), cur_batch * 2 * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int j = 0; j < cur_batch; ++j) {
      float score_0 = h_output_.get()[j * 2 + 0];
      float score_180 = h_output_.get()[j * 2 + 1];

      if (score_180 > score_0 && score_180 > kClsThresh) {
        auto &box = boxes[beg + j];
        std::swap(box[0], box[2]);
        std::swap(box[1], box[3]);
      }
    }
  }
}
