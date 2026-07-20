#pragma once

#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <opencv2/core.hpp>

// GPU-accelerated JPEG encoder using nvJPEG.
// Encodes an interleaved 8-bit BGR cv::Mat -> JPEG bytes entirely on the GPU,
// so the entropy coding (DCT + Huffman) never touches the CPU. This is the
// mirror of NvJpegDecoder and is used by the PDF page-image export path: on a
// CPU-starved host (e.g. a single-core server) it frees the one core that the
// PDF rasterizer needs.
//
// Threading: one instance is NOT safe for concurrent encode() calls (nvJPEG
// encoder state is single-threaded). The page-image path uses one instance
// per worker thread (thread_local), so each owns its handle, state, stream
// and reusable device buffer.

namespace turbo_ocr::encode {

class NvJpegEncoder {
public:
  NvJpegEncoder() {
    nvjpegStatus_t st = nvjpegCreateSimple(&handle_);
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpegEnc] create handle failed: {}\n",
                               static_cast<int>(st));
      handle_ = nullptr;
      return;
    }
    if (nvjpegEncoderStateCreate(handle_, &enc_state_, nullptr) != NVJPEG_STATUS_SUCCESS ||
        nvjpegEncoderParamsCreate(handle_, &enc_params_, nullptr) != NVJPEG_STATUS_SUCCESS) {
      std::cerr << "[NvJpegEnc] encoder state/params create failed\n";
      destroy();
      return;
    }
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      (void)cudaGetLastError();
      stream_ = nullptr;  // fall back to default stream (still correct)
    }
    // 4:2:0 chroma subsampling matches the CPU libjpeg-turbo path (TJSAMP_420).
    nvjpegEncoderParamsSetSamplingFactors(enc_params_, NVJPEG_CSS_420, stream_);
  }

  ~NvJpegEncoder() noexcept { destroy(); }

  NvJpegEncoder(const NvJpegEncoder &) = delete;
  NvJpegEncoder &operator=(const NvJpegEncoder &) = delete;
  NvJpegEncoder(NvJpegEncoder &&) = delete;
  NvJpegEncoder &operator=(NvJpegEncoder &&) = delete;

  [[nodiscard]] bool available() const noexcept { return handle_ != nullptr; }

  // Encode an 8-bit, 3-channel interleaved BGR image to JPEG bytes.
  // Returns an empty vector on any failure (caller should fall back to CPU).
  [[nodiscard]] std::vector<uint8_t> encode(const cv::Mat &bgr, int quality) {
    if (!handle_ || bgr.empty() || bgr.type() != CV_8UC3)
      return {};

    // nvjpegEncodeImage reads a contiguous device buffer with pitch = w*3.
    cv::Mat src = bgr.isContinuous() ? bgr : bgr.clone();
    const int w = src.cols, h = src.rows;
    const size_t bytes = static_cast<size_t>(w) * h * 3;

    if (!ensure_device_buffer(bytes))
      return {};

    if (cudaMemcpyAsync(d_buf_, src.data, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
      (void)cudaGetLastError();
      return {};
    }

    nvjpegImage_t source{};
    source.channel[0] = d_buf_;
    source.pitch[0]   = static_cast<unsigned int>(w) * 3;

    nvjpegEncoderParamsSetQuality(enc_params_, quality, stream_);

    if (nvjpegEncodeImage(handle_, enc_state_, enc_params_, &source,
                          NVJPEG_INPUT_BGRI, w, h, stream_) != NVJPEG_STATUS_SUCCESS) {
      (void)cudaGetLastError();
      return {};
    }

    size_t len = 0;
    if (nvjpegEncodeRetrieveBitstream(handle_, enc_state_, nullptr, &len, stream_) != NVJPEG_STATUS_SUCCESS) {
      (void)cudaGetLastError();
      return {};
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      (void)cudaGetLastError();
      return {};
    }

    std::vector<uint8_t> out(len);
    if (nvjpegEncodeRetrieveBitstream(handle_, enc_state_, out.data(), &len, stream_) != NVJPEG_STATUS_SUCCESS) {
      (void)cudaGetLastError();
      return {};
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      (void)cudaGetLastError();
      return {};
    }
    out.resize(len);
    return out;
  }

private:
  bool ensure_device_buffer(size_t bytes) {
    if (bytes <= d_cap_) return true;
    if (d_buf_) { cudaFree(d_buf_); d_buf_ = nullptr; d_cap_ = 0; }
    if (cudaMalloc(reinterpret_cast<void **>(&d_buf_), bytes) != cudaSuccess) {
      (void)cudaGetLastError();
      d_buf_ = nullptr;
      return false;
    }
    d_cap_ = bytes;
    return true;
  }

  void destroy() noexcept {
    // A thread_local encoder's dtor can run at process exit, AFTER the CUDA
    // runtime has torn the context down. Calling nvjpeg*Destroy against a dead
    // context faults (cudaErrorCudartUnloading) and a swallowed
    // cudaGetLastError() can't undo that — so if the context is already gone,
    // leak the handles (the process is exiting anyway) rather than fault.
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
      (void)cudaGetLastError();
      handle_ = nullptr; enc_state_ = nullptr; enc_params_ = nullptr;
      stream_ = nullptr; d_buf_ = nullptr; d_cap_ = 0;
      return;
    }
    // Context alive: drain any in-flight encode, then destroy nvJPEG objects
    // (built against handle_) BEFORE the stream/buffer they ran on.
    if (stream_) (void)cudaStreamSynchronize(stream_);
    if (enc_params_) { nvjpegEncoderParamsDestroy(enc_params_); enc_params_ = nullptr; }
    if (enc_state_)  { nvjpegEncoderStateDestroy(enc_state_);   enc_state_  = nullptr; }
    if (handle_)     { nvjpegDestroy(handle_);                  handle_     = nullptr; }
    if (d_buf_)      { cudaFree(d_buf_);            d_buf_ = nullptr; d_cap_ = 0; }
    if (stream_)     { cudaStreamDestroy(stream_);  stream_ = nullptr; }
    (void)cudaGetLastError();
  }

  nvjpegHandle_t        handle_     = nullptr;
  nvjpegEncoderState_t  enc_state_  = nullptr;
  nvjpegEncoderParams_t enc_params_ = nullptr;
  cudaStream_t          stream_     = nullptr;  // nullptr == default stream
  unsigned char        *d_buf_      = nullptr;
  size_t                d_cap_      = 0;
};

} // namespace turbo_ocr::encode
