#pragma once

#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/base/env_utils.h"

// GPU-accelerated JPEG decoder using nvJPEG.
// Decodes JPEG bytes -> cv::Mat (BGR, on CPU) significantly faster than cv::imdecode.
// Falls back to cv::imdecode for non-JPEG formats.

namespace turbo_ocr::decode {

class NvJpegDecoder {
public:
  NvJpegDecoder() {
    // Try NVDEC hardware decoder first (offloads Huffman to dedicated HW)
    nvjpegStatus_t st = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, 0, &handle_);
    if (st == NVJPEG_STATUS_SUCCESS) {
      std::cerr << "[NvJpeg] Using HARDWARE (NVDEC) backend\n";
    } else {
      // Fallback to GPU_HYBRID (GPU-assisted Huffman, frees compute cores)
      st = nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr, nullptr, 0, &handle_);
      if (st == NVJPEG_STATUS_SUCCESS) {
        std::cerr << "[NvJpeg] Using GPU_HYBRID backend\n";
      } else {
        // Final fallback to simple (default hybrid CPU+GPU)
        st = nvjpegCreateSimple(&handle_);
        if (st == NVJPEG_STATUS_SUCCESS) {
          std::cerr << "[NvJpeg] Using default (simple) backend\n";
        }
      }
    }
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Failed to create handle: {}", static_cast<int>(st)) << '\n';
      handle_ = nullptr;
      return;
    }
    st = nvjpegJpegStateCreate(handle_, &state_);
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Failed to create state: {}", static_cast<int>(st)) << '\n';
      nvjpegDestroy(handle_);
      handle_ = nullptr;
      return;
    }

    // nvJPEG requires DEVICE output pointers. On HMM/ATS systems
    // (cudaDevAttrPageableMemoryAccess=1) any host pointer IS a valid device
    // address, so decoding straight into cv::Mat host memory is both legal
    // and the fastest path (one PCIe traversal, no scratch). Everywhere else
    // that same pointer faults from GPU kernels — the GitHub #22 crash — so
    // those machines take the device-scratch + copy-back path.
    // NVJPEG_DEVICE_COPY=1 forces the scratch path (ops escape hatch/tests).
    int dev = 0, pageable = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, dev) ==
            cudaSuccess) {
      direct_host_output_ = pageable == 1;
    } else {
      (void)cudaGetLastError();
    }
    if (std::string env = turbo_ocr::env::env_or("NVJPEG_DEVICE_COPY", "");
        !env.empty() && env[0] == '1')
      direct_host_output_ = false;
  }

  ~NvJpegDecoder() noexcept {
    // Safety contract: every public call (decode/batch_decode/decode_to_gpu
    // when caller-synced) returns only after its CUDA stream has either
    // synchronized or been queued behind the caller's sync. So when this
    // dtor runs there is no in-flight work referencing state_/handle_.
    // If a future caller adds an unsynchronized path, sync the default
    // stream HERE before destroying handles to keep the contract.
    //
    // A thread_local decoder's dtor can run at process exit, AFTER the CUDA
    // runtime has torn the context down (same hazard as NvJpegEncoder). If
    // the context is already gone, leak the handles (the process is exiting
    // anyway) rather than fault against a dead runtime.
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
      (void)cudaGetLastError();
      return;
    }
    if (staging_) (void)cudaFreeHost(staging_);
    if (state_) nvjpegJpegStateDestroy(state_);
    if (handle_) nvjpegDestroy(handle_);
  }

  // Non-copyable, non-movable (owns GPU resources)
  NvJpegDecoder(const NvJpegDecoder &) = delete;
  NvJpegDecoder &operator=(const NvJpegDecoder &) = delete;
  NvJpegDecoder(NvJpegDecoder &&) = delete;
  NvJpegDecoder &operator=(NvJpegDecoder &&) = delete;

  // Decode JPEG bytes to BGR cv::Mat. Returns empty Mat on failure.
  //
  // nvJPEG requires the nvjpegImage_t channel pointers to be DEVICE memory
  // (for nvjpegDecode as much as for nvjpegDecodeBatched), so the decode
  // lands in a stream-ordered device scratch buffer and is copied back into
  // the host Mat on the same stream. Handing nvJPEG a host pointer is UB:
  // the batched path dereferences it from GPU kernels and poisons the CUDA
  // context with an illegal memory access (GitHub #22).
  [[nodiscard]] cv::Mat decode(const unsigned char *data, size_t len, cudaStream_t stream = 0) {
    if (!handle_ || !is_jpeg(data, len))
      return {};  // Not JPEG -- caller should fall back to cv::imdecode

    int nComponents;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegStatus_t st = nvjpegGetImageInfo(handle_, data, len,
                                            &nComponents, &subsampling,
                                            widths, heights);
    if (st != NVJPEG_STATUS_SUCCESS)
      return {};

    const int w = widths[0], h = heights[0];
    const size_t bytes = static_cast<size_t>(h) * static_cast<size_t>(w) * 3;
    cv::Mat result(h, w, CV_8UC3);

    if (direct_host_output_) {
      // HMM: the Mat's host memory is a valid device address — decode into it.
      if (!decode_to_gpu(data, len, result.data, static_cast<size_t>(w) * 3, w, h, stream))
        return {};
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        (void)cudaGetLastError();
        return {};
      }
      return result;
    }

    DeviceBuffer dbuf(bytes, stream);
    if (!dbuf.ptr)
      return {};

    if (!decode_to_gpu(data, len, dbuf.ptr, static_cast<size_t>(w) * 3, w, h, stream))
      return {};

    // Copy back via pinned staging when available (full-bandwidth D2H),
    // falling back to a direct pageable copy.
    unsigned char *stage = ensure_staging(bytes);
    if (cudaMemcpyAsync(stage ? stage : result.data, dbuf.ptr, bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      (void)cudaGetLastError();
      return {};
    }
    // Sync before returning so the host copy is materialized and safe to hand
    // to other threads, and so the scratch free below is stream-ordered
    // behind the copy.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      (void)cudaGetLastError();
      return {};
    }
    if (stage) std::memcpy(result.data, stage, bytes);
    return result;
  }

  // Check if data is a JPEG (starts with FF D8).
  [[nodiscard]] static bool is_jpeg(const unsigned char *data, size_t len) noexcept {
    return len >= 2 && data[0] == 0xFF && data[1] == 0xD8;
  }

  // Get JPEG image dimensions without decoding.
  // Returns {width, height} or {0, 0} on failure.
  [[nodiscard]] std::pair<int, int> get_dimensions(const unsigned char *data, size_t len) {
    if (!handle_ || !is_jpeg(data, len))
      return {0, 0};
    int nComponents;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegStatus_t st = nvjpegGetImageInfo(handle_, data, len,
                                            &nComponents, &subsampling,
                                            widths, heights);
    if (st != NVJPEG_STATUS_SUCCESS)
      return {0, 0};
    return {widths[0], heights[0]};
  }

  // Decode JPEG bytes directly to a GPU buffer (device memory).
  // The caller provides a pre-allocated device buffer with the given pitch.
  // The buffer must be large enough for h * pitch bytes (pitch >= w * 3).
  // Returns true on success. The decode is async on the given stream;
  // the caller must synchronize the stream before reading the output.
  // w/h describe the caller's buffer-size contract (see above); nvjpeg infers
  // the actual decode dimensions, so they are unused inside this method.
  [[nodiscard]] bool decode_to_gpu(const unsigned char *data, size_t len,
                                    void *d_output, size_t pitch,
                                    int /*w*/, int /*h*/,
                                    cudaStream_t stream = 0) {
    if (!handle_ || !is_jpeg(data, len))
      return false;

    nvjpegImage_t output;
    output.channel[0] = static_cast<unsigned char *>(d_output);
    output.pitch[0] = static_cast<unsigned int>(pitch);
    for (int i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
      output.channel[i] = nullptr;
      output.pitch[i] = 0;
    }

    // Decode directly to GPU memory (interleaved BGR)
    nvjpegStatus_t st = nvjpegDecode(handle_, state_, data, len,
                                      NVJPEG_OUTPUT_BGRI, &output, stream);
    return st == NVJPEG_STATUS_SUCCESS;
  }

  // Batch decode multiple JPEG images in one nvjpegDecodeBatched call.
  // Input: vector of (data, length) pairs — must all be valid JPEGs.
  // Returns: vector of cv::Mat (BGR). Failed images get empty Mat.
  // Non-JPEG images should be filtered out by the caller and decoded separately.
  //
  // All batch outputs decode into ONE stream-ordered device allocation
  // (per-image 256B-aligned slices) and are copied back to host afterwards:
  // nvjpegDecodeBatched writes through the channel pointers from GPU kernels,
  // so host Mats here caused sticky illegal-memory-access faults (GitHub #22).
  [[nodiscard]] std::vector<cv::Mat> batch_decode(
      const std::vector<std::pair<const unsigned char *, size_t>> &jpeg_buffers,
      cudaStream_t stream = 0) {

    size_t n = jpeg_buffers.size();
    std::vector<cv::Mat> results(n);

    if (!handle_ || n == 0)
      return results;

    auto fallback_single = [&] {
      for (size_t i = 0; i < n; ++i)
        results[i] = decode(jpeg_buffers[i].first, jpeg_buffers[i].second, stream);
      return results;
    };

    // Parse every header first: per-image dims + total scratch size.
    struct Dims { int w, h; size_t offset, bytes; };
    std::vector<Dims> dims(n);
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) {
      auto [data, len] = jpeg_buffers[i];
      if (!is_jpeg(data, len))
        return fallback_single();

      int nComponents;
      nvjpegChromaSubsampling_t subsampling;
      int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
      if (nvjpegGetImageInfo(handle_, data, len, &nComponents, &subsampling,
                             widths, heights) != NVJPEG_STATUS_SUCCESS)
        return fallback_single();

      dims[i].w = widths[0];
      dims[i].h = heights[0];
      dims[i].bytes = static_cast<size_t>(dims[i].h) * static_cast<size_t>(dims[i].w) * 3;
      dims[i].offset = total;
      total += (dims[i].bytes + 255) & ~size_t{255};
    }

    DeviceBuffer dbuf(direct_host_output_ ? size_t{0} : total, stream);
    if (!direct_host_output_ && !dbuf.ptr) {
      std::cerr << std::format("[NvJpeg] Batch scratch alloc failed ({} bytes), "
                               "falling back to single decode", total) << '\n';
      return fallback_single();
    }

    // Initialize batch decode state
    nvjpegStatus_t st = nvjpegDecodeBatchedInitialize(
        handle_, state_, static_cast<int>(n), 1 /*max_cpu_threads*/,
        NVJPEG_OUTPUT_BGRI);
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Batch init failed: {}, falling back to single decode",
                               static_cast<int>(st)) << '\n';
      return fallback_single();
    }

    std::vector<nvjpegImage_t> outputs(n);
    std::vector<const unsigned char *> data_ptrs(n);
    std::vector<size_t> lengths(n);
    for (size_t i = 0; i < n; ++i) {
      if (direct_host_output_) {
        // HMM: Mat host memory is a valid device address (see ctor comment).
        results[i] = cv::Mat(dims[i].h, dims[i].w, CV_8UC3);
        outputs[i].channel[0] = results[i].data;
      } else {
        outputs[i].channel[0] = static_cast<unsigned char *>(dbuf.ptr) + dims[i].offset;
      }
      outputs[i].pitch[0] = static_cast<unsigned int>(dims[i].w * 3);
      for (int c = 1; c < NVJPEG_MAX_COMPONENT; c++) {
        outputs[i].channel[c] = nullptr;
        outputs[i].pitch[c] = 0;
      }
      data_ptrs[i] = jpeg_buffers[i].first;
      lengths[i] = jpeg_buffers[i].second;
    }

    // Batch decode all JPEGs in one call
    st = nvjpegDecodeBatched(handle_, state_,
                              data_ptrs.data(), lengths.data(),
                              outputs.data(), stream);
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Batch decode failed: {}, falling back to single decode",
                               static_cast<int>(st)) << '\n';
      return fallback_single();
    }

    // Contract-safe path: copy back through the persistent pinned staging
    // buffer when possible — one full-bandwidth D2H of the whole scratch,
    // then host memcpys into the Mats. Direct mode decoded into the Mats
    // already and only needs the sync below.
    bool copies_ok = true;
    unsigned char *stage = nullptr;
    if (!direct_host_output_) {
      stage = ensure_staging(total);
      if (stage) {
        copies_ok = cudaMemcpyAsync(stage, dbuf.ptr, total,
                                    cudaMemcpyDeviceToHost, stream) == cudaSuccess;
      } else {
        for (size_t i = 0; i < n && copies_ok; ++i) {
          results[i] = cv::Mat(dims[i].h, dims[i].w, CV_8UC3);
          copies_ok = cudaMemcpyAsync(results[i].data, outputs[i].channel[0],
                                      dims[i].bytes, cudaMemcpyDeviceToHost,
                                      stream) == cudaSuccess;
        }
      }
    }

    // Sync so the host copies are materialized and the scratch free (in
    // ~DeviceBuffer, stream-ordered behind the copies) can never race the
    // decode/copy work.
    if (cudaStreamSynchronize(stream) != cudaSuccess || !copies_ok) {
      cudaError_t err = cudaGetLastError();
      std::cerr << std::format("[NvJpeg] Batch copy-back failed: {}",
                               cudaGetErrorString(err)) << '\n';
      for (auto &m : results) m.release();  // never hand out unmaterialized Mats
      return results;
    }

    if (stage) {
      for (size_t i = 0; i < n; ++i) {
        results[i] = cv::Mat(dims[i].h, dims[i].w, CV_8UC3);
        std::memcpy(results[i].data, stage + dims[i].offset, dims[i].bytes);
      }
    }

    return results;
  }

  [[nodiscard]] bool available() const noexcept { return handle_ != nullptr; }

private:
  // Stream-ordered device scratch (cudaMallocAsync pools make per-request
  // allocation cheap). ptr == nullptr signals allocation failure.
  struct DeviceBuffer {
    void *ptr = nullptr;
    cudaStream_t stream;
    DeviceBuffer(size_t bytes, cudaStream_t s) : stream(s) {
      if (bytes == 0 || cudaMallocAsync(&ptr, bytes, s) != cudaSuccess) {
        (void)cudaGetLastError();
        ptr = nullptr;
      }
    }
    ~DeviceBuffer() {
      if (ptr && cudaFreeAsync(ptr, stream) != cudaSuccess)
        (void)cudaGetLastError();
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  };

  // Persistent pinned staging for D2H copy-back (grown geometrically, freed
  // in the dtor behind the dead-context guard). nullptr return = allocation
  // failed; callers fall back to direct pageable copies.
  [[nodiscard]] unsigned char *ensure_staging(size_t bytes) {
    if (bytes <= staging_cap_) return staging_;
    if (staging_) {
      (void)cudaFreeHost(staging_);
      staging_ = nullptr;
      staging_cap_ = 0;
    }
    size_t want = std::max(bytes, staging_cap_ * 2);
    void *p = nullptr;
    if (cudaMallocHost(&p, want) != cudaSuccess) {
      (void)cudaGetLastError();
      return nullptr;
    }
    staging_ = static_cast<unsigned char *>(p);
    staging_cap_ = want;
    return staging_;
  }

  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
  unsigned char *staging_ = nullptr;
  size_t staging_cap_ = 0;
  bool direct_host_output_ = false;
};

} // namespace turbo_ocr::decode
