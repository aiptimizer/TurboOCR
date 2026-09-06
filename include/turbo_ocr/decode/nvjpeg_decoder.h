#pragma once

#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/decode/jpeg_codec.h"

// GPU-accelerated JPEG decoder using nvJPEG.
// Decodes JPEG bytes -> cv::Mat (BGR, on CPU) significantly faster than cv::imdecode.
// Falls back to cv::imdecode for non-JPEG formats.

namespace turbo_ocr::decode {

class NvJpegDecoder {
public:
  // Which nvJPEG backend to prefer. Hardware (the NVJPG engine) decodes
  // baseline/extended-sequential JPEG fastest and off the SMs; Hybrid runs
  // Huffman on the CPU or GPU and the rest on the GPU, and also decodes what
  // the hardware path reports as unsupported (progressive, arithmetic).
  // Replicas keep one of each: hardware first, hybrid for the rest, so a
  // bitstream reaches the host codec only when neither GPU backend takes it.
  enum class Backend { Hardware, Hybrid };

  explicit NvJpegDecoder(Backend preferred = Backend::Hardware) : preferred_(preferred) {
    // nvJPEG's internal scratch (the per-decoder device and pinned buffers it
    // allocates on first use) comes through OUR allocators: device memory
    // from the stream-ordered pool (cudaMallocAsync, whose release threshold
    // the server sets) and pinned host memory from cudaHostAlloc. Decoder
    // memory is then part of the same explicit budget as everything else
    // instead of whatever cudaMalloc happened to grab.
    static nvjpegDevAllocatorV2_t dev_alloc{&dev_malloc_v2_, &dev_free_v2_, nullptr};
    static nvjpegPinnedAllocatorV2_t pinned_alloc{&pinned_malloc_v2_, &pinned_free_v2_, nullptr};
    // Try NVDEC hardware decoder first (offloads Huffman to dedicated HW)
    nvjpegStatus_t st = NVJPEG_STATUS_NOT_INITIALIZED;
    if (preferred_ == Backend::Hardware)
      st = nvjpegCreateExV2(NVJPEG_BACKEND_HARDWARE, &dev_alloc, &pinned_alloc, 0, &handle_);
    if (st == NVJPEG_STATUS_SUCCESS) {
      // The HARDWARE backend is accepted on every GPU, but only devices with
      // an NVJPG engine (A100/H100/GB10 class) decode on it; elsewhere nvJPEG
      // runs the same GPU kernels as the hybrid backend. Say which, so a log
      // line never claims engine offload the card does not have.
      unsigned engines = 0, cores = 0;
      if (nvjpegGetHardwareDecoderInfo(handle_, &engines, &cores) != NVJPEG_STATUS_SUCCESS)
        engines = 0;
      if (engines > 0)
        TOCR_LOG_INFO("nvJPEG hardware backend", "nvjpg_engines", engines, "cores_per_engine", cores);
      else
        TOCR_LOG_INFO("nvJPEG hardware backend without an NVJPG engine on this GPU; decoding on GPU kernels");
    } else {
      // Fallback to GPU_HYBRID (GPU-assisted Huffman, frees compute cores)
      st = nvjpegCreateExV2(NVJPEG_BACKEND_GPU_HYBRID, &dev_alloc, &pinned_alloc, 0, &handle_);
      if (st == NVJPEG_STATUS_SUCCESS) {
        TOCR_LOG_INFO("nvJPEG GPU-hybrid backend");
      } else {
        // Final fallback to simple (default hybrid CPU+GPU)
        st = nvjpegCreateSimple(&handle_);
        if (st == NVJPEG_STATUS_SUCCESS) {
          TOCR_LOG_INFO("nvJPEG default backend");
        }
      }
    }
    if (st != NVJPEG_STATUS_SUCCESS) {
      TOCR_LOG_ERROR("nvJPEG handle creation failed", "status", static_cast<int>(st));
      handle_ = nullptr;
      return;
    }
    st = nvjpegJpegStateCreate(handle_, &state_);
    if (st != NVJPEG_STATUS_SUCCESS) {
      TOCR_LOG_ERROR("nvJPEG state creation failed", "status", static_cast<int>(st));
      nvjpegDestroy(handle_);
      handle_ = nullptr;
      return;
    }

    // Remember the device so bind_calling_thread_() can attach any thread
    // that later uses this decoder to the same context (see decode()).
    if (cudaGetDevice(&device_) != cudaSuccess) {
      (void)cudaGetLastError();
      device_ = 0;
    }

    // Output always lands in a stream-ordered device scratch and is copied
    // back through pinned staging. Decoding straight into pageable host
    // memory on HMM/ATS systems (where a host pointer is a valid device
    // address) was tried and removed: the GPU then writes, through page
    // faults, into memory that a general-purpose allocator is purging and
    // reusing underneath it, which is where multi-minute UVM stalls came
    // from under jemalloc. The copy costs a memcpy; the fault storm cost the
    // process.
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

  // A host-side decode and how it went. `image` is empty unless status is Ok.
  // `nvjpeg_status` is the raw nvjpegStatus_t for logs.
  struct HostDecode {
    cv::Mat image;
    JpegDecodeStatus status = JpegDecodeStatus::Failed;
    int nvjpeg_status = 0;
  };

  // Decode JPEG bytes to a BGR cv::Mat on the GPU.
  //
  // nvJPEG requires the nvjpegImage_t channel pointers to be DEVICE memory
  // (for nvjpegDecode as much as for nvjpegDecodeBatched), so the decode
  // lands in a stream-ordered device scratch buffer and is copied back into
  // the host Mat on the same stream. Handing nvJPEG a host pointer is UB:
  // the batched path dereferences it from GPU kernels and poisons the CUDA
  // context with an illegal memory access (GitHub #22).
  [[nodiscard]] HostDecode decode(const unsigned char *data, size_t len, cudaStream_t stream = 0) {
    if (!handle_) return {{}, JpegDecodeStatus::Failed, nvjpeg_status::kSuccess};
    if (!is_jpeg(data, len)) return {{}, JpegDecodeStatus::Unsupported, nvjpeg_status::kSuccess};
    bind_calling_thread_();

    int nComponents;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegStatus_t st = nvjpegGetImageInfo(handle_, data, len,
                                            &nComponents, &subsampling,
                                            widths, heights);
    if (st != NVJPEG_STATUS_SUCCESS)
      return {{}, classify_nvjpeg_status(static_cast<int>(st)), static_cast<int>(st)};

    const int w = widths[0], h = heights[0];
    const size_t bytes = static_cast<size_t>(h) * static_cast<size_t>(w) * 3;
    cv::Mat result(h, w, CV_8UC3);

    DeviceBuffer dbuf(bytes, stream);
    if (!dbuf.ptr)
      return {{}, JpegDecodeStatus::Failed, nvjpeg_status::kSuccess};

    const JpegDecodeStatus ds =
        decode_to_gpu(data, len, dbuf.ptr, static_cast<size_t>(w) * 3, w, h, stream);
    if (ds != JpegDecodeStatus::Ok) return {{}, ds, last_nvjpeg_status_};

    // Copy back via pinned staging when available (full-bandwidth D2H),
    // falling back to a direct pageable copy.
    unsigned char *stage = ensure_staging(bytes);
    if (cudaMemcpyAsync(stage ? stage : result.data, dbuf.ptr, bytes,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      (void)cudaGetLastError();
      return {{}, JpegDecodeStatus::Failed, nvjpeg_status::kSuccess};
    }
    // Sync before returning so the host copy is materialized and safe to hand
    // to other threads, and so the scratch free below is stream-ordered
    // behind the copy.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      (void)cudaGetLastError();
      return {{}, JpegDecodeStatus::Failed, nvjpeg_status::kSuccess};
    }
    if (stage) std::memcpy(result.data, stage, bytes);
    return {std::move(result), JpegDecodeStatus::Ok, nvjpeg_status::kSuccess};
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
    bind_calling_thread_();
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
  // The decode is async on the given stream; the caller must synchronize the
  // stream before reading the output. w/h describe the caller's buffer-size
  // contract (see above); nvjpeg infers the actual decode dimensions, so they
  // are unused inside this method. The raw nvjpegStatus_t of the last call is
  // kept in last_nvjpeg_status() for error messages.
  [[nodiscard]] JpegDecodeStatus decode_to_gpu(const unsigned char *data, size_t len,
                                                void *d_output, size_t pitch,
                                                int /*w*/, int /*h*/,
                                                cudaStream_t stream = 0) {
    if (!handle_) return JpegDecodeStatus::Failed;
    if (!is_jpeg(data, len)) return JpegDecodeStatus::Unsupported;
    bind_calling_thread_();

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
    last_nvjpeg_status_ = static_cast<int>(st);
    return classify_nvjpeg_status(last_nvjpeg_status_);
  }

  [[nodiscard]] int last_nvjpeg_status() const noexcept { return last_nvjpeg_status_; }

  // Batch decode multiple JPEG images in one nvjpegDecodeBatched call.
  // Input: vector of (data, length) pairs — must all be valid JPEGs.
  // Returns one HostDecode per input, in order. When the batched call cannot
  // serve the set as a whole (an unsupported member, batch init failure,
  // scratch exhaustion) every image is decoded singly so each carries its own
  // status; a device fault therefore never masquerades as an unsupported image.
  //
  // All batch outputs decode into ONE stream-ordered device allocation
  // (per-image 256B-aligned slices) and are copied back to host afterwards:
  // nvjpegDecodeBatched writes through the channel pointers from GPU kernels,
  // so host Mats here caused sticky illegal-memory-access faults (GitHub #22).
  [[nodiscard]] std::vector<HostDecode> batch_decode(
      const std::vector<std::pair<const unsigned char *, size_t>> &jpeg_buffers,
      cudaStream_t stream = 0) {
    bind_calling_thread_();

    size_t n = jpeg_buffers.size();
    std::vector<HostDecode> results(n);

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

    DeviceBuffer dbuf(total, stream);
    if (!dbuf.ptr) {
      TOCR_LOG_WARN_RL("nvJPEG batch scratch allocation failed; decoding the images one by one on the GPU",
                       "bytes", total);
      return fallback_single();
    }

    // Initialize batch decode state
    nvjpegStatus_t st = nvjpegDecodeBatchedInitialize(
        handle_, state_, static_cast<int>(n), 1 /*max_cpu_threads*/,
        NVJPEG_OUTPUT_BGRI);
    if (st != NVJPEG_STATUS_SUCCESS) {
      TOCR_LOG_INFO_RL("nvJPEG batched decode not initialised for this batch; decoding the images one by one on the GPU",
                       "status", static_cast<int>(st));
      return fallback_single();
    }

    std::vector<nvjpegImage_t> outputs(n);
    std::vector<const unsigned char *> data_ptrs(n);
    std::vector<size_t> lengths(n);
    for (size_t i = 0; i < n; ++i) {
      outputs[i].channel[0] = static_cast<unsigned char *>(dbuf.ptr) + dims[i].offset;
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
      TOCR_LOG_INFO_RL("nvJPEG batched decode refused this batch; decoding the images one by one on the GPU",
                       "status", static_cast<int>(st));
      return fallback_single();
    }

    // Contract-safe path: copy back through the persistent pinned staging
    // buffer when possible — one full-bandwidth D2H of the whole scratch,
    // then host memcpys into the Mats. Direct mode decoded into the Mats
    // already and only needs the sync below.
    bool copies_ok = true;
    unsigned char *stage = ensure_staging(total);
    if (stage) {
      copies_ok = cudaMemcpyAsync(stage, dbuf.ptr, total,
                                  cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    } else {
      for (size_t i = 0; i < n && copies_ok; ++i) {
        results[i].image = cv::Mat(dims[i].h, dims[i].w, CV_8UC3);
        copies_ok = cudaMemcpyAsync(results[i].image.data, outputs[i].channel[0],
                                    dims[i].bytes, cudaMemcpyDeviceToHost,
                                    stream) == cudaSuccess;
      }
    }

    // Sync so the host copies are materialized and the scratch free (in
    // ~DeviceBuffer, stream-ordered behind the copies) can never race the
    // decode/copy work.
    if (cudaStreamSynchronize(stream) != cudaSuccess || !copies_ok) {
      (void)cudaGetLastError();
      for (auto &r : results) r = {{}, JpegDecodeStatus::Failed, nvjpeg_status::kSuccess};
      return results;
    }
    for (size_t i = 0; i < n; ++i) {
      if (stage) {
        results[i].image = cv::Mat(dims[i].h, dims[i].w, CV_8UC3);
        std::memcpy(results[i].image.data, stage + dims[i].offset, dims[i].bytes);
      }
      results[i].status = JpegDecodeStatus::Ok;
      results[i].nvjpeg_status = nvjpeg_status::kSuccess;
    }
    return results;
  }

  [[nodiscard]] bool available() const noexcept { return handle_ != nullptr; }
  [[nodiscard]] Backend preferred_backend() const noexcept { return preferred_; }

private:
  // nvJPEG allocates its per-decoder buffers lazily inside nvjpegDecode /
  // nvjpegDecodeBatched, through an allocator that needs the CALLING thread
  // bound to a CUDA context. A thread that has never made a CUDA runtime call
  // is not bound, and the allocation fails with NVJPEG_STATUS_ALLOCATOR_FAILURE
  // (no CUDA error is raised). A decoder used only by the thread that built
  // it never hits this, because the constructor's runtime calls bound that
  // thread; a decoder used from any other thread does. Each replica owns its
  // decoder and uses it on its own thread, so this is defence in depth: bind
  // once per thread, per device (a thread_local flag keyed on the device
  // index) before any call that may allocate, and no future caller can
  // reintroduce the silent failure by moving a decoder across threads.
  // nvJPEG allocator callbacks (V2 = stream-ordered). Return 0 on success.
  static int dev_malloc_v2_(void *, void **ptr, size_t size, cudaStream_t stream) noexcept {
    if (cudaMallocAsync(ptr, size, stream) == cudaSuccess) return 0;
    (void)cudaGetLastError();
    return 1;
  }
  static int dev_free_v2_(void *, void *ptr, size_t, cudaStream_t stream) noexcept {
    if (ptr && cudaFreeAsync(ptr, stream) != cudaSuccess) { (void)cudaGetLastError(); return 1; }
    return 0;
  }
  static int pinned_malloc_v2_(void *, void **ptr, size_t size, cudaStream_t) noexcept {
    if (cudaHostAlloc(ptr, size, cudaHostAllocPortable) == cudaSuccess) return 0;
    (void)cudaGetLastError();
    return 1;
  }
  static int pinned_free_v2_(void *, void *ptr, size_t, cudaStream_t) noexcept {
    if (ptr && cudaFreeHost(ptr) != cudaSuccess) { (void)cudaGetLastError(); return 1; }
    return 0;
  }

  void bind_calling_thread_() const noexcept {
    thread_local int bound_device = -1;
    if (bound_device == device_) return;
    if (cudaSetDevice(device_) == cudaSuccess) bound_device = device_;
    else (void)cudaGetLastError();
  }

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
  int device_ = 0;
  int last_nvjpeg_status_ = 0;
  Backend preferred_ = Backend::Hardware;
};

} // namespace turbo_ocr::decode
