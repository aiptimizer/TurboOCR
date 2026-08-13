// Regression test for the gRPC + JPEG NVDEC race that poisoned the CUDA
// context (see `NvJpegDecoder::decode` and `grpc_jpeg_decode_and_infer`).
// Reproduces the production handler-thread -> dispatcher-worker pattern:
// thread A decodes JPEGs and pushes cv::Mats into a bounded queue;
// thread B drains the queue and does memcpy + cudaMemcpy2DAsync on a
// cudaStreamNonBlocking stream. The fix (decode() syncs its stream before
// returning) means the cv::Mat is materialized before the handoff; without
// it, NVDEC's pending DMA either races the memcpy or writes into freed
// pages after thread A drops the Mat.
//
// Exit 0 on N successful iters, 1 on any CUDA fault.
//
// Usage:  test_nvjpeg_race <jpeg-file> [iters=200]

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "nvidia/support/nvjpeg_decoder.h"

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      std::cerr << "CUDA " #x " -> " << cudaGetErrorString(_e) << " at "       \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <jpeg> [iters=200]\n";
    return 2;
  }
  std::ifstream f(argv[1], std::ios::binary);
  if (!f) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  std::string jpeg((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
  int iters = argc >= 3 ? std::atoi(argv[2]) : 200;

  // Producer-side decoder + buffers — modeling a gRPC handler thread.
  // Bounded queue, single consumer (modeling one dispatcher worker).
  std::mutex m;
  std::condition_variable have_cv, room_cv;
  std::vector<cv::Mat> q;
  const size_t cap = 2;
  std::atomic<bool> done{false};
  std::atomic<int> fault{-1};
  std::string fault_msg;

  cudaStream_t stream = nullptr;
  CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  void *d_buf = nullptr, *h_pinned = nullptr;
  size_t d_pitch = 0, d_cap = 0, h_cap = 0;

  std::thread consumer([&] {
    int n = 0;
    while (true) {
      cv::Mat img;
      {
        std::unique_lock<std::mutex> lk(m);
        have_cv.wait(lk, [&] { return !q.empty() || done.load(); });
        if (q.empty() && done.load()) break;
        img = std::move(q.front());
        q.erase(q.begin());
      }
      room_cv.notify_one();

      size_t need_d = static_cast<size_t>(img.cols) * 3 * img.rows;
      if (need_d > d_cap) {
        if (d_buf) cudaFree(d_buf);
        if (cudaMallocPitch(&d_buf, &d_pitch,
                            static_cast<size_t>(img.cols) * 3,
                            static_cast<size_t>(img.rows)) != cudaSuccess) {
          fault.store(n);
          fault_msg = "cudaMallocPitch failed";
          break;
        }
        d_cap = need_d;
      }
      size_t need_h = static_cast<size_t>(img.rows) * img.step;
      if (need_h > h_cap) {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (cudaHostAlloc(&h_pinned, need_h, cudaHostAllocWriteCombined) !=
            cudaSuccess) {
          fault.store(n);
          fault_msg = "cudaHostAlloc failed";
          break;
        }
        h_cap = need_h;
      }
      std::memcpy(h_pinned, img.data, need_h);
      cudaError_t e = cudaMemcpy2DAsync(d_buf, d_pitch, h_pinned, img.step,
                                        static_cast<size_t>(img.cols) * 3,
                                        static_cast<size_t>(img.rows),
                                        cudaMemcpyHostToDevice, stream);
      if (e != cudaSuccess) {
        fault.store(n);
        fault_msg = std::string("cudaMemcpy2DAsync: ") + cudaGetErrorString(e);
        (void)cudaGetLastError();
        break;
      }
      e = cudaStreamSynchronize(stream);
      if (e != cudaSuccess) {
        fault.store(n);
        fault_msg = std::string("stream sync: ") + cudaGetErrorString(e);
        (void)cudaGetLastError();
        break;
      }
      ++n;
    }
    done.store(true);
    have_cv.notify_all();
  });

  thread_local turbo_ocr::decode::NvJpegDecoder dec;
  if (!dec.available()) {
    std::cerr << "FAIL: nvJPEG not available\n";
    done.store(true);
    have_cv.notify_all();
    consumer.join();
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();
  int produced = 0;
  for (int i = 0; i < iters; ++i) {
    cv::Mat img = dec.decode(
        reinterpret_cast<const unsigned char *>(jpeg.data()), jpeg.size());
    if (img.empty()) {
      std::cerr << "FAIL: decode returned empty at iter " << i << "\n";
      done.store(true);
      have_cv.notify_all();
      consumer.join();
      return 1;
    }
    {
      std::unique_lock<std::mutex> lk(m);
      room_cv.wait(lk, [&] { return q.size() < cap || done.load(); });
      if (done.load()) break;
      q.push_back(std::move(img));
    }
    have_cv.notify_one();
    ++produced;
    if (fault.load() >= 0) break;
  }
  done.store(true);
  have_cv.notify_all();
  consumer.join();
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (d_buf) cudaFree(d_buf);
  if (h_pinned) cudaFreeHost(h_pinned);
  cudaStreamDestroy(stream);

  int fi = fault.load();
  if (fi >= 0) {
    std::cout << "FAIL at iter " << fi << " (produced=" << produced
              << ") in " << ms << " ms — " << fault_msg << "\n";
    return 1;
  }
  std::cout << "OK " << produced << " iters in " << ms << " ms ("
            << (ms / produced) << " ms/iter)\n";
  return 0;
}
