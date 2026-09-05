// Decode-time bench: how long does the replica spend decoding a JPEG on the
// GPU, by image size, on the hardware and the hybrid backend? Puts a number on
// "decode is not on the critical path": compare the per-image mean with the
// replica's per-request time under load (throughput / replicas).
//
// Usage:  bench_nvjpeg_decode <dir-of-jpegs> [passes=3]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/decode/nvjpeg_decoder.h"

using turbo_ocr::decode::JpegDecodeStatus;
using turbo_ocr::decode::NvJpegDecoder;

namespace {
struct Acc { double ms = 0; int n = 0; int unsupported = 0; };
const char *size_class(long mp10) {  // tenths of a megapixel
  if (mp10 < 30) return "<3 MP";
  if (mp10 < 90) return "3-9 MP";
  if (mp10 < 150) return "9-15 MP";
  return ">=15 MP";
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: bench_nvjpeg_decode <dir> [passes]\n"); return 2; }
  const int passes = argc >= 3 ? std::atoi(argv[2]) : 3;
  std::vector<std::vector<unsigned char>> files;
  for (auto &ent : std::filesystem::directory_iterator(argv[1])) {
    auto ext = ent.path().extension().string();
    if (ext != ".jpg" && ext != ".jpeg" && ext != ".JPG") continue;
    std::ifstream f(ent.path(), std::ios::binary);
    files.emplace_back(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
  std::printf("%zu JPEGs, %d passes\n", files.size(), passes);

  for (auto backend : {NvJpegDecoder::Backend::Hardware, NvJpegDecoder::Backend::Hybrid}) {
    NvJpegDecoder dec(backend);
    if (!dec.available()) { std::printf("backend unavailable\n"); continue; }
    cudaStream_t stream; cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    // One device buffer sized for the largest image, like the pipeline's.
    size_t max_bytes = 0;
    for (auto &f : files) { auto [w, h] = dec.get_dimensions(f.data(), f.size()); max_bytes = std::max(max_bytes, size_t(w) * size_t(h) * 3); }
    void *d_buf = nullptr; cudaMalloc(&d_buf, max_bytes);
    std::map<std::string, Acc> by_class;
    // warm-up pass (first-use allocations), then timed passes
    for (int p = 0; p <= passes; ++p) {
      for (auto &f : files) {
        auto [w, h] = dec.get_dimensions(f.data(), f.size());
        if (w <= 0 || h <= 0) continue;
        const auto t0 = std::chrono::steady_clock::now();
        const auto st = dec.decode_to_gpu(f.data(), f.size(), d_buf, size_t(w) * 3, w, h, stream);
        cudaStreamSynchronize(stream);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (p == 0) continue;
        auto &a = by_class[size_class(long(w) * h / 100000)];
        if (st == JpegDecodeStatus::Ok) { a.ms += ms; ++a.n; } else if (st == JpegDecodeStatus::Unsupported) ++a.unsupported;
      }
    }
    std::printf("== %s backend ==\n", backend == NvJpegDecoder::Backend::Hardware ? "hardware" : "hybrid");
    for (auto &[cls, a] : by_class)
      std::printf("  %-8s n=%5d  mean %6.2f ms/image  unsupported=%d\n", cls.c_str(), a.n, a.n ? a.ms / a.n : 0.0, a.unsupported);
    cudaFree(d_buf); cudaStreamDestroy(stream);
  }
  return 0;
}
