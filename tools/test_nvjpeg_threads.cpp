// Regression test: an NvJpegDecoder must decode on threads other than the
// one that constructed it, including the first (allocating) decode and a
// later larger image that forces nvJPEG to reallocate.
//
// nvJPEG allocates lazily inside nvjpegDecode through an allocator that needs
// the calling thread bound to a CUDA context; an unbound thread gets
// NVJPEG_STATUS_ALLOCATOR_FAILURE and no CUDA error. The per-thread decoders
// of v3.5.0 never crossed threads; the shared NvJpegDecoderPool (v3.5.2) does,
// and NvJpegDecoder::bind_calling_thread_() is what makes that safe. Without
// it the server silently fell back to CPU decode on the base64 routes.
//
// Exit 0 when every case decodes, 1 otherwise.
//
// Usage:  test_nvjpeg_threads <small.jpg> <large.jpg>

#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/decode/nvjpeg_decoder.h"

using turbo_ocr::decode::NvJpegDecoder;

namespace {

std::vector<unsigned char> load(const char *path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

int failures = 0;

void expect_decode(const char *tag, NvJpegDecoder &d,
                   const std::vector<unsigned char> &jpg) {
  cv::Mat m = d.decode(jpg.data(), jpg.size());
  const bool ok = !m.empty();
  std::cout << (ok ? "ok   " : "FAIL ") << tag << " (" << m.cols << "x" << m.rows << ")\n";
  if (!ok) ++failures;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: test_nvjpeg_threads <small.jpg> <large.jpg>\n";
    return 2;
  }
  const auto small = load(argv[1]);
  const auto large = load(argv[2]);

  NvJpegDecoder probe;
  if (!probe.available()) {
    std::cout << "nvJPEG unavailable on this machine; nothing to test\n";
    return 0;
  }

  // Constructed on the main thread, first decode on a fresh thread.
  {
    NvJpegDecoder d;
    std::thread([&] { expect_decode("main-constructed, first decode on T1", d, small); }).join();
  }
  // Constructed on one worker, used on two others, with a size increase.
  {
    std::unique_ptr<NvJpegDecoder> d;
    std::thread([&] { d = std::make_unique<NvJpegDecoder>(); }).join();
    std::thread([&] { expect_decode("T1-constructed, small on T2", *d, small); }).join();
    std::thread([&] { expect_decode("T1-constructed, larger on T3 (reallocates)", *d, large); }).join();
    std::thread([&] { expect_decode("T1-constructed, small again on T4", *d, small); }).join();
  }
  // Batch path across threads.
  {
    NvJpegDecoder d;
    std::thread([&] {
      std::vector<std::pair<const unsigned char *, size_t>> in{
          {small.data(), small.size()}, {large.data(), large.size()}};
      auto out = d.batch_decode(in);
      const bool ok = out.size() == 2 && !out[0].empty() && !out[1].empty();
      std::cout << (ok ? "ok   " : "FAIL ") << "main-constructed, batch_decode on T1\n";
      if (!ok) ++failures;
    }).join();
  }
  std::cout << (failures ? "FAILED" : "PASSED") << " (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
