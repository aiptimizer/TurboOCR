// Regression test: an NvJpegDecoder must decode on threads other than the
// one that constructed it, including the first (allocating) decode and a
// later larger image that forces nvJPEG to reallocate.
//
// nvJPEG allocates lazily inside nvjpegDecode through an allocator that needs
// the calling thread bound to a CUDA context; an unbound thread gets
// NVJPEG_STATUS_ALLOCATOR_FAILURE and no CUDA error. v3.5.2 shared decoders
// across threads and hit this; the server then silently fell back to CPU
// decode on the base64 routes. Each replica now owns its decoder, and
// NvJpegDecoder::bind_calling_thread_() keeps cross-thread use safe anyway.
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
  auto r = d.decode(jpg.data(), jpg.size());
  const bool ok = r.status == turbo_ocr::decode::JpegDecodeStatus::Ok && !r.image.empty();
  std::cout << (ok ? "ok   " : "FAIL ") << tag << " (" << r.image.cols << "x" << r.image.rows
            << ", status " << turbo_ocr::decode::to_string(r.status) << "/" << r.nvjpeg_status << ")\n";
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
      const bool ok = out.size() == 2 &&
                      out[0].status == turbo_ocr::decode::JpegDecodeStatus::Ok && !out[0].image.empty() &&
                      out[1].status == turbo_ocr::decode::JpegDecodeStatus::Ok && !out[1].image.empty();
      std::cout << (ok ? "ok   " : "FAIL ") << "main-constructed, batch_decode on T1\n";
      if (!ok) ++failures;
    }).join();
  }
  // Optional third argument: a progressive JPEG. The hardware backend must
  // report it Unsupported (not Failed) and the hybrid backend must decode it.
  if (argc >= 4) {
    const auto prog = load(argv[3]);
    NvJpegDecoder hw(NvJpegDecoder::Backend::Hardware);
    NvJpegDecoder hy(NvJpegDecoder::Backend::Hybrid);
    auto a = hw.decode(prog.data(), prog.size());
    auto b = hy.decode(prog.data(), prog.size());
    const bool ok = a.status != turbo_ocr::decode::JpegDecodeStatus::Failed &&
                    b.status == turbo_ocr::decode::JpegDecodeStatus::Ok && !b.image.empty();
    std::cout << (ok ? "ok   " : "FAIL ") << "progressive: hardware=" << turbo_ocr::decode::to_string(a.status)
              << " hybrid=" << turbo_ocr::decode::to_string(b.status) << " (" << b.image.cols << "x" << b.image.rows << ")\n";
    if (!ok) ++failures;
  }
  std::cout << (failures ? "FAILED" : "PASSED") << " (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
