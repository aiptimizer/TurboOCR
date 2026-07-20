// GPU-only safety unit tests (C4 per-request timeout primitive, H4 sticky-CUDA
// fault classifier). Guarded: both headers pull in CUDA/TRT, so this file is
// empty in the CPU build and its tests run under the GPU build's ctest.
#include <catch_amalgamated.hpp>

#ifndef USE_CPU_ONLY

#include <future>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"

using turbo_ocr::pipeline::get_with_timeout;

TEST_CASE("C4: get_with_timeout throws TimeoutError on deadline overrun", "[c4][timeout]") {
  // A promise that is never fulfilled models a wedged GPU worker: the wait must
  // give up at the deadline and throw (so the route maps it to 504), not block.
  std::promise<int> p;
  auto fut = p.get_future();
  REQUIRE_THROWS_AS(get_with_timeout(fut, /*timeout_ms=*/30), turbo_ocr::TimeoutError);
}

TEST_CASE("C4: get_with_timeout returns the value when it resolves in time", "[c4][timeout]") {
  std::promise<int> p;
  auto fut = p.get_future();
  std::thread producer([&p] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    p.set_value(7);
  });
  REQUIRE(get_with_timeout(fut, /*timeout_ms=*/2000) == 7);
  producer.join();
}

TEST_CASE("H4: is_sticky_cuda_error flags context-poisoning faults only", "[h4][cuda]") {
  // Sticky faults poison the whole CUDA context -> must fail-fast for restart.
  REQUIRE(turbo_ocr::is_sticky_cuda_error(cudaErrorIllegalAddress));
  REQUIRE(turbo_ocr::is_sticky_cuda_error(cudaErrorLaunchFailure));
  REQUIRE(turbo_ocr::is_sticky_cuda_error(cudaErrorMisalignedAddress));
  // Recoverable / benign codes must NOT trip the fail-fast.
  REQUIRE_FALSE(turbo_ocr::is_sticky_cuda_error(cudaSuccess));
  REQUIRE_FALSE(turbo_ocr::is_sticky_cuda_error(cudaErrorInvalidValue));
}

// Regression (GitHub #22): nvjpegDecodeBatched writes through the output
// channel pointers from GPU kernels, so they must be DEVICE memory. The old
// batch_decode handed it host cv::Mat pointers — 2+ JPEGs per batch raised a
// sticky cudaErrorIllegalAddress that poisoned the context and killed the
// server. No test ever sent >=2 JPEGs through one batch, so it shipped.
TEST_CASE("NvJpeg: batch_decode of 2+ JPEGs matches single decode and leaves "
          "the CUDA context healthy", "[nvjpeg][batch]") {
  if (cudaSetDevice(0) != cudaSuccess) {
    (void)cudaGetLastError();
    SKIP("no CUDA device");
  }

  // Three differently-sized synthetic images (exercises per-image slice
  // offsets in the batched scratch buffer).
  std::vector<std::vector<unsigned char>> jpegs;
  std::vector<cv::Mat> originals;
  for (int i = 0; i < 3; ++i) {
    cv::Mat img(200 + 64 * i, 320 + 96 * i, CV_8UC3);
    cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::GaussianBlur(img, img, cv::Size(9, 9), 0);  // JPEG-friendly content
    std::vector<unsigned char> buf;
    REQUIRE(cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 95}));
    jpegs.push_back(std::move(buf));
    originals.push_back(img);
  }

  std::vector<std::pair<const unsigned char *, size_t>> buffers;
  for (const auto &j : jpegs) buffers.emplace_back(j.data(), j.size());

  auto run_case = [&] {
    turbo_ocr::decode::NvJpegDecoder dec;
    if (!dec.available())
      SKIP("nvJPEG unavailable");
    auto batch = dec.batch_decode(buffers);
    REQUIRE(batch.size() == 3);
    for (size_t i = 0; i < batch.size(); ++i) {
      REQUIRE_FALSE(batch[i].empty());
      CHECK(batch[i].rows == originals[i].rows);
      CHECK(batch[i].cols == originals[i].cols);
      // Batched output must match the per-image nvJPEG decode of the same
      // bytes (same library; <=1 LSB tolerance in case a backend pairs
      // different IDCT kernels for the two entry points).
      cv::Mat single = dec.decode(buffers[i].first, buffers[i].second);
      REQUIRE_FALSE(single.empty());
      CHECK(cv::norm(batch[i], single, cv::NORM_INF) <= 1);
    }
    // The context must be healthy afterwards: the old host-pointer bug left a
    // sticky illegal-memory-access that any later CUDA call would surface.
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);
  };

  SECTION("default mode (direct on HMM machines, device-scratch elsewhere)") {
    ::unsetenv("NVJPEG_DEVICE_COPY");
    run_case();
  }

  SECTION("forced contract-safe device-scratch path (the non-HMM/4090 path)") {
    ::setenv("NVJPEG_DEVICE_COPY", "1", 1);
    run_case();
    ::unsetenv("NVJPEG_DEVICE_COPY");
  }
}

// The single-pass BUF hooking merge claims to produce the exact 8-connected
// partition. That claim must hold for ARBITRARY topologies, not just axis-
// aligned text blobs — an under-merge (component split across blocks) or an
// over-merge changes the component COUNT, so counting against OpenCV's
// reference labeller on pathological shapes (staircases, spirals, snakes,
// checkerboards, dense fuzz) is a complete oracle for both failure modes.
// h_num_total counts EVERY root (even past the kMaxGpuComponents id cap), so
// the over-cap dot grid also exercises the cap path.
TEST_CASE("GPU CCL partition matches OpenCV connectedComponents on "
          "pathological shapes and fuzz", "[ccl][partition]") {
  if (cudaSetDevice(0) != cudaSuccess) {
    (void)cudaGetLastError();
    SKIP("no CUDA device");
  }
  using turbo_ocr::kernels::GpuDetBox;
  using turbo_ocr::kernels::kMaxGpuComponents;

  constexpr int kMaxW = 256, kMaxH = 256;
  const size_t max_px = (size_t)kMaxW * kMaxH;

  uint8_t *d_bitmap = nullptr;
  float *d_pred = nullptr;
  int *d_labels = nullptr, *d_compact = nullptr, *d_counter = nullptr,
      *d_num = nullptr;
  GpuDetBox *d_bboxes = nullptr, *h_boxes = nullptr;
  REQUIRE(cudaMalloc(&d_bitmap, max_px) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_pred, max_px * sizeof(float)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_labels, max_px * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_compact, max_px * sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_counter, sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_num, sizeof(int)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_bboxes, (size_t)kMaxGpuComponents * 2 * sizeof(GpuDetBox)) ==
          cudaSuccess);
  REQUIRE(cudaMallocHost(&h_boxes, (size_t)kMaxGpuComponents * sizeof(GpuDetBox)) ==
          cudaSuccess);
  const std::vector<float> ones(max_px, 1.0f);

  auto gpu_component_count = [&](const cv::Mat &bm) {
    const int w = bm.cols, h = bm.rows;
    cv::Mat cont = bm.isContinuous() ? bm : bm.clone();
    REQUIRE(cudaMemcpy(d_bitmap, cont.data, (size_t)w * h,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_pred, ones.data(), (size_t)w * h * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    int h_num = 0, h_total = 0;
    (void)turbo_ocr::kernels::cuda_gpu_ccl_detect(
        d_bitmap, d_pred, w, h, /*box_thresh=*/0.0f, d_labels, d_compact,
        d_counter, d_bboxes, d_num, h_boxes, &h_num, /*stream=*/0, &h_total);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    return h_total;
  };
  auto cv_component_count = [](const cv::Mat &bm) {
    cv::Mat lbl;
    return cv::connectedComponents(bm, lbl, 8, CV_32S) - 1;  // minus background
  };
  auto check = [&](const char *name, const cv::Mat &bm) {
    INFO(name << " (" << bm.cols << "x" << bm.rows << ")");
    CHECK(gpu_component_count(bm) == cv_component_count(bm));
  };

  SECTION("degenerate and trivial") {
    check("empty", cv::Mat::zeros(64, 64, CV_8U));
    check("full", cv::Mat(64, 64, CV_8U, cv::Scalar(255)));
    cv::Mat corners = cv::Mat::zeros(65, 63, CV_8U);
    corners.at<uint8_t>(0, 0) = corners.at<uint8_t>(0, 62) = 255;
    corners.at<uint8_t>(64, 0) = corners.at<uint8_t>(64, 62) = 255;
    corners.at<uint8_t>(32, 31) = 255;
    check("corner+center dots", corners);
    cv::Mat row(1, 97, CV_8U, cv::Scalar(0));
    for (int x = 0; x < 97; ++x) row.at<uint8_t>(0, x) = (x % 3 != 2) ? 255 : 0;
    check("1xN dashed row", row);
    check("Nx1 dashed col", row.t());
  }

  SECTION("diagonals and snakes (worst case for block merging)") {
    cv::Mat diag = cv::Mat::zeros(129, 127, CV_8U);
    for (int i = 0; i < 127; ++i) diag.at<uint8_t>(i, i) = 255;
    check("main-diagonal staircase", diag);
    cv::Mat anti = cv::Mat::zeros(127, 129, CV_8U);
    for (int i = 0; i < 127; ++i) anti.at<uint8_t>(i, 128 - i) = 255;
    check("anti-diagonal staircase", anti);
    // Both diagonals of an X cross exactly once -> 1 component.
    cv::Mat cross = cv::Mat::zeros(101, 101, CV_8U);
    for (int i = 0; i < 101; ++i) {
      cross.at<uint8_t>(i, i) = 255;
      cross.at<uint8_t>(i, 100 - i) = 255;
    }
    check("X cross", cross);
    // Boustrophedon snake: full rows joined by alternating 1px columns —
    // one component threading through every 2x2 block row of the image.
    cv::Mat snake = cv::Mat::zeros(96, 96, CV_8U);
    for (int y = 0; y < 96; y += 4) {
      snake.row(y).setTo(255);
      if (y + 4 < 96) {
        const int x = (y / 4) % 2 == 0 ? 95 : 0;
        for (int yy = y; yy <= y + 4; ++yy) snake.at<uint8_t>(yy, x) = 255;
      }
    }
    check("boustrophedon snake", snake);
    // Concentric 1px square rings, 3px apart: one component per ring.
    cv::Mat rings = cv::Mat::zeros(128, 128, CV_8U);
    for (int m = 0; m + 1 < 64; m += 4)
      cv::rectangle(rings, {m, m}, {127 - m, 127 - m}, cv::Scalar(255), 1);
    check("concentric rings", rings);
  }

  SECTION("dense adversarial") {
    cv::Mat checker(128, 128, CV_8U);
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 128; ++x)
        checker.at<uint8_t>(y, x) = ((x + y) & 1) ? 255 : 0;
    check("checkerboard (8-conn: one component)", checker);
    cv::Mat stripes_h = cv::Mat::zeros(127, 64, CV_8U);
    for (int y = 0; y < 127; y += 2) stripes_h.row(y).setTo(255);
    check("1px horizontal stripes", stripes_h);
    check("1px vertical stripes", stripes_h.t());
    // Isolated dot grid larger than kMaxGpuComponents: exercises the compact-id
    // cap while h_num_total still counts every root.
    cv::Mat dots = cv::Mat::zeros(256, 256, CV_8U);
    for (int y = 0; y < 256; y += 3)
      for (int x = 0; x < 256; x += 3) dots.at<uint8_t>(y, x) = 255;
    REQUIRE((256 / 3 + 1) * (256 / 3 + 1) > kMaxGpuComponents);
    check("over-cap dot grid", dots);
  }

  SECTION("seeded fuzz across densities and odd dimensions") {
    uint32_t state = 0x9E3779B9u;  // deterministic LCG, no time dependence
    auto next = [&state] {
      state = state * 1664525u + 1013904223u;
      return state >> 16;
    };
    const int dims[][2] = {{33, 47}, {64, 64}, {127, 129}, {254, 256}};
    for (auto [w, h] : dims) {
      for (int density : {25, 50, 75}) {
        cv::Mat bm(h, w, CV_8U);
        for (int y = 0; y < h; ++y)
          for (int x = 0; x < w; ++x)
            bm.at<uint8_t>(y, x) = (int)(next() % 100) < density ? 255 : 0;
        std::string name = "fuzz " + std::to_string(w) + "x" + std::to_string(h) +
                           " d" + std::to_string(density);
        check(name.c_str(), bm);
      }
    }
  }

  cudaFree(d_bitmap); cudaFree(d_pred); cudaFree(d_labels);
  cudaFree(d_compact); cudaFree(d_counter); cudaFree(d_num);
  cudaFree(d_bboxes); cudaFreeHost(h_boxes);
}

#endif  // !USE_CPU_ONLY
