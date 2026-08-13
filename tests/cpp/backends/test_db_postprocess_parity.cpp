// CROSS-BACKEND PARITY for IKernels::db_postprocess.
//
// The NVIDIA README carried this as a bring-up gate: "CudaKernels::db_postprocess
// reproduces the mode-2 axis-aligned JFA path for generic callers and must be
// byte-diffed against PaddleDet on hardware before it replaces any detector call
// site." That framing is not testable as written — PaddleDet's probability map
// and its mode-2 helper are both private, so there is no seam to compare across.
//
// The property that actually matters is stronger and IS testable: db_postprocess
// is a SEAM op with more than one implementation, and every implementation must
// answer the same question the same way. NVIDIA's is unused by its own detector
// today (PaddleDet owns its post-process); AMD's RocmDetector calls the seam op
// directly. So the day a detector switches to the seam, this is the test that
// says whether it may.
//
// The oracle is HostKernels — OpenCV connected components plus the shared unclip
// policy, the same reference the CPU path ships. Synthetic maps, so the input is
// exactly controlled rather than whatever a model happened to emit.

#include <catch_amalgamated.hpp>

#ifndef USE_CPU_ONLY

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "cpu/kernels_host/host_kernels.h"
#include "cpu/queue/host_device_queue.h"
#include "nvidia/kernels_cuda/cuda_kernels.h"
#include "nvidia/queue/cuda_device_queue.h"
#include "turbo_ocr/backend/kernels.h"

namespace {

struct Map {
  int w = 0, h = 0;
  std::vector<float> pred;
  std::vector<std::uint8_t> bitmap;
};

// A map with `blobs` filled rectangles. Probabilities are well clear of the
// threshold so the test measures GEOMETRY, not threshold tie-breaking.
Map make_map(int w, int h, const std::vector<std::array<int, 4>> &blobs) {
  Map m;
  m.w = w;
  m.h = h;
  m.pred.assign(static_cast<size_t>(w) * h, 0.05F);
  m.bitmap.assign(static_cast<size_t>(w) * h, 0);
  for (const auto &b : blobs) {
    for (int y = b[1]; y < b[1] + b[3] && y < h; ++y)
      for (int x = b[0]; x < b[0] + b[2] && x < w; ++x) {
        m.pred[static_cast<size_t>(y) * w + x] = 0.95F;
        m.bitmap[static_cast<size_t>(y) * w + x] = 1;
      }
  }
  return m;
}

// Compare on axis-aligned bounds so the check does not depend on corner ORDER,
// only on the region each implementation decided to emit. aabb() is the shared
// helper in geometry/box.h — no second copy of "min/max over four corners".
std::vector<std::array<int, 4>>
sorted_bounds(const std::vector<turbo_ocr::Box> &boxes) {
  std::vector<std::array<int, 4>> out;
  out.reserve(boxes.size());
  for (const auto &b : boxes) out.push_back(turbo_ocr::aabb(b));
  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
    if (a[1] != b[1]) return a[1] < b[1];
    return a[0] < b[0];
  });
  return out;
}

} // namespace

TEST_CASE("db_postprocess agrees between the CUDA and host backends",
          "[dbpost][parity]") {
  if (cudaSetDevice(0) != cudaSuccess) {
    (void)cudaGetLastError();
    SKIP("no CUDA device");
  }

  const std::vector<std::pair<const char *, Map>> cases = {
      {"one rectangle", make_map(128, 96, {{20, 20, 40, 16}})},
      {"two separated rectangles",
       make_map(160, 120, {{10, 10, 30, 12}, {90, 60, 40, 20}})},
      {"row of words", make_map(200, 64,
                                {{8, 20, 24, 14}, {40, 20, 24, 14},
                                 {72, 20, 24, 14}, {104, 20, 24, 14}})},
      {"tall and thin", make_map(96, 128, {{40, 10, 8, 100}})},
  };

  turbo_ocr::nvidia::CudaKernels cuda;
  turbo_ocr::cpu::HostKernels host;
  turbo_ocr::nvidia::CudaDeviceQueue cuda_q(/*owns=*/true);
  turbo_ocr::cpu::HostDeviceQueue host_q;

  // ONE set of params for both backends — which is the point. These two used to
  // declare DISJOINT capability (host: rotated quads only, CUDA: AABB only), so
  // no value of `oriented` was served by both and a portable caller could not
  // exist. Worse, asking either for the other mode did not fail at the call
  // site: the contract guard returns an EMPTY box list, indistinguishable from
  // "this page has no text". This test found it by asking both for
  // oriented=false and getting 1 box from CUDA and 0 from the host.
  //
  // HostKernels now reduces its quads to bounding rects for this mode, so
  // oriented=false is the portable one and this comparison is a real parity
  // check rather than two different questions.
  turbo_ocr::backend::DbPostParams params;   // shared detection defaults
  params.oriented = false;                   // the mode BOTH backends serve

  for (const auto &[name, m] : cases) {
    INFO("case " << name);
    const size_t px = static_cast<size_t>(m.w) * m.h;

    float *d_pred = nullptr;
    std::uint8_t *d_bitmap = nullptr;
    REQUIRE(cudaMalloc(&d_pred, px * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_bitmap, px) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_pred, m.pred.data(), px * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_bitmap, m.bitmap.data(), px,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const auto gpu_boxes =
        cuda.db_postprocess(d_pred, d_bitmap, m.w, m.h, params, cuda_q);
    const auto cpu_boxes = host.db_postprocess(m.pred.data(), m.bitmap.data(),
                                               m.w, m.h, params, host_q);

    cudaFree(d_pred);
    cudaFree(d_bitmap);

    // Component COUNT is the first-order property: a split or a merge is a
    // different reading of the page, not a rounding difference.
    REQUIRE(gpu_boxes.size() == cpu_boxes.size());

    // Then the geometry. The unclip radius is computed from an integer
    // perimeter on one side and a polygon offset on the other, so exact equality
    // is not the contract — landing on the same region within a pixel or two is.
    const auto g = sorted_bounds(gpu_boxes);
    const auto c = sorted_bounds(cpu_boxes);
    constexpr int kTol = 2;
    for (size_t i = 0; i < g.size(); ++i) {
      INFO("box " << i << " gpu=[" << g[i][0] << "," << g[i][1] << ","
                  << g[i][2] << "," << g[i][3] << "] cpu=[" << c[i][0] << ","
                  << c[i][1] << "," << c[i][2] << "," << c[i][3] << "]");
      for (int k = 0; k < 4; ++k) CHECK(std::abs(g[i][k] - c[i][k]) <= kTol);
    }
  }
}

#endif // USE_CPU_ONLY
