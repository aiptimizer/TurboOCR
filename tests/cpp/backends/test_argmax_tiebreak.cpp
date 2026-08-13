// IKernels::argmax TIE-BREAK contract, cross-backend.
//
// The contract (from the shared CPU reference, ctc_decode.cpp): on equal
// scores the LOWER class index wins. This sounds cosmetic and is not: the CTC
// blank is class 0, so a blank-vs-character tie decides whether a glyph exists
// at all, and a reduction that resolves ties by "whichever partial arrived
// last" gives different text on different WAVEFRONT WIDTHS. The AMD README
// documents exactly this trap (risk #1): the CUDA-inherited warp-synchronous
// tail was wrong on wave64 in two ways that cancelled by accident, and the
// rewritten wavefront-agnostic tree must keep ties order-independent. A
// natural corpus will essentially never present an exact float tie — this
// synthetic test is the only thing standing between "passes on MI250" and
// "wrong text on RX 7900".
//
// Structure mirrors test_db_postprocess_parity.cpp: the host oracle always
// runs; each device backend section compiles when its toolkit is in the
// configure (TURBO_TESTS_HAVE_AMD from CMake's amd block; !USE_CPU_ONLY for
// CUDA) and SKIPs without a device.

#include <catch_amalgamated.hpp>

#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/kernels_host/host_kernels.h"
#include "cpu/queue/host_device_queue.h"
#include "turbo_ocr/backend/kernels.h"

namespace {

struct Case {
  int batch = 1, seq = 1, classes = 0;
  std::vector<float> probs;                // [batch*seq*classes]
  std::vector<int> want_idx;               // [batch*seq]
  std::vector<float> want_score;           // [batch*seq]
};

// Rows engineered around the reduction's seams: ties within one wave's lanes,
// ties straddling the 32/64 lane boundary, ties across the num_classes >
// blockDim tail loop, and the all-equal row (must answer 0, the blank).
Case make_tie_case() {
  Case c;
  c.batch = 2;
  c.seq = 3;
  c.classes = 6627; // real rec dict scale: forces the per-thread strided loop
  const auto row = [&](std::initializer_list<int> winners_tied_at_one) {
    std::vector<float> r(static_cast<size_t>(c.classes), 0.25F);
    for (int w : winners_tied_at_one) r[static_cast<size_t>(w)] = 1.0F;
    return r;
  };
  std::vector<std::vector<float>> rows = {
      row({5, 42}),        // tie inside the first wave -> 5
      row({0, 6626}),      // blank vs last class, across the tail loop -> 0
      row({30, 6000}),     // tie across distant strides -> 30
      row({31, 32}),       // straddles the wave32 lane boundary -> 31
      row({63, 64}),       // straddles the wave64 lane boundary -> 63
      row({}),             // all equal -> 0
  };
  c.want_idx = {5, 0, 30, 31, 63, 0};
  c.want_score = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.25F}; // all-equal row: max IS 0.25
  for (auto &r : rows) c.probs.insert(c.probs.end(), r.begin(), r.end());
  return c;
}

} // namespace

TEST_CASE("host argmax breaks ties toward the lower class index",
          "[argmax][tiebreak]") {
  const Case c = make_tie_case();
  std::vector<int> idx(static_cast<size_t>(c.batch) * c.seq, -1);
  std::vector<float> score(static_cast<size_t>(c.batch) * c.seq, 0.F);

  turbo_ocr::cpu::HostKernels host;
  turbo_ocr::cpu::HostDeviceQueue q;
  host.argmax(c.probs.data(), idx.data(), score.data(), c.batch, c.seq,
              c.classes, q);
  q.synchronize();

  for (size_t i = 0; i < idx.size(); ++i) {
    INFO("row " << i);
    CHECK(idx[i] == c.want_idx[i]);
    CHECK(score[i] == c.want_score[i]);
  }
}

#ifdef TURBO_TESTS_HAVE_AMD

#include <hip/hip_runtime_api.h>

#include "amd/kernels_hip/hip_kernels.h"
#include "amd/memory/hip_allocator.h"
#include "amd/queue/hip_queue.h"

TEST_CASE("hip argmax matches the host contract on exact ties",
          "[argmax][tiebreak][amd]") {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    SKIP("no HIP device");

  const Case c = make_tie_case();
  const size_t rows = static_cast<size_t>(c.batch) * c.seq;

  float *d_probs = nullptr;
  int *d_idx = nullptr;
  float *d_score = nullptr;
  REQUIRE(hipMalloc(&d_probs, c.probs.size() * sizeof(float)) == hipSuccess);
  REQUIRE(hipMalloc(&d_idx, rows * sizeof(int)) == hipSuccess);
  REQUIRE(hipMalloc(&d_score, rows * sizeof(float)) == hipSuccess);
  REQUIRE(hipMemcpy(d_probs, c.probs.data(), c.probs.size() * sizeof(float),
                    hipMemcpyHostToDevice) == hipSuccess);

  turbo_ocr::amd::HipKernels hip(
      std::make_shared<turbo_ocr::amd::HipAllocator>());
  turbo_ocr::amd::HipStreamQueue q(/*device_id=*/0);
  hip.argmax(d_probs, d_idx, d_score, c.batch, c.seq, c.classes, q);
  q.synchronize();

  std::vector<int> idx(rows, -1);
  std::vector<float> score(rows, 0.F);
  REQUIRE(hipMemcpy(idx.data(), d_idx, rows * sizeof(int),
                    hipMemcpyDeviceToHost) == hipSuccess);
  REQUIRE(hipMemcpy(score.data(), d_score, rows * sizeof(float),
                    hipMemcpyDeviceToHost) == hipSuccess);
  (void)hipFree(d_probs);
  (void)hipFree(d_idx);
  (void)hipFree(d_score);

  for (size_t i = 0; i < rows; ++i) {
    INFO("row " << i << " (wavefront-order sensitive)");
    CHECK(idx[i] == c.want_idx[i]);
    CHECK(score[i] == c.want_score[i]);
  }
}

#endif // TURBO_TESTS_HAVE_AMD
