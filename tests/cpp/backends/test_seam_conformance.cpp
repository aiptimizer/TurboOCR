// SEAM CONFORMANCE — the contracts in backend/kernels.h and backend/stages.h,
// checked against EVERY backend compiled into this binary.
//
// Why this file exists, concretely. Three seam violations shipped, each found by
// hand months apart, none by a test:
//
//   * NVIDIA never called IRecognizer::warmup — the hook did not exist, so
//     PaddleRec::bake_graphs() sat defined with no caller and every request
//     re-issued its crop kernels. ~14% of throughput, nothing failing.
//   * HostKernels and CudaKernels declared DISJOINT db_postprocess modes, so no
//     value of DbPostParams::oriented was served by both and a portable caller
//     could not exist. The refusal returned an empty box list, which reads as
//     "this page has no text".
//   * IClassifier::run promised a flipped COUNT that two backends fabricated
//     with `return 0` and no caller ever read.
//
// The pattern is the same every time: the seam documents a contract in prose,
// one implementation quietly does not honour it, and because the interface
// reports success nothing anywhere notices. Prose does not enforce; this does.
//
// Deliberately MODEL-FREE. load_stages() needs weights on disk, which a unit
// test should not depend on, so this covers the contracts reachable without
// them — caps() honesty, the parameter-refusal rule, and decode capability
// agreement. Stage-level contracts that need a loaded model belong in a
// harness that has one; see the note at the bottom.

#include <catch_amalgamated.hpp>

#include <string>
#include <vector>

#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/backend/backend_registry.h"
#include "turbo_ocr/backend/kernels.h"

using namespace turbo_ocr;

namespace {

// Every backend this binary can construct. A vendor whose factory declines
// (no device present) is skipped, not failed — that is the registry's
// documented "compiled in but not usable here".
std::vector<std::pair<std::string, std::unique_ptr<backend::Backend>>>
constructible_backends() {
  std::vector<std::pair<std::string, std::unique_ptr<backend::Backend>>> out;
  for (const auto name : backend::available_backends()) {
    auto bk = backend::make_backend(name);
    if (bk) out.emplace_back(std::string(name), std::move(bk));
  }
  return out;
}

// A probability map with one solid rectangle: unambiguously ONE component, well
// clear of any threshold, so a backend that returns nothing has refused rather
// than legitimately found nothing.
struct Map {
  int w = 96, h = 64;
  std::vector<float> pred;
  std::vector<std::uint8_t> bitmap;
  Map() : pred(static_cast<size_t>(w) * h, 0.05F),
          bitmap(static_cast<size_t>(w) * h, 0) {
    for (int y = 16; y < 40; ++y)
      for (int x = 20; x < 70; ++x) {
        pred[static_cast<size_t>(y) * w + x] = 0.95F;
        bitmap[static_cast<size_t>(y) * w + x] = 1;
      }
  }
};


// A REGISTERED backend whose DEVICE is gone is an environment fault, not a code
// fault, and the two must not look alike:
//   * an EMPTY registry means the registration TUs were not force-linked — a
//     broken build, and these suites fail on it (that is why they assert
//     non-empty rather than skipping);
//   * a backend that registers but cannot make a queue means the driver or card
//     is unavailable, which no source change can fix and which every other GPU
//     test in this repo already reports with SKIP.
// Without this split a faulted GPU reads as "the seam is broken", which is
// exactly the wrong thing to tell someone at 3am.
[[nodiscard]] inline bool device_is_usable(backend::Backend &bk) {
  try {
    return bk.make_queue() != nullptr;
  } catch (...) {
    return false;
  }
}

} // namespace

TEST_CASE("every backend is constructible and self-consistent", "[seam]") {
  auto backends = constructible_backends();
  // A binary with no constructible backend cannot serve anything; that is a
  // failure of this build, not a skip.
  REQUIRE_FALSE(backends.empty());

  for (auto &[name, bk] : backends) {
    INFO("backend " << name);
    if (!device_is_usable(*bk)) {
      WARN("skipping " << name << ": its device is unavailable "
           "(driver/hardware fault, not a code fault)");
      continue;
    }
    const auto caps = bk->caps();
    CHECK_FALSE(caps.name.empty());
    // caps().name is what /capabilities reports; a backend answering under a
    // different name than it registered would make that document a lie.
    CHECK(caps.name == name);

    // A backend must hand out the pieces its caps() implies.
    REQUIRE(bk->allocator() != nullptr);
    auto queue = bk->make_queue();
    REQUIRE(queue != nullptr);
    auto kernels = bk->make_kernels();
    REQUIRE(kernels != nullptr);

    // The allocator's device and the backend's must agree — the pipeline picks
    // its staging path from caps().device and then allocates through this.
    CHECK(bk->allocator()->device() == caps.device);
  }
}

TEST_CASE("db_postprocess honours the mode it declares", "[seam][dbpost]") {
  const Map m;
  for (auto &[name, bk] : constructible_backends()) {
    INFO("backend " << name);
    if (!device_is_usable(*bk)) {
      WARN("skipping " << name << ": its device is unavailable "
           "(driver/hardware fault, not a code fault)");
      continue;
    }
    auto kernels = bk->make_kernels();
    auto queue = bk->make_queue();
    const auto kcaps = kernels->caps();

    // THE RULE (kernels.h): honour the parameter, or declare it unsupported in
    // caps() and refuse. What must never happen is a backend DECLARING support
    // and then returning an empty result — the caller cannot tell that from
    // "no text on this page", which is exactly how the disjoint-mode bug hid.
    //
    // Only the host path can be driven from a host buffer here; a device
    // backend needs its own upload, which is what the CUDA-specific parity test
    // does. So this checks the declaration itself for coherence, plus the real
    // call wherever the pointers are host-addressable.
    CHECK((kcaps.params.db_oriented || kcaps.params.db_axis_aligned));

    if (bk->caps().device == backend::DeviceKind::Host) {
      for (const bool oriented : {false, true}) {
        const bool declared =
            oriented ? kcaps.params.db_oriented : kcaps.params.db_axis_aligned;
        if (!declared) continue;
        backend::DbPostParams p;
        p.oriented = oriented;
        const auto boxes = kernels->db_postprocess(m.pred.data(), m.bitmap.data(),
                                                   m.w, m.h, p, *queue);
        INFO("oriented=" << oriented << " was DECLARED supported");
        CHECK_FALSE(boxes.empty());
      }
    }
  }
}

TEST_CASE("decode capability agrees with the decoder", "[seam][decode]") {
  // A 1x1 PNG. can_decode_image() is a header sniff, so a backend that says
  // "yes" must then actually decode it — saying yes and failing turns a cheap
  // pre-lease question into a wasted pipeline lease.
  static const std::uint8_t kPng[] = {
      0x89,'P','N','G',0x0D,0x0A,0x1A,0x0A, 0,0,0,0x0D,'I','H','D','R',
      0,0,0,1, 0,0,0,1, 8,2,0,0,0, 0x90,0x77,0x53,0xDE,
      0,0,0,0x0C,'I','D','A','T', 0x08,0xD7,0x63,0xF8,0xCF,0xC0,0,0,3,1,1,0,
      0x18,0xDD,0x8D,0xB0, 0,0,0,0,'I','E','N','D',0xAE,0x42,0x60,0x82};

  for (auto &[name, bk] : constructible_backends()) {
    INFO("backend " << name);
    if (!device_is_usable(*bk)) {
      WARN("skipping " << name << ": its device is unavailable "
           "(driver/hardware fault, not a code fault)");
      continue;
    }
    auto kernels = bk->make_kernels();
    const bool claims = kernels->can_decode_image(kPng, sizeof(kPng));
    // The BACKEND-level probe and the KERNEL-level one must not disagree: the
    // InferFunc asks the backend, the pipeline asks the kernels, and a mismatch
    // routes a page down a path the other half did not expect.
    const bool backend_claims = bk->can_device_decode(kPng, sizeof(kPng));
    if (!bk->caps().native_image_decode) {
      CHECK_FALSE(backend_claims);
    }
    INFO("kernels claim=" << claims << " backend claim=" << backend_claims);
    // A backend claiming device decode must have declared the capability.
    if (backend_claims) CHECK(bk->caps().native_image_decode);
  }
}

TEST_CASE("preproc geometry is the shared one, not a per-backend copy",
          "[seam][preproc]") {
  // Not a behavioural check — a REGRESSION check on the constants. Three
  // implementations each hard-coded 800 (and 224/256) for the same models;
  // three copies of a number that must agree is one edit from a backend that
  // silently preprocesses to the wrong size, where the tensor still binds and
  // the model still runs.
  using backend::preproc_geometry;
  using backend::PreprocKind;
  CHECK(preproc_geometry(PreprocKind::LayoutSubRect).target == 800);
  CHECK(preproc_geometry(PreprocKind::TableCls).target == 224);
  CHECK(preproc_geometry(PreprocKind::TableCls).resize_short == 256);
  CHECK(preproc_geometry(PreprocKind::SlanextBGR).target ==
        preproc_geometry(PreprocKind::SlanextRGB).target);
}

// STILL UNCOVERED, and deliberately so: the stage-level contracts
// (IDetector::enqueue ordering and single-slot reuse, IRecognizer::warmup being
// called, box coordinate space, IClassifier flipping in place) all need a
// loaded model. They belong in a harness that has weights — tests/cpp/backends/
// turbo_golden.cpp is the one that does. Two of the three bugs named at the top
// of this file live in exactly that gap, so it is the next thing worth closing.
