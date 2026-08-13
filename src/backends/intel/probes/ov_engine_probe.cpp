// ov_engine_probe — an OFF-HARDWARE functional test of OpenVINOEngine.
//
// WHAT THIS DOES AND DOES NOT PROVE
//
// The Intel backend's device half (SYCL kernels, USM residency, Level Zero
// queues, RemoteTensor binding) cannot be exercised without an Intel GPU. But
// the ENGINE — which is where the performance-gate machinery lives (per-shape
// CompiledModel/InferRequest caching, shape keying, variant selection, tensor
// binding, staging, the data-dependent-output lease path) — is device-agnostic
// C++ over the OpenVINO Runtime, and OpenVINO ships a CPU plugin that runs
// anywhere. So this probe drives the REAL OpenVINOEngine, with the REAL
// PP-OCRv6 ONNX models, through the OV_DEVICE=CPU path and asserts:
//
//   1. load() parses the model and reports its IO names.
//   2. prebuild() compiles one variant per (batch,width) from the SHARED
//      recognition::rec_shape_matrix ladder, and output_shape() then reports the
//      model's real [batch, seq, classes] for each — i.e. a stage can SIZE its
//      buffers from the model instead of assuming a /8 stride.
//   3. run() on a prebuilt shape does NOT increment shape_misses() — the hot
//      path found its cached artefact and compiled nothing.
//   4. run() on a shape that was never prebuilt still succeeds (dynamic
//      fallback) and DOES increment shape_misses() — so a bad warmup matrix is
//      observable rather than silent.
//   5. the det model runs at a real canvas and writes a plausible probability
//      map into the caller's output buffer.
//
// It does NOT prove: any SYCL kernel, any USM/RemoteTensor interop, any
// accuracy number, or anything at all about Intel GPU performance.
//
// Build/run: see run_probe.sh next to this file.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "intel/engine/openvino_engine.h"
#include "intel/memory/l0_allocator.h"
#include "intel/queue/l0_device_queue.h"
#include "turbo_ocr/analysis/recognition/rec_batching.h"

using namespace turbo_ocr;

namespace {

int failures = 0;

void check(bool cond, const std::string &what) {
  std::printf("  [%s] %s\n", cond ? " ok " : "FAIL", what.c_str());
  if (!cond)
    ++failures;
}

} // namespace

int main(int argc, char **argv) {
  const std::string rec_model = argc > 1 ? argv[1] : "models/rec_tiny.onnx";
  const std::string det_model = argc > 2 ? argv[2] : "models/det_tiny.onnx";

  auto alloc = std::make_shared<intel::L0Allocator>(-1);
  intel::L0DeviceQueue queue(-1);

  std::printf("allocator has_device=%d (0 == host path, expected off Intel hw)\n",
              static_cast<int>(alloc->has_device()));

  // ---------------------------------------------------------------- recognizer
  std::printf("\n== recognition engine (%s) ==\n", rec_model.c_str());
  intel::OpenVINOEngine rec(intel::OpenVINOEngine::DeviceType::CPU, alloc);
  check(rec.load(rec_model), "load()");
  if (!rec.is_loaded())
    return 1;

  std::printf("  inputs :");
  for (const auto &n : rec.input_names())
    std::printf(" %s", n.c_str());
  std::printf("\n  outputs:");
  for (const auto &n : rec.output_names())
    std::printf(" %s", n.c_str());
  std::printf("\n");

  const auto caps = rec.caps();
  check(caps.caller_owns_outputs, "caps().caller_owns_outputs");
  check(!caps.async, "caps().async == false (documented sync contract)");
  check(!caps.thread_safe_concurrent, "caps().thread_safe_concurrent == false");
  check(caps.io_space == backend::DeviceKind::Host,
        "caps().io_space == Host on the CPU plugin");

  // Shapes come from the SHARED ladder, restricted to the small end so the probe
  // stays quick. This is the same call the real IntelRecognizer::load() makes.
  const int H = 48;
  std::vector<std::vector<std::int64_t>> shapes;
  std::vector<std::pair<int, int>> wb;
  for (const auto &[w, b] :
       recognition::rec_shape_matrix(recognition::kRecWidthBuckets, H)) {
    if (w > 320 || b > 8)
      continue;
    shapes.push_back({b, 3, H, w});
    wb.emplace_back(w, b);
  }
  const std::size_t built = rec.prebuild(shapes);
  std::printf("  prebuilt %zu / %zu (width,batch) variants\n", built, shapes.size());
  check(built == shapes.size(), "prebuild() compiled every requested shape");

  const std::string out0 = rec.output_names().front();
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    const auto os = rec.output_shape(shapes[i], out0);
    std::printf("    (w=%4d, b=%3d) -> out %s\n", wb[i].first, wb[i].second,
                [&] {
                  std::string s = "[";
                  for (auto d : os)
                    s += std::to_string(d) + ",";
                  return s + "]";
                }()
                    .c_str());
    check(os.size() == 3, "output_shape() reports a 3-D logits shape");
  }

  // --- 3. a prebuilt shape must not miss.
  const auto miss_before = rec.shape_misses();
  {
    const int b = wb.front().second, w = wb.front().first;
    const auto os = rec.output_shape({b, 3, H, w}, out0);
    std::vector<float> input(static_cast<std::size_t>(b) * 3 * H * w, 0.0f);
    std::vector<float> logits(static_cast<std::size_t>(os[0]) * os[1] * os[2], 0.0f);
    std::vector<backend::DeviceTensor> in(1), out(1);
    in[0] = {rec.input_names().front(), input.data(), backend::DeviceKind::Host,
             backend::DType::F32, 0, {b, 3, H, w}};
    out[0] = {out0, logits.data(), backend::DeviceKind::Host, backend::DType::F32,
              0, os};
    std::vector<backend::OutputLease> leases;
    check(rec.run(in, out, leases, queue), "run() on a PREBUILT shape");
    check(rec.shape_misses() == miss_before,
          "prebuilt shape did not increment shape_misses()");
    bool nonzero = false;
    for (float v : logits)
      if (v != 0.0f) {
        nonzero = true;
        break;
      }
    check(nonzero, "run() wrote into the caller's output buffer");
  }

  // --- 4. a shape that was never prebuilt must still work, and must be counted.
  {
    const int b = 3, w = 336; // deliberately off the ladder
    std::vector<float> input(static_cast<std::size_t>(b) * 3 * H * w, 0.0f);
    std::vector<backend::DeviceTensor> in(1), out(1);
    in[0] = {rec.input_names().front(), input.data(), backend::DeviceKind::Host,
             backend::DType::F32, 0, {b, 3, H, w}};
    // Data-dependent-free, but let the engine own it so we exercise the lease
    // path too (data == nullptr).
    out[0] = {out0, nullptr, backend::DeviceKind::Host, backend::DType::F32, 0, {}};
    std::vector<backend::OutputLease> leases;
    check(rec.run(in, out, leases, queue), "run() on a NON-prebuilt shape");
    check(rec.shape_misses() > miss_before,
          "non-prebuilt shape incremented shape_misses()");
    check(leases.size() == 1 && leases[0].data != nullptr,
          "engine-owned output came back as an OutputLease");
    if (!leases.empty()) {
      std::printf("    lease %s shape [", leases[0].name.c_str());
      for (auto d : leases[0].shape)
        std::printf("%lld,", static_cast<long long>(d));
      std::printf("]\n");
    }
  }

  // ---------------------------------------------------------------- detector
  std::printf("\n== detection engine (%s) ==\n", det_model.c_str());
  intel::OpenVINOEngine det(intel::OpenVINOEngine::DeviceType::CPU, alloc);
  check(det.load(det_model), "load()");
  if (det.is_loaded()) {
    const int rh = 640, rw = 640;
    std::vector<float> img(static_cast<std::size_t>(3) * rh * rw, 0.0f);
    std::vector<float> prob(static_cast<std::size_t>(rh) * rw, -1.0f);
    std::vector<backend::DeviceTensor> in(1), out(1);
    in[0] = {det.input_names().front(), img.data(), backend::DeviceKind::Host,
             backend::DType::F32, 0, {1, 3, rh, rw}};
    out[0] = {det.output_names().front(), prob.data(), backend::DeviceKind::Host,
              backend::DType::F32, 0, {1, 1, rh, rw}};
    std::vector<backend::OutputLease> leases;
    check(det.run(in, out, leases, queue), "run() at a 640x640 canvas");
    bool in_range = true;
    for (float v : prob)
      if (!(v >= 0.0f && v <= 1.0f)) {
        in_range = false;
        break;
      }
    check(in_range, "probability map is in [0,1] (sigmoid output, buffer written)");
  }

  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
