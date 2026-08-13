// backend_probe.cpp — evidence that ONE binary holds SEVERAL backends and that
// --backend picks between them at runtime.
//
// This probe links ONLY the pieces that decide backend selection —
// src/pipeline/backend_registry.o plus every vendor registration TU — because
// that is the minimal set the question needs, and keeping it minimal is what
// makes a failure here unambiguous. (It is NOT a workaround for the unified
// server being unlinkable on macOS: turboocr-server builds here, with the PDF
// subsystem coming up via the `elseif(APPLE)` arm and the vendored
// third_party/pdfium/lib/libpdfium.dylib — see src/service/server/unified/unified_server.cmake.)
// What it exercises:
//
//   * available_backends()  -> must list every linked vendor
//   * make_backend("cpu")   -> CpuBackend
//   * make_backend("apple") -> AppleBackend
//   * make_backend("")      -> highest-priority USABLE one (auto-detect)
//   * make_backend("nope")  -> nullptr
//
// Built as the CMake target `turbo_backend_probe` (see CMakeLists.txt); run it
// with no arguments.

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "turbo_ocr/backend/backend_registry.h"

namespace {

int try_one(const std::string &name, const char *label) {
  std::fflush(stdout);
  try {
    auto b = turbo_ocr::backend::make_backend(name);
    if (!b) {
      std::printf("  %-22s -> (null: not compiled in / no device)\n", label);
      return 0;
    }
    const auto caps = b->caps();
    std::printf("  %-22s -> backend='%s' device='%s' async=%d pool=%d\n", label,
                caps.name.c_str(),
                turbo_ocr::backend::device_kind_name(caps.device),
                static_cast<int>(caps.async), caps.recommended_pool_size);
    return 1;
  } catch (const std::exception &e) {
    std::printf("  %-22s -> threw: %s\n", label, e.what());
    return 0;
  }
}

} // namespace

int main(int argc, char **argv) {
  const auto names = turbo_ocr::backend::available_backends();
  std::printf("available_backends() [%zu]:", names.size());
  for (auto n : names) std::printf(" %.*s", static_cast<int>(n.size()), n.data());
  std::printf("\n");

  // FAILURE ACCOUNTING. This probe exists to catch the WHOLE_ARCHIVE
  // registrar-stripping regression (see the header comment) — which manifests
  // as available_backends() returning EMPTY. It used to end `return 0;`
  // unconditionally, so it passed on exactly that regression: a green check
  // that could not go red guards nothing.
  int failures = 0;
  if (names.empty()) {
    std::fprintf(stderr, "FAIL: no backends registered — the registrar objects "
                         "were stripped (WHOLE_ARCHIVE regression)\n");
    ++failures;
  }

  std::printf("make_backend():\n");
  // What MUST construct anywhere: auto-detect (the registry's contract is that
  // it always lands on a usable backend — cpu is the floor) and "cpu" itself.
  // A registered DEVICE vendor may legitimately return null on hardware-less
  // machines (that is the registry's decline-and-try-next contract), so those
  // are probed and printed but never counted as failures.
  failures += try_one("", "\"\" (auto-detect)") ? 0 : 1;
  for (auto n : names) {
    const int ok = try_one(std::string(n), std::string(n).c_str());
    if (n == "cpu" && !ok) ++failures;
  }
  try_one("metal", "\"metal\" (alias)");
  try_one("host", "\"host\" (alias)");
  // Unknown names must yield null — a backend materializing for "nope" means
  // name matching is broken.
  if (turbo_ocr::backend::make_backend("nope") != nullptr) {
    std::fprintf(stderr, "FAIL: make_backend(\"nope\") returned a backend\n");
    ++failures;
  } else {
    std::printf("  %-22s -> (null, as required)\n", "\"nope\" (unknown)");
  }
  for (int i = 1; i < argc; ++i) try_one(argv[i], argv[i]);
  if (failures)
    std::fprintf(stderr, "backend_probe: %d failure(s)\n", failures);
  return failures != 0;
}
