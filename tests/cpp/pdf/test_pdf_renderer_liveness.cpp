// Regression test for: PDF renderer daemon failures are detected at
// construction time instead of silently stalling the first render request.
//
// Both cases below are about a daemon that cannot serve. They differ in who is
// at fault, and therefore in what the renderer owes the operator:
//
//   * FASTPDF2PNG_PATH points at nothing — the operator asked for a specific
//     binary that is not there. Honour the request or fail; running a
//     different renderer than the one named would be a silent substitution.
//     => still throws.
//   * the binary is there but dies on exec — nobody asked for anything, the
//     environment is just broken. The in-process renderer can serve this
//     perfectly well, so falling back keeps the server up.
//     => no throw; the fallback must actually rasterize.
#include <catch_amalgamated.hpp>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/core.hpp>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"

namespace fs = std::filesystem;
using turbo_ocr::render::PdfRenderer;

namespace {

// Write a shell script that exits with the given code and return its path.
// Caller must unlink; the test uses /tmp so the kernel cleans up on reboot.
std::string write_stub(const char *basename, int exit_code) {
  std::string path = std::string("/tmp/") + basename + "_" + std::to_string(::getpid());
  {
    std::ofstream o(path);
    o << "#!/bin/sh\nexit " << exit_code << '\n';
  }
  REQUIRE(chmod(path.c_str(), 0755) == 0);
  return path;
}

// Slurp a fixture. Returns empty on a missing file so the caller can REQUIRE
// on it and fail with "fixture missing" rather than a confusing render error.
std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in),
                              std::istreambuf_iterator<char>());
}

struct EnvScope {
  std::string key;
  bool had;
  std::string prev;
  EnvScope(const char *k, const char *v) : key(k) {
    if (const char *p = std::getenv(k)) { had = true; prev = p; } else { had = false; }
    ::setenv(k, v, 1);
  }
  ~EnvScope() {
    if (had) ::setenv(key.c_str(), prev.c_str(), 1);
    else ::unsetenv(key.c_str());
  }
};

} // namespace

TEST_CASE("PdfRenderer ctor throws when the binary path doesn't exist", "[pdf_renderer][liveness]") {
  EnvScope scope{"FASTPDF2PNG_PATH", "/nonexistent/fastpdf2png-does-not-exist"};
  REQUIRE_THROWS_AS(PdfRenderer(1, 1), turbo_ocr::PdfRenderError);
}

TEST_CASE("PdfRenderer falls back to in-process when the daemon binary exits immediately",
          "[pdf_renderer][liveness]") {
  std::string stub = write_stub("turbo_ocr_pdf_liveness_stub", 7);
  EnvScope scope{"FASTPDF2PNG_PATH", stub.c_str()};

  // The invariant under test has always been "a dead daemon must never leave a
  // renderer that blocks on the first request". The liveness probe still
  // detects the dead child — what changed is the response: it used to THROW,
  // which took the whole server down at startup, and now it selects the
  // in-process renderer instead. Asserting only "does not throw" would be a
  // weaker test than the one it replaces, so this renders a real document and
  // requires pages back: that is only possible on a live fallback, never on
  // the dead pipe the original test was written to prevent.
  std::vector<uint8_t> pdf = read_file(std::string(TURBO_TEST_PDF_DIR) +
                                       "/tables_document.pdf");
  REQUIRE_FALSE(pdf.empty());

  PdfRenderer renderer(2, 1);
  std::vector<cv::Mat> pages = renderer.render(pdf.data(), pdf.size(), 72);
  REQUIRE_FALSE(pages.empty());
  REQUIRE_FALSE(pages.front().empty());

  ::unlink(stub.c_str());
}
