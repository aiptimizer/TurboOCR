// Regression test for: PDF renderer daemon failures are now detected at
// construction time instead of silently stalling the first render request.
#include <catch_amalgamated.hpp>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/render/pdf_renderer.h"

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

TEST_CASE("PdfRenderer ctor throws when the binary exits immediately (execl succeeds, program fails)",
          "[pdf_renderer][liveness]") {
  std::string stub = write_stub("turbo_ocr_pdf_liveness_stub", 7);
  EnvScope scope{"FASTPDF2PNG_PATH", stub.c_str()};

  // Without the liveness probe this would return cleanly and the first render
  // call would block on a dead pipe. With the fix it raises before we ever
  // accept requests.
  REQUIRE_THROWS_AS(PdfRenderer(2, 1), turbo_ocr::PdfRenderError);

  ::unlink(stub.c_str());
}
