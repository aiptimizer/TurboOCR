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

// A binary built for another CPU fails exec() with ENOEXEC and, before this
// check existed, surfaced as "fastpdf2png binary not found" although the file
// was right there. The probe must name the architecture instead.
TEST_CASE("PdfRenderer ctor names a wrong-architecture binary", "[pdf_renderer][liveness][elf]") {
  // 20-byte ELF header for the architecture this test was NOT built for.
#if defined(__x86_64__)
  const unsigned char e_machine[2] = {0xb7, 0x00};  // EM_AARCH64
  const char *expected = "built for aarch64; this machine is x86-64";
#else
  const unsigned char e_machine[2] = {0x3e, 0x00};  // EM_X86_64
  const char *expected = "built for x86-64; this machine is aarch64";
#endif
  std::string path = "/tmp/turbo_ocr_pdf_wrong_arch_" + std::to_string(::getpid());
  {
    unsigned char hdr[20] = {0x7f, 'E', 'L', 'F', 2, 1, 1};
    hdr[16] = 2; hdr[17] = 0;
    hdr[18] = e_machine[0]; hdr[19] = e_machine[1];
    std::ofstream o(path, std::ios::binary);
    o.write(reinterpret_cast<const char *>(hdr), sizeof hdr);
  }
  REQUIRE(chmod(path.c_str(), 0755) == 0);
  EnvScope scope{"FASTPDF2PNG_PATH", path.c_str()};

  REQUIRE_THROWS_WITH(PdfRenderer(1, 1), Catch::Matchers::ContainsSubstring(expected));

  ::unlink(path.c_str());
}
