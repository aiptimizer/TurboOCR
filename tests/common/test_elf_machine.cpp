// The ELF machine probe behind the "fastpdf2png is built for x86-64" message.
#include <catch_amalgamated.hpp>

#include <array>
#include <cstdio>
#include <string>
#include <unistd.h>

#include "turbo_ocr/common/elf_machine.h"

using namespace turbo_ocr::elf;

namespace {

// A minimal 20-byte ELF prefix: magic, 64-bit class, byte order, version, then
// e_type and e_machine. Everything else is irrelevant to the probe.
std::array<unsigned char, kHeaderBytes> elf_prefix(std::uint16_t e_machine, bool little_endian = true) {
  std::array<unsigned char, kHeaderBytes> b{};
  b[0] = 0x7f; b[1] = 'E'; b[2] = 'L'; b[3] = 'F';
  b[4] = 2;                          // ELFCLASS64
  b[5] = little_endian ? 1 : 2;      // EI_DATA
  b[6] = 1;                          // EV_CURRENT
  b[16] = 2; b[17] = 0;              // ET_EXEC (little-endian layout; unused by the probe)
  if (little_endian) { b[18] = e_machine & 0xff; b[19] = e_machine >> 8; }
  else               { b[18] = e_machine >> 8;   b[19] = e_machine & 0xff; }
  return b;
}

}  // namespace

TEST_CASE("parse_header tells x86-64 from aarch64", "[elf]") {
  auto x86 = parse_header(elf_prefix(62));
  REQUIRE(x86.has_value());
  CHECK(x86->is_elf);
  CHECK(x86->machine == Machine::X86_64);
  CHECK(x86->e_machine == 62);

  auto arm = parse_header(elf_prefix(183));
  REQUIRE(arm.has_value());
  CHECK(arm->machine == Machine::AArch64);

  auto riscv = parse_header(elf_prefix(243));
  REQUIRE(riscv.has_value());
  CHECK(riscv->machine == Machine::Other);
  CHECK(riscv->e_machine == 243);
}

TEST_CASE("parse_header honours the byte-order byte", "[elf]") {
  auto big = parse_header(elf_prefix(183, /*little_endian=*/false));
  REQUIRE(big.has_value());
  CHECK(big->machine == Machine::AArch64);
}

TEST_CASE("parse_header reports non-ELF files and short reads distinctly", "[elf]") {
  const std::string script = "#!/bin/sh\nexit 0\n   padding   ";
  auto sh = parse_header(std::span<const unsigned char>(
      reinterpret_cast<const unsigned char *>(script.data()), script.size()));
  REQUIRE(sh.has_value());
  CHECK_FALSE(sh->is_elf);
  CHECK(sh->machine == Machine::Unknown);

  std::array<unsigned char, 7> tiny{0x7f, 'E', 'L', 'F', 2, 1, 1};
  CHECK_FALSE(parse_header(tiny).has_value());
}

TEST_CASE("read_header classifies a file on disk and tolerates a missing one", "[elf]") {
  const std::string path = "/tmp/turbo_ocr_elf_probe_" + std::to_string(::getpid());
  {
    auto bytes = elf_prefix(183);
    std::FILE *f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
  }
  auto h = read_header(path);
  ::unlink(path.c_str());
  REQUIRE(h.has_value());
  CHECK(h->machine == Machine::AArch64);

  CHECK_FALSE(read_header("/nonexistent/turbo_ocr_elf_probe").has_value());
}

TEST_CASE("host_machine names the architecture this test was compiled for", "[elf]") {
  constexpr Machine host = host_machine();
#if defined(__x86_64__)
  STATIC_CHECK(host == Machine::X86_64);
  CHECK(to_string(host) == "x86-64");
#elif defined(__aarch64__)
  STATIC_CHECK(host == Machine::AArch64);
  CHECK(to_string(host) == "aarch64");
#else
  CHECK(host == Machine::Other);
#endif
  CHECK(to_string(Machine::Unknown) == "unknown");
}
