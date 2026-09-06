// ELF machine probe: tells an x86-64 binary from an aarch64 one before we exec
// or link it, so a wrong-architecture helper is reported as exactly that
// instead of "not found" (the exec fails with ENOEXEC and looks like a missing
// file from the outside). Header-only and dependency-free; the same 20 bytes
// are what `file` reads.
#pragma once

#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace turbo_ocr::elf {

enum class Machine { Unknown, X86_64, AArch64, Other };

struct Header {
  bool is_elf = false;       // false: the file is something else (a script, a text file)
  Machine machine = Machine::Unknown;
  std::uint16_t e_machine = 0;  // raw ELF e_machine value, for messages about "Other"
};

inline constexpr std::size_t kHeaderBytes = 20;  // e_ident (16) + e_type (2) + e_machine (2)

// Classify the first bytes of a file image. Returns nullopt when fewer than
// kHeaderBytes were supplied (the caller could not read a whole header).
[[nodiscard]] inline std::optional<Header> parse_header(std::span<const unsigned char> bytes) noexcept {
  if (bytes.size() < kHeaderBytes) return std::nullopt;
  if (!(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F')) return Header{};
  const bool little_endian = bytes[5] == 1;  // EI_DATA: 1 = LSB, 2 = MSB
  const std::uint16_t m = little_endian
      ? static_cast<std::uint16_t>(bytes[18] | (bytes[19] << 8))
      : static_cast<std::uint16_t>((bytes[18] << 8) | bytes[19]);
  Header h;
  h.is_elf = true;
  h.e_machine = m;
  h.machine = m == 62 ? Machine::X86_64 : m == 183 ? Machine::AArch64 : Machine::Other;  // EM_X86_64, EM_AARCH64
  return h;
}

// Read and classify a file on disk. nullopt when the file cannot be opened or
// is shorter than a header; callers treat that as "not an ELF we can judge".
[[nodiscard]] inline std::optional<Header> read_header(const std::string &path) noexcept {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) return std::nullopt;
  unsigned char buf[kHeaderBytes];
  const std::size_t n = std::fread(buf, 1, sizeof buf, f);
  std::fclose(f);
  return parse_header(std::span<const unsigned char>(buf, n));
}

// The architecture this program was compiled for.
[[nodiscard]] constexpr Machine host_machine() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  return Machine::X86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
  return Machine::AArch64;
#else
  return Machine::Other;
#endif
}

[[nodiscard]] constexpr std::string_view to_string(Machine m) noexcept {
  switch (m) {
    case Machine::X86_64:  return "x86-64";
    case Machine::AArch64: return "aarch64";
    case Machine::Other:   return "another architecture";
    case Machine::Unknown: break;
  }
  return "unknown";
}

}  // namespace turbo_ocr::elf
