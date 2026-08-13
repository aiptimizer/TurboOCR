#pragma once

// harness.h — the ONE shared test-side toolbox for every rebuild test binary.
//
// WHY THIS FILE EXISTS: tests/cpp/backends/ used to hold FOUR near-identical drivers
// (funsd_unified_cpu.cpp, funsd_unified_apple.mm, funsd_unified_apple_conc.mm,
// cls_golden_apple.mm). Each re-spelled the same JSON escaper, the same FUNSD
// image loop, the same arg parser, the same timing block — and, because each was
// pinned to one backend, none of their numbers were comparable to another
// machine's. That is precisely the per-backend duplication IMPLEMENTATION_PLAN.md
// forbids, applied to the tests instead of the library.
//
// THE KEY INSIGHT that makes ONE driver possible for every backend: a test that
// only calls backend::make_backend(<name>) + UnifiedOcrPipeline needs NO
// Objective-C++ and no vendor header. All the Metal/MPSGraph/ObjC lives inside
// libturbo_ocr_backend_apple.a behind the seam; CUDA will live inside
// libturbo_ocr_backend_nvidia.a the same way. So every test here is plain .cpp
// and works unmodified on an NVIDIA box.
//
// Everything in this header is backend-neutral by construction: it names no
// vendor type, and the only place a backend name appears is in the DEFAULT MODEL
// PATH table (default_models()), which is data, not logic.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// gethostname / uname / popen are the entire POSIX surface of this harness, and
// MSVC has none of the three under those names. Windows spells them
// GetComputerNameA, RtlGetVersion + GetNativeSystemInfo, and _popen. Everything
// here is provenance metadata for a benchmark record — the numbers do not depend
// on it — but a run whose header cannot say which machine produced it is exactly
// the thing this harness exists to prevent.
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>     // gethostname
#include <sys/utsname.h> // uname
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <crt_externs.h> // _NSGetEnviron
#endif

namespace turbo_bench_platform {
// The environment block: the same NUL-terminated char** of "K=V" everywhere,
// reached three different ways.
//   Windows — no ::environ exists; MSVC spells it _environ in <stdlib.h>.
//   Apple   — <unistd.h> does NOT declare environ for this translation unit,
//             and Apple's documented accessor is _NSGetEnviron() (plain
//             `environ` is not even exported to a dylib).
//   glibc   — ::environ, declared by <unistd.h>.
// Wrapped in a function rather than a macro so the name cannot leak into
// anything that includes this header.
inline char **environ_block() {
#if defined(_WIN32)
  return _environ;
#elif defined(__APPLE__)
  return *_NSGetEnviron();
#else
  return ::environ;
#endif
}
} // namespace turbo_bench_platform

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/core/model_catalog.h" // kV6DetConfigTiny: tier det base
#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/backend/backend_registry.h"
#include "turbo_ocr/base/geometry/box.h"

// The process environment, at GLOBAL scope (a namespace-scope `extern char
// **environ` would declare a different symbol). Executables get this from the
// C runtime on both Darwin and Linux.
extern "C" char **environ;

namespace turbo_ocr::harness {

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

inline double ms_since(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// ---------------------------------------------------------------------------
// Arguments — ONE parser for all three binaries.
//
// Supports `--flag value`, bare `--flag` booleans, and leading POSITIONALS. The
// positionals exist only for backward compatibility with the retired
// `funsd_unified_cpu <cache_dir> <N> <out.json>` CLI, which existing gates and
// notes reference (see tests/cpp/backends/README.md "backward compatibility").
// ---------------------------------------------------------------------------
class Args {
public:
  Args(int argc, char **argv) {
    prog_ = argv[0];
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.rfind("--", 0) == 0) {
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
          flags_[a.substr(2)] = argv[i + 1];
          ++i;
        } else {
          flags_[a.substr(2)] = "1";
        }
      } else {
        pos_.push_back(a);
      }
    }
  }

  [[nodiscard]] bool has(const char *f) const { return flags_.count(f) != 0; }
  [[nodiscard]] std::string get(const char *f, std::string def = {}) const {
    auto it = flags_.find(f);
    return it == flags_.end() ? std::move(def) : it->second;
  }
  [[nodiscard]] int get_int(const char *f, int def) const {
    auto it = flags_.find(f);
    return it == flags_.end() ? def : std::atoi(it->second.c_str());
  }
  [[nodiscard]] double get_double(const char *f, double def) const {
    auto it = flags_.find(f);
    return it == flags_.end() ? def : std::atof(it->second.c_str());
  }
  [[nodiscard]] bool get_bool(const char *f, bool def) const {
    auto it = flags_.find(f);
    if (it == flags_.end()) return def;
    return !(it->second == "0" || it->second == "false" || it->second == "no");
  }
  [[nodiscard]] const std::vector<std::string> &positionals() const { return pos_; }
  [[nodiscard]] const std::string &prog() const { return prog_; }
  // Every unknown flag is a typo the operator must hear about: a silently
  // ignored `--treads 16` is how a benchmark reports the wrong configuration.
  [[nodiscard]] std::vector<std::string> unknown(std::initializer_list<const char *> known) const {
    std::vector<std::string> bad;
    for (const auto &[k, v] : flags_) {
      bool ok = false;
      for (const char *n : known) if (k == n) { ok = true; break; }
      if (!ok) bad.push_back(k);
    }
    return bad;
  }

private:
  std::string prog_;
  std::map<std::string, std::string> flags_;
  std::vector<std::string> pos_;
};

// ---------------------------------------------------------------------------
// JSON emission (minimal, dependency-free — the tests must link without drogon).
// ---------------------------------------------------------------------------
inline std::string jesc(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          o += b;
        } else {
          o += c;
        }
    }
  }
  return o;
}
inline std::string jstr(std::string_view s) { return "\"" + jesc(s) + "\""; }

// Per-image word lists — the FUNSD prediction format tools/bench/score_funsd.py eats.
// Written IDENTICALLY by every backend so two machines' outputs are diffable.
inline bool write_words_json(const std::string &path,
                             const std::vector<std::vector<std::string>> &words) {
  FILE *f = std::fopen(path.c_str(), "w");
  if (!f) return false;
  std::fputc('[', f);
  for (std::size_t i = 0; i < words.size(); ++i) {
    std::fputc('[', f);
    for (std::size_t k = 0; k < words[i].size(); ++k)
      std::fprintf(f, "\"%s\"%s", jesc(words[i][k]).c_str(),
                   k + 1 < words[i].size() ? "," : "");
    std::fprintf(f, "]%s", i + 1 < words.size() ? "," : "");
  }
  std::fputc(']', f);
  std::fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// SHA-256 (public-domain style compact implementation).
//
// Model hashes are the load-bearing part of cross-machine comparison: "my box
// gets 78% F1" means nothing until we know it ran the same weights. Every
// metrics JSON therefore carries {path, sha256, bytes} per model.
// ---------------------------------------------------------------------------
class Sha256 {
public:
  Sha256() { reset(); }
  void reset() {
    h_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    len_ = 0;
    buf_len_ = 0;
  }
  void update(const unsigned char *p, std::size_t n) {
    len_ += n;
    while (n) {
      std::size_t take = std::min(n, std::size_t(64) - buf_len_);
      std::memcpy(buf_ + buf_len_, p, take);
      buf_len_ += take;
      p += take;
      n -= take;
      if (buf_len_ == 64) { block(buf_); buf_len_ = 0; }
    }
  }
  std::string hex() {
    std::uint64_t bits = len_ * 8;
    unsigned char pad = 0x80;
    update(&pad, 1);
    unsigned char z = 0;
    while (buf_len_ != 56) update(&z, 1);
    unsigned char L[8];
    for (int i = 0; i < 8; ++i) L[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
    len_ -= 8; // do not count the length field itself
    update(L, 8);
    static const char *hx = "0123456789abcdef";
    std::string o;
    o.reserve(64);
    for (std::uint32_t v : h_)
      for (int i = 3; i >= 0; --i) {
        unsigned char b = static_cast<unsigned char>(v >> (8 * i));
        o += hx[b >> 4];
        o += hx[b & 15];
      }
    return o;
  }

private:
  static std::uint32_t ror(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const unsigned char *p) {
    static const std::uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
             (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
    for (int i = 16; i < 64; ++i) {
      std::uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
      std::uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    std::uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];
    for (int i = 0; i < 64; ++i) {
      std::uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
      std::uint32_t ch = (e & f) ^ (~e & g);
      std::uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      std::uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
      std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      std::uint32_t t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    std::uint32_t add[8] = {a, b, c, d, e, f, g, hh};
    for (int i = 0; i < 8; ++i) h_[i] += add[i];
  }
  std::array<std::uint32_t, 8> h_{};
  std::uint64_t len_ = 0;
  unsigned char buf_[64]{};
  std::size_t buf_len_ = 0;
};

struct ArtifactHash {
  std::string path;
  std::string sha256;   // "" when missing
  std::uint64_t bytes = 0;
  bool is_dir = false;
  int file_count = 0;
};

inline std::string sha256_file(const fs::path &p, std::uint64_t *bytes_out = nullptr) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return {};
  Sha256 s;
  std::vector<char> buf(1 << 16);
  std::uint64_t total = 0;
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    std::streamsize got = in.gcount();
    if (got <= 0) break;
    s.update(reinterpret_cast<const unsigned char *>(buf.data()),
             static_cast<std::size_t>(got));
    total += static_cast<std::uint64_t>(got);
  }
  if (bytes_out) *bytes_out = total;
  return s.hex();
}

// Hash a model artefact. Apple/NVIDIA "models" are DIRECTORIES (MPSGraph export
// dirs, TRT plan dirs) while CPU models are single .onnx files — one function
// covers both: a directory hashes as SHA256 over (relative path, file hash) of
// every regular file, sorted, so it is stable across machines.
inline ArtifactHash hash_artifact(const std::string &path) {
  ArtifactHash a;
  a.path = path;
  if (path.empty()) return a;
  std::error_code ec;
  if (!fs::exists(path, ec)) return a;
  if (fs::is_directory(path, ec)) {
    a.is_dir = true;
    std::vector<fs::path> files;
    for (auto &e : fs::recursive_directory_iterator(path, ec))
      if (e.is_regular_file(ec)) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    Sha256 s;
    for (const auto &f : files) {
      std::string rel = fs::relative(f, path, ec).string();
      s.update(reinterpret_cast<const unsigned char *>(rel.data()), rel.size());
      std::uint64_t n = 0;
      std::string h = sha256_file(f, &n);
      s.update(reinterpret_cast<const unsigned char *>(h.data()), h.size());
      a.bytes += n;
      ++a.file_count;
    }
    a.sha256 = s.hex();
  } else {
    a.sha256 = sha256_file(path, &a.bytes);
    a.file_count = 1;
  }
  return a;
}

// ---------------------------------------------------------------------------
// Environment provenance — WHAT MAKES A REMOTE RUN COMPARABLE TO THIS ONE.
//
// Nothing here is decorative. Every field is something that has already changed
// a number in this project: chip/thermal state, thread count, and above all the
// env vars (TURBO_APPLE_REC_BUCKETS silently changes the recognition ladder and
// costs 2x throughput if unset; TURBO_APPLE_PROFILE distorts high-K throughput;
// DISABLE_COREML changes the CPU execution provider).
// ---------------------------------------------------------------------------
inline std::string host_name() {
#if defined(_WIN32)
  char b[MAX_COMPUTERNAME_LENGTH + 1] = {0};
  DWORD n = sizeof b;
  if (!GetComputerNameA(b, &n)) return "unknown";
  return b;
#else
  char b[256] = {0};
  if (gethostname(b, sizeof b - 1) != 0) return "unknown";
  return b;
#endif
}
inline std::string os_string() {
#if defined(_WIN32)
  // Same shape as the POSIX branch: "<sysname> <release> <machine>".
  //
  // NOT GetVersionEx: since 8.1 it reports 6.2 unless the binary carries a
  // compatibility manifest, so a Windows 11 box would record itself as Windows 8
  // in the benchmark header — a silently wrong provenance field is worse than
  // none. RtlGetVersion is the documented way to get the real numbers and is not
  // manifest-sensitive; it lives in ntdll and is resolved dynamically because
  // there is no import library for it in the SDK.
  std::string release = "unknown";
  using RtlGetVersionFn = LONG(WINAPI *)(PRTL_OSVERSIONINFOW);
  if (HMODULE nt = ::GetModuleHandleW(L"ntdll.dll")) {
    auto fn = reinterpret_cast<RtlGetVersionFn>(
        reinterpret_cast<void *>(::GetProcAddress(nt, "RtlGetVersion")));
    RTL_OSVERSIONINFOW vi{};
    vi.dwOSVersionInfoSize = sizeof vi;
    if (fn && fn(&vi) == 0)
      release = std::to_string(vi.dwMajorVersion) + "." +
                std::to_string(vi.dwMinorVersion) + "." +
                std::to_string(vi.dwBuildNumber);
  }
  SYSTEM_INFO si{};
  ::GetNativeSystemInfo(&si);
  const char *machine =
      si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64   ? "x86_64"
      : si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64 ? "arm64"
                                                                  : "unknown";
  return "Windows " + release + " " + machine;
#else
  struct utsname u {};
  if (uname(&u) != 0) return "unknown";
  return std::string(u.sysname) + " " + u.release + " " + u.machine;
#endif
}
inline std::string cpu_brand() {
#if defined(__APPLE__)
  char b[256] = {0};
  std::size_t n = sizeof b;
  if (sysctlbyname("machdep.cpu.brand_string", b, &n, nullptr, 0) == 0) return b;
  return "apple-silicon";
#else
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto c = line.find(':');
      if (c != std::string::npos) return line.substr(c + 2);
    }
  }
  return "unknown";
#endif
}

// Env vars that have been observed to change a result in this project, plus the
// vendor knobs a GPU box will need. Anything matching a prefix is captured too.
inline std::vector<std::pair<std::string, std::string>> relevant_env() {
  static const char *kPrefixes[] = {"TURBO_", "OCR_", "ORT_", "CUDA_", "TRT_",
                                    "HIP_",   "ROCM_", "ZE_",  "OMP_",  "MPS"};
  static const char *kExact[] = {"DISABLE_COREML", "MallocNanoZone",
                                 "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"};
  std::vector<std::pair<std::string, std::string>> out;
  for (char **e = turbo_bench_platform::environ_block(); e && *e; ++e) {
    std::string kv = *e;
    auto eq = kv.find('=');
    if (eq == std::string::npos) continue;
    std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    bool keep = false;
    for (const char *p : kPrefixes) if (k.rfind(p, 0) == 0) { keep = true; break; }
    if (!keep) for (const char *x : kExact) if (k == x) { keep = true; break; }
    if (keep) out.emplace_back(k, v);
  }
  std::sort(out.begin(), out.end());
  return out;
}

inline std::string provenance_json(const backend::BackendCaps &caps,
                                   const std::vector<ArtifactHash> &models,
                                   const std::vector<std::pair<std::string, std::string>> &extra) {
  std::string j = "{";
  j += "\"hostname\":" + jstr(host_name());
  j += ",\"os\":" + jstr(os_string());
  j += ",\"chip\":" + jstr(cpu_brand());
  j += ",\"hw_concurrency\":" + std::to_string(std::thread::hardware_concurrency());
  j += ",\"backend\":" + jstr(caps.name);
  j += ",\"device\":" + jstr(backend::device_kind_name(caps.device));
  j += ",\"async\":" + std::string(caps.async ? "true" : "false");
  j += ",\"supports_batch\":" + std::string(caps.supports_batch ? "true" : "false");
  j += ",\"recommended_pool_size\":" + std::to_string(caps.recommended_pool_size);
  j += ",\"available_backends\":[";
  {
    auto names = backend::available_backends();
    for (std::size_t i = 0; i < names.size(); ++i)
      j += (i ? "," : "") + jstr(std::string(names[i]));
  }
  j += "]";
  j += ",\"models\":{";
  for (std::size_t i = 0; i < models.size(); ++i) {
    const auto &m = models[i];
    j += (i ? "," : "") + jstr(m.path) + ":{\"sha256\":" + jstr(m.sha256) +
         ",\"bytes\":" + std::to_string(m.bytes) +
         ",\"files\":" + std::to_string(m.file_count) +
         ",\"is_dir\":" + (m.is_dir ? "true" : "false") + "}";
  }
  j += "}";
  j += ",\"env\":{";
  {
    auto env = relevant_env();
    for (std::size_t i = 0; i < env.size(); ++i)
      j += (i ? "," : "") + jstr(env[i].first) + ":" + jstr(env[i].second);
  }
  j += "}";
  for (const auto &[k, v] : extra) j += "," + jstr(k) + ":" + v;
  j += "}";
  return j;
}

// ---------------------------------------------------------------------------
// Model path defaults — the ONLY place a backend name appears in test logic.
//
// It is a TABLE, not a branch: adding nvidia means adding rows, never editing a
// driver. A backend with no entry falls back to the CPU-tier .onnx paths, which
// is right for every ONNX/TensorRT-from-ONNX backend (nvidia/amd/intel all
// consume the same models/*.onnx today).
// ---------------------------------------------------------------------------
struct ModelPaths {
  std::string det, rec, keys, cls, layout;
};

inline std::string home_dir() {
  const char *h = std::getenv("HOME");
  return h ? h : "";
}

inline ModelPaths default_models(const std::string &backend_name,
                                 const std::string &tier) {
  ModelPaths m;
  // EVERY backend measures the SHIPPED tier models. The apple branch used to
  // pin ~/.apple_ocr_ml/exports/det_tiny992 (the Phase-0 prototype exports)
  // for every tier — so "--tier small" silently benchmarked tiny-at-992 and
  // reported it as small, a 4.6x flattering lie the numbers of which were
  // almost mistaken for a Python-binding regression. AppleBackend discovers
  // its MPSGraph export dirs (models/det_<tier>/graph.json) from the .onnx
  // path, exactly as the server and the Python binding do; prototype exports
  // remain reachable via the explicit --det/--rec/--cls overrides.
  if (tier == "tiny") {
    m.det = "models/det_tiny.onnx";
    m.rec = "models/rec_tiny.onnx";
    m.keys = "models/keys_tiny.txt";
  } else if (tier == "small") {
    m.det = "models/det_small.onnx";
    m.rec = "models/rec_small.onnx";
    m.keys = "models/keys.txt";
  } else { // medium / default
    m.det = "models/det.onnx";
    m.rec = "models/rec.onnx";
    m.keys = "models/keys.txt";
  }
  m.cls = "models/cls.onnx";
  return m;
}

// Apply --det/--rec/--keys/--cls/--layout overrides on top of the defaults.
inline ModelPaths resolve_models(const Args &a, const std::string &backend_name,
                                 const std::string &tier) {
  ModelPaths m = default_models(backend_name, tier);
  m.det = a.get("det", m.det);
  m.rec = a.get("rec", m.rec);
  m.keys = a.get("keys", m.keys);
  m.cls = a.has("cls") ? a.get("cls") : m.cls; // --cls "" disables the classifier
  m.layout = a.get("layout", m.layout);
  // Install the tier's official detection base, exactly like the server
  // (build_backend_runtime) and the Python binding do — a bench that measures
  // a tier must run that tier's real thresholds (tiny: box_thresh 0.40). An
  // explicit --det override keeps the defaults, mirroring resolve_model's
  // "overridden detector discards the entry's det_cfg" rule.
  if (!a.has("det") && tier == "tiny")
    detection::set_det_config_base(server::kV6DetConfigTiny.resize,
                                   server::kV6DetConfigTiny.db);
  return m;
}

inline backend::BackendConfig to_config(const ModelPaths &m) {
  backend::BackendConfig cfg;
  cfg.det_model = m.det;
  cfg.rec_model = m.rec;
  cfg.rec_dict = m.keys;
  cfg.cls_model = m.cls;
  if (!m.layout.empty()) { cfg.layout_model = m.layout; cfg.want_layout = true; }
  // TURBO_ENGINE_MODE=native|ultra|onnx|fast picks WHICH path to the silicon
  // (backend/engine_mode.h). It must be reachable from the harness or the two
  // paths can never be A/B'd against each other on the same images — which is
  // the only way to know what the native engine is actually buying.
  if (const char *m_env = std::getenv("TURBO_ENGINE_MODE"))
    cfg.mode = backend::parse_engine_mode(m_env);
  if (const char *d = std::getenv("TURBO_EP_DEVICE")) cfg.ep.device = d;
  if (const char *f = std::getenv("TURBO_EP_FP16"))
    cfg.ep.fp16 = (std::strcmp(f, "0") != 0);
  return cfg;
}

inline std::vector<ArtifactHash> hash_models(const ModelPaths &m) {
  std::vector<ArtifactHash> v;
  for (const std::string *p : {&m.det, &m.rec, &m.keys, &m.cls, &m.layout})
    if (!p->empty()) v.push_back(hash_artifact(*p));
  return v;
}

// ---------------------------------------------------------------------------
// Image set. FUNSD naming (funsd_%03d.png) by default; any directory of images
// works via --glob-any, so the same harness benchmarks a customer corpus.
// ---------------------------------------------------------------------------
struct ImageSet {
  std::vector<cv::Mat> imgs;
  std::vector<std::string> names;
  std::string sha256; // hash over the file list -> proves both boxes ran the same pages
};

inline ImageSet load_images(const std::string &dir, int count, bool any = false) {
  ImageSet s;
  std::vector<fs::path> files;
  std::error_code ec;
  if (!any) {
    for (int i = 0; i < count; ++i) {
      char p[512];
      std::snprintf(p, sizeof p, "%s/funsd_%03d.png", dir.c_str(), i);
      if (fs::exists(p, ec)) files.emplace_back(p);
    }
  }
  if (files.empty()) { // fall back to "every image in the dir, sorted"
    for (auto &e : fs::directory_iterator(dir, ec)) {
      if (!e.is_regular_file(ec)) continue;
      auto ext = e.path().extension().string();
      for (auto &c : ext) c = static_cast<char>(std::tolower(c));
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
          ext == ".tif" || ext == ".tiff")
        files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (count > 0 && static_cast<int>(files.size()) > count) files.resize(count);
  }
  Sha256 h;
  for (const auto &f : files) {
    cv::Mat m = cv::imread(f.string(), cv::IMREAD_COLOR);
    if (m.empty()) {
      std::fprintf(stderr, "harness: cannot read %s\n", f.string().c_str());
      continue;
    }
    s.imgs.push_back(std::move(m));
    s.names.push_back(f.filename().string());
    std::uint64_t n = 0;
    std::string fh = sha256_file(f, &n);
    h.update(reinterpret_cast<const unsigned char *>(fh.data()), fh.size());
  }
  s.sha256 = h.hex();
  return s;
}

// ---------------------------------------------------------------------------
// Bag-of-words F1 — a C++ transcription of tools/bench/score_funsd.py (which is itself
// tests/benchmark/scoring/bench_funsd_local.py's Counter-based metric).
//
// It lives here so the harness can obey the rule "NEVER report a throughput
// number without its accuracy" in-process, with no python on the path. The
// python scorer remains the gate of record; this is verified to agree with it
// exactly (see README).
// ---------------------------------------------------------------------------
inline std::vector<std::string> tokenize(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  for (unsigned char c : text) {
    unsigned char lc = static_cast<unsigned char>(std::tolower(c));
    if ((lc >= 'a' && lc <= 'z') || (lc >= '0' && lc <= '9')) {
      cur += static_cast<char>(lc);
    } else if (!cur.empty()) {
      out.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

struct Accuracy {
  double f1 = 0, precision = 0, recall = 0;
  int pages = 0;
  bool scored = false;
};

// GT file: a JSON list of lists of strings (tests/benchmark/funsd_gt_words.json).
// Parsed with a 40-line reader rather than a JSON dependency, because this
// binary must link with nothing but OpenCV + the seam on a bare GPU box.
inline bool read_words_json(const std::string &path,
                            std::vector<std::vector<std::string>> &out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::size_t i = 0;
  auto skip = [&] { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; };
  skip();
  if (i >= s.size() || s[i] != '[') return false;
  ++i;
  for (;;) {
    skip();
    if (i >= s.size()) return false;
    if (s[i] == ']') { ++i; break; }
    if (s[i] == ',') { ++i; continue; }
    if (s[i] != '[') return false;
    ++i;
    std::vector<std::string> page;
    for (;;) {
      skip();
      if (i >= s.size()) return false;
      if (s[i] == ']') { ++i; break; }
      if (s[i] == ',') { ++i; continue; }
      if (s[i] != '"') return false;
      ++i;
      std::string w;
      while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
          ++i;
          switch (s[i]) {
            case 'n': w += '\n'; break;
            case 't': w += '\t'; break;
            case 'r': w += '\r'; break;
            case 'u': {
              // \uXXXX -> UTF-8 (BMP only; the GT has no surrogate pairs)
              unsigned cp = 0;
              for (int k = 0; k < 4 && i + 1 < s.size(); ++k) {
                ++i;
                char c = s[i];
                cp = cp * 16 + static_cast<unsigned>(c <= '9' ? c - '0'
                                : (c | 0x20) - 'a' + 10);
              }
              if (cp < 0x80) w += static_cast<char>(cp);
              else if (cp < 0x800) {
                w += static_cast<char>(0xC0 | (cp >> 6));
                w += static_cast<char>(0x80 | (cp & 0x3F));
              } else {
                w += static_cast<char>(0xE0 | (cp >> 12));
                w += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                w += static_cast<char>(0x80 | (cp & 0x3F));
              }
              break;
            }
            default: w += s[i];
          }
        } else {
          w += s[i];
        }
        ++i;
      }
      ++i; // closing quote
      page.push_back(std::move(w));
    }
    out.push_back(std::move(page));
  }
  return true;
}

inline Accuracy score_words(const std::vector<std::vector<std::string>> &preds,
                            const std::string &gt_path) {
  Accuracy a;
  std::vector<std::vector<std::string>> gt;
  if (!read_words_json(gt_path, gt)) return a;
  const std::size_t n = std::min(preds.size(), gt.size());
  double sf = 0, sp = 0, sr = 0;
  for (std::size_t i = 0; i < n; ++i) {
    std::string gj, pj;
    for (const auto &w : gt[i]) { gj += w; gj += ' '; }
    for (const auto &w : preds[i]) { pj += w; pj += ' '; }
    auto gtok = tokenize(gj), ptok = tokenize(pj);
    if (gtok.empty() && ptok.empty()) { sf += 1; sp += 1; sr += 1; continue; }
    if (gtok.empty() || ptok.empty()) continue;
    std::map<std::string, int> gb, pb;
    for (auto &t : gtok) ++gb[t];
    for (auto &t : ptok) ++pb[t];
    int tp = 0;
    for (auto &[k, v] : gb) {
      auto it = pb.find(k);
      if (it != pb.end()) tp += std::min(v, it->second);
    }
    double r = static_cast<double>(tp) / static_cast<double>(gtok.size());
    double p = static_cast<double>(tp) / static_cast<double>(ptok.size());
    double f1 = (r + p) > 0 ? 2 * r * p / (r + p) : 0.0;
    sf += f1; sp += p; sr += r;
  }
  a.pages = static_cast<int>(n);
  if (n) { a.f1 = sf / n; a.precision = sp / n; a.recall = sr / n; a.scored = true; }
  return a;
}

// Default GT location, relative to the repo root the binary is run from.
inline std::string default_gt_path() {
  const char *env = std::getenv("TURBO_FUNSD_GT");
  if (env && *env) return env;
  return "tests/benchmark/funsd_gt_words.json";
}

// ---------------------------------------------------------------------------
// MEASUREMENT DISCIPLINE (these were all learned the hard way — see
// turboocr-rebuild-progress "MEASUREMENT DISCIPLINE"). They are enforced here,
// once, for every backend, so no future harness can quietly drop one.
// ---------------------------------------------------------------------------
inline constexpr double kMinWindowSeconds = 15.0;   // shorter windows are noise
inline constexpr double kWallClockTolerance = 0.05; // 5% reported-vs-wall drift

struct TimingVerdict {
  double window_s = 0;        // the timed window (steady_clock, one reading)
  double accounted_s = 0;     // sum(per-image latency)/threads — independent
  double rate = 0;            // images / window_s
  double rate_accounted = 0;  // images / accounted_s
  double skew = 0;            // |window - accounted| / window
  bool window_long_enough = false;
  bool wall_clock_agrees = false;
};

// The cross-check that caught the bogus "288 img/s".
//
// TWO INDEPENDENT clocks measure the SAME timed region: (a) one steady_clock
// span around the whole run, (b) the sum of the per-image latencies each worker
// measured, divided by the number of workers. If the reported rate is real those
// agree to within a few percent. They disagree exactly when the timed window
// contains work that is not per-image OCR — model load, graph JIT, thread
// spin-up — which is how a 288 img/s reading appeared from a window that was
// mostly model load. A >5% gap makes the number untrustworthy, so it is a LOUD
// failure, not a footnote.
inline TimingVerdict check_timing(double window_ms,
                                  const std::vector<double> &per_image_ms,
                                  int threads, long images) {
  TimingVerdict v;
  v.window_s = window_ms / 1000.0;
  double sum = 0;
  for (double d : per_image_ms) sum += d;
  v.accounted_s = (sum / std::max(1, threads)) / 1000.0;
  v.rate = v.window_s > 0 ? static_cast<double>(images) / v.window_s : 0.0;
  v.rate_accounted = v.accounted_s > 0 ? static_cast<double>(images) / v.accounted_s : 0.0;
  v.skew = v.window_s > 0 ? std::fabs(v.window_s - v.accounted_s) / v.window_s : 1.0;
  v.window_long_enough = v.window_s >= kMinWindowSeconds;
  v.wall_clock_agrees = v.skew <= kWallClockTolerance;
  return v;
}

inline void print_timing_verdict(const TimingVerdict &v, bool strict) {
  std::printf("\n--- measurement discipline ---\n");
  std::printf("timed window          : %.2f s  (minimum %.0f s)\n", v.window_s, kMinWindowSeconds);
  std::printf("accounted busy time   : %.2f s  (sum per-image latency / threads)\n", v.accounted_s);
  std::printf("rate (window)         : %.1f img/s\n", v.rate);
  std::printf("rate (accounted)      : %.1f img/s\n", v.rate_accounted);
  std::printf("wall-clock skew       : %.2f%%  (tolerance %.0f%%)\n", v.skew * 100.0,
              kWallClockTolerance * 100.0);
  if (!v.window_long_enough)
    std::printf("!! WINDOW TOO SHORT: %.2f s < %.0f s. A short window is dominated by "
                "load/JIT and has produced fabricated numbers here before (a 288 img/s "
                "reading). Raise --count or --repeat.%s\n",
                v.window_s, kMinWindowSeconds, strict ? " [FAIL]" : " [warn]");
  if (!v.wall_clock_agrees)
    std::printf("!! WALL-CLOCK CROSS-CHECK FAILED: the timed window and the summed "
                "per-image latencies disagree by %.1f%% (>%.0f%%). The timed region "
                "contains work that is not per-image OCR, so the reported rate is NOT "
                "trustworthy.%s\n",
                v.skew * 100.0, kWallClockTolerance * 100.0, strict ? " [FAIL]" : " [warn]");
  if (v.window_long_enough && v.wall_clock_agrees)
    std::printf("OK: window >= %.0fs and both clocks agree within %.0f%%.\n",
                kMinWindowSeconds, kWallClockTolerance * 100.0);
}

// ---------------------------------------------------------------------------
// DEVICE SATURATION SAMPLING — the cheapest possible evidence of whether a
// throughput number was device-bound or host-bound.
//
// A throughput figure alone cannot distinguish "the GPU is saturated, this is
// the hardware limit" from "the GPU idles half the time waiting on the host".
// Sampling utilization DURING the timed window answers that for free, needs no
// root, and lands in the metrics JSON so a remote run is interpretable.
//
// Sources, per platform:
//   Apple : ioreg -r -c IOAccelerator  ->  "Device Utilization %"
//   NVIDIA: nvidia-smi --query-gpu=utilization.gpu
//   AMD   : rocm-smi --showuse   (parsed from "GPU use (%)")
//
// TRAP — DO NOT "FIX" THIS BY ADDING ane0's busy COUNTER. ioreg's ane0
// `busy (N ms)` field is IOKit DEVICE-MATCHING state, not compute utilization:
// it does not move under ANE load and reading it as utilization produces a false
// "ANE at 0%". There is no user-space ANE utilization counter, so this harness
// reports NO ANE utilization field at all rather than a wrong one.
//
// What it DOES report for the ANE is contention: ANECompilerService pegs a full
// CPU core while ANE programs are being compiled, and that compilation can run
// DURING a benchmark (not only at load), stealing a core from the host half of
// the pipeline. A long sustained K=16 run measured 55.1 img/s against 103 img/s
// on a short one — that is not simply thermal drift, so the compiler's CPU share
// is sampled and reported next to the rate.
// ---------------------------------------------------------------------------
inline std::string run_cmd_capture(const char *cmd) {
  // MSVC ships the same two calls under the underscore-prefixed names.
#if defined(_WIN32)
  auto open_pipe = [](const char *c) { return ::_popen(c, "r"); };
  auto close_pipe = [](FILE *f) { return ::_pclose(f); };
#else
  auto open_pipe = [](const char *c) { return ::popen(c, "r"); };
  auto close_pipe = [](FILE *f) { return ::pclose(f); };
#endif
  std::string out;
  FILE *p = open_pipe(cmd);
  if (!p) return out;
  char buf[512];
  while (std::fgets(buf, sizeof buf, p)) out += buf;
  close_pipe(p);
  return out;
}

// Percent GPU utilization, or -1 when this platform exposes none.
inline double sample_gpu_utilization() {
#if defined(__APPLE__)
  // Needs NO root. Several accelerators may match (integrated + discrete);
  // take the busiest, which is the one the backend is actually using.
  const std::string s =
      run_cmd_capture("ioreg -r -c IOAccelerator 2>/dev/null | grep 'Device Utilization %'");
  // ioreg prints the WHOLE PerformanceStatistics dict on one line, so the value
  // must be read immediately after this exact key. (Scanning for the next '='
  // on the line picks up "lastRecoveryTime"=1029878500420 instead — a 1e12
  // "utilization" that still passes a >=90% saturation test. Ask how I know.)
  static const std::string kKey = "\"Device Utilization %\"=";
  double best = -1;
  std::size_t pos = 0;
  while ((pos = s.find(kKey, pos)) != std::string::npos) {
    pos += kKey.size();
    double v = std::atof(s.c_str() + pos);
    if (v > best) best = v;
  }
  return best;
#else
  std::string s = run_cmd_capture(
      "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null");
  if (!s.empty()) return std::atof(s.c_str());
  s = run_cmd_capture("rocm-smi --showuse 2>/dev/null | grep -o '[0-9]\\+%' | head -1");
  if (!s.empty()) return std::atof(s.c_str());
  return -1;
#endif
}

// Total %CPU held by ANECompilerService (Apple only; -1 elsewhere, 0 when idle).
inline double sample_ane_compiler_cpu() {
#if defined(__APPLE__)
  const std::string s =
      run_cmd_capture("ps -Ao pcpu,comm 2>/dev/null | grep ANECompilerService | grep -v grep");
  if (s.empty()) return 0.0;
  double total = 0;
  std::size_t i = 0;
  while (i < s.size()) {
    total += std::atof(s.c_str() + i);
    auto nl = s.find('\n', i);
    if (nl == std::string::npos) break;
    i = nl + 1;
  }
  return total;
#else
  return -1;
#endif
}

struct UtilStats {
  bool have = false;
  double min = 0, median = 0, max = 0;
  int samples = 0;
};

inline UtilStats summarize(std::vector<double> v) {
  UtilStats s;
  v.erase(std::remove_if(v.begin(), v.end(), [](double d) { return d < 0; }), v.end());
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.have = true;
  s.samples = static_cast<int>(v.size());
  s.min = v.front();
  s.max = v.back();
  s.median = v[v.size() / 2];
  return s;
}

// Samples utilization on a background thread for the duration of the timed
// window. One popen per second per metric is far below the noise floor of a
// benchmark that runs for >= 15 s, and it is OUTSIDE the measured work.
class UtilSampler {
public:
  void start(int period_ms = 1000) {
    stop_ = false;
    th_ = std::thread([this, period_ms] {
      while (!stop_) {
        gpu_.push_back(sample_gpu_utilization());
        ane_.push_back(sample_ane_compiler_cpu());
        for (int i = 0; i < period_ms / 50 && !stop_; ++i)
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  void stop() {
    stop_ = true;
    if (th_.joinable()) th_.join();
  }
  [[nodiscard]] UtilStats gpu() const { return summarize(gpu_); }
  [[nodiscard]] UtilStats ane_compiler() const { return summarize(ane_); }

private:
  std::thread th_;
  std::atomic<bool> stop_{true};
  std::vector<double> gpu_, ane_;
};

inline void print_utilization(const UtilStats &gpu, const UtilStats &ane) {
  if (gpu.have)
    std::printf("device utilization    : median %.1f%%  (min %.1f, max %.1f, n=%d)  "
                "-> %s\n",
                gpu.median, gpu.min, gpu.max, gpu.samples,
                gpu.median >= 90 ? "DEVICE-BOUND (saturated)"
                                 : "NOT saturated — the host, not the device, is the limit");
  else
    std::printf("device utilization    : unavailable on this platform\n");
  if (ane.have && ane.max > 0)
    std::printf("ANECompilerService CPU: median %.0f%%  max %.0f%%  — ANE program "
                "compilation was running DURING the window and steals a host core; "
                "this run is not comparable to one without it\n",
                ane.median, ane.max);
  else if (ane.have)
    std::printf("ANECompilerService CPU: 0%% (no ANE compilation during the window)\n");
  std::printf("NOTE: no ANE *utilization* is reported. ioreg's ane0 \"busy (N ms)\" is "
              "IOKit device-matching state, not compute utilization — it never moves "
              "under load, and reporting it would be a false 0%%.\n");
}

inline void print_thermal_warning() {
  std::printf(
      "NOTE: absolute throughput on this class of machine DRIFTS ~12%% downward over a\n"
      "      long session (thermal / sustained load). A number from another session — or\n"
      "      another machine — is NOT comparable head-to-head. For any A-vs-B claim use\n"
      "      the INTERLEAVED paired mode: --ab <backendA,backendB>.\n");
}

// ---------------------------------------------------------------------------
// Boxes: comparison utilities for the conformance/golden tests.
// IoU is computed over the axis-aligned bounding box of each quad — quads here
// are near-axis-aligned document lines, and an AABB IoU needs no geometry
// dependency, so the check behaves identically on every platform.
// ---------------------------------------------------------------------------
struct Aabb { double x0, y0, x1, y1; };

inline Aabb aabb_of(const Box &b) {
  Aabb r{static_cast<double>(b.pts[0][0]), static_cast<double>(b.pts[0][1]),
         static_cast<double>(b.pts[0][0]), static_cast<double>(b.pts[0][1])};
  for (const auto &p : b.pts) {
    r.x0 = std::min(r.x0, static_cast<double>(p[0]));
    r.y0 = std::min(r.y0, static_cast<double>(p[1]));
    r.x1 = std::max(r.x1, static_cast<double>(p[0]));
    r.y1 = std::max(r.y1, static_cast<double>(p[1]));
  }
  return r;
}

inline double box_iou(const Box &a, const Box &b) {
  Aabb A = aabb_of(a), B = aabb_of(b);
  const double ix = std::max(0.0, std::min(A.x1, B.x1) - std::max(A.x0, B.x0));
  const double iy = std::max(0.0, std::min(A.y1, B.y1) - std::max(A.y0, B.y0));
  const double inter = ix * iy;
  const double ua = (A.x1 - A.x0) * (A.y1 - A.y0) + (B.x1 - B.x0) * (B.y1 - B.y0) - inter;
  return ua > 0 ? inter / ua : 0.0;
}

// ---------------------------------------------------------------------------
// Page upload — the ONE device-agnostic way a test hands a cv::Mat to a stage.
// Mirrors UnifiedOcrPipeline::upload_image_ (which is private); host backends
// zero-copy-wrap the Mat, device backends stage one H2D copy. Needed only by
// turbo_golden, which drives individual stages rather than the whole pipeline.
// ---------------------------------------------------------------------------
struct Uploaded {
  backend::DeviceBuffer buf;
  std::vector<unsigned char> staging;
  backend::ImageView view;
};

inline Uploaded upload_page(backend::Backend &b,
                            const std::shared_ptr<backend::IDeviceAllocator> &alloc,
                            backend::DeviceQueue &q, const cv::Mat &img) {
  Uploaded u;
  const auto caps = b.caps();
  if (caps.device == backend::DeviceKind::Host || !alloc) {
    u.view = backend::ImageView{img.data, static_cast<std::size_t>(img.step),
                                img.rows, img.cols, backend::DeviceKind::Host};
    return u;
  }
  const std::size_t bytes = static_cast<std::size_t>(img.rows) * img.cols * 3;
  u.staging.resize(bytes);
  if (img.isContinuous()) {
    std::memcpy(u.staging.data(), img.data, bytes);
  } else {
    const std::size_t row = static_cast<std::size_t>(img.cols) * 3;
    for (int r = 0; r < img.rows; ++r)
      std::memcpy(u.staging.data() + r * row, img.ptr(r), row);
  }
  u.buf = alloc->allocate_buffer(bytes);
  alloc->copy_h2d(u.buf.data(), u.staging.data(), bytes, q);
  u.view = backend::ImageView{u.buf.data(), static_cast<std::size_t>(img.cols) * 3,
                              img.rows, img.cols, caps.device};
  return u;
}

// ---------------------------------------------------------------------------
// Backend construction with a clear diagnostic — every test opens this way.
// ---------------------------------------------------------------------------
inline std::unique_ptr<backend::Backend> open_backend(const std::string &name) {
  std::unique_ptr<backend::Backend> b;
  try {
    b = backend::make_backend(name);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "make_backend(\"%s\") threw: %s\n", name.c_str(), e.what());
    return nullptr;
  }
  if (!b) {
    std::fprintf(stderr, "make_backend(\"%s\") returned null. Backends compiled in:",
                 name.c_str());
    for (auto n : backend::available_backends())
      std::fprintf(stderr, " %.*s", static_cast<int>(n.size()), n.data());
    std::fprintf(stderr, "\n  (build with -DTURBO_BACKENDS=\"cpu;nvidia\" etc.)\n");
  }
  return b;
}

} // namespace turbo_ocr::harness
