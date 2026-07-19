#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/logger.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <format>
#include <memory>
#include <mutex>
#include <thread>

#include <pthread.h>

#include <opencv2/imgproc.hpp>
#include <fcntl.h>
#include <poll.h>
#include <sys/inotify.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace turbo_ocr::render;

// PPM RGB→BGR swap implementation selector (TURBO_PPM_SWAP=scalar forces the
// old byte loop; validated to {simd, scalar} at startup by ServerConfig).
// Default uses OpenCV's SIMD cvtColor, markedly faster on a single core.
// CPU-only path — no GPU required.
static bool ppm_swap_use_simd() {
  static const bool simd = [] {
    const char *e = std::getenv("TURBO_PPM_SWAP");  // pre-commit-allow-getenv
    return !(e && std::strcmp(e, "scalar") == 0);
  }();
  return simd;
}

// Max rendered pixels per page (width*height). The per-side 16384 cap below
// still bounds a single dimension, but a 16384x16384 page is ~268MP → ~768MB
// raster + a same-size encoded image held in the response: this area cap
// rejects such pages (decode_ppm returns empty → the route reports a decode
// failure). Reads MAX_PDF_PAGE_PIXELS_MP (megapixels); ServerConfig validates
// it to [1,268] at startup. Default 40 MP (e.g. 5000x8000 at ~600 DPI A4).
static int64_t ppm_max_pixels() {
  static const int64_t px = [] {
    // env_int clamps to [1,268] and returns the default (40) on any malformed
    // value; a garbage value would previously revert to 40 with no diagnostic.
    const int def = 40;
    const int mp = turbo_ocr::env::env_int("MAX_PDF_PAGE_PIXELS_MP", def, 1, 268);
    if (const char *e = std::getenv("MAX_PDF_PAGE_PIXELS_MP");  // pre-commit-allow-getenv
        e && *e && mp == def) {
      // Distinguish "explicitly set to 40" from "malformed -> default".
      char *end = nullptr;
      std::strtol(e, &end, 10);
      if (end == e || *end != '\0')
        TOCR_LOG_WARN("MAX_PDF_PAGE_PIXELS_MP malformed; using default",
                      "value", e, "default", def);
    }
    return static_cast<int64_t>(mp) * 1000000;
  }();
  return px;
}

static std::string find_binary() {
  // Explicit override — used by tests and by deployments that put the binary
  // in a non-standard location. Fails fast if the configured path is missing
  // rather than falling back to the default search (surprises hurt in prod).
  if (const char *env = std::getenv("FASTPDF2PNG_PATH"); env && *env) {  // pre-commit-allow-getenv
    if (std::filesystem::exists(env)) return env;
    throw turbo_ocr::PdfRenderError(
        std::format("FASTPDF2PNG_PATH does not exist: {}", env));
  }
  static constexpr const char *paths[] = {
    "/app/bin/fastpdf2png",
    "/usr/local/bin/fastpdf2png",
    "./build/fastpdf2png",
    "./bin/fastpdf2png",
  };
  for (const char *p : paths) {
    if (std::filesystem::exists(p)) return p;
  }
  throw turbo_ocr::PdfRenderError("fastpdf2png binary not found");
}

static bool try_write_file(const char *tmpl, const uint8_t *data, size_t len,
                           std::string &out) {
  char path[64];
  std::strncpy(path, tmpl, sizeof(path) - 1);
  path[sizeof(path) - 1] = '\0';
  int fd = mkstemp(path);
  if (fd < 0) return false;
  size_t written = 0;
  while (written < len) {
    auto n = ::write(fd, data + written, len - written);
    if (n <= 0) { close(fd); unlink(path); return false; }
    written += n;
  }
  close(fd);
  out = path;
  return true;
}

static std::string write_temp_pdf(const uint8_t *data, size_t len) {
  std::string path;
  if (try_write_file("/dev/shm/ocr_pdf_XXXXXX", data, len, path)) return path;
  if (try_write_file("/tmp/ocr_pdf_XXXXXX", data, len, path)) return path;
  throw turbo_ocr::PdfRenderError("Failed to create temp PDF file");
}

static std::string make_temp_dir() {
  // /tmp first: PPM files for large PDFs can exhaust Docker's default 64 MB
  // /dev/shm. The mmap in decode_ppm still benefits from page cache warmth
  // on /tmp, and the GPU inference dominates wall time regardless.
  const char *templates[] = {"/tmp/ocr_out_XXXXXX", "/dev/shm/ocr_out_XXXXXX"};
  for (auto *tmpl : templates) {
    char path[64];
    std::strncpy(path, tmpl, sizeof(path) - 1);
    path[sizeof(path) - 1] = '\0';
    if (mkdtemp(path)) return path;
  }
  throw turbo_ocr::PdfRenderError("Failed to create temp output dir");
}

// RAII guard for temp file/directory cleanup.
struct TempGuard {
  std::string path;
  bool is_dir;
  TempGuard(std::string p, bool dir) : path(std::move(p)), is_dir(dir) {}
  ~TempGuard() noexcept {
    if (path.empty()) return;
    // Best-effort cleanup from a noexcept destructor: a failed unlink/remove
    // only leaks a temp file the OS reclaims, and we must not throw here.
    try {
      if (is_dir) std::filesystem::remove_all(path);
      else unlink(path.c_str());
    } catch (...) { /* noexcept dtor: leaked temp is reclaimed by the OS */ }
  }
  void release() { path.clear(); }
  TempGuard(const TempGuard &) = delete;
  TempGuard &operator=(const TempGuard &) = delete;
};

// Parsed PPM (P5/P6) header. `valid` is false for a malformed or bomb-sized
// header; `payload_offset` is the byte index where pixel data begins and
// `payload_bytes` its exact declared length, so a complete file is exactly
// payload_offset + payload_bytes long. One parser shared by decode_ppm and
// the streamed safety-net's completeness check.
struct PpmHeader {
  bool   valid = false;
  bool   gray = false;
  int    w = 0, h = 0;
  size_t payload_offset = 0;
  size_t payload_bytes = 0;
};

static PpmHeader parse_ppm_header(const unsigned char *base, size_t len) {
  PpmHeader hdr;
  const unsigned char *end = base + len;
  const unsigned char *p = base;
  if (len < 3 || p[0] != 'P' || (p[1] != '5' && p[1] != '6')) return hdr;
  hdr.gray = (p[1] == '5');
  p += 2;

  // Consume one header token (int), skipping whitespace and '#'-comments.
  auto next_int = [&](int &out) -> bool {
    while (p < end) {
      unsigned char c = *p;
      if (c == '#') { while (p < end && *p != '\n') ++p; continue; }
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++p; continue; }
      break;
    }
    if (p >= end || *p < '0' || *p > '9') return false;
    int v = 0;
    while (p < end && *p >= '0' && *p <= '9') {
      v = v * 10 + (*p - '0');
      if (v > 100000) return false;
      ++p;
    }
    out = v;
    return true;
  };

  int maxval = 0;
  if (!next_int(hdr.w) || !next_int(hdr.h) || !next_int(maxval)) return hdr;
  if (hdr.w <= 0 || hdr.h <= 0 || hdr.w > 16384 || hdr.h > 16384 || maxval != 255)
    return hdr;
  if (static_cast<int64_t>(hdr.w) * hdr.h > ppm_max_pixels())
    return hdr;  // area bomb guard
  // Exactly one whitespace byte separates maxval from the payload.
  if (p >= end) return hdr;
  ++p;

  hdr.payload_offset = static_cast<size_t>(p - base);
  hdr.payload_bytes =
      static_cast<size_t>(hdr.w) * hdr.h * (hdr.gray ? 1 : 3);
  hdr.valid = true;
  return hdr;
}

// True when `path` is a fully-written PPM: a parseable header plus the entire
// declared pixel payload present on disk. Used by the streamed safety-net so a
// file still being flushed by a forked worker is retried, not delivered
// truncated. Reads only the header prefix, then stats the size.
static bool ppm_is_complete(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st{};
  if (::fstat(fd, &st) < 0) { ::close(fd); return false; }
  const auto file_size = static_cast<size_t>(st.st_size);
  unsigned char buf[128];
  const ssize_t got = ::read(fd, buf, sizeof(buf));
  ::close(fd);
  if (got <= 0) return false;
  const PpmHeader hdr = parse_ppm_header(buf, static_cast<size_t>(got));
  return hdr.valid && file_size >= hdr.payload_offset + hdr.payload_bytes;
}

// PPM → BGR decoder. mmap the file, copy pixels into a cv::Mat with a
// single-pass RGB→BGR swap, then unlink the file. Unlinking immediately
// after mmap keeps /dev/shm usage bounded by the number of in-flight
// workers rather than the total page count — critical for large PDFs
// where N × ~3 MB/page would exhaust the default 64 MB Docker shm.
cv::Mat PdfRenderer::decode_ppm(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return {};
  struct stat st{};
  if (::fstat(fd, &st) < 0 || st.st_size < 3) {
    ::close(fd);
    return {};
  }
  const size_t file_size = static_cast<size_t>(st.st_size);
  void *map = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map == MAP_FAILED) return {};
  // Unlink now: MAP_PRIVATE made a CoW snapshot so the mapping survives.
  // This frees /dev/shm space immediately instead of after StreamHandle cleanup.
  std::remove(path.c_str());

  struct Unmap {
    void *p;
    size_t n;
    ~Unmap() noexcept { if (p && p != MAP_FAILED) ::munmap(p, n); }
  } guard{map, file_size};

  const unsigned char *base = static_cast<const unsigned char *>(map);
  const PpmHeader hdr = parse_ppm_header(base, file_size);
  if (!hdr.valid) return {};
  const bool gray = hdr.gray;
  const int w = hdr.w, h = hdr.h;
  const unsigned char *p = base + hdr.payload_offset;
  const size_t expected = hdr.payload_bytes;
  if (file_size - hdr.payload_offset < expected) return {};

  if (gray) {
    cv::Mat g(h, w, CV_8UC1);
    std::memcpy(g.data, p, expected);
    cv::Mat bgr;
    cv::cvtColor(g, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }

  // Color: RGB (PPM) → BGR. OpenCV's cvtColor is a SIMD-vectorized channel
  // swap — meaningfully faster than a scalar byte loop on the single core that
  // bottlenecks PDF page-image throughput. The scalar loop is kept as a
  // fallback selectable with TURBO_PPM_SWAP=scalar.
  cv::Mat bgr(h, w, CV_8UC3);
  if (ppm_swap_use_simd()) {
    // Header over the mmap'd RGB payload — no copy; cvtColor reads it and
    // writes the owned `bgr`, both completing before the munmap at return.
    cv::Mat rgb(h, w, CV_8UC3, const_cast<unsigned char *>(p));
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  } else {
    const unsigned char *src = p;
    unsigned char *dst = bgr.data;
    const size_t n_px = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n_px; ++i) {
      dst[0] = src[2];
      dst[1] = src[1];
      dst[2] = src[0];
      src += 3;
      dst += 3;
    }
  }
  return bgr;
}

// NOTE: We deliberately do NOT install a process-wide SIGCHLD reaper.
// In the container the OCR process is PID 1, which means orphaned grand-
// children (the daemon's worker subprocesses) get re-parented to us.
// A waitpid(-1, …, WNOHANG) drain in our SIGCHLD handler would race
// the daemon's own waitpid() and steal the worker's exit status; the
// daemon then thinks the worker is still alive, never delivers the
// PPM, and OCR sees a missing file. The zombie risk the reaper was
// added to address is bounded by the daemon count (16) and never
// materialises under normal operation — the daemon stays alive across
// the process lifetime. ~PdfRenderer reaps the daemons explicitly.

// Fork+exec a fresh daemon into `d`. Shared by the constructor and the
// runtime crash-recovery path (respawn_daemon). The pipes are created
// O_CLOEXEC so every sibling daemon's parent-side fd (and our own unused pipe
// ends) is dropped automatically at execl(); only the two fds dup2()'d onto
// STDIN/STDOUT survive (dup2 clears CLOEXEC on its target). This is what makes
// runtime respawn safe: the forked child never reads another daemon's FILE*
// fields, so it can't race a concurrent sibling respawn mutating them. pipe2()
// sets the flag atomically — no fd-leak window for a concurrent fork.
// L4: dup2() can fail (EBADF/EINTR/EMFILE); an un-rewired child would speak the
// daemon protocol on the wrong fds and wedge the parent's pipe, so check both
// dup2() calls and _exit on failure.
void PdfRenderer::spawn_daemon(Daemon &d) {
  int in_pipe[2], out_pipe[2];
  if (pipe2(in_pipe, O_CLOEXEC) < 0)
    throw turbo_ocr::PdfRenderError("pipe2() failed for PDF renderer daemon");
  if (pipe2(out_pipe, O_CLOEXEC) < 0) {
    close(in_pipe[0]); close(in_pipe[1]);
    throw turbo_ocr::PdfRenderError("pipe2() failed for PDF renderer daemon");
  }

  pid_t pid = fork();
  if (pid < 0) {
    close(in_pipe[0]); close(in_pipe[1]);
    close(out_pipe[0]); close(out_pipe[1]);
    throw turbo_ocr::PdfRenderError("fork() failed for PDF renderer daemon");
  }

  if (pid == 0) {
    // dup2 clears CLOEXEC on STDIN/STDOUT so they survive exec; all other
    // (CLOEXEC) fds — our unused pipe ends and every sibling's pipe fd — are
    // closed automatically by execl(). No manual cross-daemon close loop, so
    // nothing here touches another daemon's FILE* (race-free vs respawn).
    if (dup2(in_pipe[0], STDIN_FILENO) < 0) _exit(127);
    if (dup2(out_pipe[1], STDOUT_FILENO) < 0) _exit(127);
    execl(binary_path_.c_str(), binary_path_.c_str(), "--daemon", nullptr);
    _exit(1);
  }

  close(in_pipe[0]);
  close(out_pipe[1]);
  d.pid = pid;
  d.cmd_in = fdopen(in_pipe[1], "w");
  d.result_out = fdopen(out_pipe[0], "r");
  if (!d.cmd_in || !d.result_out)
    throw turbo_ocr::PdfRenderError("fdopen failed for PDF renderer daemon");
}

// Crash recovery for a wedged/dead daemon. Caller holds d.mutex. Reap the
// dead child (best-effort; the no-SIGCHLD-reaper design means it's still
// ours to wait on), tear down its stale pipe handles, then fork a fresh one.
// Returns false (without throwing) if the respawn fails so send_cmd can
// surface the original protocol error instead of masking it.
bool PdfRenderer::respawn_daemon(Daemon &d) {
  if (d.cmd_in)     { fclose(d.cmd_in);     d.cmd_in = nullptr; }
  if (d.result_out) { fclose(d.result_out); d.result_out = nullptr; }
  if (d.pid > 0) {
    // The dead child may be a zombie (exited) or still dying (e.g. crashing
    // mid-write). Don't block shutdown-style: try non-blocking, then SIGKILL.
    if (waitpid(d.pid, nullptr, WNOHANG) == 0) {
      kill(d.pid, SIGKILL);
      waitpid(d.pid, nullptr, 0);
    }
  }
  d.pid = 0;
  try {
    spawn_daemon(d);
  } catch (const std::exception &) {
    // Leave the slot dead (pid==0, null handles); a later request retries.
    return false;
  }
  return d.pid > 0;
}

PdfRenderer::PdfRenderer(int pool_size, int workers_per_render)
    : pool_size_(pool_size), workers_per_render_(workers_per_render),
      daemons_(pool_size) {
  binary_path_ = find_binary();

  for (int i = 0; i < pool_size_; ++i)
    spawn_daemon(daemons_[i]);

  // Liveness probe: if the binary was missing a shared lib, wasn't executable,
  // or crashed during its own startup, the child calls _exit(1) within ~micro-
  // seconds of fork. Give every child 200ms to either exec successfully or
  // die, then reap any corpses. Without this, the first render request would
  // block on a pipe whose reader is dead.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (int i = 0; i < pool_size_; ++i) {
    int status = 0;
    pid_t reaped = waitpid(daemons_[i].pid, &status, WNOHANG);
    if (reaped != daemons_[i].pid) continue;  // 0 = still running, expected

    // Child already exited — record details, then null out the handles so
    // ~PdfRenderer() doesn't SIGPIPE writing QUIT to a dead pipe.
    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    fclose(daemons_[i].cmd_in);     daemons_[i].cmd_in = nullptr;
    fclose(daemons_[i].result_out); daemons_[i].result_out = nullptr;
    daemons_[i].pid = 0;
    throw turbo_ocr::PdfRenderError(std::format(
        "PDF renderer daemon {}/{} exited immediately after fork "
        "(binary={}, exit={}) — likely missing shared library, "
        "non-executable binary, or crash during startup.",
        i, pool_size_, binary_path_, exit_code));
  }
}

PdfRenderer::~PdfRenderer() noexcept {
  for (auto &d : daemons_) {
    if (d.cmd_in) {
      fprintf(d.cmd_in, "QUIT\n");
      fflush(d.cmd_in);
      fclose(d.cmd_in);
    }
    if (d.result_out) fclose(d.result_out);
    if (d.pid > 0) {
      // Wait briefly, then force-kill to avoid hanging on shutdown
      if (waitpid(d.pid, nullptr, WNOHANG) == 0) {
        kill(d.pid, SIGKILL);
        waitpid(d.pid, nullptr, 0);
      }
    }
  }
}

int PdfRenderer::acquire_daemon() {
  static thread_local int hint = 0;
  for (int i = 0; i < pool_size_; ++i) {
    int idx = (hint + i) % pool_size_;
    if (daemons_[idx].mutex.try_lock()) {
      hint = (idx + 1) % pool_size_;
      return idx;
    }
  }
  int idx = hint % pool_size_;
  // Lock is acquired here and released via std::unique_lock in render()
  daemons_[idx].mutex.lock();
  hint = (idx + 1) % pool_size_;
  return idx;
}

// Block SIGPIPE on the calling thread for the lifetime of the guard, draining
// any pending instance on destruction. Writing to a daemon whose read end has
// died would otherwise raise SIGPIPE and take down the whole server — exactly
// the M9 crash we must instead recover from. Thread-local mask only; no
// process-wide signal disposition change.
namespace {
struct SigpipeBlocker {
  sigset_t old_set;
  bool blocked = false;
  SigpipeBlocker() {
    sigset_t pipe_set;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    if (pthread_sigmask(SIG_BLOCK, &pipe_set, &old_set) == 0) blocked = true;
  }
  ~SigpipeBlocker() {
    if (!blocked) return;
    // Drain a SIGPIPE that fired while blocked so it doesn't get delivered
    // once we unblock. sigtimedwait with a zero timeout is non-blocking.
    sigset_t pipe_set;
    sigemptyset(&pipe_set);
    sigaddset(&pipe_set, SIGPIPE);
    struct timespec zero{0, 0};
    while (sigtimedwait(&pipe_set, nullptr, &zero) >= 0) {}
    pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
  }
};

// Parse the page count out of a daemon "OK <n>" reply. A wedged or killed
// child can hand back arbitrary bytes; that must surface as PdfRenderError,
// not as std::invalid_argument escaping the error taxonomy.
[[nodiscard]] int parse_daemon_page_count(const std::string &resp) {
  if (!resp.starts_with("OK ")) return 0;
  constexpr int kMaxDaemonPages = 100000;
  int n = 0;
  const char *first = resp.data() + 3;
  const char *last = resp.data() + resp.size();
  auto [ptr, ec] = std::from_chars(first, last, n);
  if (ec != std::errc{} || ptr == first || n < 0 || n > kMaxDaemonPages)
    throw turbo_ocr::PdfRenderError(
        std::format("PDF daemon returned malformed page count: {}", resp));
  return n;
}
} // namespace

// Single write+read round-trip to the daemon. Returns false (not throws) on a
// pipe write/read failure so the caller (send_cmd) can decide whether to
// respawn+retry vs. surface the error.
bool PdfRenderer::send_cmd_once(Daemon &d, const std::string &cmd,
                                std::string &out) {
  if (!d.cmd_in || !d.result_out) return false;
  {
    SigpipeBlocker no_sigpipe;
    if (fprintf(d.cmd_in, "%s\n", cmd.c_str()) < 0) return false;
    if (fflush(d.cmd_in) != 0) return false;  // EPIPE if reader is dead
  }
  // Bound the blocking reply read: a daemon that accepted the command but never
  // answers (wedged mid-render) would otherwise hang the worker forever. The
  // initial "OK N" wait is NOT covered by the 30 s missed-page net in
  // render_streamed, so poll the result fd first. On timeout/error return false
  // and let send_cmd's existing respawn+retry path recover. Configurable via
  // PDF_RENDER_REPLY_TIMEOUT_MS (default 120000).
  static const int reply_timeout_ms = [] {
    return turbo_ocr::env::env_int("PDF_RENDER_REPLY_TIMEOUT_MS", 120000, 1, 3600000);
  }();
  struct pollfd pfd = {fileno(d.result_out), POLLIN, 0};
  // poll() returns -1/EINTR on signal delivery — that is NOT a daemon failure, so
  // retry against the remaining budget rather than tripping the respawn path. A
  // genuine timeout (0) or real error (-1, errno != EINTR) still returns false.
  {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(reply_timeout_ms);
    for (;;) {
      const auto now = std::chrono::steady_clock::now();
      int remaining = static_cast<int>(
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
              .count());
      if (remaining < 0) remaining = 0;
      const int pr = ::poll(&pfd, 1, remaining);
      if (pr > 0) break;                        // fd readable
      if (pr == 0) return false;                // timed out
      if (errno == EINTR) continue;             // signal: retry with remaining budget
      return false;                             // real poll error
    }
  }

  char buf[4096];
  if (!fgets(buf, sizeof(buf), d.result_out)) return false;
  auto len = std::strlen(buf);
  if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
  out.assign(buf);
  return true;
}

// M9: on a read/write failure the daemon has crashed. Under the caller-held
// per-daemon mutex, reap+re-fork a fresh daemon and retry the command exactly
// once. A single retry caps tight re-fork loops: a daemon that dies again on
// the retry surfaces the error (and a brief backoff before the re-fork avoids
// hammering exec when the binary itself is the problem). Steady-state callers
// hit zero overhead — the first attempt succeeds and we never touch fork().
std::string PdfRenderer::send_cmd(Daemon &d, const std::string &cmd) {
  std::string out;
  if (send_cmd_once(d, cmd, out)) return out;

  // First attempt failed — assume crash, re-fork and retry once.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));  // re-fork backoff
  if (!respawn_daemon(d))
    throw turbo_ocr::PdfRenderError(
        "PDF renderer daemon crashed and could not be re-forked");
  if (send_cmd_once(d, cmd, out)) return out;

  throw turbo_ocr::PdfRenderError(
      "PDF renderer daemon read failed after re-fork (daemon may be crash-looping)");
}

std::vector<cv::Mat> PdfRenderer::render(const uint8_t *data, size_t len,
                                         int dpi) {
  TempGuard tmpfile(write_temp_pdf(data, len), false);
  TempGuard tmpdir(make_temp_dir(), true);
  std::string pattern = std::format("{}/p_%04d.ppm", tmpdir.path);

  int idx = acquire_daemon();
  // acquire_daemon() already locked the mutex; adopt it into RAII unique_lock
  std::unique_lock<std::mutex> daemon_lock(daemons_[idx].mutex, std::adopt_lock);
  std::string resp = send_cmd(daemons_[idx],
      std::format("RENDER\t{}\t{}\t{}\t{}\t-1",
                  tmpfile.path, pattern, dpi, workers_per_render_));
  daemon_lock.unlock();

  if (!resp.starts_with("OK"))
    throw turbo_ocr::PdfRenderError(std::format("PDF render failed: {}", resp));

  const int num_pages = parse_daemon_page_count(resp);

  // Read PPM files — parallel for multi-page PDFs (each read_ppm is
  // independent: thread-safe fopen/fread, creates its own cv::Mat).
  std::vector<cv::Mat> pages(num_pages);
  if (num_pages <= 2) {
    for (int i = 0; i < num_pages; ++i)
      pages[i] = read_ppm(std::format("{}/p_{:04d}.ppm", tmpdir.path, i + 1));
  } else {
    std::vector<std::thread> readers;
    int n_readers = std::min(num_pages, 4);
    readers.reserve(n_readers);
    std::atomic<int> next{0};
    for (int t = 0; t < n_readers; ++t) {
      readers.emplace_back([&]() {
        while (true) {
          int idx = next.fetch_add(1, std::memory_order_relaxed);
          if (idx >= num_pages) break;
          pages[idx] = read_ppm(
              std::format("{}/p_{:04d}.ppm", tmpdir.path, idx + 1));
        }
      });
    }
    for (auto &th : readers) th.join();
  }

  // TempGuard destructors clean up tmpfile and tmpdir automatically
  return pages;
}

// StreamHandle cleanup: unlink the tmpfile and remove the tmpdir (and
// any remaining PPMs inside it). Called from the destructor when the
// caller finally drops the handle — which MUST be after all OCR workers
// finish decoding, otherwise workers will try to open a file that's been
// unlinked under them.
void PdfRenderer::StreamHandle::cleanup() noexcept {
  // Best-effort cleanup from a noexcept path: a failed unlink/remove only
  // leaks a temp file the OS reclaims later, and we must not throw here.
  try {
    if (!pdf_tmpfile.empty()) ::unlink(pdf_tmpfile.c_str());
    if (!ppm_tmpdir.empty())  std::filesystem::remove_all(ppm_tmpdir);
  } catch (...) { /* noexcept cleanup: leaked temp is reclaimed by the OS */ }
  pdf_tmpfile.clear();
  ppm_tmpdir.clear();
  num_pages = 0;
}

// ---------------------------------------------------------------------------
// render_streamed: overlap rendering with OCR using inotify
// ---------------------------------------------------------------------------
// The daemon's RenderMulti forks worker processes that write PPM files
// independently. inotify CLOSE_WRITE events tell us the moment each PPM
// lands, so we can hand the path to an OCR worker while later pages are
// still rendering.
//
// The decode step (mmap + RGB→BGR swap, ~3-5 ms/page on A4) now runs in
// the CALLER's thread — OCR workers pop ppm_path strings from their
// queue and call decode_ppm() themselves. Parallelizing decode across
// `num_workers` lifts the single-threaded poll-loop ceiling (~90 p/s)
// close to the GPU OCR ceiling.
//
// Timeline comparison (20-page PDF, pool_size=5):
//   Old streamed: [render     ][poll thread: serial read_ppm + dispatch] → ~90 p/s
//   New streamed: [render     ][poll: dispatch path      ]
//                                [worker: decode + OCR  ] × pool → GPU-bound

PdfRenderer::StreamHandle
PdfRenderer::render_streamed(const uint8_t *data, size_t len, int dpi,
                             PageCallback on_page) {
  TempGuard tmpfile(write_temp_pdf(data, len), false);
  TempGuard tmpdir(make_temp_dir(), true);
  std::string pattern = std::format("{}/p_%04d.ppm", tmpdir.path);

  // Set up inotify BEFORE sending RENDER to avoid missing early pages.
  // CLOSE_WRITE fires when a worker finishes writing a PPM file. The
  // RAII guard ensures the fd is closed even if any later step throws
  // (acquire_daemon, std::thread ctor, std::stoi on the daemon reply).
  struct InotifyFdGuard {
    int fd = -1;
    int wd = -1;
    ~InotifyFdGuard() noexcept {
      if (wd >= 0 && fd >= 0) ::inotify_rm_watch(fd, wd);
      if (fd >= 0) ::close(fd);
    }
  };
  InotifyFdGuard inotify;
  inotify.fd = ::inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (inotify.fd < 0)
    throw turbo_ocr::PdfRenderError("inotify_init1 failed");
  inotify.wd = ::inotify_add_watch(inotify.fd, tmpdir.path.c_str(), IN_CLOSE_WRITE);
  if (inotify.wd < 0)
    throw turbo_ocr::PdfRenderError("inotify_add_watch failed");
  const int inotify_fd = inotify.fd;

  // Track which pages have been delivered to avoid duplicates.
  // Uses a bitset-style vector; pages delivered via inotify are marked here
  // so the safety-net scan at the end skips them. We start with a generous
  // pre-allocation (resized as needed when page indices arrive).
  std::vector<bool> delivered(256, false); // pre-alloc for typical PDFs

  // Launch render in a background thread so we can process inotify events
  // concurrently. The daemon mutex is held for the duration of RENDER.
  // Adopt the lock OUTSIDE the lambda so a std::thread-ctor failure
  // (bad_alloc on stack) doesn't strand the daemon mutex.
  int idx = acquire_daemon();
  std::unique_lock<std::mutex> daemon_lock(daemons_[idx].mutex, std::adopt_lock);
  std::atomic<bool> render_done{false};
  std::string render_resp;
  std::exception_ptr render_error;

  std::thread render_thread([&, daemon_lock = std::move(daemon_lock)]() mutable {
    try {
      render_resp = send_cmd(daemons_[idx],
          std::format("RENDER\t{}\t{}\t{}\t{}\t-1",
                      tmpfile.path, pattern, dpi, workers_per_render_));
    } catch (...) {
      render_error = std::current_exception();
    }
    // daemon_lock RAII releases here.
    render_done.store(true, std::memory_order_release);
  });

  // Helper: parse inotify events and invoke callback for each completed PPM
  int pages_delivered = 0;
  alignas(struct inotify_event) char ev_buf[4096];

  auto process_events = [&]() {
    while (true) {
      auto nread = ::read(inotify_fd, ev_buf, sizeof(ev_buf));
      if (nread <= 0) break;
      for (char *ptr = ev_buf; ptr < ev_buf + nread; ) {
        auto *event = reinterpret_cast<struct inotify_event *>(ptr);
        ptr += sizeof(struct inotify_event) + event->len;
        if (event->len == 0 || !(event->mask & IN_CLOSE_WRITE)) continue;

        // Parse page number from "p_NNNN.ppm"
        std::string_view name(event->name);
        if (!name.starts_with("p_") || !name.ends_with(".ppm")) continue;
        auto num_part = name.substr(2, name.size() - 6);
        int page_num = 0;
        for (char c : num_part) {
          if (c < '0' || c > '9') { page_num = -1; break; }
          page_num = page_num * 10 + (c - '0');
        }
        if (page_num <= 0) continue;

        int page_idx = page_num - 1; // 0-based
        if (page_idx >= static_cast<int>(delivered.size()))
          delivered.resize(page_idx + 1, false);
        if (delivered[page_idx]) continue;
        delivered[page_idx] = true;

        std::string ppm_path = std::format("{}/{}", tmpdir.path, static_cast<const char*>(event->name));
        // Hand the path to the caller; decode + OCR happens in their
        // worker thread, off the critical poll loop.
        on_page(page_idx, std::move(ppm_path));
        ++pages_delivered;
      }
    }
  };

  // Poll loop: process inotify events while render is in progress
  struct pollfd pfd = {inotify_fd, POLLIN, 0};
  while (!render_done.load(std::memory_order_acquire)) {
    int ret = poll(&pfd, 1, 2); // 2ms timeout — low latency, low CPU
    if (ret > 0 && (pfd.revents & POLLIN))
      process_events();
  }

  render_thread.join();

  // Drain any remaining inotify events. inotify fd + watch are released
  // by the InotifyFdGuard dtor at function exit.
  process_events();

  if (render_error) std::rethrow_exception(render_error);

  if (!render_resp.starts_with("OK"))
    throw turbo_ocr::PdfRenderError(
        std::format("PDF render failed: {}", render_resp));

  const int num_pages = parse_daemon_page_count(render_resp);

  // Safety net: deliver any pages missed by inotify (race, coalesced events).
  // The daemon may respond "OK N" before its forked workers finish writing
  // the last PPM files, so retry until either every page lands or we hit
  // a generous wall-clock cap. Earlier this loop was 50 × 10 ms = 500 ms
  // total — too short for 20+ page PDFs where the workers can still be
  // mid-flush after the daemon's "OK N" reply. Extending to 30 s is safe
  // because real renders never need this long; we only stop early so a
  // truly missing page doesn't hang the request forever.
  if (pages_delivered < num_pages) {
    if (num_pages > static_cast<int>(delivered.size()))
      delivered.resize(num_pages, false);
    using clock = std::chrono::steady_clock;
    const auto deadline = clock::now() + std::chrono::seconds(30);
    // The primary path waits on IN_CLOSE_WRITE (write complete); this net
    // must not deliver a page that merely EXISTS but is still being flushed
    // by a forked worker (the daemon can reply "OK N" before its workers
    // finish writing). Require a complete PPM — full header plus the declared
    // pixel payload — before delivering, so a mid-flush file is retried
    // instead of handed on truncated (decode_ppm would otherwise drop it).
    while (pages_delivered < num_pages && clock::now() < deadline) {
      bool found_any = false;
      for (int i = 0; i < num_pages; ++i) {
        if (delivered[i]) continue;
        std::string ppm_path = std::format("{}/p_{:04d}.ppm", tmpdir.path, i + 1);
        if (!ppm_is_complete(ppm_path)) continue;
        delivered[i] = true;
        on_page(i, std::move(ppm_path));
        ++pages_delivered;
        found_any = true;
      }
      if (pages_delivered < num_pages && !found_any)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // Transfer tmpfile/tmpdir ownership into the StreamHandle so they
  // outlive this stack frame — OCR workers in the caller are still
  // decoding PPM files from the tmpdir and must not race the cleanup.
  StreamHandle handle;
  handle.pdf_tmpfile = tmpfile.path;
  handle.ppm_tmpdir  = tmpdir.path;
  handle.num_pages   = num_pages;
  tmpfile.release();
  tmpdir.release();
  return handle;
}
