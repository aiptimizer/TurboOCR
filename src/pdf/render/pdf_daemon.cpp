// fastpdf2png daemon pool: spawn/respawn, liveness, command protocol.
#include "turbo_ocr/pdf/render/pdf_renderer.h"

#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

#include "pdf_renderer_internal.h"
#include "turbo_ocr/common/elf_machine.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/common/errors.h"

using namespace turbo_ocr::render;

namespace {

// Why a candidate that exists cannot be used — spelled out for the operator.
// A binary built for the wrong CPU fails exec() with ENOEXEC, which from the
// outside looks exactly like a missing file; naming the architecture is the
// difference between a five-second fix and an afternoon.
struct RejectedCandidate {
  std::string path;
  std::string reason;
};

// Returns the path when it can be exec'd on this machine; otherwise records
// why not (or nothing at all when the path simply does not exist).
std::optional<std::string> usable_binary(const std::string &path,
                                         std::vector<RejectedCandidate> &rejected) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (!fs::exists(path, ec)) return std::nullopt;
  if (fs::is_directory(path, ec)) {
    rejected.push_back({path, "is a directory"});
    return std::nullopt;
  }
  if (auto h = turbo_ocr::elf::read_header(path);
      h && h->is_elf && h->machine != turbo_ocr::elf::host_machine()) {
    rejected.push_back({path, std::format("is built for {}; this machine is {}",
                                          turbo_ocr::elf::to_string(h->machine),
                                          turbo_ocr::elf::to_string(turbo_ocr::elf::host_machine()))});
    return std::nullopt;
  }
  if (::access(path.c_str(), X_OK) != 0) {
    rejected.push_back({path, "is not executable"});
    return std::nullopt;
  }
  return path;
}

// Directory of the running server executable, so a build tree that is not
// called `build/` (or a relocated install) still finds the renderer that was
// built next to it. Empty when /proc/self/exe cannot be resolved.
std::string own_executable_dir() {
  std::error_code ec;
  auto exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec) return {};
  return exe.parent_path().string();
}

std::string find_binary() {
  std::vector<RejectedCandidate> rejected;

  // Explicit override — used by tests and by deployments that put the binary
  // in a non-standard location. Fails fast if the configured path is unusable
  // rather than falling back to the default search (surprises hurt in prod).
  if (const char *env = std::getenv("FASTPDF2PNG_PATH"); env && *env) {  // pre-commit-allow-getenv
    if (auto p = usable_binary(env, rejected)) return *p;
    if (rejected.empty())
      throw turbo_ocr::PdfRenderError(std::format("FASTPDF2PNG_PATH does not exist: {}", env));
    throw turbo_ocr::PdfRenderError(
        std::format("FASTPDF2PNG_PATH {} {}", rejected.front().path, rejected.front().reason));
  }

  std::vector<std::string> candidates;
  if (const std::string dir = own_executable_dir(); !dir.empty())
    candidates.push_back(dir + "/fastpdf2png");
  for (const char *p : {"/app/bin/fastpdf2png", "/usr/local/bin/fastpdf2png",
                        "./build/fastpdf2png", "./bin/fastpdf2png"})
    candidates.emplace_back(p);

  for (const auto &c : candidates) {
    if (auto p = usable_binary(c, rejected)) return *p;
  }

  std::string msg = "fastpdf2png binary not found (searched:";
  for (const auto &c : candidates) msg += " " + c;
  msg += ").";
  for (const auto &r : rejected) msg += std::format(" {} {}.", r.path, r.reason);
  msg += " Build it with the server (cmake -DTURBO_BUILD_FASTPDF2PNG=ON, the default), run "
         "scripts/install_fastpdf2png.sh, or point FASTPDF2PNG_PATH at a binary for this machine.";
  throw turbo_ocr::PdfRenderError(msg);
}

// Block SIGPIPE on the calling thread for the lifetime of the guard, draining
// any pending instance on destruction. Writing to a daemon whose read end has
// died would otherwise raise SIGPIPE and take down the whole server — exactly
// the M9 crash we must instead recover from. Thread-local mask only; no
// process-wide signal disposition change.
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

} // namespace

namespace turbo_ocr::render::pdfrdetail {

// Parse the page count out of a daemon "OK <n>" reply. A wedged or killed
// child can hand back arbitrary bytes; that must surface as PdfRenderError,
// not as std::invalid_argument escaping the error taxonomy.
[[nodiscard]] int parse_daemon_page_count(const std::string &resp) {
  if (!resp.starts_with("OK ")) return 0;
  int n = 0;
  const char *first = resp.data() + 3;
  const char *last = resp.data() + resp.size();
  auto [ptr, ec] = std::from_chars(first, last, n);
  if (ec != std::errc{} || ptr == first || n < 0 || n > kMaxDaemonPages)
    throw turbo_ocr::PdfRenderError(
        std::format("PDF daemon returned malformed page count: {}", resp));
  return n;
}

} // namespace turbo_ocr::render::pdfrdetail

using turbo_ocr::render::pdfrdetail::parse_daemon_page_count;

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
      binary_path_(find_binary()), daemons_(pool_size) {
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
