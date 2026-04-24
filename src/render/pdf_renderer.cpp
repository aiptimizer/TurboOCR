#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/common/errors.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <fcntl.h>
#include <poll.h>
#include <sys/inotify.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace turbo_ocr::render;

static std::string find_binary() {
  // Explicit override — used by tests and by deployments that put the binary
  // in a non-standard location. Fails fast if the configured path is missing
  // rather than falling back to the default search (surprises hurt in prod).
  if (const char *env = std::getenv("FASTPDF2PNG_PATH"); env && *env) {
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
    try {
      if (is_dir) std::filesystem::remove_all(path);
      else unlink(path.c_str());
    } catch (...) {}
  }
  void release() { path.clear(); }
  TempGuard(const TempGuard &) = delete;
  TempGuard &operator=(const TempGuard &) = delete;
};

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
  const unsigned char *end  = base + file_size;
  const unsigned char *p    = base;

  // Magic: "P5" (gray) or "P6" (color RGB).
  if (p[0] != 'P' || (p[1] != '5' && p[1] != '6')) return {};
  const bool gray = (p[1] == '5');
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

  int w = 0, h = 0, maxval = 0;
  if (!next_int(w) || !next_int(h) || !next_int(maxval)) return {};
  if (w <= 0 || h <= 0 || w > 16384 || h > 16384 || maxval != 255) return {};
  // After maxval there's exactly one whitespace byte before the payload.
  if (p >= end) return {};
  ++p;

  const size_t expected = static_cast<size_t>(w) * h * (gray ? 1 : 3);
  if (static_cast<size_t>(end - p) < expected) return {};

  if (gray) {
    cv::Mat g(h, w, CV_8UC1);
    std::memcpy(g.data, p, expected);
    cv::Mat bgr;
    cv::cvtColor(g, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }

  // Color: single-pass RGB→BGR copy, one write-back over the pixels.
  cv::Mat bgr(h, w, CV_8UC3);
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
  return bgr;
}

PdfRenderer::PdfRenderer(int pool_size, int workers_per_render)
    : pool_size_(pool_size), workers_per_render_(workers_per_render),
      daemons_(pool_size) {
  binary_path_ = find_binary();

  for (int i = 0; i < pool_size_; ++i) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0)
      throw turbo_ocr::PdfRenderError("pipe() failed for PDF renderer daemon");

    pid_t pid = fork();
    if (pid < 0) throw turbo_ocr::PdfRenderError("fork() failed for PDF renderer daemon");

    if (pid == 0) {
      dup2(in_pipe[0], STDIN_FILENO);
      dup2(out_pipe[1], STDOUT_FILENO);
      close(in_pipe[0]); close(in_pipe[1]);
      close(out_pipe[0]); close(out_pipe[1]);
      for (int j = 0; j < i; ++j) {
        if (daemons_[j].cmd_in) fclose(daemons_[j].cmd_in);
        if (daemons_[j].result_out) fclose(daemons_[j].result_out);
      }
      execl(binary_path_.c_str(), binary_path_.c_str(), "--daemon", nullptr);
      _exit(1);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);
    daemons_[i].pid = pid;
    daemons_[i].cmd_in = fdopen(in_pipe[1], "w");
    daemons_[i].result_out = fdopen(out_pipe[0], "r");
    if (!daemons_[i].cmd_in || !daemons_[i].result_out)
      throw turbo_ocr::PdfRenderError("fdopen failed for PDF renderer daemon");
  }

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

std::string PdfRenderer::send_cmd(Daemon &d, const std::string &cmd) {
  fprintf(d.cmd_in, "%s\n", cmd.c_str());
  fflush(d.cmd_in);
  char buf[4096];
  if (!fgets(buf, sizeof(buf), d.result_out))
    throw turbo_ocr::PdfRenderError("PDF renderer daemon read failed (daemon may have crashed)");
  auto len = std::strlen(buf);
  if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
  return buf;
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

  int num_pages = 0;
  if (resp.starts_with("OK "))
    num_pages = std::stoi(resp.substr(3));

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
  try {
    if (!pdf_tmpfile.empty()) ::unlink(pdf_tmpfile.c_str());
    if (!ppm_tmpdir.empty())  std::filesystem::remove_all(ppm_tmpdir);
  } catch (...) {}
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
  // CLOSE_WRITE fires when a worker finishes writing a PPM file.
  int inotify_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (inotify_fd < 0)
    throw turbo_ocr::PdfRenderError("inotify_init1 failed");

  int wd = inotify_add_watch(inotify_fd, tmpdir.path.c_str(), IN_CLOSE_WRITE);
  if (wd < 0) {
    close(inotify_fd);
    throw turbo_ocr::PdfRenderError("inotify_add_watch failed");
  }

  // Track which pages have been delivered to avoid duplicates.
  // Uses a bitset-style vector; pages delivered via inotify are marked here
  // so the safety-net scan at the end skips them. We start with a generous
  // pre-allocation (resized as needed when page indices arrive).
  std::vector<bool> delivered(256, false); // pre-alloc for typical PDFs

  // Launch render in a background thread so we can process inotify events
  // concurrently. The daemon mutex is held for the duration of RENDER.
  int idx = acquire_daemon();
  std::atomic<bool> render_done{false};
  std::string render_resp;
  std::exception_ptr render_error;

  std::thread render_thread([&]() {
    try {
      std::unique_lock<std::mutex> daemon_lock(daemons_[idx].mutex,
                                                std::adopt_lock);
      render_resp = send_cmd(daemons_[idx],
          std::format("RENDER\t{}\t{}\t{}\t{}\t-1",
                      tmpfile.path, pattern, dpi, workers_per_render_));
    } catch (...) {
      render_error = std::current_exception();
    }
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

  // Drain any remaining inotify events
  process_events();

  // Clean up inotify
  inotify_rm_watch(inotify_fd, wd);
  close(inotify_fd);

  if (render_error) std::rethrow_exception(render_error);

  if (!render_resp.starts_with("OK"))
    throw turbo_ocr::PdfRenderError(
        std::format("PDF render failed: {}", render_resp));

  int num_pages = 0;
  if (render_resp.starts_with("OK "))
    num_pages = std::stoi(render_resp.substr(3));

  // Safety net: deliver any pages missed by inotify (race, coalesced events).
  // The daemon may respond "OK N" before its forked workers finish writing
  // the last PPM files, so retry briefly if expected files are missing.
  if (pages_delivered < num_pages) {
    if (num_pages > static_cast<int>(delivered.size()))
      delivered.resize(num_pages, false);
    for (int retry = 0; retry < 50 && pages_delivered < num_pages; ++retry) {
      for (int i = 0; i < num_pages; ++i) {
        if (delivered[i]) continue;
        std::string ppm_path = std::format("{}/p_{:04d}.ppm", tmpdir.path, i + 1);
        if (!std::filesystem::exists(ppm_path)) continue;
        delivered[i] = true;
        on_page(i, std::move(ppm_path));
        ++pages_delivered;
      }
      if (pages_delivered < num_pages)
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
