// Render entry points: temp scratch, blocking render(), streamed render.
#include "turbo_ocr/pdf/render/pdf_renderer.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <poll.h>
#include <sys/inotify.h>
#include <unistd.h>

#include "pdf_renderer_internal.h"
#include "turbo_ocr/common/errors.h"

using namespace turbo_ocr::render;
using turbo_ocr::render::pdfrdetail::kMaxDaemonPages;
using turbo_ocr::render::pdfrdetail::parse_daemon_page_count;
using turbo_ocr::render::pdfrdetail::ppm_is_complete;

namespace {

bool try_write_file(const char *tmpl, const uint8_t *data, size_t len,
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

std::string write_temp_pdf(const uint8_t *data, size_t len) {
  std::string path;
  if (try_write_file("/dev/shm/ocr_pdf_XXXXXX", data, len, path)) return path;
  if (try_write_file("/tmp/ocr_pdf_XXXXXX", data, len, path)) return path;
  throw turbo_ocr::PdfRenderError("Failed to create temp PDF file");
}

std::string make_temp_dir() {
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

} // namespace

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
          if (c < '0' || c > '9' || page_num > kMaxDaemonPages) {
            page_num = -1;
            break;
          }
          page_num = page_num * 10 + (c - '0');
        }
        if (page_num <= 0 || page_num > kMaxDaemonPages) continue;

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
