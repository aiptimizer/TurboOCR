#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace turbo_ocr::render {

// Pool of persistent fastpdf2png v2.0 daemons.
// Each daemon = separate process with own PDFium + fork workers.
// Raw PPM output (-c -1) to /dev/shm = zero PNG overhead, ramdisk I/O.
// 16 daemons handle concurrent requests independently (no shared state).

class PdfRenderer {
public:
  explicit PdfRenderer(int pool_size = 16, int workers_per_render = 4);
  ~PdfRenderer() noexcept;

  PdfRenderer(const PdfRenderer &) = delete;
  PdfRenderer &operator=(const PdfRenderer &) = delete;
  PdfRenderer(PdfRenderer &&) = delete;
  PdfRenderer &operator=(PdfRenderer &&) = delete;

  /// True when this build can actually rasterize (the daemon/in-process
  /// renderer is compiled in). The no-PDF build still CONSTRUCTS a renderer
  /// (its stubs throw), so "pointer != nullptr" is NOT an availability test —
  /// that exact test made the gRPC PDF_NOT_AVAILABLE guard dead code and a
  /// streamed PDF on a no-PDF build came back as a generic inference error.
  [[nodiscard]] static bool can_render() noexcept;

  /// Render all pages of a PDF document to cv::Mat images.
  [[nodiscard]] std::vector<cv::Mat> render(const uint8_t *data, size_t len, int dpi = 100);

  /// Opaque handle returned by render_streamed_begin. Owns the scratch
  /// tmpfile + tmpdir and must outlive any OCR worker that still holds a
  /// PPM path from the callback — otherwise the file is unlinked from
  /// under the worker. Move-only RAII; destruction cleans both paths.
  struct StreamHandle {
    std::string pdf_tmpfile;
    std::string ppm_tmpdir;
    int num_pages = 0;

    StreamHandle() = default;
    StreamHandle(StreamHandle &&o) noexcept
        : pdf_tmpfile(std::move(o.pdf_tmpfile)),
          ppm_tmpdir(std::move(o.ppm_tmpdir)), num_pages(o.num_pages) {
      o.num_pages = 0;
    }
    StreamHandle &operator=(StreamHandle &&o) noexcept {
      if (this != &o) {
        cleanup();
        pdf_tmpfile = std::move(o.pdf_tmpfile);
        ppm_tmpdir  = std::move(o.ppm_tmpdir);
        num_pages   = o.num_pages;
        o.num_pages = 0;
      }
      return *this;
    }
    StreamHandle(const StreamHandle &) = delete;
    StreamHandle &operator=(const StreamHandle &) = delete;
    ~StreamHandle() noexcept { cleanup(); }

   private:
    void cleanup() noexcept;
  };

  /// Callback type: (page_index, ppm_path) -> void.
  /// `ppm_path` is an absolute path to a PPM file on /dev/shm. The worker
  /// decodes it via decode_ppm() at the point of use — this keeps the
  /// decode cost parallel across OCR workers instead of funnelling it
  /// through the single inotify poll thread.
  using PageCallback = std::function<void(int page_idx, std::string ppm_path)>;

  /// Render pages and invoke `on_page` for each PPM file as soon as it
  /// lands in the tmpdir. Returns a StreamHandle that owns the tmpdir;
  /// callers MUST keep the handle alive until every OCR worker has
  /// finished decoding its page, otherwise the tmpdir is removed early.
  StreamHandle render_streamed(const uint8_t *data, size_t len, int dpi,
                                PageCallback on_page);

  /// Zero-copy PPM → BGR decoder. mmap()s the file, parses the P5/P6
  /// header in-place, allocates one cv::Mat, and does a single-pass
  /// RGB→BGR swap (or GRAY→BGR copy). Safe to call concurrently from
  /// multiple worker threads on different paths; pure function, no
  /// shared state. Returns an empty cv::Mat on malformed/missing file.
  [[nodiscard]] static cv::Mat decode_ppm(const std::string &path);

private:
  struct Daemon {
    // The daemon pool is the POSIX path (pdf_daemon.cpp: fork/waitpid/pipe2)
    // and is not compiled on platforms without inotify — but this header is
    // included by everything that touches PdfRenderer, so the MEMBER still has
    // to name a type that exists. Windows has no pid_t; intptr_t holds a
    // process handle or a pid on every platform this builds for.
#if defined(_WIN32)
    intptr_t pid = -1;
#else
    pid_t pid = -1;
#endif
    FILE *cmd_in = nullptr;
    FILE *result_out = nullptr;
    std::mutex mutex;
  };

  int pool_size_;
  int workers_per_render_;
  std::string binary_path_;
  std::vector<Daemon> daemons_;

  [[nodiscard]] int acquire_daemon();
  [[nodiscard]] std::string send_cmd(Daemon &d, const std::string &cmd);
  // One write+read round-trip; returns false on a pipe write/read failure
  // (daemon crash) instead of throwing, so send_cmd can respawn+retry.
  [[nodiscard]] bool send_cmd_once(Daemon &d, const std::string &cmd,
                                   std::string &out);
  // Fork+exec a fresh fastpdf2png daemon into `d`, closing the inherited
  // fds of every OTHER daemon in the child. Caller must hold d.mutex (or be
  // the single-threaded constructor). Throws PdfRenderError on pipe/fork
  // failure; on success populates d.pid/cmd_in/result_out. Does NOT reap any
  // prior child occupying `d` — see respawn_daemon for the crash-recovery path.
  void spawn_daemon(Daemon &d);
  // Crash recovery: reap the dead child in `d` and spawn a replacement.
  // Caller MUST hold d.mutex. Returns true if a live daemon now occupies `d`.
  [[nodiscard]] bool respawn_daemon(Daemon &d);
  // Back-compat: forwards to decode_ppm(). Used only by the legacy
  // non-streamed render() code path.
  [[nodiscard]] static cv::Mat read_ppm(const std::string &path) {
    return decode_ppm(path);
  }
};

} // namespace turbo_ocr::render
