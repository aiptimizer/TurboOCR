#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/grpc/grpc_response_mode.h"
#include "turbo_ocr/server/language_paths.h"

#include <CLI11.hpp>

namespace turbo_ocr::server {

enum class Profile { Gpu, Cpu };

/// Build-time default profile derived from USE_CPU_ONLY. Lets the same
/// from_env() call site work in either binary without hard-coding.
constexpr Profile build_profile() noexcept {
#ifdef USE_CPU_ONLY
  return Profile::Cpu;
#else
  return Profile::Gpu;
#endif
}

/// Centralized server configuration. Owns every env-var and CLI-flag read
/// for the HTTP/gRPC servers. Loaded once at startup via load_or_die().
struct ServerConfig {
  // ---- Network ----
  // Binds all interfaces by default. Set BIND_HOST (or TURBO_OCR_HOST / --host)
  // to restrict, e.g. 127.0.0.1 when the only client is a co-located proxy.
  std::string host = "0.0.0.0";  // TURBO_OCR_HOST / BIND_HOST / --host
  int http_port = 8080;          // PORT / --http-port
  int grpc_port = 50051;         // GRPC_PORT / --grpc-port

  // ---- Request lifecycle ----
  // Per-request inference deadline (ms). When > 0, a submit whose future does
  // not resolve within this window returns 504 INFERENCE_TIMEOUT and frees its
  // slot. The effective default (set by from_env) is 60000 — ON by default, so a
  // wedged GPU slot is recovered instead of hanging unbounded; set
  // REQUEST_TIMEOUT_MS=0 to opt out (unbounded wait, the pre-hardening behavior).
  // Applies to single-image / batch / gRPC; the PDF path scales it by page count
  // and surfaces a timeout as 504 (see pdf_job.h).
  int request_timeout_ms = 0;  // REQUEST_TIMEOUT_MS (from_env default 60000; 0 = off)

  // ---- Body limits ----
  int max_body_mb     = 100;     // MAX_BODY_MB / --max-body-mb
  int max_body_mem_mb = 1024;    // MAX_BODY_MEMORY_MB / --max-body-memory-mb

  // ---- Pipeline / threading ----
  // `pipeline_pool_size`, `http_threads` and `nvjpeg_decoders` are deliberately
  // `std::optional`: `nullopt` means "operator did not set this; caller
  // chooses the default." GPU main.cpp picks via VRAM auto-detect when pool
  // size is unset; CPU cpu_main.cpp uses a hard-coded 4. http_threads defaults
  // to `max(pool_size*32, 128)` on GPU and isn't read on CPU. nvjpeg_decoders
  // defaults to the pool size (one shared decoder per replica) on GPU.
  std::optional<int> pipeline_pool_size;  // PIPELINE_POOL_SIZE / --pool-size
  std::optional<int> http_threads;        // HTTP_THREADS / --http-threads (GPU only consumer)
  std::optional<int> nvjpeg_decoders;     // NVJPEG_DECODERS / --nvjpeg-decoders (GPU only consumer)
  int pdf_daemons            = 16;        // PDF_DAEMONS / --pdf-daemons
  int pdf_workers            = 4;         // PDF_WORKERS / --pdf-workers
  int shutdown_grace_seconds = 30;        // SHUTDOWN_GRACE_SECONDS / --shutdown-grace

  // ---- gRPC tuning ----
  int grpc_cqs           = 10;            // GRPC_CQS / --grpc-cqs
  int grpc_batch_workers = 8;             // GRPC_BATCH_WORKERS / --grpc-batch-workers
  int max_pdf_pages      = 2000;          // MAX_PDF_PAGES / --max-pdf-pages
  // Max images per /ocr/batch (HTTP + gRPC RecognizeBatch) request. The
  // handlers allocate O(n) per-slot vectors + an n*1KB JSON reserve before
  // touching any image, so an unbounded array is an OOM lever — reject over
  // this with 400. MAX_BATCH_IMAGES.
  int max_batch_images   = 1024;
  // Max rendered pixels per PDF page (width*height after mediabox*dpi). Caps
  // both the rasterized buffer and any inline page image, independent of the
  // per-side decode_ppm cap. ~33.5M = 16384*2048; a full 16384² page is ~268M
  // and would be rejected. MAX_PDF_PAGE_PIXELS.
  int64_t max_pdf_page_pixels = 40000000;  // ~40 MP (e.g. 5000x8000)
  GrpcResponseMode grpc_response_mode = GrpcResponseMode::json_bytes;

  // ---- Model paths ----
  std::string det_onnx;
  std::string cls_onnx;
  std::string layout_onnx = "models/layout/layout.onnx";
  std::optional<std::string> layout_trt;
  // Document-orientation model (PP-LCNet_x1_0_doc_ori) for /ocr/pdf?autorotate=1.
  // Optional — if the file is absent, autorotate requests are rejected.
  std::string doc_ori_onnx = "models/doc_ori.onnx";
  RecPaths    rec_paths;
  // Registry name of the selected OCR model (OCR_MODEL / OCR_LANG alias /
  // default "tiny"). Always set to the resolved registry entry; an explicit
  // REC override replaces the path but not the name. `ocr_lang_value` is the
  // deprecated alias kept for back-compat logging and the "ocr_lang" JSON
  // field; it mirrors selected_model_name.
  std::string selected_model_name;
  std::string ocr_lang_value;
  // Per-model detection inference config (resize policy + DB params) for the
  // selected model. Threaded into each detector at construction; env per-field
  // overrides are applied there (read_det_resize/read_db_params) so env wins.
  DetInferConfig det_cfg{};

  // ---- TensorRT / decode tuning (consumed by engine + decode subsystems;
  //      strict-validated here so malformed input fails fast at boot) ----
  // Effective engine optimization-profile MAX side, resolved by from_env the
  // same way the detector + TRT builder do: the selected model's max_side_limit
  // (1280) with DET_LIMIT_*/DET_MAX_SIDE_LIMIT folded in, DET_MAX_SIDE winning,
  // /32-ceiled and clamped. The 1280 placeholder is never read — from_env
  // always recomputes it before validation. DET_MAX_SIDE [32, 4096].
  int det_max_side = 1280;
  int         trt_opt_level   = 5;     // TRT_OPT_LEVEL [0, 5]
  std::string trt_engine_cache;        // TRT_ENGINE_CACHE (empty = "~/.cache/turbo-ocr")
  int         max_image_dim   = 16384; // MAX_IMAGE_DIM [64, 65535]

  // ---- Page-image export / PDF render tuning (validated here; consumed by
  //      the encode + render subsystems via the same env vars they read) ----
  // page_image_encoder is honored only by the GPU build's JPEG page-image
  // path; on the CPU-only build it is reported but inert (no nvJPEG present).
  std::string page_image_encoder = "gpu";  // TURBO_PDF_IMAGE_ENCODER {cpu, gpu}
  std::string ppm_swap           = "simd"; // TURBO_PPM_SWAP {simd, scalar}

  // ---- Logging ----
  std::string log_level  = "info";     // LOG_LEVEL {debug, info, warn, error}
  std::string log_format = "json";     // LOG_FORMAT {json, text}

  // ---- Feature toggles ----
  bool        disable_angle_cls = false;
  // CLS_ALL_BOXES=1: classify orientation on every crop, not just vertical
  // ones. cls_explicit: CLS_ONNX/CLS_MODEL was set by the user (missing file
  // is then fatal instead of a silent disable).
  bool        cls_all_boxes     = false;
  bool        cls_explicit      = false;
  bool        layout_disabled   = false;
  pdf::PdfMode default_pdf_mode = pdf::PdfMode::Ocr;
  bool        default_pdf_mode_was_set = false;

  /// Effective profile this config was loaded for. Set by from_env.
  Profile profile = build_profile();

  /// Errors accumulated during parsing or cross-field validation. If
  /// non-empty after load, load_or_die() prints them and exit(2).
  std::vector<std::string> errors;

  /// Warnings — recorded but not fatal. Logged at startup.
  std::vector<std::string> warnings;

  /// Parse env vars + CLI flags. CLI overrides env; both override defaults.
  /// Returns a ServerConfig with .errors populated for any malformed input.
  /// Pass argc=0, argv=nullptr for env-only loading.
  static ServerConfig from_env_and_cli(int argc, char **argv,
                                        Profile p = build_profile());

  /// Same as from_env_and_cli but env-only (for tests and library use).
  static ServerConfig from_env(Profile p = build_profile()) {
    return from_env_and_cli(0, nullptr, p);
  }

  /// Load, validate, and exit(2) with a diagnostic list on any error.
  /// Also handles --print-config / --check-config (those exit(0)).
  static ServerConfig load_or_die(int argc, char **argv,
                                   Profile p = build_profile());

  /// Emit one structured INFO log line with every resolved value, so
  /// operators can grep a single post-mortem source of truth.
  void log_effective() const;

  /// JSON dump used by both --print-config and log_effective().
  [[nodiscard]] std::string to_json() const;
};

namespace detail {

inline std::string_view grpc_mode_str(GrpcResponseMode m) noexcept {
  return m == GrpcResponseMode::structured ? "structured" : "json_bytes";
}

inline std::string_view profile_str(Profile p) noexcept {
  return p == Profile::Cpu ? "cpu" : "gpu";
}
} // namespace detail

} // namespace turbo_ocr::server
