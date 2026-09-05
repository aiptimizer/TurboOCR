// ServerConfig implementation (cold path: startup-only parsing, validation,
// and diagnostics). Declarations in turbo_ocr/server/bootstrap/server_config.h.
#include "turbo_ocr/server/bootstrap/server_config.h"
#include "turbo_ocr/decode/image_config.h"

#include <cstdio>
#include <cstdlib>

#include "turbo_ocr/common/log/logger.h"

namespace turbo_ocr::server {

size_t ServerConfig::max_image_pixels_bytes() const {
  return static_cast<size_t>(decode::max_image_pixels()) * 3;
}

using namespace env; // env_* parsers (was re-exported by the deleted server/env_utils.h shim)

namespace detail {


/// Minimal JSON string escape — keeps quotes and backslashes safe.
std::string esc(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 2);
  out += '"';
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else out += c;
  }
  out += '"';
  return out;
}

/// Render an optional as a JSON value: `null` when empty, otherwise the
/// underlying value. Strings are escaped through esc() so quotes and
/// backslashes can't corrupt the output.
template <typename T>
std::string opt_json(const std::optional<T> &v) {
  if (!v) return "null";
  if constexpr (std::is_same_v<T, std::string>)
    return esc(*v);
  else
    return std::to_string(*v);
}

/// Cross-field validation shared by both the env-only and CLI-included
/// loader paths. Errors land in `c.errors` (fatal at load_or_die time);
/// warnings land in `c.warnings` (advisory, surfaced by log_effective).
/// The mem-cap clamp also runs here so the field is invariant by the
/// time the caller reads it.
void cross_field_validate(ServerConfig &c, bool mem_explicit) {
  if (c.http_port == c.grpc_port)
    c.errors.push_back("PORT and GRPC_PORT must differ (both = " +
                       std::to_string(c.http_port) + ")");
  if (c.pdf_workers > c.pdf_daemons)
    c.warnings.push_back("PDF_WORKERS (" + std::to_string(c.pdf_workers) +
                         ") exceeds PDF_DAEMONS (" + std::to_string(c.pdf_daemons) +
                         "); excess workers will sit idle");
  if (c.max_body_mem_mb > c.max_body_mb) {
    if (mem_explicit)
      c.warnings.push_back("MAX_BODY_MEMORY_MB (" + std::to_string(c.max_body_mem_mb) +
                           ") > MAX_BODY_MB (" + std::to_string(c.max_body_mb) +
                           "); clamping to body cap");
    c.max_body_mem_mb = c.max_body_mb;
  }
}


} // namespace detail

std::string ServerConfig::to_json() const {
  using detail::esc;
  using detail::opt_json;
  std::string j = "{";
  j += "\"profile\":"            + esc(detail::profile_str(profile));
  j += ",\"host\":"              + esc(host);
  j += ",\"http_port\":"         + std::to_string(http_port);
  j += ",\"grpc_port\":"         + std::to_string(grpc_port);
  j += ",\"request_timeout_ms\":" + std::to_string(request_timeout_ms);
  j += ",\"max_body_mb\":"       + std::to_string(max_body_mb);
  j += ",\"max_body_mem_mb\":"   + std::to_string(max_body_mem_mb);
  j += ",\"pipeline_pool_size\":" + opt_json(pipeline_pool_size);
  j += ",\"http_threads\":"      + opt_json(http_threads);
  j += ",\"pdf_daemons\":"       + std::to_string(pdf_daemons);
  j += ",\"pdf_workers\":"       + std::to_string(pdf_workers);
  j += ",\"shutdown_grace_seconds\":" + std::to_string(shutdown_grace_seconds);
  j += ",\"grpc_cqs\":"          + std::to_string(grpc_cqs);
  j += ",\"grpc_batch_workers\":" + std::to_string(grpc_batch_workers);
  j += ",\"max_pdf_pages\":"     + std::to_string(max_pdf_pages);
  j += ",\"max_batch_images\":"  + std::to_string(max_batch_images);
  j += ",\"max_pdf_page_pixels\":" + std::to_string(max_pdf_page_pixels);
  j += ",\"grpc_response_mode\":" + esc(detail::grpc_mode_str(grpc_response_mode));
  j += ",\"det_onnx\":"          + esc(det_onnx);
  j += ",\"cls_onnx\":"          + esc(cls_onnx);
  j += ",\"layout_onnx\":"       + esc(layout_onnx);
  j += ",\"layout_trt\":"        + (layout_trt ? esc(*layout_trt) : std::string("null"));
  j += ",\"doc_ori_onnx\":"      + esc(doc_ori_onnx);
  j += ",\"rec\":"               + esc(rec_paths.rec);
  j += ",\"rec_dict\":"          + esc(rec_paths.dict);
  j += ",\"ocr_model\":" + esc(selected_model_name);
  j += ",\"ocr_lang\":"          + esc(ocr_lang_value);
  j += ",\"disable_angle_cls\":" + std::string(disable_angle_cls ? "true" : "false");
  j += ",\"cls_all_boxes\":"     + std::string(cls_all_boxes ? "true" : "false");
  j += ",\"layout_disabled\":"   + std::string(layout_disabled ? "true" : "false");
  j += ",\"default_pdf_mode\":"  + esc(pdf::mode_name(default_pdf_mode));
  // Report the EFFECTIVE det config (per-model base + DET_* env overrides) so
  // the post-mortem matches what the detector actually runs. det_max_side was
  // already resolved this way in from_env. (GPU_BOX_THRESH/GPU_UNCLIP_SCALE are
  // a further GPU-only layer applied in paddle_det.cpp and not surfaced here.)
  const auto eff_resize = detection::read_det_resize(det_cfg.resize);
  const auto eff_db = detection::read_db_params(det_cfg.db);
  j += ",\"det_max_side\":"      + std::to_string(det_max_side);
  j += ",\"det_limit_type\":" + esc(eff_resize.limit_type);
  j += ",\"det_limit_side_len\":" + std::to_string(eff_resize.limit_side_len);
  j += ",\"det_max_side_limit\":" + std::to_string(eff_resize.max_side_limit);
  j += ",\"det_db_thresh\":" + std::to_string(eff_db.thresh);
  j += ",\"det_box_thresh\":" + std::to_string(eff_db.box_thresh);
  j += ",\"det_unclip_ratio\":" + std::to_string(eff_db.unclip_ratio);
  j += ",\"trt_opt_level\":"     + std::to_string(trt_opt_level);
  j += ",\"trt_engine_cache\":"  + esc(trt_engine_cache);
  j += ",\"max_image_dim\":"     + std::to_string(max_image_dim);
  j += ",\"page_image_encoder\":" + esc(page_image_encoder);
  j += ",\"ppm_swap\":"          + esc(ppm_swap);
  j += ",\"log_level\":"         + esc(log_level);
  j += ",\"log_format\":"        + esc(log_format);
  j += "}";
  return j;
}

void ServerConfig::log_effective() const {
  TOCR_LOG_INFO("Effective server config", "config", std::string_view(to_json()));
  for (const auto &w : warnings)
    TOCR_LOG_WARN("Config warning", "detail", std::string_view(w));
}

ServerConfig ServerConfig::from_env_and_cli(int argc, char **argv,
                                                    Profile p) {
  ServerConfig c;
  c.profile = p;

  const bool is_gpu = (p == Profile::Gpu);
  const char *rec_env = is_gpu ? "REC_ONNX" : "REC_MODEL";
  const char *det_env = is_gpu ? "DET_ONNX" : "DET_MODEL";
  const char *cls_env = is_gpu ? "CLS_ONNX" : "CLS_MODEL";

  // ---- Pass 1: load from env, accumulating parse errors ----
  // Host accepts BIND_HOST as an alias for TURBO_OCR_HOST; the latter wins when
  // both are set (it predates BIND_HOST). Default 0.0.0.0 (see field doc).
  c.host      = env_or("TURBO_OCR_HOST", env_or("BIND_HOST", "0.0.0.0"));
  c.http_port = env_int_strict("PORT",      8080,  1, 65535, c.errors);
  c.grpc_port = env_int_strict("GRPC_PORT", 50051, 1, 65535, c.errors);

  // Default 60 s: an out-of-the-box 504 beats an unbounded worker hang. Operators
  // may still set 0 explicitly to opt back into unbounded blocking.
  c.request_timeout_ms =
      env_int_strict("REQUEST_TIMEOUT_MS", 60000, 0, 3600000, c.errors);

  c.max_body_mb     = env_int_strict("MAX_BODY_MB",        100,  1, 102400, c.errors);
  c.max_body_mem_mb = env_int_strict("MAX_BODY_MEMORY_MB", 1024, 1, 102400, c.errors);
  // Default of 1024 MB is intentionally a soft ceiling — bodies up to
  // MAX_BODY_MB stay in RAM. Only warn when the operator explicitly raised
  // the in-memory cap above the body cap (likely a misconfiguration).
  const bool mem_explicit = env_present("MAX_BODY_MEMORY_MB");

  if (env_present("PIPELINE_POOL_SIZE"))
    c.pipeline_pool_size = env_int_strict("PIPELINE_POOL_SIZE", 1, 1, 4096, c.errors);
  if (env_present("HTTP_THREADS"))
    c.http_threads = env_int_strict("HTTP_THREADS", 1, 1, 4096, c.errors);
  // Retired in 3.5.3: JPEG decodes on each replica's own decoder, so there
  // is no shared pool to size. Warn instead of erroring so a 3.5.2 deployment
  // still boots; the value is ignored.
  if (env_present("NVJPEG_DECODERS"))
    c.warnings.push_back("NVJPEG_DECODERS is no longer used (JPEG decodes on the "
                         "replica that runs inference); ignoring it");
  c.pdf_daemons = env_int_strict("PDF_DAEMONS", is_gpu ? 16 : 4, 1, 1024, c.errors);
  c.pdf_workers = env_int_strict("PDF_WORKERS", is_gpu ? 4  : 2, 1, 1024, c.errors);
  c.shutdown_grace_seconds = env_int_strict("SHUTDOWN_GRACE_SECONDS", 30, 0, 600, c.errors);

  c.grpc_cqs           = env_int_strict("GRPC_CQS", 10, 1, 1024, c.errors);
  c.grpc_batch_workers = env_int_strict("GRPC_BATCH_WORKERS", 8, 1, 256, c.errors);
  c.max_pdf_pages      = env_int_strict("MAX_PDF_PAGES", 2000, 1, 100000, c.errors);
  c.max_batch_images   = env_int_strict("MAX_BATCH_IMAGES", 1024, 1, 1000000, c.errors);
  c.max_pdf_page_pixels = static_cast<int64_t>(
      env_int_strict("MAX_PDF_PAGE_PIXELS_MP", 40, 1, 268, c.errors)) * 1000000;

  {
    auto mode_s = env_choice_strict("GRPC_RESPONSE_MODE", "json_bytes",
                                     {"json_bytes", "structured"}, c.errors);
    c.grpc_response_mode = (mode_s == "structured")
        ? GrpcResponseMode::structured : GrpcResponseMode::json_bytes;
  }

  // CLS_ONNX/CLS_MODEL takes a path or a shorthand variant name ("x0_25",
  // "x1_0"). cls_explicit records that the user set it, so a missing file can
  // refuse to boot instead of silently disabling orientation handling.
  c.cls_explicit = env_present(cls_env);
  c.cls_onnx    = classification::resolve_cls_shorthand(
      env_or(cls_env, "models/cls.onnx"));
  c.cls_all_boxes = env_bool_strict("CLS_ALL_BOXES", false, c.errors);
  c.layout_onnx = env_or("LAYOUT_ONNX", "models/layout/layout.onnx");
  c.doc_ori_onnx = env_or("DOC_ORI_ONNX", "models/doc_ori.onnx");
  if (env_present("LAYOUT_TRT"))
    c.layout_trt = env_or("LAYOUT_TRT", "");
  // Resolve det/rec/dict from the model registry in one pass: explicit
  // DET_ONNX/REC_ONNX/REC_DICT overrides win per-stage, else OCR_MODEL, else
  // the deprecated OCR_LANG alias (warns), else default "tiny". Unknown
  // OCR_MODEL is fatal (pushed into c.errors).
  {
    ResolvedModel m = resolve_model(rec_env, det_env, "REC_DICT", &c.errors, &c.warnings);
    c.det_onnx = std::move(m.det);
    c.rec_paths = RecPaths{.rec = std::move(m.rec), .dict = std::move(m.dict)};
    c.det_cfg = m.det_cfg;
    c.selected_model_name = m.name;
    c.ocr_lang_value = std::move(m.name);
  }

  // ---- TensorRT / decode / logging knobs (validated here, consumed by
  //      other subsystems via the same env vars they always read) ----
  // Strict-validate DET_MAX_SIDE so a malformed string fails fast, then report
  // the effective engine-profile MAX exactly as the detector + engine builder
  // compute it: effective_det_max_side(read_det_resize(cfg.resize)) — the
  // model's max_side_limit (1280) with DET_LIMIT_*/DET_MAX_SIDE_LIMIT overrides
  // folded in, then DET_MAX_SIDE winning, clamped to [32, 4096].
  (void)env_int_strict("DET_MAX_SIDE", 1280, 32, 4096, c.errors);
  // Validate the rest of the detection-knob family the same way: these are
  // consumed leniently in det_config.h/paddle_det.cpp (shared with CLI/tools),
  // so the server is where a typo must fail fast instead of silently zeroing
  // a threshold or thumbnailing every image (the GitHub #23 failure class).
  (void)env_choice_strict("DET_LIMIT_TYPE", "min", {"min", "max"}, c.errors);
  (void)env_int_strict("DET_LIMIT_SIDE_LEN", 64, 32, 4096, c.errors);
  (void)env_int_strict("DET_MAX_SIDE_LIMIT", 1280, 32, 4096, c.errors);
  (void)env_float_strict("DET_DB_THRESH", 0.2f, 0.001f, 1.0f, c.errors);
  (void)env_float_strict("DET_BOX_THRESH", 0.45f, 0.001f, 1.0f, c.errors);
  (void)env_float_strict("DET_UNCLIP", 1.4f, 0.1f, 10.0f, c.errors);
  (void)env_int_strict("GPU_CCL", 1, 0, 2, c.errors);
  (void)env_float_strict("GPU_BOX_THRESH", 0.45f, 0.001f, 1.0f, c.errors);
  (void)env_float_strict("GPU_UNCLIP_SCALE", 1.0f, 0.1f, 10.0f, c.errors);
  c.det_max_side = detection::effective_det_max_side(detection::read_det_resize(c.det_cfg.resize));
  c.trt_opt_level      = env_int_strict("TRT_OPT_LEVEL", 5,     0,  5,     c.errors);
  c.trt_engine_cache   = env_or("TRT_ENGINE_CACHE", "");
  c.max_image_dim      = env_int_strict("MAX_IMAGE_DIM", 16384, 64, 65535, c.errors);
  c.page_image_encoder = env_choice_strict("TURBO_PDF_IMAGE_ENCODER", "gpu",
      {"cpu", "gpu"}, c.errors);
  c.ppm_swap           = env_choice_strict("TURBO_PPM_SWAP", "simd",
      {"simd", "scalar"}, c.errors);
  c.log_level  = env_choice_strict("LOG_LEVEL",  "info",
      {"debug", "info", "warn", "error"}, c.errors);
  c.log_format = env_choice_strict("LOG_FORMAT", "json",
      {"json", "text"}, c.errors);
  // Consumed by the layout module (layout_postfilter.h); validated here so a
  // typo fails startup instead of silently falling back to the default.
  // Canonical names are "all"/"outer"/"inner"; the old "union"/"large"/"small"
  // strings are still accepted as deprecated aliases.
  (void)env_choice_strict("LAYOUT_MERGE_MODE", "all",
      {"all", "outer", "inner", "union", "large", "small"}, c.errors);

  c.disable_angle_cls = env_bool_strict("DISABLE_ANGLE_CLS", false, c.errors);
  if (c.cls_all_boxes && c.disable_angle_cls)
    c.warnings.push_back(
        "CLS_ALL_BOXES=1 has no effect with DISABLE_ANGLE_CLS=1 — the angle "
        "classifier is disabled entirely");
  c.layout_disabled   = env_bool_strict("DISABLE_LAYOUT",    false, c.errors);

  // ENABLE_LAYOUT removed — operators must migrate.
  if (env_present("ENABLE_LAYOUT")) {
    c.errors.push_back(
        "ENABLE_LAYOUT is no longer supported. Set DISABLE_LAYOUT=1 to "
        "disable layout, or remove this env var (layout is on by default).");
  }

  // Table/formula recognition runs on the regions LAYOUT classifies, so a
  // configured table/formula backend REQUIRES layout. Refuse to boot with one
  // configured while layout is disabled — the stage would load and then
  // silently produce nothing (no regions to fill). The enable set mirrors
  // gpu_pipeline_pool.h::maybe_load_router_models.
  if (c.layout_disabled &&
      (env_present("FORMULA_BACKEND") || env_present("FORMULA_ONNX") ||
       env_present("TABLE_BACKEND") || env_present("TABLE_CLS_TRT") ||
       env_present("TABLE_SLANEXT_ENCODER_ONNX") ||
       env_present("VLLM_TABLE_BASE_URL") || env_present("TURBO_ROUTING_CONFIG"))) {
    c.errors.push_back(
        "DISABLE_LAYOUT is set but a table/formula backend is configured; those "
        "stages recognize layout regions and would produce nothing. Remove "
        "DISABLE_LAYOUT, or unset the table/formula backend env.");
  }

  if (env_present("ENABLE_PDF_MODE")) {
    auto m = env_choice_strict("ENABLE_PDF_MODE", "ocr",
        {"ocr", "geometric", "auto", "auto_verified"}, c.errors);
    c.default_pdf_mode = pdf::parse_pdf_mode(m);
    c.default_pdf_mode_was_set = true;
  }

  // ---- Pass 2: CLI flags (override env) ----
  if (argc > 0 && argv) {
    CLI::App app{"TurboOCR server — GPU/CPU OCR + layout HTTP/gRPC service"};
    app.set_help_flag("-h,--help", "Print this help message and exit");
    bool flag_print_config = false;
    bool flag_check_config = false;
    app.add_flag("--print-config", flag_print_config,
                 "Print resolved configuration as JSON and exit (zero)");
    app.add_flag("--check-config", flag_check_config,
                 "Validate configuration and exit (zero on valid, 2 on errors)");

    app.add_option("--host",        c.host,        "Bind address for HTTP and gRPC (default 0.0.0.0)")->capture_default_str();
    app.add_option("--http-port",   c.http_port,   "HTTP port")->capture_default_str()->check(CLI::Range(1, 65535));
    app.add_option("--grpc-port",   c.grpc_port,   "gRPC port")->capture_default_str()->check(CLI::Range(1, 65535));
    app.add_option("--request-timeout-ms", c.request_timeout_ms,
        "Per-request inference deadline (ms); 0 = disabled (default); >0 returns 504 on overrun")
        ->capture_default_str()->check(CLI::Range(0, 3600000));
    app.add_option("--max-body-mb", c.max_body_mb, "Max request body size (MB)")->capture_default_str()->check(CLI::Range(1, 102400));
    app.add_option("--max-body-memory-mb", c.max_body_mem_mb,
        "In-memory body buffer cap (MB); always clamped to --max-body-mb so effective default is min(1024, MAX_BODY_MB)")
        ->check(CLI::Range(1, 102400));

    int pool_size_cli = c.pipeline_pool_size.value_or(0);
    int http_threads_cli = c.http_threads.value_or(0);
    auto *opt_pool = app.add_option("--pool-size", pool_size_cli,
        "Pipeline pool size (0 = auto from VRAM on GPU / 4 on CPU)")->check(CLI::Range(0, 4096));
    auto *opt_http = app.add_option("--http-threads", http_threads_cli,
        "HTTP work pool threads (0 = auto from pool size)")->check(CLI::Range(0, 4096));
    // Retired in 3.5.3 (see NVJPEG_DECODERS above): still accepted so a 3.5.2
    // command line keeps booting; the value is ignored with a warning.
    int nvjpeg_decoders_retired = 0;
    auto *opt_nvjpeg = app.add_option("--nvjpeg-decoders", nvjpeg_decoders_retired,
        "Retired (no longer used): JPEG decodes on the replica that runs inference")->check(CLI::Range(0, 256));
    app.add_option("--pdf-daemons",  c.pdf_daemons,  "PDF render daemons")->capture_default_str()->check(CLI::Range(1, 1024));
    app.add_option("--pdf-workers",  c.pdf_workers,  "PDF render workers")->capture_default_str()->check(CLI::Range(1, 1024));
    app.add_option("--shutdown-grace", c.shutdown_grace_seconds, "Graceful drain seconds before exit")->capture_default_str()->check(CLI::Range(0, 600));
    app.add_option("--grpc-cqs",            c.grpc_cqs,            "gRPC completion queue count")->capture_default_str()->check(CLI::Range(1, 1024));
    app.add_option("--grpc-batch-workers",  c.grpc_batch_workers,  "Parallel workers in gRPC RecognizeBatch")->capture_default_str()->check(CLI::Range(1, 256));
    app.add_option("--max-pdf-pages",       c.max_pdf_pages,       "Max pages per PDF request")->capture_default_str()->check(CLI::Range(1, 100000));
    app.add_option("--max-batch-images",    c.max_batch_images,    "Max images per /ocr/batch request")->capture_default_str()->check(CLI::Range(1, 1000000));

    std::string grpc_mode_cli = (c.grpc_response_mode == GrpcResponseMode::structured)
                                  ? "structured" : "json_bytes";
    app.add_option("--grpc-response-mode", grpc_mode_cli, "gRPC response mode")
        ->capture_default_str()
        ->check(CLI::IsMember({"json_bytes", "structured"}));

    app.add_option("--det-onnx",    c.det_onnx,    "Detection model ONNX path")->capture_default_str();
    app.add_option("--cls-onnx",    c.cls_onnx,    "Angle-classification model ONNX path")->capture_default_str();
    app.add_option("--layout-onnx", c.layout_onnx, "Layout-detection model ONNX path")->capture_default_str();
    std::string layout_trt_cli = c.layout_trt.value_or("");
    auto *opt_layout_trt = app.add_option("--layout-trt", layout_trt_cli,
        "Pre-built layout TRT engine (GPU only; overrides --layout-onnx build)");

    app.add_option("--det-max-side",      c.det_max_side,    "Max detection input side (px); changes invalidate cached TRT engine")
        ->capture_default_str()->check(CLI::Range(32, 4096));
    app.add_option("--trt-opt-level",     c.trt_opt_level,   "TensorRT builder optimization level (0=fast build / 5=fast runtime)")
        ->capture_default_str()->check(CLI::Range(0, 5));
    app.add_option("--trt-engine-cache",  c.trt_engine_cache,"Directory for cached TensorRT engines (empty = ~/.cache/turbo-ocr)")
        ->capture_default_str();
    app.add_option("--max-image-dim",     c.max_image_dim,   "Max image width/height (px) accepted on decode routes")
        ->capture_default_str()->check(CLI::Range(64, 65535));
    app.add_option("--log-level",         c.log_level,       "Log level")
        ->capture_default_str()->check(CLI::IsMember({"debug", "info", "warn", "error"}));
    app.add_option("--log-format",        c.log_format,      "Log output format")
        ->capture_default_str()->check(CLI::IsMember({"json", "text"}));

    app.add_flag("--disable-angle-cls", c.disable_angle_cls, "Skip angle classifier");
    app.add_flag("--disable-layout",    c.layout_disabled,   "Skip layout detection");

    std::string pdf_mode_cli = std::string(pdf::mode_name(c.default_pdf_mode));
    auto *opt_pdf_mode = app.add_option("--default-pdf-mode", pdf_mode_cli, "Default PDF extraction mode")
        ->capture_default_str()
        ->check(CLI::IsMember({"ocr", "geometric", "auto", "auto_verified"}));

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      // --help and --version exit 0 here; anything else is a hard error.
      std::exit(app.exit(e));
    }

    // Reflect CLI overrides back onto the optional / enum fields.
    if (opt_pool->count() > 0)
      c.pipeline_pool_size = pool_size_cli > 0 ? std::optional<int>(pool_size_cli) : std::nullopt;
    if (opt_http->count() > 0)
      c.http_threads = http_threads_cli > 0 ? std::optional<int>(http_threads_cli) : std::nullopt;
    if (opt_nvjpeg->count() > 0)
      c.warnings.push_back("--nvjpeg-decoders is no longer used (JPEG decodes on the "
                           "replica that runs inference); ignoring it");
    if (opt_layout_trt->count() > 0)
      c.layout_trt = layout_trt_cli.empty() ? std::nullopt : std::optional<std::string>(layout_trt_cli);
    c.grpc_response_mode = (grpc_mode_cli == "structured")
        ? GrpcResponseMode::structured : GrpcResponseMode::json_bytes;
    if (opt_pdf_mode->count() > 0) {
      c.default_pdf_mode = pdf::parse_pdf_mode(pdf_mode_cli);
      c.default_pdf_mode_was_set = true;
    }

    detail::cross_field_validate(c, mem_explicit);

    // ---- Print/check modes ----
    if (flag_print_config) {
      std::cout << c.to_json() << "\n";
      std::exit(0);
    }
    if (flag_check_config) {
      if (c.errors.empty()) {
        std::cerr << "config OK\n";
        std::exit(0);
      }
      for (const auto &e : c.errors) std::cerr << "[config error] " << e << "\n";
      std::exit(2);
    }
  } else {
    detail::cross_field_validate(c, mem_explicit);
  }

  return c;
}

ServerConfig ServerConfig::load_or_die(int argc, char **argv,
                                               Profile p) {
  ServerConfig c = from_env_and_cli(argc, argv, p);
  if (!c.errors.empty()) {
    for (const auto &e : c.errors)
      std::cerr << "[config error] " << e << "\n";
    std::cerr << "Refusing to start with invalid configuration. "
                 "Use --check-config to validate without booting the pipeline.\n";
    std::exit(2);
  }
  return c;
}


} // namespace turbo_ocr::server
