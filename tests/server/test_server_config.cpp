#include <catch_amalgamated.hpp>

#include <cstdlib>

#include "turbo_ocr/server/bootstrap/server_config.h"

using turbo_ocr::server::Profile;
using turbo_ocr::server::ServerConfig;

namespace {

// Every env var ServerConfig::from_env touches — wiped between cases so each
// test starts from a known baseline.
const char* const kAllEnvVars[] = {
    "TURBO_OCR_HOST",
    "BIND_HOST",
    "REQUEST_TIMEOUT_MS",
    "PORT",
    "GRPC_PORT",
    "MAX_BODY_MB",
    "MAX_BODY_MEMORY_MB",
    "PIPELINE_POOL_SIZE",
    "HTTP_THREADS",
    "PDF_DAEMONS",
    "PDF_WORKERS",
    "SHUTDOWN_GRACE_SECONDS",
    "GRPC_CQS",
    "GRPC_BATCH_WORKERS",
    "MAX_PDF_PAGES",
    "GRPC_RESPONSE_MODE",
    "DET_ONNX",
    "DET_MODEL",
    "CLS_ONNX",
    "CLS_MODEL",
    "LAYOUT_ONNX",
    "LAYOUT_TRT",
    "REC_ONNX",
    "REC_MODEL",
    "REC_DICT",
    "OCR_LANG",
    "OCR_MODEL",
    "TURBO_OCR_REC_ALLOW_TINY",
    "DISABLE_ANGLE_CLS",
    "DISABLE_LAYOUT",
    "LAYOUT_MERGE_MODE",
    "ENABLE_LAYOUT",
    "ENABLE_PDF_MODE",
    "DET_MAX_SIDE",
    "DET_MAX_SIDE_LIMIT",
    "DET_LIMIT_TYPE",
    "DET_LIMIT_SIDE_LEN",
    "DET_DB_THRESH",
    "DET_BOX_THRESH",
    "DET_UNCLIP",
    "TRT_OPT_LEVEL",
    "TRT_ENGINE_CACHE",
    "MAX_IMAGE_DIM",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "MAX_BATCH_IMAGES",
    "MAX_PDF_PAGE_PIXELS_MP",
    "DOC_ORI_ONNX",
    "CLS_ALL_BOXES",
};

void reset_env() {
  for (const char *v : kAllEnvVars) ::unsetenv(v);
}

} // namespace

TEST_CASE("from_env defaults are sane (GPU)", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env(Profile::Gpu);
  CHECK(c.errors.empty());
  CHECK(c.host == "0.0.0.0");
  CHECK(c.http_port == 8080);
  CHECK(c.grpc_port == 50051);
  CHECK(c.max_body_mb == 100);
  // Default mem (1024) is silently clamped to body cap (100); no warning.
  CHECK(c.max_body_mem_mb == 100);
  CHECK(c.warnings.empty());
  CHECK_FALSE(c.pipeline_pool_size.has_value());
  CHECK_FALSE(c.http_threads.has_value());
  CHECK(c.pdf_daemons == 16);
  CHECK(c.pdf_workers == 4);
  CHECK(c.shutdown_grace_seconds == 30);
  CHECK(c.grpc_cqs == 10);
  CHECK(c.grpc_batch_workers == 8);
  CHECK(c.max_pdf_pages == 2000);
  // Default model is "tiny" (the throughput tier), whose per-tier detector is
  // det_tiny.onnx (not the shared det.onnx, which backs only the V5Lang rows).
  CHECK(c.det_onnx == "models/det_tiny.onnx");
  CHECK(c.cls_onnx == "models/cls.onnx");
  CHECK_FALSE(c.disable_angle_cls);
  CHECK_FALSE(c.layout_disabled);
  CHECK_FALSE(c.default_pdf_mode_was_set);
  CHECK(c.profile == Profile::Gpu);
}

TEST_CASE("cls shorthand names resolve to bundled paths", "[server_config]") {
  reset_env();
  ::setenv("CLS_ONNX", "x1_0", 1);
  auto c = ServerConfig::from_env(Profile::Gpu);
  CHECK(c.errors.empty());
  CHECK(c.cls_onnx == "models/cls_x1_0.onnx");
  CHECK(c.cls_explicit);

  ::setenv("CLS_ONNX", "x0_25", 1);
  auto c2 = ServerConfig::from_env(Profile::Gpu);
  CHECK(c2.cls_onnx == "models/cls.onnx");

  // CPU profile uses CLS_MODEL for the same shorthand.
  reset_env();
  ::setenv("CLS_MODEL", "x1_0", 1);
  auto c3 = ServerConfig::from_env(Profile::Cpu);
  CHECK(c3.cls_onnx == "models/cls_x1_0.onnx");
  CHECK(c3.cls_explicit);

  // A plain path passes through untouched, and an unset env is not explicit.
  reset_env();
  ::setenv("CLS_ONNX", "/opt/models/custom_cls.onnx", 1);
  auto c4 = ServerConfig::from_env(Profile::Gpu);
  CHECK(c4.cls_onnx == "/opt/models/custom_cls.onnx");
  reset_env();
  auto c5 = ServerConfig::from_env(Profile::Gpu);
  CHECK_FALSE(c5.cls_explicit);
}

TEST_CASE("CLS_ALL_BOXES parses strictly", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env(Profile::Gpu);
  CHECK_FALSE(c.cls_all_boxes);

  ::setenv("CLS_ALL_BOXES", "1", 1);
  auto c1 = ServerConfig::from_env(Profile::Gpu);
  CHECK(c1.errors.empty());
  CHECK(c1.cls_all_boxes);

  ::setenv("CLS_ALL_BOXES", "banana", 1);
  auto c2 = ServerConfig::from_env(Profile::Gpu);
  CHECK_FALSE(c2.errors.empty());  // garbage must fail loud, not default

  // Contradictory combo warns (classifier disabled entirely).
  ::setenv("CLS_ALL_BOXES", "1", 1);
  ::setenv("DISABLE_ANGLE_CLS", "1", 1);
  auto c3 = ServerConfig::from_env(Profile::Gpu);
  CHECK(c3.errors.empty());
  CHECK_FALSE(c3.warnings.empty());
}

TEST_CASE("runtime truthy reader matches env_bool_strict's accepted set",
          "[server_config]") {
  // The boot validator (env_bool_strict) and the pipelines' runtime reader
  // (truthy_env_value) must agree, or CLS_ALL_BOXES=true would validate at
  // boot yet silently run with the feature off.
  using turbo_ocr::classification::truthy_env_value;
  for (const char *v : {"1", "true", "TRUE", "yes", "on", "On"})
    CHECK(truthy_env_value(v));
  for (const char *v : {"0", "false", "no", "off", "", "banana"})
    CHECK_FALSE(truthy_env_value(v));
  CHECK_FALSE(truthy_env_value(nullptr));
}

TEST_CASE("from_env defaults differ on CPU profile", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env(Profile::Cpu);
  CHECK(c.errors.empty());
  CHECK(c.pdf_daemons == 4);
  CHECK(c.pdf_workers == 2);
  CHECK(c.profile == Profile::Cpu);
}

TEST_CASE("from_env uses profile-specific model env names", "[server_config]") {
  reset_env();
  ::setenv("DET_ONNX",  "trt_det.onnx", 1);
  ::setenv("DET_MODEL", "cpu_det.onnx", 1);
  CHECK(ServerConfig::from_env(Profile::Gpu).det_onnx == "trt_det.onnx");
  CHECK(ServerConfig::from_env(Profile::Cpu).det_onnx == "cpu_det.onnx");
}

TEST_CASE("from_env accepts valid integer values", "[server_config]") {
  reset_env();
  ::setenv("PORT",      "9000",  1);
  ::setenv("GRPC_PORT", "50061", 1);
  ::setenv("MAX_BODY_MB", "250", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  CHECK(c.http_port == 9000);
  CHECK(c.grpc_port == 50061);
  CHECK(c.max_body_mb == 250);
}

TEST_CASE("from_env rejects malformed integers", "[server_config]") {
  reset_env();
  ::setenv("PORT", "abc", 1);
  auto c = ServerConfig::from_env();
  REQUIRE(c.errors.size() == 1);
  CHECK(c.errors[0].find("PORT") != std::string::npos);
  CHECK(c.errors[0].find("abc")  != std::string::npos);
  // Falls back to default while reporting the error.
  CHECK(c.http_port == 8080);
}

TEST_CASE("from_env rejects out-of-range integers", "[server_config]") {
  reset_env();
  ::setenv("PORT", "70000", 1);
  auto c = ServerConfig::from_env();
  REQUIRE_FALSE(c.errors.empty());
  CHECK(c.errors[0].find("70000") != std::string::npos);
  CHECK(c.errors[0].find("[1, 65535]") != std::string::npos);
}

TEST_CASE("from_env accumulates multiple errors", "[server_config]") {
  reset_env();
  ::setenv("PORT",         "abc",   1);
  ::setenv("GRPC_PORT",    "xyz",   1);
  ::setenv("MAX_BODY_MB",  "9999999", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.size() == 3);
}

TEST_CASE("from_env optional vars stay nullopt when unset", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env();
  CHECK_FALSE(c.pipeline_pool_size.has_value());
  CHECK_FALSE(c.http_threads.has_value());
  CHECK_FALSE(c.layout_trt.has_value());
}

TEST_CASE("from_env optional vars populate when set", "[server_config]") {
  reset_env();
  ::setenv("PIPELINE_POOL_SIZE", "7",  1);
  ::setenv("HTTP_THREADS",       "64", 1);
  ::setenv("LAYOUT_TRT",         "/opt/layout.trt", 1);
  auto c = ServerConfig::from_env();
  REQUIRE(c.pipeline_pool_size.has_value());
  CHECK(*c.pipeline_pool_size == 7);
  REQUIRE(c.http_threads.has_value());
  CHECK(*c.http_threads == 64);
  REQUIRE(c.layout_trt.has_value());
  CHECK(*c.layout_trt == "/opt/layout.trt");
}

TEST_CASE("from_env strict bool parses many spellings", "[server_config]") {
  reset_env();
  for (const char *t : {"1", "true", "TRUE", "Yes", "on"}) {
    ::setenv("DISABLE_ANGLE_CLS", t, 1);
    auto c = ServerConfig::from_env();
    CHECK(c.disable_angle_cls);
    CHECK(c.errors.empty());
  }
  for (const char *f : {"0", "false", "FALSE", "no", "off"}) {
    ::setenv("DISABLE_ANGLE_CLS", f, 1);
    auto c = ServerConfig::from_env();
    CHECK_FALSE(c.disable_angle_cls);
    CHECK(c.errors.empty());
  }
}

TEST_CASE("from_env rejects malformed bool", "[server_config]") {
  reset_env();
  ::setenv("DISABLE_ANGLE_CLS", "maybe", 1);
  auto c = ServerConfig::from_env();
  REQUIRE_FALSE(c.errors.empty());
  CHECK(c.errors[0].find("DISABLE_ANGLE_CLS") != std::string::npos);
}

TEST_CASE("ENABLE_LAYOUT always errors with a migration message", "[server_config]") {
  // Any value of ENABLE_LAYOUT now fails fast — v2.2.x parsing was buggy
  // (only "1" did what the name said, every other value silently disabled
  // layout), so we force explicit migration rather than try to guess intent.
  for (const char *p : {"1", "0", "true", "false", "yes", "no", "on", "off", "maybe"}) {
    for (auto profile : {Profile::Gpu, Profile::Cpu}) {
      reset_env();
      ::setenv("ENABLE_LAYOUT", p, 1);
      auto c = ServerConfig::from_env(profile);
      REQUIRE_FALSE(c.errors.empty());
      bool found = false;
      for (const auto &e : c.errors)
        if (e.find("ENABLE_LAYOUT") != std::string::npos &&
            e.find("DISABLE_LAYOUT=1") != std::string::npos) found = true;
      CHECK(found);
    }
  }
  // Unset is clean — no error, no warning.
  reset_env();
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  CHECK(c.warnings.empty());
}

TEST_CASE("ENABLE_PDF_MODE validates against allowed set", "[server_config]") {
  reset_env();
  ::setenv("ENABLE_PDF_MODE", "auto", 1);
  auto ok = ServerConfig::from_env();
  CHECK(ok.errors.empty());
  CHECK(ok.default_pdf_mode_was_set);
  CHECK(ok.default_pdf_mode == turbo_ocr::pdf::PdfMode::Auto);

  reset_env();
  ::setenv("ENABLE_PDF_MODE", "bogus", 1);
  auto bad = ServerConfig::from_env();
  REQUIRE_FALSE(bad.errors.empty());
  CHECK(bad.errors[0].find("ENABLE_PDF_MODE") != std::string::npos);
}

TEST_CASE("GRPC_RESPONSE_MODE validates", "[server_config]") {
  reset_env();
  ::setenv("GRPC_RESPONSE_MODE", "structured", 1);
  CHECK(ServerConfig::from_env().grpc_response_mode ==
        turbo_ocr::server::GrpcResponseMode::structured);

  reset_env();
  ::setenv("GRPC_RESPONSE_MODE", "json_typo", 1);
  auto bad = ServerConfig::from_env();
  REQUIRE_FALSE(bad.errors.empty());
}

TEST_CASE("LAYOUT_MERGE_MODE validates", "[server_config]") {
  // Canonical names plus the deprecated "large"/"small"/"union" aliases.
  for (const char *ok :
       {"all", "outer", "inner", "large", "small", "union"}) {
    reset_env();
    ::setenv("LAYOUT_MERGE_MODE", ok, 1);
    CHECK(ServerConfig::from_env().errors.empty());
  }
  reset_env();
  ::setenv("LAYOUT_MERGE_MODE", "outerr", 1);  // typo must be fatal, not silent
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());
}

TEST_CASE("cross-field: PORT == GRPC_PORT is fatal", "[server_config]") {
  reset_env();
  ::setenv("PORT",      "9000", 1);
  ::setenv("GRPC_PORT", "9000", 1);
  auto c = ServerConfig::from_env();
  REQUIRE_FALSE(c.errors.empty());
  bool found = false;
  for (const auto &e : c.errors)
    if (e.find("must differ") != std::string::npos) found = true;
  CHECK(found);
}

TEST_CASE("cross-field: PDF_WORKERS > PDF_DAEMONS is a warning", "[server_config]") {
  reset_env();
  ::setenv("PDF_DAEMONS", "2",  1);
  ::setenv("PDF_WORKERS", "16", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  REQUIRE_FALSE(c.warnings.empty());
}

TEST_CASE("cross-field: MAX_BODY_MEMORY_MB > MAX_BODY_MB clamps + warns",
          "[server_config]") {
  reset_env();
  ::setenv("MAX_BODY_MB",        "50",   1);
  ::setenv("MAX_BODY_MEMORY_MB", "1024", 1);
  auto c = ServerConfig::from_env();
  CHECK(c.errors.empty());
  CHECK(c.max_body_mem_mb == 50);  // clamped to MAX_BODY_MB
  REQUIRE_FALSE(c.warnings.empty());
}

TEST_CASE("from_env strict-validates engine/decode/logging knobs", "[server_config]") {
  reset_env();
  auto def = ServerConfig::from_env();
  // Effective det max-side comes from the selected model's per-model
  // max_side_limit when DET_MAX_SIDE is unset. Every v6 tier caps at 1280 (the
  // official PaddleOCR 4000 OOMs the pooled pre-allocation — det_config.h), so
  // the default is 1280, not 4000; the env knob still wins when set (below).
  CHECK(def.det_max_side == 1280);
  CHECK(def.trt_opt_level == 5);
  CHECK(def.max_image_dim == 16384);
  CHECK(def.log_level == "info");
  CHECK(def.log_format == "json");
  CHECK(def.errors.empty());

  reset_env();
  ::setenv("DET_MAX_SIDE", "abc", 1);
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("DET_MAX_SIDE", "10", 1);  // below min 32
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("TRT_OPT_LEVEL", "9", 1);  // above max 5
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("LOG_LEVEL", "trace", 1);
  auto bad_log = ServerConfig::from_env();
  REQUIRE_FALSE(bad_log.errors.empty());

  reset_env();
  ::setenv("LOG_FORMAT", "xml", 1);
  auto bad_fmt = ServerConfig::from_env();
  REQUIRE_FALSE(bad_fmt.errors.empty());

  reset_env();
  ::setenv("MAX_IMAGE_DIM", "10", 1);  // below min 64
  REQUIRE_FALSE(ServerConfig::from_env().errors.empty());

  reset_env();
  ::setenv("DET_MAX_SIDE",  "2048", 1);
  ::setenv("TRT_OPT_LEVEL", "3",    1);
  ::setenv("MAX_IMAGE_DIM", "8192", 1);
  ::setenv("LOG_LEVEL",     "warn", 1);
  ::setenv("LOG_FORMAT",    "text", 1);
  auto ok = ServerConfig::from_env();
  CHECK(ok.errors.empty());
  CHECK(ok.det_max_side == 2048);
  CHECK(ok.trt_opt_level == 3);
  CHECK(ok.max_image_dim == 8192);
  CHECK(ok.log_level == "warn");
  CHECK(ok.log_format == "text");
}

TEST_CASE("opt_json escapes optional strings with embedded quotes", "[server_config]") {
  // Latent-bug guard: opt_json must escape strings, not just concatenate.
  reset_env();
  auto c = ServerConfig::from_env();
  c.layout_trt = std::string("/path/with\"quote.trt");
  auto j = c.to_json();
  // Should not contain raw unescaped quote inside the string value.
  CHECK(j.find("with\\\"quote") != std::string::npos);
}

TEST_CASE("to_json produces non-empty JSON object", "[server_config]") {
  reset_env();
  auto c = ServerConfig::from_env();
  auto j = c.to_json();
  REQUIRE(j.size() > 2);
  CHECK(j.front() == '{');
  CHECK(j.back() == '}');
  CHECK(j.find("\"host\":\"0.0.0.0\"") != std::string::npos);
  CHECK(j.find("\"http_port\":8080") != std::string::npos);
}

TEST_CASE("exposure-hardening caps: defaults, env override, bounds", "[server_config]") {
  reset_env();
  auto def = ServerConfig::from_env(Profile::Gpu);
  CHECK(def.errors.empty());
  CHECK(def.max_batch_images == 1024);
  CHECK(def.max_pdf_page_pixels == 40000000);  // 40 MP

  reset_env();
  ::setenv("MAX_BATCH_IMAGES", "64", 1);
  ::setenv("MAX_PDF_PAGE_PIXELS_MP", "20", 1);
  auto on = ServerConfig::from_env(Profile::Gpu);
  CHECK(on.errors.empty());
  CHECK(on.max_batch_images == 64);
  CHECK(on.max_pdf_page_pixels == 20000000);
  auto j = on.to_json();
  CHECK(j.find("\"max_batch_images\":64") != std::string::npos);

  reset_env();
  ::setenv("MAX_BATCH_IMAGES", "0", 1);          // below min 1
  ::setenv("MAX_PDF_PAGE_PIXELS_MP", "999", 1);  // above max 268
  auto bad = ServerConfig::from_env(Profile::Gpu);
  REQUIRE_FALSE(bad.errors.empty());
}

TEST_CASE("BIND_HOST aliases the bind address; REQUEST_TIMEOUT_MS validates", "[server_config]") {
  // BIND_HOST is an additive alias for TURBO_OCR_HOST (non-breaking; default
  // stays 0.0.0.0 — auth/exposure are the fronting gateway's job).
  reset_env();
  ::setenv("BIND_HOST", "127.0.0.1", 1);
  CHECK(ServerConfig::from_env(Profile::Gpu).host == "127.0.0.1");

  // TURBO_OCR_HOST wins when both are set (it predates BIND_HOST). A
  // non-default value on BOTH sides, or "wins" would be indistinguishable
  // from "everything ignored, default served".
  reset_env();
  ::setenv("BIND_HOST", "127.0.0.1", 1);
  ::setenv("TURBO_OCR_HOST", "10.1.2.3", 1);
  CHECK(ServerConfig::from_env(Profile::Gpu).host == "10.1.2.3");

  // REQUEST_TIMEOUT_MS validates like the other strict ints (M3).
  reset_env();
  ::setenv("REQUEST_TIMEOUT_MS", "not_a_number", 1);
  CHECK_FALSE(ServerConfig::from_env(Profile::Gpu).errors.empty());
}
