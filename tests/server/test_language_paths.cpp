#include <catch_amalgamated.hpp>

#include <cstdlib>

#include "turbo_ocr/server/language_paths.h"

using turbo_ocr::server::resolve_rec_paths;

namespace {
// Clear all env vars this resolver reads so each case starts clean.
void reset_env() {
  ::unsetenv("OCR_LANG");
  ::unsetenv("OCR_MODEL");
  ::unsetenv("REC_ONNX");
  ::unsetenv("REC_DICT");
  ::unsetenv("REC_MODEL");
  ::unsetenv("TURBO_OCR_REC_ALLOW_TINY");
}
} // namespace

TEST_CASE("resolve_rec_paths falls back to the registry default", "[language_paths]") {
  reset_env();
  auto p = resolve_rec_paths("REC_ONNX");
  // No env -> kDefaultModel ("tiny"): rec_tiny.onnx + its own keys_tiny.txt.
  CHECK(p.rec == "models/rec_tiny.onnx");
  CHECK(p.dict == "models/keys_tiny.txt");
}

TEST_CASE("resolve_rec_paths routes OCR_LANG onto its V5Lang registry entry", "[language_paths]") {
  reset_env();
  ::setenv("OCR_LANG", "korean", 1);
  auto p = resolve_rec_paths("REC_ONNX");
  CHECK(p.rec == "models/rec/korean/rec.onnx");
  CHECK(p.dict == "models/rec/korean/dict.txt");
}

TEST_CASE("resolve_rec_paths collapses OCR_LANG with no entry onto the default",
          "[language_paths]") {
  reset_env();
  // v6 covers chinese, so it has no V5Lang entry and must NOT fabricate a
  // models/rec/chinese/ path (which doesn't exist) — it falls back to default.
  ::setenv("OCR_LANG", "chinese", 1);
  auto p = resolve_rec_paths("REC_ONNX");
  CHECK(p.rec == "models/rec_tiny.onnx");
  CHECK(p.dict == "models/keys_tiny.txt");
}

TEST_CASE("resolve_rec_paths lets REC_ONNX override OCR_LANG", "[language_paths]") {
  reset_env();
  ::setenv("OCR_LANG", "chinese", 1);
  ::setenv("REC_ONNX", "/custom/rec.onnx", 1);
  ::setenv("REC_DICT", "/custom/dict.txt", 1);
  auto p = resolve_rec_paths("REC_ONNX");
  CHECK(p.rec == "/custom/rec.onnx");
  CHECK(p.dict == "/custom/dict.txt");
}

TEST_CASE("resolve_rec_paths treats empty OCR_LANG as unset", "[language_paths]") {
  reset_env();
  ::setenv("OCR_LANG", "", 1);
  auto p = resolve_rec_paths("REC_ONNX");
  CHECK(p.rec == "models/rec_tiny.onnx");
  CHECK(p.dict == "models/keys_tiny.txt");
}

TEST_CASE("resolve_rec_paths treats OCR_LANG=latin as a default sentinel", "[language_paths]") {
  reset_env();
  ::setenv("OCR_LANG", "latin", 1);
  auto p = resolve_rec_paths("REC_ONNX");
  // Latin is served by the default v6 tier; it must not be routed to
  // models/rec/latin/ (which doesn't exist and would crash the TRT builder).
  CHECK(p.rec == "models/rec_tiny.onnx");
  CHECK(p.dict == "models/keys_tiny.txt");
}

TEST_CASE("resolve_rec_paths honors REC_MODEL alias for CPU server", "[language_paths]") {
  reset_env();
  ::setenv("REC_MODEL", "/cpu/rec.onnx", 1);
  auto p = resolve_rec_paths("REC_MODEL");
  CHECK(p.rec == "/cpu/rec.onnx");
}
