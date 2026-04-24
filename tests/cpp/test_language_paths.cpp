#include <catch_amalgamated.hpp>

#include <cstdlib>

#include "turbo_ocr/server/language_paths.h"

using turbo_ocr::server::resolve_rec_paths;

namespace {
// Clear all env vars this resolver reads so each case starts clean.
void reset_env() {
  ::unsetenv("OCR_LANG");
  ::unsetenv("REC_ONNX");
  ::unsetenv("REC_DICT");
  ::unsetenv("REC_MODEL");
}
} // namespace

TEST_CASE("resolve_rec_paths falls back to legacy flat layout", "[language_paths]") {
  reset_env();
  auto p = resolve_rec_paths("REC_ONNX");
  CHECK(p.rec == "models/rec.onnx");
  CHECK(p.dict == "models/keys.txt");
}

TEST_CASE("resolve_rec_paths honors OCR_LANG nested layout", "[language_paths]") {
  reset_env();
  ::setenv("OCR_LANG", "chinese", 1);
  auto p = resolve_rec_paths("REC_ONNX");
  CHECK(p.rec == "models/rec/chinese/rec.onnx");
  CHECK(p.dict == "models/rec/chinese/dict.txt");
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
  CHECK(p.rec == "models/rec.onnx");
  CHECK(p.dict == "models/keys.txt");
}

TEST_CASE("resolve_rec_paths treats OCR_LANG=latin as flat-layout sentinel", "[language_paths]") {
  reset_env();
  ::setenv("OCR_LANG", "latin", 1);
  auto p = resolve_rec_paths("REC_ONNX");
  // Latin ships in the flat layout; must not be routed to models/rec/latin/
  // (which doesn't exist in-image and would crash the TRT builder).
  CHECK(p.rec == "models/rec.onnx");
  CHECK(p.dict == "models/keys.txt");
}

TEST_CASE("resolve_rec_paths honors REC_MODEL alias for CPU server", "[language_paths]") {
  reset_env();
  ::setenv("REC_MODEL", "/cpu/rec.onnx", 1);
  auto p = resolve_rec_paths("REC_MODEL");
  CHECK(p.rec == "/cpu/rec.onnx");
}
