#include <catch_amalgamated.hpp>
#include <cstdlib>
#include <string>
#include <vector>

#include "turbo_ocr/server/language_paths.h"
#include "turbo_ocr/server/model_catalog.h"

using turbo_ocr::server::find_model;
using turbo_ocr::server::kDefaultDet;
using turbo_ocr::server::kDefaultModel;
using turbo_ocr::server::ModelFamily;
using turbo_ocr::server::resolve_model;

namespace {
// Clear every env var resolve_model consults so each case starts clean.
void reset_env() {
  ::unsetenv("OCR_MODEL");
  ::unsetenv("OCR_LANG");
  ::unsetenv("REC_ONNX");
  ::unsetenv("REC_MODEL");
  ::unsetenv("REC_DICT");
  ::unsetenv("DET_ONNX");
  ::unsetenv("DET_MODEL");
  ::unsetenv("TURBO_OCR_REC_ALLOW_TINY");
}

// GPU-profile resolution (REC_ONNX/DET_ONNX) with strict error/warning capture.
turbo_ocr::server::ResolvedModel resolve(std::vector<std::string>* errors = nullptr,
                                         std::vector<std::string>* warnings = nullptr) {
  return resolve_model("REC_ONNX", "DET_ONNX", "REC_DICT", errors, warnings);
}
}  // namespace

TEST_CASE("registry resolves the v6 tiny default", "[dict_resolution]") {
  reset_env();
  auto m = resolve();
  CHECK(m.name == std::string(kDefaultModel));  // "tiny" — the throughput default
  CHECK(m.name == "tiny");
  CHECK(m.rec == "models/rec_tiny.onnx");
  // tiny ships its own 6,904-char dict, not the shared 18,708-char keys.txt.
  CHECK(m.dict == "models/keys_tiny.txt");
  // Each v6 tier ships its own detector.
  CHECK(m.det == "models/det_tiny.onnx");
}

TEST_CASE("OCR_MODEL=tiny selects the divergent tiny rec + dict",
          "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "tiny", 1);
  std::vector<std::string> errors;
  auto m = resolve(&errors);
  // tiny is selectable directly — no opt-in, no fatal error.
  CHECK(errors.empty());
  CHECK(m.name == "tiny");
  CHECK(m.rec == "models/rec_tiny.onnx");
  // tiny ships its own 6,904-char dict, NOT the shared 18,708-char keys.txt.
  CHECK(m.dict == "models/keys_tiny.txt");
  // tiny ships its own detector too.
  CHECK(m.det == "models/det_tiny.onnx");
}

TEST_CASE("OCR_MODEL=small reuses the medium dict on its own rec", "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "small", 1);
  auto m = resolve();
  CHECK(m.name == "small");
  CHECK(m.rec == "models/rec_small.onnx");
  CHECK(m.dict == "models/keys.txt");  // medium == small dict
}

TEST_CASE("OCR_MODEL=arabic selects the retained v5 script recognizer", "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "arabic", 1);
  auto m = resolve();
  CHECK(m.name == "arabic");
  CHECK(m.rec == "models/rec/arabic/rec.onnx");
  CHECK(m.dict == "models/rec/arabic/dict.txt");
  // Even a non-Latin script routes detection through the shared v6 detector.
  CHECK(m.det == std::string(kDefaultDet));
  CHECK(find_model("arabic")->family == ModelFamily::V5Lang);
}

TEST_CASE("unknown OCR_MODEL is rejected and falls back to the default", "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "ultra", 1);
  std::vector<std::string> errors;
  auto m = resolve(&errors);
  REQUIRE_FALSE(errors.empty());
  CHECK(errors[0].find("OCR_MODEL") != std::string::npos);
  CHECK(errors[0].find("ultra") != std::string::npos);
  // The valid-names list is surfaced so the operator can self-correct.
  CHECK(errors[0].find("medium") != std::string::npos);
  CHECK(errors[0].find("arabic") != std::string::npos);
  // Resolution still yields the default so non-strict callers stay bootable.
  CHECK(m.name == std::string(kDefaultModel));
  CHECK(m.rec == "models/rec_tiny.onnx");
}

TEST_CASE("OCR_LANG aliases a non-Latin script onto its registry entry", "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_LANG", "korean", 1);
  std::vector<std::string> warnings;
  auto m = resolve(nullptr, &warnings);
  CHECK(m.name == "korean");
  CHECK(m.rec == "models/rec/korean/rec.onnx");
  CHECK(m.dict == "models/rec/korean/dict.txt");
  // OCR_LANG is deprecated; resolving via it must warn but still work.
  REQUIRE_FALSE(warnings.empty());
  CHECK(warnings[0].find("OCR_LANG") != std::string::npos);
  CHECK(warnings[0].find("deprecated") != std::string::npos);
}

TEST_CASE("OCR_LANG with no dedicated entry aliases onto the default", "[dict_resolution]") {
  reset_env();
  // v6 medium covers Latin/Chinese/Japanese, so these have no V5Lang entry
  // and must collapse onto the default model rather than fabricate a path.
  for (const char* lang : {"latin", "chinese", "japanese", "en"}) {
    ::setenv("OCR_LANG", lang, 1);
    auto m = resolve();
    CHECK(m.name == std::string(kDefaultModel));
    CHECK(m.rec == "models/rec_tiny.onnx");
    CHECK(m.dict == "models/keys_tiny.txt");
  }
}

TEST_CASE("OCR_MODEL takes precedence over a conflicting OCR_LANG", "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "small", 1);
  ::setenv("OCR_LANG", "arabic", 1);
  auto m = resolve();
  CHECK(m.name == "small");
  CHECK(m.rec == "models/rec_small.onnx");
}

TEST_CASE("explicit REC_ONNX override bypasses the registry", "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "arabic", 1);
  ::setenv("REC_ONNX", "/custom/rec.onnx", 1);
  ::setenv("REC_DICT", "/custom/dict.txt", 1);
  auto m = resolve();
  // The override wins per-stage; the registry name still reflects the entry
  // so logging/telemetry stays truthful, but rec+dict are the overrides.
  CHECK(m.name == "arabic");
  CHECK(m.rec == "/custom/rec.onnx");
  CHECK(m.dict == "/custom/dict.txt");
}

TEST_CASE("a DET_ONNX override leaves rec+dict on the selected model", "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "small", 1);
  ::setenv("DET_ONNX", "/custom/det.onnx", 1);
  auto m = resolve();
  CHECK(m.det == "/custom/det.onnx");
  CHECK(m.rec == "models/rec_small.onnx");
  CHECK(m.dict == "models/keys.txt");
}

TEST_CASE("per-model det_cfg: tiny diverges in box_thresh; medium/small match",
          "[dict_resolution]") {
  // The detection box_thresh is part of each tier's official config: tiny uses
  // 0.40 (its inference.yml), medium/small use 0.45.
  CHECK(find_model("tiny")->det_cfg.db.box_thresh == 0.40f);
  CHECK(find_model("small")->det_cfg.db.box_thresh == 0.45f);
  CHECK(find_model("medium")->det_cfg.db.box_thresh == 0.45f);
}

TEST_CASE("a DET_ONNX override drops the per-model det_cfg to the official base",
          "[dict_resolution]") {
  reset_env();
  ::setenv("OCR_MODEL", "tiny", 1);
  ::setenv("DET_ONNX", "/custom/det.onnx", 1);
  auto m = resolve();
  // rec stays tiny, but with a detector supplied outside the registry the
  // per-model det_cfg no longer applies — it reverts to the official defaults
  // (box_thresh 0.45), NOT tiny's 0.40.
  CHECK(m.rec == "models/rec_tiny.onnx");
  CHECK(m.det == "/custom/det.onnx");
  CHECK(m.det_cfg.db.box_thresh == 0.45f);
}
