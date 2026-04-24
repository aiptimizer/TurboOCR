#pragma once

#include <cstdlib>
#include <string>

#include "turbo_ocr/server/env_utils.h"

namespace turbo_ocr::server {

struct RecPaths {
  std::string rec;
  std::string dict;
};

// Resolve the recognition model + character dictionary paths.
//
// Selection order (highest precedence first):
//   1. `rec_env` / `dict_env` — per-variable env overrides, always win.
//   2. `OCR_LANG=<lang>` (where <lang> is anything but empty or "latin") —
//      nested layout: models/rec/<lang>/{rec.onnx,dict.txt}.
//   3. Otherwise — flat layout: models/rec.onnx + models/keys.txt.
//      This covers OCR_LANG unset, OCR_LANG="", and the explicit sentinel
//      OCR_LANG="latin".
//
// `rec_env` is "REC_ONNX" for GPU (TensorRT) code paths and "REC_MODEL" for
// the CPU/ONNX-Runtime server. `dict_env` is always "REC_DICT".
//
// Detection and angle classification are language-agnostic and keep their
// own shared env vars at the call site.
[[nodiscard]] inline RecPaths resolve_rec_paths(const char *rec_env,
                                                const char *dict_env = "REC_DICT") {
  const char *lang_c = std::getenv("OCR_LANG");
  std::string lang = (lang_c && *lang_c) ? std::string(lang_c) : std::string{};

  std::string rec_default = "models/rec.onnx";
  std::string dict_default = "models/keys.txt";
  if (!lang.empty() && lang != "latin") {
    rec_default = "models/rec/" + lang + "/rec.onnx";
    dict_default = "models/rec/" + lang + "/dict.txt";
  }

  return RecPaths{
      .rec  = env_or(rec_env, rec_default),
      .dict = env_or(dict_env, dict_default),
  };
}

[[nodiscard]] inline std::string ocr_lang() {
  const char *v = std::getenv("OCR_LANG");
  return (v && *v) ? std::string(v) : std::string{};
}

} // namespace turbo_ocr::server
