#pragma once

#include <filesystem>
#include <string>

#include "turbo_ocr/base/log/logger.h" // TOCR_LOG_ERROR

// SLANeXt model-path derivation shared by the GPU (slanext_table_recognizer)
// and CPU (ort_slanext_table) backends: the decoder blob and structure dict
// default to files sitting next to the encoder ONNX, overridable via
// TABLE_SLANEXT_DECODER_BIN / TABLE_SLANEXT_DICT (env::env_or at the sites).
namespace turbo_ocr::table {

// Resolve the encoder ONNX for TABLE_BACKEND=slanext: the env override wins,
// otherwise the path the release bundle / Docker image ships. Returns "" (and
// prints why) when neither exists — the caller fails its load.
//
// SHARED on purpose. This policy briefly lived only in the TRT loader, so
// `TABLE_BACKEND=slanext` alone booted the GPU server but made the CPU/unified
// server abort with "table backend failed to load" even though the shipped
// encoder was on disk. Path policy is generic; keep it in ONE place.
[[nodiscard]] inline std::string
resolve_slanext_encoder(const std::string &env_value) {
  static constexpr const char *kDefaultEncoder =
      "models/table/slanext_encoder/SLANeXt_wired_encoder.onnx";
  if (!env_value.empty()) return env_value;
  if (std::filesystem::exists(kDefaultEncoder)) return kDefaultEncoder;
  // TOCR_LOG, not std::cerr: this is a boot-time failure an operator has to see
  // in the same structured stream as everything else.
  TOCR_LOG_ERROR("table encoder not found: set TABLE_SLANEXT_ENCODER_ONNX",
                 "expected_path", kDefaultEncoder);
  return {};
}

// Derive the decoder.bin path that sits next to the encoder ONNX.
[[nodiscard]] inline std::string
slanext_default_decoder_bin(const std::string &enc) {
  const std::string suf = "_encoder.onnx";
  std::string d = enc;
  auto p = d.rfind(suf);
  if (p != std::string::npos) d.replace(p, suf.size(), "_decoder.bin");
  return d;
}

[[nodiscard]] inline std::string
slanext_default_dict(const std::string &enc) {
  auto slash = enc.find_last_of('/');
  std::string dir =
      (slash == std::string::npos) ? std::string() : enc.substr(0, slash + 1);
  return dir + "SLANeXt_dict_infer.txt";
}

} // namespace turbo_ocr::table
