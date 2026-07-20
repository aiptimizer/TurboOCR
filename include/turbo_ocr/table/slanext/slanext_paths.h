#pragma once

#include <string>

// SLANeXt model-path derivation shared by the GPU (slanext_table_recognizer)
// and CPU (cpu_slanext_table) backends: the decoder blob and structure dict
// default to files sitting next to the encoder ONNX, overridable via
// TABLE_SLANEXT_DECODER_BIN / TABLE_SLANEXT_DICT (env::env_or at the sites).
namespace turbo_ocr::table {

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
