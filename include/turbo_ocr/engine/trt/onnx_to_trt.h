#pragma once

#include <filesystem>
#include <string>

namespace turbo_ocr::engine {

/// Get the cache directory for TRT engines.
/// Docker: /tmp/turbo-ocr-engines/
/// Native: ~/.cache/turbo-ocr/ (or /tmp/ as fallback)
[[nodiscard]] std::string get_engine_cache_dir();

/// Get the cached TRT engine path for an ONNX model.
/// Uses hash of: ONNX file size + mtime + TRT version → unique filename.
/// Returns: {cache_dir}/{type}_{hash}.trt
[[nodiscard]] std::string get_cached_engine_path(const std::string &onnx_path,
                                                  const std::string &type);

/// Ensure a TRT engine exists in cache for the given ONNX model.
/// Builds if missing. Returns the path to the cached .trt file.
/// type: "det", "rec", or "cls" — controls optimization profile shapes.
[[nodiscard]] std::string ensure_trt_engine(const std::string &onnx_path,
                                            const std::string &type);

/// Remove orphaned `*.trt.tmp.*` files left behind by a previous process
/// that crashed between fs::write and fs::rename. Skips files modified
/// in the last `min_age_seconds` so an in-progress build by a sibling
/// replica isn't deleted out from under it. Safe to call from any
/// process; logs a one-line summary.
void sweep_orphan_engine_temps(int min_age_seconds = 60);

} // namespace turbo_ocr::engine
