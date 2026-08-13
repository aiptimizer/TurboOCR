#pragma once

// ONNX Runtime takes model paths as ORTCHAR_T*, which is `char` on POSIX and
// `wchar_t` on Windows — so the same Ort::Session(env, path.c_str(), opts) that
// compiles on Linux fails to resolve an overload on MSVC. Every session
// construction in this tree goes through ort_path() so the difference lives
// here and not at six call sites.
//
// UTF-8 in, UTF-16 out on Windows: model paths come from config and the CLI and
// may contain non-ASCII (a user directory with an umlaut is the common case),
// and CP_UTF8 is the only conversion that round-trips them. Narrowing through
// the system codepage would silently mangle such a path into "file not found".

#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace turbo_ocr::onnx {

#if defined(_WIN32)

using OrtPathString = std::wstring;

[[nodiscard]] inline OrtPathString ort_path(const std::string &utf8) {
  if (utf8.empty()) return {};
  const int need = ::MultiByteToWideChar(
      CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
  if (need <= 0) return {};
  OrtPathString out(static_cast<std::size_t>(need), L'\0');
  ::MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
                        out.data(), need);
  return out;
}

#else

using OrtPathString = std::string;

// By value, not by reference: the call sites write ort_path(p).c_str(), and
// returning a reference to the argument would be fine here but becomes a
// dangling read the moment a caller passes a temporary. One copy of a path
// string, once per session load, is not a cost worth that hazard.
[[nodiscard]] inline OrtPathString ort_path(const std::string &p) { return p; }

#endif

} // namespace turbo_ocr::onnx
