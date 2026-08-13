#pragma once

// TEST-ONLY POSIX shims for MSVC. Force-included into turbo_ocr_tests via /FI
// (see CMakeLists.txt) so the nine test files that call setenv/unsetenv need no
// #ifdef of their own — the alternative was editing each one, which buries the
// platform difference in nine places instead of naming it in one.
//
// Deliberately NOT in include/: production code reads the environment through
// turbo_ocr::env (base/env_utils.h) and never writes it. Only tests write, to
// set up a case. If production ever needs to WRITE an env var, it should get a
// real cross-platform helper, not this.

#if defined(_WIN32)

#include <cstdlib>
#include <string>

// POSIX setenv returns 0 on success. _putenv_s returns errno_t (0 on success),
// so the return values already agree. `overwrite == 0` means "leave an existing
// value alone", which _putenv_s has no equivalent for — hence the explicit
// lookup. getenv is fine here: single-threaded test setup, before any worker.
inline int setenv(const char *name, const char *value, int overwrite) {
  if (!name || !value) return -1;
  if (!overwrite) {
    std::size_t len = 0;
    if (getenv_s(&len, nullptr, 0, name) == 0 && len > 0) return 0; // already set
  }
  return _putenv_s(name, value) == 0 ? 0 : -1;
}

// POSIX unsetenv removes the variable. On Windows, assigning an EMPTY value is
// the documented way to delete it — _putenv_s(name, "") unsets rather than
// setting it to "". That difference matters: a test that unsets and then reads
// expects "not present", not "present but empty".
inline int unsetenv(const char *name) {
  if (!name) return -1;
  return _putenv_s(name, "") == 0 ? 0 : -1;
}

#endif // _WIN32
