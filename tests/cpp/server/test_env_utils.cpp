// ENV_UTILS EDGE SEMANTICS — the exact parsing contracts of env_or/env_enabled/
// env_int, plus env::observed(). Production behaviour depends on the PRECISE
// edges here, not the happy path: an operator writing `VAR=` in a compose file
// expects "not configured", not "configured to the empty string", and a knob
// that clamps into range must not quietly fall back to a default instead —
// those are different failure stories to debug from.
//
// tests/cpp/common/test_env_utils.cpp already covers the *_strict family; this
// file is the lenient trio (env_or/env_enabled/env_int) plus the one thing
// nothing else exercises: the observed() inventory that env_utils.h exists to
// build (see the file-header comment on env_utils.h — 80 ungoverned getenv call
// sites was the whole motivation for this header).

#include <catch_amalgamated.hpp>

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/base/env_utils.h"

using namespace turbo_ocr::env;

namespace {

// RAII env var: set on construction, restore prior state on destruction so a
// CHECK failure mid-case can't leak a value into a later, unrelated test.
class ScopedEnv {
public:
  ScopedEnv(const char *name, const char *value) : name_(name) {
    if (const char *prev = std::getenv(name)) {
      had_prev_ = true;
      prev_ = prev;
    }
    if (value)
      setenv(name, value, 1);
    else
      unsetenv(name);
  }
  ~ScopedEnv() {
    if (had_prev_)
      setenv(name_.c_str(), prev_.c_str(), 1);
    else
      unsetenv(name_.c_str());
  }
  ScopedEnv(const ScopedEnv &) = delete;
  ScopedEnv &operator=(const ScopedEnv &) = delete;

private:
  std::string name_;
  std::string prev_;
  bool had_prev_ = false;
};

// Distinct from tests/cpp/common/test_env_utils.cpp's kVar so the two suites'
// reads never collide inside the process-wide observed() log (see the
// dedicated observed() test below, which depends on a name nothing else in
// the binary has ever touched).
constexpr const char *kVar = "TURBO_OCR_SERVER_TEST_ENV_VAR";

} // namespace

TEST_CASE("env_or: unset -> default, SET-BUT-EMPTY -> default, set -> value",
          "[env_utils][server]") {
  {
    ScopedEnv e(kVar, nullptr);
    CHECK(env_or(kVar, "def") == "def");
  }
  {
    // The deliberate one: `VAR=` in a compose file means "not configured",
    // not "configured to the empty string" — env_or must not hand back "".
    ScopedEnv e(kVar, "");
    CHECK(env_or(kVar, "def") == "def");
  }
  {
    ScopedEnv e(kVar, "configured");
    CHECK(env_or(kVar, "def") == "configured");
  }
}

TEST_CASE("env_enabled is true for exactly \"1\", nothing else",
          "[env_utils][server]") {
  {
    ScopedEnv e(kVar, "1");
    CHECK(env_enabled(kVar));
  }
  // Every one of these reads as "on" to a careless strcmp/atoi and must not
  // here: "" is unset-shaped, "0"/"false"/"yes" are other truthy/falsy
  // spellings a caller might expect, "01" is "1" with an extra character that
  // a prefix check would wrongly accept.
  for (const char *v : {"", "0", "true", "yes", "01"}) {
    ScopedEnv e(kVar, v);
    INFO("value=\"" << v << "\"");
    CHECK_FALSE(env_enabled(kVar));
  }
}

TEST_CASE("env_int: garbage falls back to default, out-of-range CLAMPS",
          "[env_utils][server]") {
  {
    ScopedEnv e(kVar, nullptr);
    CHECK(env_int(kVar, 5, 1, 10) == 5);
  }
  {
    ScopedEnv e(kVar, "not-a-number");
    CHECK(env_int(kVar, 5, 1, 10) == 5); // garbage -> default, not 0
  }
  {
    ScopedEnv e(kVar, "7trailing");
    CHECK(env_int(kVar, 5, 1, 10) == 5); // trailing junk is still garbage
  }
  {
    ScopedEnv e(kVar, "6");
    CHECK(env_int(kVar, 5, 1, 10) == 6); // valid -> parsed, not the default
  }
  // Out-of-range is NOT garbage: it clamps to the boundary, it does not fall
  // back to `def`. A caller reading env_int(name, 5, 1, 10) with VAR=1000
  // must get 10 (the ceiling the value overshot), never 5 (the unrelated
  // default) and never 1000 (unclamped passthrough).
  {
    ScopedEnv e(kVar, "1000");
    CHECK(env_int(kVar, 5, 1, 10) == 10);
  }
  {
    ScopedEnv e(kVar, "-1000");
    CHECK(env_int(kVar, 5, 1, 10) == 1);
  }
  {
    ScopedEnv e(kVar, "0");
    CHECK(env_int(kVar, 5, 1, 10) == 1); // 0 is below min_val=1 -> clamps up
  }
}

TEST_CASE("env::observed() logs a knob once, keyed by the FIRST value read",
          "[env_utils][server]") {
  // A name nothing else in this binary reads, so the inventory's count for it
  // starts at zero regardless of test execution order.
  constexpr const char *kKnob = "TURBO_OCR_SERVER_ENV_UTILS_OBSERVED_KNOB";

  auto find_it = [](const std::vector<std::pair<std::string, std::string>> &v,
                    const std::string &name) {
    int count = 0;
    std::string value;
    for (const auto &kv : v)
      if (kv.first == name) {
        ++count;
        value = kv.second;
      }
    return std::make_pair(count, value);
  };

  {
    const auto [count, value] = find_it(observed(), kKnob);
    (void)value;
    CHECK(count == 0);
  }

  {
    ScopedEnv e(kKnob, "first");
    CHECK(env_or(kKnob, "def") == "first");
  }
  {
    const auto [count, value] = find_it(observed(), kKnob);
    REQUIRE(count == 1); // one read through env_or -> one inventory entry
    CHECK(value == "first");
  }

  // Reading the SAME name again, now with a DIFFERENT live value, must not
  // duplicate the entry (a knob read from ten call sites would otherwise
  // flood the inventory with copies of itself) and must not overwrite it
  // either — detail::record() documents "first read wins; later reads are
  // the same" precisely so the inventory reflects what governed the FIRST
  // observation, not whatever happened to be set last when someone called
  // observed(). This is also the property that makes duplicate reads of one
  // name across 39 files safe to leave ungoverned.
  {
    ScopedEnv e(kKnob, "second");
    CHECK(env_or(kKnob, "def") == "second"); // the live read sees the change...
  }
  {
    const auto [count, value] = find_it(observed(), kKnob);
    CHECK(count == 1);        // ...the inventory still has exactly one entry...
    CHECK(value == "first");  // ...holding the value first observed.
  }

  // A different helper (env_int, not env_or) reading a fresh name must feed
  // the SAME inventory -- observed() is not env_or-specific bookkeeping, it is
  // the one log every parser in this header writes through.
  constexpr const char *kIntKnob = "TURBO_OCR_SERVER_ENV_UTILS_OBSERVED_INT_KNOB";
  {
    ScopedEnv e(kIntKnob, "42");
    CHECK(env_int(kIntKnob, 0, 0, 100) == 42);
  }
  {
    const auto [count, value] = find_it(observed(), kIntKnob);
    CHECK(count == 1);
    CHECK(value == "42");
  }
}
