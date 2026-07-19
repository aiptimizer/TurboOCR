#include <catch_amalgamated.hpp>

#include <cstdlib>
#include <string>
#include <vector>

#include "turbo_ocr/common/env_utils.h"

using namespace turbo_ocr::env;

namespace {

// RAII env var: set on construction, restore prior state on destruction so
// tests can't leak state into each other regardless of CHECK failures.
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

constexpr const char *kVar = "TURBO_OCR_TEST_ENV_VAR";

} // namespace

TEST_CASE("env_or returns value, default on unset, default on empty", "[env_utils]") {
  {
    ScopedEnv e(kVar, "hello");
    CHECK(env_or(kVar, "def") == "hello");
  }
  {
    ScopedEnv e(kVar, nullptr);
    CHECK(env_or(kVar, "def") == "def");
  }
  {
    // Set-but-empty means "not configured" — same contract as every parser.
    ScopedEnv e(kVar, "");
    CHECK(env_or(kVar, "def") == "def");
  }
}

TEST_CASE("env_enabled only accepts exactly \"1\"", "[env_utils]") {
  for (const char *v : {"1"}) {
    ScopedEnv e(kVar, v);
    CHECK(env_enabled(kVar));
  }
  for (const char *v : {"0", "true", "yes", "11", ""}) {
    ScopedEnv e(kVar, v);
    CHECK_FALSE(env_enabled(kVar));
  }
}

TEST_CASE("env_int: default, parse, clamp, garbage", "[env_utils]") {
  {
    ScopedEnv e(kVar, nullptr);
    CHECK(env_int(kVar, 7, 1, 100) == 7);
  }
  {
    ScopedEnv e(kVar, "42");
    CHECK(env_int(kVar, 7, 1, 100) == 42);
  }
  {
    ScopedEnv e(kVar, "0");
    CHECK(env_int(kVar, 7, 1, 100) == 1); // clamped up
  }
  {
    ScopedEnv e(kVar, "1000");
    CHECK(env_int(kVar, 7, 1, 100) == 100); // clamped down
  }
  {
    ScopedEnv e(kVar, "12abc");
    CHECK(env_int(kVar, 7, 1, 100) == 7); // trailing garbage -> default
  }
  {
    ScopedEnv e(kVar, "abc");
    CHECK(env_int(kVar, 7, 1, 100) == 7);
  }
}

TEST_CASE("env_int_strict pushes errors instead of guessing", "[env_utils]") {
  std::vector<std::string> errors;
  {
    ScopedEnv e(kVar, "55");
    CHECK(env_int_strict(kVar, 7, 1, 100, errors) == 55);
    CHECK(errors.empty());
  }
  {
    ScopedEnv e(kVar, "abc");
    CHECK(env_int_strict(kVar, 7, 1, 100, errors) == 7);
    REQUIRE(errors.size() == 1);
    CHECK(errors[0].find(kVar) != std::string::npos);
  }
  errors.clear();
  {
    ScopedEnv e(kVar, "101");
    CHECK(env_int_strict(kVar, 7, 1, 100, errors) == 7);
    REQUIRE(errors.size() == 1);
  }
  errors.clear();
  {
    // Overflow beyond long long must be an error, not a wrapped value.
    ScopedEnv e(kVar, "99999999999999999999999999");
    CHECK(env_int_strict(kVar, 7, 1, 100, errors) == 7);
    CHECK(errors.size() == 1);
  }
}

TEST_CASE("env_bool_strict accepts the documented spellings only", "[env_utils]") {
  std::vector<std::string> errors;
  for (const char *v : {"1", "true", "YES", "On"}) {
    ScopedEnv e(kVar, v);
    CHECK(env_bool_strict(kVar, false, errors));
  }
  for (const char *v : {"0", "false", "no", "OFF"}) {
    ScopedEnv e(kVar, v);
    CHECK_FALSE(env_bool_strict(kVar, true, errors));
  }
  CHECK(errors.empty());
  {
    ScopedEnv e(kVar, "maybe");
    CHECK(env_bool_strict(kVar, true, errors) == true); // default kept
    CHECK(errors.size() == 1);
  }
}

TEST_CASE("env_float_strict rejects NaN and range violations", "[env_utils]") {
  std::vector<std::string> errors;
  {
    ScopedEnv e(kVar, "0.25");
    CHECK(env_float_strict(kVar, 1.0f, 0.0f, 1.0f, errors) == Catch::Approx(0.25f));
    CHECK(errors.empty());
  }
  {
    ScopedEnv e(kVar, "nan");
    CHECK(env_float_strict(kVar, 1.0f, 0.0f, 1.0f, errors) == 1.0f);
    CHECK(errors.size() == 1);
  }
  errors.clear();
  {
    ScopedEnv e(kVar, "2.5");
    CHECK(env_float_strict(kVar, 1.0f, 0.0f, 1.0f, errors) == 1.0f);
    CHECK(errors.size() == 1);
  }
}

TEST_CASE("env_choice_strict enforces the choice list", "[env_utils]") {
  std::vector<std::string> errors;
  {
    ScopedEnv e(kVar, "b");
    CHECK(env_choice_strict(kVar, "a", {"a", "b"}, errors) == "b");
    CHECK(errors.empty());
  }
  {
    ScopedEnv e(kVar, "z");
    CHECK(env_choice_strict(kVar, "a", {"a", "b"}, errors) == "a");
    REQUIRE(errors.size() == 1);
    CHECK(errors[0].find("must be one of") != std::string::npos);
  }
}
