#include <catch_amalgamated.hpp>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Unique temp-file suffix, so two concurrent ctest runs don't collide. POSIX
// spells it getpid(); Windows spells it GetCurrentProcessId() and has no
// <unistd.h>. Wrapped rather than #ifdef'd at the call site — there is one
// caller and it only wants "a number unique to this process".
#if defined(_WIN32)
#include <process.h>
static inline long turbo_test_pid() { return static_cast<long>(_getpid()); }
#else
#include <unistd.h>
static inline long turbo_test_pid() { return static_cast<long>(::getpid()); }
#endif

#include "turbo_ocr/analysis/formula/ppformulanet/formula_tokenizer.h"

using turbo_ocr::formula::FormulaTokenizer;
namespace fs = std::filesystem;

namespace {

// RAII temp tokenizer file.
class TempJson {
public:
  explicit TempJson(const std::string &content) {
    path_ = fs::temp_directory_path() /
            ("tokenizer_test_" + std::to_string(turbo_test_pid()) + "_" +
             std::to_string(counter_++) + ".json");
    std::ofstream(path_) << content;
  }
  ~TempJson() { std::error_code ec; fs::remove(path_, ec); }
  [[nodiscard]] std::string path() const { return path_.string(); }

private:
  static inline int counter_ = 0;
  fs::path path_;
};

} // namespace

TEST_CASE("tokenizer loads a well-formed vocab and decodes", "[formula_tokenizer]") {
  TempJson f(R"({
    "model": {"vocab": {"x": 0, "y": 1, "+": 2}},
    "added_tokens": [
      {"id": 3, "content": "</s>", "special": true}
    ]
  })");
  auto tok = FormulaTokenizer::load(f.path());
  REQUIRE(tok.has_value());
  // Sequence decodes up to EOS; special tokens are skipped.
  const std::vector<int64_t> ids{0, 2, 1, 3, 0};
  const std::string out = tok->decode(ids, /*post_process=*/false);
  CHECK(out == "x+y");
}

TEST_CASE("tokenizer rejects a hostile huge token id instead of allocating",
          "[formula_tokenizer]") {
  TempJson f(R"({
    "model": {"vocab": {"x": 4294967295}}
  })");
  CHECK_FALSE(FormulaTokenizer::load(f.path()).has_value());
}

TEST_CASE("tokenizer rejects a hostile huge added_tokens id", "[formula_tokenizer]") {
  TempJson f(R"({
    "model": {"vocab": {"x": 0}},
    "added_tokens": [{"id": 4294967295, "content": "boom"}]
  })");
  CHECK_FALSE(FormulaTokenizer::load(f.path()).has_value());
}

TEST_CASE("tokenizer survives malformed shapes without throwing",
          "[formula_tokenizer]") {
  // Non-numeric id: a type-mismatch throw inside nlohmann must resolve to
  // nullopt, never escape the loader.
  TempJson bad_id(R"({
    "model": {"vocab": {"x": "zero"}}
  })");
  CHECK_FALSE(FormulaTokenizer::load(bad_id.path()).has_value());

  TempJson no_vocab(R"({"model": {}})");
  CHECK_FALSE(FormulaTokenizer::load(no_vocab.path()).has_value());

  TempJson vocab_not_object(R"({"model": {"vocab": [1, 2]}})");
  CHECK_FALSE(FormulaTokenizer::load(vocab_not_object.path()).has_value());

  TempJson not_json("this is not json at all {");
  CHECK_FALSE(FormulaTokenizer::load(not_json.path()).has_value());
}

TEST_CASE("tokenizer added_tokens without id are skipped, not fatal",
          "[formula_tokenizer]") {
  TempJson f(R"({
    "model": {"vocab": {"x": 0}},
    "added_tokens": [{"content": "orphan"}, {"id": 1, "content": "</s>"}]
  })");
  auto tok = FormulaTokenizer::load(f.path());
  REQUIRE(tok.has_value());
  const std::vector<int64_t> ids{0};
  CHECK(tok->decode(ids, false) == "x");
}
