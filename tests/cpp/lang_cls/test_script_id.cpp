// Tests for script_id_types.h: the enum whose numeric values are a contract
// with the training recipe, and the wire-name / --rec-mode parsing the
// recognizer selection path depends on.

#include <catch_amalgamated.hpp>

#include "turbo_ocr/analysis/lang_cls/script_id_types.h"

using turbo_ocr::script_id::ScriptId;
using turbo_ocr::script_id::parse_rec_mode;
using turbo_ocr::script_id::parse_script;
using turbo_ocr::script_id::to_string;

TEST_CASE("ScriptId enum values are stable (training-recipe contract)",
          "[script_id]") {
  REQUIRE(static_cast<int>(ScriptId::Latin)   == 0);
  REQUIRE(static_cast<int>(ScriptId::Chinese) == 1);
  REQUIRE(static_cast<int>(ScriptId::Arabic)  == 2);
  REQUIRE(static_cast<int>(ScriptId::ESlav)   == 3);
  REQUIRE(static_cast<int>(ScriptId::Greek)   == 4);
  REQUIRE(static_cast<int>(ScriptId::Korean)  == 5);
  REQUIRE(static_cast<int>(ScriptId::Thai)    == 6);
}

TEST_CASE("ScriptId wire names round-trip", "[script_id]") {
  for (auto s : {ScriptId::Latin, ScriptId::Chinese, ScriptId::Arabic,
                 ScriptId::ESlav, ScriptId::Greek, ScriptId::Korean,
                 ScriptId::Thai}) {
    auto name = to_string(s);
    auto parsed = parse_script(name);
    REQUIRE(parsed.has_value());
    REQUIRE(*parsed == s);
  }
}

TEST_CASE("rec_mode parses single + multi + multilingual specs",
          "[script_id]") {
  REQUIRE(parse_rec_mode("")->count(ScriptId::Latin) == 1);
  REQUIRE(parse_rec_mode("latin")->count(ScriptId::Latin) == 1);
  auto multi = parse_rec_mode("latin,chinese");
  REQUIRE(multi.has_value());
  REQUIRE(multi->size() == 2);
  REQUIRE(multi->count(ScriptId::Chinese) == 1);
  auto all = parse_rec_mode("multilingual");
  REQUIRE(all.has_value());
  REQUIRE(all->size() == 7);
  REQUIRE_FALSE(parse_rec_mode("bogus").has_value());
}

// NOTE: the ScriptIdEngine and ScriptIdResult cases went with the class itself.
// ScriptIdEngine was CUDA-only and instantiated by NOTHING in src/ — these
// tests, which merely default-constructed it and its result struct, were its
// only consumers in the tree. They were left behind referencing the deleted
// type and only surfaced when the GPU arm (which compiles this file) was next
// configured; the CPU arm does not build it.
