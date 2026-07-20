// CPU-side smoke tests for the ScriptIdEngine public surface. Live GPU
// classification is exercised by the server warmup + integration runs;
// here we lock the result-struct defaults, the buffer-size budget, and the
// ScriptId<->wire-name round trip the dispatcher relies on.

#include <catch_amalgamated.hpp>

#include "turbo_ocr/lang_cls/script_id.h"
#include "turbo_ocr/lang_cls/script_id_types.h"

using turbo_ocr::lang_cls::ScriptIdEngine;
using turbo_ocr::lang_cls::ScriptIdResult;
using turbo_ocr::script_id::ScriptId;
using turbo_ocr::script_id::parse_rec_mode;
using turbo_ocr::script_id::parse_script;
using turbo_ocr::script_id::to_string;

TEST_CASE("ScriptIdResult defaults to Latin with zero confidence",
          "[script_id]") {
  ScriptIdResult r{};
  REQUIRE(r.script == ScriptId::Latin);
  REQUIRE(r.confidence == 0.0f);
}

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

TEST_CASE("ScriptIdEngine is default-constructible without GPU",
          "[script_id]") {
  ScriptIdEngine e;
  (void)e;  // no GPU resources touched until load_model() / allocate_buffers()
}
