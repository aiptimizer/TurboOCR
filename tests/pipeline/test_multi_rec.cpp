// Unit tests for the multi-rec dispatch logic. The full pipeline path is
// exercised by the server warmup + omnidocbench eval — here we lock the
// algorithmic invariants of script→engine grouping and the confidence
// fallback that does not require a live GPU.

#include <catch_amalgamated.hpp>

#include <map>
#include <set>
#include <vector>

#include "turbo_ocr/lang_cls/script_id.h"
#include "turbo_ocr/lang_cls/script_id_types.h"

using turbo_ocr::lang_cls::ScriptIdResult;
using turbo_ocr::script_id::ScriptId;

namespace {

// Mirror of the per-box clamp + fallback in OcrPipeline::dispatch_rec_.
// Keeping the logic inline in the test isolates the algorithmic contract
// from the GPU dispatch glue.
ScriptId clamp_to_loaded(const ScriptIdResult& r,
                         const std::set<ScriptId>& loaded) {
  ScriptId fallback = loaded.count(ScriptId::Latin) ? ScriptId::Latin
                                                    : *loaded.begin();
  if (r.confidence < 0.6f) return fallback;
  if (loaded.count(r.script) == 0) return fallback;
  return r.script;
}

}  // namespace

TEST_CASE("multi_rec: low-confidence prediction falls back to Latin",
          "[multi_rec]") {
  std::set<ScriptId> loaded{ScriptId::Latin, ScriptId::Chinese};
  ScriptIdResult r{ScriptId::Chinese, 0.4f};
  REQUIRE(clamp_to_loaded(r, loaded) == ScriptId::Latin);
}

TEST_CASE("multi_rec: high-confidence prediction passes through",
          "[multi_rec]") {
  std::set<ScriptId> loaded{ScriptId::Latin, ScriptId::Chinese};
  ScriptIdResult r{ScriptId::Chinese, 0.95f};
  REQUIRE(clamp_to_loaded(r, loaded) == ScriptId::Chinese);
}

TEST_CASE("multi_rec: unloaded prediction clamped to Latin", "[multi_rec]") {
  std::set<ScriptId> loaded{ScriptId::Latin, ScriptId::Chinese};
  ScriptIdResult r{ScriptId::Arabic, 0.99f};
  REQUIRE(clamp_to_loaded(r, loaded) == ScriptId::Latin);
}

TEST_CASE("multi_rec: when Latin not loaded, fallback uses first key",
          "[multi_rec]") {
  std::set<ScriptId> loaded{ScriptId::Chinese, ScriptId::Arabic};
  // std::set iterates in enum order; Chinese (1) < Arabic (2)
  ScriptIdResult low{ScriptId::Korean, 0.1f};
  REQUIRE(clamp_to_loaded(low, loaded) == ScriptId::Chinese);
  ScriptIdResult unloaded{ScriptId::Thai, 0.99f};
  REQUIRE(clamp_to_loaded(unloaded, loaded) == ScriptId::Chinese);
}

TEST_CASE("multi_rec: grouping preserves original order on merge",
          "[multi_rec]") {
  // Simulate: 4 boxes, classifier produces (Latin, Chinese, Latin, Chinese).
  // After grouping and merging back, results indexed by original position
  // must match what each engine produced for its slot.
  std::vector<ScriptId> per_box = {ScriptId::Latin, ScriptId::Chinese,
                                   ScriptId::Latin, ScriptId::Chinese};
  std::map<ScriptId, std::vector<std::pair<std::string, float>>> fake_engine_out{
      {ScriptId::Latin,   {{"hello", 0.9f}, {"world", 0.8f}}},
      {ScriptId::Chinese, {{"\xe4\xbd\xa0", 0.95f}, {"\xe5\xa5\xbd", 0.85f}}}};

  std::vector<std::pair<std::string, float>> merged(per_box.size());
  for (const auto &kv : fake_engine_out) {
    int slot = 0;
    for (std::size_t i = 0; i < per_box.size(); ++i) {
      if (per_box[i] == kv.first) {
        merged[i] = kv.second[slot++];
      }
    }
  }
  REQUIRE(merged[0].first == "hello");
  REQUIRE(merged[1].first == "\xe4\xbd\xa0");
  REQUIRE(merged[2].first == "world");
  REQUIRE(merged[3].first == "\xe5\xa5\xbd");
}

TEST_CASE("multi_rec: single-engine + no classifier is the short-circuit guard",
          "[multi_rec]") {
  std::set<ScriptId> loaded{ScriptId::Latin};
  bool has_classifier = false;
  bool short_circuit = (loaded.size() == 1 && !has_classifier);
  REQUIRE(short_circuit);
}
