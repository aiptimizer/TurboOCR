// The Python binding's adapter over the shared request-option gate.
//
// Scope: parse_python_options (validation/python_options.h), the third and last
// adapter over parse_options_core — HTTP's is query_options.h, gRPC's is
// proto_options.h. Before it existed the nanobind module took four plain bools
// straight into pipeline::RunFlags, so a Python caller reached the pipeline with
// NO availability gate, no reading_order/as_blocks implications and no text=0
// combination rules. That is the same drift options_core.h was written to end,
// on the one transport nobody had counted.
//
// Every case below asserts PARITY: the python adapter's (error, error_code) for
// a request must be byte-identical to what the HTTP-shaped reader produces for
// the same request. Asserting the strings by hand would let the two drift the
// moment a message is reworded in options_core.h — comparing the adapters
// against each other cannot.

#include <catch_amalgamated.hpp>

#include <map>
#include <string>
#include <string_view>

#include "turbo_ocr/service/validation/options_core.h"
#include "turbo_ocr/service/validation/python_options.h"

using turbo_ocr::capability::CapabilityId;
using turbo_ocr::capability::CapabilityMask;
using turbo_ocr::capability::capability_info;
using turbo_ocr::server::InferOptions;
using turbo_ocr::server::ParseOptionsResult;
using turbo_ocr::server::parse_options_core;
using turbo_ocr::server::parse_python_options;

namespace {

// The HTTP side of every parity check: the same wire-string flag source
// test_query_options.cpp uses, standing in for query_options.h's
// parse_bool_flag (whose only contribution over this is resolving a name
// against drogon's query string / JSON body — no policy of its own).
struct WireFlags {
  std::map<std::string, std::string> values;

  std::string operator()(std::string_view name, bool *value,
                         bool *present) const {
    *value = false;
    const auto it = values.find(std::string(name));
    *present = it != values.end();
    if (!*present) return {};
    *value = it->second == "1";
    return {};
  }
};

// Run one request through BOTH adapters and require they agree on everything a
// caller can observe: the message, the code, and the parsed options.
struct Parity {
  ParseOptionsResult result; // identical on both sides once checked
  InferOptions opts;
};

Parity both(const std::map<std::string, bool> &flags, CapabilityMask loaded,
            bool allow_image_only = false,
            CapabilityMask acts_on = CapabilityMask::all()) {
  std::map<std::string, std::string> wire;
  for (const auto &[k, v] : flags) wire.emplace(k, v ? "1" : "0");

  InferOptions py_opts;
  const auto py =
      parse_python_options(flags, loaded, &py_opts, allow_image_only, acts_on);

  InferOptions http_opts;
  WireFlags src{wire};
  const auto http = parse_options_core(src, loaded, &http_opts,
                                       allow_image_only, acts_on);

  CHECK(py.error == http.error);
  CHECK(py.error_code == http.error_code);
  CHECK(py_opts.want_layout == http_opts.want_layout);
  CHECK(py_opts.want_tables == http_opts.want_tables);
  CHECK(py_opts.want_formulas == http_opts.want_formulas);
  CHECK(py_opts.want_reading_order == http_opts.want_reading_order);
  CHECK(py_opts.want_blocks == http_opts.want_blocks);
  CHECK(py_opts.want_text == http_opts.want_text);
  return {py, py_opts};
}

} // namespace

// ---------------------------------------------------------------------------
// Implications — the binding passes flags by their capability-registry name, so
// the dependency pull-ins are the core's, not a copy.
// ---------------------------------------------------------------------------

TEST_CASE("python: tables=True alone implies layout, same as HTTP",
          "[python_options][implies]") {
  const auto p = both({{"tables", true}}, CapabilityMask::all());
  REQUIRE(p.result.error.empty());
  CHECK(p.opts.want_tables);
  CHECK(p.opts.want_layout);
}

TEST_CASE("python: reading_order=True implies layout, same as HTTP",
          "[python_options][implies]") {
  const auto p = both({{"reading_order", true}}, CapabilityMask::all());
  REQUIRE(p.result.error.empty());
  CHECK(p.opts.want_reading_order);
  CHECK(p.opts.want_layout);
}

// ---------------------------------------------------------------------------
// The availability gate — the whole point of the task. A Python caller asking
// for a stage this pipeline did not load is REJECTED with the operator-facing
// hint, not handed a result with the stage quietly skipped.
// ---------------------------------------------------------------------------

TEST_CASE("python: a capability this pipeline did not load is rejected",
          "[python_options][availability]") {
  const auto p = both({{"layout", true}}, CapabilityMask::none());
  REQUIRE_FALSE(p.result.error.empty());
  CHECK(p.result.error_code ==
        std::string(capability_info(CapabilityId::Layout).error_code));
  CHECK(p.result.error.find(capability_info(CapabilityId::Layout).hint) !=
        std::string::npos);
}

TEST_CASE("python: tables=True with no table backend is rejected, not dropped",
          "[python_options][availability]") {
  const auto p =
      both({{"tables", true}},
           CapabilityMask::none().set(CapabilityId::Layout, true));
  REQUIRE_FALSE(p.result.error.empty());
  CHECK(p.result.error_code ==
        std::string(capability_info(CapabilityId::Table).error_code));
}

TEST_CASE("python: a request the pipeline can serve passes the gate",
          "[python_options][availability]") {
  const auto p = both({{"layout", true}, {"tables", true}},
                      CapabilityMask::all());
  REQUIRE(p.result.error.empty());
  CHECK(p.opts.want_layout);
  CHECK(p.opts.want_tables);
}

// ---------------------------------------------------------------------------
// text=0 — the four combination rules, each reaching Python with the exact
// message HTTP returns. `allow_image_only` stays false: the binding has no
// image-only route, so a layout-only request is judged the same way an /ocr
// request is.
// ---------------------------------------------------------------------------

TEST_CASE("python: text=False with tables names the conflicting flag",
          "[python_options][text0]") {
  const auto p =
      both({{"text", false}, {"tables", true}}, CapabilityMask::all());
  REQUIRE_FALSE(p.result.error.empty());
  CHECK(p.result.error.find("tables=1") != std::string::npos);
  CHECK(p.result.error_code == "INVALID_PARAMETER");
}

TEST_CASE("python: text=False with formulas names the conflicting flag",
          "[python_options][text0]") {
  const auto p =
      both({{"text", false}, {"formulas", true}}, CapabilityMask::all());
  REQUIRE_FALSE(p.result.error.empty());
  CHECK(p.result.error.find("formulas=1") != std::string::npos);
}

TEST_CASE("python: text=False with as_blocks names the conflicting flag",
          "[python_options][text0]") {
  // The binding does not offer as_blocks today (it cannot aggregate blocks), so
  // this rule is unreachable from the current keyword set — pinned anyway,
  // because the adapter is what a future blocks-capable binding would go
  // through, and the rule must already be live when it does.
  const auto p =
      both({{"text", false}, {"as_blocks", true}}, CapabilityMask::all());
  REQUIRE_FALSE(p.result.error.empty());
  CHECK(p.result.error.find("as_blocks=1") != std::string::npos);
}

TEST_CASE("python: text=False with reading_order names the conflicting flag",
          "[python_options][text0]") {
  const auto p =
      both({{"text", false}, {"reading_order", true}}, CapabilityMask::all());
  REQUIRE_FALSE(p.result.error.empty());
  CHECK(p.result.error.find("reading_order=1") != std::string::npos);
}

TEST_CASE("python: a bare text=False is rejected, never run as full OCR",
          "[python_options][text0]") {
  // What the Python layer actually sends for OCR.read(text=False) with no
  // structure requested. Accepting it would have returned a complete det+rec
  // result for a request that asked for no text at all.
  const auto p = both({{"text", false}}, CapabilityMask::all());
  REQUIRE_FALSE(p.result.error.empty());
  CHECK(p.result.error_code == "INVALID_PARAMETER");
}

// ---------------------------------------------------------------------------
// Presence — the one place a map-shaped flag source could differ from a wire
// one. `text` is the only opt-OUT flag, so "key missing" and "key present and
// false" must stay distinguishable through this adapter too.
// ---------------------------------------------------------------------------

TEST_CASE("python: an absent text key leaves the default true",
          "[python_options][presence]") {
  const auto p = both({{"layout", true}}, CapabilityMask::all());
  REQUIRE(p.result.error.empty());
  CHECK(p.opts.want_text);
}

TEST_CASE("python: text=True present is not read as text=0",
          "[python_options][presence]") {
  // A binding that always inserts the key must not thereby turn every call into
  // an explicit opt-out — that would reject every request the moment `text`
  // became a keyword argument.
  const auto p = both({{"layout", true}, {"text", true}}, CapabilityMask::all());
  REQUIRE(p.result.error.empty());
  CHECK(p.opts.want_text);
  CHECK(p.opts.want_layout);
}

TEST_CASE("python: an opt-in flag present and false is not requested",
          "[python_options][presence]") {
  // The binding passes every keyword argument, including the ones left at their
  // default False. Those must read as "not requested" — otherwise a pipeline
  // with no layout model would reject every single call.
  const auto p = both({{"layout", false},
                       {"tables", false},
                       {"formulas", false},
                       {"reading_order", false},
                       {"text", true}},
                      CapabilityMask::none());
  REQUIRE(p.result.error.empty());
  CHECK_FALSE(p.opts.requested.any());
}
