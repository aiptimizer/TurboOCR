#include <catch_amalgamated.hpp>

#include <cstdlib>

#include "turbo_ocr/backend_routing/routing_config.h"

using turbo_ocr::backend_routing::BackendSpec;
using turbo_ocr::backend_routing::Kind;
using turbo_ocr::backend_routing::load_routing_config;
using turbo_ocr::backend_routing::parse_inline_backend;
using turbo_ocr::backend_routing::parse_parser;
using turbo_ocr::backend_routing::Parser;
using turbo_ocr::backend_routing::routable_backend_names;
using turbo_ocr::backend_routing::RoutingConfigError;

namespace {
// Every env var the routing loader / synth path reads — wiped between cases so
// each test starts from a known baseline.
void reset_env() {
  ::unsetenv("TURBO_ROUTING_CONFIG");
  ::unsetenv("FORMULA_BACKEND");
  ::unsetenv("TABLE_BACKEND");
  ::unsetenv("TABLE_SLANEXT_ENCODER_ONNX");
  ::unsetenv("VLLM_TABLE_BASE_URL");
}
} // namespace

// ---------------------------------------------------------------------------
// api_key env-indirection (security: no literal secrets accepted)
// ---------------------------------------------------------------------------

TEST_CASE("api_key literal secret is rejected", "[routing_config][security]") {
  // A bare value (not 'env:NAME') must fail — never accept an inline secret.
  const std::string j =
      R"({"kind":"openai","base_url":"http://h:1","model":"m",)"
      R"("prompt":"Table Recognition:","parser":"otsl","api_key":"sk-literal"})";
  CHECK_THROWS_AS(parse_inline_backend("table", j), RoutingConfigError);
}

TEST_CASE("api_key env:UNSET_VAR is rejected", "[routing_config][security]") {
  ::unsetenv("TURBO_TEST_UNSET_KEY");
  const std::string j =
      R"({"kind":"openai","base_url":"http://h:1","model":"m",)"
      R"("prompt":"Table Recognition:","parser":"otsl",)"
      R"("api_key":"env:TURBO_TEST_UNSET_KEY"})";
  CHECK_THROWS_AS(parse_inline_backend("table", j), RoutingConfigError);
}

TEST_CASE("api_key env:SET_VAR resolves the secret", "[routing_config][security]") {
  ::setenv("TURBO_TEST_SET_KEY", "sk-from-env", 1);
  const std::string j =
      R"({"kind":"openai","base_url":"http://h:1","model":"m",)"
      R"("prompt":"Table Recognition:","parser":"otsl",)"
      R"("api_key":"env:TURBO_TEST_SET_KEY"})";
  BackendSpec b;
  CHECK_NOTHROW(b = parse_inline_backend("table", j));
  CHECK(b.kind == Kind::Openai);
  CHECK(b.api_key == "sk-from-env");
  ::unsetenv("TURBO_TEST_SET_KEY");
}

// ---------------------------------------------------------------------------
// Malformed JSON must throw, never crash (DoS regression guard for /infer)
// ---------------------------------------------------------------------------

TEST_CASE("parse_inline_backend rejects malformed JSON without crashing",
          "[routing_config][security]") {
  // Wrong-typed fields and outright non-JSON: every one must surface as a
  // RoutingConfigError (mapped to a 400) rather than a throw of a json type
  // error or an abort.
  CHECK_THROWS_AS(parse_inline_backend("table", R"({"kind":123})"),
                  RoutingConfigError);
  CHECK_THROWS_AS(
      parse_inline_backend(
          "table",
          R"({"kind":"openai","base_url":"http://h:1","model":"m",)"
          R"("prompt":"Table Recognition:","max_tokens":"x"})"),
      RoutingConfigError);
  CHECK_THROWS_AS(parse_inline_backend("table", "not json"),
                  RoutingConfigError);
  CHECK_THROWS_AS(parse_inline_backend("table", ""), RoutingConfigError);
}

// ---------------------------------------------------------------------------
// kind:local engine must be valid for the modality (ROUTING_MODALITY_MISMATCH)
// ---------------------------------------------------------------------------

TEST_CASE("kind:local rejects an engine invalid for the modality",
          "[routing_config]") {
  // A formula engine routed as a table backend is a mismatch.
  const std::string j = R"({"kind":"local","engine":"formulanet"})";
  REQUIRE_THROWS_AS(parse_inline_backend("table", j), RoutingConfigError);
  try {
    parse_inline_backend("table", j);
  } catch (const RoutingConfigError &e) {
    CHECK(std::string(e.what()).find("ROUTING_MODALITY_MISMATCH") !=
          std::string::npos);
  }
}

TEST_CASE("kind:local accepts every valid modality/engine pair",
          "[routing_config]") {
  CHECK_NOTHROW(
      parse_inline_backend("table", R"({"kind":"local","engine":"slanext"})"));
  CHECK_NOTHROW(
      parse_inline_backend("table", R"({"kind":"local","engine":"vlm"})"));
  CHECK_NOTHROW(parse_inline_backend(
      "formula", R"({"kind":"local","engine":"formulanet"})"));
  CHECK_NOTHROW(parse_inline_backend(
      "formula", R"({"kind":"local","engine":"ppformulanet_s"})"));
  CHECK_NOTHROW(
      parse_inline_backend("formula", R"({"kind":"local","engine":"vlm"})"));
}

// ---------------------------------------------------------------------------
// parser mapping
// ---------------------------------------------------------------------------

TEST_CASE("parse_parser maps the four known parsers", "[routing_config]") {
  Parser p;
  CHECK((parse_parser("otsl", p) && p == Parser::Otsl));
  CHECK((parse_parser("latex", p) && p == Parser::Latex));
  CHECK((parse_parser("text", p) && p == Parser::Text));
  CHECK((parse_parser("raw", p) && p == Parser::Raw));
  // Empty string is the implicit Raw default.
  CHECK((parse_parser("", p) && p == Parser::Raw));
}

TEST_CASE("parse_parser rejects garbage", "[routing_config]") {
  Parser p = Parser::Otsl;
  CHECK_FALSE(parse_parser("html", p));
  CHECK_FALSE(parse_parser("OTSL", p));
  CHECK_FALSE(parse_parser("json", p));
}

TEST_CASE("openai backend with a garbage parser is rejected", "[routing_config]") {
  const std::string j =
      R"({"kind":"openai","base_url":"http://h:1","model":"m",)"
      R"("prompt":"Table Recognition:","parser":"bogus"})";
  CHECK_THROWS_AS(parse_inline_backend("table", j), RoutingConfigError);
}

// ---------------------------------------------------------------------------
// synth_from_env via load_routing_config (TURBO_ROUTING_CONFIG unset)
// ---------------------------------------------------------------------------

TEST_CASE("synth: all unset yields ppformulanet_s formula + geometric-default table",
          "[routing_config]") {
  reset_env();
  auto t = load_routing_config();

  REQUIRE(t.routes.count("formula") == 1);
  REQUIRE(t.routes.count("table") == 1);
  REQUIRE(t.routes.count("text") == 1);

  // Default is the WORKING ORT-sidecar backend, not the inert fused-Loop TRT
  // "formulanet" engine (which would serve silent-empty on TRT 10.15).
  const auto &fb = t.backends.at(t.routes.at("formula"));
  CHECK(fb.kind == Kind::Local);
  CHECK(fb.engine == "ppformulanet_s");

  // With no table backend configured, table routes to "default" (geometric
  // line-art fallback) — NOT an unconfigured slanext that would fail to load and,
  // under the fail-loud load contract, abort boot.
  CHECK(t.routes.at("table") == "default");

  // text always uses the built-in det/cls/rec path.
  CHECK(t.routes.at("text") == "default");
}

TEST_CASE("synth: FORMULA_BACKEND env overrides the formula engine",
          "[routing_config]") {
  reset_env();
  ::setenv("FORMULA_BACKEND", "ppformulanet_s", 1);
  auto t = load_routing_config();
  CHECK(t.backends.at(t.routes.at("formula")).engine == "ppformulanet_s");
}

TEST_CASE("synth: TABLE_SLANEXT_ENCODER_ONNX infers table=slanext",
          "[routing_config]") {
  reset_env();
  ::setenv("TABLE_SLANEXT_ENCODER_ONNX", "models/slanext_enc.onnx", 1);
  auto t = load_routing_config();
  CHECK(t.backends.at(t.routes.at("table")).engine == "slanext");
}

TEST_CASE("synth: VLLM_TABLE_BASE_URL (no slanext) infers table=vlm",
          "[routing_config]") {
  reset_env();
  ::setenv("VLLM_TABLE_BASE_URL", "http://localhost:8000", 1);
  auto t = load_routing_config();
  // env-only 'vlm' is a kind:Local engine routed to the tested VLMTable client.
  const auto &tb = t.backends.at(t.routes.at("table"));
  CHECK(tb.kind == Kind::Local);
  CHECK(tb.engine == "vlm");
}

TEST_CASE("synth: slanext encoder wins over a table base url",
          "[routing_config]") {
  reset_env();
  ::setenv("TABLE_SLANEXT_ENCODER_ONNX", "models/slanext_enc.onnx", 1);
  ::setenv("VLLM_TABLE_BASE_URL", "http://localhost:8000", 1);
  auto t = load_routing_config();
  CHECK(t.backends.at(t.routes.at("table")).engine == "slanext");
}

// ---------------------------------------------------------------------------
// routable_backend_names
// ---------------------------------------------------------------------------

TEST_CASE("routable_backend_names includes the modality default",
          "[routing_config]") {
  reset_env();
  ::setenv("TABLE_BACKEND", "slanext", 1);  // a configured table -> "table-env" default
  auto t = load_routing_config();
  auto names = routable_backend_names(t, "table");
  // The synthesized table default ("table-env") is the routable name.
  CHECK(names.count(t.routes.at("table")) == 1);
}

TEST_CASE("routable_backend_names includes every openai backend",
          "[routing_config]") {
  reset_env();
  ::setenv("TABLE_BACKEND", "slanext", 1);  // configured table default so it's routable
  // Build a table with an openai backend so we exercise the openai-collect path
  // alongside the modality default, without touching the filesystem loader.
  BackendSpec oa = parse_inline_backend(
      "table",
      R"({"kind":"openai","base_url":"http://h:1","model":"m",)"
      R"("prompt":"Table Recognition:","parser":"otsl"})");
  oa.name = "remote-table";

  auto t = load_routing_config();
  t.backends["remote-table"] = oa;

  auto names = routable_backend_names(t, "table");
  CHECK(names.count("remote-table") == 1);          // openai endpoint, any modality
  CHECK(names.count(t.routes.at("table")) == 1);    // modality default
  // 'default' route sentinels are never advertised as a routable name.
  CHECK(names.count("default") == 0);
}
