// The shared request-option parsing/validation policy every transport runs.
//
// Scope: this file targets parse_options_core (options_core.h) DIRECTLY,
// without going through either transport adapter. It is the transport-free
// core query_options.h (HTTP) and proto_options.h (gRPC) both delegate to —
// see options_core.h's own header comment for the history of the two
// transports' gates silently drifting apart before this existed. Testing it
// here, with a hand-written flag source, exercises the exact same logic HTTP
// and gRPC drive while staying framework-free.
//
// query_options.h's own contribution over this core is exactly one thing:
// resolving a flag name against drogon's query-string/body. That resolution
// is a 15-line, already-densely-commented function
// (parse_bool_flag/parse_bool_query) with no policy of its own — the policy
// entirely lives in parse_options_core, so it is tested here instead of
// through a hand-built drogon::HttpRequestPtr.
//
// Also covered: capability::CapabilityMask's set()/request()/without() as
// parse_options_core actually drives them (full round-trip coverage of the
// mask type lives in tests/cpp/capability/test_capability_registry.cpp; this
// file only pins the handful of invariants this policy depends on), and a
// light pass over validate_params' three-way parameter classification (full
// coverage lives in tests/cpp/validation/test_request_validation.cpp).

#include <catch_amalgamated.hpp>

#include <map>
#include <set>
#include <string>
#include <string_view>

#include "turbo_ocr/service/validation/options_core.h"
#include "turbo_ocr/service/validation/request_validation.h"

using turbo_ocr::capability::CapabilityId;
using turbo_ocr::capability::CapabilityMask;
using turbo_ocr::capability::capability_info;
using turbo_ocr::server::check_structure_backends;
using turbo_ocr::server::EndpointSpec;
using turbo_ocr::server::InferOptions;
using turbo_ocr::server::ParseOptionsResult;
using turbo_ocr::server::parse_options_core;
using turbo_ocr::server::RoutingSupport;
using turbo_ocr::server::ValidationError;
using turbo_ocr::server::validate_params;

namespace {

// A minimal flag source implementing the ReadFlag contract documented in
// options_core.h: "" on success, writes *value (false when absent), writes
// whether the request CARRIED the flag at all to *present. This is the exact
// contract query_options.h's parse_bool_flag and gRPC's proto reader both
// satisfy — using a third, independent implementation here means a bug
// shared by all three (rather than a bug in one adapter) is what these tests
// would actually be exercising, which is the point: the POLICY is under
// test, not any one transport's flag resolution.
struct FlagSource {
  std::map<std::string, std::string> values; // name -> raw wire value ("1"/"0")

  std::string operator()(std::string_view name, bool *value,
                         bool *present) const {
    *value = false;
    const auto it = values.find(std::string(name));
    *present = it != values.end();
    if (!*present) return {};
    if (it->second == "1") { *value = true; return {}; }
    if (it->second == "0") { *value = false; return {}; }
    return "Invalid " + std::string(name) + " value: '" + it->second + "'";
  }
};

ParseOptionsResult parse(const std::map<std::string, std::string> &flags,
                         CapabilityMask loaded, InferOptions *out,
                         bool allow_image_only = false,
                         CapabilityMask acts_on = CapabilityMask::all()) {
  FlagSource src{flags};
  return parse_options_core(src, loaded, out, allow_image_only, acts_on);
}

} // namespace

// ---------------------------------------------------------------------------
// Implication chains — pinned because make_infer_func.cpp's comment says
// hand-built InferOptions callers rely on these being applied consistently
// EVERYWHERE a request is parsed, not just on the happy path some endpoint
// happens to exercise.
// ---------------------------------------------------------------------------

TEST_CASE("tables=1 alone implies layout", "[query_options][implies]") {
  InferOptions opts;
  const auto r = parse({{"tables", "1"}}, CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_tables);
  CHECK(opts.want_layout); // NOT sent by the client — pulled in as a dependency
  CHECK(opts.requested.get(CapabilityId::Table));
  CHECK(opts.requested.get(CapabilityId::Layout));
}

TEST_CASE("formulas=1 alone implies layout", "[query_options][implies]") {
  InferOptions opts;
  const auto r = parse({{"formulas", "1"}}, CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_formulas);
  CHECK(opts.want_layout);
}

TEST_CASE("as_blocks=1 implies reading_order implies layout",
          "[query_options][implies]") {
  // as_blocks aggregates paragraph-level output, which needs ordered text,
  // which needs layout regions — a two-hop chain, both hops load-bearing.
  InferOptions opts;
  const auto r = parse({{"as_blocks", "1"}}, CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_blocks);
  CHECK(opts.want_reading_order);
  CHECK(opts.want_layout);
}

TEST_CASE("reading_order=1 alone implies layout but NOT as_blocks",
          "[query_options][implies]") {
  // The dependency runs one way: reading_order needs layout, but nothing
  // needs reading_order to enable as_blocks. Getting this backwards would
  // make every ?reading_order=1 request silently start aggregating blocks.
  InferOptions opts;
  const auto r = parse({{"reading_order", "1"}}, CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_reading_order);
  CHECK(opts.want_layout);
  CHECK_FALSE(opts.want_blocks);
}

TEST_CASE("layout=1 alone implies nothing further",
          "[query_options][implies]") {
  // The base capability: no downstream flag turns on just because layout ran.
  InferOptions opts;
  const auto r = parse({{"layout", "1"}}, CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_layout);
  CHECK_FALSE(opts.want_tables);
  CHECK_FALSE(opts.want_formulas);
  CHECK_FALSE(opts.want_reading_order);
  CHECK_FALSE(opts.want_blocks);
}

TEST_CASE("no flags at all requests nothing but full OCR text",
          "[query_options][implies]") {
  InferOptions opts;
  const auto r = parse({}, CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK_FALSE(opts.requested.any());
  CHECK(opts.want_text); // the one opt-OUT flag: on by default
}

// ---------------------------------------------------------------------------
// The availability gate: REJECT a capability the server did not load, never
// silently drop it. This is the exact failure this codebase keeps hitting
// (see capability.h's header comment and the gRPC-transposed-args history) —
// a request for a capability the server cannot run must come back as a 400
// with a specific code, not succeed with the stage quietly skipped.
// ---------------------------------------------------------------------------

TEST_CASE("a capability the server did not load is rejected, not dropped",
          "[query_options][availability]") {
  InferOptions opts;
  const auto r = parse({{"layout", "1"}}, CapabilityMask::none(), &opts);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error_code ==
        std::string(capability_info(CapabilityId::Layout).error_code));
  // The rejection names the OPERATOR remediation (env var / model path), not
  // a client-fixable retry — the hint text must survive into the message.
  CHECK(r.error.find(capability_info(CapabilityId::Layout).hint) !=
        std::string::npos);
}

TEST_CASE("nothing loaded: the unmet DEPENDENCY is reported first",
          "[query_options][availability]") {
  // Table TURBO_CAPABILITY_IMPLIES(Layout), and first() walks the table in
  // declaration order (Layout precedes Table). With nothing loaded, BOTH are
  // missing, so the client is told about Layout — installing only a table
  // backend would not have made this request work, and the message must say
  // so rather than naming Table and leaving Layout as a second surprise 400.
  InferOptions opts;
  const auto r = parse({{"tables", "1"}}, CapabilityMask::none(), &opts);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error_code ==
        std::string(capability_info(CapabilityId::Layout).error_code));
}

TEST_CASE("dependency satisfied: the capability itself is reported",
          "[query_options][availability]") {
  InferOptions opts;
  const auto r = parse({{"tables", "1"}},
                       CapabilityMask::none().set(CapabilityId::Layout, true),
                       &opts);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error_code ==
        std::string(capability_info(CapabilityId::Table).error_code));
}

TEST_CASE("both loaded: tables=1 succeeds and projects both bools",
          "[query_options][availability]") {
  InferOptions opts;
  const auto r =
      parse({{"tables", "1"}},
            CapabilityMask::none()
                .set(CapabilityId::Layout, true)
                .set(CapabilityId::Table, true),
            &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_tables);
  CHECK(opts.want_layout);
}

TEST_CASE("reading_order=1 with no layout loaded is rejected",
          "[query_options][availability]") {
  // reading_order has no capability id of its own — it rides in as a Layout
  // dependency (options_core.h step 2). The gate must still catch it: a
  // client asking for order over a server with no layout model must not get
  // a 200 with an empty/undefined order back.
  InferOptions opts;
  const auto r =
      parse({{"reading_order", "1"}}, CapabilityMask::none(), &opts);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error_code ==
        std::string(capability_info(CapabilityId::Layout).error_code));
}

TEST_CASE("a rejected parse never projects the pipeline-facing bools",
          "[query_options][availability]") {
  // The comment in infer_options.h says want_layout/want_tables/.../
  // want_autorotate are projected from `requested` in ONE place (step 4 of
  // parse_options_core) so they "cannot drift from requested". That
  // projection sits AFTER the availability gate, so on an error return it
  // never runs at all — want_layout stays false even though `requested`
  // already has the Layout bit set from the dependency pull-in. Every real
  // caller discards `opts` whenever ParseOptionsResult.error is non-empty
  // (request_gate.h's `fail` path returns before opts is used), so this is
  // safe in practice; pinned here so that contract cannot silently change
  // out from under the callers that rely on it.
  InferOptions opts;
  const auto r = parse({{"tables", "1"}}, CapabilityMask::none(), &opts);
  REQUIRE_FALSE(r.error.empty());
  CHECK_FALSE(opts.want_layout);
  CHECK_FALSE(opts.want_tables);
  CHECK_FALSE(opts.want_formulas);
  CHECK_FALSE(opts.want_autorotate);
}

TEST_CASE("acts_on scopes which capabilities this endpoint can even request",
          "[query_options][acts_on]") {
  // Image endpoints default to acts_on = all() minus DocOrientation (they
  // never rotate an already-decoded image). A client sending autorotate=1
  // anyway must NOT be silently ignored by THIS function — acts_on's own
  // header comment (request_validation.h EndpointSpec::acts_on) says flags
  // outside it "fall through to validate_params' classification like any
  // other unsupported param", which is a LOUD lenient-ignore-with-header /
  // strict-400, never a quiet accept-and-do-nothing. Confirm this function's
  // half: the flag is not even read, so it cannot be requested or gated here.
  InferOptions opts;
  const auto acts_on =
      CapabilityMask::all().set(CapabilityId::DocOrientation, false);
  const auto r = parse({{"autorotate", "1"}}, CapabilityMask::none(), &opts,
                       /*allow_image_only=*/false, acts_on);
  REQUIRE(r.error.empty()); // not gated here — but see validate_params below
  CHECK_FALSE(opts.want_autorotate);
  CHECK_FALSE(opts.requested.any());
}

TEST_CASE("acts_on including DocOrientation both parses and gates autorotate",
          "[query_options][acts_on]") {
  InferOptions opts;
  const auto r = parse({{"autorotate", "1"}}, CapabilityMask::none(), &opts,
                       /*allow_image_only=*/false, CapabilityMask::all());
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error_code ==
        std::string(capability_info(CapabilityId::DocOrientation).error_code));

  InferOptions opts2;
  const auto r2 = parse(
      {{"autorotate", "1"}},
      CapabilityMask::none().set(CapabilityId::DocOrientation, true), &opts2,
      /*allow_image_only=*/false, CapabilityMask::all());
  REQUIRE(r2.error.empty());
  CHECK(opts2.want_autorotate);
}

// ---------------------------------------------------------------------------
// check_structure_backends: the SAME gate, re-applied by request_gate.h
// after a post_parse hook mutates opts->requested (the /ocr/pdf markdown
// case). If a hook could add a capability without this re-check, a mutated
// request would reach the pipeline asking for something the server never
// loaded — silently, since nothing downstream re-validates.
// ---------------------------------------------------------------------------

TEST_CASE("check_structure_backends catches a hook-added capability",
          "[query_options][check_structure_backends]") {
  InferOptions opts;
  const auto r = parse({}, CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK_FALSE(opts.want_tables);

  // Simulate /ocr/pdf's markdown post_parse hook defaulting tables on from
  // what it (wrongly, in this test) believes is loaded.
  opts.requested.request(CapabilityId::Table);

  const auto gate = check_structure_backends(opts, CapabilityMask::all().set(
                                                        CapabilityId::Table,
                                                        false));
  REQUIRE_FALSE(gate.error.empty());
  CHECK(gate.error_code ==
        std::string(capability_info(CapabilityId::Table).error_code));
}

TEST_CASE("check_structure_backends passes when the mutated request is fully "
          "satisfied",
          "[query_options][check_structure_backends]") {
  InferOptions opts;
  opts.requested.request(CapabilityId::Table); // pulls in Layout too
  const auto gate = check_structure_backends(opts, CapabilityMask::all());
  CHECK(gate.error.empty());
}

// ---------------------------------------------------------------------------
// text=0 (layout-only): the four documented combination rules. Each is a
// SEPARATE branch in options_core.h precisely so the client gets the
// specific reason rather than one flat "invalid combination" message — pin
// each one so a future edit can't collapse two branches into one message.
// ---------------------------------------------------------------------------

// One contract on EVERY build. This block used to be #ifndef USE_CPU_ONLY
// with an #else pinning "text=0 is refused outright on the CPU build" — the
// exact per-build divergence options_core.h now forbids: RunFlags.text is
// honoured by the unified pipeline identically everywhere.

TEST_CASE("text absent defaults true; text=0 with layout=1 is a valid "
          "layout-only run",
          "[query_options][text0]") {
  InferOptions opts;
  const auto r = parse({{"layout", "1"}, {"text", "0"}}, CapabilityMask::all(),
                       &opts);
  REQUIRE(r.error.empty());
  CHECK_FALSE(opts.want_text);
  CHECK(opts.want_layout);
}

TEST_CASE("text=0 without layout and without allow_image_only is rejected",
          "[query_options][text0]") {
  // Nothing would come back on this endpoint — fail loud instead of
  // returning HTTP 200 with an empty results array the client cannot
  // distinguish from a genuinely blank page.
  InferOptions opts;
  const auto r = parse({{"text", "0"}}, CapabilityMask::all(), &opts,
                       /*allow_image_only=*/false);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error_code == "INVALID_PARAMETER");
}

TEST_CASE("text=0 without layout IS allowed when allow_image_only is set",
          "[query_options][text0]") {
  // /ocr/pdf?text=0&images=inline: the route re-checks against images=inline
  // itself, so the shared core must not pre-empt it.
  InferOptions opts;
  const auto r = parse({{"text", "0"}}, CapabilityMask::all(), &opts,
                       /*allow_image_only=*/true);
  REQUIRE(r.error.empty());
  CHECK_FALSE(opts.want_text);
  CHECK_FALSE(opts.want_layout);
}

TEST_CASE("text=0 rejects tables=1 and formulas=1 with the specific reason",
          "[query_options][text0]") {
  InferOptions o1;
  auto r1 = parse({{"text", "0"}, {"tables", "1"}}, CapabilityMask::all(), &o1,
                  /*allow_image_only=*/true);
  REQUIRE_FALSE(r1.error.empty());
  CHECK(r1.error.find("tables=1") != std::string::npos);

  InferOptions o2;
  auto r2 = parse({{"text", "0"}, {"formulas", "1"}}, CapabilityMask::all(),
                  &o2, /*allow_image_only=*/true);
  REQUIRE_FALSE(r2.error.empty());
  CHECK(r2.error.find("formulas=1") != std::string::npos);
}

TEST_CASE("text=0 rejects as_blocks=1 with the specific reason",
          "[query_options][text0]") {
  InferOptions opts;
  auto r = parse({{"text", "0"}, {"as_blocks", "1"}}, CapabilityMask::all(),
                 &opts, /*allow_image_only=*/true);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error.find("as_blocks=1") != std::string::npos);
}

TEST_CASE("text=0 rejects reading_order=1 with the specific reason",
          "[query_options][text0]") {
  InferOptions opts;
  auto r = parse({{"text", "0"}, {"reading_order", "1"}}, CapabilityMask::all(),
                 &opts, /*allow_image_only=*/true);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error.find("reading_order=1") != std::string::npos);
}

TEST_CASE("text=0's four rejections fire even before allow_image_only would "
          "have let a bare text=0 through",
          "[query_options][text0]") {
  // allow_image_only=true alone is not a blanket pass for text=0 — it only
  // waives the "returns nothing" check, not the text-derived-flag conflicts.
  InferOptions opts;
  auto r = parse({{"text", "0"}, {"tables", "1"}}, CapabilityMask::all(),
                 &opts, /*allow_image_only=*/true);
  REQUIRE_FALSE(r.error.empty());
}

TEST_CASE("text is the one opt-OUT flag: present-vs-absent is distinguishable",
          "[query_options][text0]") {
  // If a reader could not tell "omitted" from "sent false", every request
  // would either always read as text=... (breaking the default-true
  // contract) or silently ignore an explicit text=0. FlagSource's *present
  // out-param is exactly what makes this representable.
  InferOptions absent;
  REQUIRE(parse({}, CapabilityMask::all(), &absent).error.empty());
  CHECK(absent.want_text);

  InferOptions explicit_true;
  REQUIRE(
      parse({{"text", "1"}}, CapabilityMask::all(), &explicit_true).error.empty());
  CHECK(explicit_true.want_text);

  InferOptions explicit_false;
  REQUIRE(parse({{"layout", "1"}, {"text", "0"}}, CapabilityMask::all(),
                &explicit_false)
              .error.empty());
  CHECK_FALSE(explicit_false.want_text);
}

TEST_CASE("an invalid flag value is a parse error, not a silent false",
          "[query_options][text0]") {
  InferOptions opts;
  auto r = parse({{"layout", "banana"}}, CapabilityMask::all(), &opts);
  REQUIRE_FALSE(r.error.empty());
  CHECK(r.error_code == "INVALID_PARAMETER");
}

// ---------------------------------------------------------------------------
// capability::CapabilityMask — only the round-trips this policy leans on.
// Exhaustive coverage (every id, uniqueness, transitive closure, bit
// isolation) lives in tests/cpp/capability/test_capability_registry.cpp;
// duplicating all of it here would just be two suites to keep in sync.
// ---------------------------------------------------------------------------

TEST_CASE("CapabilityMask set/get round-trips without implications",
          "[query_options][capability_mask]") {
  CapabilityMask m;
  CHECK_FALSE(m.get(CapabilityId::Table));
  m.set(CapabilityId::Table, true);
  CHECK(m.get(CapabilityId::Table));
  CHECK_FALSE(m.get(CapabilityId::Layout)); // set() is literal, per capability.h
  m.set(CapabilityId::Table, false);
  CHECK_FALSE(m.get(CapabilityId::Table));
}

TEST_CASE("CapabilityMask::all() and none() round-trip through every "
          "capability this table defines",
          "[query_options][capability_mask]") {
  const auto all = CapabilityMask::all();
  const auto none = CapabilityMask::none();
  for (const auto id : {CapabilityId::Layout, CapabilityId::Table,
                        CapabilityId::Formula, CapabilityId::DocOrientation}) {
    CHECK(all.get(id));
    CHECK_FALSE(none.get(id));
  }
  CHECK(all.without(all) == none);
  CHECK(none.without(all) == none);
  CHECK(all.without(none) == all);
}

// ---------------------------------------------------------------------------
// validate_params — the three-way parameter classification (accepted /
// ignored-with-header / rejected in strict mode). A light pass: this
// function's full surface (routing, every EndpointSpec flag group,
// is_known_param) is covered exhaustively in
// tests/cpp/validation/test_request_validation.cpp; these cases exist so
// this file, read on its own, documents all three categories the header
// comment promises without requiring a reader to jump files.
// ---------------------------------------------------------------------------

namespace {
using Params = std::map<std::string, std::string>;
const std::set<std::string> kNoTableBackends{};
const std::set<std::string> kNoFormulaBackends{};
} // namespace

TEST_CASE("category 1: a supported param is accepted in both modes",
          "[query_options][validate_params]") {
  turbo_ocr::backend_routing::RequestRouting routing;
  EndpointSpec spec; // ocr_options = true by default
  const Params p{{"layout", "1"}};
  CHECK(validate_params(p, spec, kNoTableBackends, kNoFormulaBackends,
                        /*strict=*/false, &routing)
            .ok());
  CHECK(validate_params(p, spec, kNoTableBackends, kNoFormulaBackends,
                        /*strict=*/true, &routing)
            .ok());
}

TEST_CASE("category 2: known-but-unsupported is ignored-with-header in "
          "lenient mode, rejected in strict mode",
          "[query_options][validate_params]") {
  // dpi is a real PDF-endpoint param; an image endpoint never declares
  // pdf_options, so it lands in category 2 — dropping it can't falsify the
  // response (the endpoint never reads it), so lenient tolerates it (and
  // surfaces it via X-Ignored-Params at the request_gate.h layer, not here).
  turbo_ocr::backend_routing::RequestRouting routing;
  EndpointSpec spec;
  const Params p{{"dpi", "150"}};
  std::vector<std::string> ignored;
  auto lenient = validate_params(p, spec, kNoTableBackends, kNoFormulaBackends,
                                 /*strict=*/false, &routing, &ignored);
  CHECK(lenient.ok());
  REQUIRE(ignored.size() == 1);
  CHECK(ignored[0] == "dpi");

  auto strict = validate_params(p, spec, kNoTableBackends, kNoFormulaBackends,
                                /*strict=*/true, &routing);
  CHECK_FALSE(strict.ok());
  CHECK(strict.code == "INVALID_PARAMETER");
}

TEST_CASE("category 3: a wholly unknown param is ignored in lenient mode, "
          "rejected in strict mode",
          "[query_options][validate_params]") {
  turbo_ocr::backend_routing::RequestRouting routing;
  EndpointSpec spec;
  const Params p{{"totally_made_up", "1"}};
  CHECK(validate_params(p, spec, kNoTableBackends, kNoFormulaBackends,
                        /*strict=*/false, &routing)
            .ok());
  auto strict = validate_params(p, spec, kNoTableBackends, kNoFormulaBackends,
                                /*strict=*/true, &routing);
  CHECK_FALSE(strict.ok());
  CHECK(strict.message.find("Unknown query parameter") != std::string::npos);
}

TEST_CASE("routing overrides on a kUnsupported endpoint are rejected, never "
          "validated-then-ignored",
          "[query_options][validate_params]") {
  // A silently-ignored override would let a client believe its routing
  // choice took effect when the default backend actually ran — the same
  // silent-failure shape the availability gate exists to prevent.
  turbo_ocr::backend_routing::RequestRouting routing;
  EndpointSpec spec;
  spec.routing = RoutingSupport::kUnsupported;
  const Params p{{"route_table", "some-backend"}};
  auto r = validate_params(p, spec, {"some-backend"}, kNoFormulaBackends,
                           /*strict=*/false, &routing);
  CHECK_FALSE(r.ok());
  CHECK(r.code == "INVALID_PARAMETER");
  CHECK(routing.empty()); // never populated
}
