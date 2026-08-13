// Transport parity: every capability the server can run must be EXPRESSIBLE
// over gRPC, and the gRPC option gate must agree with the HTTP one.
//
// This suite exists because two comments in the tree promised an assertion that
// did not exist. proto_capability_bridge.h says of missing_capability_fields():
// "it is asserted by the capability registry test rather than merely logged" —
// and no test called it. test_capability_registry.cpp's own preamble names the
// failure ("`autorotate` existed over HTTP but had no proto field at all, so the
// two transports advertised the same server differently. Nothing in the build
// caught either") while asserting nothing about the proto.
//
// The build could not have caught it: turbo_ocr_tests never linked the generated
// proto. It does now, and these are the checks that make the promise true.

#include <catch_amalgamated.hpp>

#include <string>
#include <vector>

#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/service/capability/proto_capability_bridge.h"
#include "turbo_ocr/service/validation/proto_options.h"

#include "ocr.pb.h"

using turbo_ocr::capability::CapabilityId;
using turbo_ocr::capability::CapabilityMask;
using turbo_ocr::capability::kCapabilities;
using turbo_ocr::capability::missing_capability_fields;
using turbo_ocr::server::parse_proto_options;

namespace {

// Render a missing-field list into the failure message, so a red test names the
// capability to add rather than only a count.
std::string join(const std::vector<std::string> &v) {
  std::string out;
  for (const auto &s : v) {
    if (!out.empty()) out += ", ";
    out += s;
  }
  return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Every capability is expressible on the transport that can act on it
// ---------------------------------------------------------------------------

TEST_CASE("OCRPDFRequest declares a field for every capability",
          "[capability][proto]") {
  // The PDF RPC passes acts_on = all(), so it must be able to carry all() —
  // autorotate included. This is the exact gap that shipped once.
  const auto missing =
      missing_capability_fields(ocr::OCRPDFRequest::descriptor());
  INFO("OCRPDFRequest is missing a bool field for: " << join(missing));
  CHECK(missing.empty());
}

TEST_CASE("the image RPCs declare every capability they act on",
          "[capability][proto]") {
  // The image RPCs deliberately do NOT act on DocOrientation: they OCR one
  // already-decoded image and never rotate it, so omitting the field makes the
  // unsupported request unrepresentable rather than accepted-then-rejected.
  // Every OTHER capability must be present.
  for (const auto *desc :
       {ocr::OCRRequest::descriptor(), ocr::OCRBatchRequest::descriptor()}) {
    std::vector<std::string> missing;
    for (const auto &name : missing_capability_fields(desc))
      if (name != turbo_ocr::capability::capability_name(
                      CapabilityId::DocOrientation))
        missing.push_back(name);
    INFO(desc->name() << " is missing a bool field for: " << join(missing));
    CHECK(missing.empty());
  }
}

// ---------------------------------------------------------------------------
// The shared gate behaves identically through the proto adapter
// ---------------------------------------------------------------------------

TEST_CASE("a capability the server did not load is rejected, not dropped",
          "[capability][proto]") {
  // The rejection code comes from capability_table.def, which OWNS it — the
  // property that used to depend on grpc_check_layout_request agreeing with its
  // HTTP twin by hand.
  //
  // WHICH code, when several are unmet, is decided by first() "in table order",
  // and Table TURBO_CAPABILITY_IMPLIES(Layout). So the two cases differ:
  SECTION("nothing loaded: the unmet DEPENDENCY is reported first") {
    ocr::OCRRequest req;
    req.set_tables(true);
    turbo_ocr::server::InferOptions opts;
    const auto r = parse_proto_options(req, /*layout_only=*/false,
                                       CapabilityMask::none(), &opts);
    REQUIRE_FALSE(r.error.empty());
    // Layout precedes Table in the table, and tables=1 requested both. Telling
    // the caller "layout is missing" first is right: installing a table backend
    // would not have made this request work.
    CHECK(r.error_code ==
          std::string(
              turbo_ocr::capability::capability_info(CapabilityId::Layout)
                  .error_code));
  }
  SECTION("dependency satisfied: the capability itself is reported") {
    ocr::OCRRequest req;
    req.set_tables(true);
    turbo_ocr::server::InferOptions opts;
    const auto r = parse_proto_options(
        req, /*layout_only=*/false,
        CapabilityMask::none().set(CapabilityId::Layout, true), &opts);
    REQUIRE_FALSE(r.error.empty());
    CHECK(r.error_code ==
          std::string(turbo_ocr::capability::capability_info(CapabilityId::Table)
                          .error_code));
  }
}

TEST_CASE("tables implies layout through the shared core", "[capability][proto]") {
  ocr::OCRRequest req;
  req.set_tables(true);
  turbo_ocr::server::InferOptions opts;
  // Table TURBO_CAPABILITY_IMPLIES(Layout), so a server with both loaded must
  // come out of the gate with want_layout set even though the client never sent
  // layout=1. The dependency lives in the table, not in either transport.
  const auto r = parse_proto_options(req, /*layout_only=*/false,
                                     CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_tables);
  CHECK(opts.want_layout);
}

TEST_CASE("as_blocks implies reading_order implies layout",
          "[capability][proto]") {
  ocr::OCRRequest req;
  req.set_as_blocks(true);
  turbo_ocr::server::InferOptions opts;
  const auto r = parse_proto_options(req, /*layout_only=*/false,
                                     CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_blocks);
  CHECK(opts.want_reading_order);
  CHECK(opts.want_layout);
}

TEST_CASE("text defaults to true and only layout_only clears it",
          "[capability][proto]") {
  // `text` is the one opt-OUT flag and has no proto field: over gRPC it is
  // spelled layout_only. A plain request must not read as text=0.
  ocr::OCRRequest req;
  turbo_ocr::server::InferOptions opts;
  const auto r = parse_proto_options(req, /*layout_only=*/false,
                                     CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK(opts.want_text);
}

// layout_only is BUILD-INDEPENDENT now. It used to be split on USE_CPU_ONLY —
// the CPU arm asserting a flat rejection, the other arm asserting acceptance —
// which mirrored a `#ifdef USE_CPU_ONLY` inside parse_options_core itself. A
// CMake option was deciding a request-validation rule, and the macOS server
// (built USE_CPU_ONLY=ON while running the Apple backend) therefore told
// operators that "the CPU build" was the reason. Nothing implements want_text
// on ANY backend, so the rule is one rule.
TEST_CASE("layout_only rejects every text-derived flag, with the SPECIFIC reason",
          "[capability][proto]") {
  auto rejects = [](auto &&set_flag) {
    ocr::OCRRequest req;
    req.set_layout(true);
    set_flag(req);
    turbo_ocr::server::InferOptions opts;
    return parse_proto_options(req, /*layout_only=*/true, CapabilityMask::all(),
                               &opts);
  };
  // Each names the flag that actually conflicts — not the generic
  // "not implemented" answer a bare layout_only request gets.
  for (const auto &[name, r] : std::initializer_list<
           std::pair<const char *, turbo_ocr::server::ParseOptionsResult>>{
           {"tables", rejects([](auto &r) { r.set_tables(true); })},
           {"formulas", rejects([](auto &r) { r.set_formulas(true); })},
           {"as_blocks", rejects([](auto &r) { r.set_as_blocks(true); })},
           {"reading_order", rejects([](auto &r) { r.set_reading_order(true); })}}) {
    INFO("flag: " << name);
    REQUIRE_FALSE(r.error.empty());
    CHECK(r.error_code == "INVALID_PARAMETER");
    CHECK(r.error.find("not implemented") == std::string::npos);
  }
}

TEST_CASE("a well-formed layout_only request is accepted as a layout-only run",
          "[capability][proto]") {
  // The predecessor of this test pinned the refusal ("refused as
  // unimplemented ... the day a layout-only entry point lands, this is the
  // test that should fail") — and it failed on exactly that day: RunFlags.text
  // is honoured by the unified pipeline now, identically on every build, so
  // gRPC layout_only maps to want_text=false + want_layout=true and runs.
  ocr::OCRRequest req;
  req.set_layout(true);
  turbo_ocr::server::InferOptions opts;
  const auto r = parse_proto_options(req, /*layout_only=*/true,
                                     CapabilityMask::all(), &opts);
  REQUIRE(r.error.empty());
  CHECK_FALSE(opts.want_text);
  CHECK(opts.want_layout);
}
