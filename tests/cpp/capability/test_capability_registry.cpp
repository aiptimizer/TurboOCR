// The capability registry's invariants — the ones that make it impossible to
// wire a capability into some endpoints and forget it in others.
//
// Context: layout/tables/formulas used to be three positional bools threaded
// through six route registrars, and the gRPC registrar took them in a DIFFERENT
// ORDER. Transposing two compiled cleanly and silently disabled a feature.
// Separately, `autorotate` existed over HTTP but had no proto field at all, so
// the two transports advertised the same server differently. Nothing in the
// build caught either.
//
// These tests assert the properties that replaced that arrangement. Most are
// static_asserts (a violation is a compile error, not a red test); the runtime
// cases cover what the type system cannot check on its own.

#include <catch_amalgamated.hpp>

#include <set>
#include <string>
#include <string_view>

#include "turbo_ocr/core/capability.h"

using turbo_ocr::capability::CapabilityId;
using turbo_ocr::capability::CapabilityMask;
using turbo_ocr::capability::capability_by_name;
using turbo_ocr::capability::capability_info;
using turbo_ocr::capability::capability_name;
using turbo_ocr::capability::kCapabilities;
using turbo_ocr::capability::kCapabilityCount;

// ---------------------------------------------------------------------------
// The table and the enum are emitted from the same source lines
// ---------------------------------------------------------------------------

static_assert(kCapabilities.size() == kCapabilityCount,
              "descriptor table and enum disagree in length — they are emitted "
              "from the same X-macro, so this cannot happen without an edit "
              "that breaks the invariant");

TEST_CASE("every capability round-trips name <-> id", "[capability]") {
  for (const auto &cap : kCapabilities) {
    INFO("capability: " << cap.name);
    REQUIRE(capability_by_name(cap.name).has_value());
    REQUIRE(*capability_by_name(cap.name) == cap.id);
    REQUIRE(capability_name(cap.id) == cap.name);
  }
}

TEST_CASE("descriptors sit at their own enum index", "[capability]") {
  // kCapabilities is indexed BY the enum value everywhere (capability_info is
  // a raw array subscript). If a row were ever reordered relative to the enum,
  // every lookup would silently return a neighbouring capability's data.
  for (std::size_t i = 0; i < kCapabilities.size(); ++i)
    REQUIRE(static_cast<std::size_t>(kCapabilities[i].id) == i);
}

TEST_CASE("names and error codes are unique and non-empty", "[capability]") {
  // The name is the query param AND the JSON key AND the /capabilities key AND
  // the proto field name. A duplicate would make one capability unreachable.
  std::set<std::string_view> names, codes;
  for (const auto &cap : kCapabilities) {
    INFO("capability: " << cap.name);
    REQUIRE_FALSE(cap.name.empty());
    REQUIRE_FALSE(cap.error_code.empty());
    REQUIRE_FALSE(cap.hint.empty());
    REQUIRE(names.insert(cap.name).second);
    REQUIRE(codes.insert(cap.error_code).second);
  }
}

TEST_CASE("an unknown wire name resolves to nothing", "[capability]") {
  REQUIRE_FALSE(capability_by_name("layouts").has_value());
  REQUIRE_FALSE(capability_by_name("").has_value());
  REQUIRE_FALSE(capability_by_name("LAYOUT").has_value()); // case-sensitive
}

// ---------------------------------------------------------------------------
// set() states a fact; request() pulls in dependencies
// ---------------------------------------------------------------------------

TEST_CASE("request() pulls in dependencies, set() does not", "[capability]") {
  // The distinction is load-bearing. `requested` means "the client asked for
  // tables", which genuinely implies layout. `loaded` means "the table stage
  // came up" — which says nothing about layout, whose model may be missing.
  // Applying implications to the loaded axis would make a server advertise a
  // stage it cannot run.
  CapabilityMask requested;
  requested.request(CapabilityId::Table);
  REQUIRE(requested.get(CapabilityId::Table));
  REQUIRE(requested.get(CapabilityId::Layout));

  CapabilityMask loaded;
  loaded.set(CapabilityId::Table);
  REQUIRE(loaded.get(CapabilityId::Table));
  REQUIRE_FALSE(loaded.get(CapabilityId::Layout));
}

static_assert(CapabilityMask{}
                  .request(CapabilityId::Formula)
                  .get(CapabilityId::Layout),
              "formulas must imply layout");
static_assert(!CapabilityMask{}.set(CapabilityId::Formula).get(CapabilityId::Layout),
              "set() must not apply implications");

TEST_CASE("declared dependencies are self-consistent", "[capability]") {
  for (const auto &cap : kCapabilities) {
    INFO("capability: " << cap.name);
    // A capability cannot depend on itself, and every dependency bit must name
    // a capability that actually exists — a stale bit would silently request a
    // stage nothing can satisfy.
    const auto self = 1u << static_cast<unsigned>(cap.id);
    REQUIRE((cap.implies & self) == 0u);
    const auto all = CapabilityMask::all().bits();
    REQUIRE((cap.implies & ~all) == 0u);

    // Dependencies must be transitively closed: request() applies ONE level of
    // implication, so a chain (a needs b, b needs c) would silently drop c.
    CapabilityMask deps{cap.implies};
    for (const auto &dep : kCapabilities)
      if (deps.get(dep.id))
        REQUIRE((dep.implies & ~deps.bits()) == 0u);
  }
}

TEST_CASE("clearing is literal, even for a dependency", "[capability]") {
  // ?tables=0 must not drop a layout the client also asked for outright.
  CapabilityMask m;
  m.request(CapabilityId::Layout);
  m.request(CapabilityId::Table);
  m.request(CapabilityId::Table, false);
  REQUIRE_FALSE(m.get(CapabilityId::Table));
  REQUIRE(m.get(CapabilityId::Layout));
}

// ---------------------------------------------------------------------------
// "requested but not loaded" — the one question every gate asks
// ---------------------------------------------------------------------------

TEST_CASE("without() answers requested-but-not-loaded", "[capability]") {
  CapabilityMask loaded;
  loaded.set(CapabilityId::Layout);

  CapabilityMask requested;
  requested.request(CapabilityId::Table); // => tables + layout

  const auto missing = requested.without(loaded);
  REQUIRE(missing.get(CapabilityId::Table));
  REQUIRE_FALSE(missing.get(CapabilityId::Layout)); // it IS loaded
  REQUIRE(missing.first().has_value());
  REQUIRE(*missing.first() == CapabilityId::Table);
}

TEST_CASE("a fully-satisfied request has nothing missing", "[capability]") {
  const auto everything = CapabilityMask::all();
  CapabilityMask requested;
  for (const auto &cap : kCapabilities) requested.request(cap.id);
  REQUIRE_FALSE(requested.without(everything).any());
  REQUIRE_FALSE(requested.without(everything).first().has_value());
}

TEST_CASE("first() is deterministic and in table order", "[capability]") {
  // A client retrying the same rejected request must get the same error code
  // back, so the choice of which missing capability to name cannot depend on
  // bit tricks that might reorder.
  CapabilityMask requested;
  for (const auto &cap : kCapabilities) requested.request(cap.id);
  const auto missing = requested.without(CapabilityMask::none());
  REQUIRE(missing.first().has_value());
  REQUIRE(*missing.first() == kCapabilities.front().id);
}

TEST_CASE("none() and all() bracket the space", "[capability]") {
  REQUIRE_FALSE(CapabilityMask::none().any());
  for (const auto &cap : kCapabilities) {
    INFO("capability: " << cap.name);
    REQUIRE(CapabilityMask::all().get(cap.id));
    REQUIRE_FALSE(CapabilityMask::none().get(cap.id));
  }
}

TEST_CASE("every capability carries a distinct bit", "[capability]") {
  // Two capabilities sharing a bit would make requesting one enable the other.
  std::uint32_t seen = 0;
  for (const auto &cap : kCapabilities) {
    INFO("capability: " << cap.name);
    const auto bit = CapabilityMask{}.set(cap.id).bits();
    REQUIRE(bit != 0u);
    REQUIRE((seen & bit) == 0u);
    seen |= bit;
  }
  REQUIRE(seen == CapabilityMask::all().bits());
}

TEST_CASE("the error contract is complete for every capability", "[capability]") {
  // Each capability must be able to produce its own rejection: the gates format
  // {name}=1 was requested but {hint}, tagged with {error_code}. A capability
  // missing any of these could only be refused with a generic message, which is
  // how endpoints ended up inventing their own inconsistent codes.
  for (const auto &cap : kCapabilities) {
    INFO("capability: " << cap.name);
    const auto &info = capability_info(cap.id);
    REQUIRE(info.error_code == cap.error_code);
    REQUIRE(info.hint == cap.hint);
    // The gates emit "{name}=1 was requested but {hint}", so a hint that
    // restates the flag reads as "tables=1 was requested but tables=1
    // requires...". Naming an env var like DISABLE_LAYOUT=1 is fine and is
    // exactly the operator instruction we want — only the capability's own
    // wire flag is disallowed.
    const std::string echoed = std::string(cap.name) + "=1";
    REQUIRE(info.hint.find(echoed) == std::string_view::npos);
  }
}
