#pragma once

// capability.h — the ONE definition of an optional OCR capability, and the
// mask type every layer uses to talk about a SET of them.
//
// See capability_table.def for the list itself and why it is an X-macro.
//
// THREE AXES, all typed on the same CapabilityMask so nothing above the
// backend seam ever branches per vendor:
//
//   IMPLEMENTED  can this backend+mode ever build the stage   (BackendCaps)
//   LOADED       did it actually load this boot               (StageAvailability)
//   REQUESTED    did the client ask for it                    (InferOptions)
//
// Build the first two with set() (literal) and the third with request() (also
// pulls in dependencies) — see those methods for why the distinction matters.
//
// Keeping them distinct is what lets /capabilities answer "supported but not
// loaded" (operator can fix by config) separately from "not implemented on
// this backend/mode" (operator cannot) — a single flat bool conflates the two
// and leaves an operator with no idea which knob to reach for.

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

// Defines TURBO_CAPABILITY_TABLE(X). Contains only #define directives, so it is
// included here with the other headers rather than at each expansion site.
#include "turbo_ocr/core/capability_table.def"

namespace turbo_ocr::capability {

// ---- CapabilityId ----------------------------------------------------------
enum class CapabilityId : std::uint8_t {
#define TURBO_CAP_ENUM(Name, name, implies, code, hint) Name,
  TURBO_CAPABILITY_TABLE(TURBO_CAP_ENUM)
#undef TURBO_CAP_ENUM
  Count
};

inline constexpr std::size_t kCapabilityCount =
    static_cast<std::size_t>(CapabilityId::Count);
static_assert(kCapabilityCount <= 32,
              "CapabilityMask is a uint32_t bitset; widen it before adding a "
              "33rd capability");

struct CapabilityDescriptor {
  CapabilityId id;
  std::string_view name; // query param == JSON key == /capabilities key == proto field
  std::uint32_t implies; // capabilities this one requires (see the .def)
  std::string_view error_code; // stable client-facing code when not loaded
  std::string_view hint;       // what the OPERATOR must do to enable it
};

inline constexpr std::array<CapabilityDescriptor, kCapabilityCount> kCapabilities{{
#define TURBO_CAP_ROW(Name, name, implies, code, hint)                         \
  {CapabilityId::Name, name, implies, code, hint},
    TURBO_CAPABILITY_TABLE(TURBO_CAP_ROW)
#undef TURBO_CAP_ROW
}};

[[nodiscard]] constexpr const CapabilityDescriptor &
capability_info(CapabilityId id) noexcept {
  return kCapabilities[static_cast<std::size_t>(id)];
}

[[nodiscard]] constexpr std::string_view capability_name(CapabilityId id) noexcept {
  return capability_info(id).name;
}

// Wire name -> id. Used by every request parser, so a client-supplied string is
// resolved in exactly one place.
[[nodiscard]] constexpr std::optional<CapabilityId>
capability_by_name(std::string_view n) noexcept {
  for (const auto &d : kCapabilities)
    if (d.name == n) return d.id;
  return std::nullopt;
}

// ---- CapabilityMask --------------------------------------------------------
// A set of capabilities. Deliberately a distinct TYPE rather than a bag of
// bools: it cannot be transposed with a neighbouring argument (the bug this
// whole design exists to make impossible), and it can be iterated, so every
// consumer loops over the table instead of hand-listing capabilities.
class CapabilityMask {
public:
  constexpr CapabilityMask() = default;
  explicit constexpr CapabilityMask(std::uint32_t bits) noexcept : bits_(bits) {}

  [[nodiscard]] static constexpr CapabilityMask none() noexcept { return {}; }
  [[nodiscard]] static constexpr CapabilityMask all() noexcept {
    return CapabilityMask{(kCapabilityCount >= 32)
                              ? ~0u
                              : ((1u << kCapabilityCount) - 1u)};
  }

  [[nodiscard]] constexpr bool get(CapabilityId id) const noexcept {
    return (bits_ & bit(id)) != 0u;
  }

  // Set/clear exactly this capability, applying NO implications.
  //
  // This is the operation for the IMPLEMENTED and LOADED axes, where the mask
  // states a FACT: that the table stage loaded says nothing about whether the
  // layout stage did (its model may be missing), and quietly asserting layout
  // here would make a server advertise a stage it cannot run. Use request()
  // when expressing what a client ASKED for.
  constexpr CapabilityMask &set(CapabilityId id, bool on = true) noexcept {
    if (on) bits_ |= bit(id);
    else    bits_ &= ~bit(id);
    return *this;
  }

  // Request a capability, pulling in whatever it DEPENDS ON (tables/formulas
  // are per-layout-region stages, so either needs layout).
  //
  // Only meaningful on the REQUESTED axis — a request for tables genuinely is a
  // request for layout too. Encoding the dependency here means no endpoint has
  // to remember the rule, and none of them can disagree about it. Clearing is
  // deliberately literal (clearing tables must not silently drop a layout the
  // client also asked for outright).
  constexpr CapabilityMask &request(CapabilityId id, bool on = true) noexcept {
    set(id, on);
    if (on) bits_ |= kCapabilities[static_cast<std::size_t>(id)].implies;
    return *this;
  }

  [[nodiscard]] constexpr bool any() const noexcept { return bits_ != 0u; }
  [[nodiscard]] constexpr std::uint32_t bits() const noexcept { return bits_; }

  // The first capability set in this mask, in table order, or nullopt if empty.
  // Used to name ONE capability in an error message when several are missing —
  // reporting the first in declaration order makes the message deterministic
  // (a client retrying the same bad request must get the same code back).
  [[nodiscard]] constexpr std::optional<CapabilityId> first() const noexcept {
    for (const auto &d : kCapabilities)
      if (get(d.id)) return d.id;
    return std::nullopt;
  }

  // Set intersection / difference, for "requested but not loaded" questions.
  [[nodiscard]] constexpr CapabilityMask operator&(CapabilityMask o) const noexcept {
    return CapabilityMask{bits_ & o.bits_};
  }
  [[nodiscard]] constexpr CapabilityMask operator|(CapabilityMask o) const noexcept {
    return CapabilityMask{bits_ | o.bits_};
  }
  // Capabilities in *this that are absent from `o` — i.e. requested-but-absent.
  [[nodiscard]] constexpr CapabilityMask without(CapabilityMask o) const noexcept {
    return CapabilityMask{bits_ & ~o.bits_};
  }
  [[nodiscard]] constexpr bool operator==(const CapabilityMask &) const = default;

private:
  [[nodiscard]] static constexpr std::uint32_t bit(CapabilityId id) noexcept {
    return 1u << static_cast<unsigned>(id);
  }
  std::uint32_t bits_ = 0u;
};

// Convenience for the common "iterate every capability" loop:
//   for (const auto &cap : kCapabilities) { ... cap.name, mask.get(cap.id) ... }
// Deliberately NOT an iterator over set bits only — most consumers must emit a
// row for EVERY capability (e.g. /capabilities reports true AND false), and a
// set-bits-only iterator invites silently omitting the false ones.

} // namespace turbo_ocr::capability
