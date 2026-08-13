#pragma once

// proto_capability_bridge.h — turn a gRPC request into a CapabilityMask by
// REFLECTION over the capability table, instead of hand-written glue.
//
// WHY: the gRPC path used to unpack each capability by hand
// (`request->tables()`, `request->formulas()`, …) and thread the results
// positionally into a checker whose argument order differed from the HTTP one.
// Two consequences, both of which actually happened:
//
//   * transposing two of those positional flags compiled cleanly and silently
//     disabled a feature; and
//   * `autorotate` was simply never added to the proto, so a capability the
//     HTTP surface honoured did not exist over gRPC at all — and nothing in
//     the build noticed the transports had diverged.
//
// Reflection removes both. A capability's wire name in capability_table.def IS
// the proto field name, so adding a row to that table and a matching field to
// ocr.proto is the entire change: no RPC has to be edited, and a name that
// does not line up is caught by missing_capability_fields() rather than by a
// client eventually noticing a silently-dropped flag.
//
// COST: a field lookup per capability per request. FindFieldByName is a hash
// lookup on the descriptor and the table has a handful of entries, so this is
// noise next to decoding an image — and it buys the property that the two
// transports cannot drift.

#include <string>
#include <vector>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

#include "turbo_ocr/core/capability.h"

namespace turbo_ocr::capability {

// Build the REQUESTED mask from any proto message whose bool fields are named
// after capabilities. Fields the message does not declare are simply absent
// (false) — use missing_capability_fields() to assert a message declares them
// all, rather than discovering the gap in production.
[[nodiscard]] inline CapabilityMask
capabilities_from_proto(const google::protobuf::Message &msg) {
  CapabilityMask out;
  const auto *desc = msg.GetDescriptor();
  const auto *refl = msg.GetReflection();
  if (!desc || !refl) return out;
  for (const auto &cap : kCapabilities) {
    const auto *f = desc->FindFieldByName(std::string(cap.name));
    if (!f || f->type() != google::protobuf::FieldDescriptor::TYPE_BOOL ||
        f->is_repeated())
      continue;
    if (refl->GetBool(msg, f))
      out.request(cap.id); // also pulls in dependencies (tables => layout)
  }
  return out;
}

// Which capabilities this message type has NO matching bool field for.
//
// Empty means the proto and the capability table agree. A non-empty result is
// a build-time defect — the transport silently cannot express those
// capabilities — so it is asserted by the capability registry test rather than
// merely logged.
[[nodiscard]] inline std::vector<std::string>
missing_capability_fields(const google::protobuf::Descriptor *desc) {
  std::vector<std::string> missing;
  if (!desc) return missing;
  for (const auto &cap : kCapabilities) {
    const auto *f = desc->FindFieldByName(std::string(cap.name));
    if (!f || f->type() != google::protobuf::FieldDescriptor::TYPE_BOOL ||
        f->is_repeated())
      missing.emplace_back(cap.name);
  }
  return missing;
}

} // namespace turbo_ocr::capability
