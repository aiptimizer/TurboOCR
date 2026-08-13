#pragma once

#include <string>
#include <string_view>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/core/infer_options.h"
#include "turbo_ocr/service/validation/options_core.h"

// proto_options.h — the gRPC adapter over parse_options_core, the twin of the
// Drogon adapter in query_options.h.
//
// Both transports now run the SAME gate. Before this, gRPC carried
// grpc_check_layout_request ("Mirror parse_query_options()" — its own words)
// plus a hand-written copy of the text=0 combination rules in recognize_rpc.cpp.
// Those two copies had already drifted from the HTTP originals in three ways
// documented in options_core.h; a mirrored gate always does.
//
// Flags are resolved BY REFLECTION on the capability's wire name, the same
// mechanism proto_capability_bridge.h uses: a capability's name in
// capability_table.def IS its proto field name, so adding a row to that table
// plus a field to ocr.proto is the whole change — no RPC is edited, and a name
// that fails to line up is caught by missing_capability_fields() rather than by
// a client noticing a silently-dropped flag.
namespace turbo_ocr::server {

// Parse a gRPC request message into InferOptions through the shared core.
//
// `layout_only` is passed separately because it is the ONE flag whose proto
// spelling differs from its HTTP one: over HTTP a layout-only run is `text=0`,
// over gRPC it is the `layout_only` field. Mapping it here rather than teaching
// the core about two names keeps the core transport-free — and means the four
// "text=0 cannot be combined with ..." rejections now apply to gRPC verbatim
// instead of via the flatter, separately-maintained message recognize_rpc.cpp
// used to emit.
//
// PRESENCE. proto3 scalar bools have no has_() unless declared `optional`, so a
// field that is false is indistinguishable from one never set. That is exactly
// right for the opt-in flags (absent == false == not requested) and is why
// `text` cannot be a plain proto bool: it is the one opt-OUT flag, so it is
// derived from layout_only, which carries its presence in its value.
[[nodiscard]] inline ParseOptionsResult
parse_proto_options(const google::protobuf::Message &msg, bool layout_only,
                    const capability::CapabilityMask &loaded, InferOptions *out,
                    bool allow_image_only = false,
                    capability::CapabilityMask acts_on =
                        capability::CapabilityMask::all().set(
                            capability::CapabilityId::DocOrientation, false)) {
  const auto *desc = msg.GetDescriptor();
  const auto *refl = msg.GetReflection();
  const auto read_flag = [&](std::string_view name, bool *value,
                             bool *present) -> std::string {
    if (name == "text") {
      // layout_only=true IS text=0, explicitly sent. layout_only=false leaves
      // `text` unspecified, so the core keeps its default of true.
      *value = !layout_only;
      *present = layout_only;
      return {};
    }
    *value = false;
    *present = false;
    if (!desc || !refl) return {};
    const auto *f = desc->FindFieldByName(std::string(name));
    // A message that simply does not declare the field leaves the flag absent,
    // matching an HTTP request that omitted the query parameter. Whether that
    // absence is INTENDED is asserted separately by missing_capability_fields()
    // in the capability-registry test — a gap must be a build-time failure, not
    // a runtime shrug.
    if (!f || f->type() != google::protobuf::FieldDescriptor::TYPE_BOOL ||
        f->is_repeated())
      return {};
    *value = refl->GetBool(msg, f);
    *present = *value;
    return {};
  };
  return parse_options_core(read_flag, loaded, out, allow_image_only, acts_on);
}

} // namespace turbo_ocr::server
