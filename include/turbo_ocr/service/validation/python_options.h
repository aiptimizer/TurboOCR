#pragma once

#include <map>
#include <string>
#include <string_view>

#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/core/infer_options.h"
#include "turbo_ocr/service/validation/options_core.h"

// python_options.h — the Python-binding adapter over parse_options_core, the
// third of exactly three (query_options.h for Drogon, proto_options.h for gRPC).
//
// WHY A THIRD ONE. The nanobind module took its request flags as four plain
// bools straight into pipeline::RunFlags, so a Python caller reached the
// pipeline WITHOUT the capability-availability gate, without the
// reading_order/as_blocks implications and without the text=0 combination
// rules. That is the same divergence options_core.h was written to end, only on
// a transport nobody had counted: the gRPC copy of these rules had already
// drifted from the HTTP original three separate times before it was deleted.
//
// A Python caller is a transport client like any other. It gets the same
// rejections, with the same strings, or the three surfaces are back to being
// three policies.
//
// The flag source here is a NAME->VALUE map rather than a request object
// because that is what the binding actually has: nanobind resolves each keyword
// argument before the call, so presence is "the binding put the key in the map",
// not "the wire carried it". Keying by the capability's registry name is what
// keeps this table-driven — a new row in capability_table.def is requestable
// through this adapter as soon as the binding passes its name, with no edit
// here.
namespace turbo_ocr::server {

// Parse a Python-supplied flag map into InferOptions through the shared core.
//
// PRESENCE. A key in `flags` is present, whatever its value; a key absent from
// it was not requested. This matters for exactly one flag: `text` is the one
// opt-OUT flag (default true), so a binding that always inserts "text" must
// insert its real value, and one that omits it leaves the default standing.
[[nodiscard]] inline ParseOptionsResult
parse_python_options(const std::map<std::string, bool> &flags,
                     const capability::CapabilityMask &loaded, InferOptions *out,
                     bool allow_image_only = false,
                     capability::CapabilityMask acts_on =
                         capability::CapabilityMask::all().set(
                             capability::CapabilityId::DocOrientation, false)) {
  const auto read_flag = [&](std::string_view name, bool *value,
                             bool *present) -> std::string {
    const auto it = flags.find(std::string(name));
    *present = it != flags.end();
    *value = *present && it->second;
    // No parse error is representable: Python hands over a bool, not a wire
    // string. The error slot of the ReadFlag contract stays for the transports
    // that do have one.
    return {};
  };
  return parse_options_core(read_flag, loaded, out, allow_image_only, acts_on);
}

} // namespace turbo_ocr::server
