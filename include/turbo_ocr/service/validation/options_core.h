#pragma once

#include <format>
#include <string>
#include <string_view>

#include "turbo_ocr/core/capability.h"
#include "turbo_ocr/core/infer_options.h"

// options_core.h — the TRANSPORT-FREE core of request-option parsing.
//
// Everything parse_query_options does is pure logic over (flag values, loaded
// capabilities) EXCEPT reading a flag's value, which is the only part that knows
// about Drogon. Splitting exactly that out lets HTTP and gRPC share the rest:
// the capability-table sweep, the reading_order/as_blocks implications, the one
// availability gate, the mask->bools projection, and the text=0 combination
// rules.
//
// WHY THIS EXISTS. gRPC did not share any of it. It carried
// grpc_check_layout_request, whose own comment opened "Mirror
// parse_query_options()" — a hand-written copy of a gate, which is the failure
// mode this codebase keeps paying for:
//
//   * the gRPC registrar once took the layout/table/formula triple in a
//     DIFFERENT ORDER than HTTP and nothing noticed (request_gate.h);
//   * `autorotate` existed on the HTTP surface and simply was not in the proto
//     (proto_capability_bridge.h);
//   * the text=0 combination rules were written twice, and the copy in
//     recognize_rpc.cpp checked a different set of flags than this one does.
//
// A mirrored gate drifts. A shared one cannot. Adding a capability row to
// capability_table.def must change the behaviour of BOTH transports at once, and
// after this header it does.
//
// The flag reader is a TEMPLATE parameter, not a std::function: this runs per
// request per capability, and the whole point of the split is that it costs
// nothing to share.
namespace turbo_ocr::server {

// Empty `error` means success; `error_code` is the HTTP error-envelope /
// gRPC x-error-code identifier.
struct ParseOptionsResult {
  std::string error;      // empty on success
  std::string error_code; // populated when error is non-empty
};

// The contract a transport's flag reader must satisfy:
//
//   std::string read_flag(std::string_view name, bool *value, bool *present)
//
//   - returns "" on success, or a human-readable parse error;
//   - writes the flag's value to *value (false when the request omits it);
//   - writes whether the request CARRIED the flag at all to *present.
//
// `present` is not redundant with `value`: `text` is the one opt-OUT flag
// (default true), so "absent" and "sent as false" must be distinguishable. A
// reader that always reported present=true would turn every request into an
// explicit text=... and change nothing visible until someone sent text=0.
template <class ReadFlag>
[[nodiscard]] ParseOptionsResult
parse_options_core(ReadFlag &&read_flag, const capability::CapabilityMask &loaded,
                   InferOptions *out, bool allow_image_only,
                   capability::CapabilityMask acts_on) {
  using capability::CapabilityId;
  *out = {};
  bool present = false;

  // ---- 1. Every capability this endpoint acts on, from the table ------------
  // No capability is named here. Adding a row to capability_table.def makes it
  // requestable on every endpoint that can run it, on every transport, with its
  // documented rejection — which is precisely what stops a capability from being
  // wired into some surfaces and forgotten in others.
  for (const auto &cap : capability::kCapabilities) {
    if (!acts_on.get(cap.id)) continue;
    bool on = false;
    if (auto err = read_flag(cap.name, &on, &present); !err.empty())
      return {err, "INVALID_PARAMETER"};
    if (on) out->requested.request(cap.id); // also pulls in dependencies
  }

  // ---- 2. Flags that are not capabilities but imply one ---------------------
  // reading_order and as_blocks are derived VIEWS of layout output rather than
  // separate stages, so they have no availability of their own — they just need
  // layout. Auto-enabling it here is what makes ?reading_order=1 alone return a
  // populated array, as documented.
  if (auto err = read_flag("reading_order", &out->want_reading_order, &present);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (auto err = read_flag("as_blocks", &out->want_blocks, &present);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (out->want_blocks)
    out->want_reading_order = true; // aggregation consumes ordered text
  if (out->want_reading_order)
    out->requested.request(CapabilityId::Layout);

  // ---- 3. One availability gate for all of them -----------------------------
  // ONE rejection path, so every endpoint on every transport returns the same
  // code for the same condition and none can accept a request it cannot serve.
  // Message wording: "is required for this request" is true whether the client
  // sent the flag directly (tables=1) or it was pulled in as a dependency
  // (reading_order=1 needs layout).
  if (const auto missing = out->requested.without(loaded).first()) {
    const auto &info = capability::capability_info(*missing);
    return {std::format("{} is required for this request but {}. See GET "
                        "/capabilities for what this server supports.",
                        info.name, info.hint),
            std::string(info.error_code)};
  }

  // ---- 4. Project the mask onto the pipeline's plain bools ------------------
  // The single place these are assigned. Everything downstream reads them, so
  // they cannot drift from `requested` — there is nowhere else to set them.
  out->want_layout     = out->requested.get(CapabilityId::Layout);
  out->want_tables     = out->requested.get(CapabilityId::Table);
  out->want_formulas   = out->requested.get(CapabilityId::Formula);
  out->want_autorotate = out->requested.get(CapabilityId::DocOrientation);

  // ---- 5. text: the one opt-OUT flag (default true) -------------------------
  // The absent->false convention above is for opt-in flags, so only honour this
  // one when the request actually carried it.
  out->want_text = true;
  {
    bool text_value = true;
    if (auto err = read_flag("text", &text_value, &present); !err.empty())
      return {err, "INVALID_PARAMETER"};
    if (present) out->want_text = text_value;
  }
  if (!out->want_text) {
    // text=0 is the LAYOUT-ONLY run: RunFlags.text=false makes the unified
    // pipeline skip det/cls/rec and run just the layout model (or, on
    // /ocr/pdf with images=inline, skip inference entirely for a fast
    // pdf->page-images path). The pre-seam CUDA server supported this and the
    // unified pipeline briefly rejected it as unimplemented; it is implemented
    // again, IDENTICALLY on every build — the old `#ifdef USE_CPU_ONLY` split
    // (GPU accepts, CPU refuses) is exactly the per-build divergence a
    // transport- and build-free header must not carry.
    //
    // The combination rules run FIRST, so a caller who also set tables=1 gets
    // told which flag actually conflicts rather than a generic answer:
    // everything text-derived is meaningless without rec, so fail loud
    // instead of returning silently-empty tables/blocks/order.
    //
    // gRPC spells this request `layout_only`, and its own copy of these rules
    // rejected the same four flags with one flat message. Sharing them means the
    // caller now gets the specific reason, and a new text-derived flag cannot be
    // added to one transport's list and forgotten in the other's.
    if (out->want_tables || out->want_formulas)
      return {"text=0 runs the layout model only; tables=1/formulas=1 need "
              "the OCR pass. Drop text=0 or the structure flags.",
              "INVALID_PARAMETER"};
    if (out->want_blocks)
      return {"text=0 cannot be combined with as_blocks=1 (blocks aggregate "
              "recognized text)", "INVALID_PARAMETER"};
    if (out->want_reading_order)
      return {"text=0 cannot be combined with reading_order=1 (order is "
              "computed over recognized text)", "INVALID_PARAMETER"};
    // (a layout=1 request without the model was already rejected by the single
    // availability gate above — including this text=0 path)
    // Without layout the response would be empty on the image routes. On
    // /ocr/pdf (allow_image_only) the route re-checks against images=inline.
    if (!out->want_layout && !allow_image_only)
      return {"text=0 without layout=1 returns nothing on this endpoint; add "
              "layout=1 (layout-only run), or use /ocr/pdf?text=0&images=inline "
              "for page images", "INVALID_PARAMETER"};
  }

  return {};
}

// Fail loud when the client opted into a capability the server did not load.
//
// This is the SAME gate parse_options_core already applies, so validate_request
// does NOT call it again. It exists for the callers that MUTATE the requested
// mask after parsing — /ocr/pdf's markdown block defaults tables/formulas from
// availability — which must re-check before reaching the pipeline.
//
// `loaded` must reflect what the pipeline ACTUALLY loaded for the route DEFAULT
// (not the routing-name set): a per-request route_table/route_formula override
// is deliberately NOT treated as satisfying availability here, because
// `synth_from_env` always names the default formula route ("formula-env") even
// with no model loaded — honoring an override would let
// `formulas=1&route_formula=formula-env` slip past the gate and then return a
// silent empty result. To use tables/formulas the route default backend must be
// configured; an override only selects among loaded backends once this passes.
[[nodiscard]] inline ParseOptionsResult
check_structure_backends(const InferOptions &o,
                         const capability::CapabilityMask &loaded) {
  if (const auto missing = o.requested.without(loaded).first()) {
    const auto &info = capability::capability_info(*missing);
    return {std::format("{} is required for this request but {}. See GET "
                        "/capabilities for what this server supports.",
                        info.name, info.hint),
            std::string(info.error_code)};
  }
  return {};
}

// The /ocr/pdf document-output combination rules, transport-free.
//
// These are CONTRACT rules, not transport rules: whether fields=1+output=pdf is
// legal cannot depend on whether the client used HTTP or gRPC. They lived only
// in the HTTP parser, and the gRPC admit path set detect_page_fields /
// want_line_styles / want_movable_regions straight from the proto with no
// check — so over gRPC a client could request `fields+output=pdf` (run whole
// -page morphology + load the 77 MB field model, then discard the geometry) or
// `movable` without layout (lift nothing), silently. One checker, both
// transports, same messages the gRPC banner already promised.
[[nodiscard]] inline ParseOptionsResult check_pdf_doc_output_combinations(
    bool want_fields, bool want_editable, bool want_movable, bool want_markdown,
    bool want_searchable_pdf, bool want_text, bool want_layout) {
  if (want_fields && !want_text)
    return {"text=0 cannot be combined with fields=1 (field detection needs "
            "the text to label fields and to tell an empty box from a full "
            "one)", "INVALID_PARAMETER"};
  if (want_fields && want_markdown)
    return {"fields=1 cannot be combined with markdown=1 (the markdown "
            "response has nowhere to carry the field geometry)",
            "INVALID_PARAMETER"};
  if (want_fields && want_searchable_pdf)
    return {"fields=1 cannot be combined with output=pdf yet (the "
            "searchable-PDF writer does not stamp form fields); request the "
            "JSON envelope to get the field geometry", "INVALID_PARAMETER"};
  if (want_editable && !want_searchable_pdf)
    return {"editable=1 only applies to output=pdf (there is no document to "
            "rewrite when the response is the JSON envelope)",
            "INVALID_PARAMETER"};
  if (want_editable && !want_text)
    return {"text=0 cannot be combined with editable=1 (there is nothing to "
            "draw without recognised text)", "INVALID_PARAMETER"};
  if (want_movable && !want_searchable_pdf)
    return {"movable=1 only applies to output=pdf (there is no document to "
            "rewrite when the response is the JSON envelope)",
            "INVALID_PARAMETER"};
  if (want_movable && !want_layout)
    return {"movable=1 needs layout=1: the figures to lift out are the ones "
            "layout detection finds", "INVALID_PARAMETER"};
  return {};
}

} // namespace turbo_ocr::server
