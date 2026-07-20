#pragma once

#include <cctype>
#include <format>
#include <string>

#include <drogon/HttpRequest.h>

#include "turbo_ocr/backend_routing/routing_config.h"

namespace turbo_ocr::server {

/// Per-request feature flags parsed from query parameters.
struct InferOptions {
  bool want_layout = false;
  bool want_reading_order = false;
  // ?tables=1 / ?formulas=1 — strict opt-in. Even when a table/formula backend
  // is configured at startup, the stage runs ONLY when the request asks for it.
  // Both imply layout (recognition runs on layout-detected regions), so either
  // auto-enables want_layout. Default false: layout alone never triggers them.
  bool want_tables = false;
  bool want_formulas = false;
  // ?as_blocks=1 — emit a `blocks` array (paragraph-level aggregate,
  // one entry per non-empty layout cell, mirrors PaddleX's
  // PP-StructureV3 parsing_res_list granularity). Auto-enables layout
  // and reading_order since aggregation needs both.
  bool want_blocks = false;
  // ?text=0 — skip text detection/recognition entirely and run ONLY the
  // layout model (auto-enables layout; `results` comes back empty). Several
  // times cheaper than a full OCR pass. Incompatible with tables/formulas/
  // blocks/reading_order — they all consume recognized text. GPU build only.
  bool want_text = true;

  // Per-request routing override (Tier-A): a backend NAME per modality (empty
  // == use the configured route default). Parsed from /ocr/raw query params
  // (?route_table=/?route_formula=) and /ocr JSON body (routing{}). Validated
  // against the registry name-set at the route layer (unknown => 400) before
  // it reaches the pipeline. Rides the by-value `opts` capture into the
  // dispatcher lambda, so it's timeout-safe like the other flags.
  backend_routing::RequestRouting routing_override;
};

// Parse a generic boolean query param ("1"/"true"/"on"/"yes" etc.).
// Returns empty string on success and writes to *out; otherwise returns an
// error message. When the parameter is absent, *out is set to false and an
// empty string is returned.
[[nodiscard]] inline std::string parse_bool_query(const drogon::HttpRequestPtr &req,
                                                   const char *key,
                                                   bool *out) {
  *out = false;
  auto v = req->getParameter(key);
  if (v.empty()) return {};
  std::string s(v);
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "1" || s == "true" || s == "on" || s == "yes")       { *out = true; return {}; }
  if (s == "0" || s == "false" || s == "off" || s == "no")      { *out = false; return {}; }
  return std::format("Invalid {} param: '{}' "
                     "(expected 0/1, true/false, on/off, or yes/no)",
                     key, s);
}

// Parse the full set of opt-in query parameters for inference routes.
// `layout` and `reading_order` both default to 0; either being set to 1
// without the underlying layout model causes a 400 with a descriptive
// error code (LAYOUT_DISABLED).
struct ParseOptionsResult {
  std::string error;     // empty on success
  std::string error_code; // populated when error is non-empty
};
[[nodiscard]] inline ParseOptionsResult
parse_query_options(const drogon::HttpRequestPtr &req,
                    bool layout_available,
                    InferOptions *out,
                    bool allow_image_only = false) {
  *out = {};
  if (auto err = parse_bool_query(req, "layout", &out->want_layout);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (out->want_layout && !layout_available) {
    // One stable code for one condition: every "layout feature
    // unavailable" rejection (layout=1, reading_order=1, as_blocks=1)
    // returns LAYOUT_DISABLED — the code docs/api/http.md documents.
    // Malformed values stay INVALID_PARAMETER.
    return {"Layout requested but the layout model is not loaded. "
            "Either models/layout/layout.onnx is missing from the "
            "image, or the server was started with DISABLE_LAYOUT=1.",
            "LAYOUT_DISABLED"};
  }

  if (auto err = parse_bool_query(req, "reading_order",
                                   &out->want_reading_order);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (out->want_reading_order && !layout_available) {
    // Reading order is derived from layout boxes — without the model
    // there's nothing to derive from. Reject the request explicitly so
    // clients don't silently get the y/x fallback they didn't ask for.
    return {"reading_order=1 requires the layout model: start the server "
            "without DISABLE_LAYOUT=1 (layout is on by default)",
            "LAYOUT_DISABLED"};
  }
  if (out->want_reading_order && !out->want_layout) {
    // Reading order auto-enables layout so /ocr behaves as documented:
    // ?reading_order=1 alone yields a populated reading_order array.
    out->want_layout = true;
  }

  if (auto err = parse_bool_query(req, "as_blocks", &out->want_blocks);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (out->want_blocks && !layout_available) {
    return {"as_blocks=1 requires the layout model: start the server "
            "without DISABLE_LAYOUT=1 (layout is on by default)",
            "LAYOUT_DISABLED"};
  }
  if (out->want_blocks) {
    // Aggregation needs reading_order (and reading_order needs layout).
    out->want_reading_order = true;
    out->want_layout = true;
  }

  if (auto err = parse_bool_query(req, "tables", &out->want_tables);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (auto err = parse_bool_query(req, "formulas", &out->want_formulas);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  // Table/formula recognition runs on layout-detected regions, so either flag
  // implies layout. Auto-enable it (mirrors as_blocks) so ?tables=1 / ?formulas=1
  // work standalone; require the layout model to be present.
  if ((out->want_tables || out->want_formulas) && !layout_available) {
    return {"tables=1/formulas=1 require the layout model: start the server "
            "without DISABLE_LAYOUT=1 (layout is on by default)",
            "LAYOUT_DISABLED"};
  }
  if (out->want_tables || out->want_formulas)
    out->want_layout = true;

  // `text` is the one opt-OUT flag (default true); parse_bool_query's
  // absent->false convention is for opt-in flags, so only parse when present.
  out->want_text = true;
  if (!req->getParameter("text").empty()) {
    if (auto err = parse_bool_query(req, "text", &out->want_text);
        !err.empty())
      return {err, "INVALID_PARAMETER"};
  }
  if (!out->want_text) {
#ifdef USE_CPU_ONLY
    return {"text=0 (layout-only) is not supported on the CPU build",
            "INVALID_PARAMETER"};
#endif
    // Layout-only run: everything text-derived is meaningless without rec.
    // Fail loud on the combinations instead of returning silently-empty
    // tables/blocks/order.
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
    if (out->want_layout && !layout_available) {
      return {"text=0&layout=1 requests a layout-only run, which needs the "
              "layout model: start the server without DISABLE_LAYOUT=1",
              "LAYOUT_DISABLED"};
    }
    // Without layout the response would be empty on the image routes. On
    // /ocr/pdf (allow_image_only) the route re-checks against images=inline.
    if (!out->want_layout && !allow_image_only)
      return {"text=0 without layout=1 returns nothing on this endpoint; add "
              "layout=1 (layout-only run), or use /ocr/pdf?text=0&images=inline "
              "for page images", "INVALID_PARAMETER"};
  }

  return {};
}

// Fail loud when the client opted into a structure stage the server can't do:
// tables=1 / formulas=1 with no backend configured at startup. Returns
// TABLE_BACKEND_DISABLED / FORMULA_BACKEND_DISABLED (a 400) rather than the
// silent empty result a missing backend would otherwise produce — clients can
// discover availability up front via GET /capabilities. Call after
// parse_query_options at every route that honors the user's tables/formulas
// flags (NOT /ocr/markdown, which sets them best-effort).
// `table_available`/`formula_available` must reflect what the pipeline ACTUALLY
// loaded for the route DEFAULT (not the routing-name set): a per-request
// ?route_table=/?route_formula= override is deliberately NOT treated as
// satisfying availability here, because `synth_from_env` always names the
// default formula route ("formula-env") even with no model loaded — honoring an
// override would let `?formulas=1&route_formula=formula-env` slip past the gate
// and then return a silent empty result. To use tables/formulas the route
// default backend must be configured; an override only selects among loaded
// backends once the default gate passes.
[[nodiscard]] inline ParseOptionsResult
check_structure_backends(const InferOptions &o, bool table_available,
                         bool formula_available) {
  if (o.want_tables && !table_available)
    return {"tables=1 requested but no table backend is configured. Start the "
            "server with TABLE_BACKEND= + TABLE_SLANEXT_ENCODER_ONNX= (see GET "
            "/capabilities for what this server supports).",
            "TABLE_BACKEND_DISABLED"};
  if (o.want_formulas && !formula_available)
    return {"formulas=1 requested but no formula backend is configured. Start "
            "the server with FORMULA_ONNX= + FORMULA_TOKENIZER= (see GET "
            "/capabilities for what this server supports).",
            "FORMULA_BACKEND_DISABLED"};
  return {};
}

} // namespace turbo_ocr::server
