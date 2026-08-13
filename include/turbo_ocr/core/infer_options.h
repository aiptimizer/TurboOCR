#pragma once

// The parsed per-request feature flags, as a TRANSPORT-FREE leaf header.
//
// InferOptions is the only thing the pipeline needs from the validation layer:
// it is the value the route hands the pipeline entry points, and every pipeline
// TU that names it (make_infer_func, unified_ocr_pipeline, pdf_job, the gRPC
// service) used to pull in query_options.h — and with it <drogon/HttpRequest.h>
// and <json/json.h>, because the PARSERS live there. That made Drogon a
// compile-time dependency of code that never sees an HTTP request, which is
// exactly the coupling the layering is meant to forbid (and the reason a
// header-only Drogon bump recompiled the whole pipeline).
//
// So the struct lives here and the parsers stay in query_options.h, which
// includes this file. Every existing includer of query_options.h compiles
// unchanged; TUs that only need the struct include this instead.
//
// INCLUDES ARE LOAD-BEARING: capability/capability.h and
// backend_routing/routing_config.h are both std-only (<array>/<cstdint>/
// <optional>/<string_view> and <map>/<optional>/<set>/<stdexcept>/<string>
// respectively). Keep it that way — adding a transport, JSON or serialization
// header here silently re-couples the pipeline to the server.

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/core/capability.h"

namespace turbo_ocr::server {

/// Per-request feature flags parsed from the query string and/or JSON body.
struct InferOptions {
  // REQUESTED (see capability/capability.h) — the ONE parsed representation of
  // "what did the client ask for", filled by parse_query_options from the
  // capability table, so a new capability is honoured by every endpoint and
  // both request forms without touching any endpoint.
  //
  // WHICH want_* bools this mask owns, precisely: parse_query_options projects
  // want_layout / want_tables / want_formulas / want_autorotate OUT of it at the
  // end of parsing (query_options.h step 4), so for a parser-built InferOptions
  // those four must never be assigned independently. The remaining three —
  // want_reading_order, want_blocks, want_text — are parsed directly and have no
  // capability at all, so they are not projections of anything.
  //
  // WARNING for hand-built options. Several call sites construct an InferOptions
  // OUTSIDE the parser and set only the want_* bools (src/pipeline/job/pdf_job.cpp,
  // src/service/grpc/service_core.cpp). Their `requested` is EMPTY, and
  // check_structure_backends (query_options.h) reads ONLY `requested` — so such
  // an object passes that gate unconditionally. Do not hand a hand-built
  // InferOptions to check_structure_backends; gate it where it was parsed.
  capability::CapabilityMask requested;

  bool want_layout = false;
  bool want_reading_order = false;
  // ?tables=1 / ?formulas=1 — strict opt-in. Even when a table/formula backend
  // is configured at startup, the stage runs ONLY when the request asks for it.
  // Both imply layout (recognition runs on layout-detected regions), so either
  // auto-enables want_layout. Default false: layout alone never triggers them.
  bool want_tables = false;
  bool want_formulas = false;
  // ?autorotate=1 — de-rotate the page upright using the doc-orientation model
  // before OCR. Only the PDF path acts on it today; it is parsed here so the
  // flag is validated identically everywhere it is accepted.
  bool want_autorotate = false;
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

} // namespace turbo_ocr::server
