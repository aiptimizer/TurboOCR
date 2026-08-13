#pragma once

// openai_policy — the ONE definition of everything an OpenAI-compatible VLM
// endpoint does that is NOT device work.
//
// WHY THIS FILE EXISTS. There are two OpenAI endpoint classes in the tree and
// there have to be, because they implement two different recognizer seams:
//
//   turbo_ocr::vlm::OpenAIEndpoint       (src/backends/nvidia/stages/openai_endpoint.cpp)
//       the OLD CUDA-typed formula::IFormulaRecognizer / table::ITableRecognizer
//       — GpuImage + cudaStream_t; only exists in a CUDA build.
//   turbo_ocr::vlm::BackendOpenAIEndpoint (src/pipeline/unified/vlm_factory.cpp)
//       the NEW backend::IFormulaRecognizer / backend::ITableRecognizer
//       — backend::ImageView + backend::DeviceQueue; exists in every build.
//
// The discriminator is WHICH SEAM each wears, which is why the newer one is
// named for `turbo_ocr::backend`. It was briefly called RemoteOpenAIEndpoint —
// but both are remote (same HTTP endpoint, same policy), and vlm_factory.cpp
// separately uses Remote/Local for the kind==Openai vs kind==Local distinction,
// so "Remote" in a class name two hundred lines from "REMOTE branch" in a
// comment meant two different things.
//
// What did NOT have to be duplicated is the behaviour: URL trimming, the
// /v1/models probe and served-model resolution, the ready_ policy, the response
// parser dispatch, the crop rectangle math, the pool submission, and the
// score-0.0 / hit_eos-false-on-failure result semantics are all pure host work
// on a BackendSpec and a byte buffer. Forking them produced exactly the drift
// this codebase's dedup rule exists to prevent: the crop loop was written twice
// inside openai_endpoint.cpp alone (submit_crops_async and infer_crops), which
// the newer copy had already fixed by centralising it in for_each_crop.
//
// So: this header owns the policy, both classes call it, and the ONLY thing
// each class still implements itself is materialising a host-addressable page —
// GpuImage + cudaMemcpyAsync on one side, ImageView + the registered device
// readback on the other. That step is expressed as a caller-supplied callable
// returning a HostPage, which is why nothing here includes <cuda_runtime.h> or
// any backend/ header: this file must compile in a CPU-only build and inside
// the CUDA arm alike, so it can never name a device type.
//
// HEADER-ONLY ON PURPOSE. Only these two translation units include it, and both
// already pull in every dependency below (curl via crop_pool.h, nlohmann/json,
// the parsers), so nothing pays for it that was not paying already. The only
// source list either endpoint is compiled from is CMakeLists.txt
// (src/backends/nvidia/stages/openai_endpoint.cpp and src/pipeline/unified/vlm_factory.cpp, both inside the
// turbo_ocr_pipeline archive that src/service/server/unified/unified_server.cmake merely links),
// so at this size a separate TU would be churn with no upside.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "turbo_ocr/backend/routing_config.h" // BackendSpec, Parser
#include "turbo_ocr/base/geometry/box.h"            // Box, aabb
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/base/string_utils.h"     // turbo_ocr::trim
#include "turbo_ocr/analysis/formula/latex_extract.h"   // formula::extract_latex
#include "turbo_ocr/core/router_types.h"     // router::TableResult
#include "turbo_ocr/analysis/table/html_reconstruct.h"  // table::sanitize_table_html
#include "turbo_ocr/analysis/table/vlm/otsl.h"          // table::otsl_to_html (device-free)
#include "turbo_ocr/analysis/vlm/crop_pool.h"           // vlm::VLMCropPool, vlm::CropOutcome
#include "turbo_ocr/analysis/vlm/vlm_client.h"          // vlm::http_get, vlm::encode_png_bgr

namespace turbo_ocr::vlm::openai_policy {

// ---------------------------------------------------------------------------
// Page handle: whatever the caller's device-specific step produced.
// ---------------------------------------------------------------------------
//
// `pixels` points at `rows * step` contiguous, host-readable, interleaved-BGR
// bytes — the SAME contract GpuImage/ImageView carry, minus the device identity.
// A default-constructed HostPage means "this page could not be reached from the
// host": the caller declines the request instead of dereferencing device memory.
struct HostPage {
  const std::uint8_t *pixels = nullptr;
  int cols = 0;
  int rows = 0;
  std::size_t step = 0;

  [[nodiscard]] bool empty() const noexcept {
    return pixels == nullptr || rows == 0 || cols == 0;
  }
};

// One parsed crop plus whether its endpoint call actually succeeded. `ok` is
// false on a failed/timed-out/exhausted-retry transport so the result builders
// can stamp a not-confident result instead of a false-positive empty.
struct CropOut {
  std::string text;
  bool ok = false;
};

// ---------------------------------------------------------------------------
// Response parsing.
// ---------------------------------------------------------------------------

// Mirror vlm_table.cpp::otsl_or_html: a model that already returns HTML
// (`<table…`) is passed through untouched; only OTSL is converted. Without
// this, otsl_to_html would mangle a valid-HTML response from a kind:openai model.
[[nodiscard]] inline std::string otsl_or_html(const std::string &raw) {
  std::string t = turbo_ocr::trim(raw);
  // Sanitize model-emitted HTML passthrough (adversarial-image XSS vector).
  if (t.rfind("<table", 0) == 0) return table::sanitize_table_html(raw);
  return raw.empty() ? "" : table::otsl_to_html(raw);
}

// Stateless dispatch keyed on the parser enum only — no recognizer state — so
// both the sync path and the self-contained async_result_parser() snapshot
// share it.
[[nodiscard]] inline std::string parse_with(backend_routing::Parser parser,
                                            const std::string &raw) {
  switch (parser) {
  case backend_routing::Parser::Otsl:
    return otsl_or_html(raw);
  case backend_routing::Parser::Latex:
    return formula::extract_latex(raw);
  case backend_routing::Parser::Text:
    return turbo_ocr::trim(raw);
  case backend_routing::Parser::Raw:
  default:
    return raw;
  }
}

// ---------------------------------------------------------------------------
// Health check.
// ---------------------------------------------------------------------------

// GET <base_url>/v1/models; resolves the served-model-name when unset. Mutates
// `spec` (trailing slashes stripped from base_url, model filled in) and sets
// `ready` — both endpoint classes hold exactly these two pieces of state, so
// passing them in keeps the policy free of any class it belongs to.
// Returns false (=> caller disables the modality cleanly) if unreachable.
inline bool health_check(backend_routing::BackendSpec &spec, bool &ready) {
  while (!spec.base_url.empty() && spec.base_url.back() == '/')
    spec.base_url.pop_back();
  HttpResp models = http_get(spec.base_url + "/v1/models", 5, spec.api_key);
  if (!models.ok) {
    TOCR_LOG_ERROR("OpenAIEndpoint /v1/models unreachable, backend disabled",
                   "base_url", spec.base_url, "backend", spec.name);
    ready = false;
    return false;
  }
  if (spec.model.empty()) {
    try {
      auto j = nlohmann::json::parse(models.body);
      if (j.contains("data") && j["data"].is_array() && !j["data"].empty())
        spec.model = j["data"][0].value("id", "");
    } catch (const std::exception &e) {
      TOCR_LOG_WARN("OpenAIEndpoint /v1/models parse warning", "error", e.what());
    }
  }
  ready = !spec.model.empty();
  TOCR_LOG_INFO("OpenAIEndpoint ready", "base_url", spec.base_url,
                "model", spec.model,
                "parser", backend_routing::parser_name(spec.parser));
  return ready;
}

// ---------------------------------------------------------------------------
// Crop geometry.
// ---------------------------------------------------------------------------

// PNG-encode every box out of a host-addressable page and hand the bytes to
// `submit`. Shared by the sync and async paths of BOTH endpoints so the crop
// rectangle can never drift between the four call sites it used to be spelled
// out in.
//
// The clamps are load-bearing, not defensive noise: a layout box may legally
// extend past the page (the detector works on a padded canvas), and an empty
// or inverted rect would make encode_png_bgr read out of bounds. The math lives
// in turbo_ocr::clamped_crop_rect (common/geometry/box.h) — the project's
// designated single home for it, already used by the nine other VLM/formula/
// table crop sites — so this file does not spell it a third time.
template <typename Submit>
inline void for_each_crop(const HostPage &page, const std::vector<Box> &boxes,
                          Submit &&submit) {
  for (const Box &b : boxes) {
    const auto cr = turbo_ocr::clamped_crop_rect(b, page.cols, page.rows);
    const int x0 = cr[0], y0 = cr[1], w = cr[2], h = cr[3];
    const std::uint8_t *src = page.pixels +
                              static_cast<std::size_t>(y0) * page.step +
                              static_cast<std::size_t>(x0) * 3;
    submit(encode_png_bgr(src, w, h, static_cast<int>(page.step)));
  }
}

// ---------------------------------------------------------------------------
// The two request shapes.
// ---------------------------------------------------------------------------
//
// `materialise` is the ONE device-specific step, supplied by the caller:
//   HostPage materialise();
// It performs the D2H (or the zero-copy unified-memory shortcut) and returns an
// empty HostPage when the page cannot be reached. It is called at most once,
// and only after the cheap guards below have already declined — so an empty
// box list or a not-ready endpoint still costs no copy.

// Crop, PNG-encode, submit to the global pool, and return one RAW-response
// future per box (NO await, NO parse) — the async primitive.
template <typename Materialise>
[[nodiscard]] inline std::vector<std::future<std::string>>
submit_crops_async(const backend_routing::BackendSpec &spec, bool ready,
                   const std::vector<Box> &boxes, Materialise &&materialise) {
  std::vector<std::future<std::string>> futs;
  if (boxes.empty() || !ready) return futs;

  // Read the page back once on the device worker; PNG bytes are COPIED into the
  // pool at submit, so the futures outlive whatever buffer `materialise` owns —
  // it is safe to free the moment this function returns.
  const HostPage page = materialise();
  if (page.empty()) return futs;

  auto &pool = VLMCropPool::instance();
  futs.reserve(boxes.size());
  for_each_crop(page, boxes, [&](std::vector<std::uint8_t> png) {
    futs.push_back(pool.submit(std::move(png), spec.prompt, spec.model,
                               spec.max_tokens, spec.timeout_s, spec.base_url,
                               spec.api_key));
  });
  return futs;
}

// The sync path: submit_crops_async + await + parse, over the status-carrying
// pool entry point so a failed transport stays distinguishable from a clean
// empty response.
template <typename Materialise>
[[nodiscard]] inline std::vector<CropOut>
infer_crops(const backend_routing::BackendSpec &spec, bool ready,
            const std::vector<Box> &boxes, Materialise &&materialise) {
  std::vector<CropOut> out(boxes.size());
  if (boxes.empty() || !ready) return out;

  const HostPage page = materialise();
  if (page.empty()) return out;

  auto &pool = VLMCropPool::instance();
  std::vector<std::future<CropOutcome>> futs;
  futs.reserve(boxes.size());
  for_each_crop(page, boxes, [&](std::vector<std::uint8_t> png) {
    futs.push_back(pool.submit_with_status(std::move(png), spec.prompt,
                                           spec.model, spec.max_tokens,
                                           spec.timeout_s, spec.base_url,
                                           spec.api_key));
  });

  for (std::size_t i = 0; i < futs.size() && i < out.size(); ++i) {
    CropOutcome o = futs[i].get();
    // Don't parse a failed transport — the empty string is meaningless and
    // would otherwise look like a clean (confident) empty result downstream.
    out[i].ok = o.ok;
    out[i].text = o.ok ? parse_with(spec.parser, o.text) : std::string{};
  }
  return out;
}

// ---------------------------------------------------------------------------
// Result conversion.
// ---------------------------------------------------------------------------
//
// The two seams carry DIFFERENT result structs — formula::FormulaEngineResult
// (old) and backend::FormulaEngineResult (new) — with identical layout. The
// conversion is templated on the target rather than duplicated, so the ok-flag
// semantics below have one home; naming either type here would drag a seam
// header (and, on the old side, <cuda_runtime.h>) into this file.

// {latex, token_count, hit_eos}: hit_eos reflects a real completion, so it is
// false on a failed/empty endpoint call — a non-confident result must never be
// stamped as a clean stop. token_count is 0 because a remote endpoint does not
// report one.
//
// DESIGNATED INITIALISERS, not positional. This template is instantiated with
// TWO structs maintained in two different files (turbo_ocr::formula::
// FormulaEngineResult and turbo_ocr::backend::FormulaEngineResult), both with
// FOUR members {latex, token_count, hit_eos, ok}. A 3-element positional brace
// init silently starts writing the wrong slot the moment a field is inserted
// before hit_eos in either header, and nothing would catch it.
//
// `ok` is deliberately left at its default true, matching the pre-dedup
// behaviour of both endpoints (git show HEAD:src/backends/nvidia/stages/openai_endpoint.cpp).
// Setting it from c.ok would be a behaviour change with a real consequence:
// src/backends/nvidia/stages/auto_cjk_formula.cpp gates its CJK re-run on
// `res[i].ok && ...`, so clearing it would suppress the CJK fallback for exactly
// the crops that failed. Consumers currently mask the divergence with
// `!ok || latex.empty()` (unified_pipeline_dispatch.cpp).
template <typename FormulaResult>
[[nodiscard]] inline std::vector<FormulaResult>
to_formula_results(std::vector<CropOut> crops) {
  std::vector<FormulaResult> out;
  out.reserve(crops.size());
  for (auto &c : crops)
    out.push_back(FormulaResult{.latex = std::move(c.text),
                                .token_count = 0,
                                .hit_eos = c.ok});
  return out;
}

// layout_id is -1 (a remote endpoint answers per REGION, not per layout element)
// and score is 0.0 on a failed/empty endpoint call so downstream can tell a
// failure from a genuine empty table (default-confident is 1.0).
//
// BY VALUE, not by non-const reference: both builders MOVE the text out of every
// CropOut, so they consume their argument. A `std::vector<CropOut>&` advertises
// "I may modify" and makes a second call — or any read afterwards — silently
// yield empty strings. Every caller passes a dying local, so this costs nothing
// and makes the destruction visible at the call site.
[[nodiscard]] inline std::vector<router::TableResult>
to_table_results(std::vector<CropOut> crops, const std::vector<Box> &regions) {
  std::vector<router::TableResult> out;
  out.reserve(crops.size());
  for (std::size_t i = 0; i < crops.size(); ++i) {
    router::TableResult t;
    t.layout_id = -1;
    t.html = std::move(crops[i].text);
    t.score = crops[i].ok ? 1.0f : 0.0f;
    if (i < regions.size()) t.box = regions[i];
    out.push_back(std::move(t));
  }
  return out;
}

} // namespace turbo_ocr::vlm::openai_policy
