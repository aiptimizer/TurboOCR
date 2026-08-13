#pragma once

// otsl_to_html — DEVICE-FREE declaring header.
//
// The converter is pure string processing (src/analysis/table/vlm/otsl_html.cpp) with
// zero device code, and it is linked into the CPU-only unified server. Its
// declaration used to live in nvidia/stages/vlm_table.h, which also
// declares the CUDA-typed VLMTableRecognizer and therefore includes
// <cuda_runtime.h>. That single coupling cost the tree three things:
//   * src/analysis/table/vlm/otsl_html.cpp needed a fake cuda_runtime.h on its include
//     path (src/server/compat/, wired per-source in unified_server.cmake);
//   * include/turbo_ocr/analysis/vlm/openai_policy.h re-declared the function BY HAND to
//     dodge the same problem — two declarations of one entity held in sync only
//     by a comment;
//   * the tests that cover it were confined to the CUDA-only configure, so the
//     converter the CPU server actually executes had no coverage there.
// One four-line header removes all three.

#include <string>

namespace turbo_ocr::table {

// OTSL-v1.0 string -> minimal HTML. Mirrors paddlex's `convert_otsl_to_html`
// closely enough that the scorer's TEDS metric sees the same tree shape.
// Tokens supported: <fcel>, <ecel>, <nl>, <lcel> (left-merge), <ucel>
// (up-merge), <xcel> (corner, both left+up merge).
std::string otsl_to_html(const std::string &otsl);

} // namespace turbo_ocr::table
