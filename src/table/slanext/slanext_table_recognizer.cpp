#include "turbo_ocr/table/slanext/slanext_table_recognizer.h"

#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/table/slanext/slanext_paths.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "turbo_ocr/common/cuda/cuda_check.h"       // abort_on_sticky_cuda_fault
#include "turbo_ocr/engine/trt/onnx_to_trt.h"
#include "turbo_ocr/recognition/paddle_rec.h"  // per-cell crop OCR fill
#include "turbo_ocr/table/cell_matcher.h"      // OcrLine, match_cells_to_ocr
#include "turbo_ocr/table/html_reconstruct.h"  // reconstruct_html

namespace turbo_ocr::table {

namespace {
std::unique_ptr<SlanextEncSplit> build_enc(const std::string &enc_onnx,
                                           const std::string &dec,
                                           const std::string &dict) {
  const std::string trt = engine::ensure_trt_engine(enc_onnx, "slanext_encoder");
  if (trt.empty()) {
    std::cerr << "[slanext] encoder engine build failed: " << enc_onnx << '\n';
    return nullptr;
  }
  auto s = std::make_unique<SlanextEncSplit>();
  if (!s->load_model(trt, dec, dict)) {
    std::cerr << "[slanext] load_model failed: " << enc_onnx << '\n';
    return nullptr;
  }
  return s;
}
} // namespace

bool SlanextTableRecognizer::load() {
  // Auto-resolve the encoder shipped by the release bundle / Docker image so
  // TABLE_BACKEND=slanext works out of the box; an explicit env still wins.
  static constexpr const char *kDefaultEncoder =
      "models/table/slanext_encoder/SLANeXt_wired_encoder.onnx";
  std::string enc = env::env_or("TABLE_SLANEXT_ENCODER_ONNX", "");
  if (enc.empty()) {
    if (std::filesystem::exists(kDefaultEncoder)) {
      enc = kDefaultEncoder;
    } else {
      std::cerr << "[slanext] table encoder not found: set "
                   "TABLE_SLANEXT_ENCODER_ONNX (the release ships it at "
                << kDefaultEncoder << ")\n";
      return false;
    }
  }
  const std::string dec = env::env_or("TABLE_SLANEXT_DECODER_BIN", slanext_default_decoder_bin(enc));
  const std::string dict = env::env_or("TABLE_SLANEXT_DICT", slanext_default_dict(enc));
  wired_ = build_enc(enc, dec, dict);
  if (!wired_) return false;

  if (std::getenv("TABLE_SLANEXT_WIRELESS_ENCODER_ONNX"))
    std::cerr << "[slanext] wired/wireless routing removed — ignoring "
                 "TABLE_SLANEXT_WIRELESS_ENCODER_ONNX\n";

  std::cout << "[Pipeline] Table backend=slanext (wired, TRT FP16 encoder + host decode)\n";
  return true;
}

std::vector<router::TableResult>
SlanextTableRecognizer::run(const GpuImage &page, const std::vector<Box> &regions,
                            const std::vector<OCRResultItem> &page_ocr,
                            cudaStream_t stream) {
  std::vector<router::TableResult> out;
  out.reserve(regions.size());

  // Per-region scratch hoisted out of the loop: cleared (capacity retained)
  // each region so recognizing N tables on one page reuses a single set of
  // buffers instead of re-allocating per region. region_ocr and region_texts
  // stay index-aligned — the matcher returns indices into region_ocr and
  // reconstruct_html reads the same indices out of region_texts.
  std::vector<OcrLine>            region_ocr;
  std::vector<std::string>        region_texts;
  std::vector<std::array<int, 8>> quads;
  std::vector<Box>                empty_boxes;
  std::vector<std::size_t>        empty_ci;
  region_ocr.reserve(page_ocr.size());
  region_texts.reserve(page_ocr.size());

  for (std::size_t ti = 0; ti < regions.size(); ++ti) {
   try {
    region_ocr.clear();
    region_texts.clear();
    quads.clear();
    empty_boxes.clear();
    empty_ci.clear();

    const Box &region = regions[ti];
    // Crop origin the region-local cell quads are shifted back by; must equal
    // the (page top-left–clamped) origin infer() crops at. Same aabb() helper
    // infer() uses, so the two stay in lockstep by construction.
    const auto rb = aabb(region);
    const int rx = std::max(rb[0], 0), ry = std::max(rb[1], 0);
    const int rax2 = rb[2], ray2 = rb[3];

    const StructureResult sr = wired_->infer(page, region, stream);

    // Cells from the page text-OCR: geometry-match each structure-order cell
    // quad to OCR lines inside the region, then reconstruct_html substitutes.
    // The matcher reads only OcrLine.bbox, so each in-region string is copied
    // once into region_texts (its index pool); region_ocr carries geometry only.
    for (const auto &r : page_ocr) {
      int bx1 = INT_MAX, by1 = INT_MAX, bx2 = INT_MIN, by2 = INT_MIN, cx = 0, cy = 0;
      for (const auto &p : r.box.pts) {
        cx += p[0]; cy += p[1];
        bx1 = std::min(bx1, p[0]); by1 = std::min(by1, p[1]);
        bx2 = std::max(bx2, p[0]); by2 = std::max(by2, p[1]);
      }
      cx /= 4; cy /= 4;
      if (cx < rx || cx > rax2 || cy < ry || cy > ray2) continue;
      region_ocr.push_back(OcrLine{{static_cast<float>(bx1), static_cast<float>(by1),
                                    static_cast<float>(bx2), static_cast<float>(by2)},
                                   {}});
      region_texts.push_back(r.text);
    }
    // SLANeXt per-token cell quads are region-local -> shift to page coords.
    quads.reserve(sr.cells.size());
    for (const auto &c : sr.cells) {
      std::array<int, 8> q = c.bbox;
      for (int k = 0; k < 4; ++k) { q[2 * k] += rx; q[2 * k + 1] += ry; }
      quads.push_back(q);
    }
    auto matched = match_cells_to_ocr(quads, region_ocr);

    // Per-cell crop OCR: SLANeXt grid cells with no page-OCR line get recognized
    // directly from their quad crop. Recovers dense-grid cells the page text
    // detector under-segmented (the dominant local-table content loss). One
    // batched rec call per table over all empty cells; page-coordinate quads.
    if (cell_rec_ && !matched.empty()) {
      for (std::size_t ci = 0; ci < matched.size() && ci < quads.size(); ++ci) {
        if (!matched[ci].ocr_indices.empty()) continue;
        const auto &q = quads[ci];
        const int w = std::abs(q[2] - q[0]);
        const int h = std::abs(q[5] - q[1]);
        if (w < 4 || h < 4) continue;  // skip degenerate / spacer cells
        empty_boxes.push_back(
            Box{{{{q[0], q[1]}, {q[2], q[3]}, {q[4], q[5]}, {q[6], q[7]}}}});
        empty_ci.push_back(ci);
      }
      if (!empty_boxes.empty()) {
        auto rec = cell_rec_->run(page, empty_boxes, stream);
        for (std::size_t k = 0; k < rec.size() && k < empty_ci.size(); ++k) {
          const std::string &txt = rec[k].first;
          if (txt.empty() || rec[k].second < 0.5f) continue;  // drop empty / low-conf noise
          region_texts.push_back(txt);
          matched[empty_ci[k]].ocr_indices.push_back(region_texts.size() - 1);
        }
      }
    }

    router::TableResult tr;
    tr.layout_id = -1;  // stamped by caller (input order)
    tr.html      = reconstruct_html(sr.structure, matched, region_texts);
    tr.score     = sr.structure_score;
    tr.box       = region;
    out.push_back(std::move(tr));
   } catch (const std::exception &e) {
    // Per-region degrade: a CUDA/inference fault on ONE table region must not
    // abort the whole page (every other table lost) — mirror the graceful
    // "table region DROPPED" path inside infer(). Sticky faults still exit the
    // process; a recoverable error degrades just this region (empty html ->
    // counted degraded upstream) and we continue. One TableResult per region is
    // pushed unconditionally so the caller's layout_id stamping stays aligned.
    turbo_ocr::abort_on_sticky_cuda_fault("slanext_table_recognizer region");
    cudaGetLastError();  // clear the recoverable error before the next region
    std::cerr << "[slanext] table region " << ti << " FAILED (" << e.what()
              << ") — region DROPPED, continuing\n";
    router::TableResult tr;
    tr.layout_id = -1;
    tr.box       = regions[ti];  // html left empty -> degraded accounting
    out.push_back(std::move(tr));
   }
  }
  return out;
}

} // namespace turbo_ocr::table
