#include "nvidia/stages/slanext_table_recognizer.h"

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/analysis/table/slanext/slanext_paths.h"
#include "turbo_ocr/analysis/table/slanext/slanext_postprocess.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "nvidia/support/cuda_check.h"       // abort_on_sticky_cuda_fault
#include "nvidia/engine/onnx_to_trt.h"
#include "nvidia/stages/paddle_rec.h"  // per-cell crop OCR fill

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
  // Shared encoder resolution (env override, else the baked release path) —
  // one policy for the TRT and CPU loaders, see resolve_slanext_encoder.
  const std::string enc = table::resolve_slanext_encoder(
      env::env_or("TABLE_SLANEXT_ENCODER_ONNX", ""));
  if (enc.empty()) return false;
  const std::string dec = env::env_or("TABLE_SLANEXT_DECODER_BIN", slanext_default_decoder_bin(enc));
  const std::string dict = env::env_or("TABLE_SLANEXT_DICT", slanext_default_dict(enc));
  wired_ = build_enc(enc, dec, dict);
  if (!wired_) return false;

  if (env::env_present("TABLE_SLANEXT_WIRELESS_ENCODER_ONNX"))
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

  for (std::size_t ti = 0; ti < regions.size(); ++ti) {
   try {
    const Box &region = regions[ti];
    const StructureResult sr = wired_->infer(page, region, stream);

    // THE shared host postprocess (slanext_postprocess.h) — one copy of the
    // table policy for every backend; only the crop recognizer differs, and it
    // enters as a callable closing over this backend's page + stream.
    table::SlanextCellRecFn cell_fn;
    if (cell_rec_)
      cell_fn = [&](const std::vector<Box> &empty_cells) {
        return cell_rec_->run(page, empty_cells, stream);
      };
    router::TableResult tr =
        table::slanext_postprocess_region(sr, page_ocr, region, cell_fn);
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
