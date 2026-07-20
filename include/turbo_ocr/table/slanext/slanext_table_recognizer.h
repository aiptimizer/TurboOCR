#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/table/table_recognizer.h"
#include "turbo_ocr/table/slanext/slanext_enc_split.h"

namespace turbo_ocr::table {

// SLANet-Plus encoder-split table backend (default). Owns a single SLANet-Plus
// CNN encoder (the class/file keep the historical "Slanext" name). Reads its env
// knobs (TABLE_SLANEXT_ENCODER_ONNX + decoder/dict defaults next to it) in
// load(). Per region: TRT FP16 encoder + host GRU decode -> structure tokens +
// per-cell quads; cells filled from the page text-OCR via match_cells_to_ocr +
// reconstruct_html.
class SlanextTableRecognizer final : public ITableRecognizer {
public:
  [[nodiscard]] bool load() override;
  [[nodiscard]] std::vector<router::TableResult>
  run(const GpuImage &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> &page_ocr, cudaStream_t stream) override;
  [[nodiscard]] bool is_ready() const noexcept override { return wired_ != nullptr; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "slanext";
  }
  void set_cell_recognizer(recognition::PaddleRec *r) noexcept override {
    cell_rec_ = r;
  }

private:
  std::unique_ptr<SlanextEncSplit> wired_;
  recognition::PaddleRec          *cell_rec_ = nullptr;  // not owned; per-cell fill
};

} // namespace turbo_ocr::table
