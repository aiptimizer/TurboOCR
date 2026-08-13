#pragma once

// nv_table_bridge.h — the NEUTRAL boundary between the new and old table
// interfaces. It includes NEITHER backend/table_recognizer.h (new,
// turbo_ocr::backend::ITableRecognizer) NOR nvidia/stages/slanext_table_recognizer.h
// (old, turbo_ocr::table::ITableRecognizer). The namespaces are DISTINCT now
// (the new interfaces moved to backend:: to end a linker-level ODR collision),
// so the reason for this bridge is no longer a name clash — it is that the OLD
// header drags <cuda_runtime.h> and GpuImage into any TU that includes it,
// which the new-world TU must stay free of. The bridge therefore speaks only
// in types that are shared and CUDA-free — turbo_ocr::Box, OCRResultItem,
// router::TableResult — plus a POD image descriptor and a void* stream.
//
// Two TUs implement/consume this:
//   nv_table_recognizer.cpp        (new headers) -> new backend::ITableRecognizer,
//                                   forwards to NvTableImpl
//   nv_table_recognizer_impl.cpp   (old headers) -> NvTableImpl over the proven
//                                   table::SlanextTableRecognizer
// This is the standard pimpl-across-a-generation-gap pattern; it lets us WRAP
// the existing SLANeXt encoder-split + host GRU decode with zero re-derivation.

#include <memory>
#include <vector>

#include "nvidia/support/nv_image_pod.h"             // GpuImagePod
#include "turbo_ocr/base/geometry/box.h"          // turbo_ocr::Box
#include "turbo_ocr/core/types.h"                 // turbo_ocr::OCRResultItem
#include "turbo_ocr/core/router_types.h" // router::TableResult

namespace turbo_ocr::nvidia {

// Opaque wrapper the impl TU defines; the interface TU only forwards to it.
class NvTableImpl {
public:
  virtual ~NvTableImpl() = default;
  [[nodiscard]] virtual bool load() = 0;
  [[nodiscard]] virtual std::vector<router::TableResult>
  run(const GpuImagePod &page, const std::vector<turbo_ocr::Box> &regions,
      const std::vector<turbo_ocr::OCRResultItem> &page_ocr, void *stream) = 0;
  // paddle_rec is a recognition::PaddleRec* passed as void* (the concrete cell
  // recognizer the SLANeXt wrapper still expects).
  virtual void set_cell_recognizer(void *paddle_rec) = 0;
  [[nodiscard]] virtual bool is_ready() const = 0;
};

// Defined in nv_table_recognizer_impl.cpp (old headers).
std::unique_ptr<NvTableImpl> make_nv_table_impl();

} // namespace turbo_ocr::nvidia
