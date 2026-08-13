#pragma once

// NVIDIA image decode — the device mechanics behind Backend::make_image_decoder().
//
// These two functions used to live in the deleted CUDA server bootstrap
// (src/cuda/server/stages_gpu.cpp). They are pure device mechanics, so they
// belong in the vendor arm: `Backend::make_image_decoder()`
// (include/turbo_ocr/backend/backend.h) is the seam slot for exactly this, and
// every other vendor already supplies its decoder from its own directory.

#include <cstddef>

#include "turbo_ocr/core/service_fns.h"   // server::ImageDecoder

namespace turbo_ocr::nvidia {

// True when nvJPEG initialised on this machine. Logs which path is in use.
[[nodiscard]] bool probe_nvjpeg();

// JPEG via nvJPEG (GPU) when available, everything else via the shared host
// fallback (decode::decode_cpu_fallback: Wuffs for PNG, cv::imdecode for the
// rest).
//
// NOTE — this returns a HOST cv::Mat, which is why the thread_local decoder in
// the .cpp is safe: it touches no pipeline state and is callable from any
// WorkPool thread. Device-RESIDENT decode is a different piece
// (IKernels::decode_image) and must not be routed through here.
[[nodiscard]] server::ImageDecoder make_nv_image_decoder(bool nvjpeg_available);

} // namespace turbo_ocr::nvidia
