#pragma once

// Forward declarations so CUDA-free headers (route registrars, config) can
// name the decoder pool without pulling in nvjpeg.h.
namespace turbo_ocr::decode {
class NvJpegDecoder;
template <class T>
class LeasePool;
using NvJpegDecoderPool = LeasePool<NvJpegDecoder>;
} // namespace turbo_ocr::decode
