#pragma once

// The process-wide nvJPEG decoder pool: `NVJPEG_DECODERS` instances (default:
// one per pipeline replica), leased per decode by the JSON image routes
// (/ocr base64, /ocr/batch) and the gRPC byte paths that decode on work-pool
// threads. The raw-bytes route decodes GPU-direct on the replica that runs
// inference, so it keeps its per-replica decoder (GpuPipelinePool entry).
//
// Why a pool and not thread_local: each NvJpegDecoder holds ~190 MB of device
// memory and ~50 MB of host memory until process exit. One per work thread
// (128 threads by default) reaches ~24 GB of VRAM — enough to exhaust a
// 32 GB card and, on unified-memory hosts, to show up as multi-GB RSS growth
// that looks like a leak (GitHub #33).

#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

#include "turbo_ocr/decode/lease_pool.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/decode/nvjpeg_decoder_pool_fwd.h"

namespace turbo_ocr::decode {

// How long a decode waits for a free decoder before falling back to CPU
// decode. Decodes take milliseconds; this only trips when every decoder is
// held by a wedged caller, in which case a slower answer beats no answer.
inline constexpr std::chrono::milliseconds kNvJpegLeaseWait{2000};

// Open a pool of `capacity` decoders. The first one is constructed now to
// probe availability on the calling thread; the rest are constructed right
// after, so the pool never constructs a decoder inside a request (nvJPEG's
// per-decoder device buffers are still allocated on first use, so the VRAM
// footprint completes with the first `capacity` concurrent decodes, not at
// startup). Returns nullptr when nvJPEG cannot be initialised on this device
// (callers then decode on the CPU).
[[nodiscard]] inline std::unique_ptr<NvJpegDecoderPool>
open_nvjpeg_decoder_pool(size_t capacity) {
  auto pool = std::make_unique<NvJpegDecoderPool>(
      capacity, [] { return std::make_unique<NvJpegDecoder>(); });
  {
    auto probe = pool->try_acquire_for(std::chrono::seconds(30));
    if (!probe || !*probe || !(*probe)->available()) return nullptr;
  }  // the probe lease returns its decoder to the pool here
  {
    // Warm the remaining slots. Holding every lease at once forces one
    // construction per slot; dropping the vector returns them all.
    std::vector<NvJpegDecoderPool::Lease> warm;
    warm.reserve(pool->capacity());
    for (size_t i = 0; i < pool->capacity(); ++i) {
      auto l = pool->try_acquire_for(std::chrono::seconds(30));
      if (!l || !*l) break;
      warm.push_back(std::move(*l));
    }
  }
  return pool;
}

} // namespace turbo_ocr::decode
