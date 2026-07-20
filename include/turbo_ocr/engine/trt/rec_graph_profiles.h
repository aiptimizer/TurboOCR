#pragma once

namespace turbo_ocr::engine {

// Static (batch, width) shapes that get a dedicated TRT optimization profile
// (index = table index + 1; profile 0 stays the dynamic fallback) and a baked
// CUDA graph. Shared between the engine builder and PaddleRec so profile
// indices can never drift. Weighted toward narrow widths where most text
// lines land; batches that fill a bucket poorly run exact-size on the
// dynamic profile instead, so padding waste stays bounded (see
// PaddleRec::infer_bucket fill rule).
struct RecGraphProfile {
  int batch;
  int width;
};

inline constexpr RecGraphProfile kRecGraphProfiles[] = {
    {8, 320},  {16, 320}, {24, 320}, {32, 320},
    {8, 480},  {16, 480}, {24, 480}, {32, 480},
    {8, 800},  {16, 800},
    {8, 1600},
    {8, 4000},
};
inline constexpr int kNumRecGraphProfiles =
    static_cast<int>(sizeof(kRecGraphProfiles) / sizeof(kRecGraphProfiles[0]));

} // namespace turbo_ocr::engine
