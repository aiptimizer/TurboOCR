#include "turbo_ocr/analysis/recognition/ctc_decode.h"

#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "turbo_ocr/base/env_utils.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define TURBO_CTC_HAVE_AVX2 1
#else
#define TURBO_CTC_HAVE_AVX2 0
#endif

namespace turbo_ocr::recognition {

namespace {
#if TURBO_CTC_HAVE_AVX2
// AVX2 argmax over a contiguous row of floats. Matches the scalar reference
// exactly, INCLUDING tie-breaking (lowest index wins on equal values): the
// per-lane compare uses strict >, so a lane keeps its lowest index on ties,
// and the 8-lane horizontal reduce also breaks ties toward the lower index.
inline std::pair<float, int> argmax_avx2(const float *row, int n) {
  int j = 0;
  __m256 vmax = _mm256_set1_ps(-3.402823e38f);
  __m256i vidx = _mm256_setzero_si256();
  __m256i vj = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256i v8 = _mm256_set1_epi32(8);
  for (; j + 8 <= n; j += 8) {
    __m256 v = _mm256_loadu_ps(row + j);
    __m256 gt = _mm256_cmp_ps(v, vmax, _CMP_GT_OQ);
    vmax = _mm256_max_ps(vmax, v);
    vidx = _mm256_blendv_epi8(vidx, vj, _mm256_castps_si256(gt));
    vj = _mm256_add_epi32(vj, v8);
  }
  alignas(32) float vals[8];
  alignas(32) int idxs[8];
  _mm256_store_ps(vals, vmax);
  _mm256_store_si256(reinterpret_cast<__m256i *>(idxs), vidx);
  float best = vals[0];
  int bi = idxs[0];
  for (int k = 1; k < 8; k++) {
    if (vals[k] > best || (vals[k] == best && idxs[k] < bi)) {
      best = vals[k];
      bi = idxs[k];
    }
  }
  for (; j < n; j++) {
    if (row[j] > best) {
      best = row[j];
      bi = j;
    }
  }
  return {best, bi};
}

inline bool simd_ctc_enabled() {
  static const bool e = env::env_enabled("SIMD_CTC");
  return e;
}
#endif // TURBO_CTC_HAVE_AVX2
} // namespace

std::pair<std::string, float>
ctc_greedy_decode(const int *indices, const float *scores, int seq_len,
                  const std::vector<std::string> &label_list) {
  std::string text;
  text.reserve(seq_len);
  float score = 0.0f;
  int count = 0;
  int last_index = -1;

  for (int i = 0; i < seq_len; i++) {
    int index = indices[i];
    if (index != last_index) {
      // index > 0, not != 0: this overload takes CALLER-supplied indices (the
      // device-argmax backends), and a negative — e.g. a sentinel from an
      // argmax over an all-NaN row — passed the old `!= 0 && < size` pair
      // straight into label_list[-1]. Index 0 is the CTC blank either way.
      if (index > 0 && index < static_cast<int>(label_list.size())) {
        text += label_list[index];
        score += scores[i];
        count++;
      }
    }
    last_index = index;
  }
  if (count > 0)
    score /= count;
  return {text, score};
}

std::pair<std::string, float>
ctc_greedy_decode_raw(const float *logits, int seq_len, int num_classes,
                      const std::vector<std::string> &label_list) {
  std::string text;
  text.reserve(seq_len);
  float score = 0.0f;
  int count = 0;
  int last_index = -1;

#if TURBO_CTC_HAVE_AVX2
  const bool use_simd = simd_ctc_enabled();
#endif
  for (int i = 0; i < seq_len; i++) {
    const float *row = logits + i * num_classes;
    int index = 0;
    float max_val = row[0];
#if TURBO_CTC_HAVE_AVX2
    if (use_simd) {
      auto [mv, mi] = argmax_avx2(row, num_classes);
      max_val = mv;
      index = mi;
    } else
#endif
    {
      for (int j = 1; j < num_classes; j++) {
        if (row[j] > max_val) {
          max_val = row[j];
          index = j;
        }
      }
    }

    if (index != last_index) {
      if (index != 0 && index < static_cast<int>(label_list.size())) {
        text += label_list[index];
        score += max_val;
        count++;
      }
    }
    last_index = index;
  }
  if (count > 0)
    score /= count;
  return {text, score};
}

bool load_label_dict(const std::string &dict_path,
                     std::vector<std::string> &label_list) {
  std::ifstream in(dict_path);
  if (!in) [[unlikely]] {
    std::cerr << std::format("[Rec] Failed to open dictionary: {}",
                             dict_path)
              << '\n';
    return false;
  }
  std::string line;
  while (getline(in, line)) {
    if (!line.empty() && line.back() == '\n')
      line.pop_back();
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    label_list.push_back(std::move(line));
  }
  label_list.push_back(" ");
  return true;
}

} // namespace turbo_ocr::recognition
