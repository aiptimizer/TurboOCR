#pragma once

#include <string>
#include <utility>
#include <vector>

namespace turbo_ocr::recognition {

// GPU path: indices and scores already computed by GPU argmax kernel.
[[nodiscard]] std::pair<std::string, float>
ctc_greedy_decode(const int *indices, const float *scores, int seq_len,
                  const std::vector<std::string> &label_list);

// CPU path: raw logits, needs inline argmax.
[[nodiscard]] std::pair<std::string, float>
ctc_greedy_decode_raw(const float *logits, int seq_len, int num_classes,
                      const std::vector<std::string> &label_list);

// Shared dictionary loader.
// Prepends "blank" and appends " " around the file contents.
// label_list should be empty on entry (or pre-populated with "blank").
[[nodiscard]] bool load_label_dict(const std::string &dict_path,
                                    std::vector<std::string> &label_list);

} // namespace turbo_ocr::recognition
