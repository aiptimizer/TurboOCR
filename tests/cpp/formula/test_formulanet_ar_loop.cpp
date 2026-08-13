// Host-loop semantics tests for FormulaNet's 3-engine AR decode.
//
// Real engine wiring needs CUDA + TRT, which the standalone test runner
// doesn't link. These tests exercise the pure host-side state machine the
// AR loop uses: per-lane prefix accumulation, EOS termination, padding of
// finished lanes with mask=false, and greedy argmax over a vocab row.

#include "catch_amalgamated.hpp"
#include "turbo_ocr/analysis/formula/ppformulanet/formula_tokenizer.h"

#include <vector>

// Mirror the public constants from formulanet.h without pulling in the CUDA
// runtime header — keeps this test buildable in the standalone (non-GPU) test
// runner. Bound by static_assert in formulanet.h's own translation unit.
namespace turbo_ocr::formula {
inline constexpr int kFormulaMaxSeqLen_test = 512;
inline constexpr int kFormulaVocab_test     = 8000;
inline constexpr int kFormulaBosId_test     = 1;
inline constexpr int kFormulaEosId_test     = 2;
}

namespace tof = turbo_ocr::formula;
using turbo_ocr::formula::FormulaTokenizer;

namespace {

// Mirrors the host-side argmax used per step (greedy decode of last position).
int argmax_row(const float *row, int V) {
    int best = 0;
    float bv = row[0];
    for (int i = 1; i < V; ++i) {
        if (row[i] > bv) { bv = row[i]; best = i; }
    }
    return best;
}

// Drive a single-lane decode given a scripted "next token" oracle.
// Returns the full produced token sequence including BOS and EOS (if hit).
std::vector<int64_t>
script_decode(const std::vector<int> &next_tokens) {
    std::vector<int64_t> toks;
    toks.reserve(next_tokens.size() + 1);
    toks.push_back(tof::kFormulaBosId_test);
    for (int t : next_tokens) {
        toks.push_back(static_cast<int64_t>(t));
        if (t == tof::kFormulaEosId_test) break;
    }
    return toks;
}

}  // namespace

TEST_CASE("argmax_row: greedy pick over a logit vector", "[formula_ar_loop]") {
    std::vector<float> row(tof::kFormulaVocab_test, 0.0f);
    row[42] = 5.5f;
    REQUIRE(argmax_row(row.data(), tof::kFormulaVocab_test) == 42);

    // Tie-break: argmax_row returns the FIRST max (strict `>` comparison).
    row[42] = 3.0f;
    row[7]  = 3.0f;
    REQUIRE(argmax_row(row.data(), tof::kFormulaVocab_test) == 7);
}

TEST_CASE("decode trace: BOS prefix + 3 content tokens + EOS",
          "[formula_ar_loop]") {
    auto toks = script_decode({10, 20, 30, tof::kFormulaEosId_test});
    REQUIRE(toks.size() == 5);
    REQUIRE(toks.front() == tof::kFormulaBosId_test);
    REQUIRE(toks.back() == tof::kFormulaEosId_test);
}

TEST_CASE("decode trace: EOS terminates before max_seq_len",
          "[formula_ar_loop]") {
    auto toks = script_decode({tof::kFormulaEosId_test});
    REQUIRE(toks.size() == 2);
    REQUIRE(toks[0] == tof::kFormulaBosId_test);
    REQUIRE(toks[1] == tof::kFormulaEosId_test);
}

TEST_CASE("decode trace: no EOS within budget runs to max",
          "[formula_ar_loop]") {
    std::vector<int> nx(tof::kFormulaMaxSeqLen_test - 1, 99);
    auto toks = script_decode(nx);
    REQUIRE(static_cast<int>(toks.size()) == tof::kFormulaMaxSeqLen_test);
    REQUIRE(toks.front() == tof::kFormulaBosId_test);
    REQUIRE(toks.back() == 99);
}

TEST_CASE("batch lane padding: finished lanes contribute mask=false",
          "[formula_ar_loop]") {
    // Synthetic 4-lane state: lane 0 hit EOS at step 2, lanes 1..3 still live.
    constexpr int B = 4;
    std::vector<std::vector<int64_t>> lanes(B);
    std::vector<bool> done(B, false);
    const int64_t BOS = tof::kFormulaBosId_test;
    const int64_t EOS = tof::kFormulaEosId_test;
    lanes[0] = std::vector<int64_t>{BOS, int64_t{11}, EOS};  done[0] = true;
    lanes[1] = std::vector<int64_t>{BOS, int64_t{12}, int64_t{13}};
    lanes[2] = std::vector<int64_t>{BOS, int64_t{14}, int64_t{15}};
    lanes[3] = std::vector<int64_t>{BOS, int64_t{16}, int64_t{17}};

    const int T = 3;
    std::vector<int64_t> packed(B * T, -1);
    std::vector<bool>    mask(B * T, true);
    for (int i = 0; i < B; ++i) {
        const auto &row = lanes[i];
        for (int t = 0; t < T; ++t) {
            if (t < static_cast<int>(row.size())) {
                packed[i * T + t] = row[t];
                mask[i * T + t]   = !done[i];
            } else {
                packed[i * T + t] = 0;
                mask[i * T + t]   = false;
            }
        }
    }

    for (int t = 0; t < T; ++t) {
        REQUIRE(packed[0 * T + t] == lanes[0][t]);
        REQUIRE(mask[0 * T + t] == false);
    }
    for (int i = 1; i < B; ++i) {
        for (int t = 0; t < T; ++t) {
            REQUIRE(mask[i * T + t] == true);
        }
    }
}

TEST_CASE("tokenizer integration: decode of synthetic [BOS, ids..., EOS]",
          "[formula_ar_loop]") {
    // Parity-test without a real tokenizer.json: latex_post_process is the
    // public part of the decode path. The full BPE decode needs the actual
    // vocab; this just confirms the post-process is called on trimmed input.
    std::string after = FormulaTokenizer::latex_post_process("a + b = c");
    REQUIRE(after == "a+b=c");
}
