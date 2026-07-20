#pragma once

// Shared constants of the PP-FormulaNet ORT host-loop TUs
// (ppformulanet_ort.cu / ppformulanet_decode.cu / ppformulanet_plusm.cu /
// ppformulanet_gate_bench.cu).

#include "turbo_ocr/formula/ppformulanet/ppformulanet_preprocess.h"

namespace turbo_ocr::formula {

constexpr int MAX_B = 32, S = kFormulaInputSize;
// MAXLEN = the static self-attention KV buffer (must match step_batched.onnx's 1056).
// The model's learned positional embedding is 1029 long -> caps at 1026 tokens; the old
// 512 buffer corrupted long formulas (>510 tok) — the sole cause of the FAST-path gap.
// MAXIT*3 = 1026 tokens covers the full range.
constexpr int H = 16, Dh = 24, CTX = 144, VOCAB = 50000, MAXLEN = 1056, MAXIT = 342, CHECK = 16;
static_assert(MAXIT * 3 <= MAXLEN,
              "decode writes 3 KV slots per step at pos=it*3; MAXIT*3 must fit MAXLEN");

// PP-FormulaNet_plus-M: 6-layer MBart decoder, Dh=32, one greedy token/step, per-seq
// pos[B], 1056-cap static KV. decoder_step.onnx emits next_token in-graph (greedy
// argmax) so the host loop threads next_token->tokens device-to-device. Decoder start
// token = </s> (id 2, the MBart eos==decoder_start convention); decode stops on out id 2.
constexpr int PM_LAYERS = 6, PM_Dh = 32, PM_MAXIT = 1056, PM_START = 2;
constexpr int PM_MAX_N = 128;  // continuous-batch queue capacity (crops pre-encoded)
// ORT-CUDA reliably runs the plus-M step at batch <= 30 but throws at exactly MAX_B=32
// (the step graph's batch dim is dynamic, so this is a runtime quirk, not a graph cap);
// clamp the active batch here so production never hits it.
constexpr int PM_MAX_BATCH = 30;
// Length-bucketing: most formulas decode in well under 384 tokens, and the static-KV step
// attends over the FULL KV window every step, so a 384-wide bucket cuts per-step attention
// + the KV-output write ~2.75x vs the 1056 window. Crops that exceed 384 tokens re-decode
// in the 1056 buffers. 384 covers the OmniDocBench isolated-formula gate (max 331 tokens).
constexpr int PM_MAXLEN_S = 384;

}  // namespace turbo_ocr::formula
