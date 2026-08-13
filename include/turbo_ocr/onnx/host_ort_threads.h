#pragma once

// THE one answer to "how many intra-op threads should a HOST-side ORT stage
// session use?"
//
// WHY THIS IS SHARED, AND WHY IT IS NOT THREE SETTERS.
//
// Every host ORT stage picks its own intra-op cap, and every one of them wrote
// down the SAME justification for it: one session is driven concurrently by a
// worker pool, so ORT's default (a thread per physical core, per session)
// oversubscribes.
//
//     engine::OrtEngine ............... 4   (det / rec / cls / doc-orientation)
//     formula ppformulanet OrtSession . 4
//     layout::OrtPaddleLayout ......... 2
//
// That reasoning is correct exactly when the rest of the stage set is ALSO on
// the CPU. It is wrong for a backend whose det/rec run on an accelerator: there
// the CPU is idle and the cap is pure latency for no benefit. Measured on Apple
// native (MPSGraph det/rec), raising the cap gave ~1.9x on table, ~2.0x on
// formula and ~1.7x on layout.
//
// The POLICY — "is the host idle, and if so how many threads may a stage take"
// — is therefore one decision, not three. Writing it once here is what stops
// the three caps drifting the moment someone tunes one of them; per-stage
// setters would re-create exactly that drift (the repo has paid for it before).
// Each stage keeps its OWN historical default and passes it in, so a backend
// that says nothing is bit-for-bit unchanged.
//
// PRECEDENCE, highest first:
//   1. ORT_NUM_THREADS      — the pre-existing operator override. Unchanged
//                             semantics, still wins, still applies everywhere.
//   2. the backend's hint   — set_host_ort_intra_op_threads(), set once at
//                             bootstrap by a backend that knows its host is
//                             idle. Only Apple's native path sets it today.
//   3. stage_default        — what the stage used before this file existed.

namespace turbo_ocr {

// Set once, at backend bootstrap, by a backend whose main stages do NOT run on
// the CPU. n <= 0 clears the hint (back to per-stage defaults). Not thread-safe
// against concurrent readers by design: it is set during load_stages(), before
// any session is constructed.
void set_host_ort_intra_op_threads(int n) noexcept;

// The thread count this stage's session should use. `stage_default` is the
// stage's own historical cap and is returned when nothing overrides it.
// A return of <= 0 means "let ORT size it" (no stage asks for this today).
[[nodiscard]] int host_ort_intra_op_threads(int stage_default) noexcept;

} // namespace turbo_ocr
