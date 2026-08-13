# Formula — `PP-FormulaNet_plus-S` (in-process ORT-CUDA-13)

_Why the formula stage is a pure in-process C++ ONNX-Runtime decoder running a split-graph host loop — the only GPU path._

!!! abstract "TL;DR"

    - The live backend is **PP-FormulaNet_plus-S** (`FORMULA_BACKEND=ppformulanet_s`,
      alias `ppformulanet_plus_s`), implemented by `PPFormulaNetOrt`
      (`src/backends/nvidia/stages/ppformulanet_ort.cu`). The bundle dir keeps its
      historical `ppformulanet_s` name, but its weights are the official
      PP-FormulaNet_plus-S — the bundle rebuilds **byte-identically** from
      paddle's plus-S download via
      `scripts/models/onnx/export_ppformulanet_plus_s_fast.py` (verified 2026-08-05).
    - It runs **entirely in-process on ONNX Runtime CUDA-13** — no Python, no
      sidecar process, no TensorRT engine for the formula stage.
    - **GPU: the FAST split graph is the only path** — `encoder.onnx` → `prep.onnx`
      (cross-attention KV prep) → `step_batched.onnx`, driven by a host-side
      static-KV autoregressive loop on the pipeline's CUDA stream. It matches the
      fused reference exactly at **CDM ≈ 0.811**, ~8× faster than the fused graph.
      The `fast/` split graphs are **required**: if any is missing the server
      fatal-aborts at boot — there is **no fused fallback**.
    - The fused graph (`inference_trt.onnx`) is used **only by the CPU-only build**
      (`CpuFormulaRecognizer`, ORT `CPUExecutionProvider`), where the CUDA host
      loop can't run. The GPU recognizer has no fused/EXACT mode.

## Backend selection

`FORMULA_BACKEND=ppformulanet_s` resolves through `make_formula_recognizer`
(`src/backends/nvidia/stages/formula_recognizer.cpp`) to `PPFormulaNetOrt`. A *configured*
formula backend that fails to load **aborts boot** (`maybe_load_router_models`)
— the server never starts with formulas silently disabled. A dispatched formula
region that yields no LaTeX is surfaced as `formula_degraded` in the response, so
a degraded stage never looks identical to a page that genuinely had no formula.

| Env | Meaning |
| --- | --- |
| `FORMULA_BACKEND=ppformulanet_s` | Select the local PP-FormulaNet_plus-S backend (`ppformulanet_plus_s` is an accepted alias). **This is all you need** — the baked weights at `models/formula/ppformulanet_s/` auto-resolve. |
| `FORMULA_ONNX=…/ppformulanet_s/` | **Override only** — point at a non-baked model root (the GPU loads its `fast/` split graphs from it). Not needed with the baked image. |
| `FORMULA_TOKENIZER=…/ppformulanet_s/tokenizer.json` | **Override only** — non-baked tokenizer path. |
| `FORMULA_CROP_PAD` | Page-context margin (px) added around each formula box before recognition (default 4, clamped 0–64). The model is margin-sensitive: tight crops scramble the AR decoder, 8+ pulls in neighbor glyphs. Applied once at the shared dispatch — reported boxes are unchanged. |
| `PPFNS_CHUNK` | Decode batch size (default 8; clamped to [1, 32]) |
| `PPFNS_DROP_COLLAPSE=1` | Drop mode-collapsed runaway decodes to empty (opt-in; measured worse overall) |

### Swapping the local model: `plus-S` (fast) ⇄ `plus-M` (Chinese)

Two local in-process C++ formula models are selectable via `FORMULA_BACKEND`
(weights auto-resolve from `models/formula/<engine>/`; the `FORMULA_ONNX`/
`FORMULA_TOKENIZER` overrides below are optional, for non-baked locations):

| `FORMULA_BACKEND` | Model | Chinese (CJK-F1) | English | Speed¹ |
| --- | --- | --- | --- | --- |
| `ppformulanet_s` *(default)* | PP-FormulaNet_plus-S (2-layer, MTP) | ✗ **0.00** | edit-sim 0.79 · CDM-F1 0.808 | **160 crops/s** |
| `ppformulanet_plus_m` | PP-FormulaNet_plus-M (6-layer MBart) | ✓ **0.73** | edit-sim 0.82 · CDM-F1 0.994 | 12.7 crops/s |

¹ Lockstep batch-30 on the 30-crop gate set (15 EN + 15 ZH), RTX 5090. CJK-F1 + edit-sim
measured this run via the in-process C++ decode (`tools/checks/plusm_selftest.cpp`,
`PPFNS_GATE_BENCH`); CDM-F1 from the reference scorer. **`plus-S` is ~12.5× faster but
reads effectively zero Chinese in practice; `plus-M` recovers ~73% of Chinese characters
and edges it out on English, at ~1/12 the throughput.** Use `plus-S` by default, swap to
`plus-M` for Chinese documents (or `auto` to escalate only CJK crops).

```bash
# Chinese-capable (GPU only) — weights auto-resolve from the baked dir:
FORMULA_BACKEND=ppformulanet_plus_m
```

`plus-M` uses the **same in-process FAST host-loop design** as `plus-S` (its own
`encoder.onnx` + `prep.onnx` + static-KV `decoder_step.onnx`, single greedy token per
step, device-resident KV — no Python, no sidecar), decoder-verified bit-exact to the
reference greedy decode (29/30 gate crops token-identical, the 30th a single near-tie
flip). It is **6-layer / 512-d** vs `plus-S`'s 2-layer / 384-d, so per-step cost is
higher; treat it as the accuracy/Chinese option, `plus-S` as the throughput default.

## The execution path

On GPU, `PPFormulaNetOrt::load_model_dir` runs exactly one path — the FAST split
graph (`src/backends/nvidia/stages/ppformulanet_ort.cu`). There is no fused/EXACT mode and no
`FORMULA_DEVICE=cpu` here; the fused graph is reached only by the separate
CPU-only build (`CpuFormulaRecognizer`).

### FAST split graph (the only GPU path)

The Paddle export fuses the K=3 MTP decoder into a single `while`/`Loop` graph.
Running that fused loop per crop is correct but slow. The FAST path instead
splits it into three ONNX graphs and drives the autoregressive loop from C++:

| File (`…/ppformulanet_s/fast/`) | Role |
| --- | --- |
| `encoder.onnx` | image encoder → cross-attention memory (`{1, CTX, 2048}`); **bit-exact to the fused in-graph encoder** |
| `prep.onnx` | precomputes the cross-attention K/V (`ck`, `cv`) from the memory once per crop |
| `step_batched.onnx` | one autoregressive step over a **static** self-attention KV buffer |

All three run on the pipeline's CUDA stream via ORT-CUDA-13. The host loop keeps
a static KV buffer (`MAXLEN = 1056`, i.e. `MAXIT = 342` steps × 3 MTP slots =
1026 tokens — sized to the model's 1029-long positional embedding) and an
on-device argmax kernel; it syncs only every `CHECK` steps to test for EOS. This
matches the fused reference's output while amortizing the loop overhead.

!!! warning "Generating the `fast/` artifacts"

    The three `fast/` ONNX graphs are **locally generated, not part of the
    committed model bundle**. Regenerate them with
    `scratchpad/fastdec/batched_step.py` (which exports `encoder.onnx`,
    `prep.onnx`, and `step_batched.onnx` next to the fused
    `inference_trt.onnx`). The `fast/` graphs are **required** on GPU: if any is
    missing or fails to load, `load_model_dir` prints a `FATAL` and **aborts boot**
    — there is **no fused fallback**. The fast bundle must ship with the model.

### Fused graph (CPU-only build)

The fused `inference_trt.onnx` is loaded only by `CpuFormulaRecognizer`
(`src/analysis/formula/ppformulanet/cpu_formula_recognizer.cpp`) on ORT `CPUExecutionProvider`, used by
the CPU-only server where the CUDA host loop can't run. It is **not** reachable
from the GPU recognizer (no `PPFNS_EXACT`, no `FORMULA_DEVICE`).

## Decode

Crops are preprocessed (`formula_preprocess_one`, `ppformulanet_preprocess.cpp`)
to a fixed `S×S` grayscale, ImageNet-normalised canvas, batched `PPFNS_CHUNK` at
a time. The decoder emits MTP token triples; the host loop stops a crop at the
first `EOS` and strips BOS (id 0) / pad (id 1). `FormulaTokenizer`
(`src/analysis/formula/ppformulanet/formula_tokenizer.cpp`) — a BPE over the `tokenizer.json` vocab —
decodes the surviving token ids to LaTeX (`VOCAB = 50000`).

`FormulaEngineResult` carries `latex`, `token_count`, `hit_eos`, and the `ok`
status channel (false on a backend error, distinct from a clean empty result).

## C++ surface

```mermaid
classDiagram
  class PPFormulaNetOrt {
    +load_model_dir(model_dir) bool
    +load_tokenizer(tokenizer_json) bool
    +run(page, vector~Box~, stream) vector~FormulaEngineResult~
    +is_ready() bool
    -enc_ / prep_ / step_  OrtSession   "FAST split graph (GPU, only path)"
    -tok_ FormulaTokenizer
    -decode_chunk(B, out)
  }
  class FormulaEngineResult {
    +latex string
    +token_count size_t
    +hit_eos bool
    +ok bool
  }
  PPFormulaNetOrt --> FormulaEngineResult
```

## Where it plugs in

`UnifiedOcrPipeline::dispatch_router_`
(`src/pipeline/unified/unified_pipeline_dispatch.cpp`) collects the
formula layout regions, waits on `det_only_event_` so the page image is safe to
read, and calls `PPFormulaNetOrt::run` on `formula_stream_`. The local backend is
synchronous (`supports_async()==false`); only the external VLM backend defers via
`finalize_deferred`.

!!! info "See also"

    - [Table](table.md) — the table stage and its SLANeXt / VLM backends.
    - [Router](../architecture/router.md) — which layout class IDs route into formulas.
    - [Model Interactions](interactions.md) — formula step in the full-pipeline sequence.
    - [Speed vs accuracy](../benchmarks/speed-vs-accuracy.md) — measured formula CDM across configs.
