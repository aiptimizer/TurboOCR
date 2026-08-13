# Plan history

!!! abstract "TL;DR"
    internal engineering notes is the running engineering log: every architectural
    decision, recon, and post-mortem lives there as a self-contained
    markdown file. Grouped roughly by phase below; one-liners describe
    what each document captures.

## Architecture & integration (01–07, 19)

- **`00_synthesis_and_team_brief.md`** — top-level team brief tying
  every other plan together.
- **`01_router_architecture.md`** — CUA-router class shapes and the
  fan-out / reconvene contract.
- **`02_table_port.md`** — porting the Python table pipeline to C++,
  TableCls + structure recognizer staging.
- **`03_formula_port.md`** — porting PP-FormulaNet inference into the
  C++ pipeline.
- **`04_stream_graph.md`** — the CUDA-stream centerpiece: caller / rec /
  layout / table / formula lanes, every CUDA event, and the 270 ms
  text-only invariant.
- **`05_routing_logic.md`** — layout-class → destination decision tree,
  confidence tiers, salvage logic, tie-breakers.
- **`06_benchmark_harness.md`** — the bench harness contract used by
  `scripts/bench/bench_cua_loop.sh` and
  `tests/benchmark/router/bench_cua_router.py`.
- **`07_build_integration.md`** — CMake target wiring for the
  per-domain static libs (router / table / formula).
- **`19_integration_draft.md`** — end-to-end integration RFC that
  pulled the four pieces above into the merged pipeline.

## TRT gate (10)

- **`10_trt_gate_results.md`** — TensorRT 10.15 Loop-op rejection
  findings for PP-FormulaNet-S and SLANet+: the
  `makeScopeNodesContiguous` Loop optimizer choking on
  `ScatterElements` inside Loop bodies. Drove the 3-engine formula
  refactor and the Nemotron table swap.

## Forensic findings

- **`forensic_findings.md`** — repository archaeology: pre-existing
  bugs found while wiring the new pipeline, with file:line citations.
- **`review_critic.md`**, **`review_unbiased.md`** — independent code
  reviews of major change sets.

## OmniDocBench harness (omnidoc_*)

- **`omnidoc_brief.md`** — what we're measuring and why.
- **`omnidoc_class_mapping.md`** — turbo-ocr layout classes ↔
  OmniDocBench label taxonomy.
- **`omnidoc_recon.md`** — initial recon over the public benchmark.
- **`omnidoc_result.md`** — current scoring + leaderboard placement
  (consumed by `docs/benchmarks/omnidocbench.md`).

## Latency optimisation (optim_*, push_to_70, fp32_pin, ln_refine, mem_share)

- **`optim_brief.md`**, **`optim_result.md`** — the 270 ms text-only
  optimisation sweep brief and final delta table.
- **`push_to_70.md`** — pushing the formula encoder past 70 % CDM.
- **`fp32_pin.md`** — pinning specific layers to fp32 to stop accuracy
  cliffs without giving up the fp16 throughput.
- **`ln_refine.md`** — LayerNorm fusion refinements.
- **`mem_share.md`** — sharing GPU scratch between stream-bound stages.
- **`final_bench.md`** — sealing bench numbers before tag.

## RapidAI / table reconnaissance (rapidai_*, rapid_audit, table_*)

- **`rapidai_recon.md`** — recon of the RapidAI table cell-detection
  stack (alternative to slanet_plus / Nemotron).
- **`rapid_audit.md`** — diff vs. shipped pipeline.
- **`table_70_v3.md`** — pushing table TEDS past 70 %.
- **`table_fix.md`** — targeted table-postprocess regressions.

## Formula recon (formula_*)

- **`formula_bakeoff.md`** — encoder / decoder / resizer engine
  bakeoff.
- **`formula_diag.md`** — diagnosis of the original Loop-op build
  failure.
- **`formula_fix.md`** — the surgery that produced the current
  3-engine setup.
- **`formula_leak_fix.md`** — stream-scoped memory leak fix.
- **`formula_parity_diff.md`** — parity diffs against PaddleOCR
  reference.
- **`formula_standalone_debug.md`** — standalone fp16 reproducer.
- **`formula_trt_engine_probe.md`** — TRT engine probe results.

## Script-ID (script_id_*, language-router sweeps, cjk_push)

- **`script_id_recon.md`** — initial CJK script-id model recon.
- **`script_id_data_v2.md`** — synthetic data generation V2.
- **`script_id_bench_v2.md`** — V2 model bench results.
- language-router sweep briefs + result rounds (v1–v3, lang-only) —
  accuracy sweeps for the script-id router.
- **`cjk_push.md`** — Chinese / Japanese / Korean accuracy push.

## Surgery + re-exports (onnx_*, opset17, v26_*, v30_*, revert_and_det)

- **`onnx_reexport.md`** + **`onnx_reexport_script.py`** — re-export
  flow for stuck graphs.
- **`onnx_surgery_findings.md`** — what surgeries are safe vs. lossy.
- **`opset17.md`** — opset-17 migration notes.
- **`v26_consolidated.md`**, **`v30_reexport.md`** — version-stamp
  rollups.
- **`revert_and_det.md`** — detection-only revert path.

## trtllm decoupling research (A/B/C)

- **`trtllm_decouple_research_A.md`** / **`_B.md`** / **`_C.md`** —
  three parallel research tracks on decoupling from TensorRT-LLM.
- **`trtllm_formula_plugin.md`** — plugin design that ultimately got
  shelved in favor of the 3-engine ONNX flow.

## Misc

- **`cmake_fork.md`** — internal CMake fork rationale (since merged
  back).
- **`cpp_wire.md`** — early wiring notes during the C++ port.
- **`99_bench_diary.md`** — running diary of every bench sweep with the
  deltas (8 sweeps total; source for `docs/benchmarks/latency.md`).
- **`goal.md`** — original goal statement.
- **`docs_site_brief.md`** — this site's planning brief (the document
  that produced this entire docs tree).

!!! tip "Every plan is self-contained"
    Each file in internal engineering notes states the question being answered,
    the experiment run, and the verdict. Pull the matching plan when
    you need to know **why** something is the way it is.

!!! info "See also"
    - [Architecture → Overview](../architecture/overview.md) — the
      synthesis that consumed plans 01–07.
    - [Architecture → CUDA Streams](../architecture/cuda-streams.md) —
      reads `04_stream_graph.md`.
    - [Benchmarks → Latency](../benchmarks/latency.md) — reads
      `99_bench_diary.md` and `final_bench.md`.
