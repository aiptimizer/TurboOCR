# tools/

Developer tooling. Nothing here is linked into the server or shipped in the
Docker image.

**One file in this tree has an automated consumer:** `bench/score_funsd.py`, which
ctest runs as the `funsd_*_gate` tests with `--assert-f1`. Break it and the build
goes red. Everything else — including every C++ target below — is run by a human,
by hand. `grep add_test` in the root `CMakeLists.txt` if you doubt it.

Directories are named for the job you came here to do, not for the language a
tool is written in or the build file that mentions it:

| | Why it exists |
| --- | --- |
| `bench/` | Produces **numbers** for a build: throughput and accuracy. They are one directory because this repo never reports them apart — `score_funsd.py` asserts on both `--assert-f1` and `--assert-throughput`, and `cpu_profile_bench.py` emits img/s alongside an accuracy signature. Splitting "speed" from "accuracy" would cut across the grain. |
| `checks/` | Answers **"did I break this?"** for one subsystem, by comparing its output to a reference — a golden file, a config expectation, the other backend, or another build. Run one when you suspect a regression; read the numbers in `bench/` when you want to know how fast. |
| `modelgen/` | Produces a **model artefact** that something else consumes: an ONNX file, an MPSGraph export directory, a generated `.inc`. Export, conversion and quantization are one workflow, so they are one directory. |
| `probes/` | **Frozen hardware research.** Compiled by nothing, kept for the measurements it produced. One subdirectory per vendor, mirroring `src/backends/`. |
| `syntax_shims/` | Stub SDK headers plus `check.sh`, so vendor sources this machine cannot compile still get type-checked. |
| `new_backend.py` | Stamps out a backend skeleton. It sits at the top because it is the one tool you run *before* you know the layout — and it pairs with `syntax_shims/`, its other half when porting a backend. |

There is deliberately no `drivers/`, no `scripts/`, no `python/`: those name an
implementation detail, and a tool would move between them for reasons that have
nothing to do with what it does.

## bench/

| Tool | |
| --- | --- |
| `score_funsd.py` | FUNSD bag-of-words F1. **The ctest accuracy gate** — see above. Also asserts throughput. |
| `cpu_profile_bench.py` | Boots `turboocr-server` under a given env, runs timed `/ocr/batch` rounds, reports img/s + per-stage ms + an accuracy signature. |
| `cpu_sweep.py` | Runs the above across a config list and tabulates it, gating accuracy against the first row. Input format: `sweep_integration.txt`. |
| `grpc_bench.cpp`, `grpc_burst.cpp` | gRPC throughput and burst load. CMake targets. |
| `omnidoc_run.py`, `omnidoc_to_md.py` | OmniDocBench: run the server over the corpus, convert results to markdown for scoring. Orchestrated by `scripts/eval/`. |
| `omnidoc_run_paddlevl.py` | The same corpus through PaddleOCR-VL, for head-to-head comparison. |
| `analyze_vlm_profile.py` | Latency statistics from a VLM profile jsonl. |
| `formbench/` | Form-field and font-detection evaluation (CommonForms, head-to-head vs Acrobat). A Python package — `acrobat_headtohead` imports `eval_commonforms` imports `decode_reference` — so these files must stay together. |

## checks/

| Tool | Compares |
| --- | --- |
| `plusm_selftest.cpp` | plus-M FAST-path decode against a golden. No server boot. |
| `formula_swap_check.cpp` | `FORMULA_BACKEND` against the routing engine it actually resolves to. |
| `diff_ccl.cpp` | GPU connected components (`CCL=2`) against the CPU path (`CCL=1`) — box counts, IoU, missed/extra. |
| `test_nvjpeg_race.cpp` | Regression for the nvJPEG handler-thread → dispatcher-worker race. |
| `cmp_text.py` | Recognized text under config B against config A — boxes differing + character error rate. |

## modelgen/

`export_ffdetr.py` (form-field detector), `mps_export_rec.py` (MPSGraph
`graph.json` + `weights.bin` for the Apple backend), `make_glyphless_font.py`
(generates the glyphless-font `.inc` under `src/`), and the quantization
workflow: `quantize_ocr.py` → `eval_quant.py` → `bench_quant.py`, all sharing
`quant_common.py`. Those four import each other by directory and must stay
co-located.

## probes/

`probes/apple/` is the MPSGraph / Metal / CoreML research that produced the Apple
backend, kept for the measurements recorded in `docs/notes/apple-backend-log.md`.
Most of it is superseded by `src/backends/apple/`; several files are cited by no
document at all. Read the header comment before running one — it carries the
exact `clang++ -ObjC++` line, and expects to be invoked from the repo root with
`build-cpu/` present.

`mps_rec_build.h` here is a one-line forwarding header to the real translator at
`src/backends/apple/engine/mps_rec_build.h`, so the probes' `-Iinclude
-Itools/probes/apple` recipe keeps working without `-Isrc/backends`.

`probes/nvidia/fused_dwpw_bench.cu` is a standalone `nvcc` microbenchmark of a
fused depthwise+pointwise block.

## Adding a tool

Put it where its job is. If you cannot say in one clause what job a new
directory would name, it does not need one — a shared filename prefix groups
files just as well below about five of them.
