# Testing

!!! abstract "TL;DR"
    Everything compiled lives under **`tests/cpp/`** (Catch2 →
    `build/turbo_ocr_tests`, built unconditionally); everything else
    under `tests/` is **Python** (driven by `python tests/run_all.py`).
    Performance work goes through the `scripts/bench/bench_*.sh` family;
    CUA-router regressions are caught by `scripts/bench/bench_cua_loop.sh`.

See also [Testing strategy](testing-strategy.md) for the deeper playbook
(sanitizers, fuzzing, load testing, the pre-merge checklist), and
`tests/README.md` for the in-tree directory map.

## C++ tests (Catch2) — `tests/cpp/`

A single binary, `turbo_ocr_tests`, is built unconditionally alongside
the server. Its sources are listed **explicitly** in the root
`CMakeLists.txt` (`add_executable(turbo_ocr_tests …)`) — a new file
under `tests/cpp/` is not compiled until it is added there.

| Directory | Covers |
|---|---|
| `base/` | Geometry primitives, JSON envelope + escape rules, UTF-8 via simdutf, env parsing, logger throttle/bounds |
| `decode/` | Pre-decode header sniff, dim guard, PNG guards |
| `detection/`, `recognition/` | DB post-process, CTC decode, rec crop geometry |
| `layout/` | XY-cut, reading-order buckets, blocks + hierarchy, PicoDet decode |
| `table/`, `formula/` | SLANeXt dict / post-process, cell matcher, HTML + OTSL reconstruct, LaTeX extract/normalize, tokenizer |
| `pdf/` | Searchable + editable PDF, font match/style, region move, renderer liveness (non-Apple only) |
| `server/`, `validation/`, `capability/`, `backend_routing/` | Config + CLI plumbing, request validation, capability registry, backend routing config |
| `pipeline/`, `router/`, `lang_cls/`, `vlm/`, `markdown/`, `forms/`, `docassembly/` | Pipeline detail + GPU safety, CUA router, script id, remote-VLM endpoint policy, Markdown export, form fields, doc assembly |

A subset is compiled **only in the CUDA configure** (it pulls in the GPU
stack): `pipeline/test_finalize_deferred.cpp`,
`pipeline/test_multi_rec.cpp`, `table/test_otsl_to_html.cpp`,
`router/test_cua_router.cpp`, `formula/test_formula_preprocess.cpp`,
`formula/test_formulanet_ar_loop.cpp`, `lang_cls/test_script_id.cpp`.

### Run

```bash
cmake --build build --target turbo_ocr_tests
./build/turbo_ocr_tests
ctest --test-dir build --output-on-failure --no-tests=error
```

Catch2 tag filters work as usual:

```bash
./build/turbo_ocr_tests "[serialization]"
./build/turbo_ocr_tests --list-tests
```

### Cross-backend executables — `tests/cpp/backends/`

Not Catch2: four standalone drivers built when `TURBO_BACKENDS` names a
backend — `turbo_conformance` (the keystone cross-backend diff),
`turbo_golden` (per-stage golden diff), `turbo_bench` (throughput that
can never print without its accuracy) and `turbo_backend_probe`
(registered with ctest unconditionally). Their acceptance path and
measurement discipline are documented in `tests/cpp/backends/README.md`.
The FUNSD gates register only when `TURBO_FUNSD_CACHE` is set.

## Python suites

`python tests/run_all.py` is the master driver. `pytest.ini`,
`conftest.py` and `requirements.txt` live at `tests/` — the pytest
rootdir — and every suite below is relative to it.

| Suite | Path | Default? |
|---|---|---|
| `integration` | `tests/integration/{service,image,pdf,structure,grpc_api}/` | yes (needs `OCR_SERVER_URL`) |
| `regression` | `tests/regression/` | yes |
| `accuracy` | `tests/accuracy/` | yes |
| `cpp` | `tests/cpp/` via `ctest` (not pytest) | opt-in |
| `stress` | `tests/stress/` | opt-in (60 s soak) |

There is **no Python `unit` suite**: every Python test here needs a running
server, so it is an integration test. "Unit" means `tests/cpp/`.

`tests/benchmark/` and `tests/e2e/` are **not** suites either — they hold
standalone drivers with no `test_*.py`, so pytest collects nothing there.
Run them directly (CI runs `tests/e2e/docker_endpoint_matrix.py`).

### Run

=== "default"

    ```bash
    # integration + regression + accuracy
    python tests/run_all.py
    ```

=== "one suite"

    ```bash
    python tests/run_all.py --suite integration
    python tests/run_all.py --suite cpp
    ```

=== "all"

    ```bash
    python tests/run_all.py --suite all
    ```

### Environment

| Var | Default | What |
|---|---|---|
| `OCR_SERVER_URL` | `http://localhost:8000` | HTTP base used by integration / accuracy / regression. |
| `OCR_GRPC_TARGET` | `localhost:50051` | gRPC target for the same suites. |

`pytest.ini` defines markers `stress`, `accuracy` and `layout`, and runs
with `--strict-markers` so a typo'd marker fails instead of silently
matching nothing. The `layout` marker auto-skips when the server reports
layout-disabled.

## Bench scripts

Standalone shell + Python harnesses for performance work:

| Script | Purpose |
|---|---|
| `scripts/bench/bench_latency.sh`     | `hey -n 200 -c 1` sequential latency p50/p95/p99 over `/ocr/raw`. Image arg defaults to `tests/fixtures/images/png/receipt.png`. |
| `scripts/bench/bench_throughput.sh`  | `hey` at configurable `-c` for req/s sweeps. |
| `scripts/bench/bench_full.sh`        | Five-concurrency sweep (`c=1,4,8,16,32`) across the fixture set; the canonical "post-change full run". |
| `scripts/bench/bench_cua_loop.sh`    | Periodic CUA-router benchmark: health-probes `/health/ready`, then invokes `tests/benchmark/router/bench_cua_router.py` against the three scenarios (`text_only`, `formula_heavy`, `table_heavy`). |
| `tests/benchmark/router/bench_cua_router.py` | Orchestrator for the three scenarios; writes a schema-versioned JSON report under `$CUA_BENCH_OUT_DIR` (default `/tmp/cua_bench`). |
| `tests/benchmark/perf/bench_latency.py` | Python equivalent of the shell sweep. |
| `tests/benchmark/perf/bench_matrix.py` | Cross-product of fixtures × concurrency — primary regression input; writes `LATEST.{md,json}` next to itself. |
| `tests/benchmark/scoring/bench_per_document.py` | Per-fixture F1 + latency table (`PER_DOCUMENT.{md,json}`). |

!!! tip "bench_cua_loop exit codes"
    Per internal engineering notes:
    `0 = PASS`, `1 = INFRA`, `2 = server-down`,
    `3 = ALERT`, `4 = HALT`. Anything non-zero blocks the merge.

!!! warning "hey on PATH"
    All `bench_*.sh` scripts expect `hey` on `$PATH` (or at
    `~/go/bin/hey`). Install with `go install
    github.com/rakyll/hey@latest`.

## CI gating

The pre-merge gate runs three stages:

1. `cmake --build build --target turbo_ocr_tests && ./build/turbo_ocr_tests`
2. `python tests/run_all.py` (integration + regression + accuracy).
3. `scripts/bench/bench_cua_loop.sh` once against the running server — the
   regression detector exits non-zero if any of the three scenarios
   regresses past the rolling-baseline threshold.

!!! info "See also"
    - [Dev → Plan history](../notes/plan-history.md) — `06_benchmark_harness.md`
      and the diary that drives the bench cadence.
    - [Benchmarks → Latency](../benchmarks/latency.md) — what the
      shell scripts measure.
    - [API → HTTP](../reference/http.md) — the surface most tests hit.
