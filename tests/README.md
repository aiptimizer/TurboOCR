# Turbo OCR test suite

Two kinds of test, one tree. **`cpp/` is compiled and needs no server;
everything else is Python and talks to a running server over HTTP/gRPC.**

Every directory here earns its place — one line each:

| Directory | Why it exists |
|---|---|
| `cpp/` | C++ tested in-process, no server, no network — the only tests that can run in a build sandbox. |
| `cpp/backends/` | Not Catch2: standalone executables that diff one *backend* against another, so they link differently from the unit binary. |
| `integration/` | The endpoint contract: what a client actually receives, against a real server. |
| `regression/` | Behaviours that broke once and must never break again (ordering, cross-request contamination, endpoint matrix). |
| `accuracy/` | Scores output against captured ground truth, so a model swap can't quietly cost F1. |
| `stress/` | Sustained-load soaks; minutes not seconds, so they must be opt-in. |
| `benchmark/` | Measurement drivers, not tests — they print numbers instead of passing or failing. |
| `e2e/` | Drivers that boot a real server or Docker container, i.e. the only tests that exercise startup itself. |
| `fixtures/` | The corpus: images, PDFs and their captured `expected/*.json`. |
| `_grpc_generated/` | Committed protobuf stubs, so gRPC tests don't need `grpcio-tools` at run time. |

```
tests/
  pytest.ini   testpaths = integration regression accuracy
  conftest.py  shared fixtures — every suite does `from conftest import ...`
  requirements.txt
  run_all.py   master driver (pytest for the suites, ctest for cpp)

  cpp/         common/ decode/ detection/ recognition/ layout/ table/ formula/
               forms/ markdown/ docassembly/ pdf/ pipeline/ router/ lang_cls/
               vlm/ server/ validation/ capability/ backend_routing/  → turbo_ocr_tests
               backends/  → turbo_conformance · turbo_golden · turbo_bench · turbo_backend_probe

  integration/ service/    health, metrics, capabilities, boot config, error shapes
               image/      /ocr, /ocr/raw, /ocr/batch, /ocr/pixels, /ocr/markdown,
                           /infer, formats, edge cases, base64, box order
               pdf/        /ocr/pdf, pdf modes, pdf streaming, /ocr/stream
               structure/  layout, layout-only, formula auto-CJK routing
               grpc_api/   gRPC OCRService  (not `grpc/` — a package by that
                           name would shadow the real `grpc` module)
  regression/  accuracy/  stress/

  benchmark/   perf/       latency · throughput · stress · matrix (+ the shared harness)
               scoring/    FUNSD F1 · OCR tiers · per-document F1 + latency
               router/     CUA-router scenarios + rolling-baseline regression detector
               comparison/ head-to-head vs other engines (needs their deps)
  e2e/         docker + native-server matrix drivers
  fixtures/  _grpc_generated/
```

## Rules this tree follows

**A test that needs a server is an integration test.** There is no Python
`unit/` suite: the three files that used to live there (base64 decoding,
JSON envelope shape, box sort order) all POST to `/ocr`, and now sit in
`integration/image/` and `integration/service/`. "Unit" means `cpp/`.

**Every C++ file is listed by path in the root `CMakeLists.txt`.** Adding a
file under `cpp/` does not compile it — add it to
`add_executable(turbo_ocr_tests …)` too, or it is dead on arrival.

**A helper lives with its primary consumer; a second consumer adds one
documented `sys.path` insert in its `conftest.py`.** That is why
`benchmark/perf/_harness.py` (used by the four perf drivers) is reached from
`stress/conftest.py`, and `accuracy/_scoring.py` from
`benchmark/scoring/bench_per_document.py`, rather than living in a shared
`_support/` directory that would need boilerplate in every script.

**`benchmark/funsd_gt_words.json` stays at the `benchmark/` root** even
though only `scoring/` reads it from Python: the compiled FUNSD gate
hardcodes that path (`cpp/backends/harness.h`, `default_gt_path`).

## Running

```bash
pip install -r tests/requirements.txt          # once

python tests/run_all.py                        # integration → regression → accuracy
python tests/run_all.py --suite cpp            # ctest, not pytest
python tests/run_all.py --suite stress         # 60 s soak (opt-in)
python tests/run_all.py --suite all            # everything
python tests/run_all.py --server-url http://myhost:8000 --grpc-target myhost:50051

ENABLE_LAYOUT=1 python tests/run_all.py --suite integration --suite accuracy
```

Direct pytest (paths relative to the repo root):

```bash
pytest tests/integration/image/ -v
pytest tests/integration/pdf/test_pdf_modes.py -v
pytest tests/integration/image/test_ocr_endpoint.py::TestOcrEndpoint::test_detects_known_text -v
```

C++ only:

```bash
cmake --build build --target turbo_ocr_tests
ctest --test-dir build --output-on-failure --no-tests=error
```

## Benchmarks and e2e are not suites

`pytest.ini` collects only `test_*.py`, and neither directory has any —
they are drivers you run directly:

```bash
python tests/benchmark/perf/bench_matrix.py --quick     # ~2 min smoke
python tests/benchmark/scoring/bench_per_document.py
python tests/benchmark/router/bench_cua_router.py
python tests/e2e/docker_endpoint_matrix.py              # CI runs this one
```

`bench_matrix.py` writes `perf/LATEST.{md,json}`; `bench_per_document.py`
writes `scoring/PER_DOCUMENT.{md,json}`; both gitignored. See
`tests/e2e/README.md` for what each e2e driver needs.

## Optional dependencies

`reportlab` (PDF tests) · `grpcio-tools` (gRPC stubs) · `aiohttp`
(`benchmark/perf` drivers and the stress suite).

The deeper testing playbook — sanitizers, fuzzing, load testing, the
pre-merge checklist — is in the docs site under **Contributing → Testing
strategy** (`docs/contributing/testing-strategy.md`).
