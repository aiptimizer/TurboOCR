# CUA-Router Benchmark Harness

Single-client steady-state benchmark used by the 3-hour autonomous loop to
guard the **270 ms text-only p50** north-star while the CUA-router (T2–T20)
is being implemented.

Entry point: `scripts/bench/bench_cua_loop.sh` (called every 10 min).
Output dir: `/tmp/cua_bench/` (`<ts>.json`, `latest.json` symlink,
`latest.summary`, `baseline.json`).

## Scenarios

All run `c=1`, `n=60` per fixture, `warmup=10`, sequential.

| Scenario        | Endpoint              | Fixtures                                                                                                                          | Pass criteria                                         |
|-----------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `text_only`     | `POST /ocr/raw`       | png/business_letter, png/dense_text, png/multi_language, png/small_text, jpeg/03_book_page, jpeg/08_document_scan                 | p50 ≤ 270 ms (target), ≤ 280 ms (ALERT), p99 ≤ 400 ms |
| `formula_heavy` | `POST /ocr/pdf?mode=ocr` | pdf/formulas.pdf (per-page time = total / pages)                                                                              | no absolute ceiling. ALERT if (form_p50 − text_p50) > 1.20 × prior Δ |
| `table_heavy`   | `POST /ocr/raw` (image), `POST /ocr/pdf?mode=ocr` (pdf) | png/table.png, pdf/tables_document.pdf                                                                  | same shape as formula                                 |

### Text-only self-validator

At boot, the harness POSTs each text fixture to `/layout` and drops any
whose response contains `class_id ∈ {5, 11, 15, 21}` (display_formula,
formula_number, inline_formula, table). If `/layout` is unavailable, the
self-validator is skipped and a warning is logged to stderr.

## Verdict logic (plan 06 §4)

`baseline.text_only.p50_ms` = `B` = median of last 3 PASS p50s.

| Scenario | Condition                                                  | Verdict |
|----------|------------------------------------------------------------|---------|
| text     | p50 ≤ 280 ms AND ≤ 1.05·B                                  | PASS    |
| text     | 1.05·B < p50 ≤ 1.10·B (and ≤ 280 ms)                       | ALERT   |
| text     | p50 > 1.10·B OR p50 > 300 ms                               | HALT    |
| text     | p50 > 280 ms (any B)                                       | ALERT   |
| text     | two consecutive +5%-rule ALERTs                            | HALT    |
| formula  | (form_p50 − text_p50) > 1.20 × prior Δ                     | ALERT   |
| table    | (tab_p50 − text_p50) > 1.20 × prior Δ                      | ALERT   |

ALERT / HALT do **not** update the rolling baseline.
`err_rate > 0.10` → INFRA (no baseline mutation, exit 1).
`loadavg_1m > 2.0` downgrades ALERT to INFRA (concurrent builds steal CPU).

## Output schema

See `plans/06_benchmark_harness.md §5`. Top-level keys:
`schema_version`, `timestamp_utc`, `git_sha`, `server_url`, `server_up`,
`duration_s`, `loadavg_1m`, `scenarios.{text_only,formula_heavy,table_heavy}`,
`baseline`, `verdict.{overall,text_only,formula_heavy,table_heavy,consecutive_alerts,reasons}`.

## Exit codes

| code | meaning              | loop driver action                    |
|------|----------------------|---------------------------------------|
| 0    | PASS                 | quiet                                 |
| 1    | INFRA error          | retry next cycle                      |
| 2    | server not reachable | surface to user                       |
| 3    | ALERT                | log; continue                         |
| 4    | HALT                 | emit to implementation channel; stop  |

## Files

- `tests/benchmark/router/bench_cua_router.py` — orchestrator (CLI entry)
- `tests/benchmark/router/_cua_scenarios.py` — fixtures, thresholds, self-validator
- `tests/benchmark/router/_regression.py` — rolling baseline + verdict
- `scripts/bench/bench_cua_loop.sh` — health probe + python invocation + exit map
