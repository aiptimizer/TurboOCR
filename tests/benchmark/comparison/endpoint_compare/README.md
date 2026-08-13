# Endpoint-by-endpoint TurboOCR vs PaddleOCR

`paddle_server.py` wraps PaddleOCR 3.x (PP-OCRv5 server det/rec on GPU +
PP-StructureV3) behind TurboOCR-shaped HTTP endpoints on :9090.
`endpoint_compare.py` fires identical inputs at both servers and compares the
returned VALUES per endpoint — texts (token-F1), word counts, box IoU,
confidences, layout class histograms, table HTML / formula LaTeX counts,
markdown, per-page PDF texts — plus median latency.

Run on the NVIDIA box:

```bash
TABLE_BACKEND=slanext FORMULA_BACKEND=ppformulanet_s ./build-gpu/turboocr-server &
.venv/bin/python paddle_server.py &          # compare-ocrs venv (paddle-gpu)
.venv/bin/python endpoint_compare.py
```

Latest run (2026-08-05, RTX 5090, turbo tiny tier vs PaddleOCR v5 SERVER
models — note the tier asymmetry favors Paddle on accuracy):

| endpoint | words T/P | token-F1 | box IoU@match | conf T/P | ms T/P |
|---|---|---|---|---|---|
| /ocr/raw | 38/36 | 0.710 | 0.83 @ 92% | 0.939/0.908 | **8 / 155** |
| /ocr (b64) | 38/36 | 0.710 | 0.83 @ 92% | 0.939/0.908 | 9 / 156 |
| /ocr/batch[1] | 38/36 | 0.710 | 0.83 @ 92% | 0.939/0.908 | 12 / 155 |

- Turbo transports are value-identical among themselves (raw == b64 == batch).
- 10-page FUNSD aggregate: token-F1 0.848 mean (0.710 min), 553 vs 546 words.
- Layout: both emit 22 regions on page 0 with a compatible class mix; 9 ms vs 193 ms.
- Tables/formulas: 0/0 on FUNSD (forms corpus has none — needs a doc corpus).
- Markdown: 1126 vs 1228 chars, token-F1 0.614 (different export policies), 12 vs 193 ms.
- PDF (8 pages): per-page token-F1 0.607–0.880, identical page counts, **40 ms vs 1612 ms**.
- /ocr/stream emits exactly the 8 pdf pages (internal consistency).

## Old (pre-multibackend) vs new (unified) — `old_vs_new.py`

Same API on both sides, so every endpoint is compared for exact value
equality. Run 2026-08-05 (RTX 5090, both native TRT, PIPELINE_POOL_SIZE=1;
old = paddle-highspeed-cpp build of 2026-07-15 on TRT 10.15 with its own
freshly built engine cache, new = the unified server on TRT 10.16):

| endpoint | verdict | words | max conf Δ | max box Δ | token-F1 | ms new/old |
|---|---|---|---|---|---|---|
| /ocr/raw · /ocr · /ocr/batch | same counts, 2/38 borderline text flips | 38/38 | 0.015 | 1 px | 0.987 | 9/9 · 12/11 · 15/38 |
| layout (page 0) | **exactly equal** (22 regions, class+box) | — | — | — | — | 9/10 |
| /ocr/markdown | **byte-identical** | — | — | — | 1.000 | 9/9 |
| /ocr/pdf (8 pages) | same counts every page; 1–17 char-level flips/page, token-F1 0.986–1.0 | = | ≤0.33 | — | ≥0.986 | 61/48 |

Interpretation: functional parity holds — word counts, layout and markdown
are equal; the residual text/confidence deltas are TensorRT version skew
(10.15-built vs 10.16-built engines produce different float outputs, so
borderline argmax characters flip). The merge-time byte-identical gate
compared same-toolchain builds; this run deliberately crosses toolchains.
/capabilities differs by design: the new server adds profile_endpoint,
backend/device/engine_mode fields from the capability registry.

## Structure endpoints, old vs new — `old_vs_new_full.py` (2026-08-05)

Exercised on real OmniDocBench pages (found by scanning for pages where the
new server emits tables/formulas), plus test8.pdf and rotated FUNSD pages:

| endpoint | result |
|---|---|
| `?tables=1` | 2/2 tables, **HTML byte-identical**, 47+20 cells, cell-text F1 1.000 |
| `?formulas=1` | 25/25 formulas, **every LaTeX string identical** |
| `/ocr/markdown` (structured page) | **byte-identical** (1491 chars, table HTML embedded) |
| `/ocr/pdf?markdown=1` | not byte-identical (27648 vs 27626 chars, token-F1 0.996 — TRT version skew on OCR'd text) |
| `/ocr/stream` | **8/8 pages text-identical to /ocr/pdf on both servers** when matched by the 1-based `page` field (events legitimately arrive out of order) |
| `/ocr/pixels` | works on both; new pixels == new raw exactly; cross-engine F1 0.987 (same TRT skew as raw) |
| `/infer` (table crop, `backend:"table-env"`) | **200 on both, identical HTML**; the registered name is `table-env`, not the engine string — `slanext` gets the same clean 400 on both |
| `?autorotate=1` | 90°: recovered on both (new 32 words vs old 27 — new better); **180°: NOT corrected on either** (F1 0.02, identical behavior — pre-existing doc-orientation limitation, not a regression) |

Notable identical-on-both quality artifacts (model-level, not engine-level):
formula #2/#3 LaTeX is garbled the same way on both servers
("J{ \cdot o m l }" for mol, doubled superscripts) — PP-FormulaNet-S output,
carried faithfully by both pipelines.
