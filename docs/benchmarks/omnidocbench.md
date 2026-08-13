# OmniDocBench

OmniDocBench v1.7 (1651 pages, official page-match harness) is the accuracy
gate this pipeline targets. The pipeline has **two operating points** on it —
fully local, and hybrid with PaddleOCR-VL serving the structure crops — and
this page documents both, where the gap between them lives, and how to
reproduce every number.

## Headline (full set, 2026-08)

Both rows: `OCR_MODEL=medium`, full 1651-page render, 1557 pages matched and
scored by the harness (same matched set, so the rows are cell-for-cell
comparable). Composite = `((1 − Text_Edit) + Table_TEDS + Formula_CDM) / 3 × 100`.

| Operating point | text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ | Read-order Edit ↓ | **Composite** |
|---|---:|---:|---:|---:|---:|
| **Hybrid** — local text + PaddleOCR-VL tables & formulas (`TURBO_ROUTING_CONFIG`) | **0.044** | **0.924** | **0.907** | **0.130** | **92.9** |
| **Fully local** — SLANet-Plus tables + PP-FormulaNet plus-S/plus-M ladder (`FORMULA_BACKEND=auto`) | 0.073 | 0.767 | 0.766 | 0.227 | **82.0** |

Source runs: `omnidocbench/result/md_full_1651_v16_paddlevl_quick_match_*`
(hybrid) and `omnidocbench/result/md_medium_auto_0806_quick_match_*` (local,
re-measured 2026-08-06 with the fixed in-process formula stage). The hybrid
needs the vLLM sidecar ([guide](../guides/vl-separate-gpu.md)); the local
point is fully self-contained.

At 92.9 the hybrid sits above Marker (78.4) and Mistral OCR (85.7) and inside
the specialised-VLM band (93–96) on the leaderboard excerpt below. The local
point trades ~11 composite points for ~4× the throughput and zero external
dependencies.

## Speed × accuracy — every configuration, measured

Throughput: the same 100 OmniDocBench pages, full structured requests
(`?layout=1&tables=1&formulas=1`), medium tier, one pipeline replica, single
client, VL co-located on the same RTX 5090 at 0.5 GPU-memory utilization
(2026-08-07). Accuracy columns: full-set runs where they exist, per-crop
projections (same crops, VL's own per-sample scores) where marked.

| Formula engine | Table engine | img/s | CDM ↑ | TEDS ↑ | Composite |
|---|---|---:|---:|---:|---:|
| plus-S only | SLANet-Plus | **10.00** | ~0.74 | 0.766 | ~81 |
| **`auto` ladder** (plus-S, plus-M on CJK) | SLANet-Plus | **9.09** | 0.767 | 0.766 | **82.0** |
| plus-M everywhere | SLANet-Plus | 2.44 | ~0.77 | 0.766 | ~82 |
| **PaddleOCR-VL formulas only** | SLANet-Plus | **6.67** | 0.923 † | 0.766 | **≈ 87** † |
| PaddleOCR-VL both | PaddleOCR-VL | 2.50 | 0.924 | 0.907 | **92.9** |

† projected from the VL run's per-sample CDM on the same crops; not yet a
full-set scored run.

Two conclusions fall straight out:

- **`plus-M` everywhere is dominated** — it costs the same as the full VL
  hybrid (2.44 vs 2.50 img/s) for 11 fewer points. Its right place is inside
  the `auto` ladder, where escalating only CJK-context crops makes it nearly
  free (9.09 vs 10.00 img/s).
- **VL-formulas-only is the cheapest large win**: one routing-config change
  buys two-thirds of the hybrid's accuracy gap at 2.7× its speed.

## Where the local gap lives (per-crop)

Joining the two runs' per-sample scores (1,897 shared formula crops, 665
shared tables) locates the losses precisely:

- **Formulas: a failure tail, not broad weakness.** 68.7% of crops score
  CDM ≥ 0.9 locally; the mean is dragged by the bottom 16.2% under 0.3 —
  including 8.8% at exactly zero (garbled/collapsed decodes). VL beats local
  on those; it is worse than local on only 3.8% of crops.
- **Tables: spread weakness.** Only 43.5% of tables reach TEDS 0.9.
  Structure itself is decent (TEDS_structure_only 0.855); most of the loss
  is cell-text quality.
- **Text (0.073 vs 0.044) is mostly matcher collateral**: the harness aligns
  predicted blocks to ground truth globally, so weaker table/formula regions
  drag neighbouring text assignments with them. The recognizer stack is the
  same in both rows.

**Selective escalation — the measured case for it.** Because the formula loss
is a tail, escalating only the crops the local model flubs recovers almost
everything: sending the worst 20.5% of formulas (local CDM < 0.5) to the VL
yields CDM 0.911 of the VL's 0.923, and adding the worst ~15% of tables lifts
TEDS to 0.837 — a projected **≈ 89 composite at ~6.5–8.5 img/s**, near-hybrid
accuracy at roughly 3× hybrid speed. These thresholds use true scores
(an oracle); a production trigger would flag the same tail with
collapse/repetition detection (which catches the zero-CDM class), decoder
confidence, and table cell-count sanity — the same escalate-per-crop pattern
the `auto` ladder already implements for CJK, with a quality trigger and a VL
rung. Filed as the concrete next accuracy lever.

## History

| Run | Date | Config | text ↓ | CDM ↑ | TEDS ↑ | RO ↓ | Composite |
|---|---|---|---:|---:|---:|---:|---:|
| baseline | 2026-05-14 | pre-integration | 0.509 | 0.449 | 0.027 | 0.519 | 32.3 |
| model integration | 2026-05-17 | v6 text + first table wiring | 0.160 | 0.063 ‡ | 0.568 | 0.326 | ≈ 49 ‡ |
| `full_best` | 2026-06-16 | v6 text + VL structure | 0.093 | 0.896 | 0.874 | 0.226 | 89.2 |
| `v16_paddlevl` | 2026-07 | improved text + VL structure | 0.044 | 0.924 | 0.907 | 0.130 | **92.9** |
| `medium_auto_0806` | 2026-08-06 | fully local (slanext + plus-S/plus-M) | 0.073 | 0.767 | 0.766 | 0.227 | 82.0 |

‡ The May formula 0.063 was an integration bug (output mangled before
serialisation), fixed in June — the in-process formula stage now scores
CDM 0.767 full-set / 0.805 on the EN-heavy 125-doc subset. The ≈ 49
composite embeds that bug and is of historical interest only.

The big text-side win in the May integration was the CJK recognizer swap
(simplified-Chinese page edit 0.935 → 0.234); the structure wins arrived with
VL routing (June/July) and the in-process local models (June–August).

## Leaderboard context (OmniDocBench v1.6_full README, 2026-04-30)

| Methods | Class | Overall ↑ | Text Edit ↓ | Formula CDM ↑ | Table TEDS ↑ | Read Order Edit ↓ |
|---|---|---:|---:|---:|---:|---:|
| MinerU2.5-Pro | Specialised VLM (1.2B) | **95.75** | 0.036 | **97.45** | **93.42** | 0.120 |
| GLM-OCR | Specialised VLM (0.9B) | 95.22 | 0.044 | 97.18 | 92.83 | 0.133 |
| PaddleOCR-VL-1.5 | Specialised VLM (0.9B) | 94.93 | 0.038 | 96.89 | 91.67 | 0.130 |
| Youtu-Parsing | Specialised VLM (2.5B) | 93.74 | 0.044 | 93.63 | 92.02 | **0.116** |
| **TurboOCR hybrid (ours)** | Pipeline + VL crops | **92.9** | 0.044 | 92.4 | 90.7 | 0.130 |
| Gemini 3 Pro | General VLM | 92.91 | 0.064 | 95.99 | 89.15 | 0.165 |
| GPT-5.2 | General VLM | 86.59 | 0.114 | 88.21 | 82.95 | 0.193 |
| Mistral OCR | Specialised VLM | 85.66 | 0.097 | 89.91 | 76.78 | 0.171 |
| **TurboOCR local (ours)** | Pipeline tool | **82.0** | 0.073 | 76.7 | 76.6 | 0.227 |
| Marker | Pipeline tool | 78.44 | 0.157 | 85.24 | 65.77 | 0.243 |

The hybrid row uses the leaderboard VLMs' own model class for the crops it
routes out, so landing beside them is expected — the point is that the
routing keeps the C++ pipeline's speed for everything else. The local row is
the strongest pipeline-tool (non-VLM) result we know of on this set.

## Reproduce

```bash
# 1. Server — fully local best config (hybrid: add the routing config from
#    docs/guides/vl-separate-gpu.md and drop FORMULA_BACKEND/TABLE_BACKEND).
#    PIPELINE_POOL_SIZE=1: medium + both formula engines are ~23 GB resident;
#    more replicas OOM a 32 GB card mid-run.
LD_LIBRARY_PATH="$HOME/TensorRT-10.15.1.29/lib:$LD_LIBRARY_PATH" \
  OCR_MODEL=medium TABLE_BACKEND=slanext FORMULA_BACKEND=auto \
  PIPELINE_POOL_SIZE=1 \
  ./build/turboocr-server --log-level warn &

# 2. Render predictions for all 1651 pages
python tools/bench/omnidoc_run.py \
  --server http://127.0.0.1:8080 --concurrency 2 \
  --images-dir /path/to/omnidocbench/data/images \
  --out-dir /tmp/omnidoc_predictions/json

# 3. Convert raw JSON -> markdown (uses reading_order[], tables[], formulas[])
python tools/bench/omnidoc_to_md.py \
  --in-dir /tmp/omnidoc_predictions/json \
  --out-dir /tmp/omnidoc_predictions/md_myrun

# 4. Score. The result files are named after the predictions dir's basename,
#    so give it a unique name or you overwrite the previous run's results.
#    Config template: copy configs/end2end_paddle_optim.yaml and point
#    dataset.prediction.data_path at your md dir.
cd /path/to/omnidocbench && \
  .venv/bin/python pdf_validation.py --config /path/to/your_run.yaml
```

Results land in `omnidocbench/result/<mddir>_quick_match_*`: the main
`*_metric_result.json` plus per-page/per-sample breakdowns for every metric
and the run/environment reports.

## Net read

The pipeline's best accuracy (92.9) comes from the hybrid: keep detection,
recognition, layout and reading order local, route the table and formula
crops to PaddleOCR-VL. The best self-contained accuracy is 82.0 with the
`auto` formula ladder — and the measured per-crop analysis says the gap
between them is mostly a detectable failure tail, which is why selective
escalation (local first, VL only for flagged crops) is the next lever:
projected ≈ 89 at ~3× hybrid speed. plus-M's role is inside the ladder, never
as the sole engine.

!!! info "See also"
    - [Speed vs accuracy](speed-vs-accuracy.md) — dataset, metrics, and the local/hybrid/VL deep dive.
    - [PaddleOCR-VL on a separate GPU](../guides/vl-separate-gpu.md) — the hybrid's routing config.
    - [Formula](../models/formula.md) · [Table](../models/table.md) — the local structure models.
    - [Latency](latency.md) — the speed half of the benchmark story.
