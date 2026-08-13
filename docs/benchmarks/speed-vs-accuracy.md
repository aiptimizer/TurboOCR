# What this OCR server is, what it's tested on, and how the backends compare

This page describes, concretely: **what the system does**, **which models do what**, **what data it
is benchmarked on** (and what that data's ground truth contains), **what every metric means**, and a
**head-to-head of the three ways to run it** — local-only, hybrid (local + PaddleOCR-VL on regions),
and PaddleOCR-VL doing the whole page — on both speed and accuracy. All numbers are reproducible
(see [Reproducing](#6-reproducing-every-number)).

The server is a **hybrid document-OCR pipeline**: it always runs fast local text detection +
recognition, optionally runs layout detection to find regions, and can recognize table/formula
regions either with small **local** models (SLANeXt, PP-FormulaNet) or by sending each region's crop
to an external **PaddleOCR-VL** vision-language model over an OpenAI-compatible endpoint. It does *not*
have a full-page VL mode — sending a whole page to PaddleOCR-VL is a separate standalone pipeline,
benchmarked here as the accuracy ceiling.

---

## 1. Dataset & ground truth

Benchmarks use **OmniDocBench** (`omnidocbench/data/OmniDocBench.json`) — **1,651 real document
pages** (books, papers, exams, slides, newspapers; Chinese / English / mixed; 1–3 columns). Each page
has layout regions (`layout_dets[]`) typed by `category_type`, each carrying its ground truth:
`text` (text blocks), `latex` (`equation_isolated` display formulas), or `html` (tables).

| | docs | GT tables | GT display-formulas | GT text blocks |
|---|---|---|---|---|
| **Full OmniDocBench** | 1,651 | 665 | 2,066 | 16,520 |
| **Test subset used here** | **125** | **119** | **489** | **1,294** |

The 125-doc subset is a deterministic, stratified, **table/formula-heavy** cut
(`scripts/eval/omnidoc_subset_n.py`: docs with both a table and a formula first, then stratified across
source/language/layout) so the table and formula paths are genuinely exercised. **Speed and accuracy
use the identical 125 docs.** Those GT counts (119 tables / 489 formulas / 1294 text) are the
denominators — an "extracted N" only means something against them.

**Exactly which 125 documents** — the full list (every filename + its source/language/GT counts) is in
[the benchmark corpus page](corpus.md). Composition:

| by source | n | | by language | n | | every page has | n |
|---|---|---|---|---|---|---|---|
| exam_paper | 27 | | simplified_chinese | 51 | | a table only | 49 |
| book | 22 | | english | 48 | | a formula only | 48 |
| academic_literature | 16 | | en_ch_mixed | 19 | | both table + formula | 28 |
| colorful_textbook | 16 | | traditional_chinese | 7 | | | |
| research_report | 12 | | | | | | |
| PPT2PDF | 11 | | | | | | |
| newspaper | 9 | | | | | | |
| note | 7 | | | | | | |
| magazine | 5 | | | | | | |

Examples: `exam_paper-file-putnam-archive_1997_..._page_001.png` (two-column math exam),
`PPT_CalculusReview_page_014.png` (slide deck), `jiaocaineedrop_..._546.jpg` (Chinese textbook),
`docstructbench_...-j.physletb.2004.11.060.pdf_2.jpg` (two-column physics paper). 77 of the 125 have a
table, 76 have a formula; mix of 1–3 column layouts, print + handwritten notes, EN/中文/mixed.

---

## 2. Models — which model does what

**What every benchmark number in this document actually uses:** the **`tiny`** OCR tier — `det_tiny.onnx`
(1.7 MB) + `rec_tiny.onnx` (4.3 MB). This is the default (`OCR_MODEL` unset) and on these complex docs
it is also the **best** tier (medium underperforms tiny on dense/handwritten pages). There is **no
"fast" tier**; the three tiers are `tiny` / `small` / `medium`, selected by `OCR_MODEL`. The exact stack
per configuration:

- **text** (every row): `det_tiny` + `rec_tiny` + `cls` (PP-LCNet angle) + layout (PP-DocLayoutV3)
- **+ table (local)**: single SLANet-Plus encoder (GPU) + host GRU decoder
- **+ formula (local)**: PP-FormulaNet_plus-S via the **`ppformulanet_s`** in-process ORT-CUDA-13 backend
- **+ VL (hybrid or VL-only)**: **PaddleOCR-VL-1.6-0.9B** on vLLM — table/formula region crops in the
  hybrid, or the whole page in VL-only

Always-on = text path (det+rec+cls). Everything else is optional. Per-model detail lives in
[`docs/models/`](../models/); this is the inventory.

| stage | model (file) | architecture | size | selection | on by default |
|---|---|---|---|---|---|
| text detection | `models/det_tiny.onnx` (·_small 9.5 MB ·`det` 60 MB) | PP-OCRv6 DBNet | 1.7 MB | `OCR_MODEL` tier | **yes** |
| text recognition | `models/rec_tiny.onnx` (·_small 21 MB · `rec` 74 MB) | PP-OCRv6 SVTR | 4.3 MB | `OCR_MODEL` tier | **yes** |
| angle classify | `models/cls.onnx` | PP-LCNet | 1.0 MB | `DISABLE_ANGLE_CLS` | yes |
| doc orientation | `models/doc_ori.onnx` | PP-LCNet | 6.5 MB | file present | yes |
| layout | `models/layout/layout.onnx` | PP-DocLayoutV3 (RT-DETR-L) | 124 MB | `DISABLE_LAYOUT` | yes |
| table (local) | `slanext_encoder/SLANeXt_wired_encoder.onnx` + `_decoder.bin` + `_dict_infer.txt` | SLANet-Plus enc(GPU)+GRU dec(host) | 5.3 + 2.1 MB | `TABLE_BACKEND=slanext` | opt-in |
| formula (local) | `formula/ppformulanet_s/fast/{encoder,prep,step_batched}.onnx` (GPU) · `inference_trt.onnx` (CPU build) + `tokenizer.json` | PP-FormulaNet_plus-S (in-process ORT-CUDA-13, FAST split graphs) | 227 MB (fast) | `FORMULA_BACKEND=ppformulanet_s` (NOT inert `formulanet`) | opt-in |
| table / formula (external) | **PaddleOCR-VL-1.6-0.9B** (`models/vlm/paddleocr_vl_1_6`) on vLLM | VLM | 1.8 GB | `kind:openai` routing | opt-in |

OCR tiers (det+rec) trade accuracy for speed: **tiny** ~85% / ~481 img/s · **small** ~91% / ~234 ·
**medium** ~92% / ~89 (these benchmarks use tiny). The external model is **any** OpenAI-compatible
vision endpoint; PaddleOCR-VL is the one used here, and table vs formula can point at different
models/hosts. *(The standalone full-page PaddleOCR-VL pipeline in the installed `paddleocr` 3.4
supports v1.5, not v1.6, so all VL numbers here are **v1.5** for an apples-to-apples comparison.)*

---

## 3. Metrics — what every number means

Scored by the OmniDocBench scorer (`omnidocbench/pdf_validation.py`, `quick_match` element matching).

| metric | direction | what it measures |
|---|---|---|
| **text edit-dist** | ↓ lower better | normalized Levenshtein distance on recognized text vs GT |
| **table TEDS** | ↑ higher better | Tree-Edit-Distance Similarity of the table HTML (structure **and** cell text) |
| **table TEDS-structure** | ↑ | TEDS ignoring cell text — grid (rows/cols/spans) correctness only |
| **formula CDM** | ↑ | Character Detection Matching: render GT & predicted LaTeX, match characters (F1). Robust to LaTeX-syntax differences |
| **formula edit-dist** | ↓ | normalized Levenshtein on the LaTeX string |
| **reading-order edit-dist** | ↓ | edit distance of the block reading-order sequence vs GT |

"Extracted N / M GT" in the speed table is **coverage**: N regions emitted by the server vs M in the
ground truth (not an accuracy score — that's section 5).

---

## 4. Speed & VRAM (per pipeline combination)

Measured on one RTX 5090, `scripts/bench/bench_speed_matrix.py --n 125 --pool-sizes 2 --concurrency 8` over
the 125-doc subset (`POST /ocr/raw?layout=1`). **VRAM scales with `PIPELINE_POOL_SIZE`** (concurrent
GPU pipelines) — these are at **pool=2** (the throughput sweet spot; see below). `+ext` = a separate
PaddleOCR-VL process on its own GPU (~15 GB), *not* in the C++ VRAM column.

| config | images/s | p50 / p90 ms | C++ VRAM | + ext | tables /119 GT | formulas /489 GT |
|---|---|---|---|---|---|---|
| text-only (no layout) | **324** | 1 / 3 | 4.3 GB | – | – | – |
| + layout | 56 | 50 / 125 | 5.8 GB | – | – | – |
| + table SLANeXt (local) | **86** | 57 / 123 | 6.0 GB | – | 127 | – |
| + table VL | 10.3 | 278 / 1572 | 5.8 GB | +15 GB | 127 | – |
| + formula PP-FormulaNet_plus-S (local) | **4.6** | 164 / 5091 | 7.5 GB | – | – | 525 |
| + formula VL | 6.7 | 613 / 3270 | 5.8 GB | +15 GB | – | 1679 |
| + table + formula (local) | **3.8** | 302 / 5284 | 7.7 GB | – | 71 | 673 |
| + table + formula (VL) | 6.1 | 721 / 1924 | 5.8 GB | +15 GB | 127 | 1679 |
| hybrid: local table + VL formula | 12 | 344 / 1799 | 6.0 GB | +15 GB | 127 | 1679 |

Reading the rows: text detection+recognition is **fast** (324 img/s); layout and SLANeXt table add
modest cost (56 / 86 img/s). The formula-stage rows above (4.6 / 3.8 img/s) were measured with the
**prior host-side autoregressive** PP-FormulaNet_plus-S decoder, which serialized on this formula-dense subset
(every page has a formula or table; p90 ~5 s). **The FAST in-process decoder is now the default** (split
GPU encoder + batched host decode on ORT-CUDA-13): it removes that serialization and lifts the full local
pipeline to **~20–24 img/s** (concurrency 4 / pool 2–4) — clearing the 10 img/s floor and overtaking the
VL paths (hybrid 6.1, VL-only 1.3). Coverage: SLANeXt/VL over-detect tables slightly (127 vs 119 GT);
formula counts (525 / 673 / 1679 vs 489 GT) are high because layout flags inline formulas and
equation-numbers, not just the GT *display* formulas the scorer evaluates.

**VRAM and throughput vs `PIPELINE_POOL_SIZE`** (`--pool-sizes 1 2 4 8`): VRAM ≈ base + pool × ~2 GB
(both_local: 3.5 / 6.3 / 11.9 / 23.1 GB at pool 1/2/4/8); throughput **peaks around pool=2** at
concurrency 8 and then degrades from GPU contention (pool=8 even OOMs). **Set `PIPELINE_POOL_SIZE`
explicitly** — auto-detect over-provisions VRAM for no throughput gain.

---

## 5. Accuracy & the head-to-head

`scripts/eval/omnidoc_run_and_score_n.py` (local / hybrid) and `scripts/bench/bench_vl_fullpage.py` (VL full
page), all on the **same 125 docs**, scored identically (`↓` lower better, `↑` higher better).

### 5a. The three ways to run it (125 docs, PaddleOCR-VL-1.5)

| pipeline | speed | text ↓ | tbl TEDS ↑ | tbl struct ↑ | tbl edit ↓ | fml CDM ↑ | fml edit ↓ | RO ↓ |
|---|---|---|---|---|---|---|---|---|
| **Local** (tiny tier; SLANeXt + PP-FormulaNet_plus-S, FAST decoder) | ~20–24 img/s | 0.144 | 0.773 | 0.876 | 0.148 | 0.805 | 0.306 | 0.333 |
| **Hybrid** (local text+layout, VL on table/formula regions) | 6.1 img/s | 0.118 | 0.895 | 0.928 | 0.082 | 0.843 | 0.148 | 0.313 |
| **VL-only** (PaddleOCR-VL over the whole page) | 1.3 pg/s | **0.073** | **0.900** | **0.931** | **0.074** | **0.874** | **0.131** | **0.194** |

The Local row is the shipped default **`LAYOUT_MERGE_MODE=all`** on the 125-doc subset with the **FAST**
in-process PP-FormulaNet_plus-S decoder (the FAST split graphs match the fused reference: CDM ≈ 0.805).
**`LAYOUT_MERGE_MODE=outer`** is the alternative if formula CDM is the priority — it scores CDM **0.8108**
(formula edit 0.299, table TEDS 0.768, table struct 0.872, RO ~0.342) at a slight cost to the other
metrics. All Local/Hybrid numbers are the 125-doc subset; the VL-only row is the PaddleOCR-VL full-page
reference run.

### 5b. Per modality — which backend wins

| modality (metric) | local | VL on region (hybrid) | VL full page |
|---|---|---|---|
| **text** (edit ↓) | 0.144 | 0.118 | **0.073** |
| **table** (TEDS ↑) | 0.773 | 0.895 | **0.900** |
| **table** (structure ↑) | 0.876 | 0.926 | **0.931** |
| **formula** (CDM ↑) | 0.805 | 0.843 | **0.874** |
| **reading order** (edit ↓) | 0.333 | 0.313 | **0.194** |

What this says:
- **Local needs no external model and is competitive on accuracy**: formula CDM 0.805 and table TEDS
  0.773 land within striking distance of VL (0.874 / 0.900), on one GPU with no vLLM. The remaining
  table gap is structure decode on complex spanning/borderless grids; the reading-order gap (0.333 vs
  0.194) is the XY-cut algorithm itself (four reorder variants were evaluated — none beat the tuned
  baseline).
- **With the FAST in-process formula decoder (now default) local also wins on throughput** (~20–24
  img/s — see §4): the split-encoder + batched host decode removes the old serialization, so the full
  local pipeline clears the 10 img/s floor and runs faster than both VL paths (hybrid 6.1, VL-only 1.3)
  even on this formula-dense subset. The earlier host-side AR decoder bottlenecked at ~3.8 img/s. Local's
  margin widens further on text/table-heavy docs (raw OCR ~324 img/s, +table 86).
- **VL-only is the most accurate on everything** (text 0.073, table 0.900, formula 0.874, RO 0.194):
  one model over the whole page gives the most coherent result — at 1.3 pages/s.
- **Hybrid ≈ VL-only on tables/formulas** while keeping fast local text; at 6.1 img/s it batches the
  region crops across vLLM, ahead of VL-only but now behind the FAST local pipeline (~20–24 img/s). Its
  larger win is on text-heavy docs, where most pages skip the per-region VL round-trips entirely.
- **VRAM note:** the in-process formula stage (ORT-CUDA) needs headroom — sharing one 32 GB GPU with a
  vLLM (15 GB) can OOM and empty some formulas (CDM drops to ~0.70). This is **not silent**: the stage
  fails loud if it can't reach the GPU, and a per-region failure surfaces as `formula_degraded` in the
  response. Give the formula stage its own GPU or run local-only; the 0.805 above is the clean
  (vLLM-free) number.

Pick by what you optimise: **one GPU, no external model, competitive accuracy at the highest throughput
→ local** (the FAST formula decoder holds ~20–24 img/s even on formula-dense pages); **maximum accuracy
→ VL-only**; **VL-grade tables/formulas without hosting a local formula model → hybrid**. The local gap
to VL is genuine model headroom (handwriting, inline-math LaTeX, dense-table cell content).

---

## 6. Reproducing every number

```bash
# Shared 125-doc subset is generated automatically by the harnesses (scripts/eval/omnidoc_subset_n.py).

# Speed + VRAM matrix (boots each config, fixed extracted/GT counting) -> result/bench_speed_matrix.{json,md}
python3 scripts/bench/bench_speed_matrix.py --n 125 --pool-sizes 2 --concurrency 8
python3 scripts/bench/bench_speed_matrix.py --baseline result/bench_speed_matrix.json   # regression gate (>15%)

# Accuracy — boot a server with the config, then score (needs omnidocbench/.venv):
#   local:  TABLE_BACKEND=slanext + FORMULA_BACKEND=ppformulanet_s  (both auto-resolve
#           the baked weights; in-process ORT-CUDA-13 — no Python sidecar)
#   hybrid: TURBO_ROUTING_CONFIG=routing.json (table/formula -> vl), TABLE_SLANEXT_* for slanext_local
python3 scripts/eval/omnidoc_run_and_score_n.py --server-url http://localhost:8822 --experiment-name local_125

# VL-only full page (PaddleOCR-VL-1.5 standalone; needs compare-ocrs/.venv + vLLM on :8077)
python3 scripts/bench/bench_vl_fullpage.py --subset 125 --server http://localhost:8077/v1
```

Serve PaddleOCR-VL on vLLM with **both** names so the C++ hybrid (`PaddleOCR-VL`) and the standalone
pipeline (`PaddleOCR-VL-1.5-0.9B`) both resolve:
```bash
vllm serve models/vlm/paddleocr_vl_1_5 --port 8077 --trust-remote-code \
  --served-model-name PaddleOCR-VL-1.5-0.9B PaddleOCR-VL --gpu-memory-utilization 0.45
```
Build recipe (clean): `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=<trt> -DFETCH_MODELS=OFF`.

For the local pipeline, just set the formula backend (weights auto-resolve from
the baked `models/formula/ppformulanet_s/`):
```bash
FORMULA_BACKEND=ppformulanet_s ./build/turboocr-server ...
# Runs in-process on ORT-CUDA-13 (no Python, no sidecar) from the fast/ split
# graphs. It fails loud if it can't reach the GPU or the fast/ graphs are
# missing, and surfaces per-region failures as formula_degraded.
# Override the path with FORMULA_ONNX / FORMULA_TOKENIZER only for a non-baked
# location. Knob: PPFNS_CHUNK (decode batch, default 8).
```
