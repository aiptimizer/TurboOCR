# Engine comparison — TurboOCR vs other OCR engines

Like-for-like comparison on a single RTX 5090, split by what each class of engine
is built to do. Two tasks: whole-page OCR (forms/receipts) and full document
parsing (papers/books with tables and formulas).

!!! abstract "TL;DR"
    - **Forms & receipts:** TurboOCR has the best text accuracy *and* is 15–90×
      faster than every other engine.
    - **Full document parsing:** TurboOCR's local pipeline reaches **0.90 Overall**
      on a **125-doc OmniDocBench subset** at **20 pages/s**, within ~5 points of
      **PaddleOCR-VL (0.95, same subset)** — which runs at **0.9 pages/s**. A ~20×
      speed gap, fully local, no API. (Subset, not the full 1651-page set.)

## What is measured

- **Whole-page OCR:** **word-level F1** — predictions and ground truth are
  lowercased, split into ≥2-char tokens, scored by set overlap. Every engine gets
  the same image and the same metric. FUNSD (English forms) and CORD (English
  receipts), 50 pages each.
- **Full document parsing:** the **official OmniDocBench scorer** over a
  **125-document stratified subset** of OmniDocBench — **not** the full 1651-page
  set. The subset is table/formula-heavy (used for fast iteration); every pipeline
  is scored on the same 125 documents, so the comparison is apples-to-apples but
  the absolute Overall is not directly comparable to full-1651 leaderboard numbers.
  Text = 1 − text-block edit distance, Formula = CDM, Table = TEDS-structure,
  Overall = mean of the three.
- **Hardware:** RTX 5090. Throughput is steady-state at saturating concurrency;
  the full-pipeline figure runs layout + text + table + formula together.

---

## Forms & receipts (whole-page OCR)

| Engine | FUNSD F1 | FUNSD img/s | CORD F1 | CORD img/s |
|---|---:|---:|---:|---:|
| **TurboOCR-medium** | **91.9%** | 86 | **93.4%** | 193 |
| **TurboOCR-small** | 90.3% | 230 | 92.7% | 480 |
| **TurboOCR-tiny** *(default)* | 85.4% | **678** | 88.9% | **559** |
| PaddleOCR-VL-1.6 (VLM) | 91.6% | 5 | 89.4% | 7 |
| PaddleOCR PP-OCRv5 (Python) | 86.6% | 6 | 86.4% | 5 |
| RapidOCR (GPU) | 69.1% | 2 | 82.6% | 8 |
| EasyOCR | 59.8% | 3 | 67.3% | 6 |
| Tesseract | 62.3% | 2 | 38.2% | 2 |

TurboOCR has the best accuracy on both datasets and is 15–90× faster than the
next-most-accurate engine. (The tiny FUNSD row was re-measured on the v4
unified pipeline — 678 img/s pooled across replicas, F1 85.4%, ≥15 s window;
the other rows are the original measurement set.)

---

## Full document parsing (full pipeline)

Each pipeline is run end-to-end (layout → region recognition → reading-order
assembly) and scored by the official OmniDocBench scorer on a **125-document
subset** (the same documents for every pipeline; not the full 1651-page set).

| Pipeline | Text | Formula (CDM) | Table (TEDS) | Overall | pg/s |
|---|---:|---:|---:|---:|---:|
| PaddleOCR-VL-1.6 (PP-DocLayoutV3 + VLM) | 97% | 0.97 | 0.92 | **0.95** | 0.9 |
| **TurboOCR-tiny** *(default)* | 95% | 0.92 | 0.82 | **0.90** | **20** |

PaddleOCR-VL is the most accurate on dense structured pages, but TurboOCR lands
within ~5 Overall points at ~20× the speed, fully local. Accuracy is essentially
flat across the TurboOCR tiers (medium/small/tiny all score ~0.90 Overall — they
share the same layout, table and formula stages and differ only in the text
recognizer), so the tier is a speed choice; throughput on the heavy OmniDocBench
scans is memory-bound at the working `--pool-size 3` config.

## How PaddleOCR-VL is run

PaddleOCR-VL is a vision-language model served by **vLLM** over the
OpenAI-compatible API.

```bash
# Whole-page comparison — VL served directly:
vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --port 8155 --host 0.0.0.0 \
  --max-num-seqs 64 --gpu-memory-utilization 0.75 --trust-remote-code
```

In the whole-page comparison each page is sent as a base64 image with the official
`OCR:` prompt (`temperature=0`, concurrency 8). In the full-pipeline comparison VL
is driven by its PP-DocLayoutV3 layout stage (`tools/bench/omnidoc_run_paddlevl.py`).

## Reproduce

The full-pipeline OmniDocBench numbers reproduce from this repo:

```bash
# Full pipeline (OmniDocBench-125, official scorer). Enable table + formula:
FORMULA_BACKEND=ppformulanet_s TABLE_BACKEND=slanext ./build/turboocr-server --pool-size 3
python scripts/eval/omnidoc_run_and_score_n.py --server-url http://localhost:8080 \
       --n 125 --experiment-name <tier>
```

The whole-page forms/receipts comparison (FUNSD/CORD across the external engines:
PaddleOCR-VL, PaddleOCR-Python, RapidOCR, EasyOCR, Tesseract) is run from a separate
multi-engine harness — each engine scored on identical images with the same word-F1
metric. TurboOCR's own forms/receipts throughput is measured directly against a
running server (`POST /ocr/raw`, `DISABLE_LAYOUT=1`, concurrency 32).
