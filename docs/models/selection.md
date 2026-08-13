# Model Selection — `OCR_MODEL`

The recognizer is **PP-OCRv6** — a single script-agnostic model covering
**Latin + Chinese + Japanese**, shipped in three tiers. Scripts outside that
coverage are served by retained PP-OCRv5 recognizers behind the same shared v6
detector. One environment variable, `OCR_MODEL`, picks which recognizer the
server loads at startup.

## PP-OCRv6 tiers

The default is the throughput-first **`tiny`** tier. Step up to **`small`** for a
balanced accuracy/speed point, or **`medium`** for best accuracy.

| `OCR_MODEL` | FUNSD F1 | Throughput | Use it for |
| --- | ---: | ---: | --- |
| `tiny` *(default)* | ~84.5% | ~447 img/s | Maximum throughput — edge / high-volume |
| `small` | ~90.3% | ~225 img/s | Balanced accuracy/speed |
| `medium` | **~91.9%** | ~83 img/s | Best accuracy |

The three tiers trade accuracy for speed, and **`tiny` additionally trades away
Japanese**. `small` and `medium` share `keys.txt` (18,708 chars); `tiny` ships
`keys_tiny.txt` (6,904), which is a strict subset of it — and the characters it
omits include **all 180 Japanese kana**, plus 9,391 CJK ideographs. On a clean
Japanese page `tiny` recovers 81% of characters against 100% for `small` and
`medium`: with no kana in its output table it substitutes visually adjacent
kanji, turning 支払期限 into 支扎期限 and お客様番号 into 书客様番号.

For **Latin** the two dictionaries are equivalent in practice — German umlauts,
French/Spanish/Nordic accents, currency symbols and Greek are all present in
both, so `tiny`'s dictionary never limits a European document. Chinese is also
unaffected (100% on all three tiers).

**Use `small` or `medium` for Japanese. `tiny` cannot represent kana at all**,
and the failure is silent: it emits confident wrong characters rather than an
error.

!!! note "`tiny` is freely selectable"
    The `tiny` tier is now a first-class `OCR_MODEL` value — no opt-in
    environment flag is required to enable it.

## `tiny-bigdet` — full detector, tiny recognizer

`tiny-bigdet` is **not a tier**. It pairs the full-size detector (`det.onnx`)
with `tiny`'s recognizer and dictionary, for scans where text is small or
degraded. It exists because on such pages **the detector fails before the
recognizer does**: below roughly 12 px of glyph height, `tiny`'s detector
fragments lines and drops whole line-starts, and no recognizer can read text
that was never detected.

Measured on a synthetic fax-degraded page (gaussian blur 0.9 + noise σ14 +
JPEG q30), sweeping glyph height (`em px = pt/72 × dpi`), whitespace-insensitive
character accuracy:

| em px | `tiny` | `tiny-bigdet` | `small` | `medium` |
| ---: | ---: | ---: | ---: | ---: |
| 22.9 | 99.69% | 99.69% | 100.00% | 99.85% |
| 16.2 | 96.17% | 96.17% | 99.23% | 98.62% |
| 13.9 | 91.26% | 94.79% | 98.77% | 98.16% |
| 12.2 | 81.29% | 87.27% | 94.17% | 97.55% |
| 11.1 | **20.86%** | **74.54%** | 81.44% | 93.25% |

The gain is entirely at the bottom of the range: identical to `tiny` above
16 px, worth +53.7 points at 11 px.

**It is not, however, a cheaper route to `small`'s robustness — `small` beats it
outright.** The full detector's forward pass costs more than `small`'s entire
pipeline, so `tiny-bigdet` ends up both slower and less accurate:

| config | latency (per page) | accuracy @ 11.1 px | det + rec on disk |
| --- | ---: | ---: | ---: |
| `tiny` | 51-70 ms | 20.86% | 83 MB |
| `tiny-bigdet` | 128-141 ms | 74.54% | 150 MB |
| `small` | 102-115 ms | **81.44%** | 207 MB |
| `medium` | 214-272 ms | **93.25%** | 353 MB |

Latency measured on Apple native (`ORT_NUM_THREADS` unset → 4 intra-op threads);
state that setting alongside any comparison, as it moves the host-stage numbers
substantially.

So `tiny-bigdet` is **Pareto-dominated by `small` on speed and accuracy**, and
its only advantage is a 57 MB smaller footprint — because the tiny recognizer's
bucket ladder is 66 MB against `small`'s 173 MB. `small` is 24-26 ms faster at
every step of the sweep above and 3.1-6.9 points more accurate at every
degradation level (all four columns measured in one bracketed window, so the
comparison is not a cross-run artefact).

**Choose `tiny-bigdet` only when disk or memory is the binding constraint AND
the documents are Latin-only.** It carries `keys_tiny.txt`, so like `tiny` it
cannot represent Japanese kana. For every other case `small` is the better
choice on both axes.

!!! warning "Do not build this pairing with `DET_MODEL`"
    Setting `DET_MODEL=models/det.onnx` alongside `OCR_MODEL=tiny` makes
    `resolve_model` treat the detector as overridden, which **discards the
    entry's `det_cfg`** and falls back to the global defaults. For this
    particular pairing the defaults coincide with the correct values, so it
    happens to work — but the same mechanism silently runs `det_tiny` at
    `box_thresh` 0.45 instead of its own 0.40 when the override goes the other
    way. The registry row carries `det_path` and `det_cfg` together, so the
    pairing is correct by construction rather than by coincidence.

## Other scripts (PP-OCRv5)

Scripts outside PP-OCRv6's coverage are served by retained PP-OCRv5
recognizers, selected with the same `OCR_MODEL` variable:

| `OCR_MODEL` | Script |
| --- | --- |
| `arabic` | Arabic |
| `eslav` | East-Slavic Cyrillic (Russian, Ukrainian, …) |
| `korean` | Hangul + basic Latin |
| `thai` | Thai |
| `greek` | Greek (dedicated recognizer) |

## Selecting a model

`OCR_MODEL` is the selector. Set it at startup; all models are baked into the
image, so there is no runtime download. An unknown value fails startup with the
list of valid models.

```bash
# Best accuracy
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  -e OCR_MODEL=medium ghcr.io/aiptimizer/turboocr:latest

# Cyrillic
docker run --gpus all -p 8000:8000 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  -e OCR_MODEL=eslav ghcr.io/aiptimizer/turboocr:latest
```

!!! warning "`OCR_LANG` is deprecated"
    `OCR_LANG` is a deprecated alias of `OCR_MODEL`. It still works but warns on
    use, and `OCR_MODEL` wins when both are set. Prefer `OCR_MODEL`.

!!! info "See also"
    - [Engine comparison](../benchmarks/comparison.md) — full accuracy and throughput numbers across engines and datasets.
    - [Build · Configuration](../reference/configuration.md) — `OCR_MODEL` and the other environment-variable knobs.
    - [Recognition](recognition.md) — how the selected recognizer runs inside the pipeline.
    - [Detection](detection.md) — the shared v6 detector that feeds every recognizer.
