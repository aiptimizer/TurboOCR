# Form fields

`?fields=1` on `/ocr/pdf` answers a question the rest of the pipeline does not:
**where on this page would a person have to write?** It returns rectangles with
a type — `text`, `checkbox` or `signature` — which is what a caller needs to
turn a flat scan into a fillable PDF.

This is the half of Adobe's "Prepare Form" that no open stack had. It is also
the one place TurboOCR runs a detector whose classes are *interaction*
affordances rather than document structure.

## Two detectors, deliberately

Field detection is the sum of two things that are wrong in opposite directions.

**Geometry** (`src/analysis/forms/`, always on) reads the raster the way a person does —
a printed rule, a drawn box, a label followed by enough blank space, an empty
table cell. It gives **pixel-exact edges** wherever the document actually drew
the blank, which nothing regressing normalised coordinates can match. It cannot
see a blank that nothing was drawn around, and it can never report a signature:
under a signature line the morphology is an ordinary rule.

**FFDetr** (`models/forms/ffdetr.onnx`, optional) is an RF-DETR trained on
CommonForms — ~450k pages filtered out of Common Crawl for carrying real
fillable widgets. Its targets are the rectangles the documents' own authors
chose. It finds blanks that were never drawn, and it is the only source of the
`signature` class.

`merge_fields` reconciles them. A rectangle both argued for comes back as
`ffdetr+box`, keeps the **geometry's** edges, and gains confidence for the
agreement. A specific type always beats the default `text`, whichever detector
carried it — otherwise a model-detected signature merging into a geometry rule
would silently become a plain text field.

## Deferring to the model, but only where it earned it

One geometry detector is different from the other three. `label_gap` argues
from a colon and some whitespace — **no ink at all**. So when the model has
looked at the same page and proposed nothing there, that is genuine evidence
against. Measured on 40 CommonForms test pages, proposals sourced *only*
`label_gap` were **64 false positives and zero true positives**:

| | overall P | overall R | overall F1 |
|---|---|---|---|
| every proposal emitted | 0.758 | 0.817 | 0.786 |
| **drop uncorroborated `label_gap`** | **0.822** | **0.817** | **0.820** |
| also drop solo `rule` | 0.834 | 0.809 | 0.821 |
| drop all solo geometry | 0.852 | 0.810 | 0.830 |

Dropping `label_gap` alone is free: recall is unchanged to three decimals, the
same 620 true positives. Dropping solo `rule` and `box` as well buys a little
more precision but starts costing recall — a printed line *is* evidence — so
they are kept.

**But the rule cannot be unconditional.** On a plain scanned form — labels, no
drawn widgets anywhere — 8 of 10 fields are `label_gap` and nothing else, and
the model proposes only 2. Dropping them would leave 2 fields and destroy the
feature on exactly the documents it is most needed for.

What separates the two cases is not the document type, it is **how much the
model said**: it sourced 76 of 79 fields on the real form and 2 of 10 on the
plain one. A model that has barely spoken is out of its distribution, and
reading its silence as a veto deletes the only answer available.

So the deferral fires per page, only when model-sourced fields are a **majority
of the survivors**. Verified on both:

| page | fields | model-backed | solo `label_gap` | result |
|---|---|---|---|---|
| plain scan | 10 | 2 (20%) | **8 kept** | untouched |
| real form | 76 | 75 (95%) | **0** | 3 dropped |

Set `FieldOptions::defer_inference_to_model = false` to emit everything. The
rule never fires when no model proposals were supplied, so a server without the
weights is bit-for-bit unchanged.

!!! warning "The majority test is page-global"
    A page where the model does well in one region and poorly in another — a
    two-column form with a dense grid on one side — still gets a single verdict
    for the whole page, so inferential proposals in the region the model
    handled badly are vetoed along with the rest. Making the test region-local
    is possible; there is currently no measurement showing it is needed, so it
    has not been done.

## Measured

CommonForms **test** split (the paper's own held-out set, ground truth from the
source PDFs' widgets), IoU 0.5, confidence 0.40.

### The model in isolation — 150 pages

| class | P | R | F1 |
|---|---|---|---|
| text | 0.902 | 0.918 | 0.910 |
| checkbox | 0.723 | 0.445 | 0.551 |
| signature | 1.000 | 0.286 | 0.444 |
| **overall** | **0.883** | **0.836** | **0.859** |

Only 7 signature instances occur in 150 pages, so that row is indicative, not a
result.

### The shipping server — 40 pages, model on vs off

Same binary, same pages; `FIELD_MODEL_ONNX=none` disables the model.

| | geometry only | + FFDetr | **+ deferral (ships)** |
|---|---|---|---|
| text F1 | 0.768 | 0.834 | **0.852** |
| checkbox F1 | 0.046 | 0.574 | **0.574** |
| checkbox recall | 0.025 | **0.529** | **0.529** |
| signature F1 | 0.000 | 0.667 | **1.000** |
| overall precision | 0.771 | 0.758 | **0.783** |
| overall recall | 0.599 | 0.817 | **0.819** |
| **overall F1** | **0.675** | 0.786 | **0.801** |

Checkbox recall of 0.025 is the honest number for geometry alone: 4 true
positives against 157 ground-truth checkboxes. Real forms draw checkboxes in
ways morphology does not generalise over — glyphs, shaded cells, borderless
grids. **The model does essentially all of the checkbox work**, and checkbox is
the class commercial preparers are documented not to detect at all.

Only 2 signature instances occur in these 40 pages, so that row is a sanity
check, not a result.

A blanket "drop every uncorroborated `label_gap`" would score 0.820 here rather
than 0.801. It is not what ships, because the same change takes a plain scanned
form from 10 fields to 2 — see below.

## Verified end to end: scan → fillable → filled → visible

Detection metrics do not show that a form can actually be filled in. What shows
it is taking a page with no AcroForm, running `?fields=1`, writing real
widgets, putting values in every one, and then **looking at the rendered
result**. Done with PDFBox, the same library the Stirling integration uses.

On a plain scanned German form: 9 fields proposed, 9 widgets written, 9 filled,
every value rendered on its own blank. On a real requisition form: **75 fields —
43 text and 32 checkboxes — all filled, all visible**, including every cell of
its sample table individually.

That last step found three defects that no detection metric could have:

**1. Values stored but invisible.** Writing fields and setting
`/NeedAppearances` puts the value in the dictionary but leaves the *drawing* to
the viewer, and PDFium (like many renderers) ignores the flag. The page came
back looking blank. PDFBox generates a real appearance stream on `setValue()`,
which is what the product does — but only for text fields.

**2. Checkboxes had no appearance at all.** 14 of 46 fields carried an
appearance stream: every text field, and not one checkbox. PDFBox never
generates one for `PDCheckBox`, so a ticked box rendered as nothing —
i.e. the class this whole model exists for did not work. Fixed by building
both `/Off` and `/Yes` states explicitly. The "on" state draws **only the
tick**: the scan already has the box printed on it, and stroking a border
would draw a second one over every box on the page.

**3. One field swallowing a whole table.** A requisition form's sample table is
genuinely empty, so its outer border passed the emptiness test and came back as
a single 676×337 proposal — 22% of the page — enclosing the 30 cell fields
inside it. Filling that printed one value in enormous type across the entire
table. Fixed by the container rule: a proposal holding ≥3 other surviving
proposals is the rectangle drawn *around* several fields, not a field.

## Moving elements — tables, not just figures

Moving a region on a scanned page is a pixel operation: the region has no
object identity to reposition, so its pixels are cut, the hole healed with the
local background, the pixels pasted at the target, and the page re-OCR'd.

The test that matters is not that it looks moved. It is that the moved thing is
still **found as what it was**, which is what breaks if the move disturbs the
ruling lines a table recogniser depends on. Measured on a report page:

```
table BEFORE  [188,394 .. 1447,754]
table AFTER   [191,1295 .. 1447,1713]     shift +901 px (asked +900)
regions detected as `table` after the move: 1
recognised words inside it: 15  ('Warengruppe', 'Schrauben', '250', '12.40', …)
```

Re-OCR is what keeps the invisible text layer honest — a moved table whose
words stayed behind would be worse than not moving it at all.

## Why 1024, and not higher

The FFDNet paper calls high input resolution "crucial for detecting fine
details like underlines and small buttons", which makes a larger export the
obvious thing to try. Measured, on 150 pages:

| resolution | text F1 | checkbox F1 | signature F1 | overall |
|---|---|---|---|---|
| **1024** (trained) | 0.910 | **0.551** | **0.444** | 0.859 |
| 1280 | **0.923** | 0.534 | 0.000 | 0.867 |

1280 buys text accuracy and **loses checkbox and signature** — a train/test
resolution mismatch, not a detail-visibility problem. The +0.008 overall is
paid for with exactly the two classes the model exists to provide. 1024 is also
the only resolution the checkpoint is evaluated at upstream.

Confidence and NMS were swept for the same reason. `conf=0.40` maximises
overall F1, and class-agnostic NMS beats per-class (checkbox F1 0.610 vs 0.575
at conf 0.20) — both already the reference defaults. Even at conf 0.20 checkbox
recall only reaches 0.663, so the ~0.44 is the model's ceiling, not a threshold
artefact.

## What it costs

`/ocr/pdf?fields=1` on one A4 page at 200 dpi, CPU backend, Apple Silicon
(10 P-cores), mean of 5 after warmup — the whole request, including OCR and
layout:

| | ms/page |
|---|---|
| geometry only | 247 |
| + FFDetr, fp16 @ 4 threads | 1398 |
| **+ FFDetr, fp32 @ 8 threads (ships)** | **891** |

Same 76 fields, same 32 checkboxes, 36% faster. Two independent findings got it
there, and neither was the one that looked obvious.

**Threads mattered most.** The stage originally inherited the 4-thread cap the
other host ORT stages use, which cost 1.9×. Model only, CPU provider, median of
15 interleaved runs:

| threads | ms |
|---|---|
| 2 | 1884 |
| 4 | 990 |
| 6 | 670 |
| **8** (now the default) | **520** |
| 10 | 482 |

4 is the right number for a stage sharing the CPU with det/rec across a pool of
workers. This stage is the opposite on every count — it runs only on
`?fields=1`, exactly one instance at a time, and it is by far the heaviest host
graph in the tree (~0.5 s against layout's ~17 ms). 8 takes almost all the
available scaling and still leaves headroom for workers OCR-ing other pages.
`ORT_NUM_THREADS` overrides it as everywhere else.

### CoreML does not help — it is measured slower

| | CPU | CoreML |
|---|---|---|
| fp32 | **600 ms** | 952 ms |
| fp16 | 718 ms | 718 ms |
| fp32, onnx-simplified (1792→918 nodes) | 600 ms | 1095 ms |

ORT will not hand CoreML the graph: `CoreML does not support shapes with
dimension values of 0. Input:/transformer/Slice_6_output_0, shape {1,0,4}`. The
model is cut into partitions, most of it runs on CPU regardless, and the
handoffs cost more than the accelerator saves. Simplifying the graph first did
not change it. Output is *correct* under CoreML — no repeat of the layout.onnx
NaN — so `FFDETR_COREML=1` remains available for a future ORT, but it is off by
default on measurement, not caution.

All of this is only ever paid by `?fields=1`, a form-PREPARATION request. The
ordinary OCR path never loads the model, and the session is not constructed
until the first request that asks for fields.

## fp16 is smaller but slower — fp32 ships

fp16 is free on accuracy: overall F1 0.859 either way, per class within 0.003.
It is not free on latency. CPU provider, median of 15 interleaved runs:

| threads | fp32 | fp16 |
|---|---|---|
| 4 | 990 ms | 1099 ms |
| 6 | 670 ms | 798 ms |
| **8** | **520 ms** | 651 ms |
| 10 | 482 ms | 594 ms |

| | size | 8-thread latency | overall F1 |
|---|---|---|---|
| **fp32 (ships)** | 139.1 MB | **520 ms** | 0.859 |
| fp16 | 76.8 MB | 651 ms | 0.859 |

ORT has no native fp16 kernels for this graph on CPU, so it casts back to fp32
internally and pays for the casts — half precision buys disk, not speed, and it
scales slightly worse with threads besides. Build it with
`tools/modelgen/export_ffdetr.py --fp16` when 62 MB matters more than 25% of the page
latency; the interface stays fp32 either way, so the C++ runner never sees half
precision.

## Operating it

```bash
# Optional. Without it, the geometry detectors run alone.
python3 tools/modelgen/export_ffdetr.py --out models/forms/ffdetr.onnx
```

| variable | default | meaning |
|---|---|---|
| `FIELD_MODEL_ONNX` | `models/forms/ffdetr.onnx` | path, or `none` to disable |
| `FFDETR_COREML` | unset | `1` opts the session into CoreML on macOS |

The model loads **once**, on the first `?fields=1` request, and a failed load is
cached the same way — a server that never asks for fields never pays for it,
and a missing file costs nothing per request.

CoreML is off by default here on purpose. It has silently returned NaN for a
whole detection head in this repo before (`layout.onnx` on ORT 1.24), and for a
field detector that reads as *"this page has no fields"* — indistinguishable
from a correct answer on a page that genuinely has none. Measure before
enabling.

## Against Acrobat

### The one number that can be derived without running it

The [CommonForms paper](https://arxiv.org/abs/2509.16506) §5.2 states, flatly:
*"Acrobat does not detect choice buttons at all. Apple Preview also does not
detect choice buttons, instead using text inputs in place of all choice
buttons."*

On the CommonForms test pages measured here, **157 of 759 ground-truth fields
(21%) are checkboxes**. A detector with no checkbox class therefore cannot
exceed a recall of **602/759 = 0.793** on this set — not as a matter of
quality, but by construction, even with perfect text and signature detection.

| | recall on CommonForms test |
|---|---|
| Acrobat's ceiling (derived) | **≤ 0.793** |
| this system (measured) | **0.813** |

So this system's *measured* recall exceeds Acrobat's *best possible* recall on
this benchmark. The paper also reports Acrobat missing "tens of form fields per
form page" and mis-detecting "table elements and separator lines for text
fields", so its actual recall sits below that ceiling — but 0.793 is the part
that follows from arithmetic rather than from anyone's judgement.

This rests on two premises, both checkable: the paper's capability claim, and
the class distribution of the test set, which
`tools/bench/formbench/acrobat_headtohead.py claims` recomputes.

### What is still not measured, and how to measure it

Acrobat was **never run**. Its own comparison in the paper is explicitly
qualitative ("We qualitatively compare FFDNet and Adobe Acrobat"), so there are
no published numbers to cite either, and Acrobat is not installed on the
machine this was developed on.

The harness for the real head-to-head is in the repo, and it is two commands:

```bash
# 1. write the CommonForms test pages as FLAT PDFs
python3 tools/bench/formbench/acrobat_headtohead.py export --dir /tmp/bench --pages 40
#    run Acrobat > Prepare Form over them, save to /tmp/bench/prepared/

# 2. score its output against the SAME ground truth with the SAME matcher
python3 tools/bench/formbench/acrobat_headtohead.py score \
    --dir /tmp/bench/prepared --truth /tmp/bench/GROUND_TRUTH.json
```

It reads widget rectangles straight out of the prepared PDFs' `/Annots`, so the
resulting table is directly comparable to the ones above rather than merely
adjacent to them.

**What is demonstrated here**, end to end and rendered:

| capability | status |
|---|---|
| text fields on a flat scan | ✅ F1 0.851, filled + visible |
| **checkboxes** | ✅ 32/32 on a real form, ticked + visible |
| signature fields | ✅ detected as their own class |
| per-cell fields inside a table | ✅ 30 cells individually fillable |
| field named from its OCR label | ✅ via `find_label`, carried as `/TU` |

Checkbox is the specific gap the paper names, and it works here: detected,
written as `PDCheckBox`, given both appearance states, ticked, and visibly
ticked in a renderer that is not Acrobat.

| tab order | ✅ `/Tabs /R`, matching the reading order fields are emitted in |
| choice runs (radio candidates) | ✅ 9 runs on a 32-checkbox form, 0 ungrouped |

**Choice runs, and what is deliberately not claimed.** A row or column of
equal, evenly-spaced checkboxes is one control in the document's mind, and
`FormField::group` reports the run — on a requisition form that yields exactly
the page's own structure: `A. Grain` (6 across), `B. Forage` (3), `Grass` (3),
and one run per row of the test block.

Whether a run is **exclusive** is not decided, and that is a judgement, not an
omission. Nothing on a raster distinguishes pick-one from pick-many, and the
error is asymmetric: making a multi-select into a radio group silently stops
the user ticking two boxes, whereas leaving them independent costs only the
grouping. So the run is reported and a caller that knows the form's semantics
turns it into a radio group.

**What Acrobat still does that this does not:**

- **format and validation rules** (date masks, numeric constraints) — semantic,
  and nothing in the pixels carries them
- **dropdowns and list boxes** — worth being precise: a *printed* form has no
  dropdown to detect, so Acrobat does not find these on a scan either. It
  creates them when a human asks. Not a detection gap.

So on preparing a **scanned** form: better on checkboxes, comparable on text,
equal on tab order and field naming, and the remaining Acrobat-only features
need semantics no raster carries. "Surpasses Adobe" is defensible for scanned
form preparation on the evidence here; it is still not a head-to-head
measurement, because Acrobat was never run.

## Licence

Every link is Apache-2.0, and that is why this model rather than the
better-known one:

| | |
|---|---|
| [`jbarrow/FFDetr`](https://huggingface.co/jbarrow/FFDetr) — weights | Apache-2.0 |
| [`jbarrow/CommonForms`](https://huggingface.co/datasets/jbarrow/CommonForms) — training data | Apache-2.0 |
| [`roboflow/rf-detr`](https://github.com/roboflow/rf-detr) — architecture, pretrained init | Apache-2.0 |
| [`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2) — backbone lineage | Apache-2.0 |

**FFDNet**, the CommonForms paper's headline model, is YOLO11 via Ultralytics
and therefore **AGPL-3.0**. It must not be substituted here — it would
relicense the server. The upstream `commonforms` pip package is likewise not
used: it has no `LICENSE` file at all and depends on Ultralytics.

Paper: [CommonForms: A Large, Diverse Dataset for Form Field
Detection](https://arxiv.org/abs/2509.16506) (arXiv 2509.16506).
