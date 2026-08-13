# `src/` — what each top-level directory is for

Each directory below has **one** reason to exist. This file states the rule for
each, because the previous layout had three names that stated no rule at all
(`common/`, `runtime/`, `io/`) and each of them accumulated files that no rule
would have admitted.

`src/` and `include/turbo_ocr/` mirror each other: a public header's directory
and its implementation's directory have the same name. The exceptions, each for
a stated reason:

- `src/backends/` has no header mirror — it is private by construction, and its
  vendor headers are included as `"nvidia/..."` relative to `src/backends`.
- `include/turbo_ocr/core/`, `include/turbo_ocr/serialization/` and
  `include/turbo_ocr/base/` have no source mirror — all three are header-only.
  For the first two that is a rule; for `base/` it is simply where it ended up
  once `cjk_stats` (which is OCR policy, in `namespace turbo_ocr::formula`)
  moved to `analysis/formula/` where the rule below already put it.

| dir | rule |
|---|---|
| `analysis/` | one subdirectory per document-analysis task — detection, recognition, classification, layout, table, formula, forms, vlm. Each holds the model wrapper **and** its pre/post algorithms, together. |
| `backend/` | implementations of the seam contract in `include/turbo_ocr/backend/`: the link-time backend registry and routing config. Vendor-neutral. |
| `backends/` | one subdirectory per **vendor** (nvidia, amd, intel, apple, cpu). Device-specific code only; nothing outside a vendor arm may include from one. |
| `base/` | foundation with **zero** domain knowledge: geometry, logging, serialization, errors, env, string/encoding, uuid, order statistics. If it mentions OCR, it does not belong here. |
| `document/` | the assembled document — block assembly, reading structure — and its markdown serialization. |
| `image/` | image codec, both directions: decode in, encode out. |
| `onnx/` | the ONNX Runtime session layer: session creation, execution-provider options, host thread configuration. Every model wrapper in `analysis/` sits on it, and each vendor drives it with a different `EpConfig`. |
| `pdf/` | the PDF medium, both directions: page rasterization and text-layer extraction in, searchable-PDF out. |
| `pipeline/` | orchestration — staging, batching, pooling, job lifecycle. Device-free. Its `job/` subdirectory holds the multi-page PDF job; it is named for the RULE (`job lifecycle`), not the medium, because `pdf/` above already owns the medium and two directories called `pdf` stated nothing about which was which. |
| `service/` | transport only: HTTP routes, gRPC service, server bootstrap. |

Three header-only peers have no `src/` half:

- `include/turbo_ocr/core/` — the domain vocabulary: `OCRResultItem`, the
  request/response, capability, catalog and normalization types every layer may
  include and which include nothing back. It is the OCR-specific counterpart to
  `base/`, which knows no OCR at all.
- `include/turbo_ocr/serialization/` — how those domain objects become the wire
  envelope. Consumed by HTTP, gRPC and the Python binding alike, so it belongs to
  none of them.
- `include/turbo_ocr/base/` — header-only by outcome rather than by rule. Its
  last `.cpp`, `cjk_stats`, was OCR policy sitting in `namespace
  turbo_ocr::formula` inside the directory whose whole rule is "if it mentions
  OCR, it does not belong here"; it now lives in `analysis/formula/`. The same
  rule evicted `CudaError`/`HipError` into their vendor arms.

Both used to sit in `common/`, which is how `common/` stayed a junk drawer: a
`base/` holding `OCRResultItem` would have been the same drawer under a better
name. The rule is what evicts them, not the rename.

## `backend/` vs `backends/`

One letter apart, so the distinction has to be stated. `include/turbo_ocr/backend/`
is the **contract** — the `backend::` namespace: `Backend`, `IKernels`,
`IDetector`, `StageSet`, `BackendCaps`. `src/backend/` implements that contract
vendor-neutrally. `src/backends/<vendor>/` implements it *per device*.

The mirror rule is what disambiguates them in practice: `src/backend/` has a
matching `include/turbo_ocr/backend/`; `src/backends/` deliberately does not.

These two files used to sit under `src/runtime/`, whose README recorded that they
did not satisfy that directory's own rule and that the intended fix was exactly
this move. They also do not share a target — `backend_registry.cpp` compiles into
`turbo_ocr_pipeline`, `routing_config.cpp` into `turbo_ocr_common`. That follows
from what links the registry, and is not a reason to file them apart.

## Why `analysis/` and not `models/`

Over half of what is in there is not a model: `ctc_decode`, `det_postprocess`,
`xy_cut`, `reading_order`, `html_reconstruct`, `latex_normalize`, `cell_matcher`
and the rest are pure algorithms over tensors and boxes. Grouping them by task
next to the wrapper they serve is deliberate — the CTC decoder belongs beside
the recognizer, not in a separate `algorithms/` tree.
