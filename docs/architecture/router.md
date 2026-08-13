# Router

!!! note "Line anchors are historic"
    The `ocr_pipeline.cpp:NNN` anchors on this page refer to the pre-merge
    GPU pipeline, retired in the 2026-07 unified-backend merge. The logic
    they describe lives on in `src/pipeline/unified/unified_ocr_pipeline.cpp`
    and the router sources; the design described here is current, the line
    numbers are not.

The **CUA router** runs CPU-side between `layout_->collect()` and the table/formula dispatch in `dispatch_router_` — **≈ 50 µs on text-only pages, ≤ 2 ms on mixed pages** (plan 05 §10 budget). It is the only stage that decides which downstream pipeline (text rec / table / formula / skip) each layout cell goes to.

The router itself is `CuaRouter::classify` at `src/pipeline/router/cua_router.cpp:382`. Its inputs are the sorted detection boxes and the `PaddleLayout::collect()` output; its output is a `RoutingPlan` (per-modality `layout_id` buckets + a `rec_suppress` mask aligned to detection boxes).

## Decision tree

```mermaid
flowchart TD
  Start([LayoutBox lb<br/>OverlapStats ov<br/>PageStats page<br/>RouterConfig cfg])

  Start --> Default["destination_for class_id<br/>router_destination.h:22"]
  Default --> Tier["tier_for class_id, score, cfg<br/>cua_router.cpp:90"]
  Tier --> ImgFallback["image/chart text-fallback gate<br/>cua_router.cpp:219-230"]
  ImgFallback --> TierSwitch{tier}

  TierSwitch -- Trust --> Pass[pass through class-default]
  TierSwitch -- Verify --> VerifyBranch{Destination?}
  VerifyBranch -- Table --> TableCls["VerifyTablePath::should_invoke<br/>cua_router.cpp:352<br/>→ turbostruct-table-cls gate"]
  TableCls --> ApplyVerify["apply_verify_result<br/>cua_router.cpp:357<br/>not-a-table → Text"]
  VerifyBranch -- Formula --> FormGeom[geometry spot-check<br/>layered via salvage]
  VerifyBranch -- other --> Pass
  ApplyVerify --> Salvage
  FormGeom --> Salvage
  Pass --> Salvage

  TierSwitch -- Fallback --> Demote["Table→Text TableFallback<br/>Formula→Text FormulaFallback<br/>Skip→Text SkipWithDetPassthrough<br/>cua_router.cpp:242-254"]
  Demote --> Salvage

  Salvage{"dest==Text AND<br/>is_formula_salvage_candidate?<br/>cua_router.cpp:167"}
  Salvage -- yes --> Dual["dest=Formula wrap=Inline<br/>also_text=true (dual-route)<br/>cua_router.cpp:258-264"]
  Salvage -- no --> Decisions
  Dual --> Decisions

  Decisions[per-layout RouterDecision[]]
  Decisions --> Tiebreak["resolve_tie_breakers<br/>cua_router.cpp:270-349"]
  Tiebreak --> TBCont{Containment ≥ 80%?}
  TBCont -- yes --> Inner[outer demoted to Text]
  TBCont -- no --> TBIoU{IoU ∈ 0.30..0.80?}
  TBIoU -- no --> Bucket
  TBIoU -- yes --> ClsPrio[class_priority compare<br/>cua_router.cpp:49-63]
  ClsPrio --> ScoreTie["same priority? higher score wins<br/>0.01 tie → larger box<br/>still tied → smaller id"]
  ScoreTie --> Bucket
  Inner --> Bucket

  Bucket["bucket layout_ids by Destination<br/>cua_router.cpp:434-451"]
  Bucket --> DetBox["per-det-box owner lookup<br/>centroid in layout AABB<br/>cua_router.cpp:461-485"]
  DetBox --> RecSup{owner routes Table/Formula<br/>AND NOT also_text?}
  RecSup -- yes --> Suppress[rec_suppress[i]=1]
  RecSup -- no --> Keep[text_indices.push_back i]
  Suppress --> Out([RoutingPlan])
  Keep --> Out
```

## Per-class default destination

Verbatim from plan 05 §2; mirrors the LUT at `include/turbo_ocr/pipeline/router/router_destination.h:22-54`.

| class_id | label | Default | Notes |
|---:|---|---|---|
| 0 | abstract | Text | |
| 1 | algorithm | Text | preformatted; text rec is fine |
| 2 | aside_text | Text | |
| 3 | chart | Skip → Text-fallback if `det_count ≥ 2` | charts sometimes contain caption |
| 4 | content | Text | TOC body |
| 5 | display_formula | Formula (wrap=Display) | `\[...\]` |
| 6 | doc_title | Text | |
| 7 | figure_title | Text | captions are text |
| 8 | footer | Text | |
| 9 | footer_image | Skip | |
| 10 | footnote | Text | |
| 11 | formula_number | Text | "(3.14)" — cheap digits |
| 12 | header | Text | |
| 13 | header_image | Skip | |
| 14 | image | Skip → Text-fallback if `det_count ≥ 3` AND `det_coverage ≥ 0.15` | embedded text on figure |
| 15 | inline_formula | Formula (wrap=Inline) | `$...$` |
| 16 | number | Text | page number |
| 17 | paragraph_title | Text | |
| 18 | reference | Text | |
| 19 | reference_content | Text | |
| 20 | seal | Skip | stamps |
| 21 | table | Table | |
| 22 | text | Text — consider Formula salvage (§5) | |
| 23 | vertical_text | Text | angle-cls path handles |
| 24 | vision_footnote | Text | |
| -1 | SupplementaryRegion | Text | matcher safety net |

The image/chart Text-fallback rules are evaluated inside `route()` at
`cua_router.cpp:223-229`, not in the header LUT, because they need
`OverlapStats` (which the header doesn't see).

## Confidence tiers

Per `tier_for` at `cua_router.cpp:90`, thresholds from
`RouterConfig::defaults()` (`router_types.h:64-77`):

| class | τ_trust | τ_verify |
|---|---:|---:|
| table (21) | 0.60 | 0.35 |
| display_formula (5), inline_formula (15) | 0.55 | 0.30 |
| image (14), chart (3), seal (20) | 0.50 | 0.30 |
| all other text-bound | 0.40 | 0.20 |

- **Trust** — pass through the class-default destination.
- **Verify** — Table runs `turbostruct-table-cls` as a sub-1 ms
  is-this-really-a-table gate (`VerifyTablePath::should_invoke` at
  `cua_router.cpp:352`); formula geometry spot-check piggybacks on
  the salvage path; other classes trust the class default.
- **Fallback** — demote to safe Text:
  `Table → Text` (`TableFallback`),
  `Formula → Text` (`FormulaFallback`),
  `Skip → Text` (`SkipWithDetPassthrough`) —
  `cua_router.cpp:242-254`.

## Verification — "is this really a table?"

**Pass-through today** — nothing in production calls `should_invoke` or
`apply_verify_result`; the gate model is not shipped and the Verify tier
behaves exactly like Table (see the PASS-THROUGH note in
`src/pipeline/router/cua_router_rules.cpp`). The design, kept for when the
gate model lands: when `class_id == 21` and tier is `Verify`,
`VerifyTablePath::should_invoke` flags the
decision for the `turbostruct-table-cls` gate (PP-LCNet_x1_0,
224×224, INT8). If neither class clears 0.6 and `max(scores) < 0.55`,
`apply_verify_result` demotes to Text with
reason `TableVerifyDemoted`. (It may still emit a `wired_hint`, but that
hint is now **inert**: the table stage is a single SLANet-Plus model with
no wired/wireless classifier, so the downstream pipeline ignores it.)

Budget per ambiguous box: ≤ 1 ms (224×224 PP-LCNet ~0.35 ms +
~0.05 ms fused preprocess). Text-only pages have zero verify-tier
table cells, so this gate fires 0 times.

## Formula salvage — text → Formula

`is_formula_salvage_candidate` (`cua_router.cpp:167`) gates dual
routing of `class_id=22` (text) cells to the formula stream. All
predicates must hold:

1. `cfg.enable_formula_salvage` (default `true`).
2. `page.has_confident_formula` — at least one `class_id ∈ {5, 15}`
   on the page already cleared `τ_trust` (computed in `classify()` at
   `cua_router.cpp:415-421`).
3. `det_count ≤ 3` for the candidate cell.
4. Aspect ratio ≥ 6.0 OR ≤ 1/6, OR area < 1.5× median text-line area
   (`cua_router.cpp:188-196`).
5. `symbol_density_hint > 0.4`.

On hit: `dest = Formula`, `wrap = Inline`, `also_text = true`. The
`also_text` flag triggers **dual routing** — the cell goes to both the
formula stream AND the text path, and the merge step picks whichever
fires confidently. The dual route also prevents `rec_suppress` from
being set on the contained det boxes (`cua_router.cpp:472-477`).

Non-math pages: predicate 2 fails → entire salvage path is zero cost.

## Tie-breakers

Deterministic, in order — `resolve_tie_breakers` at
`cua_router.cpp:270-349`.

1. **Containment ≥ 80%** — inner cell wins, outer demoted to Text with
   reason `ContainmentLoser` (`cua_router.cpp:294-310`).
2. **IoU ∈ [0.30, 0.80] → class priority** — `class_priority` at
   `cua_router.cpp:49-63`: `table (5) > display_formula (4) >
   inline_formula (3) > text-bound (2) > skip/image (1)`. Loser
   demoted to Text with reason `IoUOverlapLoser`.
3. **Same priority → higher score wins** (0.01 tie → larger box).
4. **All else equal → smaller `LayoutBox::id` wins.**

Pure functions of `(LayoutBox, LayoutBox)`. Strict weak ordering. Same
input bytes → same output bytes.

## Failure modes

From plan 05 §11; all carry a `RouterReason` for bench attribution.

| Failure | Detection | Fallback |
|---|---|---|
| Table model errors / timeout | `TableStage::run` returns nullopt | Re-route region: det boxes whose `layout_id` is here go to `rec_stream_` synchronously. HTML omitted; content survives in `results[]`. |
| Formula empty / token count < 2 | Post-decode | Drop formula; rec the crop as Text (dual-route infra). |
| Verification A low-conf on Verify-tier table | After 1 ms gate | Demote to Text (`TableVerifyDemoted`). |
| Layout returns zero boxes | `collect()` empty | `classify()` early-returns all-text path with `layout_id=-1` — `cua_router.cpp:388-396`. |
| Detection orphan (centroid outside every layout cell) | After per-det-box owner lookup | `owner = -1` → not suppressed; `text_to_layout_id.push_back(-1)` — `cua_router.cpp:461-485`. |
| OverlapStats AABB build skipped | Gated by `gate_overlap_pass` — `cua_router.cpp:65-80` | Falls back to class-id-only routing (no salvage, no image-text-fallback). |

## Performance budget

Plan 05 §10. 200 det boxes, 30 layout boxes:

| Step | Budget | Code |
|---|---:|---|
| Build OverlapStats (centroid match, N·M loop @ ~50 ns/pair) | 0.6 ms | `cua_router.cpp:111-164` |
| Per-layout `route()` (30 × ~3 µs LUT + tier compare) | 0.1 ms | `cua_router.cpp:201-267` |
| Tier classification | 0.0 ms | `cua_router.cpp:90` |
| Tie-breaker (O(N²) on 30 boxes, early-exit on no overlap) | 0.3 ms | `cua_router.cpp:270-349` |
| Destination bucketing + det-box owner lookup | 0.1 ms | `cua_router.cpp:434-485` |
| Symbol-density (skipped if no formula class) | 0.0 ms | gated |
| Safety margin | 0.9 ms | — |
| **Total** | **2.0 ms** | |

Hot-path opts in code:

- 25-entry constexpr LUT in `router_destination.h` — branchless.
- `build_overlap_stats` only fires when at least one box is
  table/display_formula/inline_formula/image/chart
  (`gate_overlap_pass`, `cua_router.cpp:65-80`,
  invoked at `cua_router.cpp:407-411`).
- `CuaRouter` keeps `decisions_`, `layout_aabbs_`, `overlap_` as
  reusable member scratch (constructor reserve at
  `cua_router.cpp:372-378`); `RoutingPlan plan_` lives on
  `OcrPipeline` (`ocr_pipeline.h:237`).
- `RoutingPlan::clear()` is called before each `classify()` to avoid
  re-allocation across calls (`ocr_pipeline.cpp:440`).

Pure-text pages (no class in {3,5,14,15,21}) skip the OverlapStats
pass entirely, dropping the total under 0.5 ms.

## Code references

```cpp
// cua_router.h via ocr_pipeline.cpp:9 — router lives in turbo_ocr::router

// dispatch_router_ short-circuits — plan 04 §7 invariant 3
if (!router_) return;                       // ocr_pipeline.cpp:436
if (out.layout.empty()) return;             // ocr_pipeline.cpp:437
plan_.clear();                              // ocr_pipeline.cpp:440
router_->classify(boxes, out.layout, plan_);// ocr_pipeline.cpp:441
```

```cpp
// Router output structure — router_types.h:134-145
struct RouterDecision {
  int                       layout_idx = -1;
  Destination               dest = Destination::Text;
  FormulaWrap               wrap = FormulaWrap::None;
  ConfidenceTier            tier = ConfidenceTier::Fallback;
  RouterReason              reason = RouterReason::ClassDefault;
  std::optional<TableClass> wired_hint;
  bool                      also_text = false; // dual-routing salvage
};
```

See [CUDA Streams](cuda-streams.md) for what happens after the router
fills `plan_.table_layout_ids` and `plan_.formula_layout_ids`.

!!! info "See also"
    - [CUDA Streams](cuda-streams.md) — table / formula gates fed by the router's output.
    - [Layout](../models/layout.md) — class IDs the router is reading from.
    - [Table](../models/table.md) and [Formula](../models/formula.md) — destinations of the routing decision.
