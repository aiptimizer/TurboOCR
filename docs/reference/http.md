# HTTP API

!!! abstract "TL;DR"
    `POST /ocr/raw` is the fast path; `POST /ocr/pixels` skips decoding
    entirely; `POST /ocr/batch` and `POST /ocr/pdf` cover multi-image and PDF
    inputs; `POST /ocr/markdown` exports a parsed page as Markdown. `GET /health/live` and `GET /health/ready` are
    Kubernetes-friendly probes. The Docker image is fronted by nginx on
    `http://localhost:8000`; a native build binds port **8080** directly
    (`PORT`, default 8080). The examples below use `:8000` (the Docker port) —
    use `:8080` for a native server.

One server, one API. Routes live under `src/service/http/`, one endpoint
family per file; the same binary serves whichever backends were compiled in
(`nvidia`, `apple`, `intel`, `amd`, `cpu`), so every endpoint below exists on
every backend.

| Endpoint | Purpose |
|---|---|
| `POST /ocr/raw` | OCR raw image bytes (fastest) |
| `POST /ocr` | OCR a base64 image in JSON |
| `POST /ocr/pixels` | zero-decode raw pixel buffer |
| `POST /ocr/batch` | batch of images |
| `POST /ocr/pdf` | PDF → text; `?markdown=1` → whole PDF as Markdown |
| `POST /ocr/markdown` | page → faithful Markdown (requires layout) |
| `POST /ocr/stream` | PDF → newline-delimited JSON, one event per page |
| `POST /infer` | one crop through a chosen table/formula backend |
| `GET /capabilities` · `/capabilities/backend` | runtime feature and backend discovery |
| `GET /profile` | per-stage timings of recent requests |
| `GET /metrics` | Prometheus metrics ([monitoring](monitoring.md)) |
| `GET /health` · `/health/live` · `/health/ready` | probes |

## Security model — read this before exposing the server

**TurboOCR ships no authentication, no authorization and no TLS, by design.**
It is a vLLM-class inference server: auth, TLS, exposure and rate limiting are
the fronting gateway's job. That was a deliberate decision (commit `6d23a311`
removed an in-server `API_AUTH_TOKEN` bearer mechanism and a refuse-to-boot
public-bind gate); it is not an oversight and re-adding in-server auth is not
the intended fix.

What that means concretely for a deployment:

* **`BIND_HOST` defaults to `0.0.0.0`.** The server listens on every interface
  out of the box. Set `BIND_HOST=127.0.0.1` and let your gateway reach it over
  loopback, or bind it to a private network only.
* **Every endpoint is unauthenticated** — including `/metrics`, which exposes
  VRAM and topology, and `/profile`. The bundled nginx config deliberately does
  **not** add an allow/deny for `/metrics` so it stays a drop-in for existing
  scrapers; restricting it is the gateway's job.
* **The service accepts attacker-controlled images and PDFs.** Size limits
  (`MAX_BODY_MB`, `MAX_IMAGE_DIM`, `MAX_PDF_PAGES`) bound resource use, but they
  are not an authorization boundary.
* **Do not expose it directly to the internet.** Put an authenticating reverse
  proxy or API gateway in front of it and terminate TLS there.

The bundled `docker/config/nginx.conf.template` is a performance shim (keep-alive
to Drogon, slow-loris timeouts, body-size cap) — **not** a security boundary.

## Shared query parameters

Parsed by `server::parse_query_options()` in
`include/turbo_ocr/service/validation/query_options.h`. Every parameter accepts
`1` / `true` / `on` / `yes` (or the negated equivalents).

**Query string or JSON body.** On endpoints that take a JSON body (`/ocr`), a
flag may be sent either way — `?layout=1` and `{"image": ..., "layout": true}`
are equivalent. The query string wins if both are present, since it is the more
visible form (it appears in access logs and proxies). Both go through the same
parser, so the two forms cannot behave differently.

| Param | Default | Effect |
|---|---|---|
| `layout` | `0` | Run PP-DocLayoutV3 and emit a `layout` array. |
| `autorotate` | `0` | **PDF endpoints only** (`/ocr/pdf`, `/ocr/stream`): de-rotate each page upright with the doc-orientation model before OCR. Requires that model, else `400 AUTOROTATE_DISABLED`. On image endpoints it is an unsupported parameter (ignored + `x-ignored-params` header; strict mode rejects), exactly like other PDF-only params. |
| `reading_order` | `0` | XY-cut over the layout boxes; emits `reading_order`. Auto-enables `layout`. |
| `as_blocks` | `0` | Emit paragraph-level `blocks`. Auto-enables `layout` + `reading_order`. |
| `tables` | `0` | Run the table branch (SLANeXt, or a VLM backend) and emit `tables`. Strict opt-in: `1` requires a table backend configured at startup, else `400 TABLE_BACKEND_DISABLED`. Auto-enables `layout`. |
| `formulas` | `0` | Run the formula branch (PP-FormulaNet_plus-S, in-process ORT-CUDA-13) and emit `formulas`. Strict opt-in: `1` requires a formula backend configured at startup, else `400 FORMULA_BACKEND_DISABLED`. Auto-enables `layout`. |
| `fields` | `0` | `/ocr/pdf` only. Propose fillable rectangles per page and emit a `fields` array — `text`, `checkbox` or `signature`, each with a `label`, a `confidence` and the `source` detectors that argued for it. Four geometry detectors always run; [FFDetr](../models/forms.md) joins them when its weights are present. Never forces the table stage: without `tables=1` the empty-cell detector simply contributes nothing. |
| `text` | `1` | The one opt-OUT flag. `text=0` skips text detection/recognition entirely: with `layout=1` the request is a layout-only pass (`results` comes back empty); on `/ocr/pdf`, `text=0&images=inline` is the fast page-images path with zero OCR cost, and adding `layout=1` yields layout + image per page. Rejected with `tables`/`formulas`/`as_blocks`/`reading_order` (all consume recognized text), on `/ocr/batch`, and on the CPU build. |

!!! note "Optional fields stay byte-identical when empty"
    `layout`, `reading_order`, `blocks`, `tables`, `formulas` are
    emitted only when non-empty — text-only pages produce a response
    indistinguishable from the pre-feature shape. See
    `emit_pipeline_result_json` in
    `include/turbo_ocr/serialization/serialization_emit.h`.

!!! warning "Asking for a capability this server did not load"
    Every optional capability — `layout`, `tables`, `formulas`,
    `autorotate` — is refused with `400` and its own stable error code
    when the server did not load it, never accepted-and-silently-dropped.
    `GET /capabilities` reports which are available before you send a
    request.

    | Capability | Error code |
    |---|---|
    | `layout` | `LAYOUT_DISABLED` |
    | `tables` | `TABLE_BACKEND_DISABLED` |
    | `formulas` | `FORMULA_BACKEND_DISABLED` |
    | `autorotate` | `AUTOROTATE_DISABLED` |

    Because `tables` and `formulas` both require layout, requesting either
    on a `DISABLE_LAYOUT=1` server returns `LAYOUT_DISABLED` — the
    dependency is what is actually missing. `reading_order=1` and
    `as_blocks=1` are views over layout output rather than capabilities of
    their own, so they also return `LAYOUT_DISABLED`.

    The codes, their remediation text, and the parameter names all come
    from one table:
    `include/turbo_ocr/core/capability_table.def`. Adding a row
    there makes a capability requestable on every endpoint and both
    transports, advertised in `/capabilities`, and refused with its own
    code — there is no per-endpoint wiring to keep in step.

---

## `POST /ocr`

The base64 twin of `/ocr/raw` for clients that must send JSON:

```bash
curl -X POST http://localhost:8000/ocr \
  -H 'Content-Type: application/json' \
  -d "{\"image\": \"$(base64 -w0 document.png)\"}"
```

Same query parameters, same response shape, same limits as `/ocr/raw` —
plus the base64 decode cost, which is why `/ocr/raw` is preferred.

## `POST /ocr/raw`

Raw image bytes in the request body. JPEG is GPU-decoded with nvJPEG
(falling back to OpenCV); PNG goes through the Wuffs fast path;
everything else uses `cv::imdecode`.

- **Body**: raw image bytes (`image/jpeg`, `image/png`, `image/webp`,
  `image/bmp`, `image/tiff`, `image/gif`).
- **Dim guard**: `MAX_IMAGE_DIM` (default `16384`, clamped to
  `[64, 65535]`) — pre-decode for PNG/JPEG (header sniff) and
  post-decode for the rest.

### Request

=== "bash"

    ```bash
    curl -X POST http://localhost:8000/ocr/raw \
         --data-binary @page.jpg \
         -H 'Content-Type: image/jpeg'
    ```

=== "python"

    ```python
    import requests
    with open("page.jpg", "rb") as f:
        r = requests.post(
            "http://localhost:8000/ocr/raw",
            data=f.read(),
            headers={"Content-Type": "image/jpeg"},
        )
    print(r.json()["results"][0]["text"])
    ```

=== "javascript"

    ```javascript
    const bytes = await (await fetch("page.jpg")).arrayBuffer();
    const r = await fetch("http://localhost:8000/ocr/raw", {
      method: "POST",
      headers: {"Content-Type": "image/jpeg"},
      body: bytes,
    });
    console.log((await r.json()).results[0].text);
    ```

### With layout + reading order + tables

=== "bash"

    ```bash
    curl -X POST 'http://localhost:8000/ocr/raw?layout=1&reading_order=1&tables=1' \
         --data-binary @page.png \
         -H 'Content-Type: image/png'
    ```

=== "python"

    ```python
    import requests
    with open("page.png", "rb") as f:
        r = requests.post(
            "http://localhost:8000/ocr/raw",
            params={"layout": 1, "reading_order": 1, "tables": 1},
            data=f.read(),
            headers={"Content-Type": "image/png"},
        )
    ```

=== "javascript"

    ```javascript
    const bytes = await (await fetch("page.png")).arrayBuffer();
    const r = await fetch(
      "http://localhost:8000/ocr/raw?layout=1&reading_order=1&tables=1",
      {method: "POST", headers: {"Content-Type": "image/png"}, body: bytes});
    ```

### Response shape

```json
{
  "results": [
    {"id": 0, "text": "Hello world", "confidence": 0.987,
     "bounding_box": [[12,8],[180,8],[180,32],[12,32]], "layout_id": 0}
  ],
  "layout": [
    {"id": 0, "class": "chart", "class_id": 3, "confidence": 0.94,
     "bounding_box": [[10,4],[800,4],[800,40],[10,40]]},
    {"id": 1, "class": "figure_title", "class_id": 7, "confidence": 0.69,
     "bounding_box": [[32,8],[520,8],[520,30],[32,30]], "parent_id": 0}
  ],
  "reading_order": [0],
  "tables": [],
  "formulas": []
}
```

`parent_id` is the `id` of the region that CONTAINS this one — the layout model
emits genuine children (a `figure_title` inside a `chart`/`image`, a
`formula_number` inside a `display_formula`, a `paragraph_title` inside a
`content` block), and this is how that nesting is reported instead of being
flattened. A region is assigned the SMALLEST region that contains it (>=90% of
its area), so a caption inside a figure inside a content block points at the
figure. The field is **omitted** for top-level regions, which is most of them
on an ordinary page, and the hierarchy is a forest — it can never contain a
cycle or point at a region missing from the response.

Error codes: `EMPTY_BODY`, `IMAGE_DECODE_FAILED`, `DIMENSIONS_TOO_LARGE`,
`INVALID_PARAMETER`, `LAYOUT_DISABLED`, `INFERENCE_ERROR`, `SERVER_BUSY`.

---

## `POST /ocr/pixels`

Skip image decoding entirely — caller hands the server an already-decoded
BGR or grayscale buffer. Zero decode overhead, the lowest-latency entry
point.

- **Body**: raw pixel data, exactly `width × height × channels` bytes.
- **Dimensions**: pass `?width=`&`height=` query params (preferred), or the
  legacy `X-Width` / `X-Height` headers (kept for back-compat since v2.3).
  `channels`/`X-Channels` is optional (defaults to `3`; only `1` or `3`).
  Query params win when both are supplied — **unless they disagree**, which
  returns `400 DIMENSION_CONFLICT`. Missing width/height → `400 MISSING_DIMENSIONS`.
  Requests that use the legacy `X-*` headers get a `Deprecation: true` response
  header (RFC 8594) plus an `X-Deprecation-Notice` pointing to the query params.
- **Opt-in params** (`layout`, `reading_order`, `tables`, `formulas`, …) are
  query params, same as every other route.
- **Dim guard**: same `MAX_IMAGE_DIM` as `/ocr/raw`.

### Request

=== "bash"

    ```bash
    curl -X POST 'http://localhost:8000/ocr/pixels?width=1280&height=720&channels=3' \
         -H 'Content-Type: application/octet-stream' \
         --data-binary @frame.bgr
    # (legacy, still supported: -H 'X-Width: 1280' -H 'X-Height: 720' instead of the query params)
    ```

=== "python"

    ```python
    import cv2, requests
    img = cv2.imread("frame.png")  # BGR uint8
    h, w, c = img.shape
    r = requests.post(
        f"http://localhost:8000/ocr/pixels?width={w}&height={h}&channels={c}",
        data=img.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )
    ```

=== "javascript"

    ```javascript
    // bgr = Uint8Array length = w*h*3
    await fetch(`http://localhost:8000/ocr/pixels?width=${w}&height=${h}&channels=3`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: bgr,
    });
    ```

Response shape is identical to `/ocr/raw`.

Error codes: `MISSING_DIMENSIONS` (no width/height via query or header),
`DIMENSION_CONFLICT` (query param and `X-*` header disagree), `INVALID_DIMENSIONS`,
`DIMENSIONS_TOO_LARGE`, `PIXELS_TOO_LARGE`, `BODY_SIZE_MISMATCH`, plus the shared set.

---

## `POST /ocr/batch`

JSON array of base64-encoded images. Decoded with nvJPEG batch decode
when ≥2 inputs are JPEG; mixed batches fall back to per-slot decode.

!!! tip "Per-slot error alignment"
    Per-slot errors keep the response array aligned with the input order
    so a single bad image never silently drops the rest of the batch.
    Successful slots get `null`; failed slots get a string tag like
    `"base64_decode_failed"` or `"dimensions_too_large (32000x32000 > 16384x16384)"`.

### Request

=== "bash"

    ```bash
    curl -X POST 'http://localhost:8000/ocr/batch?layout=1' \
         -H 'Content-Type: application/json' \
         -d '{"images": ["'$(base64 -w0 page1.jpg)'", "'$(base64 -w0 page2.jpg)'"]}'
    ```

=== "python"

    ```python
    import base64, json, requests
    images = [base64.b64encode(open(p, "rb").read()).decode()
              for p in ("page1.jpg", "page2.jpg")]
    r = requests.post(
        "http://localhost:8000/ocr/batch",
        params={"layout": 1},
        json={"images": images},
    )
    ```

=== "javascript"

    ```javascript
    const toB64 = async (path) => {
      const buf = new Uint8Array(await (await fetch(path)).arrayBuffer());
      return btoa(String.fromCharCode(...buf));
    };
    const body = {images: [await toB64("page1.jpg"), await toB64("page2.jpg")]};
    await fetch("http://localhost:8000/ocr/batch?layout=1", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    ```

### Response shape

```json
{
  "batch_results": [
    {"results": [{"text": "page1 line", "confidence": 0.96,
                  "bounding_box": [[0,0],[200,0],[200,30],[0,30]]}],
     "layout": []},
    {"results": [], "layout": []}
  ],
  "errors": [null, "decode_failed"]
}
```

Whole-request 400s: `INVALID_JSON`, `EMPTY_BATCH`, `INVALID_PARAMETER`.

---

## `POST /ocr/pdf`

Accepts a PDF as raw body, base64 JSON, or multipart. Renders pages with
`fastpdf2png` (PDFium-backed) and runs the pipeline per page, with an
optional PDFium text-layer fast path that avoids OCR entirely when the
embedded text is trustworthy.

- **Body**: one of
    - raw bytes (`application/pdf`),
    - JSON `{"pdf": "<base64>"}`,
    - multipart with field name `file` or `pdf`.
- **`mode` query** (default `ocr`):
    - `ocr` — always render + OCR every page.
    - `geometric` — prose comes **only** from the PDF text layer; the page is
      **never OCR'd**. A page with no usable text layer (image-only / scanned)
      returns empty prose — use `auto` or `ocr` for those. The page is still
      rendered for layout and table/formula recognition when requested (those
      are vision-recognized for born-digital pages too), so `?tables=1`/
      `?formulas=1` work in geometric mode and keep the exact text-layer prose.
    - `auto` — text layer when trusted (`text_layer_quality == "trusted"`),
      OCR otherwise. This is the mode that recovers prose on image-only pages.
    - `auto_verified` — GPU only. Runs OCR, then cross-checks every
      detection against the PDF text layer; replaces matches with the
      native string (`source: "pdf"`). On CPU this aliases to `auto`.
- **`dpi` query**: 50–600 (default `100`).

!!! warning "Page cap"
    `MAX_PDF_PAGES` defaults to `2000`. Exceeding returns
    `400 PDF_TOO_LARGE` with the limit echoed back in the message.

### Searchable PDF (`?output=pdf`)

Returns the source document with the recognised words stamped on as an
invisible text layer (`application/pdf`), instead of the JSON envelope. This
is the "scanned PDF in, searchable PDF out" path — the consumer no longer has
to build a text layer itself.

```bash
curl -X POST 'http://localhost:8000/ocr/pdf?output=pdf&mode=auto' \
  --data-binary @scan.pdf -H 'Content-Type: application/pdf' -o searchable.pdf
```

- The visible pages are **untouched**: vector art, images, bookmarks and
  metadata all survive, and the save is incremental, so cost scales with what
  was added rather than with document size.
- Words the server took from the document's own text layer (`source: "pdf"`)
  are **not** overlaid — they are already searchable, and stamping them would
  duplicate every hit.
- Every script works. The text layer uses a 668-byte glyphless CID font with
  a generated `ToUnicode` CMap, so Latin, CJK, Arabic, Greek and astral-plane
  characters all extract correctly without embedding megabytes of fonts.
- `&min_confidence=0.0–1.0` drops recognised words below that score before
  stamping. Only valid with `output=pdf`.
- `&autorotate=1` also turns the output page upright, so a sideways scan
  arrives readable rather than merely searchable.
- With `&layout=1`, each detected figure, chart, table and seal is marked with
  an invisible annotation carrying its label, so a reader can select the region
  and a downstream tool can crop it without re-running detection.
- Rejected with `markdown=1` and with `text=0` (both need the text).
  `X-OCR-Dropped-Words` reports any run that had no usable geometry.

### Editable PDF (`?output=pdf&editable=1`)

Draws the recognised words as **real type in place of the print**, instead of
hiding them behind it. The page stops being a picture of text and becomes text:
it can be retyped in any PDF editor, and it still looks like the original.

```bash
curl -X POST 'http://localhost:8000/ocr/pdf?output=pdf&editable=1' \
  --data-binary @scan.pdf -H 'Content-Type: application/pdf' -o editable.pdf
```

The typeface is **identified by setting the words again**. Every candidate font
renders the very text the recogniser read, and the renderings are compared pixel
against pixel with the crop those words came from; the closest wins. It never
has to name what makes Georgia Georgia, because it is looking at Georgia while
it decides. Alongside it the page is still *measured* — stroke weight, stem
slant, the spread of a stem where it meets the baseline, the ink and paper
colours — and that measurement is fed in as a prior rather than as an answer.

Accuracy, over the 90 text faces installed on a development machine, labelled
from each font's own PANOSE and `post` metadata and rendered at scan resolution
with blur, sensor noise and JPEG:

| method | family | notes |
|---|---|---|
| measured features alone | 95.6% | reads Times New Roman and Georgia as *sans* |
| shape matching alone | 92.2% | gets both right; misses slab serifs |
| **both, as shipped** | **96.7%** | sans 100%, serif 86%, mono 100% |

Given a catalogue holding all 97 faces rather than the standard-14, the same
matcher reads the **family correctly 100%** of the time and names the **exact
face 70%** of the time — and nearly every remaining miss is between faces that
share one Latin design (Apple's Hiragino family all resolve to the same Latin
letterforms) or differ only in weight.

The whole document votes once on a family, so every line that looks alike gets
the *same* font rather than drifting page to page. Bold and italic stay per
line, judged against the document's own median — which is also what stops a scan
that went through the feeder 3° crooked from coming back entirely in italics.

Detection costs about **24 ms per page** on top of OCR.

Lines are matched to the PDF standard-14 faces (Helvetica / Times / Courier and
their bold and oblique variants). Those need no embedded font file and are
metrically compatible with the faces scanned business documents are actually set
in, so the common case costs no bytes and carries no font licence.

The matcher itself is not limited to those — it can load any `.ttf`, `.otf` or
`.ttc` and will pull a single face out of a collection when it has to, which is
how the 97-font figure above was measured. Recognising what a scan was set in
and **embedding** that font are different permissions, though: reading a font to
identify a typeface is not copying it, while writing it into the output is. Only
fonts whose licence permits embedding and redistribution belong in a shipped
catalogue, which is why the one that ships is standard-14 and nothing else.

Two kinds of line are deliberately **left exactly as scanned**, and both stay
searchable through the ordinary invisible layer:

- text sitting on ruled, shaded or printed ground, where covering the original
  would destroy page content the OCR never claimed to have read;
- text needing characters the standard-14 fonts do not have — they are Latin-1
  only, so Greek, Cyrillic and CJK are left to the scan rather than drawn as the
  wrong glyphs.

The server log reports the split as `visible` and `left_as_scan`.

Note that `editable=1` **changes the page**, which plain `output=pdf` never
does. It is rejected without `output=pdf`, and rejected with `text=0`.

### Movable figures (`?output=pdf&layout=1&movable=1`)

Cuts each figure, chart, table and seal the layout model found OUT of the page
raster and re-places it as its own image object, over a patch of the page's own
paper colour.

```bash
curl -X POST 'http://localhost:8000/ocr/pdf?output=pdf&layout=1&movable=1' \
  --data-binary @scan.pdf -H 'Content-Type: application/pdf' -o movable.pdf
```

Nothing changes to look at. What changes is what can be done with it: an editor
that drags one of those objects drags the chart, and what it leaves behind is
clean paper rather than the original showing through the gap. Marking a region
with an annotation — which `layout=1` alone does — can only ever drag an
outline, because the chart is still pixels inside one flat picture underneath.

Regions smaller than 24 px a side are skipped as detection noise, and a region
covering more than 90% of the sheet is skipped because lifting it out would
replace the scan with a copy of itself. Text regions are never lifted: they are
the text layer's business, and a picture of the words on top of the words helps
nobody. Each page has a 12 MB ceiling on lifted imagery, largest region first,
so a dense page cannot multiply the file size.

**Printed rules become shapes too.** Every table border, underline and panel
line is found by morphology and redrawn as a real filled path over a patch of
paper, in the colour it was printed in. A rule that is ink in the page image can
be looked at and nothing else; the same rule as a path can be selected, moved,
recoloured and deleted. On a ruled requisition form that is 41 rules, 2 figures
and 170 words — every mark on the page an object.

A rule is told from a letter stroke by being far longer than it is thick
(20:1 and up). Length alone is not enough: a column of tall lowercase `l`s runs
as far down the page as a short table border, and was being lifted as one.

Needs `layout=1` — the figures to lift are the ones layout detection finds — and
is rejected without it rather than silently doing nothing. Add `&mark_regions=0`
to suppress the outline annotation each region otherwise gets: wanting a figure
to be movable is not the same as wanting a box drawn round it.

### PDF → Markdown (`?markdown=1`)

One call converts the whole PDF to the same faithful Markdown as
`POST /ocr/markdown`, using the parallel page pipeline — no client-side
page splitting. Available on both the GPU and CPU builds:

```bash
curl -X POST 'http://localhost:8000/ocr/pdf?markdown=1' \
  --data-binary @paper.pdf -H 'Content-Type: application/pdf'
```

- Returns `text/markdown`: pages concatenated in order, each prefixed with an
  invisible `<!-- page N -->` marker (safe to render, easy to split on).
- `&as_pages=1` returns JSON instead — `{"pages":[{"page_index":0,
  "markdown":"…"}, …]}` — for chunked/RAG consumers; per-page
  `text_degraded` / `table_degraded` / `formula_degraded` flags appear when set.
- Implies `layout=1&reading_order=1` (requires the layout model —
  `400 LAYOUT_DISABLED` otherwise). In the default `ocr` mode, tables and
  formulas are recognized whenever their backends are loaded (→ HTML / LaTeX);
  pass `tables=0` / `formulas=0` to opt out.
- **Mode choice.** `mode=ocr` (default) renders and OCRs every page — best for
  scanned PDFs. `mode=geometric` reads a born-digital PDF's embedded text layer
  directly for the prose (exact text, faster, no OCR errors) while **still**
  recognizing tables → HTML and formulas → LaTeX on the rendered image — so a
  born-digital paper exports exact text *and* structured math/tables. `auto`
  picks per page: text layer when trustworthy, OCR otherwise.
- Figure/chart crops are embedded as base64 `data:` URIs with their OCR'd text
  as alt text; tables come back as HTML, formulas as `$…$` / `$$…$$` LaTeX.
- `dpi`, `mode` and `autorotate` work as usual. `text=0` and `images=` are
  rejected (`400 INVALID_PARAMETER`) — markdown needs the text, and the figure
  crops are already embedded.
- Per-stage degradation is aggregated in the `X-OCR-Degraded` response header
  with page numbers (e.g. `table(p3,p7)`), same contract as `/ocr/markdown`.

### Inline page-image export

`?images=inline` adds each rendered page back to the response as a
base64-encoded image, alongside the OCR result for that page. Two extra
fields appear per page: `image_b64` (the encoded bytes) and
`image_content_type` (the matching MIME type, e.g. `image/png`).

Encoding is controlled by these query parameters (parsed by
`parse_image_query_params()` in `src/service/http/pdf/pdf_request.cpp`):

| Param | Default | Effect |
|---|---|---|
| `format` | `png` | Output codec: `png`, `jpeg`, or `webp`. JPEG is GPU-encoded via nvJPEG (see `TURBO_PDF_IMAGE_ENCODER`). |
| `quality` | — | Lossy quality `1`–`100` (JPEG/WebP). Setting it implies `lossless=0` unless `lossless` is given explicitly. |
| `lossless` | `1` | WebP lossless mode. Ignored for JPEG. |
| `png_compression` | — | PNG zlib level `0`–`9` (higher = smaller, slower). |
| `max_side` | `0` | Downscale so the larger page dimension is at most `N` pixels before encoding. `0` keeps full resolution. |

Out-of-range values return `400 INVALID_PARAMETER` (e.g.
`quality must be 1-100`, `png_compression must be 0-9`,
`max_side must be >= 0`).

```bash
curl -X POST 'http://localhost:8000/ocr/pdf?images=inline&format=jpeg&quality=80&max_side=2048' \
     --data-binary @doc.pdf -H 'Content-Type: application/pdf'
```

### Auto-rotation

`?autorotate=1` corrects scanned or rotated pages before detection. The
`PP-LCNet_x1_0_doc_ori` model classifies each page's rotation
(`0` / `90` / `180` / `270`) and the page is turned upright **first**, so
the returned image, boxes, and text all come back in a single upright
frame. Each affected page reports the detected clockwise rotation in an
`orientation_deg` field. Pages handled purely from the PDF text layer
(`geometric`) and pages with a native `/Rotate` flag are skipped.

!!! warning "AUTOROTATE_DISABLED"
    `autorotate=1` requires the doc-orientation model
    (`models/doc_ori.onnx`, pointed to by `DOC_ORI_ONNX`). Against a
    server started without it, the request returns
    `400 AUTOROTATE_DISABLED` — mirroring the `LAYOUT_DISABLED` contract.

```bash
curl -X POST 'http://localhost:8000/ocr/pdf?autorotate=1&images=inline' \
     --data-binary @scan.pdf -H 'Content-Type: application/pdf'
```

### Request

=== "bash"

    ```bash
    curl -X POST 'http://localhost:8000/ocr/pdf?mode=auto&dpi=150&layout=1' \
         --data-binary @doc.pdf -H 'Content-Type: application/pdf'
    ```

=== "python"

    ```python
    import requests
    with open("doc.pdf", "rb") as f:
        r = requests.post(
            "http://localhost:8000/ocr/pdf",
            params={"mode": "auto", "dpi": 150, "layout": 1},
            data=f.read(),
            headers={"Content-Type": "application/pdf"},
        )
    ```

=== "javascript"

    ```javascript
    const bytes = await (await fetch("doc.pdf")).arrayBuffer();
    await fetch("http://localhost:8000/ocr/pdf?mode=auto&dpi=150&layout=1", {
      method: "POST",
      headers: {"Content-Type": "application/pdf"},
      body: bytes,
    });
    ```

### Response shape

```json
{
  "pages": [
    {
      "page": 1, "page_index": 0,
      "dpi": 150, "width": 1240, "height": 1754,
      "results": [
        {"text": "Title", "confidence": 1.0,
         "bounding_box": [[50,80],[420,80],[420,120],[50,120]],
         "source": "pdf"}
      ],
      "layout": [],
      "mode": "geometric",
      "text_layer_quality": "trusted"
    }
  ]
}
```

`source` is `"ocr"` (omitted) for pixel-derived text or `"pdf"` for
text-layer (or `auto_verified`-promoted) entries. `text_layer_quality` is
one of `"absent"`, `"rejected"`, `"trusted"` (see
`text_layer_quality_for()` in `src/pipeline/job/pdf_job_pages.cpp`). With
`images=inline` each page additionally carries `image_b64` +
`image_content_type`; with `autorotate=1` each de-rotated page carries
`orientation_deg`.

Error codes: `MISSING_PDF`, `MISSING_FILE`, `INVALID_MULTIPART`,
`BASE64_DECODE_FAILED`, `EMPTY_BODY`, `EMPTY_PDF`, `INVALID_DPI`,
`INVALID_PARAMETER`, `PDF_TOO_LARGE`, `PDF_RENDER_FAILED`,
`AUTOROTATE_DISABLED`.

---

## `POST /ocr/stream`

Available on every backend (the unified server registers it
unconditionally). **One streaming endpoint for PDFs and single images**
(content-sniffed by magic bytes): the response is `application/x-ndjson` —
one JSON object per line, flushed as produced. Built for streaming consumers:
a RAG ingester can start chunking/embedding page 1 while pages 2…N are still
being OCR'd, instead of waiting for the whole document.

Accepts the same query parameters as `/ocr/pdf` (`layout`, `text`, `tables`,
`formulas`, `dpi`, `mode`, `images=inline`, `autorotate`, …).

Line protocol:

```text
{"event":"meta","kind":"pdf","pages":500,"dpi":100,"mode":"ocr"}
{"event":"page", ...same shape as an /ocr/pdf pages[] element...}
{"event":"page_error","page_index":17}
{"event":"error","code":"..."}            # job-level failure mid-stream
{"event":"end","pages":500,"failed":0}
```

Page events arrive **as each page completes — out of order** (that is the
point); `page_index` identifies the page, so reorder client-side if you need
to, or don't (embedding doesn't care). Single images produce exactly
`meta` → `page` → `end`. Errors detected before the first byte are normal
HTTP 4xx; once streaming has begun they arrive as `error` events (the 200 is
already on the wire — chunked transfer has no second status line).

```bash
curl -sN -X POST 'http://localhost:8000/ocr/stream?layout=1' \
     --data-binary @doc.pdf -H 'Content-Type: application/pdf' |
  while read -r line; do echo "$line" | jq -r '.event'; done
```

```python
with requests.post(url + "/ocr/stream", data=pdf_bytes, stream=True,
                   headers={"Content-Type": "application/pdf"}) as r:
    for line in r.iter_lines():
        ev = json.loads(line)
        if ev["event"] == "page":
            embed_page(ev["page_index"], ev["results"])   # overlap with OCR
```

!!! warning "Keep-alive required"
    Chunked streaming needs a keep-alive connection. A request carrying
    `Connection: close` (python `urllib` sends this) is rejected with a 400
    rather than returning a silently-empty body — use `requests`/`httpx`.

## `POST /ocr/markdown`

Available on every backend. Runs the full pipeline (layout + reading order forced on) and
returns the page as **faithful Markdown** instead of JSON — the in-process
counterpart of PP-StructureV3 `save_to_markdown`. See
[Faithful Markdown export](markdown-output.md) for the serialization rules.

- **Body**: raw image bytes (same decoders as `/ocr/raw`).
- **`embed` query** (default `true`): figure/chart crops are always inlined as
  base64 `data:` URIs (self-contained `.md`). `embed=1` is accepted explicitly;
  `embed=0` (file-reference links) is **rejected with
  `400 INVALID_PARAMETER`** — the asset PNGs would be written to the server's
  filesystem where an HTTP client cannot retrieve them. File-reference export
  is available to library consumers via `render_markdown_with_assets`.
- **Requires layout**: against a server started with `DISABLE_LAYOUT=1` the
  request returns `400 LAYOUT_DISABLED`.
- **Response**: `text/markdown; charset=utf-8`.

```bash
# self-contained markdown (images inline as data URIs)
curl --data-binary @page.png http://localhost:8000/ocr/markdown > page.md
```

Error codes: `EMPTY_BODY`, `LAYOUT_DISABLED`, `IMAGE_DECODE_FAILED`,
`DIMENSIONS_TOO_LARGE`, plus the shared inference set.

---

## Health probes

### `GET /health/live`

Liveness probe. Always returns `200 ok` once the process is up.

```bash
curl http://localhost:8000/health/live
```

### `GET /health/ready`

Readiness probe. Invokes the `readiness_check` closure passed in by
`main.cpp` — verifies the pipeline can submit work without blocking, so
it correctly stays `503 NOT_READY` while TensorRT engines are being
built on first start.

```bash
curl http://localhost:8000/health/ready
```

!!! info "nginx 502 handling"
    The bundled `docker/config/nginx.conf.template` PRESERVES upstream
    **502** (it does NOT remap to 503) and returns a JSON error body; it
    sets no `Retry-After` header — clients should apply their own backoff
    while engines build
    (`docker/config/nginx.conf.template:45-53`).

## `GET /capabilities`

Reports what the running server actually loaded — use it to check whether
`tables`/`formulas` will work before sending `?tables=1`/`?formulas=1` (those
return `400 *_BACKEND_DISABLED` when the backend isn't loaded).

```bash
curl http://localhost:8000/capabilities
```

```json
{
  "is_gpu": true,
  "features": { "layout": true, "tables": true, "formulas": true, "autorotate": false },
  "routing": { "routes": { "table": "table-env", "formula": "formula-env", "text": "default" } }
}
```

`features.{layout,tables,formulas,autorotate}` reflect the loaded stages (formula
and table are `true` only when their backend env var was set at startup). The
`routing` block lists backend **names + kinds only** — never URLs/keys.

## `GET /capabilities/backend`

Unified server only. Reports which vendor backend actually came up, how it
behaves, and which backends this binary could have run at all.

```bash
curl http://localhost:8000/capabilities/backend
```

```json
{
  "backend": "apple",
  "device": "metal",
  "async": true,
  "native_image_decode": false,
  "supports_batch": true,
  "pool_size": 4,
  "available_backends": ["apple", "cpu"]
}
```

`backend` and `device` are also in `/capabilities`; the rest is only here.
`device` is the memory space the stages run in (`host` / `cuda` / `metal` /
`hip` / `l0`), `async` whether its queues are asynchronous,
`native_image_decode` whether it has an on-device decoder (nvJPEG, vImage),
`supports_batch` whether it does native batched inference. `pool_size` is how
many pipelines the pool holds — the number of requests that can be in
inference at once. `available_backends` lists every vendor backend linked into
this binary, by canonical name and in the order auto-detect tries them, so it
is the set `--backend` picks from (each also answers to its aliases, e.g.
`metal` for `apple`).

The document is built once at startup and is immutable for the process
lifetime.

## `POST /infer`

Runs **one crop** through a chosen table or formula backend — the low-level
building block behind `?tables=1`/`?formulas=1`, exposed for callers that
already know the region:

```json
{ "image": "<base64 crop>", "modality": "table" | "formula",
  "backend": "<registry name>" }
```

Inline ad-hoc backend specs (operator-supplied `base_url`, an SSRF surface)
are rejected unless `TURBO_ALLOW_ADHOC_BACKENDS=1`.

## `GET /profile`

Per-stage timings for recent requests (detection, classification,
recognition, layout, table, formula), as JSON — the first place to look when
latency moves. Prometheus counterparts live on
[`/metrics`](monitoring.md).

## Error envelope

Every error response carries the shared envelope:

```json
{"error": {"code": "DIMENSIONS_TOO_LARGE",
           "message": "Image dimensions 32000x32000 exceed maximum of 16384x16384"}}
```

Codes are documented inline in `proto/ocr.proto:12-18` and are the same
strings the gRPC server surfaces in the `x-error-code` trailing metadata
field.

!!! info "See also"
    - [gRPC API](grpc.md) — same surface, protobuf-shaped.
    - [CUA router](../architecture/router.md) — what `tables=1` /
      `formulas=1` actually trigger.
    - [Build → Docker](../getting-started/docker.md) — env-var matrix and nginx
      template behaviour.
