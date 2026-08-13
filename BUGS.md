# TurboOCR — defect catalogue (second pass) + RECONCILIATION

> **RECONCILIATION (2026-08-03, HEAD `99600d21`+).** This pass was written
> against a tree that was being FIXED while it read: the header below claims
> `580b38c5`, but several entries quote code that only exists in the 19-commit
> fix campaign after it (`dffe38ad`..`99600d21`). Two consequences, both
> corrected in place:
>
> 1. **Five of the seven "Refuted" entries are wrong about being wrong.**
>    R2–R6 cite the fix campaign's own patches (and in two cases quote the
>    fixes' comments verbatim) as evidence the first-pass findings never
>    existed. Those findings were real at `580b38c5`, were independently
>    verified, and were FIXED — see the corrected § Refuted. R1 (Metal
>    half-pixel) and R7 (preprocess_region, in part) are genuine refutations.
> 2. **Every entry now carries a STATUS line** (FIXED @ commit / OPEN /
>    BY-DOCUMENTED-CHOICE). The gate criticals C1–C5, C7–C8 and highs H1–H7,
>    H9 were all fixed and verified (locally + on the RTX 5090: FUNSD-50
>    nvidia tiny F1 85.37% @ 678.6 img/s pooled, golden all-stage clean)
>    before this document was written.
>
> The genuinely NEW findings of this pass — M13 (drop_score floor), M14
> (autorotate dead-AND), L1 (float [0,1] images), L2 (tiny-bigdet row), L3
> (turbo aliases), L5 (ORT skew), L8 (fallback never read), M7 (~PdfDocument
> mutex) — were all fixed in the reconciliation commit.

Branch `backup/arch-restructure-2026-08-01` @ `580b38c5` (see reconciliation note)

**This pass supersedes the first.** The first pass used nine parallel agents and marked
roughly a third of its findings `[reported]` — found by an agent, not independently confirmed.
This pass was done single-threaded by me, and **every entry below was re-proved against the
code.** There are no unverified entries.

That re-proving mattered more than expected: **seven first-pass findings turned out to be
wrong or stale**, including two ranked CRITICAL. They are listed in
[§ Refuted](#refuted) with the evidence, so nobody acts on them.

**Status markers used below:**
- **PROVED** — I executed something that demonstrates it.
- **VERIFIED** — I read the code and proved the claim.

*Excluded by design decision: the absence of an auth/TLS layer. That is a deliberate product
choice, not a defect, and is not listed.*

---

## Summary

| | count |
|---|---|
| Critical | 8 |
| High | 9 |
| Medium | 14 |
| False comments / docs | 18 |
| Dead code | 5 |
| **Refuted from pass 1** | **7** |

The dominant theme is unchanged and is worth stating first: **the gate layer does not gate.**
Four separate gates are structurally incapable of failing, one test suite scores scrambled
output as perfect, and `ctest` registers a single test. Everything else in this document was
found by reading, because the automation that should have found it cannot fail.

---

# CRITICAL

## C1 — `ctest` registers exactly one test
**STATUS: FIXED @ dffe38ad — accuracy-gates CI job + FUNSD fetch script; 13 ctest gates register with the cache.**
**PROVED**

```
$ ctest --test-dir build-cpu -N
  Test #1: turbo_ocr_unit

Total Tests: 1
```

`CMakeLists.txt:1690-1766` registers every accuracy and conformance gate inside
`if(TURBO_FUNSD_CACHE)`. No CI configure sets it — `ci.yml:31`, `:52`, `:242`.

**Never executed:** `backend_conformance`, every `golden_<backend>_<stage>`, and all five
`funsd_*_gate` F1 floors (85.7 / 92.5 / 85.2 / 85.0 / 85.0).

`tests/cpp/backends/harness.h` is 1028 lines; `turbo_bench`, `turbo_conformance` and
`turbo_golden` are 1462 lines between them. All of it feeds gates that never run.

**Failure scenario:** any accuracy regression on any backend ships green — a detector
threshold change, a broken CTC decode, a backend returning empty boxes. The F1 floors that
would catch it are evaluated on no machine, in CI or locally.

**Fix:** cache a 10-image FUNSD subset as a CI artifact and pass `-DTURBO_FUNSD_CACHE=<path>`.
If the corpus is too large, gate only the F1 floors on it and run conformance and golden
tests against synthetic fixtures.

---

## C2 — the AMD/Intel compile gate passes with a nonexistent compiler
**STATUS: FIXED @ dffe38ad — exit-status verdict + hard FATAL on a broken CXX (re-proved live).**
**PROVED**

`tools/syntax_shims/check.sh:51-58`:
```bash
  out=$("$CXX" -fsyntax-only -DPROTOBUF_USE_DLLS "${INC[@]}" -std=gnu++20 -w "$f" 2>&1 |
        grep 'error:')
  if [ -z "$out" ]; then
    echo OK
```

```
$ CXX=definitely-not-a-compiler bash tools/syntax_shims/check.sh \
    src/backends/intel/engine/openvino_engine.cpp
src/backends/intel/engine/openvino_engine.cpp        OK
EXIT=0
```

Pass/fail comes from grepping merged output for the literal string `error:`. The compiler's
exit status is discarded. With a missing compiler, the shell writes `command not found` to
stderr, the grep finds nothing, and the script prints `OK` for every source.

**Why this is critical:** this script is the **only** gate covering the AMD and Intel arms.
Neither compiles in any other CI job. CI installs `clang-18` via apt and passes
`CXX=clang++-18` — any packaging drift, rename, or failed install turns the gate into an
unconditional pass while still printing `OK` for all 37 listed sources.

**Fix:**
```bash
  "$CXX" -fsyntax-only ... "$f" 2>&1 | tee /tmp/o
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then echo OK; else echo FAIL; fail=1; fi
```
Also assert `"$CXX" --version` succeeds at the top, so a missing toolchain is a hard error.

---

## C3 — the accuracy metric is order-blind set membership
**STATUS: FIXED @ dffe38ad — ordered LCS ratio; every adversarial case lands under the 0.70 floor.**
**VERIFIED**

`tests/regression/test_accuracy_regression.py:26-33`:
```python
def _char_accuracy(expected, detected):
    if not expected:
        return 1.0
    e = expected.lower().replace(" ", "")
    d = detected.lower().replace(" ", "")
    if not e:
        return 1.0
    return sum(1 for c in e if c in d) / len(e)
```

`c in d` tests membership against the whole detected string. It measures neither order, nor
count, nor recall.

- `"HELLO"` vs `"OLLEH"` → 1.0
- `"12345"` vs `"54321"` → 1.0
- Any superset → 1.0. `"HELLO"` vs `"abcdefghijklmnopqrstuvwxyz"` → 1.0
- `"AAA"` vs `"A"` → 1.0

**Failure scenario:** the suite is parametrized over `["HELLO","WORLD","12345","ABCDEF",
"Testing"]` with `MIN_RECALL = 0.70`. A model returning the fixed string
`"abcdefghijklmnopqrstuvwxyz0123456789"` for every image passes all five at 1.0. Guarding
against a totally broken model is the suite's stated purpose and precisely what it cannot do.

**Fix:** an ordered metric — `1.0 - edit_distance(e, d) / max(len(e), 1)`, or LCS length over
`len(e)` if a dependency is unwanted.

---

## C4 — `backend_probe` is a test that cannot fail
**STATUS: FIXED @ dffe38ad — probe fails on empty registry / cpu-auto not constructing / 'nope' constructing.**
**VERIFIED**

`tests/cpp/backends/backend_probe.cpp:55-66`, registered at `CMakeLists.txt:1668`:
```cpp
  try_one("", "\"\" (auto-detect)");
  for (auto n : names) try_one(std::string(n), std::string(n).c_str());
  try_one("metal", "\"metal\" (alias)");
  try_one("host",  "\"host\" (alias)");
  try_one("nope",  "\"nope\" (unknown)");
  for (int i = 1; i < argc; ++i) try_one(argv[i], argv[i]);
  return 0;
}
```
Every return value is discarded; `main` returns 0 unconditionally. Inside `try_one`, the null
branch (`:37`) and the catch branch (`:47`) also return 0.

**It passes if:** `available_backends()` is empty, every factory returns `nullptr`, every
factory throws, or `make_backend("nope")` returns non-null.

Its own header comment (lines 13-17) says it exists to catch the WHOLE_ARCHIVE
registrar-stripping regression — which manifests as an empty `available_backends()`, the exact
case it passes on.

**Fix:** make `try_one` return `bool`, accumulate failures, assert `names` is non-empty and
`make_backend("nope") == nullptr`, `return failures != 0`.

---

## C5 — CI is red because a file move invalidated a ratchet exemption
**STATUS: FIXED @ dffe38ad — path corrected, staleness guard added, the two real violations SPLIT; ratchet green.**
**VERIFIED**

`gh run list`: the last four completed runs on this branch are all `failure`. Run
`30762082779`: `build-and-test` FAILED, `vendor-syntax-check` FAILED.

`vendor-syntax-check` runs the ratchet with no escape hatch (`ci.yml:284-285`):
```yaml
      - name: Function-length ratchet
        run: python3 tools/checks/function_size.py
```

Current violations against the 180-line limit:
```
  src/pipeline/job/pdf_job.cpp:119 (243 lines)
  src/service/server/bootstrap/server_config.cpp:138 (273 lines)
  src/service/server/unified/server_main.cpp:162 (186 lines)
```

**Root cause** — `function_size.py:26` exempts a path that does not exist:
```python
allowed = {
  "src/service/server/server_config.cpp",     # <-- no such file
```
```
$ ls src/service/server/server_config.cpp
ls: No such file or directory
$ ls src/service/server/bootstrap/server_config.cpp
src/service/server/bootstrap/server_config.cpp
```
The file moved into `bootstrap/`. The ratchet matches on exact string, so the move silently
un-exempted it.

The script's docstring says *"entries come off as functions shrink and none are ever added to
make a build pass."* It guards against entries being **added**. Nothing guards against an
entry ceasing to match anything — which is what happened.

**Fix:** correct the path, and add a staleness check:
```python
stale = allowed - set(all_scanned_paths)
if stale:
    print("ratchet entries match no file (renamed or deleted):")
    for s in sorted(stale): print("  " + s)
    sys.exit(1)
```
Then split or explicitly ratchet the two genuinely new violations.

---

## C6 — the `clang-tidy` job cannot fail
**STATUS: PARTLY FIXED @ 4319d4b9 — scope + glob corrected (100 files linted); advisory `|| true` remains BY DOCUMENTED CHOICE until the backlog is triaged.**
**VERIFIED**

`ci.yml:295+`:
```bash
clang-tidy-18 -p build --quiet "${files[@]}" 2>&1 | tee tidy.log || true
echo "findings: $(grep -c 'warning:' tidy.log || true)"
```

Three independent reasons it can never go red: `|| true` swallows the exit status; the count
is echoed and never compared; `.clang-tidy` sets `WarningsAsErrors: ''`.

**Evidence:** on run `30762082779`, where two other jobs FAILED, `clang-tidy` reported
**success**.

`.clang-tidy`'s own header essay diagnoses this exact failure from project history:
> "Nothing ran it, so nobody found that out. Wildcarding every family and then not enabling
> the tool is the same as having no tool, with extra config."

The config was fixed in response. The enforcement was not. The current state is the same
failure with a 30-minute CI job attached.

**Fix:** baseline the count and fail above it; ratchet down.

---

## C7 — AMD silently disables tables and formulas
**STATUS: FIXED @ 87af2f56 — AMD Local specs resolve to the host recognizers like every other arm.**
**VERIFIED**

`src/backends/amd/backend/rocm_backend.cpp:109-125`:
```cpp
RocmBackend::make_table_recognizer(const backend_routing::BackendSpec &spec) {
  // TODO(on-hardware): ... until then the shared factory yields the host/OTSL
  // path so tables still work (just not device-resident).
  return backend::make_table_recognizer(spec);
}
```

What the shared factory does — `src/pipeline/unified/vlm_factory.cpp:262-280`:
```cpp
// ... are answered by the Backend itself — CpuBackend returns its
// CpuTableRecognizer, CudaBackend its Nv* wrappers — BEFORE reaching this
// factory. Returning nullptr for a Local spec here is therefore correct: it
// means "no backend claimed it", and the pipeline disables the modality.

make_table_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind != backend_routing::Kind::Openai) return nullptr;
```

AMD delegates `Local` specs to a factory documented to return `nullptr` for exactly those
specs. **Every other arm special-cases the default local key first:**
`cuda_backend.cpp:150-154`, `cpu_backend.cpp:74-76`, `intel_backend.cpp:179-181`,
`apple_backend.mm:123-125` — Intel and Apple both fall back to the CPU host recognizer, which
is what AMD's comment claims to do and does not.

**Failure scenario:** `--backend amd` with `want_tables`/`want_formulas` and the default env
(`TABLE_BACKEND=slanext`, `FORMULA_BACKEND=ppformulanet_s`) returns `nullptr` for both; the
pipeline disables both modalities; requests come back empty with no error.

Latent today (AMD is never run on hardware) but a bring-up landmine — the first person to test
tables on ROCm will chase a nonexistent MIGraphX problem. See also [F1](#f1).

**Fix:**
```cpp
  if (spec.kind == backend_routing::Kind::Local &&
      (spec.engine.empty() || spec.engine == "slanext"))
    return std::make_unique<cpu::CpuTableRecognizer>(...);
  return backend::make_table_recognizer(spec);
```

---

## C8 — `HIP_CHECK` does not implement the policy its own header specifies
**STATUS: FIXED @ 87af2f56 — is_sticky_hip_error + abort_on_sticky_hip_fault; two-tier policy implemented.**
**VERIFIED**

`src/backends/amd/support/hip_check.h:3-10` promises:
> "an ordinary device-runtime error THROWS … and only a STICKY fault terminates the process.
> This file previously called std::abort() on EVERY HIP error … Generic policy is shared,
> never forked per backend."

Lines 30-38 implement:
```cpp
inline void hip_check_impl(hipError_t err, const char *expr, const char *file, int line) {
  if (err != hipSuccess) {
    auto msg = std::format(...);
    std::cerr << msg << '\n';
    throw HipError(msg);
  }
}
```
Every error throws; sticky faults are not distinguished.

```
$ grep -rn 'is_sticky\|abort_on_sticky' src/ include/
src/backends/nvidia/support/cuda_check.h:41  is_sticky_cuda_error(cudaError_t)
src/backends/nvidia/support/cuda_check.h:68  abort_on_sticky_cuda_fault(const char *)
src/backends/nvidia/stages/paddle_rec.cpp:71
src/backends/nvidia/stages/slanext_table_recognizer.cpp:84
src/backends/nvidia/engine/trt_engine.cpp:227,304
```
Only the CUDA pair exists, with four call sites. No HIP equivalent anywhere.

**This is a half-applied fix.** The file used to `abort()` on every error. That was corrected
to throw on every error — fixing the over-reaction and leaving the under-reaction. The policy
has two tiers; one was implemented.

**Failure scenario:** a genuine ROCm sticky fault (illegal address, launch failure,
uncorrectable ECC — HIP has 1:1 equivalents of every code in `cuda_check.h`'s list) throws,
the per-request handler catches it as recoverable and returns 5xx, and the poisoned HIP
context keeps serving every subsequent request. See also [F2](#f2).

**Fix:** port `is_sticky_hip_error` / `abort_on_sticky_hip_fault` and call them at the same
four call-site shapes CUDA uses.

---

# HIGH

## H1 — OCR'd and extracted text reaches Markdown output unescaped
**STATUS: FIXED @ c509b69d — emit_text entity-escapes with leading-marker neutralization.**
**VERIFIED**

`src/document/markdown/markdown_export.cpp:486-512`:
```cpp
void emit_text(const EmitContext &ctx, int li, const std::string &label, EmitState &st) {
  std::string text = gather(ctx, li);
  ...
  if (label == "doc_title")            st.parts.push_back("# " + text);
  else if (label == "paragraph_title") st.parts.push_back("## " + text);
  ...
  else                                 st.parts.push_back(text);
}
```
`text` is pushed raw — no escaping of `<`, `>`, `&`, no neutralisation of leading markers.

**The same file escapes everything else:**
- `:482` captions — `"![" + escape_md_link_text(caption) + "](" + src + ")"`
- `:95` tables — `return turbo_ocr::table::sanitize_table_html(inner);`
- `:404` formulas — gated on `latex_is_render_safe(latex)`

`emit_text` handles paragraphs, headings, abstracts, references and list items — the majority
of a document — and is the only path with no gate.

**Input is fully attacker-controlled:** in `mode=auto`, native PDF text is extracted
byte-exact. There is no OCR noise to mangle a payload.

**Reachable from six call sites**, all including `markdown_export.h`:
`markdown_route.cpp` (`POST /ocr/markdown`), `pdf_route.cpp`, `pdf_json.cpp`,
`pdf_request.cpp` (`/ocr/pdf?markdown=1`), `batch_common.cpp`, `recognize_markdown_rpc.cpp`.

**Failure scenario:** a PDF drawing the literal string `<img src=x onerror=…>` is returned
verbatim in the Markdown response. Any consumer rendering it to HTML — CommonMark and GFM both
pass raw inline HTML through by default — executes it.

**Fix:** `std::string text = escape_md_text(gather(ctx, li));` where `escape_md_text` replaces
`&` first, then `<` and `>`, and backslash-escapes a leading `#`, `-`, `>`, `|`, `*`.

---

## H2 — the `algorithm` code fence can be escaped by its own content
**STATUS: FIXED @ c509b69d — fenced_block sizes the fence past any embedded run (algorithm + latex fallback + inline_code).**
**VERIFIED**

`markdown_export.cpp:503`:
```cpp
  } else if (label == "algorithm") {
    st.parts.push_back("```\n" + text + "\n```");
```
A fixed three-backtick fence around arbitrary text, with no check for an embedded
triple-backtick. If `text` contains one, the fence terminates early and the remainder parses
as live Markdown — inside a block meant to be inert.

**The correct pattern already exists here:** `markdown_internal.h:236-241` defines
`inline_code()`, which widens the delimiter when it finds backticks. It is not used.

**Failure scenario:** a scanned page containing a Markdown fence — common in technical PDFs —
silently breaks output structure. With `text` attacker-controlled (H1), it is an injection
primitive that defeats the reader's assumption that fenced content is safe.

**Fix:**
```cpp
std::size_t run = longest_backtick_run(text);
std::string fence(std::max<std::size_t>(3, run + 1), '`');
st.parts.push_back(fence + "\n" + text + "\n" + fence);
```

---

## H3 — `/ocr/stream` decodes before checking size; every sibling checks first
**STATUS: FIXED @ c509b69d — stage-1 pre-decode sniff added; PDFium moved off the IO thread as well.**
**VERIFIED**

`src/service/http/pdf/stream_route.cpp:384-390`:
```cpp
                  cv::Mat img = decode(
                      reinterpret_cast<const unsigned char *>(body->data()),
                      body->size());                     // decode HERE
                  const int kMaxImageDim = decode::max_image_dim();
                  if (img.empty() || img.cols > kMaxImageDim ||
                      img.rows > kMaxImageDim ||
                      decode::exceeds_pixel_cap(img.cols, img.rows)) {   // checked AFTER
```

Every sibling calls the header sniff **before** the decoder — `infer_route.cpp`,
`markdown_route.cpp`, `ocr_base64_route.cpp`, `raw/raw_route.cpp` all call
`reject_if_too_large_pre(...)`. `batch/batch_common.cpp:67-75` adds a pre-decode
`peek_image_dimensions` **and** an aggregate `max_batch_pixels()` budget.

`size_guards.h:27-33` states the threat:
> "Pre-decode header sniff (PNG / JPEG): refuses oversized inputs without ever calling the
> decoder, defending against decompression bombs (a 1 KB PNG can claim 100k×100k → 30 GB
> decode buffer)."

`stream_route.cpp` implements stage 2 only.

**Exposure by path:**

| path | capped before allocation? |
|---|---|
| PNG on CPU backend | **Yes** — `fast_png_decoder.cpp:56-59, 86-92` self-caps |
| JPEG / TIFF / WebP / BMP, any backend | **No** — `cpu_image_decode.h:21-26` routes non-PNG to bare `cv::imdecode` |
| every format on Intel / Apple / AMD | **No** — bare `cv::imdecode` (`intel_backend.cpp:264-273`, `apple_backend.mm:426-433`, `rocm_backend.cpp:200-208`) |

**Failure scenario:** a ~2 KB JPEG whose SOF header declares 60000×60000. `cv::imdecode`
attempts ~10.8 GB on a `WorkPool` thread before line 388 runs. `MAX_BODY_MB` is no defence — a
bomb is small by definition.

**Fix:** add `reject_if_too_large_pre(...)` before line 384, emitting the error as an NDJSON
stream line to match this route's shape.

---

## H4 — a detector under-return is reported as a genuinely blank page
**STATUS: FIXED @ 46bbc4cd — shorted slots re-run per image (DetectionBatcher's recovery).**
**VERIFIED**

`src/pipeline/unified/unified_pipeline_batch.cpp:117-128`:
```cpp
  if (all_det.size() != static_cast<std::size_t>(n)) {
    TOCR_LOG_ERROR("IDetector::run_batch violated its size contract", ...);
    all_det.resize(static_cast<std::size_t>(n));      // pads with EMPTY vector<Box>
  }
```

The out-of-bounds read is correctly prevented, but the padding is indistinguishable from a
real result.

**Why the degradation signal never fires** — `ocr_pipeline_detail.h:45-54`:
```cpp
inline void flag_text_degraded(OcrPipelineResult &out, std::size_t num_boxes) {
  if (num_boxes > 0 && out.results.empty()) {
```
Padded pages have `boxes.size() == 0`, so the guard is false.

**The asymmetry:** the parallel `rec_->run_multi()` under-return *is* caught, because real
boxes survive to compare against a short result list. The detection path destroys the evidence
needed to detect its own failure.

**The fix already exists elsewhere:** `stage_batcher.cpp:238-263` handles the identical
violation by re-running each slot individually and incrementing `n_batch_fallback_`.

**Failure scenario:** five images batched, backend returns four without throwing. Page 4 gets
an empty box list, recognition runs on nothing, client receives HTTP 200 with an empty result —
identical to a blank page. Only trace is one log line.

**Fix:**
```cpp
  const std::size_t got = all_det.size();
  if (got != static_cast<std::size_t>(n)) {
    TOCR_LOG_ERROR(...);
    all_det.resize(static_cast<std::size_t>(n));
    for (std::size_t i = got; i < static_cast<std::size_t>(n); ++i)
      all_det[i] = det_->run(views[i], dims[i].first, dims[i].second, *queue_);
  }
```

---

## H5 — formula recognizer reads past the end of its token buffer
**STATUS: FIXED @ 46bbc4cd — rows out-param + clamp at both call sites.**
**VERIFIED**

Producer — `src/analysis/formula/ppformulanet/ort_session.cpp:213-221`:
```cpp
    L = shape.size() >= 2 ? shape[1] : 0;
    // Copy exactly what the model returned: an output batch smaller than the
    // requested B would over-read ORT's buffer if we trusted the input B.
    const int64_t rows = shape.empty() ? 0 : shape[0];
    const size_t emitted = static_cast<size_t>(rows) * static_cast<size_t>(L);
    tokens.assign(d, d + emitted);
```

Consumer — `src/analysis/formula/ppformulanet/ort_formula_recognizer.cpp:128-134`:
```cpp
    if (!fused_.run_tokens("x", "fetch_name_0", in_base, B, flat, L)) { ... continue; }
    for (int i = 0; i < B; ++i) {
      extract_content_seq(flat.data() + (size_t)i * L, L, EOS, seq);
```

`run_tokens` computes `rows`, uses it to size `flat`, then **discards it** — `rows` is a local,
not an out-parameter. The caller has only `B`, the *requested* size, and indexes to `B-1`.

The producer's comment shows the author knew `rows` can be less than `B`. The over-read was
moved from ORT's buffer into `flat` rather than eliminated.

**Failure scenario:** a chunk with `B=8` where the graph returns `rows=5`. `flat` holds `5*L`
`int64_t`. The loop reads at offsets `5*L`, `6*L`, `7*L` — up to `3*L*8` bytes past the
allocation. `extract_content_seq` then scans for `EOS` in memory it does not own: a crash, or
plausible-looking garbage LaTeX emitted for real formula regions.

**Fix:**
```cpp
bool run_tokens(..., std::vector<int64_t> &tokens, int64_t &L, int64_t &rows);
// caller
const int n_out = static_cast<int>(std::min<int64_t>(rows, B));
for (int i = 0; i < n_out; ++i) { ... }
// slots [n_out, B) stay empty — log it, don't silently skip
```

---

## H6 — HTML table placement can overwrite a cell whose rowspan is still active
**STATUS: FIXED @ 46bbc4cd — whole-span occupancy check with restart-past-blocker; regression test added.**
**VERIFIED**

`src/analysis/table/table_cells.cpp:70-83`:
```cpp
  auto place = [&](std::size_t slot, int rowspan, int colspan) {
    if (cur_row < 0 || slot >= out.size()) return;
    while (cursor < free_at.size() && free_at[cursor] > cur_row) ++cursor;
    const std::size_t c0 = cursor;
    const std::size_t end = c0 + static_cast<std::size_t>(colspan);
    if (free_at.size() < end) free_at.resize(end, 0);
    for (std::size_t k = c0; k < end; ++k) free_at[k] = cur_row + rowspan;
```

The scan finds the first free column `c0`, then writes the whole span `[c0, c0+colspan)`
**without checking whether any interior column is occupied**. `free_at[c]` holds "the first row
at which column `c` is free again", so an interior column with `free_at[k] > cur_row` belongs
to a live rowspan from an earlier row. The loop overwrites it.

The invariant this breaks is stated directly above:
> "`free_at[c]` is the first row index at which column c is free again, so a rowspan from an
> earlier row keeps later rows off it — the same occupancy rule a browser applies."

A browser shifts right rather than placing over an occupied column.

**Failure scenario:**
```
row 0:  A(rowspan=2,col0)  B(rowspan=1,col1)  C(rowspan=3,col2)  D(rowspan=1,col3)
row 1:  one cell, colspan=3
```
After row 0, `free_at = [2,1,3,1]`. At `cur_row=1` the scan skips col0 (`2 > 1`) and stops at
col1 (`1` is not `> 1`). So `c0=1`, `end=4`, and the loop sets `free_at[1..3] = 2` — clobbering
`free_at[2]=3`, C's live rowspan. The new cell reports `col=1,colspan=3` while C still reports
`col=2,rowspan=3`; they overlap at (row1,col2). No crash; the table geometry is silently wrong.

**Fix:**
```cpp
    std::size_t c0 = cursor;
    for (;;) {
      while (c0 < free_at.size() && free_at[c0] > cur_row) ++c0;
      std::size_t k = c0, end = c0 + static_cast<std::size_t>(colspan);
      while (k < end && (k >= free_at.size() || free_at[k] <= cur_row)) ++k;
      if (k == end) break;
      c0 = k + 1;
    }
```

---

## H7 — Intel's `caps()` never downgrades to ONNX mode
**STATUS: FIXED @ 87af2f56 — caps() keys on native_device(), downgrades on onnx mode.**
**VERIFIED**

`src/backends/intel/backend/intel_backend.cpp:96-108`:
```cpp
backend::BackendCaps IntelBackend::caps() const {
  const auto &I = *impl_;
  backend::BackendCaps c;
  const bool device_path = (I.device != OpenVINOEngine::DeviceType::CPU) &&
                           I.alloc->has_device();
  c.device = device_path ? backend::DeviceKind::L0 : backend::DeviceKind::Host;
  ...
  c.async = device_path;
  c.supports_batch = true;
```
`grep 'c.mode' intel_backend.cpp` → **no match.** `c.mode` keeps the `BackendCaps` default
(`EngineMode::Onnx`) regardless of what `load_stages()` resolved at `:206`.

**What every other arm does** — `rocm_backend.cpp:61-77`, matched by `cuda_backend.cpp:83-100`:
```cpp
  // ONNX MODE MUST DOWNGRADE. This is not cosmetic: UnifiedOcrPipeline picks its
  // staging path from caps().device ... In onnx mode this backend hands out
  // cpu::HostDeviceQueue + cpu::HostAllocator, so claiming a device here made the
  // pipeline run the device path over host memory and tag the ImageView with a
  // device kind it does not have.
  //
  // Also the honesty contract at backend.h:158-160: an Auto run that fell back to
  // onnx must SAY onnx. AppleBackend was the only arm doing this; the rule is
  // shared, so it belongs on every arm.
  c.mode = mode_;
  c.has_native_engine = native_device_();
  if (!native_device_()) {
    c.device = backend::DeviceKind::Host;
    c.async = false;
    c.supports_batch = false;
  }
```
Intel has neither the assignment nor the downgrade.

**Compounding — two different device signals.** `caps()` tests `I.alloc->has_device()`, a
SYCL/Level-Zero USM probe (`l0_allocator.cpp:51`, false whenever built without SYCL). But
`load_stages()` picked the mode from `OpenVINOEngine::device_available(I.device)`, an
`ov::Core::get_available_devices()` plugin query (`intel_backend.cpp:205`,
`openvino_engine.cpp:84-99`). `intel_backend_registry.cpp` documents that these disagree:
*"Losing L0 costs zero-copy, not the device."*

**Failure scenario:** OpenVINO's GPU plugin is unavailable → mode resolves to Onnx →
`make_queue()`, `allocator()`, `make_kernels()` all return Host implementations. But
`alloc->has_device()` is still true, so `caps()` reports `device=L0, async=true`. The pipeline
stages through the device ring over host memory and tags every `ImageView` with a `DeviceKind`
the backend does not have — verbatim the AppleBackend bug already fixed on three arms.

**Fix:**
```cpp
  const bool native = I.native_device();     // the SAME signal load_stages() used
  c.mode = I.mode;
  c.has_native_engine = native;
  const bool device_path = native && I.alloc->has_device();
  if (!native) { c.device = backend::DeviceKind::Host; c.async = false; c.supports_batch = false; }
```

---

## H8 — no pytest job exists in CI
**STATUS: OPEN — no pytest job yet (the C++ accuracy gates landed; the Python surface still runs nowhere in CI).**
**VERIFIED**

```
$ grep -n 'pytest\|python3 ' .github/workflows/ci.yml
63:  python3 tests/e2e/docker_endpoint_matrix.py --base-url http://127.0.0.1:18080
322:  python3 tools/checks/function_size.py
325:  python3 src/backends/layout_check.py
```
The only test-running Python invocation is line 63, inside `cpu-smoke`, gated
`if: ${{ vars.MODELS_RELEASE_URL != '' }}`.

**Executed nowhere:** `tests/integration/**` (~40 files), `tests/accuracy/**`,
`tests/regression/**` (including C3's suite), `tests/stress/**`,
`python/tests/test_smoke.py`. The nanobind `_turboocr` extension is never built in CI, so the
entire Python surface is unexercised.

**Fix:** add a job that builds `_turboocr` and runs
`pytest tests/accuracy tests/regression python/tests`.

---

## H9 — Python's default model runs the detector at the wrong threshold
**STATUS: FIXED @ 3c623e71 — bootstrap-installed det base; server + Python binding + bench harness all install it.**
**VERIFIED** (each link) · numeric delta unmeasured

The chain, link by link:

1. `python/turboocr/catalog.py:50` has the right value:
   ```python
   V6_DET_TINY = DetConfig(DetResizeParams("min", 64, 1280), DbParams(0.2, 0.40, 1.4))
   ```
   matching `model_catalog.h:37-38` `kV6DetConfigTiny`.
2. `catalog.py:80` attaches it to `tiny`; `catalog.py:73` and `pipeline.py:129` make `tiny` the
   **default model**.
3. **Nothing reads it.** `grep -rn det_cfg python/` returns only the dataclass field at
   `catalog.py:66` and a DESIGN.md mention.
4. `BackendConfig` has no det-config field — `grep -n 'det_' backend.h` returns only
   `std::string det_model;`.
5. `src/backends/cpu/stages/cpu_stages.cpp:20` calls the one-argument overload:
   ```cpp
   ready_ = det_.load_model(model_path);
   ```
   while `ort_paddle_det.h:35` declares `load_model(path, <config>)`. With no config the
   detector takes `kDbDefaults` — **box_thresh 0.45**.

**The codebase documents this exact failure** — `model_catalog.h:88-94`:
> "…the same mechanism silently runs det_tiny at box_thresh 0.45 instead of its own 0.40
> whenever the override goes the other way. A registry row carries det_path AND det_cfg
> together, so the pairing is correct by construction."

The C++ registry row solves it; the Python path never reaches the registry row.

**Failure scenario:** every default `turboocr.OCR()` run mis-thresholds `det_tiny.onnx` at
0.45, producing a different box count than the server for the same image and model.

**Fix:** add a `DetInferConfig` field to `BackendConfig` and plumb it to the two-argument
`load_model`. Then delete the dead Python config classes ([D3](#d3)).

---

# MEDIUM

## M1 — the seam's parameter-contract guard has a loophole that disables it
**STATUS: FIXED @ 87af2f56 — both seam kernels honor params.min_unclipped_side (caps true); component budget static_assert-pinned.**
**VERIFIED**

`include/turbo_ocr/backend/kernels.h:296-314`:
```cpp
  if (p.oriented && !s.db_oriented)
    return report_unhonoured(op, "DbPostParams::oriented=true");     // unconditional — correct
  ...
  if (!s.db_side_limits &&
      (p.min_box_side != detection::kMinBoxSide ||
       p.min_unclipped_side != detection::kMinUnclippedSide))        // <-- only if non-default
    return report_unhonoured(op, "DbPostParams::min_box_side/min_unclipped_side");
  if (!s.db_max_components && p.max_components != detection::kMaxDbComponents)
    return report_unhonoured(op, "DbPostParams::max_components");
```

Three of five checks are conjoined with "…and the value differs from the default." A backend
declaring `db_side_limits = false` — *"I cannot honour side limits"* — is refused only if the
caller asked for a **non-default** limit. Every caller uses `db_post_params()`, which passes
exactly the defaults. The condition is never true and the guard never fires.

The first two checks show the correct pattern: unconditional on the capability flag.

**Consequence:** the seam reports these parameters as honoured for the only configuration that
reaches it — the silent substitution the header says it prevents. Same loophole on
`db_expand_limits` and `db_max_components`.

**Fix:** refuse unconditionally when the capability is absent, matching `db_oriented`. Expect
this to start refusing calls that currently succeed — that is the point.

---

## M2 — two different functions share the name `snap_det_canvas`
**STATUS: FIXED @ 87af2f56 — pick_det_canvas / snap_det_canvas_grid.**
**VERIFIED**

| location | input | behaviour |
|---|---|---|
| `include/turbo_ocr/core/db_post_config.h:53` | **original** page dims + span of export canvases | nearest-by-aspect-ratio |
| `include/turbo_ocr/analysis/detection/det_config.h:192` | **already-resized** dims | round up to a 128 grid |

Both are `turbo_ocr::detection::snap_det_canvas`. `db_post_config.h` includes `det_config.h`,
and `db_post_config.h` is pulled in by `backend/kernels.h:61` — so both overloads are visible
in nearly every TU, and a two-argument call silently resolves to the grid version.

Current callers are both correct: Apple (`mps_detector.mm:263`) passes original dims to the
span overload; Intel (`intel_stages.cpp:166`) passes resized dims to the grid overload. **The
name is the bug** — a caller passing original dims to the two-argument overload compiles
cleanly and gets a canvas several times too large. Both files' comments call theirs "the
SHARED one" ([F9](#f9)).

**Fix:** rename to `pick_det_canvas(orig_h, orig_w, available)` and
`snap_det_canvas_grid(resized_h, resized_w, policy)`.

---

## M3 — NVIDIA's registrar never declines, so auto-detect can starve a real GPU
**STATUS: FIXED @ 4319d4b9 — registrar auto_usable probe (cudaGetDeviceCount); named selection untouched.**
**VERIFIED** · conditional on a build linking both arms

```cpp
// nv_backend_registry.cpp:24-27
std::unique_ptr<backend::Backend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();      // unconditional — never probes, never declines
}
```
```cpp
// amd_backend_registry.cpp:19-21
std::unique_ptr<backend::Backend> make_amd_backend_entry() {
  return make_rocm_backend();                  // nullptr when no HIP device
}
```
`backend_registry.cpp:129-142` walks candidates in descending priority and takes the first
non-null. NVIDIA is highest priority and never returns null, so no lower-priority backend is
ever tried; `CudaBackend` then degrades to CPU/ONNX internally.

**The registry's own comment, immediately above that loop, warns about exactly this:**
> "…a driver/library mismatch, a Metal device that would not open, or a failed Level Zero init
> turns into a host-backend server — an order-of-magnitude throughput loss on a machine that
> has the hardware, with zero operator signal, because the only startup line afterwards names
> whoever WON."

**Failure scenario:** an AMD-GPU-only machine, both arms linked, no `--backend` flag.
Auto-detect reports "nvidia", runs on CPU, never tries AMD.

**Conditional:** confirm a both-arms build is producible before prioritising.

**Fix:** gate NVIDIA's factory on `cudaGetDeviceCount() > 0`. Better, have factories report
whether they obtained a device so auto-detect can prefer one that did.

---

## M4 — AMD hand-rolls the pool-sizing ladder the shared helper owns
**STATUS: FIXED @ 87af2f56 — shared compute_pipeline_pool_size.**
**VERIFIED**

`src/backends/amd/backend/rocm_backend.cpp:51-58`:
```cpp
  // VRAM-tier pool sizing (mirror the CUDA VRAM-tier heuristic).
  if (p_->have_device) {
    size_t freeb = 0, totalb = 0;
    if (hipMemGetInfo(&freeb, &totalb) == hipSuccess) {
      const double gb = static_cast<double>(totalb) / (1024.0*1024.0*1024.0);
      c.recommended_pool_size = gb >= 48 ? 4 : gb >= 24 ? 3 : gb >= 12 ? 2 : 1;
    }
  }
```

`include/turbo_ocr/pipeline/pool_sizing.h:65-90`, called by `cuda_backend.cpp:70-82`:
```cpp
  if (vram_gb >= 14) pool_size = 5;
  else if (vram_gb >= 12) pool_size = 3;
  else if (vram_gb >= 8)  pool_size = 2;
  else                     pool_size = 1;
  // Footprint-based safety floor: ... a card that *reports* 16 GB while another
  // process already holds most of it would OOM during warmup. ... reduce the tier
  // so it fits in the FREE VRAM measured right now.
```

**Two divergences:** the tiers differ entirely (14/12/8 vs 48/24/12), and AMD reads `freeb`
from `hipMemGetInfo` and **never uses it** — only `totalb` feeds the ladder, so there is no
free-VRAM safety floor. That floor exists because, per its own comment, the flat tier "duly
died with 'CUDA Error … out of memory' at startup."

The header states the intent: *"nothing about it is transport, and nothing about it is CUDA
either: it is arithmetic over two memory numbers."* See [F10](#f10).

**Failure scenario:** a 16 GB AMD card with another process holding 10 GB picks pool size 2
from total VRAM, ignores the 6 GB actually free, and OOMs at warmup.

**Fix:** `c.recommended_pool_size = pipeline::compute_pipeline_pool_size(freeb, totalb);`

---

## M5 — NVIDIA is the least capable backend for full-frame normalization
**STATUS: OPEN (guarded) — parameterizing preprocess_kernels.cu is an on-hardware TODO; refusal path correct.**
**VERIFIED**

`src/backends/nvidia/kernels_cuda/cuda_kernels.cpp:120-160` — the CUDA full-frame preprocessor
has its constants baked into the `.cu` and serves exactly two distributions; anything else is
refused. Its own comment names the backends that do better:
> "Every other backend honours `order` (host_kernels.cpp, shaders.metal, sycl_kernels.cpp,
> preprocess_kernels.hip)."

Confirmed: `preprocess_kernels.hip` takes `rgb_out` at lines 72, 141, 167, 235, 329.
`grep 'rgb_out' src/backends/nvidia/` returns only the TODO comment.

The only backend the README describes as shipped is the one that can refuse a request every
other backend accepts — and the AMD port, never run on hardware, is strictly more capable than
the original it was ported from.

**Why MEDIUM, not higher:** it is handled correctly. The refusal routes through the shared
`backend::refuse_unbaked_norm` guard driven by `caps().params.norm_mean_std_full_frame = false`,
not a hand-rolled check — and the comment explains that hand-rolling it previously left the
capability flag unread. There is an owned TODO. A known, guarded gap.

**Fix:** port the `rgb_out` parameterization back from the `.hip`, delete the whitelist, flip
the capability flag.

---

## M6 — `infer_batch_view` hands out a view into shared member state
**STATUS: OPEN (documented) — contract stated at engine.h; per-replica ownership audit still worth an assert.**
**VERIFIED** (mechanism) · reachability unconfirmed

`include/turbo_ocr/onnx/ort_engine.h:133-135`:
```cpp
  // Owns the output tensor from the most recent infer_batch_view call so its
  // buffer outlives the returned view (until the next inference call).
  Ort::Value last_output_{nullptr};
```
`src/onnx/ort_engine.cpp:649-652` overwrites it each call and returns a pointer into it.
Callers: `ort_paddle_rec.cpp:189,276`.

The contract is documented honestly at `engine.h:56-61` ("Valid only until the next infer() on
the same engine"), and `OrtPaddleRec` also carries a mutable `batch_buf_`, so the class is
non-thread-safe **by design**. This is not a hidden bug.

**The risk:** if two threads share one `OrtEngine`, B's assignment to `last_output_` destroys
the tensor A is reading through `view.data` — a use-after-free producing garbage text rather
than a crash.

**What to check:** whether the replica pool can hand one engine to two threads. If per-replica
ownership is guaranteed, add a comment saying so.

**Related, verified:** `CpuBackend::doc_ori_` is a single instance on the Backend while
det/rec/cls/layout are per-replica. `load_stages()` runs once per replica and overwrites it, so
N−1 orientation models are built then destroyed and every replica's `orient_` closure reads the
same instance. Safe today (the path touches only stack state plus `Ort::Session::Run`), but it
breaks the isolation invariant every other stage maintains.

---

## M7 — `~PdfDocument` destroys a mutex it never acquires
**STATUS: FIXED (reconciliation commit) — destructor drains impl_->mtx before teardown.**
**VERIFIED**

Accessor — `src/pdf/text/pdf_text_layer.cpp:186-187`:
```cpp
  std::lock_guard<std::mutex> lock(impl_->mtx);
  std::lock_guard<std::mutex> gl(pdfium_lock());
```
Destructor — `:143-156`:
```cpp
PdfDocument::~PdfDocument() noexcept {
  std::lock_guard<std::mutex> gl(pdfium_lock());
  // Tear down page handles and document under the global lock ...
  if (impl_) impl_->pages.clear();
  impl_.reset();
```
The destructor takes the **global** PDFium lock but never `impl_->mtx`, then destroys `Impl` —
and with it `mtx`.

**The global lock makes the hazard more likely, not less.** A thread inside
`text_in_rect_pt` acquires `impl_->mtx` *first*, then blocks waiting for `pdfium_lock()`. If
the destructor holds `pdfium_lock()`, that thread is parked **holding `impl_->mtx`** while the
destructor destroys it — a mutex destroyed while locked, which is undefined behaviour.

The per-document `mtx` exists precisely so multiple page workers can share one `PdfDocument`,
so concurrency is the intended use.

**Currently safe by one call site:** `pdf_job.cpp`'s `run_pdf_job` calls `drain()`, joining
every page worker, before the local `unique_ptr` goes out of scope. Nothing enforces that for
other callers.

**Fix:** acquire `impl_->mtx` in the destructor before releasing the impl, or add a `closing_`
flag checked under the same mutex.

---

## M8 — `write_searchable_pdf` holds the global PDFium lock across the whole document
**STATUS: OPEN — narrowing the searchable-PDF pdfium lock needs the geometry passes hoisted out; not attempted.**
**VERIFIED**

`src/pdf/text/pdf_searchable.cpp:696` takes `detail::pdfium_lock()`; `FPDF_SaveAsCopy` is at
`:783`. The lock is held for the entire ~87-line span: the per-page stamping loop plus the
save.

`pdfium_lock()` serialises **every** PDFium call in the process — rendering on darwin, native
text extraction via `PdfDocument`, font matching. With `editable=1` the critical section also
covers repeated `FPDFPageObj_GetBounds` measurement work that needs no serialisation.

**Failure scenario:** a `?output=pdf&editable=1` request against a 300-page scanned book holds
the global lock for the full stamping pass. A concurrent one-page `/ocr/pdf` request blocks for
that entire duration. Throughput collapses to one PDF operation at a time, process-wide.

**Fix:** compute geometry outside the lock; take it per page for the PDFium calls and once for
`FPDF_SaveAsCopy`.

---

## M9 — `ctc_greedy_decode` admits a negative index
**STATUS: FIXED @ 46bbc4cd — index > 0.**
**VERIFIED** (defect) · reachability unconfirmed

`src/analysis/recognition/ctc_decode.cpp:76-82`:
```cpp
    int index = indices[i];
    if (index != last_index) {
      if (index != 0 && index < static_cast<int>(label_list.size())) {
        text += label_list[index];
```
The guard checks `!= 0` and an upper bound, no lower bound. For `index == -1`: `-1 != 0` is
true, and `-1 < static_cast<int>(size)` is true because the cast makes it a signed comparison.
`label_list[-1]` executes — out-of-bounds read on a `std::vector`, undefined behaviour.

`ctc_greedy_decode_raw` computes its index through its own argmax and cannot go negative. This
overload takes a caller-supplied `const int*` and validates one side.

**Reachability:** called from 8+ backend sites (`mps_stages.mm`, `intel_stages.cpp`,
`rocm_stages.cpp`, `paddle_rec.cpp`, several probes). Every in-tree caller appears to pass
argmax-derived indices. A device argmax over an all-NaN row is the plausible source of a
sentinel.

**Fix:** `if (index > 0 && index < static_cast<int>(label_list.size()))` — one token, no
behaviour change for valid input.

---

## M10 — the form-field model never checks that its two output heads agree
**STATUS: FIXED @ 46bbc4cd — head agreement refused at load.**
**VERIFIED**

`src/analysis/forms/field_model.cpp:180-190`:
```cpp
      if (shape.size() != 3) continue;
      if (shape[2] == 4 && dets_idx < 0) {
        dets_idx = static_cast<int>(i);
        queries = shape[1];              // Q from the BOXES head only
      } else if (shape[2] > 4 && logits_idx < 0) {
        logits_idx = static_cast<int>(i);
        num_classes = shape[2];          // C from logits; its shape[1] discarded
      }
```
`logits[q*C + c]` is later indexed for `q` in `[0, Q)`. If a re-exported graph has the heads
disagree on query count, the tail queries read past the end of ORT's logits buffer.

**Failure scenario:** a user's fine-tuned RF-DETR export with 300 boxes queries and 100 logits
queries loads successfully and over-reads on every inference.

**Fix:** compare at load time and fail loudly:
```cpp
    if (lshape[1] != queries) { std::cerr << "[ffdetr] heads disagree ...\n"; return false; }
```

---

## M11 — `preprocess_region` is implemented five times and called zero times
**STATUS: PARTLY — honesty note @ 532b23d3; routing NVIDIA table stages through the seam (or deleting) still open.**
**VERIFIED**

```
$ grep -rn '\(->\|\.\)preprocess_region(' src include tools tests
(no output)
```
No call sites anywhere. NVIDIA's table stages call `turbo_ocr::kernels::cuda_fused_*` directly,
bypassing the seam op.

Capability declarations: `host_kernels.cpp:67` true, `cuda_kernels.cpp:67` true,
`hip_kernels.cpp:108` true, `sycl_kernels.cpp:80` true (conditionally false at `:86`),
`metal_kernels.mm:48` **false**. So four backends advertise an operation nothing dispatches;
Apple correctly declares it unimplemented.

Roughly 600 lines of never-executed device code across three toolchains, and the four
`PreprocKind` geometries at `kernels.h:200-206` are enforced against nobody.

**Fix:** route the NVIDIA table stages through the seam op, or delete the op, its
implementations and the capability bit.

---

## M12 — constants defined two and three times, enforced by comment
**STATUS: FIXED @ 87af2f56 — static_asserts + shared side-limit constants in the reference detector.**
**VERIFIED**

**(a) The component budget, three definitions:** `kernels_cuda.h:130` (`kMaxGpuComponents`),
`kernels_hip.h:44` (`kMaxGpuComponents`), `db_post_config.h:35` (`kMaxDbComponents`).
`src/analysis/detection/det_postprocess.cpp:165` asserts they must be equal **in a comment**.
Raise one and the seam's `db_max_components` refusal compares against a value the kernels no
longer use: scratch sized from one constant, indices clamped by another.

**(b) The side limits, two definitions:** `ort_paddle_det.h:57-58` (class-local `kMinBoxSide`
3.0f / `kMinUnclippedSide` 5.0f) versus `db_post_config.h:29-30`. The ORT detector
(`ort_paddle_det.cpp:152`) uses its local copies while `DbPostParams` (`kernels.h:96-97`)
defaults from the shared ones. The ORT detector is the CPU reference every backend is
golden-diffed against — the worst place for a silent copy, because a change to the shared
limits leaves the reference stale and every backend then "diverges" from a wrong baseline.

**Also:** `paddle_det_ccl.cpp:196,199` retypes `kMaxExpand = 24.0f` and `min = 2.0f` as
literals beside `detection::kMaxExpand`/`kMinExpand` (`db_post_config.h:31-32`); and
`paddle_det.cpp:91` defines `GPU_UNCLIP_SCALE`, an NVIDIA-private multiplier on the shared
`unclip_ratio` that no other backend reads and that `CudaKernels::db_postprocess` does not
apply — so with the env set, NVIDIA's detector path and its own seam path disagree.

**Fix:** `static_assert` in both kernel headers; delete the class-local pair and the literals.

---

## M13 — `drop_score` below 0.5 is silently ignored
**STATUS: FIXED (reconciliation commit) — read() refuses drop_score below the engine floor.**
**VERIFIED**

`include/turbo_ocr/core/types.h:26` — `inline constexpr float kDropScore = 0.5f;`
`include/turbo_ocr/pipeline/ocr_pipeline_detail.h:117`:
```cpp
    if (rec_results[i].second < turbo_ocr::kDropScore) continue;
```
A hard filter inside the pipeline, before Python sees anything.
`python/turboocr/pipeline.py:34` — `DROP_SCORE = 0.5  # kDropScore in the C++ engine (applied
there too; a safety net here)`.

So the Python `drop_score` parameter can only **raise** the floor. Nothing plumbs a lower value
down.

**Failure scenario:** a user with a faint scan calls `ocr.read(img, drop_score=0.1)` — or
`turboocr ocr --drop-score 0.1` (`cli.py:39`) — expecting to recover low-confidence text. The
result is byte-identical to 0.5. No error, no warning.

**Fix:** plumb `min_confidence` through `RunFlags`/`InferOptions`, or `raise ValueError` when
`drop_score < DROP_SCORE`.

---

## M14 — explicit `autorotate=True` is silently dropped
**STATUS: FIXED (reconciliation commit) — explicit autorotate=True now works or raises; the dead AND is gone.**
**VERIFIED**

`python/turboocr/pipeline.py:327-328`:
```python
        do_auto = self.autorotate if autorotate is None else autorotate
        if do_auto and self.autorotate and angle == 0:
```
Line 327 correctly honours an explicit per-call `autorotate`. Line 328 then ANDs it with
`self.autorotate` again — **making line 327 dead logic.** When the instance was constructed
with `autorotate=False`, an explicit `read(autorotate=True)` is discarded.

(`:276` — `self.autorotate = autorotate and self._pipe.has_doc_ori()`.)

**The same file gets this right three lines later:** `:349-360` routes `layout`, `tables` and
`formulas` through a shared capability gate that **raises** when unavailable, with an eight-line
comment on why a refusal beats a silent no-op. `autorotate` bypasses it.

**Failure scenario:** `ocr = OCR("tiny")` then `ocr.read(sideways.png, autorotate=True)` — the
page is not rotated, OCR runs on sideways text, garbage returned with no warning.

**Fix:** drop the `and self.autorotate` conjunct and let the capability gate raise.

---

# Lower severity, verified

## L1 — float images in `[0,1]` are destroyed
**STATUS: FIXED (reconciliation commit) — [0,1] floats scaled by 255 before the cast.**
`python/turboocr/imaging.py:61-62`:
```python
        if src.dtype != np.uint8:
            src = np.clip(src, 0, 255).astype(np.uint8)
```
A float image in the conventional `[0,1]` range — `skimage.img_as_float`, `matplotlib.imread`,
a detached torch tensor, `imread(...)/255` — becomes all 0s with a few 1s. Detection finds
nothing, `read()` returns zero lines, no exception.
**Fix:** scale by 255 when `issubdtype(dtype, floating) and max() <= 1.0`.

## L2 — the Python model catalog is missing a row
**STATUS: FIXED (reconciliation commit) — tiny-bigdet row added with V6_DET (0.45) per the C++ rationale.**
`catalog.py:77-86` has 8 `ModelEntry` rows; `model_catalog.h:95-114` has 9. `tiny-bigdet`
(`model_catalog.h:108`) is absent from Python. `OCR(model="tiny-bigdet")` raises
`ValueError: unknown model` while `OCR_MODEL=tiny-bigdet` works on the server and the row is
documented at `docs/models/selection.md:41-80`.
**Fix:** generate `_CATALOG` from the C++ catalog, or expose it via `build_info()`.

## L3 — `backend="turbo"` cannot reach the NVIDIA seam backend
**STATUS: FIXED (reconciliation commit) — turbo/tensorrt/trt alias the nvidia seam backend; honest fallback preserved (incl. on Apple silicon); CLI help lists all five backends.**
`native.py:184` — `BackendAlias("nvidia", (), engine="nvidia", summary="NVIDIA (TensorRT)")`
has **no aliases**, so `"turbo"`/`"tensorrt"`/`"trt"` are absent from `_BY_ALIAS` and
`resolve_engine("turbo")` returns `"cpu"`. `configure_backend` handles them at `:309-313` and
honestly reports `"CPU (turbo/TensorRT needs the turboocr-engine-cuda wheel)"`.

**The defect is narrower than it looks:** on a CPU wheel the behaviour and message are correct.
On a build where the `nvidia` seam backend **is** compiled in, `backend="turbo"` still cannot
reach it, and the message "needs the turboocr-engine-cuda wheel" is then false — the user has it.

Compounding: `cli.py:35`'s help lists `auto|fast|turbo|cpu|cuda|openvino|coreml|...` — it
advertises `turbo` (which never reaches the seam) and omits `nvidia`, `intel`, `amd`, `apple`
(which do). Note also that `backend="cuda"` maps to the ORT CUDA EP inside the *cpu* seam
backend (`native.py:159`), not to the nvidia backend — two similar names, entirely different
paths.
**Fix:** add `("turbo","tensorrt","trt")` as aliases on the nvidia row; list all five backends
in the CLI help.

## L4 — the wheel is tagged `py3-none-any`
**STATUS: FIXED @ 99600d21 — hatchling hook stamps the real platform tag (verified cp31x-…-macosx_arm64).**
`python/pyproject.toml:1-2` uses hatchling with `packages = ["turboocr"]` and no build hook, so
the wheel tag is `py3-none-any` — while `python/turboocr/` contains
`_turboocr.cpython-311-darwin.so`. pip treats the tag as a universal match and installs it on
Linux/Windows and CPython 3.12/3.13, where `OCR()` raises `NativeExtensionMissing` for a wheel
pip considered perfect. `python/DESIGN.md:170-181` already gives the `scikit-build-core` recipe
that produces the correct `cp311-cp311-macosx_*` tag.

## L5 — ONNX Runtime version skew; CI tests the oldest
**STATUS: FIXED (reconciliation commit) — setup-cpp-deps and the cuda gate both pinned to the shipped 1.28.0.**
| consumer | version |
|---|---|
| `.github/actions/setup-cpp-deps/action.yml:49,56` — used by the unit suite, smoke, sanitizers, tidy, static-analysis | **1.22.0** |
| `CMakeLists.txt:668` (GPU) | 1.27.0 |
| `docker/Dockerfile:104,151` + `CMakeLists.txt:441` | 1.28.0 |
The action's description claims it "Mirrors docker/Dockerfile.cpu so CI and the image agree."
ORT version has silently broken this project before (CoreML NaN on 1.24.4, clean on 1.27).

## L6 — clang-tidy's scope is narrower than documented, and its glob drops two files
**STATUS: FIXED @ dffe38ad/4319d4b9 — comment, glob, and scope (100 files).**
`ci.yml:288` says "Scoped to src/pipeline + src/service + **src/models**" — there is no
`src/models`; it lints `src/analysis`. Unlinted by anything: `src/backends` (179 files),
`src/pdf`, `src/document`, `src/onnx`, `src/image`, `src/backend`.
Separately, `git ls-files 'src/pipeline/**/*.cpp'` — git's `**` requires an intervening
directory, so `finalize_deferred.cpp` and `ocr_pipeline_detail.cpp` are silently excluded.

## L7 — the `static-analysis` job cannot fail, twice over
**STATUS: BY DOCUMENTED CHOICE — the job says 'Report-only.'; revisit when a baseline exists.**
`ci.yml:67-89` carries **both** `continue-on-error: true` and `2>cppcheck.txt || true`, then
greps and echoes. Its `unusedFunction` check is the dead-code detector this job exists for
(it runs single-threaded specifically to keep that check working) — and D1–D5 below are exactly
what it would surface.

## L8 — `BackendSpec::fallback` is parsed and never read
**STATUS: FIXED (reconciliation commit) — a config that sets fallback is refused (ROUTING_FALLBACK_UNIMPLEMENTED / dangling-ref), never silently ignored.**
`include/turbo_ocr/backend/routing_config.h:55` documents it as "operator-declared backend
name". Parsed at `routing_config.cpp:174-175`; `grep '\.fallback\|->fallback'` finds no
consumers. It is also never validated — the modality-route loop at `:207-223` checks
`t.routes` targets against `t.backends`, but nothing checks that a backend's own `.fallback`
names a real entry. An operator configuring failover gets none, silently.

---

# False comments and documentation

**STATUS: F1–F18 all FIXED @ 4319d4b9** (the docs-correction commit) — except
where a later commit superseded the file. Kept below as the record of what was
false.

Each proved against the code it describes.

<a name="f1"></a>
### F1 — a comment asserting the opposite of the behaviour
`src/backends/amd/backend/rocm_backend.cpp:114-116` — "the shared factory yields the host/OTSL
path so **tables still work** (just not device-resident)." The factory returns `nullptr` and the
pipeline disables the modality. See [C7](#c7).

<a name="f2"></a>
### F2 — a safety policy described but never implemented
`src/backends/amd/support/hip_check.h:3-8` — "only a **STICKY** fault terminates the process."
Every error throws; no sticky detection exists for HIP. See [C8](#c8).

<a name="f3"></a>
### F3 — claims a missing barrier that is present
`src/backends/amd/kernels_hip/reduce_kernels.hip:19-21` — the CUDA original "reads
`s_vals[tid+32]` with **NO `__syncthreads()`**".
`src/backends/nvidia/kernels_cuda/reduce_kernels.cu:47-55` closes **every** loop iteration with
`__syncthreads()` inside the loop body:
```cpp
  for (int stride = block_size / 2; stride > 32; stride >>= 1) {
    if (tid < stride) { ... }
    __syncthreads();
  }
```
The last iteration (stride=64) therefore synchronises before the `if (tid < 32)` tail reads
`s_vals[32..63]`. **The CUDA kernel is correct.** The HIP file's *conclusion* — a
wavefront-agnostic tree on AMD — is right and should stay; only the justification is false.
**Most likely of any entry here to cause a wrong change:** it invites "fixing" a race that does
not exist.

<a name="f4"></a>
### F4 — a README command that cannot run
`README.md:196-201`:
```bash
./build/turboocr-server --backend apple \
  --layout-model models/layout.onnx \
  --doc-orient-model models/doc_ori.onnx \
  -e TABLE_BACKEND=slanext -e FORMULA_BACKEND=ppformulanet_s
```
None of the three exist. `grep -oE '"--[a-z0-9-]+"' server_config.cpp` yields
`--disable-layout`, `--layout-onnx`, `--layout-trt` and no orientation flag. Doc-orientation is
env-only (`DOC_ORI_ONNX`, `:203`); `TABLE_BACKEND`/`FORMULA_BACKEND` are env vars (`:279-280`)
and `-e` is Docker syntax. CLI11 rejects unknown options, so the documented "All stages on
Apple" command **fails at startup**.
**Fix:**
```bash
TABLE_BACKEND=slanext FORMULA_BACKEND=ppformulanet_s DOC_ORI_ONNX=models/doc_ori.onnx \
  ./build/turboocr-server --backend apple --layout-onnx models/layout.onnx
```

<a name="f5"></a>
### F5 — a status banner contradicted by its own neighbour
`src/backends/intel/README.md:3-9,157` — "not one SYCL kernel here has ever been compiled …
**No accuracy and no throughput number exists.** Any figure quoted would be fabricated."
`src/backends/intel/SETUP.md`, same directory, has per-stage golden diffs (`:262-266`), end-to-end
`F1 = 85.52%` (`:268`), and dated throughput (`:344-358`). `src/backends/README.md:15` says
intel is "ported and verified on real Intel hardware".

<a name="f6"></a>
### F6 — a documented env var that does not exist
`docs/reference/configuration.md:153` documents `PIPELINE_HARD_KILL_MS` as `_Exit`ing a wedged
worker. It appears nowhere in `src/`, `include/`, `python/` or `tools/` — the only one of 152
documented env vars with no code reference. The real variable is `TURBO_POOL_STUCK_LEASE_MS`
(`unified_pipeline_pool.cpp:59`), which only *reports*.
`make_infer_func.h:86-97` explicitly rejects the documented behaviour: *"Reporting the condition
loudly is worth having now; pretending to fix it is not."* Line 314 of the same doc documents the
real variable correctly, so the file contradicts itself. Repeated at
`docs/guides/upgrading-v3.md:18`. **An operator relying on this for hang recovery gets nothing.**

<a name="f7"></a>
### F7 — "CMake target never configured", for a configured target
`src/backends/README.md:13`. `turbo_ocr_backend_nvidia` is fully configured: sources
`CMakeLists.txt:1525-1541`, CUDA properties `:1543-1546`, includes `:1548`, link
`:1550-1555`, alias `:1557`.

<a name="f8"></a>
### F8 — three stale capability claims on Apple
`src/backends/apple/README.md:191-195` says layout is unimplemented and "`load()` returns false
⇒ backend reports layout unavailable". `apple_backend.mm:296-320` does the opposite: when no
MPSGraph export is present it constructs `HostLayoutOnDevice`, loads the ONNX model, and sets
`ss.available.optional.set(capability::CapabilityId::Layout, true)` at `:312`.
`apple_backend.h:12` — "Classifier is structural; layout/table/formula are TODOs" — is wrong on
all three: the same README says "**MpsClassifier** — VALIDATED (was structural)" at `:151`;
layout works as above; table and formula are wired at `apple_backend.mm:125-126` and `:136-137`
with capability bits at `:325-326`. Only the MPSGraph-native versions are TODO.

<a name="f9"></a>
### F9 — two comments each calling a different function "the SHARED one"
`src/backends/apple/stages/mps_stages.h:81` and `src/backends/intel/stages/intel_stages.cpp:158`
both name `detection::snap_det_canvas` as "the SHARED" one. They are different overloads with
opposite input contracts. See [M2](#m2).

<a name="f10"></a>
### F10 — "mirror the CUDA heuristic", for a different heuristic
`src/backends/amd/backend/rocm_backend.cpp:51`. Different tiers, no safety floor. See
[M4](#m4).

<a name="f11"></a>
### F11 — reasoning that covers only the harmless case
`src/backends/nvidia/backend/nv_backend_registry.cpp:1-3` — falling through "only changes which
name is reported, since both land on the same ONNX stages." True when no other accelerator
exists; false when a usable AMD or Intel GPU is present. See [M3](#m3).

<a name="f12"></a>
### F12 — claims parity with a stronger guard
`src/pipeline/unified/unified_pipeline_batch.cpp:120-122` — "The shared batcher already guards
the same call the same way; the pipeline's own direct call did not." The batcher does a per-slot
re-run (`stage_batcher.cpp:238-263`); this site pads with empty vectors. Same problem,
materially weaker handling — and that difference is what turns a detected failure into a silent
one. See [H4](#h4).

<a name="f13"></a>
### F13 — "same as every other route", for the one property that differs
`src/service/http/pdf/stream_route.cpp:391` — "Same three-way split as every other image route."
True of the three error codes; false of the ordering, which is the entire security property. See
[H3](#h3).

<a name="f14"></a>
### F14 — five stale `rebuild/` paths, one telling maintainers not to edit
`rebuild/` no longer exists.
- `include/turbo_ocr/backend/backend_registry.h:24` — "Every rebuild/*/build.sh passes these TUs
  as explicit .o files." No `build.sh` exists; registrars come in via CMake `WHOLE_ARCHIVE`
  (`CMakeLists.txt:1284,1327,1366,1429,1536`).
- `include/turbo_ocr/core/db_post_config.h:9,13-14` — "rebuild/apple/metal_kernels.mm passed
  min_unclipped_side = 2.0" and **"This header lives on the rebuild side (the main tree is
  read-only for the rebuild work)"**. The file is in the main tree; the 2.0 → 5.0 fix is already
  applied at `metal_kernels.mm:294-299`. **The "read-only" line tells a maintainer not to edit
  the file they are standing in.**
- `src/backends/intel/stages/intel_stages.h:17`, `src/backends/apple/engine/ane_rec_engine.h:15`
  — `rebuild/include/.../rec_batching.h`.
- `src/backends/apple/backend/apple_backend.h:65` — "the ONE orchestration lives in
  rebuild/pipeline"; actually `src/pipeline/unified/`.

<a name="f15"></a>
### F15 — documented classes that do not exist, and a 404 link
`docs/models/table.md:37,109-110` — `CpuSlanextTable` / `CpuSlanextEncoder` with a link to
`include/turbo_ocr/analysis/table/slanext/cpu_slanext_table.h`. Neither class name appears in
`src/` or `include/`; the header does not exist. Real path is `cpu::CpuTableRecognizer`
(`src/backends/cpu/stages/cpu_table_recognizer.h:23`) wrapping `OrtSlanextTable`.

<a name="f16"></a>
### F16 — the documented binding name is wrong
`python/DESIGN.md:83,212` — `nb::class_<CpuOcrPipeline>(m, "CpuPipeline")`. The bound name is
`Pipeline` over `UnifiedOcrPipeline`: `src/service/python/bindings.cpp:383` is
`nb::class_<PyPipeline>(m, "Pipeline")`. `python/turboocr/native.py:79` records the rename.

<a name="f17"></a>
### F17 — a default documented as absent
`docs/reference/configuration.md:347` — `OV_PERF_HINT` default column is `—`.
`src/backends/intel/engine/openvino_engine.cpp:234` is `env::env_or("OV_PERF_HINT", "latency")`,
and `intel/SETUP.md:252` documents both the default and why it flipped (throughput starved the
synchronous engine: 2.4 vs 5.5 img/s).

<a name="f18"></a>
### F18 — an "always" with an exception one line below
`include/turbo_ocr/core/db_post_config.h:43` — "Returned canvas is always one of `available`."
Line 55 is `if (available.empty()) return {want_h, want_w};` — a canvas in no list.

---

# Dead code

<a name="d1"></a>
**STATUS: D1, D2, D4, D5 deleted @ 532b23d3 (verified zero callers each). D3's
"config classes nothing reads" is OBSOLETE the other way — det_cfg is now READ
(installed into the native base @ 3c623e71).**

### D1 — `python/custom_models/` — 1149 lines, zero callers, never shipped
`__init__.py` (59), `modeling_paddleocr_vl_trt.py` (762),
`input_processor_paddleocr_vl.py` (328). `grep -rn 'custom_models'` outside the directory itself
returns nothing. Not in `[tool.hatch.build.targets.wheel] packages = ["turboocr"]`, so it never
ships. If imported, `__init__.py:37-38,57-58` swallow every registration failure into
`warnings.warn`, so a failed `AutoConfig.register` would let the model run against the wrong
config.

<a name="d2"></a>
### D2 — `python/turboocr/result.py:683-690` — dead, and silently truncating
```python
def lines_from_pairs(pairs, boxes):
    for (text, score), box in zip(pairs, boxes):
```
Zero callers. Also `zip`s without a length guard. Compare `pipeline.py:484-488`, which refuses a
mismatch **and explains why**: *"a short list would silently DROP pages, turning a backend bug
into missing text."* Same hazard, opposite handling, same file's sibling.

<a name="d3"></a>
### D3 — `python/turboocr/catalog.py:16-50,66` — config classes nothing reads
`DetResizeParams`, `DbParams`, `DetConfig`, `V6_DET`, `V6_DET_TINY`, `ModelEntry.det_cfg`.
`catalog.py:4-7` claims Python is "in lockstep with the C++ engine … same official PaddleOCR
detection inference config" — the values are computed and discarded. Direct cause of
[H9](#h9).

<a name="d4"></a>
### D4 — `python/turboocr/pdf.py:100` `render_pdf_pages` — zero callers.

<a name="d5"></a>
### D5 — `IKernels::preprocess_region` — ~600 lines across five backends, zero call sites.
See [M11](#m11).

---

<a name="refuted"></a>
# Refuted — CORRECTED

**Only R1 stands as written, and R7 in part.** R2–R6 below were "disproved"
by reading the FIX CAMPAIGN'S OWN PATCHES (this pass ran against a moving
tree): the code each entry cites as proof the finding never existed — the
searchable_pdf source-check at :144, the providers.py deletion note, the
darwin area cap with its threat comment, AMD's real-crop drop count with its
rule comment, AMD's params-driven sliver gate — was WRITTEN BY THE FIXES
(commits c509b69d, 87af2f56, 532b23d3, 99600d21). The first-pass findings
were real at 580b38c5, were independently re-verified there, and are fixed.
The right conclusion for R2–R6 is "FIXED", not "never existed".

### R1 — "Metal's resize omits the half-pixel shift" — WRONG
Reported CRITICAL, ranked #1 of 20, against the shipped macOS backend.

Claim: `shaders.metal:105-106` samples at `(gid+0.5)/dst` without CUDA's `-0.5`, so every Apple
detection box is shifted half a source pixel.

**Why it is wrong:** the comparison is not like-for-like. CUDA indexes a raw array and must
convert to texel-center space itself. Metal's sampler with `coord::normalized` and
`filter::linear` already works in texel-center space, where texel *i* is centered at *i+0.5*.
Sampling at `u = (gid+0.5)/dst_w` gives unnormalized `x = (gid+0.5)·scale`; the bilinear filter
interpolates around `x − 0.5`, i.e. an effective source position of

    (gid + 0.5) · scale − 0.5

algebraically identical to CUDA's explicit `sx`. **The −0.5 is applied by the sampler, not
omitted.** "Fixing" this would introduce the exact error it claims to remove.

### R2 — searchable-PDF `source != "pdf"` predicate — REAL, FIXED @ 99600d21
**CORRECTION: the predicate WAS missing at 580b38c5; the line and comment cited here are the fix (99600d21).**
Reported CRITICAL. `python/turboocr/searchable_pdf.py:144`:
```python
                if getattr(ln, "source", "") == "pdf":
                    continue
```
preceded by a comment explicitly citing `pdf_searchable_encoding.cpp keep()` and describing the
doubling bug it prevents. `TextLine.source` exists (`result.py:81`). The agent read the
`if not text.strip()` line above it and stopped.

*(The confidence clause of the C++ predicate is also effectively present — `_fill_lines` applies
`drop_score` upstream before these lines are reached.)*

### R3 — `providers.py` dead provider-resolution block — REAL, FIXED @ 532b23d3
**CORRECTION: the ~185 lines existed at 580b38c5; the 251-line file with the deletion note IS the fix (532b23d3).**
`wc -l python/turboocr/providers.py` → **251**. Lines 248-431 do not exist. Lines 248-251 are:
```python
# NOTE: resolve_providers()/_trt_options()/_raise_missing() were deleted
# 2026-08-03. They were a fourth, already-drifted copy of the EP alias table:
```
The functions were already removed.

### R4 — macOS render page-area cap — REAL, FIXED @ c509b69d
**CORRECTION: the cap was absent at 580b38c5; pdf_renderer_darwin.cpp:83-90 sharing the Linux constant IS the fix (c509b69d).**
Reported as a DoS. `src/pdf/render/pdf_renderer_darwin.cpp:83-90`:
```cpp
  // AREA CAP — same constant as the Linux PPM path (pdf_renderer_internal.h).
  // /MediaBox is attacker-declared, and unlike Linux (which renders in a
  // disposable fastpdf2png subprocess) this rasterizes IN the server process ...
  if (static_cast<int64_t>(w) * h > pdfrdetail::ppm_max_pixels()) {
    TOCR_LOG_WARN("page exceeds MAX_PDF_PAGE_PIXELS_MP; refusing to render", "w", w, "h", h);
    FPDF_ClosePage(page);
    return {};
  }
```
The cap exists, shares the Linux constant, and its comment describes exactly the threat the
finding claimed was unhandled.

### R5 — `dropped_crops_` padded count on AMD — REAL, FIXED @ 87af2f56 (NVIDIA half was genuinely wrong)
**CORRECTION: AMD added the padded batch at 580b38c5; the real-crop count and the rule comment ARE the fix (87af2f56). (The NVIDIA half of the first-pass claim was genuinely wrong — cur_batch was always real crops.)**
`rocm_stages.cpp:387,395`:
```cpp
      // Count the REAL crops (n), not the padded static batch B: padding slots
      // never held text, and intel/apple already count n — a per-backend split
      // here silently mis-calibrates any alert threshold on this metric by 2x.
      dropped_crops_ += n;
```
AMD already counts real crops, under a comment stating the exact rule it was accused of
violating. NVIDIA's `paddle_rec.cpp:285` adds `cur_batch`, and `:246` is
`int cur_batch = end - beg;` — also the real count. **No divergence exists.**

### R6 — four divergent DB sliver filters — REAL, FIXED @ 87af2f56
**CORRECTION: 'already fixed' is right — BY THE FIX CAMPAIGN (87af2f56), not stale in pass 1; the four divergent gate sets were real at 580b38c5.**
`hip_kernels.cpp:264-270`:
```cpp
  // Side gate = params.min_unclipped_side — the CALLER'S values, which is what
  // flips caps().params.db_side_limits to true. ... This loop used to hardcode
  // "< 3" under a comment claiming "same min-side gates as the CUDA det path" —
  // at the time there were FOUR different gate sets across the arms and this was
  // the loosest, emitting 3px slivers the shared reference (5.0 post-unclip) drops
```
Already fixed: AMD now uses the caller's `params`. The offending comment was also replaced by
this accurate history. *(The related first-pass entry "W10 — same min-side gates" is likewise
stale and removed.)*

### R7 — "all five backends declare `caps().preprocess_region = true`" — WRONG in part
`metal_kernels.mm:48` is `c.preprocess_region = false; // TODO(apple-fused-region-preproc)`, and
`sycl_kernels.cpp:86` sets it false conditionally. Four backends advertise it, not five; Apple
correctly declares it unimplemented. **The zero-call-sites half of the finding is confirmed** and
is retained as [M11](#m11).

---

# What changed between passes

| pass 1 | pass 2 |
|---|---|
| 9 parallel agents, ~⅓ of findings unverified | single-threaded, 100% verified |
| 10 critical | 8 critical (2 refuted) |
| 12 high | 9 high |
| 23 false comments | 18 (5 stale/refuted) |
| 7 dead-code items | 5 |

**Seven findings refuted, two of them CRITICAL.** The refutation rate among agent-reported,
never-verified findings was roughly one in three. Where a finding survived verification it
usually got *sharper* — M14's real defect is that line 327 is dead logic, which the first pass
did not identify; L3 is materially narrower than reported.

The verified findings themselves did not move: the gate layer is the problem, and everything
else in this document exists because the automation that should have caught it cannot fail.

# Ranked fix order

**Gates first — nothing below them is measurable until they work.**

1. **C2** syntax-shim exit status. One line; it currently gates nothing.
2. **C5** stale ratchet path + the two real over-length functions → CI green.
3. **C1** register the accuracy and conformance gates in CI.
4. **C3** replace the order-blind metric; **C4** make `backend_probe` able to fail.
5. **C6 / L7** make both analysis gates able to fail; **H8** add a pytest job.
6. **H1 / H2** escape `emit_text`, widen the algorithm fence — attacker-controlled, six routes.
7. **H3** pre-decode guard on `/ocr/stream`.
8. **H4** stop padding a detector under-return; **H5 / H6** the buffer and geometry bugs.
9. **C7 / C8** AMD recognizers and sticky-fault policy — before any ROCm bring-up.
10. **H7** Intel `caps()` downgrade; **M4** AMD → shared pool sizing; **M1** close the seam loophole.
11. **F1–F18** correct the false comments (F1, F2, F3, F4 first).
12. **H9 / M13 / M14 / L1–L4** the Python surface.
13. Delete the dead code (D1–D5).

## Open — found during CUDA engine wheel bring-up (2026-08-04, RTX 5090)

- **ORT-CUDA fast path slower than CPU — ROOT-CAUSED AND FIXED (2026-08-05)**:
  `rec_batch_num_` defaulted to 1 and nothing ever set `REC_BATCH_N`, so the
  ORT rec path ran ONE Run PER CROP everywhere (profile_dump: 41 rec_infer
  calls, 373 of 412 ms/page). Fix: device-EP defaults (batch 32 + 9-bucket
  ladder) in the shared stage — measured 2.5 -> **7.8 img/s** (~2x CPU),
  output identical. REMAINING CEILING, measured and proven ORT-inherent:
  ORT-CUDA charges ~10 ms per shape SWITCH even between warm shapes (bare
  ORT repro: 3.3 ms/run one shape, 13.4 ms/run cycling the page's 7 bucket
  shapes; io_binding does not help), and a FUNSD page cycles ~7 bucket
  shapes -> ~84 ms/page floor in rec_infer. Coarser buckets tested and
  REJECTED: only 115->105 ms and 63% text agreement (wide blank-padding
  drifts the decode — the known OOD-padding effect). Next levers are
  ORT-internal (per-shape CUDA graphs) or the native-TRT wheel.
- **Heap corruption at interpreter teardown on the CUDA EP path**:
  `corrupted double-linked list` / `free(): chunks in smallbin corrupted`
  after `OCR.close()` + exit in the wheel venv (fp32, ORT 1.27 gpu_cuda13).
  CPU path exits clean. Ordering issue between ORT CUDA EP teardown and
  process exit; needs a minimal repro against bare ORT to decide theirs/ours.
- fp16 sidecars: `make_fp16_models.py` FAILS on `rec_tiny.onnx` ("downstream
  node of the second cast node should be graph output") and its `det_tiny`
  output fails to load in the pipeline — the fp16 conversion flow is broken
  for the v6 tiny models.


## 2026-08-05 — autorotate fixed; formula -S decode bug arbitrated

- **?autorotate=1 was silently ignored on every image transport — FIXED.**
  Old-vs-new endpoint testing showed byte-identical output with/without the
  flag; the doc-orientation model itself classified 0/90/180/270 perfectly.
  Fixed at the seam (make_infer_func::maybe_autorotate) + the three image
  routes' acts_on + spec_allows. Validated on the 5090: token-F1 vs upright
  baseline = **1.000 at 90/180/270**, X-Ignored-Params clean. Inherited from
  the pre-multibackend server (identical behavior there), fixed in v4 only.
- **PP-FormulaNet garbled LaTeX — ROOT-CAUSED as crop-margin sensitivity,
  FIXED at the shared dispatch (pad 4).** The earlier "ours-only
  transposition/doubling bug in the FAST host loop" verdict was WRONG — see
  the arbitration correction below. The real mechanism: the pipeline fed the
  recognizer the layout model's TIGHT boxes; PP-FormulaNet's AR decoder
  scrambles on zero-margin crops (proven via /infer on the same crop: tight =
  garbled, +4px = clean `\boxed{10^{-16}\mathrm{erg\cdot K}^{-1}}`, +8px =
  `\begin{array}` neighbor-glyph noise). Fix: `dispatch_formulas_`
  (src/pipeline/unified/unified_pipeline_dispatch.cpp) pads every formula box
  by `FORMULA_CROP_PAD` (default 4, clamped 0-64) at the ONE shared site (both
  GPU PPFormulaNetOrt and CPU OrtFormulaRecognizer inherit it; reported boxes
  stay the layout model's own). First landed in the CPU recognizer only — no
  effect on the GPU server (different class) — moved to the shared dispatch
  per the dedup rule. Validated on the 25-formula omnidocbench page
  ...60403612.pdf_179.jpg: doubled-token artifacts 3 -> 1, crops 0 & 3 now
  clean. Pad sweep on the same page: pad6/pad8 REGRESS (crop 3 `erg`->`crg`,
  crop 2 loses content / array noise) -> 4 is the optimum, keep default.
- **Arbitration correction (identical saved crops /tmp/formula_crop_{2,3}.png):**
  | source | crop 2 | crop 3 |
  |---|---|---|
  | ours (FAST decode, pad4) | `J{ \cdot o m l }^{-1}...` (residual) | `\boxed{10^{-16}\mathrm{erg\cdot K}^{-1}}` (clean) |
  | paddle PP-FormulaNet-S (true -S) | `{\bf J}{\cdot}{\bf m o l}^{-1}{\cdot}{\bf K}^{-1}` | `\overline{{{10}^{-16}{tt e e g}...}}` (garbled) |
  | paddle PP-FormulaNet_plus-S | `\mathrm\{J{\cdot m o m}^{-1}\ \ {}K}...` (garbled) | `\boxed{10^{-16}\mathrm{erg\cdot K}^{-1}}` (clean) |
  | paddle plus-M | `\mathrm{J}\cdot\mathrm{mol}^{-1}\cdot\mathrm{K}^{-1}` | `10^{-16}\mathrm{erg}\cdot\mathrm{K}^{-1}` |
  Our engine matches paddle plus-S per-crop (clean crop 3 incl. the \boxed
  style; garbled crop 2 — paddle plus-S garbles it too). The crop-2 residual
  is a plus-S MODEL weakness, not our decode bug; true -S happens to read it,
  plus-M reads everything. Escalation for such crops =
  FORMULA_BACKEND=ppformulanet_plus_m or `auto`.

- **Formula "upgrade to plus-S": ALREADY SHIPPED — the bundle named
  `ppformulanet_s` IS PP-FormulaNet_plus-S (2026-08-05).** Proof: (1) paddle's
  official plus-S download shares 418/418 weights with our shipped bundle,
  while true PP-FormulaNet-S differs in 417/418; (2) paddle's plus-S program
  adds only 3 DEAD tensors (conv2d_80, linear_0 — declared, never consumed)
  over -S, i.e. one shared architecture (2-layer 384-d MBart, 3-token MTP);
  (3) the new `scripts/models/onnx/export_ppformulanet_plus_s_fast.py`
  (paddle2onnx -> -S patch -> weight-swap -> encoder extraction) rebuilds ALL
  FIVE shipped bundle files BYTE-IDENTICALLY from the plus-S download — the
  previously unversioned bundle recipe is now reconstructed and committed.
  Consequence: no model swap exists to do; plus-S En-BLEU 88.71 is what we
  ship. Naming made truthful instead: `ppformulanet_plus_s` accepted as an
  alias (normalized to the historical token at the ONE routing seam:
  canonical_engine in routing_config.cpp + formula_bundle_env.h), docs
  corrected. plus-M stays the optional quality/Chinese engine; `auto` ladder
  unchanged.

- **Unified server couldn't serve plus-M or auto at all — FIXED (found by
  booting every documented FORMULA_BACKEND end-to-end).** The formula bridge
  hardcoded PPFormulaNetOrt("ppformulanet_s"): plus-M got the -S graph
  contract (fast/ subdir names vs plus-M's in-dir decoder_step.onnx -> fatal
  at load) and `auto` fell through to the shared factory -> boot fatal
  "unknown formula backend 'auto'". Fixed by threading the routed engine key
  through NvFormulaRecognizer -> the ONE old-side factory. Second bridge gap
  in the same boot: wants_context_hint()/set_context_hint() were dropped, so
  the auto ladder's page-level CJK escalation never fired on the unified
  server (only crops whose plus-S OUTPUT contained CJK escalated). Both
  validated on the box: auto boots, loads plus-S + plus-M + the 384-KV
  bucket, ladder active.
- **OmniDocBench re-measured with the current best configs (2026-08-06/07)**
  — the docs' ≈49 headline was a year stale and embedded the long-fixed
  formula bug. Fresh full-set (1557 matched pages): fully local
  (medium + slanext + FORMULA_BACKEND=auto) = composite 82.0
  (text 0.073 / CDM 0.767 / TEDS 0.766); the hybrid (VL table+formula crops,
  July v16 run) = 92.9. Speed, measured same-100-pages pool-1: plus-S 10.0,
  auto 9.09, VL-formulas-only 6.67, plus-M-everywhere 2.44, VL-both 2.50
  img/s -> plus-M as sole engine is DOMINATED (VL-both costs the same and
  scores +11); auto is the only sane plus-M deployment. Per-crop join of the
  two runs: local formula loss is a TAIL (68.7% of crops >=0.9 CDM, 8.8%
  total losses), tables a spread cell-text weakness (structure-only 0.855).
  docs/benchmarks/omnidocbench.md rewritten around all of this.
- **BACKLOG — selective VL escalation ("quality ladder").** Oracle math from
  the per-crop join: escalate formulas with local CDM<0.5 (20.5% of crops)
  + worst ~15% of tables to the VL -> composite ~89 at ~3x hybrid speed;
  VL is worse than local on only 3.8% of escalated crops. Needs a runtime
  trigger instead of true scores: collapse/repetition detection (catches the
  8.8% zero-CDM class), decoder confidence, table cell-count sanity.
  Architecturally a generalization of AutoCjkFormula: same per-crop
  escalate pattern with a quality trigger and a kind:openai rung.
- **Legacy PP-OCRv5 Latin DETECTOR no longer builds on TRT — RESOLVED via the
  ORT-CUDA engine mode (2026-08-06).** det_v5.onnx fails
  `buildSerializedNetwork` on TRT 10.15.1 / sm120 with `Error Code 10: Could
  not find any implementation for node {ForeignNode[...]}`, invariant across
  TRT_DET_WORKSPACE_GB 4/8, TRT_OPT_LEVEL 5/3, det profile 1280/960, an
  EMPTY GPU (32 GB free — VRAM pressure ruled out), AND both precisions (the
  new TRT_FP16=0 fp32 escape hatch fails on a different foreign node,
  Conv.0+BatchNorm...). The fused-graph compiler rejects the whole network.
  WORKING full-v5 path: `TURBO_ENGINE_MODE=onnx` runs det_v5 + rec_v5 on the
  ORT-CUDA EP — validated on FUNSD (36 words, coherent v5-grade text, mean
  conf 0.98). rec_v5 alone also still builds on native TRT behind the v6
  detector (mean conf 0.956) — the per-script-bundle shape.
  docs/guides/legacy-ppocrv5.md documents both recipes; TRT_FP16 stays as a
  general escape hatch (cache-key-aware). Two environmental traps surfaced by the same session, both now
  documented in native.md: (1) non-interactive shells don't source the
  bashrc TRT export, and a warm engine cache hides the missing/wrong
  LD_LIBRARY_PATH until the first FRESH build fails in the BUILDER-resource
  dlopen; (2) box-specific: an unrelated process bound to 127.0.0.1:8080
  shadows the server's 0.0.0.0:8080 for loopback clients (specific bind wins)
  — reach the server via a non-loopback address or move the server's PORT.
- **OPEN — VRAM grows monotonically under sustained VARIED-SIZE load at the
  medium tier until the card exhausts; box wedged hard (2026-08-06).** Three
  OmniDocBench full-1651 render attempts (medium + slanext + auto, pools 3
  and 2, concurrency 4 and 2) all followed the same trajectory: first
  ~100-200 pages clean, then free VRAM slides ~6 GB per ~100 pages
  (measured 21.2 -> 14.7 -> 8.7 -> 6.0 GB) until allocation fails, EVERY
  subsequent request 500s (cuda_allocator OOM), and on the final attempt the
  whole box became unreachable for 2+ hours (needed physical intervention).
  Never reproduces on FUNSD — its pages are uniform; OmniDocBench's wildly
  varied page sizes are what feed the growth, so the suspect is per-shape
  device allocations that accumulate (det canvases / rec activation buffers /
  ORT arenas) instead of being reused or bounded. Needs a dedicated session
  with per-stage VRAM sampling. Workaround for full-corpus runs: chunked
  render with --skip-existing and a fresh server per chunk.
- **OPEN — pool footprint constant under-measures the MEDIUM tier: auto-sized
  pool 3 at medium+auto OOM'd at runtime (2026-08-06).** On an empty 32 GB
  card, OCR_MODEL=medium + FORMULA_BACKEND=auto auto-sized to 3 replicas
  (budget/(4.5+4) GiB) and filled the card to 2 MiB free once real pages ran
  — every request 500'd with cuda_allocator OOM and the process needed
  SIGKILL. Measured real cost at pool 2: ~26 GB total => **~13 GB per
  medium+auto replica** (the 4.5 GiB base was measured on tiny/small only;
  medium engines + CUDA graphs + activations were never in it). Workaround:
  set PIPELINE_POOL_SIZE explicitly for medium (2 on a 32 GB card with
  auto). Proper fix: per-tier footprint numbers in pool_sizing.h (medium
  ~9 GiB base) — measure once with graphs on.
- **Pool VRAM footprint model ignores the formula engine's device scratch —
  FIXED.** With FORMULA_BACKEND=auto the sizer picked pool=5 on the 32 GB
  card and boot OOM'd inside the plus-M encoder load (plus-M continuous-batch
  scratch is ~3.3 GB/replica: 4x 1056-KV + 4x 384-KV buffers + 128-crop
  cross-KV, plus its ORT sessions). Fix:
  pool_sizing.h::formula_engine_scratch_bytes adds a 4 GiB per-replica
  surcharge for plus-M/auto (plus-S is inside the measured base), and both
  vendor caps() sites feed it from the ROUTED formula engine. Validated: auto
  with no PIPELINE_POOL_SIZE now auto-sizes to the pool that fits and boots.
