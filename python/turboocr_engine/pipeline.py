"""The public OCR engine — a thin wrapper over the native C++ pipeline.

``OCR`` resolves a model + backend once at construction, loads the native
``Pipeline`` (the real C++ det → sort → cls → rec over the backend seam, with
the ORT execution provider chosen by env), then ``read`` / ``read_batch`` /
``read_pdf`` run pages.
No pre/post-processing happens in Python — it all runs in the extension.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
import queue
import threading
from typing import AsyncIterator, Generator, List, Optional

import numpy as np

from . import native
from .catalog import resolve_model
from .imaging import ImageInput, load_image, rotate_bound
from .models import ModelStore
from .native import configure_backend, load_native, quiet_stdout, set_log_level_default
from .providers import detect_hardware
from .result import (
    DocumentResult,
    FormulaRegion,
    LayoutBox,
    PageResult,
    Quad,
    TableRegion,
    TextLine,
)

from .options import (
    DEFAULT_DPI,
    DROP_SCORE,
    ENGINE_MODES,
    EngineMode,
    OnError,
    PdfMode,
)
from .options import check_on_error as _check_on_error_impl
from .options import (
    check_drop_score,
    check_pages,
    check_pdf_mode as _check_pdf_mode,
)

# Placed into the replica pool by close(): a reader that raced past the
# _closed flag and parked in Queue.get() receives this instead of blocking
# forever, re-puts it for the next parked reader, and raises.
_POOL_CLOSED = object()

# lang codes that map to a dedicated PP-OCRv5 script recognizer (tier N/A).
_SCRIPT_LANGS = {
    "ar": "arabic", "arabic": "arabic",
    "ko": "korean", "korean": "korean",
    "th": "thai", "thai": "thai",
    "el": "greek", "greek": "greek",
    "ru": "eslav", "cyrillic": "eslav", "eslav": "eslav",
}
# lang codes covered by the PP-OCRv6 tiers (Latin + Chinese + Japanese).
_V6_LANGS = {"en", "english", "latin", "ch", "chinese", "ja", "japanese", "default", ""}


def _make_progress(progress, total, unit):
    """Normalize the ``progress`` arg into a ``report(done)`` callback."""
    if not progress:
        return lambda done: None
    if callable(progress):
        return lambda done: progress(done, total)

    import sys as _sys

    def report(done):
        end = "\n" if total and done >= total else "\r"
        tot = f"/{total}" if total else ""
        print(f"\r[turboocr] {done}{tot} {unit}", end=end, file=_sys.stderr, flush=True)

    return report


def _parallel_map(items, fn, workers: int, *, ordered: bool = True,
                  lookahead: int = 1, executor=None):
    """Map ``fn`` over the ``items`` iterator with ``workers`` threads and a
    bounded look-ahead window, yielding results in input order
    (``ordered=True``) or as each completes (``ordered=False``).

    This is the page fan-out primitive under :meth:`OCR.read_pdf` /
    :meth:`OCR.read_pdf_stream` / :meth:`OCR.pdf_to_searchable`: producing an
    item (rendering a PDF page, ~5 ms) is far cheaper than mapping it (OCR,
    50-250 ms), so the producer stays on the calling thread and at most
    ``workers + lookahead`` items are in flight — which is what bounds
    retained page rasters. Ordered mode keeps progress monotone and assembles
    documents identically to a sequential run (each ``fn`` call is
    independent); completion mode never lets a slow page hold finished ones
    back — consumers reassemble by ``PageResult.page``.

    ``workers <= 1`` degenerates to a plain inline loop: zero threads, the
    exact sequential semantics (where completion order and input order
    coincide, so ``ordered`` is moot).

    A failing ``fn`` re-raises here; queued-but-unstarted work is cancelled,
    already-running calls finish and are discarded (same containment as
    read_batch's fan-out). Closing the generator early cleans up the same
    way.
    """
    if workers <= 1:
        for item in items:
            yield fn(item)
        return

    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    window = workers + max(0, lookahead)
    it = iter(items)
    pending: "deque" = deque()  # FIFO for ordered mode; a plain bag otherwise
    # An EXTERNAL executor (the engine's shared page pool) is used as-is and
    # never shut down here — sharing one pool across every stream of an
    # engine is what bounds total page-worker threads no matter how many
    # documents stream concurrently. Owning executors are torn down fully.
    own = executor is None
    ex = executor if executor is not None else ThreadPoolExecutor(max_workers=workers)
    try:
        exhausted = False
        while True:
            while not exhausted and len(pending) < window:
                try:
                    pending.append(ex.submit(fn, next(it)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            if ordered:
                yield pending.popleft().result()
            else:
                done, rest = wait(pending, return_when=FIRST_COMPLETED)
                pending = deque(rest)
                for f in done:
                    yield f.result()
    finally:
        for f in pending:
            f.cancel()
        if own:
            ex.shutdown(wait=True)


def _chunks(seq, n: int):
    """Yield ``seq`` in lists of at most ``n`` (n<=0 => one whole chunk)."""
    if n <= 0:
        yield list(seq)
        return
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


def _fill_lines(page_res: PageResult, items, drop_score: float) -> PageResult:
    """Turn native result items into ``page_res.lines`` (shared by the
    single-image and batch paths, so they can't drift apart)."""
    kept = 0
    for it in items:
        if not it.text.strip() or it.confidence < drop_score:
            continue
        box: Quad = tuple((int(p[0]), int(p[1])) for p in it.box)  # type: ignore
        page_res.lines.append(
            TextLine(
                text=it.text,
                confidence=float(it.confidence),
                box=box,
                source=it.source,
                id=it.id,
                layout_id=it.layout_id,
            )
        )
        kept += 1

    if items and kept == 0:
        # Honest attribution: if the CALLER's stricter drop_score (engine
        # floor is 0.5) is what filtered everything, that is their filter
        # working, not recognition degrading. And the C++ side reports its
        # own text_degraded flag on the structure path — never double-append.
        # Items with EMPTY text are the degradation signal (recognition
        # produced a box but no characters); items with text that only the
        # CALLER's stricter drop_score removed are their filter working.
        # Warn when the empty-text signal exists at all — one caller-filtered
        # line must not silence ten genuinely empty ones.
        genuinely_empty = any(not it.text.strip() for it in items)
        caller_filtered = any(
            it.text.strip() and DROP_SCORE <= it.confidence < drop_score
            for it in items
        )
        already = any(w.startswith("text_degraded") for w in page_res.warnings)
        if (genuinely_empty or not caller_filtered) and not already:
            page_res.warnings.append(
                "text_degraded: detection found regions but no text survived recognition"
            )
    return page_res


@contextlib.contextmanager
def _restore_env(keys):
    """Snapshot the named env keys and restore them (value or absence) on
    exit — the leak-guard around the construct block's env mutations."""
    before = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_check_on_error = _check_on_error_impl


def _failed_page(exc: BaseException, *, page: Optional[int] = None,
                 width: int = 0, height: int = 0) -> PageResult:
    """The ``on_error="skip"`` placeholder: an empty page that says WHY it is
    empty (``warnings=["page_failed: ..."]``), so a contained failure can never
    masquerade as a genuinely blank page."""
    pr = PageResult(width=width, height=height, page=page)
    pr.warnings.append(f"page_failed: {type(exc).__name__}: {exc}")
    return pr


def _ensure_stage_asset(store: ModelStore, stage: str, rel: str) -> str:
    """Resolve a structure-stage asset, converting a download failure (404,
    offline) into ModelLoadError instead of a bare urllib traceback — the
    error a user actually hit when OCR(formulas=True) fetched from a release
    that does not carry the asset."""
    from .errors import ModelLoadError

    try:
        return store.ensure_asset(rel)
    except Exception as exc:
        raise ModelLoadError(
            f"{stage}=True needs the asset '{rel}', which could not be "
            f"resolved ({type(exc).__name__}: {exc}). Point models_dir= at a "
            "directory that contains it, or check the model release."
        ) from exc


def _clamp_quad(box, w: int, h: int):
    """Clamp a region quad into the page.

    The table stage expands each region by TABLE_CROP_MARGIN (3%) before
    cropping; the CROP is clamped to the image but the REPORTED box was not, so
    a table touching the page edge came back with negative coordinates (measured
    -28 on a 960x960 fixture). A consumer slicing `img[y0:y1, x0:x1]` with a
    negative y0 silently gets the wrong strip, and drawing it lands off-canvas.
    Page dims of 0 (a contained failure) disable the clamp rather than collapse
    every point to 0."""
    if w <= 0 or h <= 0:
        return tuple((int(p[0]), int(p[1])) for p in box)
    return tuple((min(max(int(p[0]), 0), w - 1),
                  min(max(int(p[1]), 0), h - 1)) for p in box)


@dataclass(frozen=True)
class _StageRequest:
    """THE per-call stage request, resolved ONCE at the entry point.

    Every reader below the entry points sees only this object, never the raw
    keywords — so a stage flag cannot be dropped, defaulted differently, or
    mis-threaded on one path (the bug class that produced eight distinct
    silent-wrong-answer defects while the flags travelled as four parallel
    keywords through four layers of calls).

    LOAD IS NOT RUN. :meth:`resolve` treats an absent flag (None) as NOT
    REQUESTED — never "whatever this engine happens to have loaded". This
    mirrors the shared gate exactly: parse_options_core() step 1
    (validation/options_core.h:78) is `if (on) requested.request(id)`,
    unconditional, and `loaded` is read in exactly ONE place — step 3's
    availability check at :103. What is resident decides whether a request is
    LEGAL, never whether it was MADE. An EXPLICIT request is still never
    clamped to what was built: it flows through run_with_layout to the gate,
    which raises with the same message HTTP returns for the same request.

    ``run_layout`` already folds in the implication "tables/formulas IMPLY
    layout in the RESULT": those stages recognize content INSIDE layout
    regions, and withholding the regions produced an incoherent page
    (page.tables[0].layout_id pointing at a region that was not there).
    _bind_stages applies the same implication at CONSTRUCTION time."""

    run_layout: bool
    run_tables: bool
    run_formulas: bool
    run_reading_order: bool

    @classmethod
    def resolve(cls, *, layout=None, tables=None, formulas=None,
                reading_order: bool = False) -> "_StageRequest":
        """Resolve raw per-call keywords into the request, refusing the one
        contradiction: layout explicitly OFF with tables/formulas ON (the
        implication would silently switch layout back on). Entry points call
        this eagerly, so an ARGUMENT error raises once at the call — never as
        N per-page `page_failed` warnings under on_error="skip", which exists
        to contain broken pages, not broken arguments."""
        def _on(req) -> bool:
            return False if req is None else bool(req)

        use_tables, use_formulas = _on(tables), _on(formulas)
        if layout is not None and not _on(layout) and (use_tables or use_formulas):
            wanted = " and ".join(
                n for n, v in (("tables", use_tables), ("formulas", use_formulas))
                if v
            )
            raise ValueError(
                f"layout=False cannot be combined with {wanted}=True: "
                f"{wanted} are recognized INSIDE layout regions, so the layout "
                "stage has to run for them to exist. Drop layout=False, or drop "
                f"{wanted}."
            )
        return cls(
            run_layout=_on(layout) or use_tables or use_formulas,
            run_tables=use_tables,
            run_formulas=use_formulas,
            run_reading_order=bool(reading_order),
        )

    @property
    def run_structure(self) -> bool:
        """Does this request route through run_with_layout at all?
        reading_order rides the engine's layout pass even when the caller
        opted out of layout REGIONS in the result — the shared gate
        auto-enables layout for it."""
        return self.run_layout or self.run_reading_order

    def without_reading_order(self) -> "_StageRequest":
        """The same request minus reading_order — for the text-layer structure
        pass, whose recognized lines (which the indices would point into) are
        discarded in favour of the PDF layer's."""
        return _StageRequest(self.run_layout, self.run_tables,
                             self.run_formulas, False)


#: The empty request — what a call gets when it asks for no stage.
_NO_STAGES = _StageRequest(False, False, False, False)


def _fill_structure(page_res: PageResult, r, *, req: "_StageRequest") -> None:
    """Marshal the native run_with_layout result's STRUCTURE outputs
    (layout regions, tables, formulas, reading order, degradation warnings)
    into ``page_res`` — pure translation, no engine or pool state, the
    structure-side sibling of :func:`_fill_lines`."""
    if req.run_layout:
        for lb in r.layout:
            page_res.layout.append(
                LayoutBox(
                    label=lb.label, confidence=float(lb.score),
                    box=tuple((int(p[0]), int(p[1])) for p in lb.box),  # type: ignore
                    id=lb.id,
                    # getattr: bindings older than the nesting support
                    # have no parent_id attribute.
                    parent_id=getattr(lb, "parent_id", -1),
                )
            )
    if r.reading_order:
        # NOT nested under run_layout: reading order is its own request
        # (the engine may compute it while the caller opted out of layout
        # REGIONS in the result).
        page_res.reading_order = list(r.reading_order)
    if req.run_tables:
        for t in r.tables:
            page_res.tables.append(
                TableRegion(
                    html=t.content, score=float(t.score),
                    box=_clamp_quad(t.box, page_res.width, page_res.height),  # type: ignore
                    layout_id=t.layout_id,
                )
            )
        # The FLAG is authoritative; the warning string is optional detail.
        # Requiring both meant a producer that set the flag with no message
        # yielded warnings == [] — a clean-looking degraded page, which is
        # what the mechanism exists to prevent.
        if r.table_degraded:
            page_res.warnings.append(
                f"table_degraded: {r.table_warning or 'no detail'}")
    if req.run_formulas:
        for f in r.formulas:
            page_res.formulas.append(
                FormulaRegion(
                    latex=f.content, score=float(f.score),
                    box=_clamp_quad(f.box, page_res.width, page_res.height),  # type: ignore
                    layout_id=f.layout_id,
                )
            )
        if r.formula_degraded:
            page_res.warnings.append(
                f"formula_degraded: {r.formula_warning or 'no detail'}")
    if r.text_degraded:
        page_res.warnings.append(
            f"text_degraded: {r.text_warning or 'no detail'}")


def _resolve_entry(model, lang, tier):
    """Resolve (model, lang, tier) to a catalog ModelEntry. model= wins; then a
    script lang; then a Latin/CJK tier; else the default tiny tier."""
    if model:
        return resolve_model(model)
    if isinstance(lang, (list, tuple, set)):
        raise ValueError(
            "one engine handles one script — pass a single lang and build a "
            "separate OCR() per language (e.g. OCR(lang='en'), OCR(lang='ar')). "
            "Multi-language in a single engine is not supported."
        )
    if lang:
        key = lang.strip().lower()
        if key in _SCRIPT_LANGS:
            if tier:
                import warnings as _w

                _w.warn(
                    f"tier={tier!r} is ignored for script language {lang!r} "
                    "(it has a single recognizer).",
                    stacklevel=3,
                )
            return resolve_model(_SCRIPT_LANGS[key])
        if key not in _V6_LANGS:
            raise ValueError(
                f"unknown lang {lang!r}. Latin/CJK use tier=tiny|small|medium; "
                f"scripts: {sorted(set(_SCRIPT_LANGS))}."
            )
        if key in ("ja", "japanese") and (tier or "tiny") == "tiny":
            import warnings as _w

            _w.warn(
                "lang='ja' resolved to the tiny tier, which omits Japanese "
                "kana — pass tier='small' or 'medium' for Japanese.",
                stacklevel=3,
            )
    return resolve_model(tier or "tiny")


class OCR:
    """A ready-to-run OCR engine bound to one model + backend.

    Parameters
    ----------
    model:
        Catalog name or alias — ``"tiny"`` (default), ``"small"``, ``"medium"``,
        or a script (``"arabic"``, ``"korean"``, ...).
    mode:
        ``"auto"`` (default — the vendor graph engine when its artefact exists,
        else the ONNX path), ``"native"``/``"ultra"`` (prefer the graph
        engine; falls back to the ONNX path when no artefact exists —
        ``info()["mode"]`` reports what actually came up), or
        ``"onnx"``/``"fast"`` (the .onnx on the vendor's ORT provider, fp16
        where supported, no graph build).
    backend:
        ``"auto"`` (the wheel's best default: on the NVIDIA wheels this is
        the ``"tensorrt"`` engine — the first run builds a cached TensorRT
        engine — elsewhere the CPU path), ``"tensorrt"`` (also ``"trt"``;
        legacy ``"turbo"``) on the NVIDIA build,
        ``"apple"`` (native Metal/MPSGraph on macOS arm64),
        ``"cpu"``, or an explicit EP (``"cuda"``, ``"openvino"``,
        ``"coreml"``, ``"directml"``, ``"rocm"``).
    replicas:
        Number of independent native pipelines in this engine's pool
        (default 1). ``read_batch`` fans out across them, and concurrent
        ``read()`` calls from multiple threads spread across the pool — the
        same replica-pool design as the server's ``--threads N``. Each replica
        holds its own model copy in memory.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        backend: str = "auto",
        *,
        lang: Optional[str] = None,
        tier: Optional[str] = None,
        models_dir: Optional[str] = None,
        device: Optional[str] = None,
        device_id: int = 0,
        use_cls: bool = False,
        mode: EngineMode = "auto",
        replicas: int = 1,
        fp16: bool = True,
        allow_download: bool = True,
        layout: bool = False,
        tables: bool = False,
        formulas: bool = False,
        autorotate: bool = False,
        verbose: bool = False,
        keep_image: Optional[bool] = None,
    ) -> None:
        # Resolution priority: explicit model= wins; else lang(+tier); else tier;
        # else the default tiny tier. lang picks a script recognizer (ko/ar/th/
        # el/ru) or, for Latin/CJK, selects the PP-OCRv6 tier via `tier`.
        self.entry = _resolve_entry(model, lang, tier)
        self.model_name = self.entry.name
        self.use_cls = use_cls
        # Which path to the silicon: "native"/"ultra" = the vendor graph engine
        # (TensorRT / MPSGraph / OpenVINO blob — fastest, needs a built
        # artefact); "onnx"/"fast" = the .onnx on that vendor's ONNX Runtime
        # provider with fp16 where supported and NO graph build; "auto" (the
        # default) takes native when its artefact exists and falls back
        # otherwise. The resolved value is reported by info()["mode"].
        if mode not in ENGINE_MODES:
            # read_pdf(mode=) and on_error= are validated; this one silently
            # accepted any string and behaved like "auto".
            raise ValueError(f"mode must be one of {ENGINE_MODES}, got {mode!r}")
        self.requested_mode = mode
        self.fp16 = fp16
        self.verbose = verbose
        # Tri-state: None = unset, so each method applies its own default —
        # read() keeps the raster (single image, draw()/overlay is the common
        # next step), read_pdf/read_batch DROP it (hundreds of ~6 MB rasters
        # is GBs of silent retention). An explicit OCR(keep_image=...) wins
        # over the per-method default; a per-call keep_image= wins over both.
        self.keep_image = keep_image
        # One native pipeline reuses internal scratch buffers, so it is
        # single-flight even though the GIL is released during inference.
        # Concurrency therefore comes from REPLICAS: independent native
        # pipelines checked out of a queue, exactly the server's replica-pool
        # design (--threads N). replicas=3 on Apple silicon measures ~2.4x
        # one replica. Each replica holds its own copy of the models in
        # memory — size the pool accordingly.
        if replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {replicas}")
        self.replicas = replicas
        self._closed = False
        # ONE shared page-worker pool per engine (created lazily at the
        # first multi-replica stream): every PDF stream of this engine
        # submits its page work here, so M concurrent documents share
        # `replicas` worker threads instead of stacking M*replicas. This
        # replaced a per-generator document-permit gate whose hold-across-
        # yield design deadlocked nested and gathered streams (a permit held
        # while consumer code runs is hold-and-wait by construction, and its
        # thread-id-keyed reentrancy broke under CPython tid recycling).
        # Workers never wait on consumers, so queued work always drains.
        self._pdf_executor = None
        # close() coordination: idempotence + a sentinel for readers parked
        # in the pool queue (see _checkout/close); also guards lazy creation
        # of the shared executor above.
        self._close_mu = threading.Lock()
        set_log_level_default(verbose)

        _native = load_native()
        store = ModelStore(models_dir, allow_download=allow_download)
        self.paths = store.resolve(self.entry, want_cls=use_cls)
        self._store = store

        # Provision the Apple NATIVE-mode bundle (MPSGraph exports + the ANE
        # packages) BEFORE construction — the engine probes for the export
        # dirs at load time, and this must stay outside construct_lock (it may
        # download once).
        #
        # REFUSE rather than degrade. Without the bundle the Apple backend
        # silently ran its CoreML fallback, which is not a graceful
        # degradation but a TRAP: measured on an 83-page document, native
        # Apple is 27.9 pages/s, the plain CPU path 3.9, and the CoreML
        # fallback 1.7 — so asking for the fast backend and missing the bundle
        # left you SLOWER than not asking at all, announced by nothing but a
        # line on stderr. Same rule as _bind_stages below: a request that
        # cannot be honoured fails at construction with the reason, instead of
        # quietly returning something worse.
        if native.resolve_engine(backend) == "apple":
            if not store.ensure_apple_native(self.entry, self.paths):
                reason = getattr(store, "apple_native_reason", None)
                if reason:
                    from .errors import ModelLoadError

                    raise ModelLoadError(
                        f"backend='apple' needs the Apple native export, and "
                        f"{reason}. Without it this backend falls back to "
                        "CoreML, which measures SLOWER than backend='cpu' — "
                        "so it is refused rather than served quietly. Fix the "
                        "cause above, or pick the backend you actually want: "
                        "backend='cpu' (portable ORT path) or "
                        "backend='coreml' (the CoreML fallback, explicitly)."
                    )

        # Serialize env-mutation + construction: the engine reads its EP from
        # process env at construction, so two OCR(...) builds with different
        # backends must not interleave (env is global). The guard restores
        # every mutated key on exit — everything that reads them (pipe.init /
        # load_structure / warmup, all replicas) runs inside this same lock
        # hold, so one build cannot leak EP or structure-model config into the
        # next build or the caller's environment. The native DET config base
        # installed below is process-global but per-instance SAFE: every
        # backend's stages capture their det config at init() — inside this
        # same lock hold — so each engine keeps its own tier's thresholds for
        # life. The base is not restored because only future constructions
        # read it, and each installs its own first.
        _env_keys = (*native.CONSTRUCT_ENV_KEYS, "TABLE_SLANEXT_ENCODER_ONNX", "FORMULA_ONNX", "FORMULA_TOKENIZER")
        with native.construct_lock, _restore_env(_env_keys):
            self.backend, self.provider_summary = configure_backend(
                backend, device=device, device_id=device_id
            )
            native.ensure_backend_supported(backend)
            # Which C++ Backend from the seam registry runs the stages. "cpu"
            # (the ORT-based backend) unless this build actually has the
            # requested vendor backend linked in — see native.resolve_engine.
            self.engine = native.resolve_engine(backend)
            # Tables/formulas are per-layout-region stages, so they need the
            # layout model loaded too. All optional stage models are resolved
            # BEFORE init: the unified pipeline loads its whole stage set at
            # construction (env-driven table/formula bootstrap included).
            need_layout = layout or tables or formulas
            layout_path = (_ensure_stage_asset(
                store, "layout (or tables/formulas, which imply it)",
                "layout/layout.onnx") if need_layout else "")
            doc_ori_path = (_ensure_stage_asset(
                store, "autorotate", "doc_ori.onnx") if autorotate else "")
            if tables:
                os.environ["TABLE_SLANEXT_ENCODER_ONNX"] = _ensure_stage_asset(
                    store, "tables",
                    "table/slanext_encoder/SLANeXt_wired_encoder.onnx",
                )
                # The C++ table backend derives the decoder + dict paths
                # from the encoder's directory — ensuring only the encoder
                # left a cache the load then failed against.
                for sibling in (
                    "table/slanext_encoder/SLANeXt_wired_decoder.bin",
                    "table/slanext_encoder/SLANeXt_dict_infer.txt",
                ):
                    _ensure_stage_asset(store, "tables", sibling)
            if formulas:
                os.environ["FORMULA_ONNX"] = _ensure_stage_asset(
                    store, "formulas",
                    "formula/ppformulanet_s/inference_trt.onnx",
                )
                os.environ["FORMULA_TOKENIZER"] = _ensure_stage_asset(
                    store, "formulas",
                    "formula/ppformulanet_s/tokenizer.json",
                )
            # Install the tier's official detection config (catalog det_cfg —
            # tiny's box_thresh is 0.40, not the 0.45 default) into the native
            # base BEFORE constructing the pipeline. Without this the field
            # was carried by every catalog row and read by nothing, so the
            # default OCR() run mis-thresholded the detector relative to the
            # server. DET_* env overrides still win inside the native layer.
            _dc = self.entry.det_cfg
            _native.set_det_config(
                _dc.resize.limit_type, _dc.resize.limit_side_len,
                _dc.resize.max_side_limit, _dc.db.thresh, _dc.db.box_thresh,
                _dc.db.unclip_ratio,
            )
            # The engine prints load banners to stdout; hush them unless verbose.
            # All replicas are built under the SAME construct_lock hold: the
            # env/backend/det-config state read at construction must not change
            # between replica builds.
            self._pipes = []
            with quiet_stdout(not verbose):
                for _ in range(replicas):
                    pipe = _native.Pipeline()
                    if not pipe.init(
                        self.paths.det, self.paths.rec, self.paths.dict,
                        self.paths.cls or "", layout_path, doc_ori_path,
                        self.engine, mode, fp16, device or "",
                    ):
                        from .errors import ModelLoadError

                        raise ModelLoadError(
                            f"native pipeline failed to load model '{self.model_name}' "
                            f"on backend '{self.engine}' "
                            f"(det={self.paths.det}, rec={self.paths.rec}). "
                            "Check the model files exist and match the backend."
                        )
                    if tables or formulas:
                        pipe.load_structure()
                    pipe.warmup()
                    self._pipes.append(pipe)
            # Replica 0 doubles as the capability probe below; every replica
            # is init'd identically so any one of them answers for the set.
            self._pipe = self._pipes[0]
            self._pool: "queue.Queue" = queue.Queue()
            for pipe in self._pipes:
                self._pool.put(pipe)
        # What the backend actually came up on (an "auto" that fell back to the
        # ONNX path must say so rather than claim the native engine).
        self.mode = self._pipe.mode() if hasattr(self._pipe, "mode") else mode
        # Requested-but-unloaded is an ERROR, not a silent degrade: init() can
        # succeed without an optional stage (missing/unreadable model, or a
        # backend that declined it), and an OCR(tables=True) built that way
        # used to return zero tables forever — implicit reads default to the
        # built capability set, so nothing ever raised. _bind_stages raises
        # ModelLoadError for any requested stage whose probe is false, and
        # binds has_* to what was requested (== loaded, once it returns).
        self._bind_stages(self._pipe, layout=layout, tables=tables,
                          formulas=formulas, autorotate=autorotate,
                          layout_path=layout_path)
        # {capability_name: bool}, keyed by the SAME names the HTTP API uses
        # (layout/tables/formulas/autorotate), so Python and the server describe
        # the same build identically. Intersected with what THIS Pipeline was
        # asked to enable: a stage the engine loaded but the caller opted out of
        # is not available on this object.
        self.capabilities = {
            "layout": self.has_layout,
            "tables": self.has_tables,
            "formulas": self.has_formulas,
            "autorotate": self.autorotate,
        }
        # Anything the engine knows about that the dict above does not name is a
        # capability added to capability_table.def without being surfaced here.
        # Report it rather than silently hiding it from Python callers.
        engine_caps = getattr(self._pipe, "capabilities", None)
        if callable(engine_caps):
            for name, loaded in engine_caps().items():
                self.capabilities.setdefault(name, loaded)

    def _bind_stages(self, pipe, *, layout: bool, tables: bool, formulas: bool,
                     autorotate: bool, layout_path: str = "") -> None:
        """Bind the optional-stage capability flags after construction,
        raising :class:`ModelLoadError` for any REQUESTED stage whose model
        did not actually load — a request that cannot be honoured must fail
        at construction, not silently return pages without its output.
        tables/formulas IMPLY layout (they run per-layout-region, so the
        layout stage loads and is exposed as a capability — same implication
        the server's capability table applies); autorotate stays strictly
        what was requested."""
        from .errors import ModelLoadError

        need_layout = layout or tables or formulas
        if need_layout and not pipe.has_layout():
            raise ModelLoadError(
                "the layout stage was requested (layout/tables/formulas=True) "
                f"but failed to load on backend '{self.engine}'"
                + (f" (layout model: {layout_path})" if layout_path else "")
                + ". Check the model file exists and matches the backend."
            )
        if tables and not pipe.has_table_backend():
            raise ModelLoadError(
                f"tables=True but the table backend failed to load on "
                f"'{self.engine}' — check the SLANeXt encoder asset "
                "(TABLE_SLANEXT_ENCODER_ONNX)."
            )
        if formulas and not pipe.has_formula_backend():
            raise ModelLoadError(
                f"formulas=True but the formula backend failed to load on "
                f"'{self.engine}' — check the PP-FormulaNet assets "
                "(FORMULA_ONNX / FORMULA_TOKENIZER)."
            )
        if autorotate and not pipe.has_doc_ori():
            raise ModelLoadError(
                "autorotate=True but the document-orientation model failed "
                f"to load on '{self.engine}'."
            )
        self.has_layout = need_layout
        self.has_tables = tables
        self.has_formulas = formulas
        self.autorotate = autorotate

    def _live_pipe(self):
        """The capability-probe replica, or the closed error — the ONE place
        that owns the closed-engine message and the close() race (a
        concurrent close() nulls self._pipe; the local read keeps this an
        orderly RuntimeError instead of an AttributeError)."""
        pipe = self._pipe
        if self._closed or pipe is None:
            raise RuntimeError(
                "this OCR engine was closed — construct a new OCR()"
            )
        return pipe

    @staticmethod
    def _require_doc_ori(pipe) -> None:
        """Refusal-beats-silent-no-op for autorotate: an explicit request on
        an engine without the doc-orientation model must raise."""
        if not pipe.has_doc_ori():
            raise ValueError(
                "autorotate requested but this pipeline has no "
                "document-orientation model (construct OCR(..., "
                "autorotate=True) so the model is loaded)"
            )

    def _stream_executor(self):
        """The engine's shared page-worker pool (see __init__). Lazy: image-
        only users never pay for it; replicas=1 streams run inline and never
        call this."""
        from concurrent.futures import ThreadPoolExecutor

        ex = self._pdf_executor
        if ex is None:
            with self._close_mu:
                if self._closed:
                    # A stream created before close() but first advanced
                    # after it used to re-mint a pool nothing would ever
                    # shut down — `replicas` leaked threads per engine.
                    raise RuntimeError(
                        "this OCR engine was closed — construct a new OCR()"
                    )
                ex = self._pdf_executor
                if ex is None:
                    ex = ThreadPoolExecutor(
                        max_workers=self.replicas,
                        thread_name_prefix="turboocr-pdf-pages",
                    )
                    self._pdf_executor = ex
        return ex

    def _keep(self, keep_image: Optional[bool], default: bool) -> bool:
        """Resolve the raster-retention tri-state: per-call beats engine-level
        beats the calling method's own default (True for read(), False for the
        PDF/batch paths)."""
        if keep_image is not None:
            return keep_image
        if self.keep_image is not None:
            return self.keep_image
        return default

    @contextlib.contextmanager
    def _checkout(self):
        """Borrow a free replica; blocks when all are in flight.

        The queue IS the mutual exclusion: a checked-out pipeline is owned by
        exactly one thread until returned, so no per-pipeline lock is needed.
        Concurrent read() calls from user threads spread across the pool
        automatically. The _closed flag check is a fast path, not the safety
        net: a reader that passes it just before close() drains the pool
        would otherwise park in Queue.get() forever (TOCTOU) — close() puts a
        sentinel into the drained pool, and every parked reader re-puts it
        for the next one and raises."""
        if self._closed:
            raise RuntimeError(
                "this OCR engine was closed — construct a new OCR(); read() "
                "after close() used to block forever on the empty replica pool"
            )
        pipe = self._pool.get()
        if pipe is _POOL_CLOSED:
            try:
                raise RuntimeError(
                    "this OCR engine was closed — construct a new OCR()"
                )
            finally:
                # The re-put wakes the next parked reader; in a finally so
                # even an asynchronous exception in this window cannot eat
                # the only sentinel and re-hang the chain.
                self._pool.put(pipe)
        try:
            yield pipe
        finally:
            self._pool.put(pipe)

    # -- single image ------------------------------------------------------
    def read(
        self,
        image: ImageInput,
        *,
        drop_score: float = DROP_SCORE,
        rotate: int = 0,
        layout: Optional[bool] = None,
        reading_order: bool = False,
        tables: Optional[bool] = None,
        formulas: Optional[bool] = None,
        autorotate: Optional[bool] = None,
        text: bool = True,
        keep_image: Optional[bool] = None,
    ) -> PageResult:
        """OCR one image (path / bytes / numpy / PIL); returns a
        :class:`PageResult` with lines in reading order.

        ``layout=True`` also returns layout regions; ``tables=True`` /
        ``formulas=True`` return recognized tables (HTML) / formulas (LaTeX)
        (require ``OCR(tables=True)`` / ``OCR(formulas=True)``).
        ``autorotate=True`` corrects a rotated page first; an explicit
        non-zero ``rotate=`` WINS over autorotate (you told the engine the
        rotation, so the classifier is not consulted).

        ``text=False`` (a layout-only run, the library spelling of the HTTP
        ``?text=0``) goes through the SAME shared request-option gate the server
        runs — see ``include/turbo_ocr/service/validation/options_core.h`` — so
        an unsupported combination raises ``ValueError`` with the exact message
        the HTTP and gRPC surfaces return, rather than quietly coming back with
        a full OCR result labelled layout-only."""
        img = load_image(image)
        check_drop_score(drop_score)
        # Resolve the stage request ONCE, eagerly: an argument contradiction
        # raises here, before the orientation pass below does any work.
        req = _StageRequest.resolve(layout=layout, tables=tables,
                                    formulas=formulas,
                                    reading_order=reading_order)
        # The engine rotates in quarter turns; anything else silently produced a
        # differently-shaped canvas (rotate=45 gave a 659x659 image of a tilted
        # page) that no downstream stage expects. A non-int spelling reached
        # `%` and raised TypeError from string formatting instead of a
        # documented ValueError.
        if isinstance(rotate, bool) or not isinstance(rotate, int):
            raise ValueError(
                f"rotate must be an int (0, 90, 180 or 270), got "
                f"{type(rotate).__name__}"
            )
        angle = rotate % 360
        if angle % 90:
            raise ValueError(
                f"rotate must be a quarter turn (0, 90, 180 or 270), got "
                f"{rotate}. Use autorotate=True to detect the angle instead."
            )
        do_auto = self.autorotate if autorotate is None else autorotate
        # An explicit per-call autorotate=True must either WORK or RAISE — the
        # same refusal-beats-silent-no-op rule the layout/tables/formulas gate
        # below applies. This line used to AND with self.autorotate again,
        # which made the explicit override dead logic: on an instance built
        # with autorotate=False, read(autorotate=True) silently OCR'd the
        # sideways page.
        if do_auto and angle == 0:
            self._require_doc_ori(self._live_pipe())
            with self._checkout() as pipe:
                angle = int(pipe.detect_orientation(np.ascontiguousarray(img, np.uint8)))
        if angle:
            img = rotate_bound(img, angle)
        page = self._read_array(
            img, drop_score=drop_score, rotate=angle, req=req,
            want_text=text, keep_image=keep_image,
        )
        if do_auto:
            # The orientation pass RAN (whatever angle it concluded), so record
            # it. page.orientation alone cannot say this: 0 is both "checked,
            # already upright" and "never checked".
            page.stages = (*page.stages, "autorotate")
        return page

    def _read_array(
        self,
        img: np.ndarray,
        *,
        drop_score: float,  # entry points validated it (check_drop_score)
        rotate: int = 0,
        page: Optional[int] = None,
        req: _StageRequest = _NO_STAGES,
        want_text: bool = True,
        keep_image: Optional[bool] = None,
    ) -> PageResult:
        h, w = img.shape[:2]
        # Metal's maximum texture axis is 16384: on the apple backend a
        # larger image reached MTLTextureDescriptor validation and killed
        # the PROCESS with an uncatchable SIGABRT — from a plain read(), an
        # ndarray, or a PDF page rendered at high dpi. The area-based
        # TURBO_MAX_IMAGE_MP ceiling cannot catch it (17000x2000 is 34 MP).
        # A ValueError here is catchable and, on the PDF paths, containable
        # by on_error="skip".
        if getattr(self, "engine", None) == "apple" and max(h, w) > 16384:
            raise ValueError(
                f"image is {w}x{h} px — the apple backend (Metal) caps each "
                "axis at 16384. Downscale the image, or lower the render "
                "dpi for PDF pages."
            )
        page_res = PageResult(width=w, height=h, page=page, orientation=rotate % 360)

        img = np.ascontiguousarray(img, dtype=np.uint8)
        if self._keep(keep_image, True):
            page_res.image = img

        with self._checkout() as pipe:  # one replica per in-flight run (GIL released in C++)
            # `not want_text` takes this branch even with no structure requested:
            # run_with_layout is where the shared request-option gate runs AND
            # where RunFlags.text=false is honoured (layout-only run, or the
            # gate's rejection for a bare text=False with no layout). Routing it
            # to the plain run() below instead would have returned a full
            # det+rec result for a request that asked for no text at all — the
            # silent-wrong-answer the gate exists to prevent.
            if req.run_structure or not want_text:
                # tables/formulas need layout regions, so layout is on internally.
                # KEYWORDS, not position. The C++ side replaced four positional
                # bools with pipeline::RunFlags precisely so a transposition is
                # unrepresentable; passing them positionally here re-opens the
                # same hole on the Python side of the seam (swapping the last two
                # arguments silently swaps "tables" for "formulas" and nothing
                # would catch it). The nanobind arg names are declared in
                # src/service/python/bindings.cpp.
                r = pipe.run_with_layout(
                    img,
                    layout=req.run_structure,
                    reading_order=req.run_reading_order,
                    tables=req.run_tables,
                    formulas=req.run_formulas,
                    text=want_text,
                )
                items = r.items
                _fill_structure(page_res, r, req=req)
            else:
                items = pipe.run(img)  # native C++ det->sort->(cls)->rec

        _fill_lines(page_res, items, drop_score)
        # Record what actually RAN (see PageResult.stages). autorotate is a
        # pre-step owned by read(), which records its outcome in
        # page_res.orientation, so it is derived from that rather than from a
        # request flag — it is input preparation, not an output stage.
        ran = ["text"] if want_text else []
        if req.run_layout:
            ran.append("layout")
        if req.run_reading_order:
            ran.append("reading_order")
        if req.run_tables:
            ran.append("tables")
        if req.run_formulas:
            ran.append("formulas")
        page_res.stages = tuple(ran)
        return page_res

    # -- batch of images ---------------------------------------------------
    def read_batch(
        self,
        images: List[ImageInput],
        *,
        drop_score: float = DROP_SCORE,
        layout: Optional[bool] = None,
        tables: Optional[bool] = None,
        formulas: Optional[bool] = None,
        reading_order: bool = False,
        autorotate: Optional[bool] = None,
        progress=None,
        keep_image: Optional[bool] = None,
        batch_size: Optional[int] = None,
        on_error: OnError = "raise",
    ) -> DocumentResult:
        """OCR a list of images into a :class:`DocumentResult` (one page each).

        Images go through the native whole-batch submission in groups of
        ``batch_size`` (matching the server's ``/ocr/batch`` chunking), which
        lets the detector see a real batch instead of a batch of one. Requests
        that need per-image stages — layout, tables, formulas, autorotate —
        fall back to reading one at a time; see :meth:`_can_batch`.

        ``progress`` may be True (log to stderr) or a callable
        ``progress(done, total)``. ``keep_image=False`` drops each page raster
        after reading (see :meth:`read_pdf`).

        ``on_error="raise"`` (default) propagates the first failing image's
        exception and cancels images not yet started. ``on_error="skip"``
        contains a failure to ITS page: the failed image becomes an empty
        :class:`PageResult` carrying ``warnings=["page_failed: ..."]`` and
        every other image is still read — for a long unattended batch where
        one corrupt file must not cost the other results.

        Returns a DocumentResult — iterate it for the per-image pages, or use
        ``doc.to_tsv()`` / ``doc.text`` etc."""
        _check_on_error(on_error)
        check_drop_score(drop_score)
        req = _StageRequest.resolve(layout=layout, tables=tables,
                                    formulas=formulas,
                                    reading_order=reading_order)
        self._live_pipe()
        if batch_size is None:
            # Replica-aware default. Chunking into 8s feeds the detector a real
            # batch, which is the win at replicas=1 — but it also coarsens the
            # fan-out granularity, so with a replica pool the chunks stop
            # load-balancing and throughput DROPS: measured on 80 fixtures,
            # replicas=3 gave 33.1 img/s at batch_size=8 vs 38.0 at 1, and
            # replicas=6 gave 45.3 vs 50.5. Consistent at every replica count
            # >= 2, so the default follows the pool rather than a constant.
            batch_size = 8 if self.replicas == 1 else 1
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        # Batch reads default to DROPPING rasters (keep_image=False): a long
        # batch at ~6 MB/raster retains GBs invisibly. Pass keep_image=True
        # (per call or on the engine) when you need draw()/searchable PDFs.
        keep_image = self._keep(keep_image, False)
        total = len(images)
        report = _make_progress(progress, total, "images")
        doc = DocumentResult()

        if self._can_batch(req=req, autorotate=autorotate):
            chunks = list(_chunks(images, batch_size))

            def _run_chunk(chunk) -> List[PageResult]:
                # Load with per-image containment: one unreadable file in a
                # chunk must not (in skip mode) take its chunk-mates down.
                slots: List[Optional[PageResult]] = [None] * len(chunk)
                loaded = []  # (index-in-chunk, array)
                for j, im in enumerate(chunk):
                    try:
                        loaded.append(
                            (j, np.ascontiguousarray(load_image(im), dtype=np.uint8))
                        )
                    except Exception as exc:
                        if on_error == "raise":
                            raise
                        slots[j] = _failed_page(exc)
                arrays = [a for _, a in loaded]
                batch = None
                if arrays:
                    try:
                        with self._checkout() as pipe:
                            batch = pipe.run_batch(arrays)
                        # The native call returns exactly one result list per
                        # input, in input order. Assert it rather than zip(): a
                        # short list would silently DROP pages, turning a
                        # backend bug into missing text.
                        if len(batch) != len(arrays):
                            raise RuntimeError(
                                f"native run_batch returned {len(batch)} results "
                                f"for {len(arrays)} images — refusing to drop pages"
                            )
                    except Exception:
                        if on_error == "raise":
                            raise
                        # The whole native chunk failed; re-run its images one
                        # at a time so the failure lands on the image that
                        # caused it, not on the chunk.
                        batch = None
                        for j, arr in loaded:
                            try:
                                # _NO_STAGES: the batched path is only entered
                                # when the resolved request is empty
                                # (_can_batch), so the rescue run must match
                                # the pages that batched successfully.
                                slots[j] = self._read_array(
                                    arr, drop_score=drop_score,
                                    keep_image=keep_image,
                                )
                            except Exception as exc:
                                slots[j] = _failed_page(
                                    exc, width=arr.shape[1], height=arr.shape[0]
                                )
                if batch is not None:
                    for (j, arr), items in zip(loaded, batch):
                        h, w = arr.shape[:2]
                        pr = PageResult(width=w, height=h, stages=("text",))
                        if keep_image:
                            pr.image = arr
                        slots[j] = _fill_lines(pr, items, drop_score)
                if any(pr is None for pr in slots):  # pragma: no cover
                    # A real raise, not an assert: under python -O an assert
                    # vanishes and a None would land in DocumentResult.pages.
                    raise RuntimeError(
                        "internal error: a chunk slot was left unfilled — "
                        "refusing to emit None pages")
                return slots  # type: ignore[return-value]

            # Per-IMAGE progress, even though a chunk completes as a unit: the
            # callback contract is progress(done, total) counted in images, and
            # a caller's bar must not jump by batch_size (nor skip 1..n-1
            # entirely). With replicas the chunks FINISH out of order, so
            # `done` counts completed images monotonically while doc.pages is
            # assembled by chunk index — order and progress are decoupled.
            done = 0
            done_mu = threading.Lock()

            def _tick(count: int) -> None:
                nonlocal done
                with done_mu:
                    for k in range(count):
                        report(done + k + 1)
                    done += count

            if self.replicas == 1 or len(chunks) <= 1:
                for chunk in chunks:
                    doc.pages.extend(_run_chunk(chunk))
                    _tick(len(chunk))
                return doc

            from concurrent.futures import ThreadPoolExecutor, as_completed

            per_chunk: List[Optional[List[PageResult]]] = [None] * len(chunks)
            # Not `with`: __exit__ is shutdown(wait=True) WITHOUT cancellation,
            # which on a raise would still run every queued chunk to completion
            # before propagating. cancel_futures stops queueing at the failure.
            ex = ThreadPoolExecutor(max_workers=self.replicas)
            try:
                futs = {ex.submit(_run_chunk, c): i for i, c in enumerate(chunks)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    per_chunk[i] = fut.result()  # re-raises worker failures
                    _tick(len(chunks[i]))
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
            for pages in per_chunk:
                if pages is None:  # pragma: no cover — every future resolved or raised
                    raise RuntimeError(
                        "internal error: a chunk result went missing — "
                        "refusing to drop pages")
                doc.pages.extend(pages)
            return doc

        def _read_one(im) -> PageResult:
            try:
                return self.read(im, drop_score=drop_score, layout=layout,
                                 tables=tables, formulas=formulas,
                                 reading_order=reading_order,
                                 autorotate=autorotate, keep_image=keep_image)
            except Exception as exc:
                if on_error == "raise":
                    raise
                return _failed_page(exc)

        if self.replicas > 1 and total > 1:
            # Per-image stages (layout/autorotate/...) cannot use the native
            # whole-batch call, but they still spread across replicas: read()
            # checks a free one out per call.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            img_slots: List[Optional[PageResult]] = [None] * total
            done = 0
            # Same reason as the chunk fan-out above: `with` would drain the
            # queue before propagating a failure. Distinct names from the
            # chunk fan-out above — one function, two executors of different
            # result types.
            image_ex = ThreadPoolExecutor(max_workers=self.replicas)
            try:
                img_futs = {image_ex.submit(_read_one, im): i
                            for i, im in enumerate(images)}
                for img_fut in as_completed(img_futs):
                    img_slots[img_futs[img_fut]] = img_fut.result()
                    done += 1
                    report(done)
            finally:
                image_ex.shutdown(wait=True, cancel_futures=True)
            for pr in img_slots:
                if pr is None:  # pragma: no cover — every future resolved or raised
                    raise RuntimeError(
                        "internal error: an image slot was left unfilled — "
                        "refusing to emit None pages")
                doc.pages.append(pr)
            return doc

        for i, im in enumerate(images, 1):
            doc.pages.append(_read_one(im))
            report(i)
        return doc

    def _can_batch(self, *, req: _StageRequest,
                   autorotate: Optional[bool]) -> bool:
        """True when a batch can go through the native whole-batch submission.

        Only the plain det->rec path is batched: every structure stage runs
        through ``run_with_layout``, and autorotate needs a per-image
        orientation pass first, so those keep the one-image-at-a-time loop.
        The decision reads the resolved REQUEST, not what the engine happens
        to carry (Load is not Run): reading self.has_* here meant a loaded
        engine asking for plain text was denied the native whole-batch call
        for stages it never requested — silently slower, no error."""
        if not hasattr(self._pipe, "run_batch"):
            return False  # extension predates the batch binding
        if self.autorotate if autorotate is None else autorotate:
            return False
        return not req.run_structure

    # -- PDF ---------------------------------------------------------------
    def read_pdf(
        self,
        pdf: ImageInput,
        *,
        dpi: int = DEFAULT_DPI,
        pages: Optional[List[int]] = None,
        drop_score: float = DROP_SCORE,
        max_pages: Optional[int] = None,
        mode: PdfMode = "ocr",
        layout: Optional[bool] = None,
        tables: Optional[bool] = None,
        formulas: Optional[bool] = None,
        reading_order: bool = False,
        progress=None,
        keep_image: Optional[bool] = None,
        on_error: OnError = "raise",
        autorotate: Optional[bool] = None,
        password: Optional[str] = None,
    ) -> DocumentResult:
        """Read a PDF — PDF support is built in (pypdfium2 ships with the
        engine wheel). ``progress`` may be True (log to stderr) or a callable
        ``progress(done, total)``.

        ``mode`` picks how each page's text is obtained:

        * ``"ocr"`` (default) — render every page and OCR it, ignoring any
          embedded text layer. This is an OCR engine: unless you say
          otherwise, every character in the result came from the recognizer,
          on every page, whatever the input was;
        * ``"auto"`` — per page: the embedded text layer when the page has
          one AND a quality gate trusts it (~0 ms, byte-exact), OCR for the
          rest. Born-digital PDFs read ~10x faster and more accurately (no
          recognizer means no misreads). Two things to know before opting
          in: a scan that ALREADY carries a text layer from earlier,
          possibly worse, OCR software is served as-is; and text a PDF only
          contains as an IMAGE (a logo, a pasted screenshot) is invisible to
          the layer, so those lines are missing. Check ``line.source`` to
          see which path a line came from;
        * ``"text"`` — the EMBEDDED text layer only: no rendering, no OCR, no
          models. (PDFium itself is globally single-threaded — every pdfium
          call in this process serializes behind one lock, which is also what
          makes concurrent ``aread_pdf`` calls safe — so the speed here comes
          from skipping rasterization and OCR, not from workers, and
          ``replicas``/async buy nothing.) Lines carry ``source="pdf"`` and
          confidence 1.0; a scanned page simply comes back empty.

        Pages fan out across the replica pool: with ``OCR(replicas=N)``, up to
        N pages are OCR'd concurrently (rendering stays on the calling thread
        — it is ~5 ms/page against 50-250 ms of OCR). Results are assembled
        strictly in page order and each page's result is independent of the
        others, so the document is identical to a sequential run; ``replicas=1``
        IS the sequential run. At most ``replicas + 1`` page rasters are in
        flight at any moment.

        ``keep_image`` defaults to **False** here (since 4.0.0a6): a raster is
        ~6 MB at 150 DPI, so a few hundred pages silently retained GBs. Pass
        ``keep_image=True`` when you need the rasters afterwards
        (``doc.save_searchable_pdf`` / ``draw``) — or use
        :meth:`pdf_to_searchable`, which keeps them automatically. An
        engine-level ``OCR(keep_image=...)`` overrides this default;
        ``read()`` still keeps rasters by default.

        ``on_error="raise"`` (default) propagates the first failing page's
        exception; ``on_error="skip"`` contains a page failure to that page —
        it comes back as an empty :class:`PageResult` with
        ``warnings=["page_failed: ..."]`` while every other page is still
        read. A document that cannot be OPENED at all always raises.

        ``autorotate`` corrects rotated scans per page before OCR (``None``
        inherits the engine-level setting; text-layer pages are never
        rotated) — see :meth:`read_pdf_stream`.

        ``password`` unlocks an encrypted PDF (user or owner password)."""
        doc = DocumentResult(
            source=str(pdf) if isinstance(pdf, (str, os.PathLike)) else ""
        )
        for pr in self.read_pdf_stream(
            pdf, dpi=dpi, pages=pages, drop_score=drop_score,
            max_pages=max_pages, mode=mode, layout=layout, tables=tables,
            formulas=formulas, reading_order=reading_order, progress=progress,
            keep_image=keep_image, on_error=on_error, autorotate=autorotate,
            password=password,
        ):
            doc.pages.append(pr)
        return doc

    def read_pdf_stream(
        self,
        pdf: ImageInput,
        *,
        dpi: int = DEFAULT_DPI,
        pages: Optional[List[int]] = None,
        drop_score: float = DROP_SCORE,
        max_pages: Optional[int] = None,
        mode: PdfMode = "ocr",
        layout: Optional[bool] = None,
        tables: Optional[bool] = None,
        formulas: Optional[bool] = None,
        reading_order: bool = False,
        ordered: bool = True,
        progress=None,
        keep_image: Optional[bool] = None,
        on_error: OnError = "raise",
        autorotate: Optional[bool] = None,
        password: Optional[str] = None,
    ) -> Generator[PageResult, None, None]:
        """Stream a PDF's pages as a generator of :class:`PageResult`, yielding
        each page as soon as it is ready instead of assembling a whole
        :class:`DocumentResult` first — :meth:`read_pdf` is exactly this,
        drained into a document.

        Pages OCR across the replica pool with the same bounded window as
        :meth:`read_pdf` (at most ``replicas + 1`` page rasters in flight), so
        streaming a thousand-page scan holds a handful of pages of memory, not
        the document.

        ``ordered=True`` (default) yields strictly in page order — the first
        page arrives as soon as IT is done, which for a long document is far
        before the last page even renders. ``ordered=False`` yields in
        COMPLETION order: no finished page ever waits on a slower earlier one;
        use ``PageResult.page`` (1-based) to reassemble. Each page's result is
        independent, so both modes produce the same set of results.

        ``mode`` works exactly as in :meth:`read_pdf` — ``"ocr"`` (the
        default) OCRs every page, ``"text"`` streams the embedded text layer
        with no OCR at all, ``"auto"`` serves trusted text layers and OCRs
        everything else. Under ``"auto"``, engines built
        with layout/tables/formulas still run those STRUCTURE stages on
        text-layer pages (the page is rendered for the structure pass while
        its text comes byte-exact from the layer — the server's Geometric
        behavior); ``reading_order`` is not computed for text-layer pages.

        ``progress`` counts YIELDED pages (monotone in both modes). The
        generator cleans up after itself when closed early — breaking out of
        the loop cancels queued pages. Argument validation happens at CALL
        time (this returns an already-validated generator), so a bad ``mode``
        raises here, not at the first ``next()`` on some other thread.

        ``on_error`` works as in :meth:`read_pdf`: ``"skip"`` turns a failing
        page — OCR failures AND page render/extract failures — into an empty
        result with a ``page_failed`` warning instead of ending the stream.

        Concurrency: every stream of one engine shares a single pool of
        ``replicas`` page-worker threads, so any number of concurrent
        documents — gathered, interleaved (``zip(a_stream, b_stream)``), or
        nested inside each other's loops, sync or async — always make
        progress and never stack threads. Each OPEN stream still holds its
        own bounded look-ahead window (at most ``replicas + 1`` rendered
        pages), so raster memory scales with concurrently-open streams;
        close or exhaust streams you are done with. ``mode="text"`` streams
        use no workers at all.

        ``autorotate`` works as in :meth:`read`: ``None`` inherits the
        engine-level setting, ``True``/``False`` overrides per call. A rotated
        scan is detected per page (0/90/180/270), rotated upright before OCR,
        and the applied angle lands in ``PageResult.orientation``. Text-layer
        pages (``mode="text"``/``"auto"``) are inherently upright — the layer
        stores logical text — so only OCR'd pages are ever rotated."""
        # EAGER validation: everything below raises at the CALL, not at the
        # first next() (which may run on another thread via the async
        # wrapper, detaching the traceback from the caller).
        _check_pdf_mode(mode)
        _check_on_error(on_error)
        from .options import check_max_pages

        check_max_pages(max_pages)
        check_pages(pages)
        check_drop_score(drop_score)
        req = _StageRequest.resolve(layout=layout, tables=tables,
                                    formulas=formulas,
                                    reading_order=reading_order)
        if mode == "text" and req.run_structure:
            wanted = ", ".join(
                n for n, v in (("layout", layout), ("tables", tables),
                               ("formulas", formulas),
                               ("reading_order", reading_order)) if v
            )  # raw flags: name only what the CALLER spelled out
            # Refusal beats silent no-op: mode="text" serves the embedded
            # layer only — no rendering, no models — so these requests used to
            # come back as empty lists indistinguishable from "ran and found
            # nothing".
            raise ValueError(
                f'mode="text" cannot run {wanted}: it serves the PDF\'s '
                "embedded text layer only, with no rendering and no models. "
                'Use mode="auto" (text layer for trusted pages, structure '
                'stages on the rendered raster) or mode="ocr".'
            )
        if dpi < 1:
            # dpi=0/-100 died deep inside pdfium as "Crop exceeds page
            # dimensions" — or SUCCEEDED for text-layer pages, so the
            # failure depended on document content.
            raise ValueError(f"dpi must be >= 1, got {dpi!r}")
        # PDF reads default to DROPPING page rasters — see read_pdf's note.
        keep_image = self._keep(keep_image, False)
        do_auto = self.autorotate if autorotate is None else autorotate
        pipe0 = self._live_pipe()
        # mode="text" never rotates, so the model is not required there.
        if do_auto and mode != "text":
            self._require_doc_ori(pipe0)
        return self._stream_pdf_pages(
            pdf, dpi=dpi, pages=pages, drop_score=drop_score,
            max_pages=max_pages, text_source=mode, ordered=ordered,
            progress=progress, keep_image=keep_image, on_error=on_error,
            do_auto=do_auto, password=password, req=req,
        )

    def _stream_pdf_pages(
        self, pdf, *, dpi, pages, drop_score, max_pages, text_source, ordered,
        progress, keep_image, on_error, do_auto, password,
        req: _StageRequest = _NO_STAGES,
    ) -> Generator[PageResult, None, None]:
        """The generator behind :meth:`read_pdf_stream` — arguments arrive
        pre-validated and pre-resolved (keep_image and do_auto are concrete
        bools). The public ``mode=`` keyword becomes ``text_source`` at this
        boundary: "mode" already means the ENGINE execution path
        (OCR(mode=)/info()["mode"]), and one word carrying both meanings is
        how pdf_to_searchable earned a six-line disambiguation comment."""
        from .pdf import iter_pdf_pages, pdf_page_count

        if progress:
            # Totals consult the document so a pages= list with out-of-range
            # entries cannot overshoot (a bar that never completes).
            n_doc = pdf_page_count(pdf, password=password)
            if pages is not None:
                total = len([p for p in pages if 1 <= p <= n_doc])
            else:
                total = n_doc
            if max_pages:
                total = min(total, max_pages)
        else:
            total = 0
        report = _make_progress(progress, total, "pages")

        def _text_page(page_no, w, h, lines) -> PageResult:
            # "text" is honest here: the page HAS text, it just came from the
            # PDF's own layer rather than the recognizer (every line carries
            # source="pdf"). Structure stages, if requested, are appended by the
            # _read_array pass that follows.
            pr = PageResult(width=w, height=h, page=page_no, stages=("text",))
            for text, quad in lines:
                pr.lines.append(
                    TextLine(text=text, confidence=1.0, box=quad, source="pdf")
                )
            pr.dpi = dpi
            return pr

        if text_source == "text":
            # Embedded text layer only — no rendering, no OCR, no engine
            # work; serialized behind the process-wide PDFium lock (held per
            # page, so concurrent callers interleave). Routed through
            # iter_pdf_pages so on_error="skip" contains a failing page here
            # too (it used to end a 500-page extraction at page one), and a
            # layer-less page carries a no_text_layer warning instead of
            # being indistinguishable from a blank page.
            for i, item in enumerate(
                iter_pdf_pages(pdf, dpi=dpi, pages=pages, max_pages=max_pages,
                               mode="text", password=password,
                               on_error=on_error), 1
            ):
                kind, page_no, *rest = item
                if kind == "error":
                    pr = PageResult(page=page_no)
                    pr.warnings.append(f"page_failed: {rest[0]}")
                    pr.dpi = dpi
                else:
                    w, h, lines, _arr, page_warns = rest
                    pr = _text_page(page_no, w, h, lines)
                    pr.warnings.extend(page_warns)
                yield pr
                report(i)
            return

        # Text-layer pages carry no raster by default; render one anyway when
        # the engine runs structure stages (they need pixels) or the caller
        # asked to keep page images (draw()/save_searchable_pdf afterwards).
        # Load is not Run: what the CALL asked for, not what the engine holds.
        # Reading self.has_* here made a text-layer page and a rendered page in
        # the SAME document disagree about which stages ran. run_layout folds
        # in tables/formulas; reading_order alone is NOT structure here (it is
        # never computed for text-layer pages — its indices would point into
        # the discarded OCR line list).
        structure_wanted = req.run_layout
        text_with_raster = structure_wanted or keep_image

        def _process(item):
            kind, page_no, *rest = item
            try:
                if kind == "error":
                    # Producer-side containment (on_error="skip"): the page
                    # failed to load/extract/render inside iter_pdf_pages.
                    pr = PageResult(page=page_no)
                    pr.warnings.append(f"page_failed: {rest[0]}")
                    pr.dpi = dpi
                    return pr
                if kind == "text":
                    w, h, lines, arr, page_warns = rest
                    pr = _text_page(page_no, w, h, lines)
                    pr.warnings.extend(page_warns)
                    if arr is not None:
                        if keep_image:
                            pr.image = np.ascontiguousarray(arr, dtype=np.uint8)
                        if structure_wanted:
                            # Server parity (Geometric + layout): the page's
                            # TEXT stays byte-exact from the layer while the
                            # structure stages run on the rendered raster.
                            #
                            # want_text is NOT simply False here. Tables and
                            # formulas RECOGNIZE text inside their regions, so
                            # the shared request gate rejects text=0 alongside
                            # them ("tables=1/formulas=1 need the OCR pass") —
                            # which made mode="auto" raise ValueError on every
                            # text-layer page of an engine built with
                            # tables=True or formulas=True. A layout-ONLY pass
                            # can still skip recognition, and does, because
                            # that is the cheaper half of the common case.
                            # Either way sr.lines is discarded: the page's
                            # lines stay the layer's.
                            sr = self._read_array(
                                arr, drop_score=drop_score, page=page_no,
                                keep_image=False,
                                req=req.without_reading_order(),
                                want_text=req.run_tables or req.run_formulas,
                            )
                            pr.layout = sr.layout
                            pr.tables = sr.tables
                            pr.formulas = sr.formulas
                            pr.warnings.extend(sr.warnings)
                            # The structure pass's record joins the page's.
                            # Its "text" entry is dropped: the recognized
                            # lines were discarded above, and the page's
                            # "text" (the layer's) is already recorded.
                            pr.stages = (*pr.stages,
                                         *(s for s in sr.stages if s != "text"))
                    return pr
                arr = rest[0]
                angle = 0
                if do_auto:
                    # Per-page, exactly like read(): detect on the raster,
                    # rotate upright, then OCR. _read_array records the angle
                    # as PageResult.orientation via rotate=.
                    arr = np.ascontiguousarray(arr, dtype=np.uint8)
                    with self._checkout() as pipe:
                        angle = int(pipe.detect_orientation(arr))
                    if angle:
                        arr = rotate_bound(arr, angle)
                pr = self._read_array(
                    arr, drop_score=drop_score, page=page_no,
                    keep_image=keep_image, rotate=angle, req=req,
                )
                if do_auto:
                    # Same record as read(): the orientation pass RAN,
                    # whatever angle it concluded.
                    pr.stages = (*pr.stages, "autorotate")
                pr.dpi = dpi
                return pr
            except Exception as exc:
                if on_error == "raise":
                    raise
                w = h = 0
                if kind == "img" and rest and hasattr(rest[0], "shape"):
                    h, w = rest[0].shape[:2]
                pr = _failed_page(exc, page=page_no, width=w, height=h)
                pr.dpi = dpi
                return pr

        # Page work rides the engine's SHARED executor, so any number of
        # concurrent streams (sync, async, nested inside each other's loops)
        # share `replicas` worker threads and always make progress — workers
        # never wait on consumers. Memory note: each open stream still keeps
        # its own bounded look-ahead window of at most replicas+1 rendered
        # pages, so raster memory scales with the streams a caller holds
        # open concurrently; threads do not.
        executor = self._stream_executor() if self.replicas > 1 else None
        try:
            for i, pr in enumerate(
                _parallel_map(
                    iter_pdf_pages(pdf, dpi=dpi, pages=pages,
                                   max_pages=max_pages, mode=text_source,
                                   password=password,
                                   text_with_raster=text_with_raster,
                                   on_error=on_error),
                    _process, self.replicas, ordered=ordered,
                    executor=executor,
                ),
                1,
            ):
                yield pr
                report(i)
        except RuntimeError as exc:
            if "cannot schedule new futures" in str(exc):
                # close() shut the shared executor mid-stream: name the real
                # cause, not the executor mechanics.
                raise RuntimeError(
                    "this OCR engine was closed — construct a new OCR()"
                ) from exc
            raise

    def pdf_to_searchable(
        self,
        input_pdf: ImageInput,
        output_pdf: str,
        *,
        dpi: int = DEFAULT_DPI,
        pages: Optional[List[int]] = None,
        max_pages: Optional[int] = None,
        drop_score: float = DROP_SCORE,
        progress=None,
        on_error: OnError = "raise",
        autorotate: Optional[bool] = None,
        password: Optional[str] = None,
    ) -> str:
        """Stream a PDF through OCR into a SEARCHABLE PDF (page image + invisible
        text layer), writing ``output_pdf``. This is :meth:`read_pdf_stream`
        (ordered) piped into the PDF writer: pages OCR across the replica pool
        but the output is written strictly in page order, holding at most
        ``replicas + 1`` page rasters at a time — so it scales to large scans.
        ``keep_image=True`` is forced: the output embeds each page's raster,
        regardless of the engine-level ``keep_image`` default.
        ``on_error="skip"`` writes a failed page as an empty (image-less)
        placeholder page instead of aborting the whole document."""
        from .searchable_pdf import build_searchable_pdf

        build_searchable_pdf(
            self.read_pdf_stream(
                input_pdf, dpi=dpi, pages=pages, drop_score=drop_score,
                max_pages=max_pages, progress=progress, keep_image=True,
                on_error=on_error, autorotate=autorotate, password=password,
                # Pinned page-text source: the output EMBEDS rasters, so
                # every page must render.
                mode="ocr",
            ),
            out_path=output_pdf,
        )
        return output_pdf

    # -- introspection -----------------------------------------------------
    def detect_orientation(self, image: ImageInput) -> int:
        """Detected page rotation (0/90/180/270), if a doc-orientation model is
        loaded; else 0."""
        img = np.ascontiguousarray(load_image(image), dtype=np.uint8)
        with self._checkout() as pipe:
            return int(pipe.detect_orientation(img))

    # -- asyncio -----------------------------------------------------------
    # Thin, honest sugar: each coroutine runs its sync twin in a worker thread
    # via asyncio.to_thread. The parallelism is real and comes from the layers
    # below — the GIL is released during native inference and one OCR object
    # is thread-safe against its replica pool — so `asyncio.gather()` over
    # these genuinely overlaps work up to `replicas`. There is no separate
    # async engine to configure, and awaiting with replicas=1 serializes
    # exactly like the sync API (construct with replicas=N to scale).

    async def aread(self, image: ImageInput, **kwargs) -> PageResult:
        """Async :meth:`read` — same parameters, same :class:`PageResult`."""
        import asyncio

        return await asyncio.to_thread(self.read, image, **kwargs)

    async def aread_batch(self, images: List[ImageInput], **kwargs) -> DocumentResult:
        """Async :meth:`read_batch` — same parameters, same result.

        Prefer this over gathering many :meth:`aread` calls when you have the
        list up front: the batch path feeds the detector real batches."""
        import asyncio

        return await asyncio.to_thread(self.read_batch, images, **kwargs)

    async def aread_pdf(self, pdf: ImageInput, **kwargs) -> DocumentResult:
        """Async :meth:`read_pdf` — same parameters, same result. Pages
        already fan out across the replica pool inside one call, so a single
        awaited ``aread_pdf`` uses every replica; gathering several documents
        at once shares the same pool between them."""
        import asyncio

        return await asyncio.to_thread(self.read_pdf, pdf, **kwargs)

    async def aread_pdf_stream(
        self, pdf: ImageInput, **kwargs
    ) -> AsyncIterator[PageResult]:
        """Async :meth:`read_pdf_stream` — ``async for page in
        ocr.aread_pdf_stream(pdf)`` yields each :class:`PageResult` as it is
        ready (``ordered=False`` for completion order), without blocking the
        event loop. Breaking out of the loop cleans up the queued work.

        Every step of one stream runs on its OWN dedicated thread, not
        asyncio's shared default executor. Two hangs required that:
        gathering more streams than the shared executor had threads parked
        every thread in ``_doc_gate.acquire()`` and wedged the whole loop
        (the permit-holding stream could never get a thread to advance); and
        cancelling a task mid-``next()`` had the cleanup close the generator
        from a second thread while it was still executing — ``ValueError:
        generator already executing`` instead of ``CancelledError``. One
        single-thread executor per stream gives gate waiters their own
        threads and serializes close() AFTER any in-flight step."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        gen = self.read_pdf_stream(pdf, **kwargs)
        sentinel = object()
        loop = asyncio.get_running_loop()
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aread_pdf_stream")
        try:
            while True:
                item = await loop.run_in_executor(ex, next, gen, sentinel)
                if item is sentinel:
                    return
                # mypy sees `object` through run_in_executor's erasure; the
                # generator yields PageResult by its own annotation.
                yield item  # type: ignore[misc]
        finally:
            # Close on the SAME single worker: queued behind any in-flight
            # next(), so the generator is never entered from two threads.
            # Shielded so a second cancellation cannot orphan the close —
            # and even if the await itself is cancelled, the close call is
            # already queued on the executor and WILL run.
            close_fut = loop.run_in_executor(ex, gen.close)
            try:
                await asyncio.shield(close_fut)
            except asyncio.CancelledError:
                pass
            finally:
                ex.shutdown(wait=False)

    def close(self) -> None:
        """Release the native ONNX sessions. The engine is unusable afterward.
        Optional — fine to rely on GC for the common one-engine-per-process
        case; use this (or the context manager) when churning many engines.
        A read that already CHECKED OUT a replica finishes before its pipe
        is dropped (every replica is reclaimed from the pool first); a read
        that has not yet checked out raises the closed error instead.
        Idempotent and race-safe: two threads closing concurrently used to
        each try to drain `replicas` pipes from a pool holding only
        `replicas` — both hung forever."""
        with self._close_mu:
            if self._closed:
                return  # second closer returns; the first is draining
            self._closed = True  # checked by _checkout: raise, never hang
            n = len(self._pipes)
            ex = self._pdf_executor
            self._pdf_executor = None
        if ex is not None:
            # Queued page work is cancelled; running pages finish. A stream
            # still pumping surfaces the closed error (the submit-after-
            # shutdown RuntimeError is translated in _stream_pdf_pages).
            ex.shutdown(wait=False, cancel_futures=True)
        for _ in range(n):
            pipe = self._pool.get()
            if pipe is _POOL_CLOSED:  # cannot happen before our sentinel; belt
                self._pool.put(pipe)
                break
        self._pipes = []
        self._pipe = None
        # Wake any reader that raced past the _closed check into Queue.get().
        self._pool.put(_POOL_CLOSED)

    def __enter__(self) -> "OCR":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def info(self) -> dict:
        return {
            "model": self.model_name,
            "backend": self.backend,
            "engine": self.engine,
            "mode": getattr(self, "mode", None),
            # what the caller ASKED for, next to what they got — the pair
            # that answers "did native actually come up?" at a glance.
            "requested_mode": getattr(self, "requested_mode", None),
            "replicas": self.replicas,
            "fp16": self.fp16,
            "provider_summary": self.provider_summary,
            "use_cls": self.use_cls,
            "layout": self.has_layout,
            "capabilities": dict(self.capabilities),
            "det": self.paths.det,
            "rec": self.paths.rec,
            "dict": self.paths.dict,
            "hardware": detect_hardware().vendor,
            "native": True,
            # True once ANY layout session in this process dropped CoreML and
            # rebuilt on the CPU provider (process-wide one-way latch — see
            # docs/reference/python.md, Accelerator degradation). Layout still
            # works, just unaccelerated. getattr: extensions older than the
            # binding cannot have latched, so False is the honest answer.
            "layout_coreml_dropped": bool(
                getattr(native.load_native(), "coreml_layout_wedged",
                        lambda: False)()
            ),
        }
