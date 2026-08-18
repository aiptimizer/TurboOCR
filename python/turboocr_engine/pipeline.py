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
import queue
import threading
from typing import List, Optional

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

DROP_SCORE = 0.5  # kDropScore in the C++ engine (applied there too; a safety net here)

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
        page_res.warnings.append(
            "text_degraded: detection found regions but no text survived recognition"
        )
    return page_res


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
        else the ONNX path), ``"native"``/``"ultra"`` (require the graph
        engine), or ``"onnx"``/``"fast"`` (the .onnx on the vendor's ORT
        provider, fp16 where supported, no graph build).
    backend:
        ``"auto"`` (fast-setup — best no-build EP for your hardware), ``"turbo"``
        (TensorRT on the NVIDIA build), ``"cpu"``, or an explicit EP
        (``"cuda"``, ``"openvino"``, ``"coreml"``, ``"directml"``, ``"rocm"``).
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
        mode: str = "auto",
        replicas: int = 1,
        fp16: bool = True,
        allow_download: bool = True,
        layout: bool = False,
        tables: bool = False,
        formulas: bool = False,
        autorotate: bool = False,
        verbose: bool = False,
        keep_image: bool = True,
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
        self.requested_mode = mode
        self.fp16 = fp16
        self.verbose = verbose
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
        set_log_level_default(verbose)

        _native = load_native()
        store = ModelStore(models_dir, allow_download=allow_download)
        self.paths = store.resolve(self.entry, want_cls=use_cls)
        self._store = store

        # Provision the Apple NATIVE-mode bundle (MPSGraph exports + the ANE
        # packages) BEFORE construction — the engine probes for the export
        # dirs at load time, and this must stay outside construct_lock (it may
        # download once). Best-effort by contract: without the bundle,
        # backend="apple" runs its CoreML fallback exactly as before.
        if native.resolve_engine(backend) == "apple":
            store.ensure_apple_native(self.entry, self.paths)

        # Serialize env-mutation + construction: the engine reads its EP from
        # process env at construction, so two OCR(...) builds with different
        # backends must not interleave (env is global).
        with native.construct_lock:
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
            layout_path = store.ensure_asset("layout/layout.onnx") if need_layout else ""
            doc_ori_path = store.ensure_asset("doc_ori.onnx") if autorotate else ""
            if tables:
                os.environ["TABLE_SLANEXT_ENCODER_ONNX"] = store.ensure_asset(
                    "table/slanext_encoder/SLANeXt_wired_encoder.onnx"
                )
            if formulas:
                os.environ["FORMULA_ONNX"] = store.ensure_asset(
                    "formula/ppformulanet_s/inference_trt.onnx"
                )
                os.environ["FORMULA_TOKENIZER"] = store.ensure_asset(
                    "formula/ppformulanet_s/tokenizer.json"
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
        # LOADED, not requested. `need_layout` is only the constructor's
        # intent; the engine can come up without the layout stage (missing or
        # unreadable layout.onnx, or a backend that declined it) while init()
        # still succeeds, because layout is optional. Reporting intent made
        # info()/capabilities() claim a capability the engine does not have —
        # and let an explicit layout=True silently return zero regions instead
        # of the capability-unavailable rejection the shared gate now raises.
        # Intersected with the request for the same reason as the three below:
        # a stage the engine loaded but this object opted out of is not
        # available here (tables=True alone still loads layout internally).
        self.has_layout = need_layout and self._pipe.has_layout()
        self.has_tables = tables and self._pipe.has_table_backend()
        self.has_formulas = formulas and self._pipe.has_formula_backend()
        self.autorotate = autorotate and self._pipe.has_doc_ori()
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

    @contextlib.contextmanager
    def _checkout(self):
        """Borrow a free replica; blocks when all are in flight.

        The queue IS the mutual exclusion: a checked-out pipeline is owned by
        exactly one thread until returned, so no per-pipeline lock is needed.
        Concurrent read() calls from user threads spread across the pool
        automatically."""
        pipe = self._pool.get()
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
        ``autorotate=True`` corrects a rotated page first.

        ``text=False`` (a layout-only run, the library spelling of the HTTP
        ``?text=0``) goes through the SAME shared request-option gate the server
        runs — see ``include/turbo_ocr/service/validation/options_core.h`` — so
        an unsupported combination raises ``ValueError`` with the exact message
        the HTTP and gRPC surfaces return, rather than quietly coming back with
        a full OCR result labelled layout-only."""
        img = load_image(image)
        # The C++ engine applies its own hard floor (kDropScore = 0.5) BEFORE
        # Python sees any item, so a lower drop_score here can only pretend:
        # read(drop_score=0.1) returned output byte-identical to 0.5 with no
        # warning. Refuse until the floor is plumbed through RunFlags.
        if drop_score < DROP_SCORE:
            raise ValueError(
                f"drop_score={drop_score} is below the engine's hard floor "
                f"({DROP_SCORE}): the C++ pipeline filters at "
                f"{DROP_SCORE} before Python sees results, so lower values "
                "have no effect. Use drop_score >= 0.5."
            )
        angle = rotate % 360
        do_auto = self.autorotate if autorotate is None else autorotate
        # An explicit per-call autorotate=True must either WORK or RAISE — the
        # same refusal-beats-silent-no-op rule the layout/tables/formulas gate
        # below applies. This line used to AND with self.autorotate again,
        # which made the explicit override dead logic: on an instance built
        # with autorotate=False, read(autorotate=True) silently OCR'd the
        # sideways page.
        if do_auto and angle == 0:
            if not self._pipe.has_doc_ori():
                raise ValueError(
                    "autorotate requested but this pipeline has no "
                    "document-orientation model (construct OCR(..., "
                    "autorotate=True) so the model is loaded)"
                )
            with self._checkout() as pipe:
                angle = int(pipe.detect_orientation(np.ascontiguousarray(img, np.uint8)))
        if angle:
            img = rotate_bound(img, angle)
        return self._read_array(
            img, drop_score=drop_score, rotate=angle,
            want_layout=layout, want_reading_order=reading_order,
            want_tables=tables, want_formulas=formulas, want_text=text,
            keep_image=keep_image,
        )

    def _read_array(
        self,
        img: np.ndarray,
        *,
        drop_score: float,  # validated below: cannot go under the engine floor
        rotate: int = 0,
        page: Optional[int] = None,
        want_layout: Optional[bool] = None,
        want_reading_order: bool = False,
        want_tables: Optional[bool] = None,
        want_formulas: Optional[bool] = None,
        want_text: bool = True,
        keep_image: Optional[bool] = None,
    ) -> PageResult:
        h, w = img.shape[:2]
        page_res = PageResult(width=w, height=h, page=page, orientation=rotate % 360)

        img = np.ascontiguousarray(img, dtype=np.uint8)
        if self.keep_image if keep_image is None else keep_image:
            page_res.image = img

        def _resolve(req, have, name):
            # An EXPLICIT request is never clamped to the built capability set:
            # it flows through run_with_layout to the shared request-option gate
            # (validation/python_options.h over the live capability mask), which
            # raises with the same message HTTP returns for the same request.
            # This used to warn-and-ignore — a silent wrong answer (the caller
            # asked for tables and got a result without them). Only an implicit
            # request (req is None) defaults to what was built.
            del name  # the gate names the capability in its own message
            if req is None:
                return have
            return bool(req)

        use_layout = _resolve(want_layout, self.has_layout, "layout")
        use_tables = _resolve(want_tables, self.has_tables, "tables")
        use_formulas = _resolve(want_formulas, self.has_formulas, "formulas")
        use_structure = use_layout or use_tables or use_formulas

        with self._checkout() as pipe:  # one replica per in-flight run (GIL released in C++)
            # `not want_text` takes this branch even with no structure requested:
            # run_with_layout is where the shared request-option gate runs AND
            # where RunFlags.text=false is honoured (layout-only run, or the
            # gate's rejection for a bare text=False with no layout). Routing it
            # to the plain run() below instead would have returned a full
            # det+rec result for a request that asked for no text at all — the
            # silent-wrong-answer the gate exists to prevent.
            if use_structure or not want_text:
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
                    layout=use_structure,
                    reading_order=want_reading_order,
                    tables=use_tables,
                    formulas=use_formulas,
                    text=want_text,
                )
                items = r.items
                if use_layout:
                    for lb in r.layout:
                        page_res.layout.append(
                            LayoutBox(
                                label=lb.label, confidence=float(lb.score),
                                box=tuple((int(p[0]), int(p[1])) for p in lb.box),  # type: ignore
                                id=lb.id,
                            )
                        )
                    if r.reading_order:
                        page_res.reading_order = list(r.reading_order)
                if use_tables:
                    for t in r.tables:
                        page_res.tables.append(
                            TableRegion(
                                html=t.content, score=float(t.score),
                                box=tuple((int(p[0]), int(p[1])) for p in t.box),  # type: ignore
                                layout_id=t.layout_id,
                            )
                        )
                    # The FLAG is authoritative; the warning string is optional
                    # detail. Requiring both meant a producer that set the flag
                    # with no message yielded warnings == [] — a clean-looking
                    # degraded page, which is what the mechanism exists to
                    # prevent. Every C++ emitter tests the flag first.
                    if r.table_degraded:
                        page_res.warnings.append(
                            f"table_degraded: {r.table_warning or 'no detail'}")
                if use_formulas:
                    for f in r.formulas:
                        page_res.formulas.append(
                            FormulaRegion(
                                latex=f.content, score=float(f.score),
                                box=tuple((int(p[0]), int(p[1])) for p in f.box),  # type: ignore
                                layout_id=f.layout_id,
                            )
                        )
                    if r.formula_degraded:
                        page_res.warnings.append(
                            f"formula_degraded: {r.formula_warning or 'no detail'}")
                if r.text_degraded:
                    page_res.warnings.append(
                        f"text_degraded: {r.text_warning or 'no detail'}")
            else:
                items = pipe.run(img)  # native C++ det->sort->(cls)->rec

        _fill_lines(page_res, items, drop_score)
        return page_res

    # -- batch of images ---------------------------------------------------
    def read_batch(
        self,
        images: List[ImageInput],
        *,
        drop_score: float = DROP_SCORE,
        layout: Optional[bool] = None,
        autorotate: Optional[bool] = None,
        progress=None,
        keep_image: Optional[bool] = None,
        batch_size: int = 8,
    ) -> DocumentResult:
        """OCR a list of images into a :class:`DocumentResult` (one page each).

        Images go through the native whole-batch submission in groups of
        ``batch_size`` (matching the server's ``/ocr/batch`` chunking), which
        lets the detector see a real batch instead of a batch of one. Requests
        that need per-image stages — layout, tables, formulas, autorotate —
        fall back to reading one at a time; see :meth:`_can_batch`.

        ``progress`` may be True (log to stderr) or a callable
        ``progress(done, total)``. ``keep_image=False`` drops each page raster
        after reading (see :meth:`read_pdf`). Returns a DocumentResult —
        iterate it for the per-image pages, or use ``doc.to_tsv()`` /
        ``doc.text`` etc."""
        total = len(images)
        report = _make_progress(progress, total, "images")
        doc = DocumentResult()

        if self._can_batch(layout=layout, autorotate=autorotate):
            keep = self.keep_image if keep_image is None else keep_image
            chunks = list(_chunks(images, batch_size))

            def _run_chunk(chunk) -> List[PageResult]:
                arrays = [
                    np.ascontiguousarray(load_image(im), dtype=np.uint8)
                    for im in chunk
                ]
                with self._checkout() as pipe:
                    batch = pipe.run_batch(arrays)
                # The native call returns exactly one result list per input, in
                # input order. Assert it rather than zip(): a short list would
                # silently DROP pages, turning a backend bug into missing text.
                if len(batch) != len(arrays):
                    raise RuntimeError(
                        f"native run_batch returned {len(batch)} results for "
                        f"{len(arrays)} images — refusing to drop pages"
                    )
                pages = []
                for arr, items in zip(arrays, batch):
                    h, w = arr.shape[:2]
                    pr = PageResult(width=w, height=h)
                    if keep:
                        pr.image = arr
                    pages.append(_fill_lines(pr, items, drop_score))
                return pages

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
            with ThreadPoolExecutor(max_workers=self.replicas) as ex:
                futs = {ex.submit(_run_chunk, c): i for i, c in enumerate(chunks)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    per_chunk[i] = fut.result()  # re-raises worker failures
                    _tick(len(chunks[i]))
            for pages in per_chunk:
                assert pages is not None  # every future resolved or raised
                doc.pages.extend(pages)
            return doc

        if self.replicas > 1 and total > 1:
            # Per-image stages (layout/autorotate/...) cannot use the native
            # whole-batch call, but they still spread across replicas: read()
            # checks a free one out per call.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            slots: List[Optional[PageResult]] = [None] * total
            done = 0
            with ThreadPoolExecutor(max_workers=self.replicas) as ex:
                futs = {
                    ex.submit(self.read, im, drop_score=drop_score, layout=layout,
                              autorotate=autorotate, keep_image=keep_image): i
                    for i, im in enumerate(images)
                }
                for fut in as_completed(futs):
                    slots[futs[fut]] = fut.result()
                    done += 1
                    report(done)
            for pr in slots:
                assert pr is not None
                doc.pages.append(pr)
            return doc

        for i, im in enumerate(images, 1):
            doc.pages.append(
                self.read(im, drop_score=drop_score, layout=layout,
                          autorotate=autorotate, keep_image=keep_image)
            )
            report(i)
        return doc

    def _can_batch(self, *, layout: Optional[bool], autorotate: Optional[bool]) -> bool:
        """True when a batch can go through the native whole-batch submission.

        Only the plain det->rec path is batched: layout / tables / formulas run
        through ``run_with_layout``, and autorotate needs a per-image
        orientation pass first, so those keep the one-image-at-a-time loop."""
        if not hasattr(self._pipe, "run_batch"):
            return False  # extension predates the batch binding
        if self.autorotate if autorotate is None else autorotate:
            return False
        want_layout = self.has_layout if layout is None else layout
        return not (want_layout or self.has_tables or self.has_formulas)

    # -- PDF ---------------------------------------------------------------
    def read_pdf(
        self,
        pdf: ImageInput,
        *,
        dpi: int = 150,
        pages: Optional[List[int]] = None,
        drop_score: float = DROP_SCORE,
        max_pages: Optional[int] = None,
        progress=None,
        keep_image: Optional[bool] = None,
    ) -> DocumentResult:
        """Render a PDF with PDFium and OCR each page. Requires the ``pdf``
        extra (``pip install "turboocr[cpu,pdf]"``). ``progress`` may be True (log to
        stderr) or a callable ``progress(done, total)``.

        ``keep_image=False`` drops each page raster once the page is read.
        The default keeps them (so ``save_searchable_pdf`` / ``draw`` work),
        but a raster is ~6 MB at 150 DPI, so a few hundred pages is GBs of
        retained memory — pass False when you only want the text."""
        from .pdf import pdf_page_count, render_pdf

        doc = DocumentResult(
            source=str(pdf) if isinstance(pdf, (str, os.PathLike)) else ""
        )
        if pages is not None:
            total = min(len(pages), max_pages) if max_pages else len(pages)
        else:
            total = pdf_page_count(pdf) if progress else 0
            if max_pages:
                total = min(total, max_pages)
        report = _make_progress(progress, total, "pages")
        for i, (page_no, arr) in enumerate(
            render_pdf(pdf, dpi=dpi, pages=pages, max_pages=max_pages), 1
        ):
            pr = self._read_array(
                arr, drop_score=drop_score, page=page_no, keep_image=keep_image
            )
            pr.dpi = dpi
            doc.pages.append(pr)
            report(i)
        return doc

    def pdf_to_searchable(
        self,
        input_pdf: ImageInput,
        output_pdf: str,
        *,
        dpi: int = 150,
        pages: Optional[List[int]] = None,
        max_pages: Optional[int] = None,
        drop_score: float = DROP_SCORE,
        progress=None,
    ) -> str:
        """Stream a PDF through OCR into a SEARCHABLE PDF (page image + invisible
        text layer), writing ``output_pdf``. Streams page-by-page — only one
        page raster is held at a time — so it scales to large scans. Requires
        the ``pdf`` extra (pypdfium2 + reportlab)."""
        from .pdf import pdf_page_count, render_pdf
        from .searchable_pdf import build_searchable_pdf

        if pages is not None:
            total = min(len(pages), max_pages) if max_pages else len(pages)
        else:
            total = pdf_page_count(input_pdf) if progress else 0
            if max_pages:
                total = min(total, max_pages)
        report = _make_progress(progress, total, "pages")

        def _page_stream():
            for i, (page_no, arr) in enumerate(
                render_pdf(input_pdf, dpi=dpi, pages=pages, max_pages=max_pages), 1
            ):
                pr = self._read_array(arr, drop_score=drop_score, page=page_no)
                pr.dpi = dpi
                yield pr
                report(i)  # raster is dropped as soon as this page is written

        build_searchable_pdf(_page_stream(), out_path=output_pdf)
        return output_pdf

    # -- introspection -----------------------------------------------------
    def detect_orientation(self, image: ImageInput) -> int:
        """Detected page rotation (0/90/180/270), if a doc-orientation model is
        loaded; else 0."""
        img = np.ascontiguousarray(load_image(image), dtype=np.uint8)
        with self._checkout() as pipe:
            return int(pipe.detect_orientation(img))

    def close(self) -> None:
        """Release the native ONNX sessions. The engine is unusable afterward.
        Optional — fine to rely on GC for the common one-engine-per-process
        case; use this (or the context manager) when churning many engines.
        Waits for in-flight reads: every replica is reclaimed from the pool
        before any is dropped, so a concurrent read() finishes rather than
        racing a teardown."""
        for _ in range(len(self._pipes)):
            self._pool.get()
        self._pipes = []
        self._pipe = None

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
        }
