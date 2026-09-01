"""TurboOCR — fast, multi-backend OCR for Python.

Quick start
-----------
>>> import turboocr
>>> ocr = turboocr.OCR()             # tiny model, fast-setup backend (auto)
>>> page = ocr.read("document.png")
>>> print(page.text)

Pick a backend explicitly, or see what to install:

>>> turboocr.doctor()                # install panel for your GPU
>>> ocr = turboocr.OCR("medium", backend="cuda")   # NVIDIA, no engine build
>>> ocr = turboocr.OCR("medium", backend="tensorrt")  # NVIDIA TensorRT (opt-in)

PDF — built in, no extra needed; pages fan out across the replica pool:

>>> doc = ocr.read_pdf("paper.pdf", dpi=150)
>>> print(doc.to_markdown())
>>> for page in ocr.read_pdf_stream("paper.pdf"):  # stream pages as ready
...     print(page.page, page.text[:40])

Async twins exist for every read — ``aread`` / ``aread_batch`` /
``aread_pdf`` / ``aread_pdf_stream`` — real concurrency up to
``OCR(replicas=N)``.

Backends: ``auto`` (the wheel's best default — on the NVIDIA wheels it
resolves to ``turbo`` and the first run builds a cached TensorRT engine;
elsewhere the CPU path), ``turbo`` (TensorRT on NVIDIA), ``apple`` (native
Metal/MPSGraph — the fast path on Apple silicon), ``cpu``, and explicit EPs
``cuda`` / ``rocm`` / ``openvino`` / ``directml`` / ``coreml``.
"""

from __future__ import annotations

import threading as _threading
from typing import Optional
from collections import OrderedDict

from ._version import __version__
from .catalog import DEFAULT_MODEL, list_models, resolve_model
from .catalog import catalog as model_catalog
from .doctor import available_backends, build_report, doctor, recommend
from .errors import (
    BackendUnavailable,
    ModelLoadError,
    NativeExtensionMissing,
    TurboOCRError,
)
from .pipeline import OCR
from .providers import HardwareInfo, detect_hardware
from .result import (
    DocumentResult,
    FormulaRegion,
    LayoutBox,
    PageResult,
    TableRegion,
    TextLine,
)

# The essentials most users need. Plumbing (recommend, build_report,
# detect_hardware, resolve_providers, HardwareInfo, onnxruntime_available, ...)
# stays importable from turboocr.providers / turboocr.doctor but is kept out of
# the top-level `*` surface to keep discoverability sharp.
__all__ = [
    "DEFAULT_MODEL",
    "OCR",
    "BackendUnavailable",
    "DocumentResult",
    "FormulaRegion",
    "LayoutBox",
    "ModelLoadError",
    "NativeExtensionMissing",
    "PageResult",
    "TableRegion",
    "TextLine",
    "TurboOCRError",
    "__version__",
    "available_backends",
    "doctor",
    "list_models",
    "model_catalog",
    "read",
    "read_pdf",
    "resolve_model",
]

# Lazily-constructed process-wide default engines for the module-level
# helpers: a small keyed cache behind a lock. The old single (engine, key)
# slot rebuilt on EVERY call when two keys alternated (read(layout=True) /
# read() ping-pong = a full model load each time), and two threads racing the
# slot could each construct — one engine leaked unclosed.
_DEFAULT_LOCK = _threading.Lock()
_DEFAULT_CACHE: "OrderedDict" = OrderedDict()  # key -> OCR, LRU, small cap
_DEFAULT_CACHE_CAP = 4
_DEFAULT_KEY_LOCKS: dict = {}  # key -> Lock serializing THAT key's build


def _default_engine(
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    *,
    lang: Optional[str] = None,
    tier: Optional[str] = None,
    layout: bool = False,
    tables: bool = False,
    formulas: bool = False,
    autorotate: bool = False,
) -> "OCR":
    # Cache keyed on EVERY capability that changes what the engine can do —
    # read(tables=True) must build a table-capable engine, not hit a cached
    # incapable one (the shared gate would then reject the request loudly,
    # but the feature would be unreachable from the one-shot API).
    # lang/tier are part of the KEY: they select a different recognizer, so
    # two calls that differ only in tier must not share one cached engine.
    key = (resolve_model(model).name, backend, lang, tier, layout, tables,
           formulas, autorotate)
    with _DEFAULT_LOCK:
        eng = _DEFAULT_CACHE.get(key)
        if eng is not None:
            _DEFAULT_CACHE.move_to_end(key)
            return eng
        klock = _DEFAULT_KEY_LOCKS.setdefault(key, _threading.Lock())
    # Construction happens under a PER-KEY lock, not the cache lock: a
    # same-key race still builds exactly once (the loser finds it cached on
    # re-check), while a cache HIT for another key no longer waits out a
    # minutes-long cold build (model download, bundle provisioning).
    with klock:
        with _DEFAULT_LOCK:
            eng = _DEFAULT_CACHE.get(key)
            if eng is not None:
                _DEFAULT_CACHE.move_to_end(key)
                return eng
        try:
            eng = OCR(model, backend, lang=lang, tier=tier, layout=layout,
                      tables=tables, formulas=formulas, autorotate=autorotate)
        except Exception:
            # A failed build must not leave its key lock behind forever —
            # with backend/model strings from user input that dict would
            # grow unboundedly. (A concurrent same-key caller may briefly
            # mint a fresh lock and retry the build; that is the correct
            # behavior after a failure anyway.)
            with _DEFAULT_LOCK:
                _DEFAULT_KEY_LOCKS.pop(key, None)
            raise
        with _DEFAULT_LOCK:
            _DEFAULT_CACHE[key] = eng
            while len(_DEFAULT_CACHE) > _DEFAULT_CACHE_CAP:
                # Dropped, NOT closed: a caller may still hold a reference
                # from an earlier call — GC reclaims the sessions once they
                # let go. The key lock stays in _DEFAULT_KEY_LOCKS (a handful
                # of tiny objects; correctness over churn).
                _DEFAULT_CACHE.popitem(last=False)
        return eng


def _one_shot(model, backend, lang, tier, layout, tables, formulas,
              reading_order, autorotate):
    """Shared half of the one-shot wrappers: the cached engine AND the per-call
    stage kwargs, together — so a wrapper cannot build a capable engine and
    then forget to ask (read_pdf shipped exactly that bug: an engine with the
    table model resident and a call that never requested tables).

    Load is not Run: capability flags must do BOTH jobs. They size the cached
    engine (reading_order rides the layout model, so it must LOAD layout even
    though the shared gate auto-enables the layout STAGE per call), and they
    travel with the call as explicit requests. autorotate alone stays an
    engine property — input preparation, inherited by every read path."""
    eng = _default_engine(model, backend, lang=lang, tier=tier,
                          layout=layout or reading_order,
                          tables=tables, formulas=formulas,
                          autorotate=autorotate)
    return eng, {"layout": layout or None, "tables": tables or None,
                 "formulas": formulas or None, "reading_order": reading_order}


def read(
    image,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    lang: Optional[str] = None,
    tier: Optional[str] = None,
    layout: bool = False,
    tables: bool = False,
    formulas: bool = False,
    reading_order: bool = False,
    autorotate: bool = False,
    **kwargs,
):
    """One-shot convenience: OCR a single image with a cached default engine.

    ``layout``/``tables``/``formulas``/``autorotate`` build (and cache) an
    engine with those capabilities — they are NOT silently ignored, and
    ``lang``/``tier`` select the recognizer the same way they do on
    :class:`OCR` (they used to land in ``**kwargs`` and reach ``OCR.read()``,
    which raised TypeError). For repeated calls, construct an :class:`OCR`
    once and reuse it.
    """
    eng, stage_kwargs = _one_shot(model, backend, lang, tier, layout, tables,
                                  formulas, reading_order, autorotate)
    return eng.read(image, autorotate=autorotate or None,
                    **stage_kwargs, **kwargs)


def read_pdf(
    pdf,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    lang: Optional[str] = None,
    tier: Optional[str] = None,
    layout: bool = False,
    tables: bool = False,
    formulas: bool = False,
    reading_order: bool = False,
    autorotate: bool = False,
    **kwargs,
):
    """One-shot convenience: OCR a PDF with a cached default engine (the
    capability flags work as in :func:`read`)."""
    eng, stage_kwargs = _one_shot(model, backend, lang, tier, layout, tables,
                                  formulas, reading_order, autorotate)
    return eng.read_pdf(pdf, **stage_kwargs, **kwargs)
