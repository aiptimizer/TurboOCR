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
>>> ocr = turboocr.OCR("medium", backend="turbo")  # NVIDIA TensorRT (opt-in)

PDF (needs `pip install "turboocr[cpu,pdf]"`):

>>> doc = ocr.read_pdf("paper.pdf", dpi=150)
>>> print(doc.to_markdown())

Backends: ``auto`` (default, fast-setup — no engine build), ``turbo``
(TensorRT on NVIDIA), ``cpu``, and explicit EPs ``cuda`` / ``rocm`` /
``openvino`` / ``directml`` / ``coreml``.
"""

from __future__ import annotations

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
    "__version__",
    "OCR",
    "read",
    "read_pdf",
    "doctor",
    "available_backends",
    "PageResult",
    "DocumentResult",
    "TextLine",
    "LayoutBox",
    "TableRegion",
    "FormulaRegion",
    "list_models",
    "model_catalog",
    "resolve_model",
    "DEFAULT_MODEL",
    "TurboOCRError",
    "ModelLoadError",
    "BackendUnavailable",
    "NativeExtensionMissing",
]

# Lazily-constructed process-wide default engine for the module-level helpers.
_DEFAULT_OCR = None
_DEFAULT_KEY = None


def _default_engine(
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    *,
    layout: bool = False,
    autorotate: bool = False,
) -> "OCR":
    # Cache keyed on capabilities too, so `read(layout=True)` actually builds a
    # layout-capable engine instead of silently no-opping against a cached one.
    global _DEFAULT_OCR, _DEFAULT_KEY
    key = (resolve_model(model).name, backend, layout, autorotate)
    if _DEFAULT_OCR is None or _DEFAULT_KEY != key:
        _DEFAULT_OCR = OCR(model, backend, layout=layout, autorotate=autorotate)
        _DEFAULT_KEY = key
    return _DEFAULT_OCR


def read(
    image,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    layout: bool = False,
    autorotate: bool = False,
    **kwargs,
):
    """One-shot convenience: OCR a single image with a cached default engine.

    ``layout``/``autorotate`` build (and cache) an engine with those
    capabilities — they are NOT silently ignored. For repeated calls, construct
    an :class:`OCR` once and reuse it.
    """
    eng = _default_engine(model, backend, layout=layout, autorotate=autorotate)
    return eng.read(image, layout=layout or None, autorotate=autorotate or None, **kwargs)


def read_pdf(
    pdf,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    layout: bool = False,
    autorotate: bool = False,
    **kwargs,
):
    """One-shot convenience: OCR a PDF with a cached default engine."""
    eng = _default_engine(model, backend, layout=layout, autorotate=autorotate)
    return eng.read_pdf(pdf, **kwargs)
