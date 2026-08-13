"""Typed exceptions so callers can branch on failure mode.

All inherit ``RuntimeError`` so existing ``except RuntimeError`` handlers keep
working — this only adds structure on top.
"""

from __future__ import annotations


class TurboOCRError(RuntimeError):
    """Base class for all TurboOCR errors."""


class NativeExtensionMissing(TurboOCRError):
    """The compiled ``_turboocr`` extension isn't available for this env."""


class ModelLoadError(TurboOCRError):
    """A model (detector/recognizer/dict/layout) failed to load."""


class BackendUnavailable(TurboOCRError):
    """The requested backend's execution provider isn't in this build."""
