"""Model catalog — a Python port of ``server/model_catalog.h`` and
``detection/det_config.h``.

Keeps the Python bindings in lockstep with the C++ engine: same tiers, same
per-model detector, same official PaddleOCR detection inference config
(resize policy + DB post-processing thresholds). Adding a model is one row,
exactly as in the C++ registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DetResizeParams:
    """PaddleOCR ``DetResizeForTest`` (resize_image_type0) policy.

    ``limit_type`` "min" grows the shorter side up to ``limit_side_len``;
    "max" shrinks the longer side down to it. Both clamp the output so the
    longer side never exceeds ``max_side_limit``, then round each side to a
    multiple of 32 (the detector requires /32 input dims).
    """

    limit_type: str = "min"
    limit_side_len: int = 64
    max_side_limit: int = 1280


@dataclass(frozen=True)
class DbParams:
    """DBNet post-processing thresholds."""

    thresh: float = 0.2  # probability-map binarization threshold
    box_thresh: float = 0.45  # per-box mean-score cutoff
    unclip_ratio: float = 1.4  # polygon expansion ratio


@dataclass(frozen=True)
class DetConfig:
    resize: DetResizeParams = field(default_factory=DetResizeParams)
    db: DbParams = field(default_factory=DbParams)


# Official PaddleOCR det config for the PP-OCRv6 tiers (medium, small) and the
# retained V5Lang recognizers that reuse the shared v6 detector.
V6_DET = DetConfig(DetResizeParams("min", 64, 1280), DbParams(0.2, 0.45, 1.4))
# The tiny tier differs only in box_thresh (0.40 per its inference.yml).
V6_DET_TINY = DetConfig(DetResizeParams("min", 64, 1280), DbParams(0.2, 0.40, 1.4))


@dataclass(frozen=True)
class ModelEntry:
    """One selectable OCR model.

    Paths are relative to a models root. ``det`` empty => the shared default
    detector (``det.onnx``). ``dict`` is the recognizer's character dictionary.
    """

    name: str
    rec: str
    dict: str
    det: str  # "" => DEFAULT_DET
    family: str = "v6"  # "v6" | "v5lang"
    det_cfg: DetConfig = field(default_factory=lambda: V6_DET)

    def det_path(self) -> str:
        return self.det or DEFAULT_DET


DEFAULT_DET = "det.onnx"
DEFAULT_MODEL = "tiny"

# The registry. v6 tiers share keys.txt except tiny (its own 6,904-char dict).
# Legacy scripts keep their v5 recognizer + dict and the shared v6 detector.
_CATALOG: Tuple[ModelEntry, ...] = (
    ModelEntry("medium", "rec.onnx", "keys.txt", "det.onnx", "v6", V6_DET),
    ModelEntry("small", "rec_small.onnx", "keys.txt", "det_small.onnx", "v6", V6_DET),
    ModelEntry("tiny", "rec_tiny.onnx", "keys_tiny.txt", "det_tiny.onnx", "v6", V6_DET_TINY),
    # Full-size detector + tiny recognizer (mirrors model_catalog.h — see the
    # long rationale there). det_cfg is V6_DET (0.45): the DB parameters belong
    # to the detector that RUNS, and det.onnx is the medium detector. The dict
    # MUST stay keys_tiny.txt (rec_tiny has 6,904 classes).
    ModelEntry("tiny-bigdet", "rec_tiny.onnx", "keys_tiny.txt", "det.onnx", "v6", V6_DET),
    ModelEntry("arabic", "rec/arabic/rec.onnx", "rec/arabic/dict.txt", "", "v5lang", V6_DET),
    ModelEntry("eslav", "rec/eslav/rec.onnx", "rec/eslav/dict.txt", "", "v5lang", V6_DET),
    ModelEntry("korean", "rec/korean/rec.onnx", "rec/korean/dict.txt", "", "v5lang", V6_DET),
    ModelEntry("thai", "rec/thai/rec.onnx", "rec/thai/dict.txt", "", "v5lang", V6_DET),
    ModelEntry("greek", "rec/greek/rec.onnx", "rec/greek/dict.txt", "", "v5lang", V6_DET),
)

_BY_NAME: Dict[str, ModelEntry] = {e.name: e for e in _CATALOG}

# Friendly aliases so callers can say language names or "default"/"fast"/"best".
_ALIASES: Dict[str, str] = {
    "default": DEFAULT_MODEL,
    "fast": "tiny",
    "fastest": "tiny",
    "base": "medium",
    "best": "medium",
    "accurate": "medium",
    "latin": DEFAULT_MODEL,
    "en": DEFAULT_MODEL,
    "english": DEFAULT_MODEL,
    "ch": DEFAULT_MODEL,
    "chinese": DEFAULT_MODEL,
    "ja": DEFAULT_MODEL,
    "japanese": DEFAULT_MODEL,
    "ar": "arabic",
    "ko": "korean",
    "th": "thai",
    "el": "greek",
    "ru": "eslav",
    "cyrillic": "eslav",
}


def find_model(name: str) -> Optional[ModelEntry]:
    """Resolve a model by name or alias. Returns None if unknown."""
    key = (name or "").strip().lower()
    key = _ALIASES.get(key, key)
    return _BY_NAME.get(key)


def resolve_model(name: str) -> ModelEntry:
    """Resolve a model by name/alias, raising a clear error on a miss."""
    e = find_model(name)
    if e is None:
        raise ValueError(
            f"unknown model '{name}'. Available: {', '.join(list_models())}"
        )
    return e


def list_models() -> List[str]:
    return [e.name for e in _CATALOG]


def catalog() -> Tuple[ModelEntry, ...]:
    return _CATALOG
