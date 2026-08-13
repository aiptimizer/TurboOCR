"""Image loading + small geometry helpers.

Everything downstream (detection, recognition) works on a BGR uint8 HxWx3
numpy array — the same layout OpenCV uses — so all the different input shapes a
caller might pass (path, bytes, PIL image, numpy array) funnel through
:func:`load_image` into that one canonical form.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cv2

# cv2 is imported lazily (see _cv2()): it costs ~35 ms and pulls a large native
# library, which every `import turboocr` used to pay even when the caller never
# decoded an image. result.py already lazy-imports it the same way.
_CV2: Any = None


def _cv2() -> Any:
    global _CV2
    if _CV2 is None:
        import cv2 as _mod

        _CV2 = _mod
    return _CV2


ImageInput = Union[str, "os.PathLike[str]", bytes, bytearray, memoryview, np.ndarray]


def _from_numpy(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:  # grayscale -> BGR
        return _cv2().cvtColor(arr.astype(np.uint8, copy=False), _cv2().COLOR_GRAY2BGR)
    if arr.ndim == 3:
        if arr.shape[2] == 4:  # assume BGRA
            return _cv2().cvtColor(arr, _cv2().COLOR_BGRA2BGR)
        if arr.shape[2] == 3:
            return np.ascontiguousarray(arr)
    raise ValueError(f"unsupported image array shape {arr.shape}")


def load_image(src: ImageInput) -> np.ndarray:
    """Decode any supported input into a BGR uint8 HxWx3 array.

    Accepts a filesystem path, raw encoded bytes (PNG/JPEG/...), a PIL image,
    or a numpy array (grayscale, BGR, or BGRA). numpy arrays are assumed to be
    in OpenCV's channel order already.
    """
    # PIL image (duck-typed to avoid a hard Pillow dependency here).
    if hasattr(src, "mode") and hasattr(src, "size") and hasattr(src, "convert"):
        rgb = np.asarray(src.convert("RGB"))
        return _cv2().cvtColor(rgb, _cv2().COLOR_RGB2BGR)

    if isinstance(src, np.ndarray):
        if src.dtype != np.uint8:
            # A float image in the conventional [0,1] range (skimage
            # img_as_float, matplotlib.imread, a detached torch tensor,
            # imread(...)/255) must be scaled, not clipped: the bare
            # clip-to-[0,255] cast turned such an image into all-zeros with a
            # few ones, detection found nothing, and read() returned zero
            # lines with no error.
            if np.issubdtype(src.dtype, np.floating) and src.size and float(src.max()) <= 1.0:
                src = (src * 255.0).round()
            src = np.clip(src, 0, 255).astype(np.uint8)
        return _from_numpy(src)

    if isinstance(src, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(bytes(src), dtype=np.uint8)
        img = _cv2().imdecode(buf, _cv2().IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image bytes")
        return img

    # Path-like.
    path = os.fspath(src)
    if not os.path.exists(path):
        raise FileNotFoundError(f"image not found: {path}")
    # imread mangles non-ASCII paths on some platforms; decode via a byte read.
    with open(path, "rb") as fh:
        buf = np.frombuffer(fh.read(), dtype=np.uint8)
    img = _cv2().imdecode(buf, _cv2().IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image file: {path}")
    return img


def rotate_bound(img: np.ndarray, angle_cw: int) -> np.ndarray:
    """Rotate clockwise by 0/90/180/270 degrees with no cropping."""
    a = angle_cw % 360
    if a == 0:
        return img
    if a == 90:
        return _cv2().rotate(img, _cv2().ROTATE_90_CLOCKWISE)
    if a == 180:
        return _cv2().rotate(img, _cv2().ROTATE_180)
    if a == 270:
        return _cv2().rotate(img, _cv2().ROTATE_90_COUNTERCLOCKWISE)
    # arbitrary angle fallback
    h, w = img.shape[:2]
    m = _cv2().getRotationMatrix2D((w / 2, h / 2), -angle_cw, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    return _cv2().warpAffine(img, m, (nw, nh), borderValue=(255, 255, 255))
