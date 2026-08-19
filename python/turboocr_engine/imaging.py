"""Image loading + small geometry helpers.

Everything downstream (detection, recognition) works on a BGR uint8 HxWx3
numpy array — the same layout OpenCV uses — so all the different input shapes a
caller might pass (path, bytes, PIL image, numpy array) funnel through
:func:`load_image` into that one canonical form.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

import numpy as np

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

# The PDF spec tolerates junk before the %PDF- header as long as it appears
# early, so sniff a window rather than trusting the file extension — a PDF
# misnamed .png gets the same actionable error as doc.pdf. Shared with the
# reverse check in pdf.py (read_pdf on a non-PDF), so the two readers can
# never disagree about what counts as a PDF.
_PDF_SNIFF_WINDOW = 1024


def looks_like_pdf(head: bytes) -> bool:
    """Is this the start of a PDF document? ``head`` = the first bytes."""
    return b"%PDF-" in head[:_PDF_SNIFF_WINDOW]


def _tiff_page_count(data: bytes) -> int:
    """Number of IFDs (pages) in a classic TIFF, by walking the IFD chain.

    Returns 0 for anything that is not a well-formed classic TIFF (BigTIFF's
    magic 43 included) — those fall through to the normal decoder, which
    accepts or rejects them on its own. A truncated chain stops counting
    rather than raising: the decoder owns the corruption error."""
    import struct

    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return 0
    endian = "<" if data[:2] == b"II" else ">"

    def u16(off: int) -> int:
        return struct.unpack_from(endian + "H", data, off)[0]

    def u32(off: int) -> int:
        return struct.unpack_from(endian + "I", data, off)[0]

    if u16(2) != 42:
        return 0
    seen = set()
    off = u32(4)
    count = 0
    while off and off not in seen and count < 100000:
        if off + 2 > len(data):
            break
        seen.add(off)
        next_ptr = off + 2 + 12 * u16(off)
        if next_ptr + 4 > len(data):
            break
        count += 1
        off = u32(next_ptr)
    return count


def _reject_multipage_tiff(data: bytes, what: str) -> None:
    """cv2.imdecode on a multi-page TIFF silently decodes page 1 and DROPS
    the rest — silent data loss, the worst failure mode. Refuse with the page
    count and the way out instead."""
    n = _tiff_page_count(data)
    if n > 1:
        raise ValueError(
            f"{what} is a multi-page TIFF ({n} pages) — read() decodes ONE "
            "image, and decoding this would silently drop every page but the "
            "first. Split the pages (cv2.imreadmulti or tifffile) and use "
            "read_batch() on them."
        )


_DEFAULT_MAX_MP = 96.0  # ~a 12000x8000 scan; far above any sane OCR page


def _max_image_mp() -> float:
    """The decode ceiling in megapixels (TURBO_MAX_IMAGE_MP; 0 disables)."""
    v = os.environ.get("TURBO_MAX_IMAGE_MP", "").strip()
    if not v:
        return _DEFAULT_MAX_MP
    try:
        f = float(v)
    except ValueError:
        return _DEFAULT_MAX_MP
    if f != f or f < 0:  # NaN / negative: a typo must not disable the guard
        return _DEFAULT_MAX_MP
    return f if f > 0 else float("inf")  # 0 explicitly disables


def _sniff_dims(data: bytes) -> "Optional[Tuple[int, int]]":
    """(width, height) straight from a PNG IHDR / JPEG SOFn header, without
    decoding; None for other formats (those get the post-decode check)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 and data[12:16] == b"IHDR":
        return (int.from_bytes(data[16:20], "big"),
                int.from_bytes(data[20:24], "big"))
    if data[:2] == b"BM" and len(data) >= 26:
        # The DIB header SIZE at offset 14 keys the layout: 12 is the OS/2
        # BITMAPCOREHEADER with UNSIGNED 16-bit dims at 18/20; the 40+ family
        # (INFOHEADER/V4/V5) has signed 32-bit dims at 18/22 (negative height
        # = top-down orientation). Reading 32-bit dims out of a core header
        # rejected valid decodable BMPs with a nonsense gigapixel message,
        # and any "BM"-prefixed text file got the same treatment — unknown
        # header sizes now fall through to the post-decode check instead.
        dib = int.from_bytes(data[14:18], "little")
        if dib == 12:
            return (int.from_bytes(data[18:20], "little"),
                    int.from_bytes(data[20:22], "little"))
        if dib in (40, 52, 56, 64, 108, 124):
            w = int.from_bytes(data[18:22], "little", signed=True)
            h = int.from_bytes(data[22:26], "little", signed=True)
            return (abs(w), abs(h))
        return None
    if data[:2] == b"\xff\xd8":
        i, n = 2, len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker == 0xFF:
                i += 1
                continue
            if marker == 0x00:
                # FF 00 is a STUFFED data byte, not a marker. Treating it as
                # an unknown segment desynchronized the walk (skipping garbage
                # "lengths"), returned None, and silently bypassed the
                # pre-decode ceiling for a file libjpeg still decodes.
                i += 2
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2  # standalone markers carry no length
                continue
            if marker in (0xD9, 0xDA):
                break  # EOI / entropy-coded scan: no SOF before pixels
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h = int.from_bytes(data[i + 5:i + 7], "big")
                w = int.from_bytes(data[i + 7:i + 9], "big")
                return (w, h)
            i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def _check_pixel_ceiling(w: int, h: int, what: str) -> None:
    limit = _max_image_mp()
    mp = (int(w) * int(h)) / 1e6
    if mp > limit:
        raise ValueError(
            f"{what}: {w}x{h} px is {mp:.0f} megapixels — over the safety "
            f"ceiling of {limit:.0f} MP. Decoding costs ~{w * h * 3 / 1e9:.1f} GB "
            "before OCR even starts, which usually means a corrupt or hostile "
            "header rather than a real page. Set TURBO_MAX_IMAGE_MP to raise "
            "the ceiling (or 0 to disable)."
        )


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

    The TURBO_MAX_IMAGE_MP decode ceiling applies to the bytes/path branches
    (where it prevents the allocation); numpy/PIL inputs are DELIBERATELY
    exempt — their memory is already committed by the caller, and the
    engine's own DET_MAX_SIDE_LIMIT bounds what detection will allocate.
    """
    # PIL image (duck-typed to avoid a hard Pillow dependency here).
    if hasattr(src, "mode") and hasattr(src, "size") and hasattr(src, "convert"):
        # Refusal scope mirrors the bytes/path TIFF guard, no wider:
        # * TIFF only — frame 0 of an animated GIF/WebP/APNG is the normal
        #   meaning of "the image", not data loss;
        # * only at tell()==0 — a caller who explicitly seek()ed picked
        #   their frame, exactly like passing a single-page file;
        # * n_frames may be a lazily-computed property that raises on damaged
        #   files — treat failure as single-frame and let convert() surface
        #   the real error.
        try:
            n_frames = int(getattr(src, "n_frames", 1) or 1)
            seeked = int(getattr(src, "tell", lambda: 0)() or 0) > 0
        except Exception:
            n_frames, seeked = 1, False
        if (n_frames > 1 and not seeked
                and str(getattr(src, "format", "")).upper() == "TIFF"):
            raise ValueError(
                f"this PIL image has {n_frames} frames — read() decodes ONE "
                "image, and converting it would silently drop every frame "
                "but the first. Per frame: im.seek(i) then "
                "read(im.copy()); or split with PIL.ImageSequence "
                "(copy() each frame) and use read_batch()."
            )
        rgb = np.asarray(src.convert("RGB"))
        return _cv2().cvtColor(rgb, _cv2().COLOR_RGB2BGR)

    if isinstance(src, np.ndarray):
        if src.dtype != np.uint8:
            # Every non-uint8 dtype must land in [0,255] by its VALUE RANGE,
            # not its dtype range. Two symmetric failure modes both end in
            # "read() returned zero lines silently": clipping a true 16-bit
            # scan saturates 99+% of pixels to white; but scaling by the
            # DTYPE's max crushes the far more common 8-bit-values-in-a-wide-
            # dtype case (PIL convert("I") -> int32, tifffile 8-in-16,
            # arr.astype(int)) to black. So: values already in [0,255] pass
            # through; only genuinely wide-range data is rescaled.
            if src.dtype == np.bool_:
                src = src.astype(np.uint8) * 255
            elif np.issubdtype(src.dtype, np.integer):
                info = np.iinfo(src.dtype)
                data_max = int(src.max()) if src.size else 0
                if info.max > 255 and data_max > 255:
                    # Scale by the DATA's own maximum — same rule as the
                    # float branch below. A dtype-based divisor destroyed
                    # 16-bit-in-int32 scans (everything rounded to zero) and
                    # a fixed 65535 rung crushed 10/12/14-bit sensor data
                    # into the bottom sliver of the range (measured: a
                    # 12-bit page OCR'd "INV0ICE" — contrast, not absolute
                    # luminance, is what recognition needs).
                    # float32 temporaries: float64 tripled the peak memory.
                    divisor = float(data_max)
                    src = src.astype(np.float32)
                    np.clip(src, 0, divisor, out=src)
                    np.multiply(src, np.float32(255.0 / divisor), out=src)
                    np.rint(src, out=src)
            elif np.issubdtype(src.dtype, np.floating) and src.size:
                # The range test must see only FINITE values: one NaN (or one
                # +inf, if it were mapped into the codomain first) used to
                # defeat the rescale and blacken the whole image.
                finite = src[np.isfinite(src)]
                fmax = float(finite.max()) if finite.size else 0.0
                # +inf maps to the BRIGHTEST FINITE value — mapping it to a
                # fixed 255 before the wide-scale rescale sent it to ~1
                # (near-black) on a 16-bit-scale float image.
                src = np.nan_to_num(src.astype(np.float32, copy=True),
                                    nan=0.0,
                                    posinf=(fmax if fmax > 0 else 255.0),
                                    neginf=0.0)
                if fmax <= 1.0:
                    src = (src * 255.0).round()
                elif fmax > 255.0:
                    # Float data on a wide scale (a 16-bit scan cast to
                    # float): clipping saturated it to two values.
                    src = (src * (255.0 / fmax)).round()
            src = np.clip(src, 0, 255).astype(np.uint8)
        return _from_numpy(src)

    if isinstance(src, (bytes, bytearray, memoryview)):
        raw = bytes(src)
        if not raw:
            # cv2.imdecode asserts on a zero-size buffer instead of returning
            # None — callers catching the documented ValueError got a raw
            # OpenCV error for an empty upload.
            raise ValueError("could not decode image bytes: empty input")
        _reject_multipage_tiff(raw, "these bytes")
        dims = _sniff_dims(raw)
        if dims:
            # BEFORE decode: a hostile/corrupt PNG or JPEG header claiming
            # absurd dimensions must fail in microseconds, not after the
            # allocator dies trying.
            _check_pixel_ceiling(dims[0], dims[1], "these bytes")
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = _cv2().imdecode(buf, _cv2().IMREAD_COLOR)
        if img is None:
            # Only sniffed on FAILURE: the happy path pays nothing, and the
            # by-far most common failure (someone handing a whole PDF to the
            # image reader) gets pointed at the right method instead of a
            # bare "could not decode".
            if looks_like_pdf(raw):
                raise ValueError(
                    "these bytes are a PDF document, not an image — read() "
                    "decodes single images. Use read_pdf() (whole document), "
                    "read_pdf_stream() (page by page), or their async twins."
                )
            raise ValueError("could not decode image bytes")
        # Post-decode fallback for formats without a header sniff (TIFF, BMP,
        # WebP, ...): the memory is spent, but downstream stages are spared.
        _check_pixel_ceiling(img.shape[1], img.shape[0], "these bytes")
        return img

    # Path-like.
    path = os.fspath(src)
    if not os.path.exists(path):
        raise FileNotFoundError(f"image not found: {path}")
    # imread mangles non-ASCII paths on some platforms; decode via a byte read.
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw:
        raise ValueError(f"could not decode image file: {path} is empty")
    _reject_multipage_tiff(raw, path)
    dims = _sniff_dims(raw)
    if dims:
        _check_pixel_ceiling(dims[0], dims[1], path)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = _cv2().imdecode(buf, _cv2().IMREAD_COLOR)
    if img is None:
        if looks_like_pdf(buf[:_PDF_SNIFF_WINDOW].tobytes()):
            raise ValueError(
                f"{path} is a PDF document, not an image — read() decodes "
                "single images. Use read_pdf() (whole document), "
                "read_pdf_stream() (page by page), or their async twins."
            )
        raise ValueError(f"could not decode image file: {path}")
    _check_pixel_ceiling(img.shape[1], img.shape[0], path)
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
