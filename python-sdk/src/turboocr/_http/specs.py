from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from .._core.content import ContentProvider, ImageInput, read_image_bytes, streamable_content
from .._core.options import OcrOptions
from ..models import PdfMode
from .transport import encode_base64

type BoolParam = bool | None


@dataclass(frozen=True, slots=True)
class RequestSpec:
    method: Literal["GET", "POST"]
    path: str
    params: Mapping[str, str] = field(default_factory=dict)
    content: ContentProvider | None = None
    json_body: Mapping[str, object] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    # When set, httpx is given a `files=` arg (multipart/form-data) instead of
    # `content=`. Tuple shape matches httpx: (field_name, (filename, body, mime)).
    # Multipart bodies are bytes-only (loaded into memory) — multipart is an
    # interop format here, not a streaming optimization.
    files: tuple[str, tuple[str, bytes, str]] | None = None


def recognize_image_spec(image: ImageInput, *, opts: OcrOptions) -> RequestSpec:
    return RequestSpec(
        method="POST",
        path="/ocr/raw",
        params=opts.to_query_params(),
        content=streamable_content(image),
        headers={"Content-Type": "application/octet-stream"},
    )


def recognize_base64_spec(b64: str, *, opts: OcrOptions) -> RequestSpec:
    return RequestSpec(
        method="POST",
        path="/ocr",
        params=opts.to_query_params(),
        json_body={"image": b64},
    )


def recognize_pixels_spec(
    pixels: ImageInput,
    *,
    width: int,
    height: int,
    channels: int,
    opts: OcrOptions,
) -> RequestSpec:
    # Dimensions travel as query params — the legacy X-Width/X-Height headers
    # are deprecated server-side (the server answers them with a
    # `Deprecation: true` header since v3.1).
    params = dict(opts.to_query_params())
    params["width"] = str(width)
    params["height"] = str(height)
    params["channels"] = str(channels)
    return RequestSpec(
        method="POST",
        path="/ocr/pixels",
        params=params,
        content=streamable_content(pixels),
        headers={"Content-Type": "application/octet-stream"},
    )


def recognize_batch_spec(images: Iterable[ImageInput], *, opts: OcrOptions) -> RequestSpec:
    b64_images = [encode_base64(read_image_bytes(img)) for img in images]
    return RequestSpec(
        method="POST",
        path="/ocr/batch",
        params=opts.to_query_params(),
        json_body={"images": b64_images},
    )


def _pdf_params(
    *,
    dpi: int | None,
    mode: PdfMode | str | None,
    autorotate: bool | None,
    opts: OcrOptions,
) -> dict[str, str]:
    params = dict(opts.to_query_params())
    if dpi is not None:
        params["dpi"] = str(dpi)
    if mode is not None:
        params["mode"] = mode.value if isinstance(mode, PdfMode) else str(mode)
    if autorotate is not None:
        params["autorotate"] = "1" if autorotate else "0"
    return params


def recognize_pdf_spec(
    pdf: ImageInput,
    *,
    dpi: int | None,
    mode: PdfMode | str | None,
    autorotate: bool | None = None,
    opts: OcrOptions,
) -> RequestSpec:
    return RequestSpec(
        method="POST",
        path="/ocr/pdf",
        params=_pdf_params(dpi=dpi, mode=mode, autorotate=autorotate, opts=opts),
        content=streamable_content(pdf),
        headers={"Content-Type": "application/pdf"},
    )


def pdf_markdown_spec(
    pdf: ImageInput,
    *,
    dpi: int | None,
    mode: PdfMode | str | None,
    autorotate: bool | None,
    as_pages: bool,
    opts: OcrOptions,
) -> RequestSpec:
    params = _pdf_params(dpi=dpi, mode=mode, autorotate=autorotate, opts=opts)
    params["markdown"] = "1"
    if as_pages:
        params["as_pages"] = "1"
    return RequestSpec(
        method="POST",
        path="/ocr/pdf",
        params=params,
        content=streamable_content(pdf),
        headers={"Content-Type": "application/pdf"},
    )


def page_markdown_spec(image: ImageInput, *, embed: bool) -> RequestSpec:
    params: dict[str, str] = {}
    if not embed:
        params["embed"] = "0"
    return RequestSpec(
        method="POST",
        path="/ocr/markdown",
        params=params,
        content=streamable_content(image),
        headers={"Content-Type": "application/octet-stream"},
    )


def stream_spec(
    document: ImageInput,
    *,
    dpi: int | None,
    mode: PdfMode | str | None,
    autorotate: bool | None,
    opts: OcrOptions,
) -> RequestSpec:
    # /ocr/stream sniffs PDF vs image by magic bytes; the same spec serves both.
    return RequestSpec(
        method="POST",
        path="/ocr/stream",
        params=_pdf_params(dpi=dpi, mode=mode, autorotate=autorotate, opts=opts),
        content=streamable_content(document),
        headers={"Content-Type": "application/octet-stream"},
    )


def capabilities_spec() -> RequestSpec:
    return RequestSpec(method="GET", path="/capabilities")


type HealthEndpoint = Literal["health", "live", "ready"]


def health_spec(*, endpoint: HealthEndpoint = "health") -> RequestSpec:
    match endpoint:
        case "health":
            return RequestSpec(method="GET", path="/health")
        case "live":
            return RequestSpec(method="GET", path="/health/live")
        case "ready":
            return RequestSpec(method="GET", path="/health/ready")
