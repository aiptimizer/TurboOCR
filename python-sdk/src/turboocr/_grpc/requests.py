from __future__ import annotations

from collections.abc import Iterable

from .._core.options import OcrOptions
from ..errors import InvalidParameter
from ..models import PdfMode
from ._stubs import ocr_pb2 as pb2


# proto3 without `optional` has no field presence (verified via DESCRIPTOR;
# has_presence=False for every bool). So `None` collapses to False on the
# wire — same effective behavior as HTTP today because the server defaults
# each option to False, but if the server ever changes a default, gRPC users
# would have to start passing the option explicitly. The HTTP transport
# preserves tristate by omitting the query param.
def build_recognize_request(image: bytes, opts: OcrOptions) -> pb2.OCRRequest:
    return pb2.OCRRequest(
        image=image,
        layout=bool(opts.layout),
        reading_order=bool(opts.reading_order),
        as_blocks=bool(opts.include_blocks),
        tables=bool(opts.tables),
        formulas=bool(opts.formulas),
    )


def build_recognize_pixels_request(
    pixels: bytes,
    *,
    width: int,
    height: int,
    channels: int,
    opts: OcrOptions,
) -> pb2.OCRRequest:
    return pb2.OCRRequest(
        pixels=pixels,
        width=width,
        height=height,
        channels=channels,
        layout=bool(opts.layout),
        reading_order=bool(opts.reading_order),
        as_blocks=bool(opts.include_blocks),
        tables=bool(opts.tables),
        formulas=bool(opts.formulas),
    )


def build_recognize_batch_request(
    images: Iterable[bytes], opts: OcrOptions
) -> pb2.OCRBatchRequest:
    return pb2.OCRBatchRequest(
        images=list(images),
        layout=bool(opts.layout),
        reading_order=bool(opts.reading_order),
        as_blocks=bool(opts.include_blocks),
        tables=bool(opts.tables),
        formulas=bool(opts.formulas),
    )


def build_recognize_pdf_request(
    pdf: bytes,
    *,
    dpi: int | None,
    mode: PdfMode | str | None,
    opts: OcrOptions,
) -> pb2.OCRPDFRequest:
    if opts.reading_order:
        raise InvalidParameter(
            "reading_order=True is not supported by the gRPC PDF endpoint; "
            "use the HTTP client (POST /ocr/pdf) or omit reading_order"
        )
    mode_str = ""
    if mode is not None:
        mode_str = mode.value if isinstance(mode, PdfMode) else str(mode)
    return pb2.OCRPDFRequest(
        pdf_data=pdf,
        mode=mode_str,
        dpi=dpi if dpi is not None else 0,
        layout=bool(opts.layout),
        as_blocks=bool(opts.include_blocks),
        tables=bool(opts.tables),
        formulas=bool(opts.formulas),
    )
