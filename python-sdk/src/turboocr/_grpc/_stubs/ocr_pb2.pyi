from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OCRRequest(_message.Message):
    __slots__ = ("image", "layout", "pixels", "width", "height", "channels", "reading_order", "as_blocks", "tables", "formulas")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    PIXELS_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    READING_ORDER_FIELD_NUMBER: _ClassVar[int]
    AS_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    TABLES_FIELD_NUMBER: _ClassVar[int]
    FORMULAS_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    layout: bool
    pixels: bytes
    width: int
    height: int
    channels: int
    reading_order: bool
    as_blocks: bool
    tables: bool
    formulas: bool
    def __init__(self, image: _Optional[bytes] = ..., layout: bool = ..., pixels: _Optional[bytes] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., channels: _Optional[int] = ..., reading_order: bool = ..., as_blocks: bool = ..., tables: bool = ..., formulas: bool = ...) -> None: ...

class OCRBatchRequest(_message.Message):
    __slots__ = ("images", "det_batch_num", "layout", "reading_order", "as_blocks", "tables", "formulas")
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    DET_BATCH_NUM_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    READING_ORDER_FIELD_NUMBER: _ClassVar[int]
    AS_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    TABLES_FIELD_NUMBER: _ClassVar[int]
    FORMULAS_FIELD_NUMBER: _ClassVar[int]
    images: _containers.RepeatedScalarFieldContainer[bytes]
    det_batch_num: int
    layout: bool
    reading_order: bool
    as_blocks: bool
    tables: bool
    formulas: bool
    def __init__(self, images: _Optional[_Iterable[bytes]] = ..., det_batch_num: _Optional[int] = ..., layout: bool = ..., reading_order: bool = ..., as_blocks: bool = ..., tables: bool = ..., formulas: bool = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: _containers.RepeatedScalarFieldContainer[float]
    y: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, x: _Optional[_Iterable[float]] = ..., y: _Optional[_Iterable[float]] = ...) -> None: ...

class OCRResult(_message.Message):
    __slots__ = ("text", "confidence", "bounding_box")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    text: str
    confidence: float
    bounding_box: _containers.RepeatedCompositeFieldContainer[BoundingBox]
    def __init__(self, text: _Optional[str] = ..., confidence: _Optional[float] = ..., bounding_box: _Optional[_Iterable[_Union[BoundingBox, _Mapping]]] = ...) -> None: ...

class OCRResponse(_message.Message):
    __slots__ = ("results", "num_detections", "json_response", "reading_order", "error")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    NUM_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    JSON_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    READING_ORDER_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[OCRResult]
    num_detections: int
    json_response: bytes
    reading_order: _containers.RepeatedScalarFieldContainer[int]
    error: str
    def __init__(self, results: _Optional[_Iterable[_Union[OCRResult, _Mapping]]] = ..., num_detections: _Optional[int] = ..., json_response: _Optional[bytes] = ..., reading_order: _Optional[_Iterable[int]] = ..., error: _Optional[str] = ...) -> None: ...

class OCRBatchResponse(_message.Message):
    __slots__ = ("batch_results", "total_images")
    BATCH_RESULTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_IMAGES_FIELD_NUMBER: _ClassVar[int]
    batch_results: _containers.RepeatedCompositeFieldContainer[OCRResponse]
    total_images: int
    def __init__(self, batch_results: _Optional[_Iterable[_Union[OCRResponse, _Mapping]]] = ..., total_images: _Optional[int] = ...) -> None: ...

class OCRPDFRequest(_message.Message):
    __slots__ = ("pdf_data", "mode", "dpi", "layout", "as_blocks", "tables", "formulas")
    PDF_DATA_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    DPI_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    AS_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    TABLES_FIELD_NUMBER: _ClassVar[int]
    FORMULAS_FIELD_NUMBER: _ClassVar[int]
    pdf_data: bytes
    mode: str
    dpi: int
    layout: bool
    as_blocks: bool
    tables: bool
    formulas: bool
    def __init__(self, pdf_data: _Optional[bytes] = ..., mode: _Optional[str] = ..., dpi: _Optional[int] = ..., layout: bool = ..., as_blocks: bool = ..., tables: bool = ..., formulas: bool = ...) -> None: ...

class OCRPDFResponse(_message.Message):
    __slots__ = ("pages",)
    PAGES_FIELD_NUMBER: _ClassVar[int]
    pages: _containers.RepeatedCompositeFieldContainer[OCRPageResult]
    def __init__(self, pages: _Optional[_Iterable[_Union[OCRPageResult, _Mapping]]] = ...) -> None: ...

class OCRPageResult(_message.Message):
    __slots__ = ("page_number", "results", "width", "height", "dpi", "mode", "text_layer_quality", "json_response")
    PAGE_NUMBER_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    DPI_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    TEXT_LAYER_QUALITY_FIELD_NUMBER: _ClassVar[int]
    JSON_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    page_number: int
    results: _containers.RepeatedCompositeFieldContainer[OCRResult]
    width: int
    height: int
    dpi: int
    mode: str
    text_layer_quality: str
    json_response: bytes
    def __init__(self, page_number: _Optional[int] = ..., results: _Optional[_Iterable[_Union[OCRResult, _Mapping]]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., dpi: _Optional[int] = ..., mode: _Optional[str] = ..., text_layer_quality: _Optional[str] = ..., json_response: _Optional[bytes] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "response_mode")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_MODE_FIELD_NUMBER: _ClassVar[int]
    status: str
    response_mode: str
    def __init__(self, status: _Optional[str] = ..., response_mode: _Optional[str] = ...) -> None: ...
