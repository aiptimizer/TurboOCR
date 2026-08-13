from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

# Skip the whole gRPC test subdir cleanly if grpcio isn't installed.
pytest.importorskip("grpc")
import grpc
import grpc.aio

from turboocr._grpc._stubs import ocr_pb2 as pb2
from turboocr._grpc._stubs import ocr_pb2_grpc as pb2_grpc


def ocr_json_payload() -> dict[str, object]:
    """Same shape as the HTTP test's _ocr_payload — proves the fast path."""
    return {
        "results": [
            {
                "id": 0,
                "text": "hello",
                "confidence": 0.99,
                "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                "layout_id": 0,
            }
        ],
        "layout": [
            {
                "id": 0,
                "class": "paragraph_title",
                "class_id": 17,
                "confidence": 0.9,
                "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
            }
        ],
        "reading_order": [0],
        "blocks": [
            {
                "id": 0,
                "layout_id": 0,
                "class": "paragraph_title",
                "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                "content": "hello",
                "order_index": 0,
            }
        ],
    }


def ocr_pdf_page_payload(page_index: int = 0) -> dict[str, object]:
    return {
        "page": page_index + 1,
        "page_index": page_index,
        "dpi": 100,
        "width": 800,
        "height": 600,
        "results": [
            {
                "id": 0,
                "text": "hello",
                "confidence": 0.99,
                "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
            }
        ],
        "mode": "ocr",
        "text_layer_quality": "absent",
    }


def build_ocr_response_proto(
    *, with_json: bool = True, with_results: bool = False
) -> pb2.OCRResponse:
    resp = pb2.OCRResponse(num_detections=1, reading_order=[0])
    if with_json:
        resp.json_response = json.dumps(ocr_json_payload()).encode("utf-8")
    if with_results:
        resp.results.append(
            pb2.OCRResult(
                text="hello",
                confidence=0.99,
                bounding_box=[pb2.BoundingBox(x=[0, 10, 10, 0], y=[0, 0, 5, 5])],
            )
        )
    return resp


class _DefaultOCRService(pb2_grpc.OCRServiceServicer):
    """Default in-process handler. Tests override via `register_handler`."""

    def __init__(self) -> None:
        self._recognize: Callable[[Any, Any], pb2.OCRResponse] | None = None
        self._recognize_batch: Callable[[Any, Any], pb2.OCRBatchResponse] | None = None
        self._recognize_pdf: Callable[[Any, Any], pb2.OCRPDFResponse] | None = None
        self._health: Callable[[Any, Any], pb2.HealthResponse] | None = None

    def set(self, rpc: str, fn: Callable[[Any, Any], Any]) -> None:
        setattr(self, f"_{rpc}", fn)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if hasattr(value, "__await__"):
            return await value
        return value

    async def Recognize(
        self, request: pb2.OCRRequest, context: grpc.aio.ServicerContext
    ) -> pb2.OCRResponse:
        if self._recognize is None:
            return build_ocr_response_proto()
        return await self._maybe_await(self._recognize(request, context))

    async def RecognizeBatch(
        self, request: pb2.OCRBatchRequest, context: grpc.aio.ServicerContext
    ) -> pb2.OCRBatchResponse:
        if self._recognize_batch is None:
            return pb2.OCRBatchResponse(
                batch_results=[build_ocr_response_proto()], total_images=1
            )
        return await self._maybe_await(self._recognize_batch(request, context))

    async def RecognizePDF(
        self, request: pb2.OCRPDFRequest, context: grpc.aio.ServicerContext
    ) -> pb2.OCRPDFResponse:
        if self._recognize_pdf is None:
            page = pb2.OCRPageResult(
                page_number=1,
                json_response=json.dumps(ocr_pdf_page_payload(0)).encode("utf-8"),
            )
            return pb2.OCRPDFResponse(pages=[page])
        return await self._maybe_await(self._recognize_pdf(request, context))

    async def Health(
        self, request: pb2.HealthRequest, context: grpc.aio.ServicerContext
    ) -> pb2.HealthResponse:
        if self._health is None:
            return pb2.HealthResponse(status="ok")
        return await self._maybe_await(self._health(request, context))


@pytest.fixture
async def grpc_server() -> AsyncIterator[tuple[str, _DefaultOCRService]]:
    """Start an in-process async gRPC server bound to localhost:0.

    Yields (target, service) — tests mutate `service.set("recognize", fn)` to
    swap in canned handlers before issuing the call. Server lifecycle is
    bounded by the fixture scope, so each test gets a fresh port.
    """
    service = _DefaultOCRService()
    server = grpc.aio.server()
    pb2_grpc.add_OCRServiceServicer_to_server(service, server)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    try:
        yield f"localhost:{port}", service
    finally:
        await server.stop(grace=None)
