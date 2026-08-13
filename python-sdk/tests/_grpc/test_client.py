from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("grpc")

from turboocr import AsyncGrpcClient, GrpcClient
from turboocr._core.retry import RetryPolicy


async def test_sync_recognize_image_happy_path(grpc_server) -> None:
    target, _service = grpc_server

    def call() -> str:
        with GrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
            resp = client.recognize_image(b"\x89PNG")
            return resp.results[0].text

    text = await asyncio.to_thread(call)
    assert text == "hello"


async def test_async_recognize_image_happy_path(grpc_server) -> None:
    target, _service = grpc_server
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        resp = await client.recognize_image(b"\x89PNG")
    assert resp.results[0].text == "hello"
    assert resp.blocks[0].class_name == "paragraph_title"


async def test_async_recognize_batch(grpc_server) -> None:
    target, _service = grpc_server
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        batch = await client.recognize_batch([b"\x89PNG", b"\x89PNG"])
    # Default handler returns a single result; what matters is the SDK parses
    # the BatchResponse shape — errors list parallels batch_results.
    assert len(batch.batch_results) == len(batch.errors)
    assert batch.batch_results[0].results[0].text == "hello"


async def test_async_recognize_pdf(grpc_server) -> None:
    target, _service = grpc_server
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        pdf_resp = await client.recognize_pdf(b"%PDF-1.4 stub", dpi=100, mode="ocr")
    assert len(pdf_resp.pages) == 1
    assert pdf_resp.pages[0].page == 1
    assert pdf_resp.pages[0].results[0].text == "hello"


async def test_grpc_recognize_pdf_reading_order_raises(grpc_server) -> None:
    from turboocr import InvalidParameter

    target, _service = grpc_server
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        with pytest.raises(InvalidParameter, match="reading_order"):
            await client.recognize_pdf(b"%PDF-1.4 stub", dpi=100, reading_order=True)


async def test_async_health(grpc_server) -> None:
    target, _service = grpc_server
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        status = await client.health()
    assert status.ok
    assert status.body == "ok"
    assert status.body_json == {"status": "ok"}


async def test_sync_repr_does_not_leak_key(grpc_server) -> None:
    target, _service = grpc_server

    def call() -> str:
        with GrpcClient(target=target, api_key="sk-secret") as client:
            return repr(client)

    text = await asyncio.to_thread(call)
    assert "sk-secret" not in text
    assert "auth=" in text
    assert "/set" in text


async def test_async_interceptor_fires_on_call(grpc_server) -> None:
    """User-supplied async interceptors should be invoked on every RPC.

    This is the gRPC analogue of HTTP's `on_request` / `on_response` hooks —
    the integration point for OpenTelemetry, Datadog, custom metrics, etc.
    """
    import grpc.aio

    target, _service = grpc_server
    method_names: list[str] = []

    class _RecordingInterceptor(grpc.aio.UnaryUnaryClientInterceptor):
        async def intercept_unary_unary(
            self, continuation, client_call_details, request
        ):
            method = client_call_details.method
            method_names.append(method.decode() if isinstance(method, bytes) else method)
            return await continuation(client_call_details, request)

    async with AsyncGrpcClient(
        target=target,
        retry=RetryPolicy(attempts=1),
        interceptors=[_RecordingInterceptor()],
    ) as client:
        await client.recognize_image(b"\x89PNG")
        await client.health()

    assert method_names == ["/ocr.OCRService/Recognize", "/ocr.OCRService/Health"]
