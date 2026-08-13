from __future__ import annotations

import pytest

grpc = pytest.importorskip("grpc")

from turboocr import (  # noqa: E402
    AsyncGrpcClient,
    DimensionsTooLarge,
    NetworkError,
    PoolExhausted,
    ServerError,
    Timeout,
)
from turboocr._core.retry import RetryPolicy  # noqa: E402


async def _abort(context, status, message: str, error_code: str | None = None) -> None:
    if error_code is not None:
        await context.send_initial_metadata([])
        context.set_trailing_metadata((("x-error-code", error_code),))
    await context.abort(status, message)


async def test_x_error_code_wins_over_status(grpc_server) -> None:
    """RESOURCE_EXHAUSTED + x-error-code=DIMENSIONS_TOO_LARGE → DimensionsTooLarge,
    not PoolExhausted. The explicit code wins because it carries semantic
    information the status code alone cannot express."""
    target, service = grpc_server

    async def handler(_req, context):
        await _abort(
            context, grpc.StatusCode.RESOURCE_EXHAUSTED, "image too big",
            error_code="DIMENSIONS_TOO_LARGE",
        )

    service.set("recognize", handler)
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        with pytest.raises(DimensionsTooLarge) as ei:
            await client.recognize_image(b"\x89PNG")
    assert ei.value.code == "DIMENSIONS_TOO_LARGE"
    assert ei.value.status_code == 429  # gRPC RESOURCE_EXHAUSTED → 429


async def test_unavailable_without_metadata_maps_to_network_error(grpc_server) -> None:
    target, service = grpc_server

    async def handler(_req, context):
        await _abort(context, grpc.StatusCode.UNAVAILABLE, "backend down")

    service.set("recognize", handler)
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        with pytest.raises(NetworkError) as ei:
            await client.recognize_image(b"\x89PNG")
    assert ei.value.status_code == 503
    assert ei.value.code is None


async def test_deadline_exceeded_maps_to_timeout(grpc_server) -> None:
    target, service = grpc_server

    async def handler(_req, context):
        await _abort(context, grpc.StatusCode.DEADLINE_EXCEEDED, "slow")

    service.set("recognize", handler)
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        with pytest.raises(Timeout) as ei:
            await client.recognize_image(b"\x89PNG")
    assert ei.value.status_code == 504


async def test_internal_maps_to_server_error(grpc_server) -> None:
    target, service = grpc_server

    async def handler(_req, context):
        await _abort(context, grpc.StatusCode.INTERNAL, "boom")

    service.set("recognize", handler)
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        with pytest.raises(ServerError) as ei:
            await client.recognize_image(b"\x89PNG")
    assert ei.value.status_code == 500


async def test_resource_exhausted_without_code_maps_to_pool_exhausted(
    grpc_server,
) -> None:
    target, service = grpc_server

    async def handler(_req, context):
        await _abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, "busy")

    service.set("recognize", handler)
    async with AsyncGrpcClient(target=target, retry=RetryPolicy(attempts=1)) as client:
        with pytest.raises(PoolExhausted) as ei:
            await client.recognize_image(b"\x89PNG")
    assert ei.value.status_code == 429
