from __future__ import annotations

import pytest

grpc = pytest.importorskip("grpc")

from turboocr import AsyncGrpcClient  # noqa: E402
from turboocr._core.retry import DEFAULT_RETRY_GRPC_STATUSES, RetryPolicy  # noqa: E402

from .conftest import build_ocr_response_proto  # noqa: E402


async def test_retries_on_unavailable_then_succeeds(grpc_server) -> None:
    target, service = grpc_server
    attempts = {"n": 0}

    async def handler(_req, context):
        attempts["n"] += 1
        if attempts["n"] == 1:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "transient")
        return build_ocr_response_proto()

    service.set("recognize", handler)
    async with AsyncGrpcClient(
        target=target,
        retry=RetryPolicy(attempts=2, backoff=0.0, jitter=0.0),
    ) as client:
        resp = await client.recognize_image(b"\x89PNG")
    assert resp.results[0].text == "hello"
    assert attempts["n"] == 2


async def test_does_not_retry_on_invalid_argument(grpc_server) -> None:
    target, service = grpc_server
    attempts = {"n": 0}

    async def handler(_req, context):
        attempts["n"] += 1
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "bad")

    service.set("recognize", handler)
    async with AsyncGrpcClient(
        target=target,
        retry=RetryPolicy(attempts=3, backoff=0.0, jitter=0.0),
    ) as client:
        with pytest.raises(Exception):
            await client.recognize_image(b"\x89PNG")
    assert attempts["n"] == 1  # non-retryable status → single attempt


def test_default_grpc_statuses_match_documentation() -> None:
    assert frozenset(
        {"UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED"}
    ) == DEFAULT_RETRY_GRPC_STATUSES


async def test_custom_retry_grpc_statuses_honored(grpc_server) -> None:
    target, service = grpc_server
    attempts = {"n": 0}

    async def handler(_req, context):
        attempts["n"] += 1
        if attempts["n"] == 1:
            await context.abort(grpc.StatusCode.ABORTED, "transient-aborted")
        return build_ocr_response_proto()

    service.set("recognize", handler)
    policy = RetryPolicy(
        attempts=2,
        backoff=0.0,
        jitter=0.0,
        retry_grpc_statuses=frozenset({"ABORTED"}),
    )
    async with AsyncGrpcClient(target=target, retry=policy) as client:
        resp = await client.recognize_image(b"\x89PNG")
    assert resp.results[0].text == "hello"
    assert attempts["n"] == 2


async def test_unavailable_skipped_when_excluded(grpc_server) -> None:
    target, service = grpc_server
    attempts = {"n": 0}

    async def handler(_req, context):
        attempts["n"] += 1
        await context.abort(grpc.StatusCode.UNAVAILABLE, "transient")

    service.set("recognize", handler)
    policy = RetryPolicy(
        attempts=3,
        backoff=0.0,
        jitter=0.0,
        retry_grpc_statuses=frozenset({"DEADLINE_EXCEEDED"}),
    )
    async with AsyncGrpcClient(target=target, retry=policy) as client:
        with pytest.raises(Exception):
            await client.recognize_image(b"\x89PNG")
    assert attempts["n"] == 1
