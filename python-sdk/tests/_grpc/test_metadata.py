from __future__ import annotations

import pytest

grpc = pytest.importorskip("grpc")

from turboocr import AsyncGrpcClient  # noqa: E402
from turboocr._core.retry import RetryPolicy  # noqa: E402
from turboocr._grpc.metadata import build_metadata  # noqa: E402

from .conftest import build_ocr_response_proto  # noqa: E402


def test_build_metadata_bearer_scheme_is_lowercase() -> None:
    md = build_metadata(
        api_key="sk-abc",
        auth_scheme="bearer",
        request_id="req-1",
        extra={"X-Custom": "v"},
    )
    keys = {k for k, _ in md}
    # gRPC requires lowercase keys; mixed case from `extra` is normalized.
    assert all(k == k.lower() for k in keys)
    assert "authorization" in keys
    assert "x-custom" in keys
    assert "x-request-id" in keys
    assert "user-agent" in keys


def test_build_metadata_x_api_key_scheme() -> None:
    md = build_metadata(
        api_key="sk-z", auth_scheme="x-api-key", request_id="req-2", extra=None
    )
    pairs = dict(md)
    assert pairs["x-api-key"] == "sk-z"
    assert "authorization" not in pairs


def test_build_metadata_callable_scheme_is_lowercased() -> None:
    def scheme(key: str) -> tuple[str, str]:
        return ("X-My-Auth", f"k:{key}")

    md = build_metadata(api_key="abc", auth_scheme=scheme, request_id="r", extra=None)
    pairs = dict(md)
    assert pairs["x-my-auth"] == "k:abc"


async def test_request_metadata_present_on_wire(grpc_server) -> None:
    """End-to-end: the metadata the client sends actually reaches the server."""
    target, service = grpc_server
    captured: dict[str, str] = {}

    async def handler(_req, context):
        for k, v in context.invocation_metadata():
            captured[k] = v
        return build_ocr_response_proto()

    service.set("recognize", handler)
    async with AsyncGrpcClient(
        target=target,
        api_key="sk-abc",
        retry=RetryPolicy(attempts=1),
    ) as client:
        await client.recognize_image(b"\x89PNG")

    assert captured.get("authorization") == "Bearer sk-abc"
    assert "x-request-id" in captured
    # gRPC normalizes inbound keys to lowercase; verify nothing arrived in mixed case.
    assert all(k == k.lower() for k in captured)
