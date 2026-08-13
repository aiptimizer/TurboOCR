from __future__ import annotations

import base64
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType, TracebackType
from typing import Any, Final, NotRequired, Self, TypedDict, Unpack

import grpc
import grpc.aio

from .._core.content import ImageInput, read_image_bytes
from .._core.env import DEFAULT_TIMEOUT, AuthScheme, resolve_api_key
from .._core.ids import short_request_id
from .._core.options import OcrOptions
from .._core.retry import RetryPolicy
from ..models import BatchResponse, HealthStatus, OcrResponse, PdfMode, PdfResponse
from ..searchable_pdf import make_searchable_pdf as _overlay
from ._stubs import ocr_pb2 as pb2
from ._stubs import ocr_pb2_grpc as pb2_grpc
from .channel import (
    AsyncInterceptor,
    ChannelOption,
    SyncInterceptor,
    make_async_channel,
    make_channel,
    resolve_grpc_target,
)
from .errors import classify_rpc_error
from .metadata import build_metadata
from .parse import parse_batch_response, parse_ocr_response, parse_pdf_response
from .requests import (
    build_recognize_batch_request,
    build_recognize_pdf_request,
    build_recognize_pixels_request,
    build_recognize_request,
)
from .retry import execute_grpc_with_retries, execute_grpc_with_retries_async

logger = logging.getLogger("turboocr.grpc")

type BoolParam = bool | None

_HEALTH_BODY: Final[str] = "ok"


def _log_rpc(rpc: str, req_id: str, elapsed_ms: float) -> None:
    logger.debug("turbo-ocr gRPC %s (%.1fms) [req=%s]", rpc, elapsed_ms, req_id)


class _BaseGrpcClient:
    def __init__(
        self,
        *,
        target: str | None,
        api_key: str | None,
        auth_scheme: AuthScheme,
        timeout: float,
        default_metadata: Mapping[str, str] | None,
        retry: RetryPolicy | None,
    ) -> None:
        self._target = resolve_grpc_target(target)
        self._api_key = resolve_api_key(api_key)
        self._auth_scheme: AuthScheme = auth_scheme
        self._default_metadata: dict[str, str] = dict(default_metadata or {})
        self._timeout = timeout
        self._retry = retry if retry is not None else RetryPolicy()

    @property
    def target(self) -> str:
        return self._target

    @property
    def auth_scheme(self) -> AuthScheme:
        return self._auth_scheme

    @property
    def default_metadata(self) -> Mapping[str, str]:
        return MappingProxyType(self._default_metadata)

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def retry(self) -> RetryPolicy:
        return self._retry

    def _build_metadata(self, request_id: str) -> list[tuple[str, str]]:
        return build_metadata(
            api_key=self._api_key,
            auth_scheme=self._auth_scheme,
            request_id=request_id,
            extra=self._default_metadata,
        )

    def _repr_fields(self, kind: str) -> str:
        masked = "set" if self._api_key else "none"
        return (
            f"{kind}(target={self._target!r}, "
            f"timeout={self._timeout}, "
            f"auth={self._auth_scheme!r}/{masked}, "
            f"retry=attempts={self._retry.attempts})"
        )


class _GrpcClientKwargs(TypedDict, total=False):
    target: NotRequired[str | None]
    api_key: NotRequired[str | None]
    auth_scheme: NotRequired[AuthScheme]
    secure: NotRequired[bool]
    credentials: NotRequired[grpc.ChannelCredentials | None]
    timeout: NotRequired[float]
    default_metadata: NotRequired[Mapping[str, str] | None]
    retry: NotRequired[RetryPolicy | None]
    channel_options: NotRequired[list[ChannelOption] | None]
    interceptors: NotRequired[Sequence[SyncInterceptor] | None]
    channel: NotRequired[grpc.Channel | None]


class GrpcClient(_BaseGrpcClient):
    def __init__(
        self,
        *,
        target: str | None = None,
        api_key: str | None = None,
        auth_scheme: AuthScheme = "bearer",
        secure: bool = False,
        credentials: grpc.ChannelCredentials | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        default_metadata: Mapping[str, str] | None = None,
        retry: RetryPolicy | None = None,
        channel_options: list[ChannelOption] | None = None,
        interceptors: Sequence[SyncInterceptor] | None = None,
        channel: grpc.Channel | None = None,
    ) -> None:
        super().__init__(
            target=target,
            api_key=api_key,
            auth_scheme=auth_scheme,
            timeout=timeout,
            default_metadata=default_metadata,
            retry=retry,
        )
        if channel is not None:
            self._channel = channel
            self._owns_channel = False
        else:
            self._channel = make_channel(
                self._target,
                secure=secure,
                credentials=credentials,
                options=channel_options,
                interceptors=interceptors,
            )
            self._owns_channel = True
        self._stub = pb2_grpc.OCRServiceStub(self._channel)  # type: ignore[no-untyped-call]

    def __repr__(self) -> str:
        return self._repr_fields("GrpcClient")

    @classmethod
    def from_env(cls, **overrides: Unpack[_GrpcClientKwargs]) -> Self:
        """Construct a `GrpcClient` driven by `TURBO_OCR_GRPC_TARGET` and friends."""
        return cls(**overrides)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the channel, but only if this client created it.

        When you passed `channel=` at construction time, ownership stays with
        the caller — `close()` is a no-op (with a DEBUG-level log) so a shared
        channel isn't accidentally torn down underneath other consumers.
        """
        if self._owns_channel:
            self._channel.close()
        else:
            logger.debug("GrpcClient.close(): external channel — caller owns lifecycle")

    def _call(
        self,
        rpc: str,
        unary: Any,
        request: Any,
        *,
        timeout: float | None,
    ) -> Any:
        per_call_timeout = timeout if timeout is not None else self._timeout

        def attempt() -> Any:
            req_id = short_request_id()
            md = self._build_metadata(req_id)
            start = time.monotonic()
            try:
                result = unary(request, timeout=per_call_timeout, metadata=md)
            finally:
                _log_rpc(rpc, req_id, (time.monotonic() - start) * 1000.0)
            return result

        try:
            return execute_grpc_with_retries(policy=self._retry, rpc=rpc, attempt_send=attempt)
        except grpc.RpcError as exc:
            raise classify_rpc_error(exc) from exc

    def recognize_image(
        self,
        image: ImageInput,
        *,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> OcrResponse:
        """gRPC equivalent of [`Client.recognize_image`][turboocr.Client.recognize_image]."""
        req = build_recognize_request(
            read_image_bytes(image),
            OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = self._call("Recognize", self._stub.Recognize, req, timeout=timeout)
        return parse_ocr_response(resp)

    def recognize_base64(
        self,
        b64: str,
        *,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> OcrResponse:
        """gRPC equivalent of [`Client.recognize_base64`][turboocr.Client.recognize_base64].

        Note: gRPC has no native base64 wire encoding, so the SDK decodes
        client-side and sends raw bytes. The convenience method is kept for
        surface parity with the HTTP client.
        """
        return self.recognize_image(
            base64.b64decode(b64),
            layout=layout,
            reading_order=reading_order,
            include_blocks=include_blocks,
            tables=tables,
            formulas=formulas,
            timeout=timeout,
        )

    def recognize_pixels(
        self,
        pixels: ImageInput,
        *,
        width: int,
        height: int,
        channels: int = 3,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> OcrResponse:
        """gRPC equivalent of [`Client.recognize_pixels`][turboocr.Client.recognize_pixels]."""
        req = build_recognize_pixels_request(
            read_image_bytes(pixels),
            width=width,
            height=height,
            channels=channels,
            opts=OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = self._call("Recognize", self._stub.Recognize, req, timeout=timeout)
        return parse_ocr_response(resp)

    def recognize_batch(
        self,
        images: Iterable[ImageInput],
        *,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> BatchResponse:
        """gRPC equivalent of [`Client.recognize_batch`][turboocr.Client.recognize_batch]."""
        req = build_recognize_batch_request(
            (read_image_bytes(img) for img in images),
            OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = self._call("RecognizeBatch", self._stub.RecognizeBatch, req, timeout=timeout)
        return parse_batch_response(resp)

    def recognize_pdf(
        self,
        pdf: ImageInput,
        *,
        dpi: int | None = None,
        mode: PdfMode | str | None = None,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> PdfResponse:
        """gRPC equivalent of [`Client.recognize_pdf`][turboocr.Client.recognize_pdf].

        Passing `reading_order=True` raises
        [`InvalidParameter`][turboocr.InvalidParameter] — the gRPC proto lacks
        the field. Use the HTTP client for PDFs that need reading order.
        """
        req = build_recognize_pdf_request(
            read_image_bytes(pdf),
            dpi=dpi,
            mode=mode,
            opts=OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = self._call("RecognizePDF", self._stub.RecognizePDF, req, timeout=timeout)
        return parse_pdf_response(resp)

    def health(self) -> HealthStatus:
        """Probe the gRPC server's `Health` RPC."""
        resp: pb2.HealthResponse = self._call(
            "Health", self._stub.Health, pb2.HealthRequest(), timeout=None
        )
        return HealthStatus(
            ok=True,
            status_code=200,
            body=resp.status or _HEALTH_BODY,
            body_json={"status": resp.status} if resp.status else None,
        )

    def make_searchable_pdf(
        self,
        source: ImageInput,
        *,
        dpi: int = 200,
        mode: PdfMode | str | None = None,
        font_path: str | None = None,
    ) -> bytes:
        """gRPC equivalent of
        [`Client.make_searchable_pdf`][turboocr.Client.make_searchable_pdf].
        """
        raw = read_image_bytes(source)
        if raw.startswith(b"%PDF-"):
            response: OcrResponse | PdfResponse = self.recognize_pdf(raw, dpi=dpi, mode=mode)
        else:
            response = self.recognize_image(
                raw, layout=True, reading_order=True, include_blocks=True
            )
        return _overlay(raw, response, dpi=dpi, font_path=font_path)


class _AsyncGrpcClientKwargs(TypedDict, total=False):
    target: NotRequired[str | None]
    api_key: NotRequired[str | None]
    auth_scheme: NotRequired[AuthScheme]
    secure: NotRequired[bool]
    credentials: NotRequired[grpc.ChannelCredentials | None]
    timeout: NotRequired[float]
    default_metadata: NotRequired[Mapping[str, str] | None]
    retry: NotRequired[RetryPolicy | None]
    channel_options: NotRequired[list[ChannelOption] | None]
    interceptors: NotRequired[Sequence[AsyncInterceptor] | None]
    channel: NotRequired[grpc.aio.Channel | None]


class AsyncGrpcClient(_BaseGrpcClient):
    def __init__(
        self,
        *,
        target: str | None = None,
        api_key: str | None = None,
        auth_scheme: AuthScheme = "bearer",
        secure: bool = False,
        credentials: grpc.ChannelCredentials | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        default_metadata: Mapping[str, str] | None = None,
        retry: RetryPolicy | None = None,
        channel_options: list[ChannelOption] | None = None,
        interceptors: Sequence[AsyncInterceptor] | None = None,
        channel: grpc.aio.Channel | None = None,
    ) -> None:
        super().__init__(
            target=target,
            api_key=api_key,
            auth_scheme=auth_scheme,
            timeout=timeout,
            default_metadata=default_metadata,
            retry=retry,
        )
        if channel is not None:
            self._channel = channel
            self._owns_channel = False
        else:
            self._channel = make_async_channel(
                self._target,
                secure=secure,
                credentials=credentials,
                options=channel_options,
                interceptors=interceptors,
            )
            self._owns_channel = True
        self._stub = pb2_grpc.OCRServiceStub(self._channel)  # type: ignore[no-untyped-call]

    def __repr__(self) -> str:
        return self._repr_fields("AsyncGrpcClient")

    @classmethod
    def from_env(cls, **overrides: Unpack[_AsyncGrpcClientKwargs]) -> Self:
        """Async equivalent of [`GrpcClient.from_env`][turboocr.GrpcClient.from_env]."""
        return cls(**overrides)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Async equivalent of [`GrpcClient.close`][turboocr.GrpcClient.close]."""
        if self._owns_channel:
            await self._channel.close()
        else:
            logger.debug("AsyncGrpcClient.aclose(): external channel — caller owns lifecycle")

    async def _call(
        self,
        rpc: str,
        unary: Any,
        request: Any,
        *,
        timeout: float | None,
    ) -> Any:
        per_call_timeout = timeout if timeout is not None else self._timeout

        async def attempt() -> Any:
            req_id = short_request_id()
            md = self._build_metadata(req_id)
            start = time.monotonic()
            try:
                result = await unary(request, timeout=per_call_timeout, metadata=md)
            finally:
                _log_rpc(rpc, req_id, (time.monotonic() - start) * 1000.0)
            return result

        try:
            return await execute_grpc_with_retries_async(
                policy=self._retry, rpc=rpc, attempt_send=attempt
            )
        except grpc.RpcError as exc:
            raise classify_rpc_error(exc) from exc

    async def recognize_image(
        self,
        image: ImageInput,
        *,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> OcrResponse:
        """Async gRPC equivalent of [`Client.recognize_image`][turboocr.Client.recognize_image]."""
        req = build_recognize_request(
            read_image_bytes(image),
            OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = await self._call("Recognize", self._stub.Recognize, req, timeout=timeout)
        return parse_ocr_response(resp)

    async def recognize_base64(
        self,
        b64: str,
        *,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> OcrResponse:
        """Async gRPC equivalent of
        [`Client.recognize_base64`][turboocr.Client.recognize_base64].
        """
        return await self.recognize_image(
            base64.b64decode(b64),
            layout=layout,
            reading_order=reading_order,
            include_blocks=include_blocks,
            tables=tables,
            formulas=formulas,
            timeout=timeout,
        )

    async def recognize_pixels(
        self,
        pixels: ImageInput,
        *,
        width: int,
        height: int,
        channels: int = 3,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> OcrResponse:
        """Async gRPC equivalent of
        [`Client.recognize_pixels`][turboocr.Client.recognize_pixels].
        """
        req = build_recognize_pixels_request(
            read_image_bytes(pixels),
            width=width,
            height=height,
            channels=channels,
            opts=OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = await self._call("Recognize", self._stub.Recognize, req, timeout=timeout)
        return parse_ocr_response(resp)

    async def recognize_batch(
        self,
        images: Iterable[ImageInput],
        *,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> BatchResponse:
        """Async gRPC equivalent of [`Client.recognize_batch`][turboocr.Client.recognize_batch]."""
        req = build_recognize_batch_request(
            (read_image_bytes(img) for img in images),
            OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = await self._call("RecognizeBatch", self._stub.RecognizeBatch, req, timeout=timeout)
        return parse_batch_response(resp)

    async def recognize_pdf(
        self,
        pdf: ImageInput,
        *,
        dpi: int | None = None,
        mode: PdfMode | str | None = None,
        layout: BoolParam = None,
        reading_order: BoolParam = None,
        include_blocks: BoolParam = None,
        tables: BoolParam = None,
        formulas: BoolParam = None,
        timeout: float | None = None,
    ) -> PdfResponse:
        """Async gRPC equivalent of
        [`GrpcClient.recognize_pdf`][turboocr.GrpcClient.recognize_pdf].
        """
        req = build_recognize_pdf_request(
            read_image_bytes(pdf),
            dpi=dpi,
            mode=mode,
            opts=OcrOptions(layout, reading_order, include_blocks, tables, formulas),
        )
        resp = await self._call("RecognizePDF", self._stub.RecognizePDF, req, timeout=timeout)
        return parse_pdf_response(resp)

    async def health(self) -> HealthStatus:
        """Async equivalent of [`GrpcClient.health`][turboocr.GrpcClient.health]."""
        resp: pb2.HealthResponse = await self._call(
            "Health", self._stub.Health, pb2.HealthRequest(), timeout=None
        )
        return HealthStatus(
            ok=True,
            status_code=200,
            body=resp.status or _HEALTH_BODY,
            body_json={"status": resp.status} if resp.status else None,
        )

    async def make_searchable_pdf(
        self,
        source: ImageInput,
        *,
        dpi: int = 200,
        mode: PdfMode | str | None = None,
        font_path: str | None = None,
    ) -> bytes:
        """Async equivalent of
        [`GrpcClient.make_searchable_pdf`][turboocr.GrpcClient.make_searchable_pdf].
        """
        raw = read_image_bytes(source)
        if raw.startswith(b"%PDF-"):
            response: OcrResponse | PdfResponse = await self.recognize_pdf(raw, dpi=dpi, mode=mode)
        else:
            response = await self.recognize_image(
                raw, layout=True, reading_order=True, include_blocks=True
            )
        return _overlay(raw, response, dpi=dpi, font_path=font_path)
