from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Final

import grpc
import grpc.aio

DEFAULT_GRPC_TARGET: Final[str] = "localhost:50051"

type ChannelOption = tuple[str, int | str]
type SyncInterceptor = (
    grpc.UnaryUnaryClientInterceptor
    | grpc.UnaryStreamClientInterceptor
    | grpc.StreamUnaryClientInterceptor
    | grpc.StreamStreamClientInterceptor
)
type AsyncInterceptor = grpc.aio.ClientInterceptor

# 64 MB ceilings — PDFs and pixel buffers can be tens of MB, and json_response
# can fan out beyond the 4 MB gRPC default for dense pages.
_MAX_MSG_BYTES: Final[int] = 64 * 1024 * 1024

_DEFAULT_OPTIONS: Final[tuple[ChannelOption, ...]] = (
    ("grpc.max_receive_message_length", _MAX_MSG_BYTES),
    ("grpc.max_send_message_length", _MAX_MSG_BYTES),
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_permit_without_calls", 1),
)


def resolve_grpc_target(target: str | None) -> str:
    if target is not None:
        return target
    return os.environ.get("TURBO_OCR_GRPC_TARGET", DEFAULT_GRPC_TARGET)


def build_channel_options(
    extra: list[ChannelOption] | None = None,
) -> list[ChannelOption]:
    merged: dict[str, int | str] = {k: v for k, v in _DEFAULT_OPTIONS}
    if extra:
        for k, v in extra:
            merged[k] = v
    return list(merged.items())


def make_channel(
    target: str,
    *,
    secure: bool = False,
    credentials: grpc.ChannelCredentials | None = None,
    options: list[ChannelOption] | None = None,
    interceptors: Sequence[SyncInterceptor] | None = None,
) -> grpc.Channel:
    opts = build_channel_options(options)
    if secure:
        creds = credentials if credentials is not None else grpc.ssl_channel_credentials()
        channel: grpc.Channel = grpc.secure_channel(target, creds, options=opts)
    else:
        channel = grpc.insecure_channel(target, options=opts)
    if interceptors:
        # grpc.intercept_channel wraps the channel; later interceptors run inside earlier ones.
        channel = grpc.intercept_channel(channel, *interceptors)
    return channel


def make_async_channel(
    target: str,
    *,
    secure: bool = False,
    credentials: grpc.ChannelCredentials | None = None,
    options: list[ChannelOption] | None = None,
    interceptors: Sequence[AsyncInterceptor] | None = None,
) -> grpc.aio.Channel:
    opts = build_channel_options(options)
    # Async channels take interceptors at construction; the sync API wraps
    # post-construction via grpc.intercept_channel instead.
    interceptor_arg = list(interceptors) if interceptors else None
    if secure:
        creds = credentials if credentials is not None else grpc.ssl_channel_credentials()
        return grpc.aio.secure_channel(target, creds, options=opts, interceptors=interceptor_arg)
    return grpc.aio.insecure_channel(target, options=opts, interceptors=interceptor_arg)
