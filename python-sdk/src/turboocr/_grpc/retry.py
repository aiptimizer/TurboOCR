from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import grpc

from .._core.retry import RetryPolicy
from .errors import rpc_status_code

logger = logging.getLogger("turboocr.grpc.retry")

def should_retry_grpc_status(
    code: grpc.StatusCode,
    policy: RetryPolicy,
) -> bool:
    return policy.should_retry_grpc_status_name(code.name)


def _log_retry(attempt: int, total: int, rpc: str, status: grpc.StatusCode) -> None:
    logger.warning(
        "retrying gRPC %s (attempt %d/%d) — status %s", rpc, attempt, total, status.name
    )


def execute_grpc_with_retries[T](
    *,
    policy: RetryPolicy,
    rpc: str,
    attempt_send: Callable[[], T],
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last_exc: grpc.RpcError | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return attempt_send()
        except grpc.RpcError as exc:
            last_exc = exc
            status = rpc_status_code(exc)
            if attempt == policy.attempts or not should_retry_grpc_status(status, policy):
                raise
            _log_retry(attempt, policy.attempts, rpc, status)
            sleep(policy.delay_for(attempt))
    assert last_exc is not None
    raise last_exc


async def execute_grpc_with_retries_async[T](
    *,
    policy: RetryPolicy,
    rpc: str,
    attempt_send: Callable[[], Awaitable[T]],
) -> T:
    last_exc: grpc.RpcError | None = None
    for attempt in range(1, policy.attempts + 1):
        try:
            return await attempt_send()
        except grpc.RpcError as exc:
            last_exc = exc
            status = rpc_status_code(exc)
            if attempt == policy.attempts or not should_retry_grpc_status(status, policy):
                raise
            _log_retry(attempt, policy.attempts, rpc, status)
            await asyncio.sleep(policy.delay_for(attempt))
    assert last_exc is not None
    raise last_exc
