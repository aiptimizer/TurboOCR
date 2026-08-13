from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Final

import httpx

from .._core.retry import RetryPolicy, log_retry

RETRYABLE_EXC: Final[tuple[type[BaseException], ...]] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


# Retries happen at the SDK layer (here), not as an httpx transport wrapper.
# That's deliberate: each attempt re-invokes `attempt_send`, which rebuilds
# the request kwargs — including freshly invoking any body factory. Streamed
# Path inputs therefore replay correctly across retries, which a transport
# wrapper cannot do (httpx exhausts the body iterable on first send).


def execute_with_retries(
    *,
    policy: RetryPolicy,
    method: str,
    path: str,
    attempt_send: Callable[[], httpx.Response],
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    last_exc: BaseException | None = None
    method_retryable = policy.should_retry_method(method)
    for attempt in range(1, policy.attempts + 1):
        try:
            response = attempt_send()
        except RETRYABLE_EXC as exc:
            last_exc = exc
            if attempt == policy.attempts or not method_retryable:
                raise
            log_retry(attempt, policy.attempts, method, path, exc=exc)
            sleep(policy.delay_for(attempt))
            continue

        if (
            not method_retryable
            or not policy.should_retry_status(response.status_code)
            or attempt == policy.attempts
        ):
            return response
        delay = policy.delay_with_retry_after(attempt, response.headers.get("Retry-After"))
        response.close()
        log_retry(attempt, policy.attempts, method, path, status=response.status_code)
        sleep(delay)
    assert last_exc is not None
    raise last_exc


async def execute_with_retries_async(
    *,
    policy: RetryPolicy,
    method: str,
    path: str,
    attempt_send: Callable[[], Awaitable[httpx.Response]],
) -> httpx.Response:
    last_exc: BaseException | None = None
    method_retryable = policy.should_retry_method(method)
    for attempt in range(1, policy.attempts + 1):
        try:
            response = await attempt_send()
        except RETRYABLE_EXC as exc:
            last_exc = exc
            if attempt == policy.attempts or not method_retryable:
                raise
            log_retry(attempt, policy.attempts, method, path, exc=exc)
            await asyncio.sleep(policy.delay_for(attempt))
            continue

        if (
            not method_retryable
            or not policy.should_retry_status(response.status_code)
            or attempt == policy.attempts
        ):
            return response
        delay = policy.delay_with_retry_after(attempt, response.headers.get("Retry-After"))
        await response.aclose()
        log_retry(attempt, policy.attempts, method, path, status=response.status_code)
        await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
