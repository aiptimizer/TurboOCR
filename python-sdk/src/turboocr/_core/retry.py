from __future__ import annotations

import email.utils
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Final

logger = logging.getLogger("turboocr.retry")

DEFAULT_RETRY_STATUSES: Final[frozenset[int]] = frozenset({429, 502, 503, 504})
DEFAULT_RETRY_METHODS: Final[frozenset[str]] = frozenset({"GET", "POST"})

# gRPC status names retried by default — keyed by name (not grpc.StatusCode)
# to keep `_core` transport-agnostic. The gRPC retry executor converts to
# StatusCode at use site. INTERNAL is deliberately excluded — it signals a
# server-side bug, not a transient blip.
DEFAULT_RETRY_GRPC_STATUSES: Final[frozenset[str]] = frozenset({
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "RESOURCE_EXHAUSTED",
})


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff retry policy for transient HTTP/gRPC failures.

    Each retry attempt waits `min(backoff_cap, backoff * 2 ** (attempt - 1))
    + random.uniform(0, jitter)` seconds before firing. When the server
    sends a `Retry-After` header (HTTP) and `respect_retry_after` is `True`,
    the effective delay is `max(backoff, retry_after)` — server hints
    extend backoff but never shorten it.

    Note: retries replay the whole request, including the body. The server
    does not support range/resume uploads, so a 502 at the tail end of a
    large PDF upload triggers a full re-upload on the next attempt. For
    very large payloads over unstable links, keep `attempts` low and let
    your own pipeline manage the retry budget.

    Args:
        attempts: Total request attempts before giving up. `1` disables
            retries entirely. Default `3` (one initial + two retries).
        backoff: Base delay in seconds for the first retry. Doubles each
            subsequent attempt up to `backoff_cap`.
        backoff_cap: Maximum exponential delay (seconds) before jitter
            is added. Caps the geometric growth.
        jitter: Upper bound (seconds) of uniform random jitter added to
            each delay. Prevents thundering-herd reconnects.
        retry_statuses: HTTP status codes considered transient. Defaults
            to `{429, 502, 503, 504}`.
        retry_methods: HTTP methods eligible for retry. Defaults to
            `{"GET", "POST"}`; OCR uploads are POST.
        retry_grpc_statuses: gRPC status names eligible for retry.
            Defaults to `{"UNAVAILABLE", "DEADLINE_EXCEEDED",
            "RESOURCE_EXHAUSTED"}`. `INTERNAL` is deliberately excluded —
            it signals a server-side bug, not a transient blip.
        respect_retry_after: When `True`, honour `Retry-After` headers as
            a lower bound on the next attempt's delay. Disable to keep
            client-driven backoff regardless of server hints.

    Example:
        ```python
        from turboocr import Client, RetryPolicy

        policy = RetryPolicy(attempts=5, backoff=0.5, backoff_cap=10.0)
        client = Client(retry=policy)
        ```
    """

    attempts: int = 3
    backoff: float = 0.25
    backoff_cap: float = 8.0
    jitter: float = 0.1
    retry_statuses: frozenset[int] = DEFAULT_RETRY_STATUSES
    retry_methods: frozenset[str] = field(default_factory=lambda: DEFAULT_RETRY_METHODS)
    retry_grpc_statuses: frozenset[str] = field(
        default_factory=lambda: DEFAULT_RETRY_GRPC_STATUSES
    )
    respect_retry_after: bool = True

    def should_retry_status(self, code: int) -> bool:
        return code in self.retry_statuses

    def should_retry_method(self, method: str) -> bool:
        return method.upper() in self.retry_methods

    def should_retry_grpc_status_name(self, status_name: str) -> bool:
        return status_name in self.retry_grpc_statuses

    def delay_for(self, attempt: int) -> float:
        base: float = min(self.backoff_cap, self.backoff * (2 ** (attempt - 1)))
        return base + random.uniform(0.0, self.jitter)

    def delay_with_retry_after(self, attempt: int, retry_after_header: str | None) -> float:
        backoff = self.delay_for(attempt)
        if not self.respect_retry_after or not retry_after_header:
            return backoff
        retry_after = parse_retry_after(retry_after_header)
        return max(backoff, retry_after) if retry_after is not None else backoff


def parse_retry_after(header: str) -> float | None:
    s = header.strip()
    if not s:
        return None
    if s.isdigit():
        return float(s)
    try:
        parsed = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "malformed Retry-After header %r (%s); using default backoff", header, exc
        )
        return None
    return max(0.0, parsed.timestamp() - time.time())


def log_retry(
    attempt: int,
    total: int,
    method: str,
    path: str,
    *,
    status: int | None = None,
    exc: BaseException | None = None,
) -> None:
    reason = f"status {status}" if status is not None else f"{type(exc).__name__}: {exc}"
    logger.warning(
        "retrying %s %s (attempt %d/%d) — %s", method, path, attempt, total, reason
    )
