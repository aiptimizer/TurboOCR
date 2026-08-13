"""gRPC error adapter — translates ``grpc.RpcError`` to the SDK's exception hierarchy.

Distinct from the top-level ``turboocr.errors`` module: this is the
transport-specific adapter that reads the ``x-error-code`` trailing-metadata
slot defined in ``proto/ocr.proto`` and maps gRPC status codes to HTTP-ish
status integers so callers can branch on ``exc.status_code`` regardless of
transport.

gRPC StatusCode → HTTP status mapping (synthetic, for parity with HTTP transport):

    UNAVAILABLE          -> 503
    DEADLINE_EXCEEDED    -> 504
    RESOURCE_EXHAUSTED   -> 429
    INVALID_ARGUMENT     -> 400
    FAILED_PRECONDITION  -> 400
    NOT_FOUND            -> 404
    UNAUTHENTICATED      -> 401
    PERMISSION_DENIED    -> 403
    UNIMPLEMENTED        -> 501
    INTERNAL / UNKNOWN   -> 500
"""

from __future__ import annotations

from typing import Final

import grpc

from ..errors import (
    _CODE_TO_EXC,
    NetworkError,
    PoolExhausted,
    ServerError,
    Timeout,
    TurboOcrError,
)

_GRPC_STATUS_TO_HTTP: Final[dict[grpc.StatusCode, int]] = {
    grpc.StatusCode.UNAVAILABLE: 503,
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
    grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
    grpc.StatusCode.INVALID_ARGUMENT: 400,
    grpc.StatusCode.FAILED_PRECONDITION: 400,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.UNAUTHENTICATED: 401,
    grpc.StatusCode.PERMISSION_DENIED: 403,
    grpc.StatusCode.UNIMPLEMENTED: 501,
    grpc.StatusCode.INTERNAL: 500,
    grpc.StatusCode.UNKNOWN: 500,
    grpc.StatusCode.ABORTED: 500,
    grpc.StatusCode.CANCELLED: 499,
    grpc.StatusCode.DATA_LOSS: 500,
    grpc.StatusCode.OUT_OF_RANGE: 400,
    grpc.StatusCode.ALREADY_EXISTS: 409,
}

# Fallback when no x-error-code is present: pick the closest SDK exception
# class from the gRPC status alone. INTERNAL/UNKNOWN -> ServerError;
# transport-level transients map to the connection family.
_GRPC_STATUS_TO_EXC: Final[dict[grpc.StatusCode, type[TurboOcrError]]] = {
    grpc.StatusCode.UNAVAILABLE: NetworkError,
    grpc.StatusCode.DEADLINE_EXCEEDED: Timeout,
    grpc.StatusCode.RESOURCE_EXHAUSTED: PoolExhausted,
    grpc.StatusCode.INTERNAL: ServerError,
    grpc.StatusCode.UNKNOWN: ServerError,
    grpc.StatusCode.DATA_LOSS: ServerError,
    grpc.StatusCode.ABORTED: ServerError,
}


def _trailing_error_code(exc: grpc.RpcError) -> str | None:
    trailers = getattr(exc, "trailing_metadata", None)
    if trailers is None:
        return None
    try:
        md = trailers()
    except Exception:
        return None
    if md is None:
        return None
    for key, value in md:
        if key.lower() == "x-error-code":
            return value if isinstance(value, str) else value.decode("utf-8", "replace")
    return None


def rpc_status_code(exc: grpc.RpcError) -> grpc.StatusCode:
    """Defensively extract a StatusCode from an RpcError.

    The grpc-python typing claims `RpcError.code()` returns a StatusCode but in
    practice the attribute can be absent (custom subclasses) or raise. Used by
    both error classification and retry-decision paths.
    """
    code_fn = getattr(exc, "code", None)
    if code_fn is None:
        return grpc.StatusCode.UNKNOWN
    try:
        result = code_fn()
    except Exception:
        return grpc.StatusCode.UNKNOWN
    if isinstance(result, grpc.StatusCode):
        return result
    return grpc.StatusCode.UNKNOWN


def _details(exc: grpc.RpcError) -> str:
    details_fn = getattr(exc, "details", None)
    if details_fn is None:
        return str(exc)
    try:
        msg = details_fn()
    except Exception:
        return str(exc)
    return msg or str(exc)


def classify_rpc_error(exc: grpc.RpcError) -> TurboOcrError:
    status = rpc_status_code(exc)
    http_status = _GRPC_STATUS_TO_HTTP.get(status, 500)
    message = _details(exc)
    error_code = _trailing_error_code(exc)

    # Explicit x-error-code wins over status-based heuristics: it carries the
    # same string the HTTP API would return inside {"error":{"code":"..."}}.
    if error_code is not None:
        exc_cls = _CODE_TO_EXC.get(error_code, ServerError)
        return exc_cls(message, code=error_code, status_code=http_status)

    # No explicit code — pick the closest SDK class from the gRPC status.
    fallback_cls = _GRPC_STATUS_TO_EXC.get(status, ServerError)
    return fallback_cls(message, code=None, status_code=http_status)
