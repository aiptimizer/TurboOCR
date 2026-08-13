"""gRPC transport for turboocr.

Importing this subpackage requires the optional ``[grpc]`` extra. The
import-time guard below gives a clear error if ``grpc`` is missing rather
than letting a downstream ``ImportError`` surface from a generated stub.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("grpc") is None:  # pragma: no cover - import-time guard
    raise ImportError(
        "install turboocr[grpc] to use the gRPC client"
    )

from .client import AsyncGrpcClient, GrpcClient

__all__ = ["AsyncGrpcClient", "GrpcClient"]
