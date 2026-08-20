__version__ = "4.0.0a7"
SERVER_API_VERSION_MIN = "3.1.0"
SERVER_API_VERSION_MAX_EXCLUSIVE = "5.0.0"

from ._core.retry import RetryPolicy  # noqa: E402
from ._http.client import AsyncClient, Client  # noqa: E402
from .errors import (  # noqa: E402
    APIConnectionError,
    BackendDisabled,
    DimensionsTooLarge,
    EmptyBody,
    ImageDecodeError,
    InferenceTimeout,
    InvalidParameter,
    LayoutDisabled,
    NetworkError,
    PdfRenderError,
    PoolExhausted,
    ProtocolError,
    ServerError,
    Timeout,
    TurboOcrError,
)
from .markdown import (  # noqa: E402
    MarkdownDocument,
    MarkdownNode,
    MarkdownStyle,
    NodeKind,
    render_to_markdown,
)
from .models import (  # noqa: E402
    BatchFailure,
    BatchResponse,
    BatchResult,
    BatchSuccess,
    Block,
    BoundingBox,
    Capabilities,
    CapabilityFeatures,
    CapabilityLimits,
    CapabilityPdf,
    Formula,
    HealthStatus,
    LayoutBox,
    LayoutLabel,
    MarkdownPage,
    MarkdownPagesResponse,
    OcrResponse,
    PdfMode,
    PdfPage,
    PdfResponse,
    StreamEvent,
    Table,
    TextItem,
)


def supports_server_version(server_version: str) -> bool:
    def _parse(v: str) -> tuple[int, ...]:
        try:
            return tuple(int(p) for p in v.split("."))
        except ValueError as exc:
            raise InvalidParameter(f"invalid version: {v!r}") from exc

    version = _parse(server_version)
    return _parse(SERVER_API_VERSION_MIN) <= version < _parse(SERVER_API_VERSION_MAX_EXCLUSIVE)


__all__ = [
    "SERVER_API_VERSION_MAX_EXCLUSIVE",
    "SERVER_API_VERSION_MIN",
    "APIConnectionError",
    "AsyncClient",
    "BackendDisabled",
    "BatchFailure",
    "BatchResponse",
    "BatchResult",
    "BatchSuccess",
    "Block",
    "BoundingBox",
    "Capabilities",
    "CapabilityFeatures",
    "CapabilityLimits",
    "CapabilityPdf",
    "Client",
    "DimensionsTooLarge",
    "EmptyBody",
    "Formula",
    "HealthStatus",
    "ImageDecodeError",
    "InferenceTimeout",
    "InvalidParameter",
    "LayoutBox",
    "LayoutDisabled",
    "LayoutLabel",
    "MarkdownDocument",
    "MarkdownNode",
    "MarkdownPage",
    "MarkdownPagesResponse",
    "MarkdownStyle",
    "NetworkError",
    "NodeKind",
    "OcrResponse",
    "PdfMode",
    "PdfPage",
    "PdfRenderError",
    "PdfResponse",
    "PoolExhausted",
    "ProtocolError",
    "RetryPolicy",
    "ServerError",
    "StreamEvent",
    "Table",
    "TextItem",
    "Timeout",
    "TurboOcrError",
    "__version__",
    "render_to_markdown",
    "supports_server_version",
]

# pypdf + reportlab ship with the core install, so searchable-PDF exports
# always work. noqa F401: ruff can't see the runtime __all__.extend below.
from .searchable_pdf import (  # noqa: E402, F401
    FontError,
    FontGlyphMissing,
)

__all__.extend(["FontError", "FontGlyphMissing"])

# gRPC transport is gated behind the [grpc] extra (grpcio + protobuf). We
# probe with find_spec so `import turboocr` succeeds even without
# grpc installed; the actual import of generated stubs happens lazily.
import importlib.util as _importlib_util  # noqa: E402

if _importlib_util.find_spec("grpc") is not None:
    from ._grpc.client import AsyncGrpcClient, GrpcClient  # noqa: F401

    __all__.extend(["AsyncGrpcClient", "GrpcClient"])


# ---------------------------------------------------------------------------
# Embedded engine (optional)
# ---------------------------------------------------------------------------
# `turboocr` is pure Python and installs everywhere — it has always been, and
# remains, a CLIENT for a running TurboOCR server. Since v4 it is ALSO the
# front door to the in-process native engine, which ships as a separate
# platform wheel (`turboocr-engine-cpu` / `-cuda` / `-openvino` / `-rocm`,
# selected by the extras in pyproject.toml). Keeping them as two distributions
# is deliberate: a client must stay installable on any Python and any OS, and
# it must not drag ~1 GB of CUDA runtime onto a laptop that only wants to POST
# an image to a server.
#
# When an engine wheel is present we re-export its API here, so one import
# serves both modes:
#
#     import turboocr
#     turboocr.Client("http://server:8000").ocr("page.png")   # remote
#     turboocr.OCR().read("page.png")                         # in-process
#
# NAME COLLISION, resolved deliberately: both packages define `LayoutBox`
# (client = the server's JSON model, engine = the native result type). The
# CLIENT's stays bound at top level so no existing import changes meaning;
# the engine's is reachable as `turboocr.engine.LayoutBox`.
try:  # noqa: E402
    import turboocr_engine as engine  # noqa: F401

    from turboocr_engine import (  # noqa: E402, F401
        DEFAULT_MODEL,
        OCR,
        BackendUnavailable,
        DocumentResult,
        FormulaRegion,
        ModelLoadError,
        NativeExtensionMissing,
        PageResult,
        TableRegion,
        TextLine,
        TurboOCRError,
        available_backends,
        doctor,
        list_models,
        model_catalog,
        read,
        read_pdf,
        resolve_model,
    )

    __all__.extend([
        "DEFAULT_MODEL", "OCR", "BackendUnavailable", "DocumentResult",
        "FormulaRegion", "ModelLoadError", "NativeExtensionMissing",
        "PageResult", "TableRegion", "TextLine", "TurboOCRError",
        "available_backends", "doctor", "engine", "list_models",
        "model_catalog", "read", "read_pdf", "resolve_model",
    ])
    HAS_ENGINE = True
except ImportError:  # pragma: no cover - depends on what is installed
    engine = None
    HAS_ENGINE = False

    def _engine_missing(name: str):
        def _raise(*_a, **_k):
            raise ImportError(
                f"turboocr.{name} needs the in-process engine, which ships as a "
                "separate wheel. Install one for your hardware:\n"
                "    pip install 'turboocr[cpu]'      # any machine\n"
                "    pip install 'turboocr[cuda12]'   # NVIDIA, driver R525+\n"
                "    pip install 'turboocr[cuda13]'   # NVIDIA, driver R580+\n"
                "    pip install 'turboocr[openvino]' # Intel\n"
                "    pip install 'turboocr[rocm]'     # AMD\n"
                "Or keep using the client against a running server: "
                "turboocr.Client(...)."
            )
        return _raise

    # Bound as callables that fail with an actionable message, rather than
    # raising AttributeError from a bare namespace — the failure should say
    # what to install, not just that the name is absent.
    OCR = _engine_missing("OCR")          # type: ignore[assignment]
    read = _engine_missing("read")        # type: ignore[assignment]
    read_pdf = _engine_missing("read_pdf")  # type: ignore[assignment]
    doctor = _engine_missing("doctor")    # type: ignore[assignment]
    __all__.extend(["OCR", "read", "read_pdf", "doctor", "engine", "HAS_ENGINE"])

__all__.append("HAS_ENGINE")
