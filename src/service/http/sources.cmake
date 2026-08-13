# HTTP (drogon) route sources.
#
# ONE list, because there is ONE server. This was two — TURBO_HTTP_COMMON_SRCS
# plus a TURBO_HTTP_CPU_SRCS "CPU target only" tail — back when a cpu- and a
# gpu-server executable each picked their own subset. Both lists have gone into
# the same add_executable since the mains merged, so the split named a
# distinction that no longer existed, and the routes in the "cpu" tail are as
# device-neutral as the rest (they are typed on InferFunc, not on any device
# type). The files it held were renamed off their _cpu suffix for the same
# reason.
set(TURBO_HTTP_SRCS
    src/service/http/admin/capabilities_route.cpp
    src/service/http/admin/observability_middleware.cpp
    src/service/http/admin/health_route.cpp
    src/service/http/admin/profile_route.cpp
    src/service/http/pdf/pdf_request.cpp
    src/service/http/pdf/pdf_json.cpp
    src/service/http/pdf/pdf_route.cpp
    # RESTORED: /ocr/stream. Deleted with src/cuda/'s HTTP layer and never
    # ported, leaving PdfJobOptions' streaming hooks with no consumer.
    src/service/http/pdf/stream_route.cpp
    src/service/http/image/ocr_base64_route.cpp
    src/service/http/image/batch/batch_common.cpp
    src/service/http/image/raw/raw_route.cpp
    src/service/http/image/pixels/pixels_route.cpp
    # RESTORED endpoints. Both were deleted with src/cuda/'s duplicate HTTP layer
    # and never ported, so /ocr/markdown and /infer 404'd on every backend. They
    # are device-agnostic (InferFunc / InferOneFunc).
    src/service/http/image/markdown_route.cpp
    src/service/http/image/infer_route.cpp
    # The old batch_route_cpu.cpp (typed on the deleted CpuPipelinePool) is
    # replaced by src/service/http/unified_routes.cpp, typed on UnifiedPipelinePool.
)
