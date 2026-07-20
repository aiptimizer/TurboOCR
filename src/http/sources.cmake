# HTTP (drogon) route sources, grouped by build-target membership.
# Naming: *_gpu.cpp -> GPU target only, *_cpu.cpp -> CPU target only,
# no suffix -> shared (compiled into both targets).
set(TURBO_HTTP_COMMON_SRCS
    src/http/admin/capabilities_route.cpp
    src/http/admin/observability_middleware.cpp
    src/http/admin/health_route.cpp
    src/http/pdf/pdf_request.cpp
    src/http/pdf/pdf_json.cpp
    src/http/pdf/pdf_route.cpp
    src/http/image/ocr_base64_route.cpp
    src/http/image/batch/batch_common.cpp
)
set(TURBO_HTTP_GPU_SRCS
    src/http/image/image_routes_gpu.cpp
    src/http/image/raw/raw_route_gpu.cpp
    src/http/image/batch/batch_route_gpu.cpp
    src/http/image/batch/batch_support_gpu.cpp
    src/http/image/pixels/pixels_route_gpu.cpp
    src/http/image/ocr_markdown_route_gpu.cpp
    src/http/image/infer_route_gpu.cpp
    src/http/pdf/pdf_stream_route_gpu.cpp
)
set(TURBO_HTTP_CPU_SRCS
    src/http/admin/profile_route.cpp
    src/http/image/raw/raw_route_cpu.cpp
    src/http/image/pixels/pixels_route_cpu.cpp
    src/http/image/batch/batch_route_cpu.cpp
)
