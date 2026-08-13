# src/service/server/unified/unified_server.cmake — the ONE multi-backend TurboOCR server.
#
# server_main.cpp is VENDOR-NEUTRAL: it only ever calls backend::make_backend().
# Which vendors it can select among is a LINK-time choice, made below by
# force-linking every turbo_ocr_backend_<b> named in TURBO_BACKENDS.
#
# Included from the root CMakeLists.txt in the CPU configure (it reuses
# turbo_ocr_cpu and the root's proto codegen variables, which share this
# directory scope because include() does not create a new one).

# ---------------------------------------------------------------------------
# PDF availability — DECOMPOSED into the facts that are actually independent.
#
# The old single `if(NOT APPLE)` conflated three unrelated things and so lost
# ALL of PDF whenever any one of them was missing:
#
#   1. PDFIUM      — the library. Vendored for linux AND mac-arm64
#                    (scripts/setup/install_pdfium.sh); the old check looked only for
#                    libpdfium.so, so macOS "had no pdfium" while the .dylib
#                    sat right there. Gates: text layer, mode=auto_verified,
#                    searchable-PDF output.
#   2. TURBOJPEG   — page-image export only.
#   3. INOTIFY etc — pdf_daemon/pdf_renderer ONLY. Gates rasterizing a scanned
#                    PDF, i.e. the OCR-a-PDF path.
#
# Decomposed, macOS gets everything except rasterization: a text-layer PDF is
# extracted, verified and written back as a searchable PDF, natively.
# ---------------------------------------------------------------------------
option(TURBO_ENABLE_PDF "Build the PDF subsystem into the unified server" ON)

set(TURBO_PDF_RENDER_AVAILABLE ${TURBO_HAVE_PDF_RENDER})
# TURBO_HAVE_PDF_RENDER is now ON in every arm of the root list — the no-inotify
# case compiles the in-process renderer and still sets it ON — so the only way
# the renderer is absent is an explicit build-time opt-out. The old
# `elseif(NOT TURBO_HAVE_PDF_RENDER)` arm (blaming missing inotify) was
# unreachable and has been removed; the 501 body it fed said the same wrong thing.
if(NOT TURBO_ENABLE_PDF)
    set(TURBO_PDF_RENDER_AVAILABLE OFF)
    set(_pdf_why "disabled by -DTURBO_ENABLE_PDF=OFF")
endif()

if(TURBO_PDF_RENDER_AVAILABLE)
    message(STATUS "unified server: PDF fully ENABLED (render + text + searchable)")
else()
    message(STATUS "unified server: PDF RENDERER disabled — ${_pdf_why}. "
                   "Text layer / auto_verified / searchable-PDF remain ENABLED; "
                   "only requests that must RASTERIZE a page answer 501.")
endif()

# ---------------------------------------------------------------------------
# Main-tree server TUs this target reuses (the CPU server's sources minus its
# main and stage bootstrap, which server_main.cpp / backend_stages.cpp replace).
# ---------------------------------------------------------------------------
set(_us_main_srcs
    src/service/server/bootstrap/server_config.cpp
    src/pipeline/job/pdf_job.cpp
    src/pipeline/job/pdf_job_pages.cpp
    # Remote-VLM transport: device-agnostic; src/pipeline/unified/vlm_factory.cpp needs it.
    src/pipeline/finalize_deferred.cpp
    src/analysis/vlm/vlm_client.cpp
    src/analysis/vlm/crop_pool.cpp
    src/analysis/vlm/crop_pool_transport.cpp
    src/analysis/table/vlm/otsl_html.cpp
)

add_executable(turboocr-server
    src/service/server/unified/server_main.cpp
    src/service/server/unified/backend_stages.cpp
    src/service/http/unified_routes.cpp
    ${_us_main_srcs}
    ${TURBO_HTTP_SRCS}
    ${TURBO_GRPC_SRCS}
    ${PROTO_GEN_CC}
    ${GRPC_GEN_CC}
)
if(NOT TURBO_PDF_RENDER_AVAILABLE)
    # Supplies ONLY the PdfRenderer definitions (see the file): every other
    # pdf::* symbol is compiled for real into turbo_ocr_cpu on this platform.
    target_sources(turboocr-server PRIVATE src/service/server/unified/pdf_unavailable.cpp)
endif()

# NOTE (removed): there used to be a set_source_files_properties() here putting
# src/server/compat (a fake <cuda_runtime.h>) on the include path of
# src/analysis/table/vlm/otsl_html.cpp and src/pipeline/finalize_deferred.cpp. Both
# are device-free and neither reaches a CUDA type any more: otsl_html.cpp now
# includes the device-free turbo_ocr/analysis/table/vlm/otsl.h instead of vlm_table.h, and
# finalize_deferred.cpp's two unused CUDA-typed seam includes were deleted.
# src/server/compat/ is gone with them.

target_include_directories(turboocr-server PRIVATE
    "${PROTO_GEN_DIR}"
    "${CMAKE_SOURCE_DIR}/third_party/cli11"
)
target_link_libraries(turboocr-server PRIVATE
    turbo_ocr_pipeline          # the seam + turbo_ocr_common
    # ORT + the main-tree host stages. NOT the literal turbo_ocr_cpu: that
    # target only exists in the CPU configure. The GPU configure carries the
    # same classes in turbo_ocr_cpu_host, and TURBO_CPU_HOST_LIB (set in the
    # root list) names whichever one this configure built.
    ${TURBO_CPU_HOST_LIB}
    Drogon::Drogon
    ${GRPC_LIBRARIES}
    protobuf::libprotobuf
    ${OpenCV_LIBS}
)
turbo_link_backends(turboocr-server ${TURBO_BACKENDS})
find_package(CURL QUIET)
if(TARGET CURL::libcurl)
    target_link_libraries(turboocr-server PRIVATE CURL::libcurl)
else()
    target_link_libraries(turboocr-server PRIVATE curl)
endif()
target_compile_options(turboocr-server PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
if(TARGET turbo_apple_metallib)
    add_dependencies(turboocr-server turbo_apple_metallib)
endif()
if(TARGET fetch_models)
    add_dependencies(turboocr-server fetch_models)
endif()
if(_ipo_ok AND CMAKE_BUILD_TYPE STREQUAL "Release")
    set_property(TARGET turboocr-server PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
install(TARGETS turboocr-server RUNTIME DESTINATION bin)
