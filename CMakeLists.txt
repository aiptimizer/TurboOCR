cmake_minimum_required(VERSION 3.20)
project(turbo_ocr VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Per-target optimization flags (applied below instead of global CMAKE_CXX_FLAGS)
option(NATIVE_ARCH "Use -march=native (faster but not portable)" ON)
if(NATIVE_ARCH)
  set(TURBO_OCR_CXX_FLAGS -O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter)
else()
  set(TURBO_OCR_CXX_FLAGS -O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

option(USE_CPU_ONLY "Build CPU-only mode with ONNX Runtime" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
# Native builds have nothing in models/ (repo excludes the binaries), so
# fetch the PP-OCRv5 bundle from the pinned GitHub Release after build.
# Off-by-default for Docker, where fetch_release_models.sh runs as its own
# RUN layer; the container Dockerfiles set -DFETCH_MODELS=OFF explicitly.
option(FETCH_MODELS "After build, fetch OCR model bundles into ./models/" ON)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Drogon CONFIG REQUIRED)

# c-ares is a transitive dependency of gRPC's cmake config.  The apt
# package provides only pkg-config, not cmake config, so we create the
# imported target manually if cmake can't find it.
find_package(c-ares CONFIG QUIET)
if(NOT c-ares_FOUND AND NOT TARGET c-ares::cares)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(_CARES QUIET libcares)
    endif()
    add_library(c-ares::cares INTERFACE IMPORTED)
    if(_CARES_FOUND)
        set_target_properties(c-ares::cares PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_CARES_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${_CARES_LINK_LIBRARIES}")
    else()
        find_library(_CARES_LIB cares)
        if(_CARES_LIB)
            set_target_properties(c-ares::cares PROPERTIES
                INTERFACE_LINK_LIBRARIES "${_CARES_LIB}")
        endif()
    endif()
endif()

# --- Shared library: turbo_ocr_common ---
add_library(turbo_ocr_common STATIC
    src/detection/det_postprocess.cpp
    src/recognition/crop_utils.cpp
    src/recognition/ctc_decode.cpp
    src/decode/fast_png_decoder.cpp
    third_party/clipper/clipper.cpp
    third_party/simdutf/simdutf.cpp
)
target_include_directories(turbo_ocr_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party/simdutf>
)
target_link_libraries(turbo_ocr_common PUBLIC ${OpenCV_LIBS})
target_compile_options(turbo_ocr_common PRIVATE ${TURBO_OCR_CXX_FLAGS})
add_library(turbo_ocr::common ALIAS turbo_ocr_common)

# --- C++ unit tests (Catch2, always built) ---
add_executable(turbo_ocr_tests
    tests/cpp/test_box.cpp
    tests/cpp/test_encoding.cpp
    tests/cpp/test_serialization.cpp
    tests/cpp/test_language_paths.cpp
    tests/cpp/test_pdf_renderer_liveness.cpp
    tests/cpp/test_ctc_decode.cpp
    tests/cpp/test_perspective.cpp
    tests/cpp/test_det_postprocess.cpp
    src/render/pdf_renderer.cpp
    third_party/catch2/catch_amalgamated.cpp
)
target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_common)
target_include_directories(turbo_ocr_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/catch2
)
target_compile_options(turbo_ocr_tests PRIVATE ${TURBO_OCR_CXX_FLAGS})

if(USE_CPU_ONLY)
    message(STATUS "Building CPU-ONLY mode (ONNX Runtime backend)")

    find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib)
    if(NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR "ONNX Runtime not found")
    endif()

    # Protobuf + gRPC (same discovery logic as GPU build)
    find_package(Protobuf REQUIRED)
    find_package(gRPC CONFIG QUIET)
    if(NOT gRPC_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GRPC REQUIRED grpc++ grpc)
        set(GRPC_LIBRARIES ${GRPC_LINK_LIBRARIES})
        find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
    else()
        set(GRPC_LIBRARIES gRPC::grpc++)
        get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)
    endif()

    # Proto codegen
    set(PROTO_SRC_DIR "${CMAKE_SOURCE_DIR}/proto")
    set(PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/proto_gen")
    file(MAKE_DIRECTORY ${PROTO_GEN_DIR})
    set(PROTO_FILE "${PROTO_SRC_DIR}/ocr.proto")
    set(PROTO_GEN_CC "${PROTO_GEN_DIR}/ocr.pb.cc")
    set(PROTO_GEN_H "${PROTO_GEN_DIR}/ocr.pb.h")
    set(GRPC_GEN_CC "${PROTO_GEN_DIR}/ocr.grpc.pb.cc")
    set(GRPC_GEN_H "${PROTO_GEN_DIR}/ocr.grpc.pb.h")

    add_custom_command(
        OUTPUT ${PROTO_GEN_CC} ${PROTO_GEN_H} ${GRPC_GEN_CC} ${GRPC_GEN_H}
        COMMAND protobuf::protoc
            --proto_path=${PROTO_SRC_DIR}
            --cpp_out=${PROTO_GEN_DIR}
            --grpc_out=${PROTO_GEN_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating protobuf/gRPC sources (CPU)"
    )

    # --- CPU library ---
    add_library(turbo_ocr_cpu STATIC
        src/engine/cpu_engine.cpp
        src/detection/cpu_paddle_det.cpp
        src/recognition/cpu_paddle_rec.cpp
        src/classification/cpu_paddle_cls.cpp
        src/pipeline/cpu_ocr_pipeline.cpp
        src/layout/cpu_paddle_layout.cpp
        src/render/pdf_renderer.cpp
        src/pdf/pdf_text_layer.cpp
    )
    target_include_directories(turbo_ocr_cpu PUBLIC
        ${CMAKE_SOURCE_DIR}/third_party/pdfium/include
    )
    target_link_libraries(turbo_ocr_cpu PUBLIC turbo_ocr_common ${ONNXRUNTIME_LIB}
        ${CMAKE_SOURCE_DIR}/third_party/pdfium/lib/libpdfium.so pthread)
    target_compile_options(turbo_ocr_cpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    target_compile_definitions(turbo_ocr_cpu PUBLIC USE_CPU_ONLY)
    add_library(turbo_ocr::cpu ALIAS turbo_ocr_cpu)

    # CPU HTTP + gRPC server
    add_executable(paddle_cpu_server
        src/server/cpu_main.cpp
        src/routes/common_routes.cpp
        src/routes/pdf_routes.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(paddle_cpu_server PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(paddle_cpu_server PRIVATE
        turbo_ocr_cpu
        Drogon::Drogon
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

else()
    message(STATUS "Building GPU mode (CUDA + TensorRT)")

    find_package(CUDAToolkit REQUIRED)

    set(TENSORRT_DIR "/usr/local/tensorrt" CACHE PATH "Path to TensorRT")

    # Protobuf + gRPC
    find_package(Protobuf REQUIRED)
    find_package(gRPC CONFIG QUIET)
    if(NOT gRPC_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GRPC REQUIRED grpc++ grpc)
        set(GRPC_LIBRARIES ${GRPC_LINK_LIBRARIES})
        find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
    else()
        set(GRPC_LIBRARIES gRPC::grpc++)
        get_target_property(GRPC_CPP_PLUGIN gRPC::grpc_cpp_plugin LOCATION)
    endif()

    # Proto codegen
    set(PROTO_SRC_DIR "${CMAKE_SOURCE_DIR}/proto")
    set(PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/proto_gen")
    file(MAKE_DIRECTORY ${PROTO_GEN_DIR})
    set(PROTO_FILE "${PROTO_SRC_DIR}/ocr.proto")
    set(PROTO_GEN_CC "${PROTO_GEN_DIR}/ocr.pb.cc")
    set(PROTO_GEN_H "${PROTO_GEN_DIR}/ocr.pb.h")
    set(GRPC_GEN_CC "${PROTO_GEN_DIR}/ocr.grpc.pb.cc")
    set(GRPC_GEN_H "${PROTO_GEN_DIR}/ocr.grpc.pb.h")

    add_custom_command(
        OUTPUT ${PROTO_GEN_CC} ${PROTO_GEN_H} ${GRPC_GEN_CC} ${GRPC_GEN_H}
        COMMAND protobuf::protoc
            --proto_path=${PROTO_SRC_DIR}
            --cpp_out=${PROTO_GEN_DIR}
            --grpc_out=${PROTO_GEN_DIR}
            --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating protobuf/gRPC sources"
    )

    # CUDA
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;120")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --expt-relaxed-constexpr")

    # --- GPU library ---
    add_library(turbo_ocr_gpu STATIC
        src/engine/trt_engine.cpp
        src/engine/onnx_to_trt.cpp
        src/detection/paddle_det.cpp
        src/recognition/paddle_rec.cpp
        src/classification/paddle_cls.cpp
        src/layout/paddle_layout.cpp
        src/pipeline/ocr_pipeline.cpp
        src/kernels/kernels.cu
        src/render/pdf_renderer.cpp
        src/pdf/pdf_text_layer.cpp
    )
    target_include_directories(turbo_ocr_gpu PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_DIR}/include
        ${PROTO_GEN_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/pdfium/include
    )
    target_link_directories(turbo_ocr_gpu PUBLIC ${TENSORRT_DIR}/lib)
    # Link libpdfium.so directly for in-process text-layer extraction.
    # The shared lib is vendored in third_party/pdfium/lib and also placed
    # in bin/ by scripts/install_fastpdf2png.sh (so native dev + Docker
    # builds both resolve it at runtime). Docker additionally copies it to
    # /usr/lib via ldconfig in Dockerfile.gpu:49.
    target_link_libraries(turbo_ocr_gpu PUBLIC
        turbo_ocr_common
        nvinfer
        nvinfer_plugin
        nvonnxparser
        nvjpeg
        cudart
        ${CMAKE_SOURCE_DIR}/third_party/pdfium/lib/libpdfium.so
    )
    target_compile_options(turbo_ocr_gpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    add_library(turbo_ocr::gpu ALIAS turbo_ocr_gpu)

    # GPU HTTP + gRPC server (shared pipeline pool)
    add_executable(paddle_highspeed_cpp
        src/server/main.cpp
        src/routes/common_routes.cpp
        src/routes/image_routes.cpp
        src/routes/pdf_routes.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(paddle_highspeed_cpp PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(paddle_highspeed_cpp PRIVATE
        turbo_ocr_gpu
        Drogon::Drogon
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    # Legacy standalone gRPC server removed — gRPC now runs from the main
    # binary via start_grpc_server() in grpc_service.h.

    # Benchmark tools
    add_executable(grpc_bench
        tools/grpc_bench.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(grpc_bench PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(grpc_bench PRIVATE
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    add_executable(grpc_burst
        tools/grpc_burst.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(grpc_burst PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(grpc_burst PRIVATE
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )
endif()

# --- Fetch OCR model bundles for native builds ----------------------------
# Uses det.onnx as a sentinel — ninja/make skip the step once it exists.
# Docker builds explicitly pass -DFETCH_MODELS=OFF because the Dockerfile
# runs fetch_release_models.sh as its own cacheable RUN layer.
if(FETCH_MODELS)
    set(MODELS_SENTINEL "${CMAKE_SOURCE_DIR}/models/det.onnx")
    add_custom_command(
        OUTPUT ${MODELS_SENTINEL}
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/fetch_release_models.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Fetching PP-OCRv5 models from pinned GitHub Release"
        VERBATIM
    )
    add_custom_target(fetch_models ALL DEPENDS ${MODELS_SENTINEL})
    if(USE_CPU_ONLY)
        add_dependencies(paddle_cpu_server fetch_models)
    else()
        add_dependencies(paddle_highspeed_cpp fetch_models)
    endif()
endif()
