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

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

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
    tests/cpp/test_ctc_decode.cpp
    tests/cpp/test_perspective.cpp
    tests/cpp/test_det_postprocess.cpp
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

    # --- CPU library ---
    add_library(turbo_ocr_cpu STATIC
        src/engine/cpu_engine.cpp
        src/detection/cpu_paddle_det.cpp
        src/recognition/cpu_paddle_rec.cpp
        src/classification/cpu_paddle_cls.cpp
        src/pipeline/cpu_ocr_pipeline.cpp
    )
    target_link_libraries(turbo_ocr_cpu PUBLIC turbo_ocr_common ${ONNXRUNTIME_LIB} pthread)
    target_compile_options(turbo_ocr_cpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    add_library(turbo_ocr::cpu ALIAS turbo_ocr_cpu)

    # CPU HTTP server
    add_executable(paddle_cpu_server src/server/cpu_main.cpp)
    target_link_libraries(paddle_cpu_server PRIVATE turbo_ocr_cpu)

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
        src/pipeline/ocr_pipeline.cpp
        src/kernels/kernels.cu
        src/render/pdf_renderer.cpp
    )
    target_include_directories(turbo_ocr_gpu PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_DIR}/include
        ${PROTO_GEN_DIR}
    )
    target_link_directories(turbo_ocr_gpu PUBLIC ${TENSORRT_DIR}/lib)
    target_link_libraries(turbo_ocr_gpu PUBLIC
        turbo_ocr_common
        nvinfer
        nvinfer_plugin
        nvonnxparser
        nvjpeg
        cudart
    )
    target_compile_options(turbo_ocr_gpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    add_library(turbo_ocr::gpu ALIAS turbo_ocr_gpu)

    # GPU HTTP + gRPC server (shared pipeline pool)
    add_executable(paddle_highspeed_cpp
        src/server/main.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(paddle_highspeed_cpp PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(paddle_highspeed_cpp PRIVATE
        turbo_ocr_gpu
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    # gRPC server
    add_executable(paddle_grpc_server
        src/server/grpc_server.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(paddle_grpc_server PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(paddle_grpc_server PRIVATE
        turbo_ocr_gpu
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

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
