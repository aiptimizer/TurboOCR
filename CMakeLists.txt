cmake_minimum_required(VERSION 3.20)

# Default to Release when the user didn't pick a build type. Single-config
# generators (Ninja/Make) leave CMAKE_BUILD_TYPE empty otherwise, which gives
# unoptimized builds.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Defaulting CMAKE_BUILD_TYPE to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(turbo_ocr VERSION 1.0.0 LANGUAGES CXX)

enable_testing()

# /opt/cuda ships its own protobuf cmake config. When gRPC's
# find_dependency(Protobuf) resolves to it instead of the system install, its
# protobuf-targets.cmake tries to redefine targets the system FindProtobuf
# module already created ("Some (but not all) targets in this export set were
# already defined"). Exclude just that directory so protobuf + gRPC both come
# from one consistent (system) source; CUDAToolkit discovery is unaffected.
list(APPEND CMAKE_IGNORE_PATH "/opt/cuda/lib/cmake/protobuf")

# Protobuf discovery that works on BOTH distros we build on:
#   - Arch dev host: ships protobuf-config.cmake (v35 + libupb). CONFIG mode
#     is required there so the target set matches gRPC's config-mode protobuf
#     ("some but not all targets already defined" otherwise).
#   - Ubuntu Docker images (Dockerfile.cpu/gpu): apt libprotobuf-dev ships NO
#     cmake config, so CONFIG mode fails hard — but the legacy FindProtobuf
#     MODULE works (and gRPC there has no config either, so it uses the
#     pkg-config fallback below and never re-finds protobuf in config mode,
#     meaning no target clash). Prefer CONFIG, fall back to MODULE.
macro(turbo_find_protobuf)
  find_package(Protobuf CONFIG QUIET)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf MODULE REQUIRED)
  endif()
endmacro()

# Shared by the CPU and GPU arms: protobuf/gRPC discovery + ocr.proto codegen.
# A macro (not function) on purpose — the GRPC_LIBRARIES / PROTO_GEN_* /
# GRPC_GEN_* variables must land in the calling scope, exactly as the two
# previously hand-duplicated blocks did.
macro(turbo_setup_grpc_codegen)
  turbo_find_protobuf()
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
endmacro()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Per-target optimization flags (applied below instead of global CMAKE_CXX_FLAGS)
#
# Default OFF: -march=native tunes to the BUILD host CPU, so an image built on a
# host with newer ISA (e.g. AVX-512) SIGILLs on an older deploy CPU. Off ships a
# portable baseline — x86-64-v3 still guarantees AVX2/FMA, so the hand-written
# AVX2 in ctc_decode.cpp stays valid. Turn ON only when build host == deploy host.
option(NATIVE_ARCH "Use -march=native (fastest, but only when build host CPU == deploy host CPU)" OFF)
if(NATIVE_ARCH)
  set(_turbo_arch_flag -march=native)
else()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _turbo_baseline_proc)
  if(_turbo_baseline_proc MATCHES "aarch64|arm64")
    set(_turbo_arch_flag -march=armv8.2-a)
  else()
    set(_turbo_arch_flag -march=x86-64-v3)
  endif()
endif()
set(TURBO_OCR_CXX_FLAGS -O3 ${_turbo_arch_flag} -Wall -Wextra -Wpedantic -Wno-unused-parameter)
message(STATUS "Turbo OCR arch flags: ${_turbo_arch_flag} (NATIVE_ARCH=${NATIVE_ARCH})")

# --- Target architecture detection (x86_64 / aarch64) ----------------------
# The binaries vendored in third_party/ (ONNX Runtime, PDFium) and bin/
# (fastpdf2png) are x86_64. On aarch64 the build fetches the matching
# ONNX Runtime release and PDFium build instead of linking the x86 copies.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _turbo_proc)
if(_turbo_proc MATCHES "aarch64|arm64")
  set(TURBO_ARCH "aarch64")
  set(TURBO_ORT_URL_ARCH "aarch64")
else()
  set(TURBO_ARCH "x64")
  set(TURBO_ORT_URL_ARCH "x64")
endif()
message(STATUS "Turbo OCR target arch: ${CMAKE_SYSTEM_PROCESSOR} -> ${TURBO_ARCH}")

# PDFium location. x86_64 uses the vendored copy; aarch64 expects an arm64
# PDFium unpacked to third_party/pdfium by scripts/install_pdfium.sh (which
# pulls pdfium-linux-arm64 from bblanchon/pdfium-binaries). Both layouts are
# {include/, lib/libpdfium.so}, so only the root differs.
set(TURBO_PDFIUM_DIR "${CMAKE_SOURCE_DIR}/third_party/pdfium"
    CACHE PATH "Root of the PDFium SDK (include/ + lib/)")

# Profile-Guided Optimization. Two-pass build on the SAME build dir:
#   1. cmake -DPGO_MODE=generate && build  →  run a representative workload
#   2. cmake -DPGO_MODE=use      && build  →  recompiles using the *.gcda profile
# Scoped to host C++ only ($<COMPILE_LANGUAGE:CXX>) — CUDA .cu objects are
# compiled by nvcc and left untouched. The .gcda files live next to the object
# files, so generate and use must share one build directory.
set(PGO_MODE "off" CACHE STRING "Profile-guided optimization: off|generate|use")
set_property(CACHE PGO_MODE PROPERTY STRINGS off generate use)
# Line/branch coverage. Instruments host C++ only (same $<COMPILE_LANGUAGE:CXX>
# scoping as PGO; nvcc .cu objects are left alone). Build with it on, run the
# test suite, then aggregate the *.gcda with scripts/coverage.sh (gcovr) — any
# 0-hit branch is a path no test exercises.
option(TURBO_COVERAGE "Instrument host C++ for gcov line/branch coverage" OFF)
if(NOT PGO_MODE STREQUAL "off" OR TURBO_COVERAGE)
    # The executable is linked by the C++ compiler, but CMake prepends CUDA's
    # implicit link dirs — and since CUDA 13.2's nvcc host compiler is GCC 15,
    # that dir ships a 15.x libgcov that -lgcov would resolve before the C++
    # compiler's own. Put the actual C++ compiler's gcc lib dir first so the
    # libgcov runtime matches the instrumentation the compiler emits.
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libgcov.a
        OUTPUT_VARIABLE _gcov_a OUTPUT_STRIP_TRAILING_WHITESPACE)
    get_filename_component(_gcc_libdir "${_gcov_a}" DIRECTORY)
    add_link_options("-L${_gcc_libdir}")
endif()
if(PGO_MODE STREQUAL "generate")
    message(STATUS "PGO: instrumented build (-fprofile-generate); gcov dir ${_gcc_libdir}")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fprofile-generate>)
    add_link_options(-fprofile-generate)
elseif(PGO_MODE STREQUAL "use")
    message(STATUS "PGO: profile-optimized build (-fprofile-use); gcov dir ${_gcc_libdir}")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-use>
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-correction>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-profile>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-coverage-mismatch>)
    add_link_options(-fprofile-use)
endif()
if(TURBO_COVERAGE)
    message(STATUS "Coverage: instrumented build (--coverage); gcov dir ${_gcc_libdir}")
    # -fprofile-abs-path makes the emitted .gcno reference absolute source paths
    # so gcovr resolves them regardless of the invocation cwd. -O0 keeps line
    # mapping honest (no inlining collapsing branches away).
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:--coverage>
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-abs-path>
        $<$<COMPILE_LANGUAGE:CXX>:-O0>)
    add_link_options(--coverage)
endif()

option(USE_CPU_ONLY "Build CPU-only mode with ONNX Runtime" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
# When the fetched ONNX Runtime tarball has no pinned SHA256 (the aarch64 case),
# the build fails closed. Set this to fetch unverified anyway (CI/dev only).
option(ALLOW_UNVERIFIED_ORT "Allow fetching ONNX Runtime without SHA256 verification" OFF)
# Native builds have nothing in models/ (repo excludes the binaries), so
# fetch the PP-OCRv6 bundle from the pinned GitHub Release after build.
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

# Trantor's installed cmake targets on some distros reference a bare
# `c-ares_lib` instead of the modern `c-ares::cares` namespace. Provide a
# shim target so the linker resolves it. Harmless when Trantor is upgraded.
if(NOT TARGET c-ares_lib)
    add_library(c-ares_lib INTERFACE IMPORTED)
    find_library(_SHIM_CARES_LIB cares)
    if(_SHIM_CARES_LIB)
        set_target_properties(c-ares_lib PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_SHIM_CARES_LIB}")
    elseif(TARGET c-ares::cares)
        set_target_properties(c-ares_lib PROPERTIES
            INTERFACE_LINK_LIBRARIES "c-ares::cares")
    endif()
endif()

# --- Shared library: turbo_ocr_common ---
add_library(turbo_ocr_common STATIC
    src/detection/det_postprocess.cpp
    src/formula/cjk_stats.cpp
    src/formula/latex_extract.cpp
    src/recognition/ctc_decode.cpp
    src/decode/fast_png_decoder.cpp
    src/layout/reading_order.cpp
    src/layout/match_unsorted.cpp
    src/layout/child_blocks.cpp
    src/layout/text_line_cluster.cpp
    src/docassembly/repetition_guard.cpp
    src/docassembly/title_level.cpp
    src/docassembly/table_merge.cpp
    src/routing/routing_config.cpp
    src/output/markdown_export.cpp
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
    tests/cpp/test_cjk_stats.cpp
    tests/cpp/test_encoding.cpp
    tests/cpp/test_serialization.cpp
    tests/cpp/test_language_paths.cpp
    tests/cpp/test_pdf_renderer_liveness.cpp
    tests/cpp/test_ctc_decode.cpp
    tests/cpp/test_perspective.cpp
    tests/cpp/test_det_postprocess.cpp
    tests/cpp/test_image_dims.cpp
    tests/cpp/test_xy_cut.cpp
    tests/cpp/test_logger_ratelimit.cpp
    tests/cpp/test_server_config.cpp
    tests/cpp/test_docassembly.cpp
    tests/cpp/test_dict_resolution.cpp
    tests/cpp/test_gpu_safety.cpp
    tests/cpp/test_routing_config.cpp
    tests/cpp/test_rec_geometry.cpp
    tests/cpp/test_env_utils.cpp
    tests/cpp/test_html_reconstruct.cpp
    tests/cpp/test_formula_tokenizer.cpp
    tests/cpp/test_png_guards.cpp
    tests/cpp/test_pipeline_detail.cpp
    tests/cpp/test_pdf_mode_strict.cpp
    tests/cpp/test_serialization_escape.cpp
    tests/cpp/test_logger_bounds.cpp
    tests/cpp/test_latex_extract.cpp
    third_party/catch2/catch_amalgamated.cpp
)
# These three TUs are not in turbo_ocr_common. On the CPU build the test has no
# other provider, so it compiles them directly. On the GPU build they come from
# turbo_ocr_gpu (linked below); compiling them here too would double-compile
# (static-archive dedup hides it, but it is fragile). Add them CPU-only.
if(USE_CPU_ONLY)
    target_sources(turbo_ocr_tests PRIVATE
        src/render/pdf_renderer.cpp
        src/table/html_reconstruct.cpp
        src/formula/formula_tokenizer.cpp)
endif()
target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_common)
target_include_directories(turbo_ocr_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/catch2
    ${CMAKE_SOURCE_DIR}/third_party/cli11
)
target_compile_options(turbo_ocr_tests PRIVATE ${TURBO_OCR_CXX_FLAGS})

# Register with CTest so `ctest` (and run_all.py --suite cpp) actually runs the
# suite. Without this, ctest finds no tests and exits 0 — green while testing
# nothing. Catch2's amalgamated runner returns non-zero on any failure, so a
# single add_test gates correctly. `ctest --no-tests=error` guards an empty suite.
add_test(NAME turbo_ocr_unit COMMAND turbo_ocr_tests)

# In the CPU build the test binary links turbo_ocr_common (not turbo_ocr_cpu),
# so it doesn't inherit USE_CPU_ONLY — define it here so GPU-only test files
# (test_gpu_safety.cpp, which includes CUDA headers) compile out cleanly.
if(USE_CPU_ONLY)
    target_compile_definitions(turbo_ocr_tests PRIVATE USE_CPU_ONLY)
endif()

if(USE_CPU_ONLY)
    message(STATUS "Building CPU-ONLY mode (ONNX Runtime backend)")

    # Prefer system-installed ORT (Dockerfile installs to /usr/local/lib).
    # If not found, FetchContent downloads the same pinned tarball as
    # docker/Dockerfile.cpu so native CPU builds stay reproducible.
    # The vendored third_party/onnxruntime is x86_64. On x86_64 it is preferred;
    # on aarch64 it is the wrong arch, so trust only /usr/local (Docker installs
    # the right arch there) or a fresh arch-matched fetch.
    if(TURBO_ARCH STREQUAL "x64")
        set(_ort_tp_lib ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib)
        set(_ort_tp_inc ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)
    else()
        set(_ort_tp_lib "")
        set(_ort_tp_inc "")
    endif()
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS /usr/local/lib ${_ort_tp_lib})
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS /usr/local/include ${_ort_tp_inc})
    if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
        include(FetchContent)
        set(ORT_VERSION "1.22.0" CACHE STRING "ONNX Runtime version to fetch")
        # SHA256 of onnxruntime-linux-<arch>-${ORT_VERSION}.tgz. The x64 hash is
        # pinned; the aarch64 tarball has a different hash per release. We do not
        # ship a verified default aarch64 hash, so the aarch64 build/CI must pass
        # -DORT_SHA256=... to pin it. An empty hash is a hard error rather than a
        # silent unverified download — set ALLOW_UNVERIFIED_ORT=ON to override.
        if(TURBO_ARCH STREQUAL "x64")
            set(ORT_SHA256 "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3"
                CACHE STRING "ONNX Runtime tarball SHA256")
        else()
            set(ORT_SHA256 "" CACHE STRING "ONNX Runtime aarch64 tarball SHA256 (set in CI)")
        endif()
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${TURBO_ORT_URL_ARCH}-${ORT_VERSION}.tgz")
        message(STATUS "ONNX Runtime not found; fetching v${ORT_VERSION} (${TURBO_ORT_URL_ARCH})")
        if(ORT_SHA256 STREQUAL "")
            if(NOT ALLOW_UNVERIFIED_ORT)
                message(FATAL_ERROR
                    "ORT_SHA256 unset for ${TURBO_ARCH}: refusing to fetch ONNX Runtime "
                    "without hash verification. Pin the tarball hash with -DORT_SHA256=<sha256> "
                    "(see third_party/VERSIONS.md), or pass -DALLOW_UNVERIFIED_ORT=ON to "
                    "explicitly accept an unverified download.")
            endif()
            message(WARNING "ALLOW_UNVERIFIED_ORT set; fetching ONNX Runtime for ${TURBO_ARCH} without hash verification")
            FetchContent_Declare(onnxruntime_prebuilt
                URL "${_ort_url}"
                SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
        else()
            FetchContent_Declare(onnxruntime_prebuilt
                URL "${_ort_url}"
                URL_HASH SHA256=${ORT_SHA256}
                SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
        endif()
        FetchContent_MakeAvailable(onnxruntime_prebuilt)
        find_library(ONNXRUNTIME_LIB onnxruntime
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib REQUIRED)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include REQUIRED)
    endif()
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

    # Protobuf + gRPC discovery + proto codegen (shared with the GPU arm).
    turbo_setup_grpc_codegen()

    # --- CPU library ---
    add_library(turbo_ocr_cpu STATIC
        src/engine/cpu_engine.cpp
        src/detection/cpu_paddle_det.cpp
        src/recognition/cpu_paddle_rec.cpp
        src/classification/cpu_paddle_cls.cpp
        src/classification/cpu_doc_orientation.cpp
        src/pipeline/cpu_ocr_pipeline.cpp
        src/layout/cpu_paddle_layout.cpp
        src/render/pdf_renderer.cpp
        src/pdf/pdf_text_layer.cpp
        src/pdf/page_image_encoder.cpp
        # CUDA-FREE formula (PP-FormulaNet-S fused graph on ORT-CPU) +
        # table (SLANeXt ORT-CPU encoder + host GRU decode) stages. The
        # CUDA-tied PPFormulaNetOrt/SlanextEncSplit GPU TUs are NOT built here;
        # these are their pure-C++ siblings sharing the bit-exact preprocess /
        # tokenizer / host decoder / dict / cell-matcher / html code.
        src/formula/ort_session.cpp
        src/formula/ppformulanet_preprocess.cpp
        src/formula/cpu_formula_recognizer.cpp
        src/formula/formula_tokenizer.cpp
        src/table/cpu_slanext_table.cpp
        src/table/slanext_host_decode.cpp
        src/table/slanext_postprocess.cpp
        src/table/slanext_dict.cpp
        src/table/cell_matcher.cpp
        src/table/html_reconstruct.cpp
    )
    # The SLANeXt host GRU decoder is a tight scalar GEMV/attention loop; let the
    # compiler vectorize the dot-product reductions (FP reassociation only —
    # accuracy-neutral, verified) under the default AVX2 target. Same flags the
    # GPU build applies to slanext_enc_split.cpp.
    set_source_files_properties(src/table/slanext_host_decode.cpp PROPERTIES
        COMPILE_OPTIONS "-ffast-math;-fopenmp-simd")
    # libjpeg-turbo: fast JPEG encode for inline page-image export.
    find_library(TURBOJPEG_LIB turbojpeg REQUIRED)
    target_include_directories(turbo_ocr_cpu PUBLIC
        ${TURBO_PDFIUM_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(turbo_ocr_cpu PUBLIC turbo_ocr_common ${ONNXRUNTIME_LIB}
        ${TURBO_PDFIUM_DIR}/lib/libpdfium.so pthread
        ${TURBOJPEG_LIB})
    target_compile_options(turbo_ocr_cpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    # USE_CPU_ONLY: existing CPU-build switch (page_image_encoder etc.).
    # TURBO_CPU_ONLY: compiles the CUDA execution provider out of the shared
    # OrtSession TU (src/formula/ort_session.cpp) so the CUDA-free CPU library
    # links only load_cpu() — the GPU build leaves it undefined and keeps load().
    target_compile_definitions(turbo_ocr_cpu PUBLIC USE_CPU_ONLY TURBO_CPU_ONLY)
    add_library(turbo_ocr::cpu ALIAS turbo_ocr_cpu)

    # CPU HTTP + gRPC server
    add_executable(turboocr-cpu-server
        src/server/cpu_main.cpp
        src/routes/common_routes.cpp
        src/routes/pdf_routes.cpp
        src/pipeline/pdf_job.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(turboocr-cpu-server PRIVATE
        ${PROTO_GEN_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/cli11
    )
    target_link_libraries(turboocr-cpu-server PRIVATE
        turbo_ocr_cpu
        Drogon::Drogon
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

else()
    message(STATUS "Building GPU mode (CUDA + TensorRT)")

    find_package(CUDAToolkit REQUIRED)

    set(TENSORRT_DIR "/usr/local/tensorrt" CACHE PATH "Path to TensorRT")

    # Protobuf + gRPC discovery + proto codegen (shared with the CPU arm).
    turbo_setup_grpc_codegen()

    # CUDA
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Compute capabilities baked into the GPU binary. When not set on the
    # command line, CMake pre-fills CMAKE_CUDA_ARCHITECTURES with nvcc's
    # default arch during enable_language() — the oldest arch the toolkit
    # supports (sm_75 on CUDA 13), whose embedded PTX forward-JITs onto every
    # newer GPU. That is why the published image runs on Turing and up.
    # Override for a native build of your GPU (skips JIT, e.g. RTX 50):
    #   -DCMAKE_CUDA_ARCHITECTURES=120
    # Reference SMs:
    #   75   Turing            (T4, RTX 20, GTX 16)
    #   80   Ampere DC         (A100)
    #   86   Ampere consumer   (RTX 30, A40, A10)
    #   89   Ada Lovelace      (RTX 40, L40, L4)
    #   90   Hopper            (H100, H200, GH200)              [needs CUDA 12.0+]
    #   100  Blackwell DC      (B100, B200, B300, GB200)        [needs CUDA 12.8+]
    #   120  Blackwell consumer (RTX 50-series)                 [needs CUDA 12.8+]
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --expt-relaxed-constexpr")

    # --- ONNX Runtime (GPU build: CUDA execution provider REQUIRED) ---
    # ort_session.cpp creates its sessions on ORT's CUDA EP
    # (CreateCUDAProviderOptions), which needs libonnxruntime.so PLUS
    # libonnxruntime_providers_cuda.so side by side. The official
    # onnxruntime-linux-<arch>-<ver>.tgz release asset is CPU-ONLY — it would
    # configure and link fine here and then fail at runtime when the formula
    # backend asks for the CUDA EP. So: discover, then hard-require the CUDA
    # EP at configure time instead of fetching a wrong-flavor tarball.
    # third_party/ is searched first (the vendored copy is the known-good
    # CUDA-enabled build); /usr/local second.
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib /usr/local/lib)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include /usr/local/include)
    if((NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
       AND TURBO_ARCH STREQUAL "x64")
        # Clean-clone fallback: fetch the official CUDA-13 GPU build (the
        # gpu_cuda13 release asset — the plain tarball is CPU-only and would
        # break the formula backends at runtime). Same pin as Dockerfile.gpu.
        include(FetchContent)
        set(ORT_GPU_VERSION "1.27.0" CACHE STRING "ONNX Runtime (GPU/CUDA-13) version to fetch")
        set(ORT_GPU_SHA256 "1a3227e1dc2f53d9f877c93278af500b15e26d99aa5ade877692138b3ab7d351"
            CACHE STRING "onnxruntime-linux-x64-gpu_cuda13 tarball SHA256")
        message(STATUS "ONNX Runtime not found; fetching v${ORT_GPU_VERSION} (x64, CUDA 13)")
        FetchContent_Declare(onnxruntime_gpu_prebuilt
            URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_GPU_VERSION}/onnxruntime-linux-x64-gpu_cuda13-${ORT_GPU_VERSION}.tgz"
            URL_HASH SHA256=${ORT_GPU_SHA256}
            SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)
        FetchContent_MakeAvailable(onnxruntime_gpu_prebuilt)
        find_library(ONNXRUNTIME_LIB onnxruntime
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)
    endif()
    if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR
            "ONNX Runtime not found (looked in third_party/onnxruntime and "
            "/usr/local). The GPU build needs a CUDA-enabled ONNX Runtime "
            "(libonnxruntime.so + libonnxruntime_providers_cuda.so + headers); "
            "the plain onnxruntime-linux-<arch>-<ver>.tgz release asset is "
            "CPU-only and will NOT work. Microsoft publishes no GPU aarch64 "
            "tarball — install a CUDA-enabled ORT into "
            "third_party/onnxruntime/{include,lib} or /usr/local, matching "
            "your CUDA major version.")
    endif()
    get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
    find_library(ONNXRUNTIME_CUDA_EP onnxruntime_providers_cuda
        HINTS ${ONNXRUNTIME_LIB_DIR} NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_CUDA_EP)
        message(FATAL_ERROR
            "Found ONNX Runtime at ${ONNXRUNTIME_LIB}, but it has no CUDA "
            "execution provider (libonnxruntime_providers_cuda.so is not next "
            "to it). This is the CPU-only distribution; the formula backends "
            "would fail at runtime. Install a CUDA-enabled ORT build (GPU "
            "package matching your CUDA major) into third_party/onnxruntime "
            "or /usr/local instead.")
    endif()
    message(STATUS "ONNX Runtime (CUDA EP): ${ONNXRUNTIME_LIB}")

    # --- GPU library ---
    add_library(turbo_ocr_gpu STATIC
        src/engine/trt_engine.cpp
        src/engine/engine_cache.cpp
        src/engine/onnx_to_trt.cpp
        src/detection/paddle_det.cpp
        src/recognition/paddle_rec.cpp
        src/classification/paddle_cls.cpp
        src/classification/doc_orientation.cpp
        src/lang_cls/script_id.cpp
        src/layout/paddle_layout.cpp
        src/pipeline/ocr_pipeline.cpp
        src/pipeline/ocr_pipeline_run.cpp
        src/pipeline/ocr_pipeline_detail.cpp
        src/pipeline/infer_one.cpp
        src/pipeline/recognizer_registry.cpp
        src/kernels/kernels.cu
        src/kernels/jfa.cu
        src/render/pdf_renderer.cpp
        src/pdf/pdf_text_layer.cpp
        src/pdf/page_image_encoder.cpp
        # CUA router + table + formula stages (referenced by ocr_pipeline's
        # dispatch_router_). Pulled in here so turboocr-server links;
        # the operator's prior in-progress integration left these source
        # files on disk but not on the build target.
        src/router/cua_router.cpp
        src/router/region_features.cpp
        src/router/routing_plan.cpp
        src/table/table_cell.cpp
        src/table/slanext_enc_split.cpp
        src/table/slanext_dict.cpp
        src/table/slanext_postprocess.cpp
        src/table/cell_matcher.cpp
        src/table/html_reconstruct.cpp
        src/table/table_recognizer.cpp
        src/table/slanext_table_recognizer.cpp
        src/table/vlm_table_recognizer.cpp
        src/table/vlm_table.cpp
        src/table/kernels/table_kernels.cu
        src/formula/formula_tokenizer.cpp
        src/formula/formula_recognizer.cpp
        src/formula/auto_cjk_formula.cpp
        src/formula/vlm_formula.cpp
        src/formula/ppformulanet_ort.cu
        src/formula/ppformulanet_preprocess.cpp
        src/formula/ort_session.cpp
        src/vlm/crop_pool.cpp
        src/backends/openai_endpoint.cpp
    )
    # The SLANeXt host GRU decoder is a tight scalar GEMV/attention loop; let the
    # compiler vectorize the dot-product reductions (FP reassociation only —
    # accuracy-neutral, verified) under the default AVX2 (-march=x86-64-v3) target.
    set_source_files_properties(src/table/slanext_enc_split.cpp PROPERTIES
        COMPILE_OPTIONS "-ffast-math;-fopenmp-simd")
    target_include_directories(turbo_ocr_gpu PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_DIR}/include
        ${PROTO_GEN_DIR}
        ${TURBO_PDFIUM_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}   # ort_session.cpp: formula encoder+prep+step via ORT-CUDA
    )
    target_link_directories(turbo_ocr_gpu PUBLIC ${TENSORRT_DIR}/lib)
    # Link libpdfium.so directly for in-process text-layer extraction.
    # The shared lib is vendored in third_party/pdfium/lib and also placed
    # in bin/ by scripts/install_fastpdf2png.sh (so native dev + Docker
    # builds both resolve it at runtime). Docker additionally copies it to
    # /usr/lib via ldconfig in Dockerfile.gpu:49.
    # libjpeg-turbo: CPU JPEG encode fallback for inline page-image export
    # (nvJPEG handles the GPU default path; already linked below).
    find_library(TURBOJPEG_LIB turbojpeg REQUIRED)
    message(STATUS "libjpeg-turbo: ${TURBOJPEG_LIB}")
    # libcurl: VLMFormula/VLMTable call an OpenAI-compatible vLLM endpoint.
    # The backends are runtime-opt-in but always compiled, so the dev package
    # is a hard build dependency — say which one to install instead of
    # find_library's bare "Could not find CURL_LIB".
    find_library(CURL_LIB curl)
    if(NOT CURL_LIB)
        message(FATAL_ERROR
            "libcurl development files not found. The VLM table/formula "
            "clients (src/table/vlm_table.cpp, src/formula/vlm_formula.cpp) "
            "link libcurl unconditionally even though they are runtime-opt-in. "
            "Install it first: libcurl4-openssl-dev (Debian/Ubuntu), "
            "libcurl-devel (Fedora/RHEL), or curl (Arch).")
    endif()
    message(STATUS "libcurl: ${CURL_LIB}")
    target_link_libraries(turbo_ocr_gpu PUBLIC
        turbo_ocr_common
        nvinfer
        nvinfer_plugin
        nvonnxparser
        nvjpeg
        cudart
        ${TURBO_PDFIUM_DIR}/lib/libpdfium.so
        ${TURBOJPEG_LIB}
        ${CURL_LIB}
        ${ONNXRUNTIME_LIB}
    )
    target_compile_options(turbo_ocr_gpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    add_library(turbo_ocr::gpu ALIAS turbo_ocr_gpu)

    # GPU tests link to GPU library so CUDA-dependent tests can be added here.
    target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_gpu CUDA::cudart)
    target_include_directories(turbo_ocr_tests PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_DIR}/include
    )
    # Table/router/formula/parser unit tests resolve their symbols from
    # turbo_ocr_gpu, so they only build in the GPU configuration (the CPU
    # test binary links turbo_ocr_common only).
    target_sources(turbo_ocr_tests PRIVATE
        tests/pipeline/test_finalize_deferred.cpp
        tests/pipeline/test_multi_rec.cpp
        tests/table/test_cell_matcher.cpp
        tests/table/test_otsl_to_html.cpp
        tests/table/test_html_reconstruct.cpp
        tests/table/test_slanext_dict.cpp
        tests/table/test_slanext_postprocess.cpp
        tests/table/test_slanext_postprocess_aabb.cpp
        tests/router/test_cua_router.cpp
        tests/formula/test_formula_tokenizer.cpp
        tests/formula/test_formula_preprocess.cpp
        tests/formula/test_formulanet_ar_loop.cpp
        tests/lang_cls/test_script_id.cpp
        tests/cpp/test_openai_parsers.cpp
    )

    # GPU HTTP + gRPC server (shared pipeline pool)
    add_executable(turboocr-server
        src/server/main.cpp
        src/routes/common_routes.cpp
        src/routes/image_routes.cpp
        src/routes/pdf_routes.cpp
        src/pipeline/pdf_job.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(turboocr-server PRIVATE
        ${PROTO_GEN_DIR}
        ${CMAKE_SOURCE_DIR}/third_party/cli11
    )
    target_link_libraries(turboocr-server PRIVATE
        turbo_ocr_gpu
        Drogon::Drogon
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )
    # Bake the TensorRT lib dir into the binary's RPATH so a bare `cmake && make`
    # (with -DTENSORRT_DIR) runs with no LD_LIBRARY_PATH. nvinfer dlopen()s
    # libnvinfer_builder_resource_sm*.so by bare name at first engine build;
    # without this it aborts cryptically. (onnxruntime's dir is auto-added to the
    # rpath because it's linked by absolute path; TRT is linked via
    # link_directories + bare names, which does NOT contribute to rpath.)
    #
    # --disable-new-dtags is LOAD-BEARING: it forces DT_RPATH instead of the
    # linker-default DT_RUNPATH. DT_RUNPATH is NOT consulted when a dependency
    # library (libnvinfer) dlopen()s another lib by bare name, so the resource
    # lib would not be found; DT_RPATH IS searched transitively for such loads.
    set_target_properties(turboocr-server PROPERTIES
        BUILD_RPATH "${TENSORRT_DIR}/lib"
        INSTALL_RPATH "${TENSORRT_DIR}/lib;${ONNXRUNTIME_LIB_DIR}")
    target_link_options(turboocr-server PRIVATE "-Wl,--disable-new-dtags")

    # Dev/CI-only drivers — EXCLUDE_FROM_ALL so the default `make` (and the Docker image build)
    # never compiles them into the production image; build explicitly when needed.
    # plus-M FAST-path golden self-test driver (no server boot; gate_bench).
    add_executable(plusm_selftest EXCLUDE_FROM_ALL tools/plusm_selftest.cpp)
    target_link_libraries(plusm_selftest PRIVATE turbo_ocr_gpu)

    # Formula backend swap check: FORMULA_BACKEND -> resolved routing engine.
    add_executable(formula_swap_check EXCLUDE_FROM_ALL tools/formula_swap_check.cpp)
    target_link_libraries(formula_swap_check PRIVATE turbo_ocr_gpu)

    # Legacy standalone gRPC server removed — gRPC now runs from the main
    # binary via start_grpc_server() in grpc_service.h.

    # Benchmark tools
    add_executable(grpc_bench EXCLUDE_FROM_ALL
        tools/grpc_bench.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(grpc_bench PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(grpc_bench PRIVATE
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    add_executable(grpc_burst EXCLUDE_FROM_ALL
        tools/grpc_burst.cpp
        ${PROTO_GEN_CC}
        ${GRPC_GEN_CC}
    )
    target_include_directories(grpc_burst PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(grpc_burst PRIVATE
        ${GRPC_LIBRARIES}
        protobuf::libprotobuf
    )

    # diff_ccl: side-by-side comparator for CPU CCL=1 vs GPU CCL=2 outputs.
    # Loads an image, runs both detection paths, prints box counts, IoU stats,
    # and missed/extra boxes. Used to drive the GPU JFA pipeline accuracy.
    add_executable(diff_ccl EXCLUDE_FROM_ALL tools/diff_ccl.cpp)
    target_include_directories(diff_ccl PRIVATE include)
    target_link_libraries(diff_ccl PRIVATE
        turbo_ocr_gpu
        ${OpenCV_LIBS}
    )
    set_target_properties(diff_ccl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )


    # Regression: nvJPEG handler-thread -> dispatcher-worker race
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/test_nvjpeg_race.cpp)
        add_executable(test_nvjpeg_race EXCLUDE_FROM_ALL tools/test_nvjpeg_race.cpp)
        target_include_directories(test_nvjpeg_race PRIVATE include)
        target_link_libraries(test_nvjpeg_race PRIVATE
            CUDA::cudart
            CUDA::nvjpeg
            ${OpenCV_LIBS}
        )
    endif()
endif()

# --- Interprocedural / link-time optimization (Release only) --------------
# Skipped for turbo_ocr_tests: Catch2's amalgamated TU is large and LTO
# linking adds significant time with no runtime payoff for unit tests.
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err)
if(_ipo_ok AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling IPO/LTO for Release targets")
    set_property(TARGET turbo_ocr_common PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    if(USE_CPU_ONLY)
        set_property(TARGET turbo_ocr_cpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET turboocr-cpu-server PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        set_property(TARGET turbo_ocr_gpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET turboocr-server PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
elseif(NOT _ipo_ok)
    message(STATUS "IPO/LTO not supported by this toolchain: ${_ipo_err}")
endif()

# --- Install rules --------------------------------------------------------
# Minimal layout so `cmake --install build --prefix /usr/local` works.
if(USE_CPU_ONLY)
    install(TARGETS turboocr-cpu-server RUNTIME DESTINATION bin)
else()
    install(TARGETS turboocr-server RUNTIME DESTINATION bin)
endif()
install(DIRECTORY models/ DESTINATION share/turbo-ocr/models OPTIONAL)
install(DIRECTORY scripts/ DESTINATION share/turbo-ocr/scripts USE_SOURCE_PERMISSIONS)

# --- Fetch OCR model bundles for native builds ----------------------------
# Uses a v6-specific stamp as the sentinel — NOT models/det.onnx, which a v5
# checkout already has, so keying on it would skip the fetch and the new v6
# tier siblings (det_small/det_tiny/rec_small/rec_tiny/keys_tiny) would never
# be pulled (first boot of OCR_MODEL=small/tiny then fails). The stamp is
# created only after a successful fetch, and the fetch re-runs whenever the
# script (which pins the release tag) changes. Docker builds pass
# -DFETCH_MODELS=OFF and run fetch_release_models.sh as their own RUN layer.
if(FETCH_MODELS)
    set(MODELS_SENTINEL "${CMAKE_SOURCE_DIR}/models/.v6_stamp")
    add_custom_command(
        OUTPUT ${MODELS_SENTINEL}
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/fetch_release_models.sh
        COMMAND ${CMAKE_COMMAND} -E touch ${MODELS_SENTINEL}
        DEPENDS ${CMAKE_SOURCE_DIR}/scripts/fetch_release_models.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Fetching PP-OCRv6 models from pinned GitHub Release"
        VERBATIM
    )
    add_custom_target(fetch_models ALL DEPENDS ${MODELS_SENTINEL})
    if(USE_CPU_ONLY)
        add_dependencies(turboocr-cpu-server fetch_models)
    else()
        add_dependencies(turboocr-server fetch_models)
    endif()
endif()
