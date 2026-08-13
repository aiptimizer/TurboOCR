# 3.24 is required for $<LINK_LIBRARY:WHOLE_ARCHIVE,...>, which force-links the
# per-backend registrar TUs (src/backends/*/..._registry.cpp). A registrar
# defines no symbol anybody references, so without WHOLE_ARCHIVE a static link
# may drop it and the binary silently loses that backend.
cmake_minimum_required(VERSION 3.24)
# 3.24+ warns on FetchContent URL downloads without an extraction-timestamp
# policy; NEW (use extraction time) is deliberate so a re-fetched ONNX Runtime
# tarball never looks older than the build outputs that depend on it.
cmake_policy(SET CMP0135 NEW)

# Default to Release when the user didn't pick a build type. Single-config
# generators (Ninja/Make) leave CMAKE_BUILD_TYPE empty otherwise, which gives
# unoptimized builds.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Defaulting CMAKE_BUILD_TYPE to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# VERSION must stay a bare numeric triple — project() rejects a pre-release
# suffix, so TURBO_OCR_VERSION_FULL carries the suffix alongside it.
# python/turboocr_engine/_version.py holds the PEP 440 spelling of the SAME version
# ("4.0.0a1") and is what the wheel and `turboocr --version` report; README.md
# and docs/index.md say "v4.0-alpha" in prose. Move all of them together.
project(turbo_ocr VERSION 4.0.0 LANGUAGES CXX)
set(TURBO_OCR_VERSION_FULL "4.0.0-alpha.1"
    CACHE STRING "Full version including pre-release suffix" FORCE)

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
#   - Ubuntu Docker images (Dockerfile.cpu/nvidia): apt libprotobuf-dev ships NO
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
# MSVC spells all of this differently: it has no -march (the ISA baseline is
# /arch:AVX2), rejects -O3/-W* outright with "D8021 invalid numeric argument",
# and enables the -Wpedantic equivalent via /permissive-. Keeping one list and
# translating here is what lets the same configure serve gcc, clang and cl.
if(MSVC)
  # /arch:AVX2 is the x86-64-v3 equivalent and keeps the AVX2 path in
  # ctc_decode.cpp valid. /O2 is MSVC's full optimize (there is no /O3).
  # /EHsc: standard C++ exception model. /utf-8: this tree's sources and string
  # literals are UTF-8, and MSVC otherwise reads them in the system codepage.
  set(_turbo_arch_flag /arch:AVX2)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|arm64")
    set(_turbo_arch_flag "")   # ARM64 MSVC has no /arch: selector
  endif()
  # /Zc:preprocessor is LOAD-BEARING, not tidiness: MSVC's legacy preprocessor
  # mis-tokenizes a raw string literal passed as a macro argument, so every
  # CHECK(j == R"(...\"...\"...)") in the Catch2 suite parses as a plain string
  # followed by a bogus user-defined literal suffix ("invalid literal suffix
  # 'oder'"). The conformant preprocessor handles R"()" correctly.
  set(TURBO_OCR_CXX_FLAGS /O2 ${_turbo_arch_flag} /W3 /permissive- /EHsc /utf-8
                          /Zc:preprocessor
                          /wd4100)   # 4100 = unreferenced formal parameter (== -Wno-unused-parameter)
  # <windows.h> arrives transitively (ORT, OpenCV, <process.h>) and defines
  # min/max as function-like MACROS, which then eat every
  # std::numeric_limits<T>::min() in the tree — Catch2 included. NOMINMAX is the
  # documented opt-out. WIN32_LEAN_AND_MEAN drops winsock1 etc., which otherwise
  # clash with anything pulling winsock2. Both are compile-wide on purpose: a
  # per-target definition would miss whichever header includes windows.h first.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
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
endif()
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

# PDFium location. linux-x64 and mac-arm64 are vendored (both binaries live in
# lib/ beside one shared include/); other targets are unpacked there by
# scripts/setup/install_pdfium.sh from bblanchon/pdfium-binaries.
set(TURBO_PDFIUM_DIR "${CMAKE_SOURCE_DIR}/third_party/pdfium"
    CACHE PATH "Root of the PDFium SDK (include/ + lib/)")
if(APPLE)
    set(TURBO_PDFIUM_LIB "${TURBO_PDFIUM_DIR}/lib/libpdfium.dylib")
elseif(WIN32)
    # bblanchon/pdfium-binaries ships pdfium-win-x64 as an import library
    # (pdfium.dll.lib) plus the DLL in bin/. The DLL must be beside the .exe or
    # on PATH at RUN time — the import lib only satisfies the linker.
    set(TURBO_PDFIUM_LIB "${TURBO_PDFIUM_DIR}/lib/pdfium.dll.lib")
else()
    set(TURBO_PDFIUM_LIB "${TURBO_PDFIUM_DIR}/lib/libpdfium.so")
endif()
if(NOT EXISTS "${TURBO_PDFIUM_LIB}")
    message(FATAL_ERROR
        "PDFium not found at ${TURBO_PDFIUM_LIB}. Run scripts/setup/install_pdfium.sh "
        "to fetch the one matching this OS/arch (Windows: unpack pdfium-win-x64 "
        "into ${TURBO_PDFIUM_DIR}).")
endif()
# Existing is not the same as loadable. third_party/pdfium vendors linux-x64 and
# mac-arm64 in ONE lib/ directory, so on aarch64 Linux the x86_64 libpdfium.so is
# sitting right there: the check above passes, and the build instead dies several
# hundred targets later with
#     /usr/bin/ld: libpdfium.so: error adding symbols: file in wrong format
# which names neither the cause nor the fix — on the exact platform the message
# above was written for. Read the ELF header and say it at configure time.
#
# ELF only: the Apple path is a Mach-O the loader handles, and the Windows path
# is an import library, not a shared object.
if(NOT APPLE AND NOT WIN32)
    # 20 bytes -> 40 hex chars. e_machine is bytes 18-19, little-endian.
    file(READ "${TURBO_PDFIUM_LIB}" _turbo_elf_head HEX LIMIT 20)
    string(SUBSTRING "${_turbo_elf_head}" 36 4 _turbo_elf_machine)
    if(TURBO_ARCH STREQUAL "aarch64")
        set(_turbo_elf_want "b700")   # EM_AARCH64
    else()
        set(_turbo_elf_want "3e00")   # EM_X86_64
    endif()
    if(NOT _turbo_elf_machine STREQUAL _turbo_elf_want)
        message(FATAL_ERROR
            "PDFium at ${TURBO_PDFIUM_LIB} is built for the wrong architecture "
            "(ELF e_machine 0x${_turbo_elf_machine}, this build needs "
            "0x${_turbo_elf_want} for ${TURBO_ARCH}). The vendored copy is "
            "linux-x64; run scripts/setup/install_pdfium.sh to replace it with "
            "the ${TURBO_ARCH} build.")
    endif()
endif()

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
# test suite, then aggregate the *.gcda with scripts/dev/coverage.sh (gcovr) — any
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

# USE_CPU_ONLY is a misnomer kept for back-compat. It does not mean "CPU only";
# it selects the ONNX-Runtime HOST build path — the one cpu, apple, intel AND
# amd all use — as opposed to the native CUDA + TensorRT path, which only nvidia
# needs. An Intel iGPU/NPU build is not "CPU only", yet it takes this path.
option(USE_CPU_ONLY "Use the ONNX-Runtime host build path (everything except the native NVIDIA CUDA/TensorRT path)" OFF)

# Derive the build mode from the REQUESTED backends instead of making the user
# state it twice. The mode branch (`if(USE_CPU_ONLY)`) runs ~780 lines before
# TURBO_BACKENDS is resolved, and its `else()` calls find_package(CUDAToolkit
# REQUIRED) unconditionally — so a `-DTURBO_BACKENDS="cpu;intel"` build fell into
# the CUDA branch and died hunting for a toolkit the target machine has no reason
# to own. Selecting any backend set that does not include nvidia is, by
# definition, the host path: infer it here, before the branch. nvidia (or the
# zero-flag default, which is the flagship native-TensorRT server) keeps the CUDA
# path. An explicit -DUSE_CPU_ONLY still works and an explicit nvidia selection
# still gets CUDA; this only removes the redundant second flag for the
# non-nvidia backends.
if(DEFINED TURBO_BACKENDS AND NOT "${TURBO_BACKENDS}" STREQUAL ""
   AND NOT ("nvidia" IN_LIST TURBO_BACKENDS))
  if(NOT USE_CPU_ONLY)
    message(STATUS
      "TURBO_BACKENDS='${TURBO_BACKENDS}' does not include nvidia — using the "
      "host (ONNX Runtime) build path; no CUDA toolchain required.")
  endif()
  set(USE_CPU_ONLY ON)
endif()

include(src/service/http/sources.cmake)
include(src/service/grpc/sources.cmake)
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

# Build the nanobind Python extension (`_turboocr`). Off by default so the
# server/CLI builds are unaffected; scikit-build-core turns it ON for wheels.
# ccache when available: header-heavy edits recompile many TUs, and repeat
# builds across build dirs (server, wheel, tests) redo identical work. ccache
# makes those hits ~free. Opt out with -DTURBO_CCACHE=OFF.
option(TURBO_CCACHE "Use ccache as compiler launcher when found" ON)
if(TURBO_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(TURBO_CCACHE_EXE ccache)
    if(TURBO_CCACHE_EXE)
        set(CMAKE_C_COMPILER_LAUNCHER "${TURBO_CCACHE_EXE}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${TURBO_CCACHE_EXE}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${TURBO_CCACHE_EXE}")
        message(STATUS "ccache enabled: ${TURBO_CCACHE_EXE}")
    endif()
endif()

option(BUILD_PYTHON "Build the nanobind Python extension (_turboocr)" OFF)
if(BUILD_PYTHON)
    # Every static lib in this configure ends up inside the _turboocr shared
    # object. On Linux non-PIC objects (and their local-exec TLS, e.g. the
    # logger's thread_local buffer) are illegal in a .so — the link fails with
    # "relocation R_X86_64_TPOFF32 ... local-exec is incompatible with
    # -shared". macOS is PIC by default, which is why only Linux wheel builds
    # hit this. Must be set before any add_library().
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# libcurl, found ONCE at top level so the CURL::libcurl imported target exists
# before any target that needs it. Not REQUIRED: the VLM remote-endpoint path is
# the only consumer, and a build without it still produces the library, the
# backends and the local OCR pipeline. Found here rather than at the first use
# because targets are defined in file order and the pipeline library (which
# reaches <curl/curl.h> through openai_policy.h -> crop_pool.h) is declared long
# before the test-target block that used to do the lookup.
# Threads::Threads instead of a bare `pthread`: the library is named pthread on
# Linux, is part of libSystem on darwin, and does not exist at all on MSVC
# (threading lives in the CRT), so linking the literal name fails with
# "cannot open input file 'pthread.lib'". FindThreads resolves to the right
# thing — or to nothing — on each.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(CURL QUIET)
if(CURL_FOUND)
    message(STATUS "libcurl: found (VLM remote endpoint enabled)")
else()
    message(STATUS "libcurl: NOT found — VLM remote endpoint excluded")
endif()

# BUILD_SERVER gates the ONE dependency that only turboocr-server needs.
# Drogon::Drogon is linked in exactly one place — unified_server.cmake — but the
# find_package here was unconditional, so a configure that wants only the test
# binary or the backends (a first port to a platform with no Drogon build, say)
# failed before reaching them. Default ON: every existing configure is unchanged.
option(BUILD_SERVER "Build turboocr-server (requires Drogon)" ON)
if(BUILD_SERVER)
    find_package(Drogon CONFIG REQUIRED)
endif()

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
    src/analysis/detection/det_postprocess.cpp
    src/analysis/formula/cjk_stats.cpp
    src/onnx/host_ort_threads.cpp
    src/analysis/formula/latex_extract.cpp
    src/analysis/recognition/ctc_decode.cpp
    src/image/fast_png_decoder.cpp
    src/analysis/layout/order/reading_order.cpp
    src/analysis/layout/order/xy_cut.cpp
    src/analysis/layout/blocks/match_unsorted.cpp
    src/analysis/layout/blocks/child_blocks.cpp
    src/analysis/layout/blocks/text_line_cluster.cpp
    src/analysis/forms/field_masks.cpp
    src/analysis/forms/field_labels.cpp
    src/analysis/forms/field_detector.cpp
    # NOTE: src/analysis/forms/field_model.cpp is deliberately NOT here. It runs an ONNX
    # session, and turbo_ocr_common must stay free of onnxruntime — it builds
    # for every backend. detect_form_fields() takes the model's proposals as
    # plain data, so the geometry side needs no link to the runner at all.
    # Device-free OCR result policy (text-degraded / dropped-crop accounting and
    # the combine step), shared by BOTH orchestrations: src/pipeline/ and
    # src/pipeline/unified/. It lives here rather than in either pipeline because
    # turbo_ocr_common is the one archive both of them link PUBLIC, so the policy
    # resolves exactly once in every configure. See include/turbo_ocr/pipeline/
    # ocr_pipeline_detail.h for why forking it per pipeline is a known bug class.
    src/pipeline/ocr_pipeline_detail.cpp
    src/document/docassembly/repetition_guard.cpp
    src/document/docassembly/title_level.cpp
    src/document/docassembly/table_merge.cpp
    src/backend/routing_config.cpp
    src/document/markdown/markdown_export.cpp
    src/document/markdown/markdown_latex.cpp
    src/document/markdown/markdown_columns.cpp
    src/document/markdown/markdown_assets.cpp
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
    tests/cpp/common/test_box.cpp
    tests/cpp/common/test_cjk_stats.cpp
    tests/cpp/common/test_encoding.cpp
    tests/cpp/common/test_env_utils.cpp
    tests/cpp/common/test_logger_bounds.cpp
    tests/cpp/common/test_logger_ratelimit.cpp
    tests/cpp/common/test_perspective.cpp
    tests/cpp/common/test_serialization.cpp
    tests/cpp/common/test_serialization_escape.cpp
    tests/cpp/decode/test_image_dims.cpp
    tests/cpp/decode/test_png_guards.cpp
    tests/cpp/decode/test_decode_contract.cpp
    tests/cpp/detection/test_det_postprocess.cpp
    tests/cpp/recognition/test_ctc_decode.cpp
    tests/cpp/recognition/test_rec_geometry.cpp
    tests/cpp/layout/test_blocks_aggregate.cpp
    tests/cpp/layout/test_layout_blocks.cpp
    tests/cpp/layout/test_layout_hierarchy.cpp
    tests/cpp/layout/test_picodet_decode.cpp
    tests/cpp/layout/test_reading_order_buckets.cpp
    tests/cpp/layout/test_xy_cut.cpp
    tests/cpp/table/test_cell_matcher.cpp
    tests/cpp/table/test_html_reconstruct.cpp
    tests/cpp/table/test_slanext_dict.cpp
    tests/cpp/table/test_slanext_postprocess.cpp
    tests/cpp/table/test_slanext_postprocess_aabb.cpp
    tests/cpp/table/test_slanext_region_postprocess.cpp
    tests/cpp/table/test_table_cells.cpp
    tests/cpp/formula/test_formula_tokenizer.cpp
    tests/cpp/formula/test_latex_extract.cpp
    tests/cpp/formula/test_latex_normalize.cpp
    tests/cpp/forms/test_field_detector.cpp
    tests/cpp/markdown/test_markdown_export.cpp
    tests/cpp/docassembly/test_docassembly.cpp
    tests/cpp/pdf/test_pdf_mode_strict.cpp
    tests/cpp/pipeline/test_gpu_safety.cpp
    tests/cpp/pipeline/test_pipeline_detail.cpp
    # The three seam tests — test_pipeline_pool, test_seam_conformance and
    # test_backend_caps — are NOT here. They need UnifiedPipelinePool and the
    # backend registry (available_backends / make_backend), both of which live
    # in turbo_ocr_pipeline, which exists only when TURBO_BACKENDS is non-empty.
    # It defaults to EMPTY on the CUDA configure by design (the GPU arm builds
    # the native TensorRT server, not the seam), so listing them here compiled
    # three TUs nothing could satisfy and the default GPU configure could not
    # link its own test suite at all. They are added beside the
    # target_link_libraries that satisfies them, further down.
    tests/cpp/common/test_geometry_box.cpp
    tests/cpp/service/test_query_options.cpp
    tests/cpp/service/test_python_options.cpp
    tests/cpp/service/test_error_codes.cpp
    tests/cpp/backend_routing/test_routing_config.cpp
    tests/cpp/capability/test_capability_registry.cpp
    # (test_proto_capability_parity.cpp is attached further down, once
    # turbo_setup_grpc_codegen() has defined PROTO_GEN_DIR — the macro runs
    # inside the per-arm blocks, well after this list.)
    tests/cpp/validation/test_request_validation.cpp
    tests/cpp/server/test_dict_resolution.cpp
    tests/cpp/server/test_language_paths.cpp
    tests/cpp/pipeline/test_pool_sizing.cpp
    tests/cpp/server/test_server_config.cpp
    tests/cpp/server/test_work_pool.cpp
    src/service/server/bootstrap/server_config.cpp
    # The shared remote-VLM endpoint policy + the OTSL converter under it. Both
    # are device-free and now have device-free declaring headers
    # (turbo_ocr/vlm/openai_policy.h, turbo_ocr/table/vlm/otsl.h), so they build
    # in EVERY configure — they used to sit in the CUDA-only block below, which
    # left the converter the CPU unified server actually executes untested here.
    tests/cpp/vlm/test_openai_parsers.cpp
    # NOTE: test_openai_policy.cpp is NOT here — it needs libcurl. See below.
    third_party/catch2/catch_amalgamated.cpp
)
# openai_policy.h -> crop_pool.h -> <curl/curl.h>, so this suite only compiles
# where libcurl's headers exist. find_package(CURL) above is deliberately not
# REQUIRED — a build without it still produces the library, the backends and
# the local OCR pipeline — so the test has to degrade the same way the VLM
# transport sources already do further down. Listing it unconditionally broke
# the CI runner (no libcurl) while passing on any dev box that has it.
if(CURL_FOUND OR CURL_LIB)
    target_sources(turbo_ocr_tests PRIVATE tests/cpp/vlm/test_openai_policy.cpp)
else()
    message(STATUS "libcurl not found — VLM endpoint-policy tests excluded")
endif()
# Host-logic suites above resolve their symbols from turbo_ocr_cpu on the CPU
# build and from turbo_ocr_gpu on the GPU build (linked below).
if(USE_CPU_ONLY)
    target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_cpu)
    # otsl_to_html lives in turbo_ocr_gpu on the GPU build (and in turboocr-server's
    # own source list on the unified build) but in NO library on the CPU build, so
    # the VLM suites above need the TU compiled in here — CPU configure only, or
    # the GPU link would see it twice.
    target_sources(turbo_ocr_tests PRIVATE src/analysis/table/vlm/otsl_html.cpp)
endif()
# The PDF renderer liveness suite needs the real PdfRenderer, which is inside
# the same `if(NOT APPLE)` block as the rest of the PDF subsystem (pdfium is
# not vendored for darwin and pdf_daemon needs inotify).
target_sources(turbo_ocr_tests PRIVATE
    tests/cpp/pdf/test_pdf_searchable.cpp
    tests/cpp/pdf/test_font_style.cpp
    tests/cpp/pdf/test_pdf_editable.cpp
    tests/cpp/pdf/test_font_match.cpp
    tests/cpp/pdf/test_region_move.cpp)
# Liveness test for the DAEMON renderer (pdf_daemon.cpp), so it is meaningful
# only where that path is compiled: it includes <unistd.h> and drives worker
# processes. APPLE uses the in-process renderer; Windows has neither inotify nor
# the POSIX process calls. TURBO_HAVE_PDF_RENDER is not set until the per-arm
# block far below, hence the platform test rather than the capability one.
if(NOT APPLE AND NOT WIN32)
    target_sources(turbo_ocr_tests PRIVATE tests/cpp/pdf/test_pdf_renderer_liveness.cpp)
endif()
target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_common)
target_include_directories(turbo_ocr_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/catch2
    ${CMAKE_SOURCE_DIR}/third_party/cli11
)
target_compile_options(turbo_ocr_tests PRIVATE ${TURBO_OCR_CXX_FLAGS})
if(MSVC)
    # Nine test files call POSIX setenv/unsetenv to stage a case. Force-include
    # the shim rather than #ifdef-ing each one — the platform difference then
    # lives in exactly one file (tests/cpp/support/win_posix_shim.h).
    target_compile_options(turbo_ocr_tests PRIVATE
        /FI${CMAKE_SOURCE_DIR}/tests/cpp/support/win_posix_shim.h)
endif()

# Deterministic font fixtures for the font-style tests. Rendering the specimens
# from bundled DejaVu faces (tests/fixtures/fonts/) instead of PDFium's per-OS
# standard-14 substitution makes the bold/serif/pitch measurements identical on
# every platform — the fix for the Windows failures where PDFium substituted a
# non-bold or unmeasurable face for "Helvetica-Bold"/"Courier". Absolute path
# baked at configure time so the test finds them regardless of CWD.
target_compile_definitions(turbo_ocr_tests PRIVATE
    TURBO_TEST_FONT_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/fonts"
    # Same reasoning for the PDF fixtures: the renderer-liveness test proves the
    # in-process fallback actually rasterizes, which needs a real document.
    TURBO_TEST_PDF_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures/pdf")

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
    # third_party/onnxruntime holds whatever arch this machine fetched (it is
    # gitignored, not vendored in git). Prefer it on x86_64 AND on Apple: the
    # macOS copy is the arch-matched osx-arm64 build and is NEWER than the
    # Homebrew formula CMake would otherwise pick up. That version gap is not
    # cosmetic — on ORT 1.24 the CoreML EP miscompiles the layout graph and
    # returns NaN for every score, which silently yields an EMPTY layout.
    # On linux-aarch64 the copy may be the wrong arch, so trust only /usr/local
    # (Docker installs the right arch there) or a fresh arch-matched fetch.
    if(TURBO_ARCH STREQUAL "x64" OR APPLE)
        set(_ort_tp_lib ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib)
        set(_ort_tp_inc ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include)
    else()
        set(_ort_tp_lib "")
        set(_ort_tp_inc "")
    endif()
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS /usr/local/lib ${_ort_tp_lib})
    # PATH_SUFFIXES because Homebrew installs the headers one level deeper than
    # the library search implies: /opt/homebrew/include/onnxruntime/. Without it
    # the LIB is found and the INCLUDE is not, which is the worst of the three
    # outcomes — see the cache note below.
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS /usr/local/include ${_ort_tp_inc}
        PATH_SUFFIXES onnxruntime)
    # "Found" is not the bar on Apple. Homebrew's formula sits on the default
    # search path and is currently 1.24, whose CoreML EP miscompiles the layout
    # graph and returns NaN for every score — the page then reports NO layout at
    # all, with no error anywhere. Reject it and fall through to the pinned fetch
    # instead of building something that is quietly wrong. ORT_API_VERSION tracks
    # the 1.x minor, so 27 means 1.27, the release where it was fixed.
    if(APPLE AND ONNXRUNTIME_INCLUDE_DIR
       AND EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
        file(STRINGS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_api_line
             REGEX "^#define ORT_API_VERSION")
        string(REGEX MATCH "[0-9]+" _ort_api_ver "${_ort_api_line}")
        if(_ort_api_ver AND _ort_api_ver LESS 27)
            message(STATUS
                "ONNX Runtime at ${ONNXRUNTIME_LIB} is API ${_ort_api_ver} (< 27): "
                "its CoreML EP returns NaN for the layout model. Fetching a newer build.")
            unset(ONNXRUNTIME_LIB CACHE)
            unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
        endif()
    endif()
    if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE_DIR)
        # Half a find is worse than none. find_library/find_path write CACHE
        # variables and are a NO-OP once set, so if the search above found the
        # library but not the headers, the re-find after the fetch below would
        # silently keep the SYSTEM library and pair it with the DOWNLOADED
        # headers. That is not hypothetical: on a fresh macOS clone it linked
        # Homebrew's 1.24 dylib against 1.28 headers (ORT_API_VERSION 24 vs 28)
        # and the build died on a missing coreml_provider_factory.h, because the
        # headers it had picked up came from a LINUX tarball.
        unset(ONNXRUNTIME_LIB CACHE)
        unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
        include(FetchContent)
        set(ORT_VERSION "1.28.0" CACHE STRING "ONNX Runtime version to fetch")
        # SHA256 of onnxruntime-linux-<arch>-${ORT_VERSION}.tgz. The x64 hash is
        # pinned; the aarch64 tarball has a different hash per release. We do not
        # ship a verified default aarch64 hash, so the aarch64 build/CI must pass
        # -DORT_SHA256=... to pin it. An empty hash is a hard error rather than a
        # silent unverified download — set ALLOW_UNVERIFIED_ORT=ON to override.
        # The package name is per-PLATFORM, not just per-arch. This URL used to be
        # hardcoded to onnxruntime-linux-*, so a macOS machine downloaded a Linux
        # build: .so files it cannot link and an include/ with no
        # coreml_provider_factory.h, which is what src/onnx/ort_engine.cpp needs
        # for the CoreML EP. Refresh EVERY hash whenever ORT_VERSION changes.
        if(APPLE AND TURBO_ARCH STREQUAL "aarch64")
            set(ORT_SHA256 "1268b359718099bde2cedb55787f182a130067bc4f31e8c88478c445b850d3d8"
                CACHE STRING "ONNX Runtime osx-arm64 tarball SHA256")
            set(_ort_pkg "onnxruntime-osx-arm64-${ORT_VERSION}")
        elseif(APPLE)
            message(FATAL_ERROR
                "No pinned ONNX Runtime for Intel macOS. Install one "
                "(brew install onnxruntime) and point the build at it with "
                "-DONNXRUNTIME_LIB=<path/to/libonnxruntime.dylib> "
                "-DONNXRUNTIME_INCLUDE_DIR=<path/to/include>.")
        elseif(TURBO_ARCH STREQUAL "x64")
            set(ORT_SHA256 "a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407"
                CACHE STRING "ONNX Runtime tarball SHA256")
            set(_ort_pkg "onnxruntime-linux-x64-${ORT_VERSION}")
        else()
            # aarch64 IS now pinned (computed for 1.28.0) — it no longer has to be
            # passed in by CI.
            set(ORT_SHA256 "e15ff8b5d85afe6c144d97c6fd432254bf76a219daaf17658087d6ecb3e8f0bb"
                CACHE STRING "ONNX Runtime aarch64 tarball SHA256")
            set(_ort_pkg "onnxruntime-linux-aarch64-${ORT_VERSION}")
        endif()
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ort_pkg}.tgz")
        message(STATUS "ONNX Runtime not found; fetching ${_ort_pkg}")
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
        # NO_DEFAULT_PATH: having just downloaded a matched lib+include pair, the
        # only acceptable answer is that pair. Allowing the default search here is
        # how a system library gets married to downloaded headers.
        find_library(ONNXRUNTIME_LIB onnxruntime
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib
            NO_DEFAULT_PATH REQUIRED)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include
            NO_DEFAULT_PATH REQUIRED)
    endif()
    message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIB}")

    # Protobuf + gRPC discovery + proto codegen (shared with the GPU arm).
    # Gated with the server: the generated proto is consumed by the gRPC service
    # and by exactly one test (test_proto_capability_parity.cpp), which already
    # guards itself on `DEFINED PROTO_GEN_DIR` — so skipping this drops that one
    # parity check and nothing else. Lets a platform without a protobuf/gRPC
    # build still produce the library, the backends and the rest of the suite.
    if(BUILD_SERVER)
        turbo_setup_grpc_codegen()
    endif()

    # --- CPU library ---
    add_library(turbo_ocr_cpu STATIC
        src/onnx/ort_engine.cpp
        src/analysis/detection/ort_paddle_det.cpp
        src/analysis/recognition/ort_paddle_rec.cpp
        src/analysis/classification/ort_paddle_cls.cpp
        src/analysis/classification/ort_doc_orientation.cpp
        src/analysis/layout/ort_paddle_layout.cpp
        # FFDetr (RF-DETR on CommonForms) — the learned half of field
        # detection. Lives here, beside the other host ORT stages, because it
        # needs onnxruntime; the geometry detectors stay in turbo_ocr_common.
        src/analysis/forms/field_model.cpp
        # CUDA-FREE formula (PP-FormulaNet-S fused graph on ORT-CPU) +
        # table (SLANeXt ORT-CPU encoder + host GRU decode) stages. The
        # CUDA-tied PPFormulaNetOrt/SlanextEncSplit GPU TUs are NOT built here;
        # these are their pure-C++ siblings sharing the bit-exact preprocess /
        # tokenizer / host decoder / dict / cell-matcher / html code.
        src/analysis/formula/ppformulanet/ort_session.cpp
        src/analysis/formula/ppformulanet/ppformulanet_preprocess.cpp
        src/analysis/formula/ppformulanet/ort_formula_recognizer.cpp
        src/analysis/formula/ppformulanet/formula_tokenizer.cpp
        src/analysis/formula/ppformulanet/latex_normalize.cpp
        src/analysis/table/slanext/ort_slanext_table.cpp
        src/analysis/table/slanext/slanext_host_decode.cpp
        src/analysis/table/slanext/slanext_postprocess.cpp
        src/analysis/table/slanext/slanext_dict.cpp
        src/analysis/table/cell_matcher.cpp
        src/analysis/table/html_reconstruct.cpp
        src/analysis/table/table_cells.cpp
    )
    # The SLANeXt host GRU decoder is a tight scalar GEMV/attention loop; let the
    # compiler vectorize the dot-product reductions (FP reassociation only —
    # accuracy-neutral, verified) under the default AVX2 target. Same flags the
    # GPU build applies to slanext_enc_split.cpp.
    set_source_files_properties(src/analysis/table/slanext/slanext_host_decode.cpp PROPERTIES
        COMPILE_OPTIONS "-ffast-math;-fopenmp-simd")
    target_include_directories(turbo_ocr_cpu PUBLIC
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(turbo_ocr_cpu PUBLIC turbo_ocr_common ${ONNXRUNTIME_LIB}
        Threads::Threads)

    # In-process PDFium text work — text-layer extraction, mode=auto_verified,
    # and the searchable-PDF writer. Pure library calls, so it builds on every
    # platform PDFium ships for.
    target_sources(turbo_ocr_cpu PRIVATE
        src/pdf/text/pdf_text_layer.cpp
        src/pdf/text/pdf_text_extract.cpp
        src/pdf/text/pdf_text_verify.cpp
        src/pdf/text/font_style.cpp
        src/pdf/text/font_match.cpp
        src/pdf/text/region_extract.cpp
        src/pdf/text/pdf_searchable.cpp
        src/pdf/pdf_documents.cpp
        src/pdf/text/pdf_searchable_encoding.cpp
        src/pdf/text/pdf_searchable_style.cpp
        src/pdf/text/pdf_searchable_shapes.cpp)
    target_include_directories(turbo_ocr_cpu PUBLIC ${TURBO_PDFIUM_DIR}/include)
    target_link_libraries(turbo_ocr_cpu PUBLIC ${TURBO_PDFIUM_LIB})
    # The dylib's install name is @rpath/libpdfium.dylib; point consumers at
    # the vendored directory so they resolve it wherever they run from.
    if(APPLE)
        set_target_properties(turbo_ocr_cpu PROPERTIES
            INTERFACE_LINK_DIRECTORIES "${TURBO_PDFIUM_DIR}/lib")
    endif()

    # Page-image export + PPM decode: portable. page_image_encoder needs
    # turbojpeg (present on macOS too, e.g. brew jpeg-turbo); pdf_ppm is plain
    # C++. Neither touches the daemon, so neither belongs behind the POSIX
    # guard below — lumping them there is what made "no inotify" also mean "no
    # JPEG page export".
    find_library(TURBOJPEG_LIB turbojpeg)
    # Homebrew keeps jpeg-turbo keg-only, so the headers are NOT on the default
    # include path even when the library resolves. Find the header too and fail
    # the pair together, rather than linking fine and dying on #include.
    find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h
        HINTS /opt/homebrew/opt/jpeg-turbo/include /usr/local/opt/jpeg-turbo/include)
    if(TURBOJPEG_LIB AND TURBOJPEG_INCLUDE_DIR)
        target_sources(turbo_ocr_cpu PRIVATE
            src/pdf/render/pdf_ppm.cpp
            src/image/page_image_encoder.cpp)
        target_include_directories(turbo_ocr_cpu PUBLIC ${TURBOJPEG_INCLUDE_DIR})
        target_link_libraries(turbo_ocr_cpu PUBLIC ${TURBOJPEG_LIB})
        set(TURBO_HAVE_PAGE_EXPORT ON CACHE INTERNAL "")
    else()
        message(STATUS "libturbojpeg not found — PDF page-image export disabled "
                       "(install jpeg-turbo to enable)")
        set(TURBO_HAVE_PAGE_EXPORT OFF CACHE INTERNAL "")
    endif()

    # The DAEMON page renderer is Linux-bound: pdf_daemon drives worker
    # processes with sigtimedwait/pipe2/inotify, and pdf_renderer forks them.
    # The IN-PROCESS renderer is the same contract without any of that, so it is
    # what every platform lacking inotify uses — macOS and Windows alike.
    #
    # It was gated on APPLE, which left Windows with no renderer at all and PDF
    # rasterization answering 501 there. That was never a real platform limit:
    # the file's only non-portable calls were mkdtemp and unlink, both now
    # std::filesystem. docs/guides/windows-native-build.md had already reached
    # "reuse as-is" by inspection; this is that conclusion actually wired up.
    # The in-process arm and the selector build EVERYWHERE. The daemon arm is
    # added on top where its syscalls exist, and TURBO_PDF_DAEMON lets
    # pdf_renderer_select.cpp choose between them at RUN time (binary present
    # -> daemon, else in-process). Compiling only the daemon on Linux is what
    # made a build without scripts/setup/install_fastpdf2png.sh refuse to boot.
    include(CheckIncludeFileCXX)
    check_include_file_cxx("sys/inotify.h" TURBO_HAVE_INOTIFY)
    target_sources(turbo_ocr_cpu PRIVATE
        src/pdf/render/pdf_renderer_select.cpp
        src/pdf/render/pdf_renderer_inprocess.cpp)
    if(TURBO_HAVE_INOTIFY)
        target_sources(turbo_ocr_cpu PRIVATE
            src/pdf/render/pdf_renderer.cpp
            src/pdf/render/pdf_daemon.cpp)
        # PUBLIC, not PRIVATE: this macro switches DECLARATIONS in the public
        # pdf_renderer.h, so every TU that includes it must agree on the class
        # — the same reason USE_CPU_ONLY/TURBO_CPU_ONLY are PUBLIC below.
        target_compile_definitions(turbo_ocr_cpu PUBLIC TURBO_PDF_DAEMON)
    else()
        message(STATUS "no inotify — PDF page RENDERER is in-process PDFium only")
    endif()
    set(TURBO_HAVE_PDF_RENDER ON CACHE INTERNAL "")
    target_compile_options(turbo_ocr_cpu PRIVATE ${TURBO_OCR_CXX_FLAGS})
    # USE_CPU_ONLY: existing CPU-build switch (page_image_encoder etc.).
    # TURBO_CPU_ONLY: compiles the CUDA execution provider out of the shared
    # OrtSession TU (src/analysis/formula/ppformulanet/ort_session.cpp) so the CUDA-free CPU library
    # links only load_cpu() — the GPU build leaves it undefined and keeps load().
    target_compile_definitions(turbo_ocr_cpu PUBLIC USE_CPU_ONLY TURBO_CPU_ONLY)
    add_library(turbo_ocr::cpu ALIAS turbo_ocr_cpu)

    # The CPU-configure server is the unified multi-backend one — see
    # src/service/server/unified/unified_server.cmake, included from the backends section
    # below. (The old turboocr-cpu-server, its stages_cpu bootstrap, the
    # CpuOcrPipeline it drove, and the Phase-0 apple bench tools are deleted:
    # UnifiedOcrPipeline + turbo_bench replaced them.)

else()
    message(STATUS "Building GPU mode (CUDA + TensorRT)")

    find_package(CUDAToolkit REQUIRED)

    set(TENSORRT_DIR "/usr/local/tensorrt" CACHE PATH "Path to TensorRT")

    # Protobuf + gRPC discovery + proto codegen (shared with the CPU arm).
    # GATED exactly like the CPU arm at the `if(BUILD_SERVER)` above: the
    # generated proto is consumed by the gRPC service and the parity test, and
    # by nothing else. Ungated, this made BUILD_SERVER=OFF meaningless on the
    # GPU arm — turbo_setup_grpc_codegen() hard-requires protobuf + grpc++
    # (find_package(Protobuf REQUIRED), pkg_check_modules(GRPC REQUIRED)), so a
    # container without them died at CONFIGURE time, before a single .cu file
    # compiled. That is exactly what the NVIDIA python wheels do: they are the
    # first configure to reach this branch with BUILD_SERVER=OFF, because every
    # other wheel is forced onto the host path higher up.
    if(BUILD_SERVER)
        turbo_setup_grpc_codegen()
    endif()

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
        # break the formula backends at runtime). Same pin as Dockerfile.nvidia.
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

    # The CUDA kernel TUs, named ONCE. turbo_ocr_backend_nvidia globs
    # src/backends/nvidia/*.cu as a landing pad for genuinely new vendor-private
    # kernels and subtracts this list, because these four are already compiled
    # here. When the kernels moved under src/backends/nvidia/ the glob started
    # matching them as well: the same objects landed in two archives that
    # PUBLIC-link each other, and any target linking both (turbo_ocr_tests does;
    # turboocr-server does not) got a multiple-definition link failure.
    set(TURBO_GPU_CU_SRCS
        src/backends/nvidia/kernels_cuda/preprocess_kernels.cu
        src/backends/nvidia/kernels_cuda/reduce_kernels.cu
        src/backends/nvidia/kernels_cuda/ccl_kernels.cu
        src/backends/nvidia/kernels_cuda/jfa.cu)

    # --- GPU library ---
    add_library(turbo_ocr_gpu STATIC
        src/backends/nvidia/engine/trt_engine.cpp
        src/backends/nvidia/engine/engine_loader.cpp
        src/backends/nvidia/engine/onnx_to_trt.cpp
        src/backends/nvidia/engine/trt_engine_cache.cpp
        src/backends/nvidia/engine/trt_profiles.cpp
        src/backends/nvidia/stages/paddle_det.cpp
        src/backends/nvidia/stages/paddle_det_ccl.cpp
        src/backends/nvidia/stages/paddle_rec.cpp
        src/backends/nvidia/stages/paddle_cls.cpp
        src/backends/nvidia/stages/doc_orientation.cpp
        src/backends/nvidia/stages/paddle_layout.cpp
        # FFDetr field model. NOT device code — it is a plain ORT session, and
        # this arm already links ${ONNXRUNTIME_LIB} (see target_link_libraries
        # below). It is here because src/service/http/pdf/pdf_json.cpp is in
        # TURBO_HTTP_COMMON_SRCS — compiled into BOTH servers and passed
        # straight to add_executable, so its object can never be archive-dropped
        # — and it calls forms::FieldModel::load/run UNGUARDED (pdf_json.cpp:27,
        # :387). Without this line the GPU turboocr-server fails to link with
        # undefined FieldModel::{load,run,~FieldModel}: the only definition site
        # is field_model.cpp, which was listed solely in turbo_ocr_cpu (:485) —
        # a target that does not exist in the USE_CPU_ONLY=OFF configure.
        # No CI job compiles this arm and syntax_shims only type-checks, so
        # nothing would have caught it. See docs/notes/production-readiness §2.4.
        src/analysis/forms/field_model.cpp
        # NOTE: src/pipeline/ocr_pipeline_detail.cpp is deliberately NOT here.
        # It is device-free shared result policy and lives in turbo_ocr_common
        # (which this target links PUBLIC), so both orchestrations get the SAME
        # copy. Listing it here too would compile it twice.
        src/pipeline/finalize_deferred.cpp
        ${TURBO_GPU_CU_SRCS}
        # The page renderer is compiled UNCONDITIONALLY here (the GPU configure
        # is Linux-only, where inotify/sigtimedwait/pipe2 are a given) — so
        # TURBO_HAVE_PDF_RENDER is set below to match. It used to be set only
        # in the CPU-configure branch, which made the unified server on every
        # GPU build answer 501 "no PDF page RENDERER" for scanned PDFs while
        # the renderer's own object files sat inside the very same binary.
        # Found on the RTX 5090 bring-up.
        # BOTH arms are here for the same reason as the CPU target: the daemon
        # needs the fastpdf2png binary at run time, and a GPU box that never
        # ran install_fastpdf2png.sh must still serve PDF (and still BOOT)
        # rather than die in the PdfRenderer constructor. TURBO_PDF_DAEMON is
        # defined below so the selector knows the daemon arm is available.
        src/pdf/render/pdf_renderer_select.cpp
        src/pdf/render/pdf_renderer_inprocess.cpp
        src/pdf/render/pdf_renderer.cpp
        src/pdf/render/pdf_ppm.cpp
        src/pdf/render/pdf_daemon.cpp
        src/pdf/text/pdf_text_layer.cpp
        src/pdf/text/pdf_text_extract.cpp
        src/pdf/text/pdf_text_verify.cpp
        src/pdf/text/font_style.cpp
        src/pdf/text/font_match.cpp
        src/pdf/text/region_extract.cpp
        src/pdf/text/pdf_searchable.cpp
        src/pdf/pdf_documents.cpp
        src/pdf/text/pdf_searchable_encoding.cpp
        src/pdf/text/pdf_searchable_style.cpp
        src/pdf/text/pdf_searchable_shapes.cpp
        src/image/page_image_encoder.cpp
        # CUA router + table + formula stages (referenced by ocr_pipeline's
        # dispatch_router_). Pulled in here so turboocr-server links;
        # the operator's prior in-progress integration left these source
        # files on disk but not on the build target.
        src/pipeline/router/cua_router.cpp
        src/pipeline/router/cua_router_rules.cpp
        src/pipeline/router/region_features.cpp
        src/pipeline/router/routing_plan.cpp
        src/backends/nvidia/stages/table_cell.cpp
        src/backends/nvidia/stages/slanext_enc_split.cpp
        src/analysis/table/slanext/slanext_host_decode.cpp
        src/analysis/table/slanext/slanext_dict.cpp
        src/analysis/table/slanext/slanext_postprocess.cpp
        src/analysis/table/cell_matcher.cpp
        src/analysis/table/html_reconstruct.cpp
        src/analysis/table/table_cells.cpp
        src/backends/nvidia/stages/table_recognizer.cpp
        src/backends/nvidia/stages/slanext_table_recognizer.cpp
        src/backends/nvidia/stages/vlm_table_recognizer.cpp
        src/backends/nvidia/stages/vlm_table.cpp
        src/analysis/table/vlm/otsl_html.cpp
        src/backends/nvidia/stages/table_kernels.cu
        src/analysis/formula/ppformulanet/formula_tokenizer.cpp
        src/analysis/formula/ppformulanet/latex_normalize.cpp
        src/backends/nvidia/stages/formula_recognizer.cpp
        src/backends/nvidia/stages/auto_cjk_formula.cpp
        src/backends/nvidia/stages/vlm_formula.cpp
        src/analysis/vlm/vlm_client.cpp
        src/backends/nvidia/stages/ppformulanet_ort.cu
        src/backends/nvidia/stages/ppformulanet_decode.cu
        src/backends/nvidia/stages/ppformulanet_plusm.cu
        src/backends/nvidia/stages/ppformulanet_gate_bench.cu
        src/analysis/formula/ppformulanet/ppformulanet_preprocess.cpp
        src/analysis/formula/ppformulanet/ort_session.cpp
        src/analysis/vlm/crop_pool.cpp
        src/analysis/vlm/crop_pool_transport.cpp
        src/backends/nvidia/stages/openai_endpoint.cpp
    )
    # See the note on pdf_renderer.cpp above: the renderer is in this target,
    # so the unified server's PDF gate must say so.
    set(TURBO_HAVE_PDF_RENDER ON CACHE INTERNAL "")
    # The daemon arm is compiled into this target, so the selector may pick it.
    # PUBLIC for the same reason as the CPU target: it switches declarations in
    # the public pdf_renderer.h, so every consumer must see the same class.
    target_compile_definitions(turbo_ocr_gpu PUBLIC TURBO_PDF_DAEMON)
    # The SLANeXt host GRU decoder is a tight scalar GEMV/attention loop; let the
    # compiler vectorize the dot-product reductions (FP reassociation only —
    # accuracy-neutral, verified) under the default AVX2 (-march=x86-64-v3) target.
    # slanext_host_decode.cpp is the ONE decode source shared with the CPU build;
    # identical flags here keep the GPU/CPU token streams bit-identical.
    set_source_files_properties(src/backends/nvidia/stages/slanext_enc_split.cpp
        src/analysis/table/slanext/slanext_host_decode.cpp PROPERTIES
        COMPILE_OPTIONS "-ffast-math;-fopenmp-simd")
    target_include_directories(turbo_ocr_gpu PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
        ${TENSORRT_DIR}/include
        ${PROTO_GEN_DIR}
        ${TURBO_PDFIUM_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}   # ort_session.cpp: formula encoder+prep+step via ORT-CUDA
        # This target compiles src/backends/nvidia TUs, which include their own
        # headers as "nvidia/engine/...". Every other backend library sets this;
        # turbo_ocr_gpu did not, so the GPU arm could not compile trt_engine.cpp.
        "${CMAKE_SOURCE_DIR}/src/backends"
    )
    target_link_directories(turbo_ocr_gpu PUBLIC ${TENSORRT_DIR}/lib)
    # Link libpdfium.so directly for in-process text-layer extraction.
    # The shared lib is vendored in third_party/pdfium/lib and also placed
    # in bin/ by scripts/setup/install_fastpdf2png.sh (so native dev + Docker
    # builds both resolve it at runtime). Docker additionally copies it to
    # /usr/lib via ldconfig in Dockerfile.nvidia.
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
            "clients (src/backends/nvidia/stages/vlm_table.cpp, src/backends/nvidia/stages/vlm_formula.cpp) "
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
        ${TURBO_PDFIUM_LIB}
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
    # These suites need symbols that exist only in turbo_ocr_gpu (CUDA/TRT or
    # GPU-list-only TUs), so they build in the GPU configuration only.
    target_sources(turbo_ocr_tests PRIVATE
        tests/cpp/pipeline/test_finalize_deferred.cpp
        tests/cpp/pipeline/test_multi_rec.cpp
        tests/cpp/table/test_otsl_to_html.cpp
        tests/cpp/router/test_cua_router.cpp
        tests/cpp/formula/test_formula_preprocess.cpp
        tests/cpp/formula/test_formulanet_ar_loop.cpp
        tests/cpp/lang_cls/test_script_id.cpp
    )

    # NOTE: the GPU arm no longer builds its own server. It builds the ONE
    # unified server (src/service/server/unified/) with the nvidia plugin
    # supplying whichever pipeline PIECES CUDA implements better — exactly
    # how apple/intel/amd/cpu work. The second implementation is gone.
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

    # Hand-run tools (tools/checks/, tools/bench/) — EXCLUDE_FROM_ALL so the default `make`
    # (and the Docker image build) never compiles them into the production image; build
    # explicitly when needed. No add_test() drives any of these: the only thing in tools/
    # that ctest depends on is tools/bench/score_funsd.py, below.
    # plus-M FAST-path golden self-test driver (no server boot; gate_bench).
    add_executable(plusm_selftest EXCLUDE_FROM_ALL tools/checks/plusm_selftest.cpp)
    target_link_libraries(plusm_selftest PRIVATE turbo_ocr_gpu)

    # Formula backend swap check: FORMULA_BACKEND -> resolved routing engine.
    add_executable(formula_swap_check EXCLUDE_FROM_ALL tools/checks/formula_swap_check.cpp)
    target_link_libraries(formula_swap_check PRIVATE turbo_ocr_gpu)

    # Legacy standalone gRPC server removed — gRPC now runs from the main
    # binary via start_grpc_server() in grpc_service.h.

    # Benchmark tools. Gated like turbo_setup_grpc_codegen() above: they consume
    # the generated protos and link protobuf::libprotobuf, and EXCLUDE_FROM_ALL
    # does NOT save a BUILD_SERVER=OFF configure — target_link_libraries against
    # a nonexistent imported target is a hard error at GENERATE time, before
    # anything is built (this is exactly how the NVIDIA python wheels die when
    # these sit outside the gate: their containers have no protobuf/gRPC).
    if(BUILD_SERVER)
        add_executable(grpc_bench EXCLUDE_FROM_ALL
            tools/bench/grpc_bench.cpp
            ${PROTO_GEN_CC}
            ${GRPC_GEN_CC}
        )
        target_include_directories(grpc_bench PRIVATE ${PROTO_GEN_DIR})
        target_link_libraries(grpc_bench PRIVATE
            ${GRPC_LIBRARIES}
            protobuf::libprotobuf
        )

        add_executable(grpc_burst EXCLUDE_FROM_ALL
            tools/bench/grpc_burst.cpp
            ${PROTO_GEN_CC}
            ${GRPC_GEN_CC}
        )
        target_include_directories(grpc_burst PRIVATE ${PROTO_GEN_DIR})
        target_link_libraries(grpc_burst PRIVATE
            ${GRPC_LIBRARIES}
            protobuf::libprotobuf
        )
    endif()

    # diff_ccl: side-by-side comparator for CPU CCL=1 vs GPU CCL=2 outputs.
    # Loads an image, runs both detection paths, prints box counts, IoU stats,
    # and missed/extra boxes. Used to drive the GPU JFA pipeline accuracy.
    add_executable(diff_ccl EXCLUDE_FROM_ALL tools/checks/diff_ccl.cpp)
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
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/checks/test_nvjpeg_race.cpp)
        add_executable(test_nvjpeg_race EXCLUDE_FROM_ALL tools/checks/test_nvjpeg_race.cpp)
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
# The amd backend forces the FINAL LINK through ROCm's clang++/ld.lld (the
# executables carry HIP device code from turbo_ocr_backend_amd). GCC's slim-LTO
# archive members are LTO IR that only GNU ld/gold's plugin can read — ld.lld
# reports every symbol in them as undefined (nm still shows them, which makes
# it look like an ordering bug; it is not). So a g++ + amd configure must not
# produce GCC-LTO archives. Apple (Clang LTO + ld64) and the CUDA arm (GNU ld
# + plugin) are unaffected.
if(amd IN_LIST TURBO_BACKENDS AND _ipo_ok AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_ipo_ok FALSE)
    message(STATUS "IPO/LTO disabled: backend 'amd' links with ld.lld, which "
                   "cannot read ${CMAKE_CXX_COMPILER_ID} LTO archive members")
endif()
if(_ipo_ok AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling IPO/LTO for Release targets")
    set_property(TARGET turbo_ocr_common PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    # turboocr-server is NOT touched here in either arm: it is created by
    # unified_server.cmake, which runs far below this block, and that file sets
    # its own IPO/install/fetch_models. The GPU arm used to build a second,
    # native server target here; when the two were merged into one unified
    # server these three references (IPO, install, add_dependencies) were left
    # behind and pointed at a target that no longer exists.
    if(USE_CPU_ONLY)
        set_property(TARGET turbo_ocr_cpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        set_property(TARGET turbo_ocr_gpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
elseif(NOT _ipo_ok)
    message(STATUS "IPO/LTO not supported by this toolchain: ${_ipo_err}")
endif()

# --- Install rules --------------------------------------------------------
# Minimal layout so `cmake --install build --prefix /usr/local` works.
# turboocr-server installs itself in unified_server.cmake, for both configures.
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
# The fetch is a bash script, so it needs a bash that can SEE this source tree —
# which is not the same question as "is bash on PATH". On Windows the bash found
# on PATH is normally WSL's, living in a different filesystem namespace: it runs
# happily and then reports
#   /bin/bash: C:/turbo-v4/scripts/.../fetch_release_models.sh: No such file
# and the whole build stops at 100% on a step that only downloads weights. Ask
# the candidate interpreter whether it can stat the script rather than switching
# on the OS — that also catches a broken or sandboxed bash anywhere else.
if(FETCH_MODELS)
    find_program(TURBO_BASH bash)
    if(TURBO_BASH)
        execute_process(
            COMMAND "${TURBO_BASH}" -c
                    "test -f '${CMAKE_SOURCE_DIR}/scripts/models/fetch/fetch_release_models.sh'"
            RESULT_VARIABLE _bash_sees_tree
            OUTPUT_QUIET ERROR_QUIET)
    else()
        set(_bash_sees_tree 1)
    endif()
    if(NOT _bash_sees_tree EQUAL 0)
        message(STATUS
            "FETCH_MODELS: no bash that can read this source tree "
            "(found: '${TURBO_BASH}') — skipping the automatic model fetch. "
            "Download the PP-OCRv6 bundle into models/ yourself; see "
            "docs/getting-started/install.md.")
        set(FETCH_MODELS OFF)
    endif()
endif()
if(FETCH_MODELS)
    set(MODELS_SENTINEL "${CMAKE_SOURCE_DIR}/models/.v6_stamp")
    add_custom_command(
        OUTPUT ${MODELS_SENTINEL}
        COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/models/fetch/fetch_release_models.sh
        COMMAND ${CMAKE_COMMAND} -E touch ${MODELS_SENTINEL}
        DEPENDS ${CMAKE_SOURCE_DIR}/scripts/models/fetch/fetch_release_models.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Fetching PP-OCRv6 models from pinned GitHub Release"
        VERBATIM
    )
    add_custom_target(fetch_models ALL DEPENDS ${MODELS_SENTINEL})
    # turboocr-server hooks fetch_models in unified_server.cmake, which runs
    # after this block, for both configures.
endif()

# (Python bindings are defined at the end of this file, after the backend
# targets they wrap — see "Python bindings (nanobind)" below.)

# ===========================================================================
# MULTI-BACKEND ARCHITECTURE — the ONE device-agnostic orchestration
# (src/pipeline/unified/unified_ocr_pipeline.cpp) + per-vendor backends under
# src/backends/. See docs/architecture/multi-backend.md.
#
# TURBO_BACKENDS is a LIST, not a scalar: multiple backends co-link into ONE
# binary (src/backend/backend_registry.cpp is the single link-time registry;
# each vendor ships a tiny self-registering TU) and the vendor is selected at
# RUNTIME (--backend / TURBO_BACKEND, empty = auto-detect by priority).
#
# Default: the host platform's natural set — "cpu;apple" on macOS, "cpu" on a
# CPU-only Linux configure. The GPU (CUDA) configure defaults to the native
# TensorRT server; add nvidia (and/or cpu) explicitly to build the seam-based
# targets there, which are UNVERIFIED on CUDA hardware (see
# tests/cpp/backends/README.md, "first-configure checklist").
# ===========================================================================
if(USE_CPU_ONLY)
    if(APPLE)
        set(_turbo_backends_default "cpu;apple")
    else()
        set(_turbo_backends_default "cpu")
    endif()
else()
    set(_turbo_backends_default "")
endif()
set(TURBO_BACKENDS "${_turbo_backends_default}" CACHE STRING
    "Semicolon-separated list of backends to build: cpu;apple;nvidia;amd;intel")

set(_turbo_known_backends cpu apple nvidia amd intel)
foreach(_b IN LISTS TURBO_BACKENDS)
    if(NOT _b IN_LIST _turbo_known_backends)
        message(FATAL_ERROR
            "TURBO_BACKENDS contains unknown backend '${_b}'. "
            "Known: ${_turbo_known_backends}")
    endif()
endforeach()

if(TURBO_BACKENDS)
    message(STATUS "Backends: ${TURBO_BACKENDS}")

    # AMD is written (src/backends/amd/: HIP queue/allocator/kernels + MIGraphX
    # engine) and HAS a CMake target (turbo_ocr_backend_amd, defined below) — it
    # just has never been configured or compiled, because it needs ROCm hardware
    # to validate. Warn loudly rather than let anyone mistake it for exercised
    # code; see the per-backend note at the target definition.
    if(amd IN_LIST TURBO_BACKENDS)
        # AMD is IMPLEMENTED but UNVERIFIED: it has never been configured or
        # compiled — no ROCm, no hipcc, no AMD GPU on the machines this was
        # written on. It is a real target (not a FATAL_ERROR) so an operator with
        # ROCm can build it; it is loud so nobody mistakes it for exercised code.
        # NOTE: ROCm 7.14 moved to the modular "TheRock" build system — verify
        # the HIP sources against that layout on first bring-up.
        message(WARNING
            "Backend 'amd' is UNVERIFIED — never configured or compiled on ROCm "
            "hardware. Needs hipcc + MIGraphX. See src/backends/amd/README.md.")
    endif()
    if(apple IN_LIST TURBO_BACKENDS AND NOT APPLE)
        message(FATAL_ERROR "TURBO_BACKENDS contains 'apple' but this is not an Apple platform")
    endif()
    if(nvidia IN_LIST TURBO_BACKENDS)
        # Bring-up done 2026-08-10 on Arch / CUDA 13.3.33 / TensorRT 10.15.1.29 /
        # driver 610.43.02 / RTX 5090 at sm_120. All nine steps of the
        # first-configure checklist in tests/cpp/backends/README.md pass:
        # both backends survive the link (the WHOLE_ARCHIVE registrar hazard),
        # TRT engines build and cache, conformance runs cpu-vs-nvidia, and
        # FUNSD-tiny scores 85.37% against the 85.0 gate floor. STATUS, not a
        # warning — one machine is not every machine, but it is no longer
        # "never compiled".
        message(STATUS
            "Backend 'nvidia': verified on CUDA hardware (RTX 5090, CUDA 13.3, "
            "TensorRT 10.15.1). Other SM/toolkit combinations remain untested — "
            "see the checklist in tests/cpp/backends/README.md.")
    endif()
    if(intel IN_LIST TURBO_BACKENDS AND NOT TURBO_OCR_HAS_SYCL)
        message(STATUS
            "Backend 'intel': OpenVINO engine ON, SYCL kernels OFF "
            "(host-fallback pre/post). Set OV_DEVICE=CPU|GPU|NPU at runtime.")
    endif()

    # Warning/optimization flags, scoped to C++ so they do not leak onto
    # Objective-C++ (where -Wpedantic fires on Apple's own system headers) or
    # CUDA.
    set(TURBO_BACKEND_CXX_FLAGS $<$<COMPILE_LANGUAGE:CXX>:${TURBO_OCR_CXX_FLAGS}>)

    # -----------------------------------------------------------------------
    # turbo_ocr_pipeline — the ONE device-agnostic orchestration + the ONE
    # link-time backend registry. No CUDA, no Metal, no ORT session code: it is
    # written entirely against the seam interfaces (include/turbo_ocr/backend).
    # -----------------------------------------------------------------------
    add_library(turbo_ocr_pipeline STATIC
        # UnifiedOcrPipeline is ONE class across four TUs, split along the seams
        # its own banner comments already named (single-image path / dispatch /
        # batch / stage bootstrap) once the single file passed the 900-line
        # ceiling. Adding a member function means picking the TU whose seam it
        # belongs to, not opening a fifth.
        src/pipeline/unified/unified_ocr_pipeline.cpp
        src/pipeline/unified/unified_pipeline_dispatch.cpp
        src/pipeline/unified/unified_pipeline_batch.cpp
        src/pipeline/unified/unified_pipeline_stages.cpp
        src/pipeline/unified/make_infer_func.cpp
        src/pipeline/unified/unified_pipeline_pool.cpp
        src/pipeline/unified/stage_batcher.cpp
        src/backend/backend_registry.cpp
        # The CUA router is referenced by UnifiedOcrPipeline's dispatch even on
        # the text-only path; device-neutral.
        src/pipeline/router/routing_plan.cpp
        src/pipeline/router/cua_router.cpp
        src/pipeline/router/cua_router_rules.cpp
        src/pipeline/router/region_features.cpp
    )
    target_link_libraries(turbo_ocr_pipeline PUBLIC
        turbo_ocr_common
    )
    # NOTE (removed): Drogon::Drogon used to be PUBLIC here, with the rationale
    # "server::InferOptions pulls <drogon/HttpRequest.h>". It does not any more:
    # InferOptions moved to the transport-free include/turbo_ocr/validation/
    # infer_options.h and every TU in this archive now names that instead.
    # Verified with the real compile flags + `-M -MG` over all of
    # unified_ocr_pipeline / make_infer_func / stage_batcher / vlm_factory /
    # backend_registry / the router TUs: ZERO drogon, trantor or json/json.h in
    # any include closure. The one target that needs Drogon links it itself
    # (src/service/server/unified/unified_server.cmake, and the GPU server above); the offline
    # binaries that link this archive — turbo_backend_probe, turbo_bench,
    # turbo_conformance, turbo_golden, _turboocr — deliberately must not.
    if(ONNXRUNTIME_INCLUDE_DIR)
        target_include_directories(turbo_ocr_pipeline PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
    endif()
    # The pipeline compiles against server/validation headers whose behavior is
    # switched by these two macros; PUBLIC so every backend/test/server target
    # in the seam world sees the same build flavor. (In the GPU configure they
    # must stay OFF: TURBO_CPU_ONLY would compile the CUDA execution provider
    # out of the shared OrtSession header, and USE_CPU_ONLY would flip the
    # routing policy to the CPU build's answers.)
    if(USE_CPU_ONLY)
        target_compile_definitions(turbo_ocr_pipeline PUBLIC USE_CPU_ONLY TURBO_CPU_ONLY)
    endif()
    target_compile_options(turbo_ocr_pipeline PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
    # vlm_factory.cpp -> openai_policy.h -> crop_pool.h -> <curl/curl.h>.
    # Without libcurl the header chain cannot compile, so vlm_factory.cpp is
    # only part of this archive when curl is available. Binaries that link
    # the pipeline without curl (the Python extension, offline test tools)
    # supply vlm_factory_link_support.cpp stubs for the two factory symbols.
    if(TARGET CURL::libcurl)
        target_sources(turbo_ocr_pipeline PRIVATE
            src/pipeline/unified/vlm_factory.cpp)
        target_link_libraries(turbo_ocr_pipeline PUBLIC CURL::libcurl)
    else()
        target_sources(turbo_ocr_pipeline PRIVATE
            tests/cpp/backends/vlm_factory_link_support.cpp)
    endif()
    add_library(turbo_ocr::pipeline ALIAS turbo_ocr_pipeline)

    # UnifiedPipelinePool's admission logic (tests/cpp/pipeline/test_pipeline_pool.cpp)
    # lives in this library and is device-neutral. Linked HERE, not beside the
    # other test link lines: those run earlier in this file, where
    # if(TARGET turbo_ocr_pipeline) is still false and would link nothing.
    if(TARGET turbo_ocr_tests)
        # Source and link together, so these TUs can never be compiled into a
        # configure that has nothing to satisfy them.
        target_sources(turbo_ocr_tests PRIVATE
            tests/cpp/pipeline/test_pipeline_pool.cpp
            tests/cpp/backends/test_seam_conformance.cpp
            tests/cpp/backends/test_backend_caps.cpp)
        target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_pipeline)
    endif()

    # vlm_factory.cpp (kind:openai table/formula recognisers) lives in this
    # ARCHIVE on purpose: offline binaries link
    # tests/cpp/backends/vlm_factory_link_support.cpp (a nullptr stub for the same
    # two factories) and the linker then never pulls the real member — no
    # libcurl, no VLM crop pool.

    # -----------------------------------------------------------------------
    # turbo_ocr_cpu_host — the main-tree CPU stage classes, materialised in a
    # GPU configure (where turbo_ocr_cpu does not exist) so the cpu backend can
    # still be co-linked next to nvidia. UNVERIFIED on CUDA hardware.
    # It is turbo_ocr_cpu's source list MINUS the TUs turbo_ocr_gpu already
    # compiles (the shared formula/table host files) and the PDF TUs.
    # -----------------------------------------------------------------------
    # ONE predicate for "this configure needs the main-tree CpuPaddle* classes",
    # used by both the target below and turbo_ocr_backend_onnx further down.
    # They were two hand-copied conditions and had drifted: the consumer listed
    # cpu/apple/intel/nvidia, the producer only cpu, so -DTURBO_BACKENDS=nvidia
    # hit the consumer's FATAL_ERROR with no way to satisfy it.
    set(TURBO_NEEDS_HOST_STAGES FALSE)
    # amd belongs in this list for the same reason the comment above records
    # for nvidia: turbo_ocr_backend_amd PUBLIC-links turbo_ocr_backend_onnx,
    # so a -DTURBO_BACKENDS=amd (without cpu) configure needs the host stage
    # set materialised or the link target simply does not exist.
    foreach(_b cpu apple intel nvidia amd)
        if(${_b} IN_LIST TURBO_BACKENDS)
            set(TURBO_NEEDS_HOST_STAGES TRUE)
        endif()
    endforeach()

    if(TURBO_NEEDS_HOST_STAGES AND NOT TARGET turbo_ocr_cpu)
        message(STATUS "No turbo_ocr_cpu in this configure (GPU arm) — "
                       "building turbo_ocr_cpu_host for the ONNX stage set [UNVERIFIED]")
        add_library(turbo_ocr_cpu_host STATIC
            src/onnx/ort_engine.cpp
            src/analysis/detection/ort_paddle_det.cpp
            src/analysis/recognition/ort_paddle_rec.cpp
            src/analysis/classification/ort_paddle_cls.cpp
            src/analysis/classification/ort_doc_orientation.cpp
            src/analysis/layout/ort_paddle_layout.cpp
            src/analysis/formula/ppformulanet/ort_formula_recognizer.cpp
            src/analysis/table/slanext/ort_slanext_table.cpp
        )
        target_include_directories(turbo_ocr_cpu_host PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}")
        target_link_libraries(turbo_ocr_cpu_host PUBLIC
            turbo_ocr_common "${ONNXRUNTIME_LIB}" Threads::Threads)
        # PRIVATE, not PUBLIC as on turbo_ocr_cpu: the defs must reach these 8
        # host TUs and NOT the CUDA TUs linked beside them.
        target_compile_definitions(turbo_ocr_cpu_host PRIVATE USE_CPU_ONLY TURBO_CPU_ONLY)
        target_compile_options(turbo_ocr_cpu_host PRIVATE ${TURBO_OCR_CXX_FLAGS})
    endif()
    if(TARGET turbo_ocr_cpu)
        set(TURBO_CPU_HOST_LIB turbo_ocr_cpu)
    else()
        set(TURBO_CPU_HOST_LIB turbo_ocr_cpu_host)
    endif()

    # turbo_ocr_host_kernels — the seam's HOST FALLBACK op set, SHARED by every
    # backend that lacks a native kernel for an op (backend/kernels.h caps()
    # declares which). Giving each backend its own copy would be the "fix the
    # same bug once per backend" failure the dedup rule exists to prevent.
    if(TURBO_BACKENDS)
        add_library(turbo_ocr_host_kernels STATIC src/backends/cpu/kernels_host/host_kernels.cpp)
        target_include_directories(turbo_ocr_host_kernels PUBLIC "${CMAKE_SOURCE_DIR}/src/backends")
        target_link_libraries(turbo_ocr_host_kernels PUBLIC
            turbo_ocr_pipeline
            ${OpenCV_LIBS})
        target_compile_options(turbo_ocr_host_kernels PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
    endif()

    # -----------------------------------------------------------------------
    # turbo_ocr_backend_cpu  (OpenCV pre/post + ORT-CPU; the portable fallback)
    # -----------------------------------------------------------------------
    # turbo_ocr_backend_onnx — THE shared "fast" path (backend/engine_mode.h):
    # the .onnx through an ONNX Runtime execution provider, with host pre/post.
    # It is NOT the cpu backend: NVIDIA-on-CUDA-EP, Intel-on-OpenVINO,
    # Apple-on-CoreML and AMD-on-MIGraphX are all THIS code with a different
    # EpConfig, so every vendor links it and none of them reimplements it.
    # (The cpu backend is simply this set on the default provider.)
    if(TURBO_NEEDS_HOST_STAGES)
        if(NOT TARGET turbo_ocr_cpu AND NOT TARGET turbo_ocr_cpu_host)
            message(FATAL_ERROR
                "The shared ONNX stage set wraps the main-tree CpuPaddle* classes, "
                "but neither turbo_ocr_cpu nor turbo_ocr_cpu_host was built. "
                "TURBO_NEEDS_HOST_STAGES and this check must stay in sync.")
        endif()
        add_library(turbo_ocr_backend_onnx STATIC
            src/backends/cpu/queue/host_device_queue.cpp
            src/backends/cpu/memory/host_allocator.cpp
            src/backends/cpu/stages/cpu_stages.cpp
            src/backends/cpu/engine/cpu_engine_adapter.cpp
            src/backends/cpu/stages/cpu_table_recognizer.cpp
            src/backends/cpu/stages/cpu_formula_recognizer.cpp
        )
        target_include_directories(turbo_ocr_backend_onnx PUBLIC "${CMAKE_SOURCE_DIR}/src/backends")
        target_link_libraries(turbo_ocr_backend_onnx PUBLIC
            turbo_ocr_pipeline
            turbo_ocr_host_kernels      # SHARED host fallback ops
            ${TURBO_CPU_HOST_LIB}       # the wrapped main-tree CpuPaddle* classes
        )
        target_compile_options(turbo_ocr_backend_onnx PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
    endif()

    if(cpu IN_LIST TURBO_BACKENDS)
        add_library(turbo_ocr_backend_cpu STATIC
            src/backends/cpu/backend/cpu_backend.cpp
            src/backends/cpu/backend/cpu_backend_registry.cpp   # <- registrar; needs WHOLE_ARCHIVE
        )
        target_link_libraries(turbo_ocr_backend_cpu PUBLIC turbo_ocr_backend_onnx)
        target_compile_options(turbo_ocr_backend_cpu PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
        add_library(turbo_ocr::backend_cpu ALIAS turbo_ocr_backend_cpu)
    endif()


    # -----------------------------------------------------------------------
    # turbo_ocr_backend_amd  (ROCm/HIP + MIGraphX) — UNVERIFIED, never compiled.
    #
    # Same shape as every other vendor: it supplies only the pieces HIP does
    # better (kernels, MIGraphX engine, allocator, queue, stages) and borrows the
    # generic host implementation for the rest via turbo_ocr_backend_onnx /
    # turbo_ocr_host_kernels — exactly how apple and intel are wired.
    #
    # The .hip translation units need the HIP language enabled, which only exists
    # once ROCm's CMake package is found; guard so a non-ROCm configure fails with
    # a clear message instead of an unknown-extension error deep in the generator.
    # -----------------------------------------------------------------------
    if(amd IN_LIST TURBO_BACKENDS)
        find_package(hip REQUIRED)
        find_package(migraphx REQUIRED)
        enable_language(HIP)
        set_source_files_properties(
            src/backends/amd/kernels_hip/ccl_kernels.hip
            src/backends/amd/kernels_hip/jfa.hip
            src/backends/amd/kernels_hip/preprocess_kernels.hip
            src/backends/amd/kernels_hip/reduce_kernels.hip
            src/backends/amd/kernels_hip/table_kernels.hip
            PROPERTIES LANGUAGE HIP)
        add_library(turbo_ocr_backend_amd STATIC
            src/backends/amd/queue/hip_queue.cpp
            src/backends/amd/memory/hip_allocator.cpp
            src/backends/amd/kernels_hip/hip_kernels.cpp
            src/backends/amd/kernels_hip/ccl_kernels.hip
            src/backends/amd/kernels_hip/jfa.hip
            src/backends/amd/kernels_hip/preprocess_kernels.hip
            src/backends/amd/kernels_hip/reduce_kernels.hip
            src/backends/amd/kernels_hip/table_kernels.hip
            src/backends/amd/engine/migraphx_engine.cpp
            src/backends/amd/stages/rocm_stages.cpp
            src/backends/amd/backend/rocm_backend.cpp
            src/backends/amd/backend/amd_backend_registry.cpp   # <- registrar
        )
        target_include_directories(turbo_ocr_backend_amd PUBLIC "${CMAKE_SOURCE_DIR}/src/backends")
        target_compile_definitions(turbo_ocr_backend_amd PRIVATE __HIP_PLATFORM_AMD__=1)
        target_link_libraries(turbo_ocr_backend_amd PUBLIC
            turbo_ocr_pipeline
            turbo_ocr_backend_onnx      # the SHARED fast path
            turbo_ocr_host_kernels      # SHARED host fallback ops
            # hip::host, NOT hip::device: hip::device injects `-x hip` compile
            # options onto every CXX source in the target, which g++ rejects
            # outright ("language hip not recognized") — found on the first
            # real configure. The plain .cpp TUs here use only the HIP HOST
            # API (headers + libamdhip64), which is exactly hip::host; the
            # .hip TUs are compiled by CMake's HIP language support and get
            # their device flags from CMAKE_HIP_ARCHITECTURES, not from this
            # link line.
            hip::host
            migraphx::c
            ${OpenCV_LIBS}
        )
        target_compile_options(turbo_ocr_backend_amd PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
        add_library(turbo_ocr::backend_amd ALIAS turbo_ocr_backend_amd)
    endif()

    # -----------------------------------------------------------------------
    # turbo_ocr_backend_intel  (OpenVINO Runtime; SYCL kernels optional)
    # Device is chosen at RUNTIME via OV_DEVICE (CPU|GPU|NPU) — one binary must
    # run CPU-heavy on an iGPU part and GPU-heavy on a discrete Arc without a
    # rebuild. Setup + measurements: src/backends/intel/SETUP.md.
    # -----------------------------------------------------------------------
    if(intel IN_LIST TURBO_BACKENDS)
        find_package(OpenVINO REQUIRED COMPONENTS Runtime)
        add_library(turbo_ocr_backend_intel STATIC
            src/backends/intel/queue/l0_device_queue.cpp
            src/backends/intel/memory/l0_allocator.cpp
            src/backends/intel/kernels_sycl/sycl_kernels.cpp
            src/backends/intel/engine/openvino_engine.cpp
            src/backends/intel/stages/intel_stages.cpp
            src/backends/intel/stages/intel_stages_structure.cpp
            src/backends/intel/backend/intel_backend.cpp
            src/backends/intel/backend/intel_backend_registry.cpp   # <- registrar
        )
        target_include_directories(turbo_ocr_backend_intel PUBLIC "${CMAKE_SOURCE_DIR}/src/backends")
        target_compile_definitions(turbo_ocr_backend_intel PRIVATE TURBO_OCR_HAS_OPENVINO=1)
        target_link_libraries(turbo_ocr_backend_intel PUBLIC
            turbo_ocr_pipeline
            turbo_ocr_backend_onnx      # the SHARED fast path (OpenVINO EP)
            turbo_ocr_host_kernels      # SHARED host fallback ops (no-SYCL build)
            openvino::runtime
            ${OpenCV_LIBS}
        )
        target_compile_options(turbo_ocr_backend_intel PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
        add_library(turbo_ocr::backend_intel ALIAS turbo_ocr_backend_intel)

        # Functional test of OpenVINOEngine ALONE with the real ONNX models —
        # isolates "is the engine wrapper correct" from "is the stage wiring
        # correct", which an end-to-end F1 cannot distinguish.
        add_executable(ov_engine_probe src/backends/intel/probes/ov_engine_probe.cpp)
        target_link_libraries(ov_engine_probe PRIVATE
            turbo_ocr_backend_intel turbo_ocr_pipeline openvino::runtime ${OpenCV_LIBS})
        target_compile_definitions(ov_engine_probe PRIVATE TURBO_OCR_HAS_OPENVINO=1)
        target_compile_options(ov_engine_probe PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
    endif()

    # -----------------------------------------------------------------------
    # turbo_ocr_backend_apple  (Metal / MPSGraph / ANE)
    # -----------------------------------------------------------------------
    if(apple IN_LIST TURBO_BACKENDS)
        enable_language(OBJCXX)
        set(CMAKE_OBJCXX_STANDARD 20)
        set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

        # shaders.metal -> turbo_apple.metallib, emitted next to the executables
        # so AppleBackend finds it without the operator exporting TURBO_APPLE_METALLIB.
        set(TURBO_APPLE_METALLIB "${CMAKE_BINARY_DIR}/turbo_apple.metallib")
        add_custom_command(
            OUTPUT "${CMAKE_BINARY_DIR}/shaders.air"
            COMMAND xcrun -sdk macosx metal -O2 -c
                    "${CMAKE_SOURCE_DIR}/src/backends/apple/kernels_metal/shaders.metal"
                    -o "${CMAKE_BINARY_DIR}/shaders.air"
            DEPENDS "${CMAKE_SOURCE_DIR}/src/backends/apple/kernels_metal/shaders.metal"
            COMMENT "metal: src/backends/apple/kernels_metal/shaders.metal -> shaders.air"
            VERBATIM)
        add_custom_command(
            OUTPUT "${TURBO_APPLE_METALLIB}"
            COMMAND xcrun -sdk macosx metallib
                    "${CMAKE_BINARY_DIR}/shaders.air"
                    -o "${TURBO_APPLE_METALLIB}"
            DEPENDS "${CMAKE_BINARY_DIR}/shaders.air"
            COMMENT "metallib: shaders.air -> turbo_apple.metallib"
            VERBATIM)
        add_custom_target(turbo_apple_metallib ALL DEPENDS "${TURBO_APPLE_METALLIB}")
        install(FILES "${TURBO_APPLE_METALLIB}" DESTINATION bin)

        # EXPLICIT, not file(GLOB): the sources live in per-concern subdirectories
        # now (backend/ engine/ kernels_metal/ memory/ queue/ stages/ support/), and
        # the old non-recursive glob would have silently dropped every one of them
        # and linked a backend with no stages. A recursive glob would work but
        # still hides the source set from review; every other vendor here is an
        # explicit list, so this one is too. The registrar is in the SAME list
        # because both consumers below (add_library, LANGUAGE OBJCXX) need it.
        set(TURBO_APPLE_SRC
            src/backends/apple/backend/apple_backend.mm
            src/backends/apple/backend/apple_backend_registry.cpp   # <- registrar
            src/backends/apple/support/apple_contention.mm
            src/backends/apple/support/metal_common.mm
            src/backends/apple/support/coreml_compile.mm
            src/backends/apple/engine/ane_rec_engine.mm
            src/backends/apple/engine/mps_engine.mm
            src/backends/apple/kernels_metal/metal_kernels.mm
            src/backends/apple/memory/metal_allocator.mm
            src/backends/apple/memory/metal_image.mm
            src/backends/apple/queue/metal_device_queue.mm
            src/backends/apple/stages/mps_detector.mm
            src/backends/apple/stages/mps_stages.mm
        )
        add_library(turbo_ocr_backend_apple STATIC ${TURBO_APPLE_SRC})
        # The registrar is deliberately .cpp but reaches Objective-C++ headers,
        # so compile it as OBJCXX explicitly. ARC only for Objective-C++ — a
        # blanket -fobjc-arc is an error on plain C++ TUs.
        set_source_files_properties(${TURBO_APPLE_SRC} PROPERTIES LANGUAGE OBJCXX)
        target_compile_options(turbo_ocr_backend_apple PRIVATE
            $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
            $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall>
            ${TURBO_BACKEND_CXX_FLAGS})
        target_include_directories(turbo_ocr_backend_apple PUBLIC
            "${CMAKE_SOURCE_DIR}/src/backends")
        target_link_libraries(turbo_ocr_backend_apple PUBLIC
            turbo_ocr_pipeline
            turbo_ocr_backend_onnx      # the SHARED fast path (CoreML EP)
            "-framework Foundation" "-framework Metal"
            "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph"
            "-framework CoreML"
            "-framework ImageIO" "-framework CoreGraphics")
        add_dependencies(turbo_ocr_backend_apple turbo_apple_metallib)
        add_library(turbo_ocr::backend_apple ALIAS turbo_ocr_backend_apple)
    endif()

    # -----------------------------------------------------------------------
    # turbo_ocr_backend_nvidia  (CUDA / TensorRT / ORT-CUDA / nvJPEG)
    #
    # *** NEVER COMPILED — written on a machine with no CUDA toolkit. *** It
    # WRAPS turbo_ocr_gpu (TrtEngine, PaddleDet/Rec/Cls, the .cu kernels); it
    # does not recompile any of it. First configure on real hardware is
    # bring-up: tests/cpp/backends/README.md, "first-configure checklist".
    # -----------------------------------------------------------------------
    if(nvidia IN_LIST TURBO_BACKENDS)
        if(NOT TARGET turbo_ocr_gpu)
            message(FATAL_ERROR
                "Backend 'nvidia' wraps the main-tree CUDA/TensorRT classes in "
                "turbo_ocr_gpu — configure WITHOUT -DUSE_CPU_ONLY=ON, on a box "
                "with CUDA + TensorRT + a CUDA-enabled ONNX Runtime.")
        endif()
        get_property(_turbo_langs GLOBAL PROPERTY ENABLED_LANGUAGES)
        if(NOT "CUDA" IN_LIST _turbo_langs)
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 20)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --expt-relaxed-constexpr")
        endif()
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75)  # Turing floor; PTX JITs forward
        endif()
        find_package(CUDAToolkit REQUIRED)
        if(NOT EXISTS "${TENSORRT_DIR}/include/NvInfer.h")
            message(FATAL_ERROR
                "Backend 'nvidia': no NvInfer.h under ${TENSORRT_DIR}/include. "
                "Point -DTENSORRT_DIR at the TensorRT SDK root (same cache "
                "variable the main GPU build uses).")
        endif()
        if(NOT ONNXRUNTIME_CUDA_EP)
            message(FATAL_ERROR
                "Backend 'nvidia': ONNXRUNTIME_CUDA_EP is unset — this ONNX "
                "Runtime has no CUDA execution provider; OrtCudaEngine would "
                "link and then fail at runtime.")
        endif()
        # Landing pad for a genuinely NEW nvidia-private kernel. The four
        # existing ones (TURBO_GPU_CU_SRCS) are compiled by turbo_ocr_gpu and
        # are subtracted below: this target PUBLIC-links turbo_ocr_gpu, so
        # letting the glob take them too puts identical objects in two archives
        # and anything linking both fails with multiple definitions.
        #
        # Removing them from turbo_ocr_gpu instead is NOT the alternative:
        # TURBO_BACKENDS defaults to "" on the GPU configure and
        # docker/Dockerfile.nvidia passes none, so turbo_ocr_backend_nvidia is
        # often never created and the shipping product would lose every kernel.
        #
        # GLOB_RECURSE, not GLOB: the vendor sources are in per-concern
        # subdirectories, so a flat glob would miss a new .cu added under
        # nvidia/kernels_cuda/ — exactly where a new one would land.
        file(GLOB_RECURSE TURBO_NVIDIA_CU CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/backends/nvidia/*.cu")
        foreach(_cu IN LISTS TURBO_GPU_CU_SRCS)
            list(REMOVE_ITEM TURBO_NVIDIA_CU "${CMAKE_SOURCE_DIR}/${_cu}")
        endforeach()
        add_library(turbo_ocr_backend_nvidia STATIC
            src/backends/nvidia/support/nv_image_decode.cpp
            src/backends/nvidia/queue/cuda_device_queue.cpp
            src/backends/nvidia/memory/cuda_allocator.cpp
            src/backends/nvidia/kernels_cuda/cuda_kernels.cpp
            src/backends/nvidia/engine/trt_engine_adapter.cpp
            src/backends/nvidia/engine/ort_cuda_engine.cpp
            src/backends/nvidia/stages/nv_stages.cpp
            # table + formula are a pimpl split across the old/new header
            # generations: each pair must land in the SAME library but stay
            # SEPARATE TUs. See src/backends/nvidia/README.md.
            src/backends/nvidia/stages/nv_table_recognizer.cpp
            src/backends/nvidia/stages/nv_table_recognizer_impl.cpp
            src/backends/nvidia/stages/nv_formula_recognizer.cpp
            src/backends/nvidia/stages/nv_formula_recognizer_impl.cpp
            src/backends/nvidia/backend/cuda_backend.cpp
            src/backends/nvidia/backend/nv_backend_registry.cpp   # <- registrar
            # cuda_backend.cpp calls server::probe_nvjpeg / make_gpu_image_decoder,
            # defined in stages_gpu.cpp (compiled into the server executable, not
            # turbo_ocr_gpu) — without it every executable gets two undefined syms.
            ${TURBO_NVIDIA_CU}
        )
        if(TURBO_NVIDIA_CU)
            set_source_files_properties(${TURBO_NVIDIA_CU} PROPERTIES LANGUAGE CUDA)
            set_target_properties(turbo_ocr_backend_nvidia PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        endif()
        target_include_directories(turbo_ocr_backend_nvidia PUBLIC
            "${CMAKE_SOURCE_DIR}/src/backends")
        target_link_libraries(turbo_ocr_backend_nvidia PUBLIC
            turbo_ocr_pipeline
            turbo_ocr_backend_onnx      # the SHARED fast path (CUDA EP)
            turbo_ocr_gpu               # the wrapped TRT/CUDA classes + their deps
            CUDA::cudart
        )
        target_compile_options(turbo_ocr_backend_nvidia PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
        add_library(turbo_ocr::backend_nvidia ALIAS turbo_ocr_backend_nvidia)
    endif()

    # NVIDIA runtime search paths for every executable that links the backend.
    # --disable-new-dtags is LOAD-BEARING: libnvinfer dlopen()s its builder
    # resource lib by bare name, which consults DT_RPATH but not DT_RUNPATH.
    function(turbo_nvidia_runtime_paths target)
        if(NOT TARGET turbo_ocr_backend_nvidia)
            return()
        endif()
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH "${TENSORRT_DIR}/lib"
            INSTALL_RPATH "${TENSORRT_DIR}/lib;${ONNXRUNTIME_LIB_DIR}")
        target_link_options(${target} PRIVATE "-Wl,--disable-new-dtags")
    endfunction()

    # Force-link helper: a backend's registration TU defines no referenced
    # symbol, so a static archive legitimately drops it. WHOLE_ARCHIVE is not
    # an optimisation here, it is the correctness requirement.
    function(turbo_link_backends target)
        # Dedupe first: callers pass `${TURBO_BACKENDS} cpu` and TURBO_BACKENDS
        # usually already contains cpu. ld64 silently coalesces the same
        # archive force-loaded twice, but ld.lld (the amd link driver) errors
        # with "duplicate symbol" on every member of the twice-whole-archived
        # library.
        set(_bs ${ARGN})
        list(REMOVE_DUPLICATES _bs)
        foreach(_b IN LISTS _bs)
            if(TARGET turbo_ocr_backend_${_b})
                target_link_libraries(${target} PRIVATE
                    "$<LINK_LIBRARY:WHOLE_ARCHIVE,turbo_ocr_backend_${_b}>")
            endif()
        endforeach()
    endfunction()

    # THE TEST BINARY NEEDS THE REGISTRARS TOO. Without this,
    # backend::available_backends() returns EMPTY inside turbo_ocr_tests — the
    # registration TUs are dropped exactly as the comment above describes — so
    # every seam-level test silently has nothing to run against. That is why the
    # conformance suites assert the list is non-empty rather than skipping.
    if(TARGET turbo_ocr_tests)
        turbo_link_backends(turbo_ocr_tests ${TURBO_BACKENDS} cpu)
        # The VLM transport is the remote OpenAI-compatible endpoint path. It
        # needs libcurl, and crop_pool wakes its worker with a POSIX self-pipe,
        # so it only attaches where both exist. Dropping it costs the VLM
        # table/formula backends — a runtime-opt-in feature — and nothing in the
        # local OCR pipeline the rest of the suite exercises.
        # libcurl only. The worker wake is curl_multi_wakeup(), so there is no
        # POSIX self-pipe and nothing platform-specific left in this path.
        if(CURL_FOUND OR CURL_LIB)
            target_sources(turbo_ocr_tests PRIVATE
                src/analysis/vlm/vlm_client.cpp
                src/analysis/vlm/crop_pool.cpp
                src/analysis/vlm/crop_pool_transport.cpp)
        else()
            message(STATUS "libcurl not found — VLM transport excluded from tests")
        endif()
        # Seam-kernel contract tests. Attached HERE, not in the main test list:
        # they include src/backends headers, which only exist as usage
        # requirements once the backend targets are linked above. The argmax
        # tie-break test's host half runs in every seam configure; its device
        # halves compile per-vendor. (test_db_postprocess_parity.cpp was
        # written for the nvidia bring-up but had never been attached to any
        # target — found while wiring the amd tests.)
        if(TARGET turbo_ocr_backend_cpu)
            target_sources(turbo_ocr_tests PRIVATE
                tests/cpp/backends/test_argmax_tiebreak.cpp)
        endif()
        if(nvidia IN_LIST TURBO_BACKENDS)
            target_sources(turbo_ocr_tests PRIVATE
                tests/cpp/backends/test_db_postprocess_parity.cpp)
        endif()
        if(amd IN_LIST TURBO_BACKENDS)
            target_compile_definitions(turbo_ocr_tests PRIVATE
                TURBO_TESTS_HAVE_AMD=1 __HIP_PLATFORM_AMD__=1)
        endif()
        # Prefer the imported target: CURL::libcurl carries its INCLUDE dirs,
        # which the bare find_library below does not. That matters wherever
        # libcurl is not on the default include path — a vcpkg or Homebrew
        # prefix — because crop_pool.h includes <curl/curl.h> in its PUBLIC
        # header, so every consumer needs the include dir, not just the lib.
        if(TARGET CURL::libcurl)
            target_link_libraries(turbo_ocr_tests PRIVATE CURL::libcurl)
        else()
            find_library(TURBO_TESTS_CURL_LIB curl)
            if(TURBO_TESTS_CURL_LIB)
                target_link_libraries(turbo_ocr_tests PRIVATE ${TURBO_TESTS_CURL_LIB})
            endif()
        endif()
    endif()

    # -----------------------------------------------------------------------
    # THE CROSS-BACKEND TEST HARNESS — ONE set of binaries for EVERY backend.
    # A driver that only calls backend::make_backend(name) + UnifiedOcrPipeline
    # touches no vendor type, so these are plain .cpp targets that link
    # WHOLE_ARCHIVE against every selected backend and select at runtime.
    #   turbo_bench       — accuracy + throughput, any backend, one protocol
    #   turbo_conformance — same images through EVERY linked backend, diffed
    #   turbo_golden      — per-STAGE golden diff (backend, stage) vs the CPU ref
    # -----------------------------------------------------------------------
    function(turbo_add_backend_test_exe name src)
        add_executable(${name} ${src})
        if(TARGET CURL::libcurl)
            target_sources(${name} PRIVATE
                tests/cpp/backends/vlm_factory_link_support.cpp)
        endif()
        target_link_libraries(${name} PRIVATE turbo_ocr_pipeline ${OpenCV_LIBS})
        turbo_link_backends(${name} ${TURBO_BACKENDS})
        turbo_nvidia_runtime_paths(${name})
        target_compile_options(${name} PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
        if(TARGET turbo_apple_metallib)
            add_dependencies(${name} turbo_apple_metallib)
        endif()
    endfunction()

    turbo_add_backend_test_exe(turbo_bench       tests/cpp/backends/turbo_bench.cpp)
    turbo_add_backend_test_exe(turbo_conformance tests/cpp/backends/turbo_conformance.cpp)
    turbo_add_backend_test_exe(turbo_golden      tests/cpp/backends/turbo_golden.cpp)

    # The multi-backend probe: ONE binary holding every selected backend,
    # printing available_backends() and constructing each — the regression test
    # for "registrars survived the link" (i.e. WHOLE_ARCHIVE works).
    add_executable(turbo_backend_probe
        tests/cpp/backends/backend_probe.cpp)
    if(TARGET CURL::libcurl)
        target_sources(turbo_backend_probe PRIVATE
            tests/cpp/backends/vlm_factory_link_support.cpp)
    endif()
    target_link_libraries(turbo_backend_probe PRIVATE turbo_ocr_pipeline ${OpenCV_LIBS})
    turbo_link_backends(turbo_backend_probe ${TURBO_BACKENDS})
    turbo_nvidia_runtime_paths(turbo_backend_probe)
    target_compile_options(turbo_backend_probe PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
    add_test(NAME backend_probe COMMAND turbo_backend_probe --list)

    # -----------------------------------------------------------------------
    # ctest registration. Tests that need only the models run BY DEFAULT;
    # tests that need the FUNSD image cache are registered when
    # -DTURBO_FUNSD_CACHE=<dir> points at one. Anything inapplicable exits 77
    # (SKIPPED), never a failure.
    # -----------------------------------------------------------------------
    set(TURBO_FUNSD_CACHE "" CACHE PATH "FUNSD test-set cache dir; enables the ctest accuracy gates")

    set(_turbo_test_env "")
    if(TARGET turbo_apple_metallib)
        list(APPEND _turbo_test_env "TURBO_APPLE_METALLIB=${TURBO_APPLE_METALLIB}")
    endif()
    # Pin the Apple recognition ladder: MpsRecognizer auto-discovers every
    # rec_b* export dir on disk, so an unpinned run silently uses a 42-bucket
    # ladder and halves throughput.
    list(APPEND _turbo_test_env "TURBO_APPLE_REC_BUCKETS=320,480,800,1200,1600,2000,2500,3200,4000")
    # ORT CPU/MLAS only: CoreML EP on the CPU backend is non-deterministic
    # across macOS versions — fatal for an exact-match accuracy gate.
    list(APPEND _turbo_test_env "DISABLE_COREML=1")

    if(TURBO_FUNSD_CACHE)
        # Cross-backend conformance: THE keystone test. Thresholds are tripwires
        # just under the measured cpu-vs-apple baseline (box-match 0.9657, mean
        # IoU 0.9357, per-line text agreement 0.7308 over FUNSD pages 0-9).
        add_test(NAME backend_conformance
                 COMMAND turbo_conformance --images "${TURBO_FUNSD_CACHE}" --count 10
                         --assert-box-match 0.90 --assert-mean-iou 0.88
                         --assert-text-agreement 0.65
                         --out "${CMAKE_BINARY_DIR}/conformance.json")
        set_tests_properties(backend_conformance PROPERTIES
            ENVIRONMENT "${_turbo_test_env}" SKIP_RETURN_CODE 77 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

        # Per-stage golden diffs vs the cpu reference (skip cleanly otherwise).
        foreach(_gb IN LISTS TURBO_BACKENDS)
            if(_gb STREQUAL "cpu")
                continue()
            endif()
            foreach(_gs det cls rec layout)
                add_test(NAME golden_${_gb}_${_gs}
                         COMMAND turbo_golden --backend ${_gb} --ref cpu --stage ${_gs}
                                 --images "${TURBO_FUNSD_CACHE}" --count 10
                                 --out "${CMAKE_BINARY_DIR}/golden_${_gb}_${_gs}.json")
                set_tests_properties(golden_${_gb}_${_gs} PROPERTIES
                    ENVIRONMENT "${_turbo_test_env}" SKIP_RETURN_CODE 77
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
            endforeach()
        endforeach()

        # FUNSD accuracy gates that actually SCORE and ASSERT: the run produces
        # the transcript + metrics JSON, the gate feeds both to
        # tools/bench/score_funsd.py with --assert-f1 (nonzero below the floor).
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            # name|backend|tier|min-F1  (floors = the MEASURED numbers minus a hair)
            set(_turbo_gates
                "cpu_tiny|cpu|tiny|85.7"
                "cpu_medium|cpu|medium|92.5"
                "apple_tiny|apple|tiny|85.2"
                # MEASURED on the RTX 5090 bring-up (2026-08-02): tiny 85.37%.
                # Floor stays a hair under the pre-rebuild 85.5% reference.
                "nvidia_tiny|nvidia|tiny|85.0"
                # PROVISIONAL, never run on ROCm hardware: same ONNX models
                # through MIGraphX, so the expectation is the shared-pipeline
                # tiny number (~85.5%). If the first on-hardware run lands
                # elsewhere, change THIS number (after establishing why) — do
                # not delete the gate.
                "amd_tiny|amd|tiny|85.0")
            foreach(_g IN LISTS _turbo_gates)
                string(REPLACE "|" ";" _parts "${_g}")
                list(GET _parts 0 _gname)
                list(GET _parts 1 _gbackend)
                list(GET _parts 2 _gtier)
                list(GET _parts 3 _gf1)
                if(NOT _gbackend IN_LIST TURBO_BACKENDS)
                    continue()
                endif()
                set(_words "${CMAKE_BINARY_DIR}/funsd_${_gname}.words.json")
                set(_metrics "${CMAKE_BINARY_DIR}/funsd_${_gname}.metrics.json")
                add_test(NAME funsd_${_gname}_run
                         COMMAND turbo_bench --backend ${_gbackend} --tier ${_gtier}
                                 --images "${TURBO_FUNSD_CACHE}" --count 50
                                 --words "${_words}" --out "${_metrics}")
                set_tests_properties(funsd_${_gname}_run PROPERTIES
                    ENVIRONMENT "${_turbo_test_env}"
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                    FIXTURES_SETUP funsd_${_gname})
                add_test(NAME funsd_${_gname}_gate
                         COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/bench/score_funsd.py"
                                 "${_words}" --metrics "${_metrics}" --assert-f1 ${_gf1})
                set_tests_properties(funsd_${_gname}_gate PROPERTIES
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                    FIXTURES_REQUIRED funsd_${_gname})
            endforeach()
        else()
            message(STATUS "python3 not found — FUNSD accuracy gates not registered")
        endif()
    else()
        message(STATUS "TURBO_FUNSD_CACHE unset — conformance/golden/FUNSD ctest gates not registered")
    endif()

    # -----------------------------------------------------------------------
    # The unified multi-backend server (CPU configure; the GPU configure's
    # native turboocr-server above remains canonical until the nvidia backend's
    # on-hardware bring-up).
    # -----------------------------------------------------------------------
    # ONE server for every backend, both configures.
    if(BUILD_SERVER AND (TARGET turbo_ocr_cpu OR TARGET turbo_ocr_gpu))
        include(src/service/server/unified/unified_server.cmake)
    endif()
endif()

# ---------------------------------------------------------------------------
# Transport-parity suite (attached here, not in the turbo_ocr_tests source list
# above, because PROTO_GEN_DIR is defined by turbo_setup_grpc_codegen() inside
# the per-arm blocks — referencing it earlier silently yields "/ocr.pb.cc").
#
# It asserts that ocr.proto declares a field for every capability the transport
# acts on, and that the gRPC option gate IS the shared one. Both properties were
# claimed in comments (proto_capability_bridge.h: "asserted by the capability
# registry test") while nothing checked them — the test binary had never linked
# the generated proto, so the check was not merely missing but impossible.
# ---------------------------------------------------------------------------
if(TARGET turbo_ocr_tests AND DEFINED PROTO_GEN_DIR)
    target_sources(turbo_ocr_tests PRIVATE
        tests/cpp/capability/test_proto_capability_parity.cpp
        ${PROTO_GEN_CC}
    )
    target_include_directories(turbo_ocr_tests PRIVATE ${PROTO_GEN_DIR})
    target_link_libraries(turbo_ocr_tests PRIVATE protobuf::libprotobuf)
endif()

# --- Python bindings (nanobind) -------------------------------------------
# Thin `_turboocr` extension over UnifiedOcrPipeline + the backend seam. It
# wraps backend "cpu" by default (Tier B, ONNX/EP switch via env); the compiled
# backend set follows TURBO_BACKENDS. Enable with -DBUILD_PYTHON=ON; nanobind +
# Python are located by scikit-build-core (or set nanobind_DIR /
# Python_EXECUTABLE manually).
if(BUILD_PYTHON)
    if(NOT TARGET turbo_ocr_backend_cpu)
        message(FATAL_ERROR
            "BUILD_PYTHON wraps UnifiedOcrPipeline over the cpu backend — "
            "configure with -DUSE_CPU_ONLY=ON and keep 'cpu' in TURBO_BACKENDS.")
    endif()
    # Development.SABIModule is what makes the STABLE_ABI flag below MEAN
    # anything: nanobind needs the Python::SABIModule target to build against
    # the limited API, and SILENTLY IGNORES STABLE_ABI when it is absent.
    # Without it the extension came out version-specific
    # (_turboocr.cpython-312-*.so instead of _turboocr.abi3.so) while
    # scikit-build-core still tagged the wheel cp312-abi3 from
    # python/pyproject.toml's wheel.py-api — a wheel that claims to run on every
    # Python 3.12+ but holds a 3.12-only binary. That mismatch is what
    # abi3audit was failing on; it was reporting a real defect, not being
    # pedantic. OPTIONAL, because the target only exists on CPython >= 3.12 and
    # a 3.9-3.11 build must still configure (nanobind then ignores STABLE_ABI
    # for real, and scikit-build-core falls back to a cp3XY tag to match).
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module
                 OPTIONAL_COMPONENTS Development.SABIModule)
    # Isolated builds (pip / scikit-build-core) get nanobind from the build
    # venv; ask the interpreter for its CMake dir before find_package. A
    # manually set nanobind_DIR still wins (the cache variable pre-empts this).
    if(NOT nanobind_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_VARIABLE _nb_dir OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_nb_dir)
            set(nanobind_DIR "${_nb_dir}")
        endif()
    endif()
    find_package(nanobind CONFIG REQUIRED)
    nanobind_add_module(_turboocr STABLE_ABI NB_STATIC
        src/service/python/bindings.cpp
    )
    # When curl is found, the pipeline archive contains the real vlm_factory.o
    # but the stubs compiled here satisfy the symbols first, so the linker
    # never pulls the archive member (no curl dependency in the extension).
    # When curl is absent, the pipeline already includes the stubs itself.
    if(TARGET CURL::libcurl)
        target_sources(_turboocr PRIVATE
            tests/cpp/backends/vlm_factory_link_support.cpp)
    endif()
    target_link_libraries(_turboocr PRIVATE turbo_ocr_pipeline turbo_ocr_common)
    turbo_link_backends(_turboocr ${TURBO_BACKENDS})
    target_compile_options(_turboocr PRIVATE ${TURBO_OCR_CXX_FLAGS})
    # The INSTALLED extension must still be able to resolve its @rpath dylibs
    # (libpdfium and libonnxruntime, whose install names are @rpath-relative):
    # CMake strips the build-tree RPATH at install time, which left
    # delocate-wheel unable to chase the dependencies at all (the DESIGN.md
    # section-5 RPATH landmine, hit for real). Both dirs are named explicitly —
    # USE_LINK_PATH alone missed them (PDFium enters the link via
    # INTERFACE_LINK_DIRECTORIES; ORT via the find_library full path).
    get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIB}" DIRECTORY)
    set_target_properties(_turboocr PROPERTIES
        INSTALL_RPATH "${TURBO_PDFIUM_DIR}/lib;${_ort_lib_dir}"
        INSTALL_RPATH_USE_LINK_PATH TRUE)
    # Land the built extension next to the Python package for
    # `import turboocr_engine` (the native distribution; the pure-Python
    # umbrella `turboocr` re-exports it when installed).
    # COMPONENT python scopes the wheel: scikit-build-core installs ONLY this
    # component (python/pyproject.toml install.components), so the server/model
    # install rules above never leak 1.5 GB of models into a wheel.
    install(TARGETS _turboocr LIBRARY DESTINATION turboocr_engine COMPONENT python)
    # The wheel must carry the Metal shader library beside the extension:
    # metallib_path() (metal_common.mm) resolves it via dladdr relative to the
    # loaded image, so site-packages/turboocr_engine/turbo_apple.metallib works.
    if(TARGET turbo_apple_metallib)
        add_dependencies(_turboocr turbo_apple_metallib)
        install(FILES "${TURBO_APPLE_METALLIB}" DESTINATION turboocr_engine
                COMPONENT python)
    endif()
endif()
