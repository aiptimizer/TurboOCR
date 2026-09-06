# fastpdf2png — the PDF page renderer the server forks at startup — built as a
# subproject from its pinned commit with this build's toolchain, so the binary
# matches the target architecture by construction. It lands next to the server
# executable as <build>/fastpdf2png together with libpdfium.so (loaded via
# $ORIGIN, copied there by fastpdf2png's own post-build step).
#
# FetchContent fetches the pinned commit at configure time and re-fetches when
# the pin changes; the subproject's targets take part in the normal incremental
# build, so an unchanged pin costs nothing and a bump rebuilds what changed.
# The Docker images build the renderer with scripts/install_fastpdf2png.sh into
# bin/ instead and pass -DTURBO_BUILD_FASTPDF2PNG=OFF. Offline builds point
# TURBO_FASTPDF2PNG_SOURCE_DIR at a checkout of the pinned commit.
# Requires TURBO_PDFIUM_DIR (cmake/TurboPdfium.cmake) and the NATIVE_ARCH option;
# fastpdf2png >= 2.0.9 builds as a subproject and takes the SDK from PDFium_DIR.

include_guard(GLOBAL)

option(TURBO_BUILD_FASTPDF2PNG
       "Build the fastpdf2png PDF renderer from its pinned source as part of this build" ON)
# The pin is a plain variable so a bump here takes effect on an existing build
# directory; -DTURBO_FASTPDF2PNG_GIT_TAG=<commit> on the command line still overrides it.
if(NOT DEFINED TURBO_FASTPDF2PNG_GIT_TAG)
  set(TURBO_FASTPDF2PNG_GIT_TAG "8358bdc14378c1b33ada057f24aa43f81075dbf7")  # v2.0.10; keep in sync with FASTPDF2PNG_COMMIT in scripts/install_fastpdf2png.sh
endif()
set(TURBO_FASTPDF2PNG_SOURCE_DIR "" CACHE PATH
    "Local fastpdf2png checkout to build instead of fetching TURBO_FASTPDF2PNG_GIT_TAG")

# turbo_add_fastpdf2png(<server targets...>): add the renderer subproject and
# make the given targets depend on it, so building the server yields both.
function(turbo_add_fastpdf2png)
  if(NOT TURBO_BUILD_FASTPDF2PNG)
    message(STATUS "fastpdf2png: not built here (TURBO_BUILD_FASTPDF2PNG=OFF); the server looks in "
                   "its own directory, /app/bin, /usr/local/bin, ./build, ./bin, or FASTPDF2PNG_PATH")
    return()
  endif()
  include(FetchContent)

  # EXCLUDE_FROM_ALL keeps the subproject's install rules and default targets
  # out of ours (CMake >= 3.28); older CMake simply builds it with `all`.
  set(_fp2p_exclude "")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    set(_fp2p_exclude EXCLUDE_FROM_ALL)
  endif()
  if(TURBO_FASTPDF2PNG_SOURCE_DIR)
    FetchContent_Declare(fastpdf2png SOURCE_DIR "${TURBO_FASTPDF2PNG_SOURCE_DIR}" ${_fp2p_exclude})
    set(_fp2p_what "${TURBO_FASTPDF2PNG_SOURCE_DIR}")
  else()
    FetchContent_Declare(fastpdf2png
        GIT_REPOSITORY https://github.com/aiptimizer/fastpdf2png.git
        GIT_TAG "${TURBO_FASTPDF2PNG_GIT_TAG}"
        GIT_SHALLOW FALSE
        SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/fastpdf2png-src"
        ${_fp2p_exclude})
    set(_fp2p_what "aiptimizer/fastpdf2png @ ${TURBO_FASTPDF2PNG_GIT_TAG}")
  endif()

  # The subproject's options, seen as normal variables in its scope: static
  # core library (one binary to ship), no tests or benchmarks, the SDK this
  # build already selected, and native tuning only when the server has it.
  set(FP2P_BUILD_CLI ON)
  set(FP2P_BUILD_SHARED_LIB OFF)
  set(FP2P_BUILD_TESTS OFF)
  set(FP2P_BUILD_BENCHMARKS OFF)
  if(NATIVE_ARCH)
    set(FP2P_NATIVE_ARCH ON)
  else()
    set(FP2P_NATIVE_ARCH OFF)
  endif()
  set(PDFium_DIR "${TURBO_PDFIUM_DIR}")
  FetchContent_MakeAvailable(fastpdf2png)

  if(NOT TARGET fastpdf2png_bin)
    message(FATAL_ERROR "fastpdf2png subproject at ${_fp2p_what} did not define the fastpdf2png_bin target")
  endif()
  set_target_properties(fastpdf2png_bin PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  foreach(_t IN LISTS ARGN)
    if(TARGET ${_t})
      add_dependencies(${_t} fastpdf2png_bin)
    endif()
  endforeach()
  message(STATUS "fastpdf2png: built from ${_fp2p_what} -> ${CMAKE_BINARY_DIR}/fastpdf2png")
endfunction()
