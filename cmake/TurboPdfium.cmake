# PDFium for the target architecture.
#
# The SDK vendored in third_party/pdfium is the x86-64 build. A build for any
# other architecture must not link it, so this module decides which SDK to use:
#   1. TURBO_PDFIUM_DIR, when set by the user — taken as given.
#   2. The vendored copy, when its libpdfium.so is built for the target.
#   3. Otherwise the pinned bblanchon/pdfium-binaries release for the target,
#      fetched into the build tree and verified against the recorded
#      per-architecture SHA-256 (same release and hashes as
#      scripts/install_pdfium.sh, which does the equivalent for Docker).
# Requires TURBO_ARCH ("x64" | "aarch64") from the caller.

include_guard(GLOBAL)

set(TURBO_PDFIUM_DIR "" CACHE PATH
    "Root of a PDFium SDK ({include/, lib/libpdfium.so}). Empty: the vendored copy when it matches the target architecture, otherwise the pinned bblanchon release is fetched.")
# Plain variable so a bump here takes effect on an existing build directory;
# -DTURBO_PDFIUM_RELEASE=<tag> (with -DTURBO_PDFIUM_SHA256) still overrides it.
if(NOT DEFINED TURBO_PDFIUM_RELEASE)
  set(TURBO_PDFIUM_RELEASE "chromium/7857")  # bblanchon/pdfium-binaries release; keep in sync with scripts/install_pdfium.sh
endif()
set(TURBO_PDFIUM_SHA256 "" CACHE STRING
    "SHA-256 of pdfium-linux-<arch>.tgz for TURBO_PDFIUM_RELEASE (empty: the recorded hash of the pinned release)")

# ELF e_machine of a binary, as CMake sees it: "x64", "aarch64", "other(<hex>)",
# or "" when the file does not exist. Bytes 18-19 of the header, little-endian.
function(turbo_elf_machine out_var path)
  set(${out_var} "" PARENT_SCOPE)
  if(NOT EXISTS "${path}")
    return()
  endif()
  file(READ "${path}" _hex OFFSET 18 LIMIT 2 HEX)
  if(_hex STREQUAL "3e00")
    set(${out_var} "x64" PARENT_SCOPE)
  elseif(_hex STREQUAL "b700")
    set(${out_var} "aarch64" PARENT_SCOPE)
  else()
    set(${out_var} "other(${_hex})" PARENT_SCOPE)
  endif()
endfunction()

set(_turbo_pdfium_vendored "${CMAKE_SOURCE_DIR}/third_party/pdfium")

if(TURBO_PDFIUM_DIR)
  if(NOT EXISTS "${TURBO_PDFIUM_DIR}/lib/libpdfium.so" OR NOT EXISTS "${TURBO_PDFIUM_DIR}/include/fpdfview.h")
    message(FATAL_ERROR "TURBO_PDFIUM_DIR=${TURBO_PDFIUM_DIR} does not hold a PDFium SDK "
                        "(expected lib/libpdfium.so and include/fpdfview.h)")
  endif()
  turbo_elf_machine(_turbo_pdfium_machine "${TURBO_PDFIUM_DIR}/lib/libpdfium.so")
  if(NOT _turbo_pdfium_machine STREQUAL TURBO_ARCH)
    message(WARNING "TURBO_PDFIUM_DIR=${TURBO_PDFIUM_DIR}: libpdfium.so is ${_turbo_pdfium_machine}, "
                    "the target is ${TURBO_ARCH}; linking will fail")
  endif()
  set(_turbo_pdfium_reason "TURBO_PDFIUM_DIR")
else()
  turbo_elf_machine(_turbo_pdfium_machine "${_turbo_pdfium_vendored}/lib/libpdfium.so")
  if(_turbo_pdfium_machine STREQUAL TURBO_ARCH)
    set(TURBO_PDFIUM_DIR "${_turbo_pdfium_vendored}")
    set(_turbo_pdfium_reason "vendored ${TURBO_ARCH} SDK")
  else()
    if(TURBO_ARCH STREQUAL "x64")
      set(_turbo_pdfium_url_arch "x64")
      set(_turbo_pdfium_hash "2ad1fd4237cd491201ac74a72388199b9dcf546c5cb02d8fea700725a1b80541")
    else()
      set(_turbo_pdfium_url_arch "arm64")
      set(_turbo_pdfium_hash "0e24373e73c50759136196c0078db8656860c8d03a10b2cb4a2e7b72d8068e35")
    endif()
    if(TURBO_PDFIUM_SHA256)
      set(_turbo_pdfium_hash "${TURBO_PDFIUM_SHA256}")
    elseif(NOT TURBO_PDFIUM_RELEASE STREQUAL "chromium/7857")
      message(FATAL_ERROR "TURBO_PDFIUM_RELEASE=${TURBO_PDFIUM_RELEASE} has no recorded hash; pass "
                          "-DTURBO_PDFIUM_SHA256=<sha256 of pdfium-linux-${_turbo_pdfium_url_arch}.tgz>")
    endif()
    string(REPLACE "/" "%2F" _turbo_pdfium_tag_url "${TURBO_PDFIUM_RELEASE}")
    if(_turbo_pdfium_machine STREQUAL "")
      set(_turbo_pdfium_why "no vendored SDK")
    else()
      set(_turbo_pdfium_why "vendored SDK is ${_turbo_pdfium_machine}, target is ${TURBO_ARCH}")
    endif()
    message(STATUS "PDFium: ${_turbo_pdfium_why}; fetching ${TURBO_PDFIUM_RELEASE} pdfium-linux-${_turbo_pdfium_url_arch} "
                   "(set TURBO_PDFIUM_DIR to build offline)")
    include(FetchContent)
    FetchContent_Declare(pdfium_prebuilt
      URL "https://github.com/bblanchon/pdfium-binaries/releases/download/${_turbo_pdfium_tag_url}/pdfium-linux-${_turbo_pdfium_url_arch}.tgz"
      URL_HASH SHA256=${_turbo_pdfium_hash}
      SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/pdfium-linux-${_turbo_pdfium_url_arch}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(pdfium_prebuilt)
    set(TURBO_PDFIUM_DIR "${pdfium_prebuilt_SOURCE_DIR}")
    set(_turbo_pdfium_reason "fetched ${TURBO_PDFIUM_RELEASE}, ${_turbo_pdfium_url_arch}")
  endif()
endif()
message(STATUS "PDFium: ${TURBO_PDFIUM_DIR} (${_turbo_pdfium_reason})")
