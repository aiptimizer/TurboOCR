# Turbo OCR Testing Guide

Comprehensive testing strategy for the `turbo_ocr` GPU OCR server.
Covers memory safety, thread safety, static analysis, fuzzing, load testing,
and benchmarking.

---

## Table of Contents

1. [Current Test Suite Overview](#1-current-test-suite-overview)
2. [Memory Leak Detection](#2-memory-leak-detection)
3. [Thread Safety](#3-thread-safety)
4. [Undefined Behavior Detection](#4-undefined-behavior-detection)
5. [Static Analysis](#5-static-analysis)
6. [Fuzz Testing](#6-fuzz-testing)
7. [Integration and Load Testing](#7-integration-and-load-testing)
8. [Benchmarking](#8-benchmarking)
9. [CI/CD Integration](#9-cicd-integration)
10. [Recommended Workflow](#10-recommended-workflow)

---

## 1. Current Test Suite Overview

The project already has a Python-based test suite organized into four suites:

```
tests/
  conftest.py                    # Shared fixtures (image generation, server URLs)
  run_all.py                     # Master runner (pytest wrapper)
  requirements.txt               # pytest, requests, grpcio, Pillow, etc.
  unit/                          # Tests base64, box sorting, JSON response (via HTTP)
  integration/                   # Endpoint tests: /ocr, /ocr/raw, /ocr/batch, gRPC, PDF, errors
  regression/                    # Accuracy regression, ordering consistency
  benchmark/                     # Throughput, latency, concurrency, parallel images/PDF
```

Run everything:
```bash
pip install -r tests/requirements.txt
python tests/run_all.py                     # all suites
python tests/run_all.py --suite unit        # just unit
python tests/run_all.py --suite benchmark   # just benchmarks
```

All tests require a running server instance (they are black-box HTTP/gRPC tests).
There are currently no C++ unit tests compiled as separate binaries.

---

## 2. Memory Leak Detection

### 2.1 AddressSanitizer (ASan) + LeakSanitizer (LSan)

Already supported in CMakeLists.txt via `ENABLE_ASAN`. ASan provides ~2x runtime
overhead (much faster than Valgrind's 10-50x).

**Build with ASan:**
```bash
cmake -B build_asan \
  -DENABLE_ASAN=ON \
  -DTENSORRT_DIR=/usr/local/tensorrt \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_asan -j$(nproc)
```

**Run the server under ASan:**
```bash
# LSan is included with ASan by default on Linux
ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:log_path=asan.log" \
  ./build_asan/paddle_highspeed_cpp
```

Then run the test suite against it:
```bash
python tests/run_all.py --suite integration --suite unit
```

Stop the server (Ctrl+C) and check `asan.log.*` for leak reports.

**Key ASan options for this project:**
```bash
ASAN_OPTIONS="detect_leaks=1:halt_on_error=0:suppressions=asan_suppressions.txt:log_path=asan"
```

**Known limitations with CUDA:**
- ASan instruments host code only; it cannot detect GPU memory leaks.
- Some TensorRT/CUDA runtime allocations may produce false positives.
  Create a suppressions file if needed:
  ```
  # asan_suppressions.txt
  leak:libnvinfer
  leak:libcudart
  leak:libcuda
  leak:libnvjpeg
  ```

### 2.2 Valgrind Memcheck

Valgrind provides the most thorough memory analysis but with significant overhead
(10-50x slowdown). Best for targeted testing, not load testing.

**Install:**
```bash
sudo apt install valgrind    # Debian/Ubuntu
sudo pacman -S valgrind      # Arch
```

**Run:**
```bash
valgrind --tool=memcheck \
  --leak-check=full \
  --show-leak-kinds=all \
  --track-origins=yes \
  --log-file=valgrind.log \
  ./build/paddle_highspeed_cpp
```

Then send a few requests manually and stop the server:
```bash
curl -X POST http://localhost:8000/ocr/raw \
  --data-binary @tests/test_data/png/simple_text.png \
  -H "Content-Type: image/png"
```

**Limitations:**
- Extremely slow with GPU applications (Valgrind does not understand CUDA).
- Best used with the CPU-only build:
  ```bash
  cmake -B build_valgrind -DUSE_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Debug
  cmake --build build_valgrind -j$(nproc)
  valgrind --leak-check=full ./build_valgrind/paddle_cpu_server
  ```

### 2.3 NVIDIA Compute Sanitizer (GPU Memory)

Replaces the deprecated `cuda-memcheck` (removed in CUDA 12.0+). Included in
the CUDA Toolkit.

**Memcheck (out-of-bounds, misaligned access, leaks):**
```bash
compute-sanitizer --tool memcheck \
  --leak-check full \
  --log-file compute_memcheck.log \
  ./build/paddle_highspeed_cpp
```

**Racecheck (shared memory data races):**
```bash
compute-sanitizer --tool racecheck \
  --log-file compute_racecheck.log \
  ./build/paddle_highspeed_cpp
```

**Initcheck (uninitialized GPU memory reads):**
```bash
compute-sanitizer --tool initcheck \
  --log-file compute_initcheck.log \
  ./build/paddle_highspeed_cpp
```

**Best practices for this project:**
- Compile with `-lineinfo` (already in CMAKE_CUDA_FLAGS) for source-line mapping.
- Starting with CUDA 13.1, compile-time instrumentation improves detection:
  ```bash
  # Add to CMAKE_CUDA_FLAGS for instrumented builds
  -fsanitize=address
  ```
- Run with a small number of requests (compute-sanitizer adds significant overhead).
- The kernels in `src/kernels/kernels.cu` (batch_roi_warp, argmax, fused_resize_normalize,
  threshold_to_u8, gpu_ccl) are the primary targets for GPU memory checking.

---

## 3. Thread Safety

### 3.1 ThreadSanitizer (TSan)

Already supported in CMakeLists.txt via `ENABLE_TSAN`. Detects data races,
deadlocks, and lock-order inversions.

**Build with TSan:**
```bash
cmake -B build_tsan \
  -DENABLE_TSAN=ON \
  -DTENSORRT_DIR=/usr/local/tensorrt \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_tsan -j$(nproc)
```

**Run:**
```bash
TSAN_OPTIONS="history_size=7:second_deadlock_stack=1:log_path=tsan.log" \
  ./build_tsan/paddle_highspeed_cpp
```

Then run concurrent tests:
```bash
python tests/run_all.py --suite integration
# Also specifically run the concurrency benchmarks to stress thread interactions:
pytest tests/benchmark/bench_concurrent.py -v -s --server-url http://localhost:8000
```

**Critical areas to test for races:**
- `PipelinePool<T>` acquire/release (mutex + condition variable)
- Drogon HTTP handler work pool (multiple requests hitting the pool concurrently)
- `NvJpegDecoder` thread-local instances
- `PdfRenderer` daemon pool management
- gRPC server concurrent request handling

**Important:** TSan and ASan cannot be used simultaneously. Build separate binaries.

### 3.2 Helgrind (Valgrind Thread Checker)

Alternative to TSan, runs under Valgrind. Much slower but can catch different
classes of issues.

```bash
valgrind --tool=helgrind \
  --log-file=helgrind.log \
  ./build_valgrind/paddle_cpu_server
```

Best used with the CPU-only build due to Valgrind's CUDA incompatibility.

---

## 4. Undefined Behavior Detection

### 4.1 UndefinedBehaviorSanitizer (UBSan)

Not yet in CMakeLists.txt. Detects signed integer overflow, null pointer
dereference, type punning violations, shift errors, and more.

**Recommended CMake addition** (do not apply yet -- documentation only):
```cmake
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
if(ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()
```

**Manual build (without modifying CMakeLists.txt):**
```bash
cmake -B build_ubsan \
  -DTENSORRT_DIR=/usr/local/tensorrt \
  -DCMAKE_CXX_FLAGS="-fsanitize=undefined -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build_ubsan -j$(nproc)
```

**Run:**
```bash
UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=0:log_path=ubsan.log" \
  ./build_ubsan/paddle_highspeed_cpp
```

**Areas of concern in this codebase:**
- `base64_decode()` -- pointer arithmetic with user-controlled lengths
- `snprintf` in `results_to_json()` -- float formatting
- `det_postprocess` -- polygon math, division operations
- `ctc_decode` -- index arithmetic on model output buffers
- `crop_utils` -- perspective transform matrix computations

**UBSan + ASan combined build:**
```bash
cmake -B build_sanitized \
  -DENABLE_ASAN=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=undefined" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined" \
  -DTENSORRT_DIR=/usr/local/tensorrt
```

UBSan is compatible with ASan (but not TSan).

---

## 5. Static Analysis

### 5.1 clang-tidy

The project already has `.clang-format` (Google style) and generates
`compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS ON`).

**Recommended `.clang-tidy` configuration for this project:**
```yaml
---
Checks: >
  -*,
  bugprone-*,
  cert-*,
  clang-analyzer-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers,
  -readability-identifier-length,
  -cppcoreguidelines-pro-type-reinterpret-cast,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic
WarningsAsErrors: ''
HeaderFilterRegex: 'include/turbo_ocr/.*'
FormatStyle: file
CheckOptions:
  - key: readability-function-cognitive-complexity.Threshold
    value: 50
  - key: performance-unnecessary-value-param.AllowedTypes
    value: 'cv::Mat'
```

Exclusions explained:
- `reinterpret-cast` and `pointer-arithmetic`: unavoidable in the base64 decoder,
  CUDA interop, and image buffer manipulation.
- `magic-numbers`: excessive noise in numeric computation code (kernels, geometry).
- `identifier-length`: short loop vars (`i`, `j`, `si`, `di`) are fine.

**Run clang-tidy:**
```bash
# After building (needs compile_commands.json)
cmake -B build -DTENSORRT_DIR=/usr/local/tensorrt
cmake --build build -j$(nproc)

# Single file
clang-tidy -p build src/decode/fast_png_decoder.cpp

# All source files
find src -name '*.cpp' | xargs -P$(nproc) -I{} clang-tidy -p build {}

# With fixes applied (careful -- review first)
clang-tidy -p build --fix src/recognition/ctc_decode.cpp
```

**Priority checks for this codebase:**
- `bugprone-narrowing-conversions` -- float/int conversions in image processing
- `bugprone-use-after-move` -- move-only types like GpuPipelineEntry
- `performance-move-const-arg` -- ensure moves are effective
- `clang-analyzer-core.NullDereference` -- null checks on decoded images
- `cert-err58-cpp` -- static initialization order issues

### 5.2 cppcheck

Complementary to clang-tidy; catches different bug patterns.

**Install:**
```bash
sudo apt install cppcheck       # Debian/Ubuntu
sudo pacman -S cppcheck         # Arch
```

**Run:**
```bash
cppcheck --enable=all \
  --std=c++20 \
  --suppress=missingIncludeSystem \
  --suppress=unusedFunction \
  --project=build/compile_commands.json \
  --output-file=cppcheck_report.txt \
  2>&1
```

**Targeted checks:**
```bash
# Check just the common/ code (most portable, no CUDA deps)
cppcheck --enable=all --std=c++20 \
  -I include \
  src/detection/det_postprocess.cpp \
  src/recognition/ctc_decode.cpp \
  src/recognition/crop_utils.cpp \
  src/decode/fast_png_decoder.cpp
```

### 5.3 Include What You Use (IWYU)

Ensures headers include exactly what they need -- no more, no less. Reduces
compile times and avoids transitive dependency issues.

**Install:**
```bash
sudo apt install iwyu                # Debian/Ubuntu
sudo pacman -S include-what-you-use  # Arch
```

**Run:**
```bash
iwyu_tool.py -p build -- -Xiwyu --mapping_file=iwyu_mappings.imp 2>&1 | tee iwyu_report.txt
```

You will likely need a mappings file for CUDA, TensorRT, and OpenCV headers to
avoid false positives.

---

## 6. Fuzz Testing

Fuzzing is highly valuable for this project because the server processes
untrusted user input (base64 strings, raw image bytes, PDF files).

### 6.1 libFuzzer (Recommended)

libFuzzer is built into Clang and provides in-process, coverage-guided fuzzing
with minimal setup.

**Fuzz target: base64 decoder**

Create `tests/fuzz/fuzz_base64.cpp`:
```cpp
#include "turbo_ocr/common/encoding.h"
#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  turbo_ocr::base64_decode(reinterpret_cast<const char *>(data), size);
  return 0;
}
```

**Fuzz target: JSON serializer**

Create `tests/fuzz/fuzz_json_serializer.cpp`:
```cpp
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < sizeof(float) + 1) return 0;

  turbo_ocr::OCRResultItem item;
  std::memcpy(&item.confidence, data, sizeof(float));
  item.text.assign(reinterpret_cast<const char *>(data + sizeof(float)),
                   size - sizeof(float));

  std::vector<turbo_ocr::OCRResultItem> results = {item};
  auto json = turbo_ocr::results_to_json(results);
  return 0;
}
```

**Fuzz target: PNG decoder (FastPngDecoder)**

Create `tests/fuzz/fuzz_png_decoder.cpp`:
```cpp
#include "turbo_ocr/decode/fast_png_decoder.h"
#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  turbo_ocr::decode::FastPngDecoder decoder;
  try {
    decoder.decode(data, size);
  } catch (...) {
    // Expected for malformed input
  }
  return 0;
}
```

**Build and run fuzz targets:**
```bash
# Build with libFuzzer + ASan (recommended combination)
clang++ -g -O1 -fsanitize=fuzzer,address \
  -I include -I third_party \
  -std=c++20 \
  tests/fuzz/fuzz_base64.cpp \
  -o fuzz_base64

# Run (creates corpus/ directory automatically)
mkdir -p corpus/base64
./fuzz_base64 corpus/base64/ -max_len=65536 -jobs=$(nproc)

# With a seed corpus of valid base64 images
./fuzz_base64 corpus/base64/ tests/test_data/ -max_len=1048576
```

### 6.2 AFL++

Alternative to libFuzzer with better multi-core support.

```bash
# Install
sudo apt install aflplusplus   # or build from source

# Compile (harnesses are compatible with libFuzzer)
afl-clang-fast++ -g -O1 -fsanitize=address \
  -I include -I third_party \
  -std=c++20 \
  tests/fuzz/fuzz_base64.cpp \
  -o fuzz_base64_afl

# Run
mkdir -p afl_input afl_output
echo "SGVsbG8=" > afl_input/seed.txt   # valid base64 seed
afl-fuzz -i afl_input -o afl_output -- ./fuzz_base64_afl
```

### 6.3 Priority Fuzzing Targets

Ranked by attack surface and complexity:

1. **base64_decode()** -- directly processes user input, pointer arithmetic
2. **FastPngDecoder** -- parses complex binary format (Wuffs is hardened, but the wrapper code matters)
3. **JSON request parsing** -- Drogon/jsoncpp JSON body parsing
4. **det_postprocess** -- processes model output into polygons; malformed tensors could trigger issues
5. **ctc_decode** -- index math on recognition output

---

## 7. Integration and Load Testing

### 7.1 Existing pytest Suite

The project's pytest suite provides comprehensive black-box testing:

```bash
# Full suite
python tests/run_all.py

# Specific suites
python tests/run_all.py --suite unit
python tests/run_all.py --suite integration
python tests/run_all.py --suite regression
python tests/run_all.py --suite benchmark

# Single test
pytest tests/integration/test_ocr_endpoint.py -v -s

# Against a specific server
python tests/run_all.py --server-url http://remote:8000
```

### 7.2 k6 for HTTP Load Testing

k6 (by Grafana Labs) is the best modern choice for HTTP load testing. It supports
HTTP/1.1, HTTP/2, and gRPC natively.

**Install:**
```bash
# Arch
sudo pacman -S k6
# Debian/Ubuntu
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
  --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D68
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" \
  | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt update && sudo apt install k6
```

**Example k6 script (`tests/loadtest/k6_ocr.js`):**
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import encoding from 'k6/encoding';

// Load a test image as base64
const imageBytes = open('../test_data/png/simple_text.png', 'b');
const imageB64 = encoding.b64encode(imageBytes);

export const options = {
  scenarios: {
    sustained_load: {
      executor: 'constant-vus',
      vus: 16,
      duration: '60s',
    },
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '10s', target: 50 },
        { duration: '30s', target: 50 },
        { duration: '10s', target: 0 },
      ],
      startTime: '70s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<50', 'p(99)<100'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  // Test /ocr endpoint (base64 JSON)
  const res = http.post('http://localhost:8000/ocr', JSON.stringify({
    image: imageB64,
  }), {
    headers: { 'Content-Type': 'application/json' },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
    'has results': (r) => JSON.parse(r.body).results !== undefined,
  });
}

export function rawEndpoint() {
  // Test /ocr/raw endpoint (raw bytes)
  const res = http.post('http://localhost:8000/ocr/raw', imageBytes, {
    headers: { 'Content-Type': 'image/png' },
  });

  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}
```

**Run:**
```bash
k6 run tests/loadtest/k6_ocr.js
k6 run --out json=k6_results.json tests/loadtest/k6_ocr.js
```

### 7.3 ghz for gRPC Load Testing

The project already has C++ gRPC benchmark tools (`tools/grpc_bench.cpp`,
`tools/grpc_burst.cpp`), but ghz provides richer analysis.

**Install:**
```bash
go install github.com/bojand/ghz/cmd/ghz@latest
# or download binary from https://github.com/bojand/ghz/releases
```

**Run against the gRPC server:**
```bash
# Prepare a binary payload
cat tests/test_data/png/simple_text.png | base64 > /tmp/test_b64.txt

ghz --insecure \
  --proto proto/ocr.proto \
  --call ocr.OCRService.Recognize \
  --data '{"image_data": "'"$(cat /tmp/test_b64.txt)"'"}' \
  --concurrency 16 \
  --total 1000 \
  --connections 4 \
  localhost:50051
```

### 7.4 Locust (Python-based, Good for Complex Scenarios)

Already partially covered by the existing pytest benchmark suite, but Locust
provides a web UI and distributed testing.

```bash
pip install locust
```

**Example `tests/loadtest/locustfile.py`:**
```python
import base64
from pathlib import Path
from locust import HttpUser, task, between

IMAGE_B64 = base64.b64encode(
    Path("tests/test_data/png/simple_text.png").read_bytes()
).decode()

class OCRUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(3)
    def ocr_base64(self):
        self.client.post("/ocr", json={"image": IMAGE_B64})

    @task(2)
    def ocr_raw(self):
        self.client.post(
            "/ocr/raw",
            data=Path("tests/test_data/png/simple_text.png").read_bytes(),
            headers={"Content-Type": "image/png"},
        )

    @task(1)
    def health_check(self):
        self.client.get("/health")
```

**Run:**
```bash
locust -f tests/loadtest/locustfile.py --host http://localhost:8000 --headless \
  -u 32 -r 4 --run-time 60s
```

---

## 8. Benchmarking

### 8.1 Existing Python Benchmarks

The project has comprehensive Python benchmarks:

- `bench_latency.py` -- single-request latency percentiles
- `bench_throughput.py` -- maximum requests/sec
- `bench_concurrent.py` -- concurrent request handling
- `bench_parallel_images.py` -- batch/parallel image processing
- `bench_parallel_pdf.py` -- PDF endpoint performance
- `bench_report.py` -- generates consolidated report

```bash
python tests/run_all.py --suite benchmark
python tests/benchmark/bench_report.py
```

### 8.2 Google Benchmark (C++ Micro-benchmarks)

For benchmarking individual C++ functions (base64 decode, JSON serialization,
det_postprocess, CTC decode) without server overhead.

**Install:**
```bash
sudo apt install libbenchmark-dev    # Debian/Ubuntu
sudo pacman -S benchmark             # Arch
# or build from source: https://github.com/google/benchmark
```

**Example benchmark (`tests/cpp_bench/bench_base64.cpp`):**
```cpp
#include <benchmark/benchmark.h>
#include "turbo_ocr/common/encoding.h"
#include <string>
#include <random>

static std::string make_b64(size_t decoded_size) {
  static const char charset[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result((decoded_size * 4 + 2) / 3, 'A');
  std::mt19937 rng(42);
  for (auto &c : result) c = charset[rng() % 64];
  return result;
}

static void BM_Base64Decode_1KB(benchmark::State &state) {
  auto input = make_b64(1024);
  for (auto _ : state) {
    auto result = turbo_ocr::base64_decode(input);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * 1024);
}
BENCHMARK(BM_Base64Decode_1KB);

static void BM_Base64Decode_1MB(benchmark::State &state) {
  auto input = make_b64(1024 * 1024);
  for (auto _ : state) {
    auto result = turbo_ocr::base64_decode(input);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * 1024 * 1024);
}
BENCHMARK(BM_Base64Decode_1MB);

BENCHMARK_MAIN();
```

**Example benchmark (`tests/cpp_bench/bench_json_serializer.cpp`):**
```cpp
#include <benchmark/benchmark.h>
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include <vector>

static std::vector<turbo_ocr::OCRResultItem> make_items(int n) {
  std::vector<turbo_ocr::OCRResultItem> items(n);
  for (int i = 0; i < n; ++i) {
    items[i].text = "Sample text line " + std::to_string(i);
    items[i].confidence = 0.95f;
    items[i].box = {{{10 + i, 20}, {100 + i, 20}, {100 + i, 50}, {10 + i, 50}}};
  }
  return items;
}

static void BM_JsonSerialize_10(benchmark::State &state) {
  auto items = make_items(10);
  for (auto _ : state) {
    auto json = turbo_ocr::results_to_json(items);
    benchmark::DoNotOptimize(json);
  }
}
BENCHMARK(BM_JsonSerialize_10);

static void BM_JsonSerialize_100(benchmark::State &state) {
  auto items = make_items(100);
  for (auto _ : state) {
    auto json = turbo_ocr::results_to_json(items);
    benchmark::DoNotOptimize(json);
  }
}
BENCHMARK(BM_JsonSerialize_100);

BENCHMARK_MAIN();
```

**Build (manual, without modifying CMakeLists.txt):**
```bash
g++ -std=c++20 -O3 -march=native \
  -I include -I third_party \
  tests/cpp_bench/bench_base64.cpp \
  -lbenchmark -lpthread \
  -o bench_base64

./bench_base64 --benchmark_format=json --benchmark_out=bench_base64.json
```

### 8.3 Custom Benchmarks with Real Images

The `tests/test_data/` directory contains real test images. Use the existing
Python benchmarks for end-to-end performance, and target C++ micro-benchmarks
at the hot-path functions:

| Function | Location | Why benchmark |
|---|---|---|
| `base64_decode()` | `common/encoding.h` | Every /ocr request |
| `results_to_json()` | `common/serialization.h` | Every response |
| `det_postprocess` | `detection/det_postprocess.cpp` | Post-inference CPU work |
| `ctc_greedy_decode()` | `recognition/ctc_decode.cpp` | Per-batch CPU decode |
| `get_rotate_crop_image()` | `recognition/crop_utils.cpp` | CPU crop path |
| `FastPngDecoder::decode()` | `decode/fast_png_decoder.cpp` | PNG decode path |

---

## 9. CI/CD Integration

### Recommended Pipeline Stages

```
Stage 1: Static Analysis (fast, no build required)
  - cppcheck --project=compile_commands.json
  - clang-tidy on changed files

Stage 2: Build + Sanitizer Checks
  - Build with ENABLE_ASAN=ON, run unit/integration tests
  - Build with ENABLE_TSAN=ON, run concurrent integration tests
  - Build with UBSan, run unit tests

Stage 3: Functional Tests (requires running server)
  - python tests/run_all.py --suite unit --suite integration --suite regression

Stage 4: Performance (nightly, not on every PR)
  - python tests/run_all.py --suite benchmark
  - k6 load test
  - C++ micro-benchmarks (compare against baseline)

Stage 5: Deep Analysis (weekly)
  - Valgrind memcheck on CPU build
  - Compute Sanitizer memcheck/racecheck on GPU build
  - Fuzz testing (continuous, minimum 1 hour per target)
```

### Docker Integration

Since the project uses Docker (`docker/Dockerfile.gpu`), sanitizer builds
can be added as multi-stage targets:

```bash
# Build with ASan inside container
docker exec ocr-grpc bash -c \
  "cd /app && cmake -B build_asan -DENABLE_ASAN=ON -DTENSORRT_DIR=/usr/local/tensorrt && \
   cmake --build build_asan -j\$(nproc)"
```

---

## 10. Recommended Workflow

### For Every PR

1. Run `clang-tidy` on changed `.cpp` files
2. Run `cppcheck` on changed files
3. Build normally and run `python tests/run_all.py --suite unit --suite integration`

### Weekly

1. Build with `ENABLE_ASAN=ON`, run full test suite, check for leaks
2. Build with `ENABLE_TSAN=ON`, run concurrent tests
3. Run `compute-sanitizer --tool memcheck` with a few requests
4. Run fuzz targets for 1+ hour each

### Before Major Releases

1. Full Valgrind memcheck on CPU build
2. Full Compute Sanitizer suite (memcheck, racecheck, initcheck, synccheck)
3. k6 sustained load test (10+ minutes)
4. ghz gRPC load test
5. Fuzz all targets for 24+ hours
6. Run accuracy regression suite

### Quick Reference: Build Variants

| Variant | CMake Flags | Purpose |
|---|---|---|
| Release | `-DCMAKE_BUILD_TYPE=Release` | Production |
| ASan | `-DENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo` | Memory errors |
| TSan | `-DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo` | Data races |
| UBSan | `-DCMAKE_CXX_FLAGS="-fsanitize=undefined"` | Undefined behavior |
| ASan+UBSan | `-DENABLE_ASAN=ON -DCMAKE_CXX_FLAGS="-fsanitize=undefined"` | Combined |
| CPU Debug | `-DUSE_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Debug` | Valgrind-friendly |

---

## Sources

### Memory Leak Detection
- [The Art of Detecting Memory Leaks in C++ Applications](https://www.hackerone.com/blog/art-detecting-memory-leaks-c-applications)
- [How to Detect Memory Leaks in C++ (2026 Workflow)](https://thelinuxcode.com/how-to-detect-memory-leaks-in-c-a-practical-2026-workflow/)
- [Memory Error Checking: Sanitizers vs Valgrind (Red Hat)](https://developers.redhat.com/blog/2021/05/05/memory-error-checking-in-c-and-c-comparing-sanitizers-and-valgrind)
- [Valgrind vs AddressSanitizer (Undo)](https://undo.io/resources/gdb-watchpoint/a-quick-introduction-to-using-valgrind-and-addresssanitizer/)

### CUDA/GPU Testing
- [NVIDIA Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [Compile-Time Instrumentation for Compute Sanitizer (NVIDIA Blog)](https://developer.nvidia.com/blog/better-bug-detection-how-compile-time-instrumentation-for-compute-sanitizer-enhances-memory-safety)
- [Debugging CUDA with Compute Sanitizer (NVIDIA Blog)](https://developer.nvidia.com/blog/debugging-cuda-more-efficiently-with-nvidia-compute-sanitizer/)
- [CUDA C++ Best Practices Guide (2026)](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)
- [Rigorous Static Analysis for CUDA Code (Parasoft)](https://www.parasoft.com/blog/rigorous-static-analysis-cuda-code/)

### Load Testing
- [k6 - Modern Load Testing](https://k6.io/)
- [Performance Testing gRPC with k6](https://grafana.com/docs/k6/latest/testing-guides/performance-testing-grpc-services/)
- [ghz - gRPC Benchmarking Tool](https://ghz.sh/)
- [Best Load Testing Tools 2026 Guide](https://www.vervali.com/blog/best-load-testing-tools-in-2026-definitive-guide-to-jmeter-gatling-k6-loadrunner-locust-blazemeter-neoload-artillery-and-more/)

### Fuzzing
- [libFuzzer Documentation (LLVM)](https://llvm.org/docs/LibFuzzer.html)
- [libFuzzer Testing Handbook](https://appsec.guide/docs/fuzzing/c-cpp/libfuzzer/)
- [AFL++ Fuzzing in Depth](https://aflplus.plus/docs/fuzzing_in_depth/)
- [Fuzzing C++23 with libFuzzer (2025)](https://markaicode.com/fuzzing-cpp23-libfuzzer-crash-reproduction/)

### Static Analysis
- [Clang-Tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [C++ Code Review Best Practices 2025: Static Analysis](https://markaicode.com/cpp-code-review-static-analysis-2025/)
- [Static Checks with CMake (Kitware)](https://www.kitware.com/static-checks-with-cmake-cdash-iwyu-clang-tidy-lwyu-cpplint-and-cppcheck/)

### Benchmarking
- [Google Benchmark Guide](https://gist.github.com/MangaD/4e56a9d2fac0d0a6927d9e8af24b90bc)
- [How to Benchmark C++ with Google Benchmark (Bencher)](https://bencher.dev/learn/benchmarking/cpp/google-benchmark/)
- [Catch2 Benchmarks Documentation](https://github.com/catchorg/Catch2/blob/devel/docs/benchmarks.md)
