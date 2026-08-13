// POC v0 — MPSGraph crux probe (Apple Silicon, M3 Max).
//
// Tests the two things that decide whether a GPU-resident recognition path can
// beat the ~52 ms/img CPU recognition cost:
//   (1) fixed per-command-buffer (dispatch) overhead — the research claims
//       ~10-50 us; if true, batching into one command buffer makes overhead
//       negligible vs a 68 ms budget.
//   (2) throughput of a batched matmul the size of the rec model's big output
//       projection ([B*T, H] x [H, classes]), fed from MTLBuffer (zero-copy),
//       run as an MPSGraph in one command buffer.
//
// No .metal shaders => builds with Command Line Tools alone (MPSGraph is an
// Obj-C framework, no offline metal compiler needed).
//
// Build:
//   clang++ -std=c++17 -ObjC++ -fobjc-arc -O2 tools/probes/apple/mpsgraph_probe.mm \
//     -framework Metal -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph -framework Foundation \
//     -o build-cpu/mpsgraph_probe

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t) {
  return std::chrono::duration<double, std::micro>(clk::now() - t).count();
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::printf("device: %s\n", dev.name.UTF8String);
    std::printf("unified memory: %s | max threadgroup mem: %lu KB\n\n",
                dev.hasUnifiedMemory ? "yes" : "no",
                (unsigned long)dev.maxThreadgroupMemoryLength / 1024);

    // ---- (1) Empty command-buffer overhead ----------------------------------
    {
      const int N = 500;
      // warmup
      for (int i = 0; i < 50; ++i) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [cb commit]; [cb waitUntilCompleted];
      }
      std::vector<double> t;
      t.reserve(N);
      for (int i = 0; i < N; ++i) {
        auto s = clk::now();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
        t.push_back(us_since(s));
      }
      std::printf("[1] empty command buffer (commit+wait): median %.1f us  min %.1f us\n",
                  median(t), *std::min_element(t.begin(), t.end()));
      std::printf("    => 1 command buffer/img adds ~%.3f ms to a 68 ms budget\n\n",
                  median(t) / 1000.0);
    }

    // ---- (2) Batched matmul the size of the rec output projection -----------
    // rec_tiny: [B, T, classes] output. The heavy GEMM is the final projection
    // from hidden H to classes, done for B*T positions. Representative sizes:
    //   B=37 crops, T=80 time steps, H=512 hidden, classes=6906.
    const int B = 37, T = 80, H = 512, C = 6906;
    const int M = B * T;  // 2960 rows

    MPSGraph *g = [MPSGraph new];
    MPSGraphTensor *a = [g placeholderWithShape:@[@(M), @(H)]
                                       dataType:MPSDataTypeFloat16 name:@"a"];
    MPSGraphTensor *w = [g placeholderWithShape:@[@(H), @(C)]
                                       dataType:MPSDataTypeFloat16 name:@"w"];
    MPSGraphTensor *y = [g matrixMultiplicationWithPrimaryTensor:a
                                                 secondaryTensor:w name:@"y"];

    // MTLBuffer-backed feeds (fp16). On Apple Silicon these are unified memory,
    // so wrapping is genuinely copy-free.
    auto mkbuf = [&](size_t elems) {
      id<MTLBuffer> b = [dev newBufferWithLength:elems * sizeof(uint16_t)
                                         options:MTLResourceStorageModeShared];
      uint16_t *p = (uint16_t *)b.contents;              // half=0x3C00 -> 1.0
      for (size_t i = 0; i < elems; ++i) p[i] = 0x3C00;
      return b;
    };
    id<MTLBuffer> aBuf = mkbuf((size_t)M * H);
    id<MTLBuffer> wBuf = mkbuf((size_t)H * C);
    id<MTLBuffer> yBuf = mkbuf((size_t)M * C);  // pre-allocated RESIDENT output

    MPSGraphTensorData *aTD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:aBuf shape:@[@(M), @(H)]
                                             dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *wTD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:wBuf shape:@[@(H), @(C)]
                                             dataType:MPSDataTypeFloat16];
    MPSGraphTensorData *yTD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:yBuf shape:@[@(M), @(C)]
                                             dataType:MPSDataTypeFloat16];
    NSDictionary *feeds = @{a : aTD, w : wTD};
    double flops = 2.0 * M * H * C;  // MAC*2

    std::printf("[2] batched GEMM  [%d x %d] x [%d x %d]  (fp16, rec-projection size)\n",
                M, H, H, C);

    // (2a) NAIVE: synchronous run — allocates + host-syncs the 40 MB output every
    // call (the anti-pattern; a per-stage host round-trip).
    for (int i = 0; i < 10; ++i)
      (void)[g runWithMTLCommandQueue:q feeds:feeds targetTensors:@[y] targetOperations:nil];
    {
      std::vector<double> t;
      for (int i = 0; i < 100; ++i) {
        auto s = clk::now();
        (void)[g runWithMTLCommandQueue:q feeds:feeds targetTensors:@[y] targetOperations:nil];
        t.push_back(us_since(s));
      }
      std::printf("    (2a) host-synced run : median %.3f ms  (%.0f GFLOP/s)  <- output copied to host each call\n",
                  median(t) / 1000.0, flops / (median(t) * 1e3));
    }

    // (2b) RESIDENT: compile to an executable, encode into our own command
    // buffer, output stays in yBuf (MTLBuffer) — NO host sync. This is the
    // GPU-resident path (argmax would consume yBuf on the GPU next).
    MPSGraphDevice *gdev = [MPSGraphDevice deviceWithMTLDevice:dev];
    MPSGraphExecutable *exe = [g
        compileWithDevice:gdev
                    feeds:@{a : [MPSGraphShapedType.alloc initWithShape:@[@(M),@(H)] dataType:MPSDataTypeFloat16],
                            w : [MPSGraphShapedType.alloc initWithShape:@[@(H),@(C)] dataType:MPSDataTypeFloat16]}
            targetTensors:@[y] targetOperations:nil compilationDescriptor:nil];
    for (int i = 0; i < 10; ++i) {
      MPSCommandBuffer *cb = [MPSCommandBuffer commandBufferFromCommandQueue:q];
      [exe encodeToCommandBuffer:cb inputsArray:@[aTD, wTD] resultsArray:@[yTD] executionDescriptor:nil];
      [cb commit]; [cb waitUntilCompleted];
    }
    {
      std::vector<double> wall, gpu;
      for (int i = 0; i < 100; ++i) {
        auto s = clk::now();
        MPSCommandBuffer *cb = [MPSCommandBuffer commandBufferFromCommandQueue:q];
        [exe encodeToCommandBuffer:cb inputsArray:@[aTD, wTD] resultsArray:@[yTD] executionDescriptor:nil];
        // MPSCommandBuffer defers to a root MTLCommandBuffer; commitAndContinue
        // not used — plain commit so GPUStart/EndTime are populated.
        [cb.rootCommandBuffer commit];
        [cb.rootCommandBuffer waitUntilCompleted];
        wall.push_back(us_since(s));
        double g = (cb.rootCommandBuffer.GPUEndTime - cb.rootCommandBuffer.GPUStartTime) * 1e6; // s->us
        if (g > 0) gpu.push_back(g);
      }
      std::printf("    (2b) RESIDENT wall: median %.3f ms   GPU-exec-time: median %.3f ms  min %.3f ms  (%.0f GFLOP/s on GPU-time)\n",
                  median(wall) / 1000.0,
                  gpu.empty() ? -1 : median(gpu) / 1000.0,
                  gpu.empty() ? -1 : *std::min_element(gpu.begin(), gpu.end()) / 1000.0,
                  gpu.empty() ? 0 : flops / (median(gpu) * 1e3));
    }
    std::printf("    => the full CPU rec is ~52 ms/img; this GEMM is the heaviest single rec op\n");

    return 0;
  }
}
