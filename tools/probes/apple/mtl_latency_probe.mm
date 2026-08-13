// Measure raw Metal command-buffer submit->complete latency on this machine, to
// separate "MPSGraph is slow" from "each commit+waitUntilCompleted has fixed latency".
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
#include <vector>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

int main(){ @autoreleasepool{
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q=[dev newCommandQueue];
  const int N=200;
  // warm
  for(int i=0;i<5;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; [cb commit]; [cb waitUntilCompleted]; }

  // (1) empty cb, commit+wait each
  { auto t=clk::now(); for(int i=0;i<N;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; [cb commit]; [cb waitUntilCompleted]; }
    std::printf("empty cb, commit+wait EACH:      %.3f ms/cb\n", ms(t)/N); }

  // (2) empty cb, commit all, wait only last
  { auto t=clk::now(); id<MTLCommandBuffer> last=nil; for(int i=0;i<N;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; [cb commit]; last=cb; } [last waitUntilCompleted];
    std::printf("empty cb, commit-all wait-last:  %.3f ms/cb\n", ms(t)/N); }

  // (3) empty cb, addCompletedHandler, no wait (fire and forget, then drain)
  { __block int done=0; auto t=clk::now(); id<MTLCommandBuffer> last=nil;
    for(int i=0;i<N;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; [cb addCompletedHandler:^(id<MTLCommandBuffer>){ done++; }]; [cb commit]; last=cb; }
    [last waitUntilCompleted]; std::printf("empty cb, completion-handler:     %.3f ms/cb (done=%d)\n", ms(t)/N, done); }

  // (4) a trivial blit (touch GPU) commit+wait each
  id<MTLBuffer> a=[dev newBufferWithLength:1024 options:MTLResourceStorageModeShared];
  id<MTLBuffer> b=[dev newBufferWithLength:1024 options:MTLResourceStorageModeShared];
  { auto t=clk::now(); for(int i=0;i<N;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLBlitCommandEncoder> e=[cb blitCommandEncoder]; [e copyFromBuffer:a sourceOffset:0 toBuffer:b destinationOffset:0 size:1024]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
    std::printf("blit cb, commit+wait EACH:       %.3f ms/cb\n", ms(t)/N); }

  // (5) MPSCommandBuffer (what MPSGraph uses) empty commit+wait each
  { auto t=clk::now(); for(int i=0;i<N;i++){ MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
    std::printf("MPSCommandBuffer empty c+w EACH: %.3f ms/cb\n", ms(t)/N); }
  return 0;
}}
