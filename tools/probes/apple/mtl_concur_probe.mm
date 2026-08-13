// Does a single M3 Max GPU run INDEPENDENT command buffers (from separate queues,
// separate threads) CONCURRENTLY, or serialize them? Uses a moderate compute
// kernel that UNDER-utilizes the GPU per instance (like the rec workload), so if
// the GPU can overlap independent buffers, K concurrent should be ~K x faster than
// K serial. Isolates GPU/driver behavior from MPSGraph.
#import <Metal/Metal.h>
#include <chrono>
#include <thread>
#include <vector>
#include <cstdio>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

static const char* SRC = R"(
#include <metal_stdlib>
using namespace metal;
// moderate per-thread work; few threads so one dispatch under-fills the GPU
kernel void busy(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]){
  float x = gid*1e-6f;
  for(uint i=0;i<iters;i++){ x = fma(x, 1.0000001f, 0.9999999f); x = fract(x*1.37f)+0.1f; }
  out[gid]=x;
}
)";

int main(int argc,char**argv){ @autoreleasepool{
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil;
  id<MTLLibrary> lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
  if(!lib){ std::fprintf(stderr,"lib: %s\n",e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"busy"] error:&e];
  const int THREADS = argc>1?atoi(argv[1]):4096;   // small grid => under-utilizes GPU (M3 Max has 1000s of ALUs)
  const uint ITERS = argc>2?atoi(argv[2]):200000;  // heavy per-thread loop => each dispatch takes ~ms
  const int REP = 40;
  id<MTLBuffer> itbuf=[dev newBufferWithBytes:&ITERS length:4 options:0];

  auto oneDispatch=[&](id<MTLCommandQueue> q, id<MTLBuffer> out){
    id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
    [ce setComputePipelineState:pso]; [ce setBuffer:out offset:0 atIndex:0]; [ce setBuffer:itbuf offset:0 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(THREADS,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [ce endEncoding];
    [cb commit]; [cb waitUntilCompleted]; };

  // single-dispatch time
  { id<MTLCommandQueue> q=[dev newCommandQueue]; id<MTLBuffer> out=[dev newBufferWithLength:THREADS*4 options:MTLResourceStorageModeShared];
    for(int i=0;i<3;i++) oneDispatch(q,out);  // warm
    auto t=clk::now(); for(int i=0;i<REP;i++) oneDispatch(q,out); double d=ms(t)/REP;
    std::printf("single dispatch: %.3f ms (grid=%d iters=%u)\n", d, THREADS, ITERS); }

  for(int K : {2,3,4,6,8}){
    // SERIAL baseline: K*REP dispatches on one queue
    id<MTLCommandQueue> q=[dev newCommandQueue]; id<MTLBuffer> out=[dev newBufferWithLength:THREADS*4 options:MTLResourceStorageModeShared];
    auto t=clk::now(); for(int i=0;i<K*REP;i++) oneDispatch(q,out); double serial=ms(t);
    // CONCURRENT: K threads, each own queue+buffer, each does REP dispatches, at the same time
    std::vector<id<MTLCommandQueue>> qs(K); std::vector<id<MTLBuffer>> outs(K);
    for(int k=0;k<K;k++){ qs[k]=[dev newCommandQueue]; outs[k]=[dev newBufferWithLength:THREADS*4 options:MTLResourceStorageModeShared]; }
    auto t2=clk::now();
    std::vector<std::thread> th;
    for(int k=0;k<K;k++) th.emplace_back([&,k]{ @autoreleasepool{ for(int i=0;i<REP;i++) oneDispatch(qs[k],outs[k]); }});
    for(auto& x:th) x.join(); double conc=ms(t2);
    std::printf("K=%d: serial %.1f ms | concurrent %.1f ms | speedup %.2fx  (%.0f%% of ideal %dx)\n",
      K, serial, conc, serial/conc, 100.0*(serial/conc)/K, K);
  }
  return 0;
}}
