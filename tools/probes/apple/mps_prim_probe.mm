// Does an MPS PRIMITIVE (MPSMatrixMultiplication) have low per-encode overhead
// (~us, like raw Metal) or MPSGraph-like (~ms)? Decides whether reimplementing
// rec_tiny as chained MPS GEMMs beats MPSGraph's ~5ms/executable dispatch tax.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

int main(){ @autoreleasepool{
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q=[dev newCommandQueue];
  // GEMM sizes typical of rec_tiny conv-as-GEMM: M=48*batch rows, K=~128, N=~128
  const int M=48*64, K=128, N=128;  // ~ one conv layer for batch 64
  auto mat=[&](int r,int c){ MPSMatrixDescriptor* d=[MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c rowBytes:c*4 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:[dev newBufferWithLength:(size_t)r*c*4 options:MTLResourceStorageModeShared] descriptor:d]; };
  MPSMatrix* A=mat(M,K); MPSMatrix* B=mat(K,N); MPSMatrix* C=mat(M,N);
  MPSMatrixMultiplication* gemm=[[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1 beta:0];
  // warm
  for(int i=0;i<5;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; [gemm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C]; [cb commit]; [cb waitUntilCompleted]; }

  for(int L : {1,10,37,100}){
    auto t=clk::now();
    id<MTLCommandBuffer> cb=[q commandBuffer];
    for(int i=0;i<L;i++) [gemm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C];  // L GEMMs, ONE command buffer
    [cb commit]; [cb waitUntilCompleted];
    std::printf("%3d GEMMs (M=%d K=%d N=%d) in ONE cmd buffer: %.3f ms total = %.3f ms/GEMM\n", L, M,K,N, ms(t), ms(t)/L);
  }
  // compare: L GEMMs each in its OWN command buffer commit+wait
  for(int L : {37}){
    auto t=clk::now();
    for(int i=0;i<L;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; [gemm encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B resultMatrix:C]; [cb commit]; [cb waitUntilCompleted]; }
    std::printf("%3d GEMMs each own cmd buffer commit+wait: %.3f ms = %.3f ms/GEMM\n", L, ms(t), ms(t)/L);
  }
  return 0;
}}
