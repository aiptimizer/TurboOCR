// MEASURE (Option B): rec_tiny conv backbone as im2col+GEMM via MPSMatrixMultiplication
// chained in ONE command buffer (batch 64) — the buffer-based MPS-primitive path
// that avoids MPSCNN's depthwise/texture restrictions. Verdict vs MPSGraph 118us/crop.
// Dummy zero buffers (compute is shape-dependent). Includes final FC(160->6906).
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
struct C{int ic,oc,k,s,g,ih,iw;};

int main(){ @autoreleasepool{
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  const int B=64;
  std::vector<C> cfg={
    {3,24,3,2,1,48,320},{24,48,3,2,1,24,160},{48,48,3,1,48,12,80},{48,12,1,1,1,1,1},{12,48,1,1,1,1,1},
    {48,96,1,1,1,12,80},{96,48,1,1,1,12,80},{48,48,3,1,48,12,80},{48,96,1,1,1,12,80},{96,48,1,1,1,12,80},
    {48,48,3,2,48,12,80},{48,96,1,1,1,6,40},{96,96,1,1,1,6,40},{96,96,3,1,96,6,40},{96,24,1,1,1,1,1},
    {24,96,1,1,1,1,1},{96,192,1,1,1,6,40},{192,96,1,1,1,6,40},{96,96,3,1,96,6,40},{96,192,1,1,1,6,40},
    {192,96,1,1,1,6,40},{96,96,3,2,96,6,40},{96,192,1,1,1,3,40},{192,160,1,1,1,3,40},{160,160,3,1,160,3,40},
    {160,40,1,1,1,1,1},{40,160,1,1,1,1,1},{160,320,1,1,1,3,40},{320,160,1,1,1,3,40},{160,160,3,1,160,3,40},
    {160,320,1,1,1,3,40},{320,160,1,1,1,3,40},{160,160,3,1,160,3,40},{160,320,1,1,1,3,40},{320,160,1,1,1,3,40},
    {160,160,1,1,160,1,40},{160,160,1,1,1,1,40}};
  const int T=40, CLS=6906, HID=160;

  auto mat=[&](int r,int c){ MPSMatrixDescriptor* d=[MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c rowBytes:(size_t)c*4 dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:[dev newBufferWithLength:(size_t)r*c*4 options:MTLResourceStorageModePrivate] descriptor:d]; };
  // pre-build one GEMM op + matrices per conv (im2col sizes). Depthwise: model as
  // grouped -> M=B*oh*ow*oc rows, K=k*k, N=1 (true depthwise FLOPs; tiny).
  struct G{ MPSMatrixMultiplication* op; MPSMatrix *A,*W,*Cm; };
  std::vector<G> gs; double gflop=0;
  for(auto&c:cfg){ int oh=(c.ih>1)?c.ih/c.s:1, ow=(c.iw>1)?c.iw/c.s:1;
    int M,K,N;
    if(c.g==1){ M=B*oh*ow; K=c.ic*c.k*c.k; N=c.oc; }
    else { M=B*oh*ow*c.oc; K=c.k*c.k; N=1; }         // depthwise
    gflop += 2.0*M*K*N/1e9;
    MPSMatrix* A=mat(M,K); MPSMatrix* W=mat(K,N); MPSMatrix* Cm=mat(M,N);
    MPSMatrixMultiplication* op=[[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1 beta:0];
    gs.push_back({op,A,W,Cm}); }
  // final FC 160->6906 over B*T rows
  MPSMatrix* fcA=mat(B*T,HID); MPSMatrix* fcW=mat(HID,CLS); MPSMatrix* fcC=mat(B*T,CLS);
  MPSMatrixMultiplication* fc=[[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:B*T resultColumns:CLS interiorColumns:HID alpha:1 beta:0];
  gflop += 2.0*B*T*HID*CLS/1e9;

  auto run=[&]()->double{ auto t0=clk::now(); id<MTLCommandBuffer> cb=[q commandBuffer];
    for(auto&g:gs) [g.op encodeToCommandBuffer:cb leftMatrix:g.A rightMatrix:g.W resultMatrix:g.Cm];
    [fc encodeToCommandBuffer:cb leftMatrix:fcA rightMatrix:fcW resultMatrix:fcC];
    [cb commit]; [cb waitUntilCompleted]; return ms(t0); };
  for(int i=0;i<5;i++) run();
  std::vector<double> ts; for(int i=0;i<30;i++) ts.push_back(run());
  std::sort(ts.begin(),ts.end()); double med=ts[ts.size()/2];
  std::printf("Option B: rec_tiny as %zu im2col-GEMMs + FC, batch %d, ONE command buffer\n", gs.size(), B);
  std::printf("  GEMM+FC FLOPs = %.1f GFLOP/batch  (im2col overhead NOT included)\n", gflop);
  std::printf("  median %.3f ms/batch = %.1f us/crop   (MPSGraph = 118 us/crop)\n", med, med*1000/B);
  std::printf("  => %.2fx vs MPSGraph (compute-only lower bound; +im2col kernels + elementwise would add)\n", 118.0/(med*1000/B));
  return 0;
}}
