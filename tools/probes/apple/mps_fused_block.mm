// Verdict experiment: does a FUSED Metal kernel (depthwise+pointwise+GELU, depthwise
// kept in registers) beat MPSGraph on one real rec_tiny block? Validates vs ORT
// golden, then benchmarks fused-Metal vs MPSGraph at batch 64.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cmath>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
static std::vector<float> loadbin(const char* p){ FILE* f=fopen(p,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<float> v(n/4); fread(v.data(),4,v.size(),f); fclose(f); return v; }

int main(){ @autoreleasepool{
  const int B=64, Cin=48, Cout=96, H=12, W=80;
  auto in1=loadbin("/tmp/blk_add_5.bin");     // [1,48,12,80]
  auto gold=loadbin("/tmp/blk_gelu_2.bin");   // [1,96,12,80]
  auto wdw=loadbin("/tmp/w_dw.bin");          // [48,9]
  auto wpw=loadbin("/tmp/w_pw.bin");          // [96,48]
  auto bdw=loadbin("/tmp/dw_bias.bin");       // [48]
  auto bpw=loadbin("/tmp/pw_bias.bin");       // [96]
  const size_t inN=(size_t)Cin*H*W, outN=(size_t)Cout*H*W;

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue]; NSError* e=nil;
  id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/mps_fused_block.metallib"] error:&e];
  if(!lib){ std::fprintf(stderr,"metallib: %s\n",e.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fused_dwsep"] error:&e];
  auto B_=[&](size_t n){ return [dev newBufferWithLength:std::max<size_t>(n,16) options:MTLResourceStorageModeShared]; };
  // batch-B input (replicate the single golden input B times)
  id<MTLBuffer> inBuf=B_((size_t)B*inN*4); float* ip=(float*)inBuf.contents;
  for(int b=0;b<B;b++) memcpy(ip+(size_t)b*inN, in1.data(), inN*4);
  id<MTLBuffer> wdwB=[dev newBufferWithBytes:wdw.data() length:wdw.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bdwB=[dev newBufferWithBytes:bdw.data() length:bdw.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> wpwB=[dev newBufferWithBytes:wpw.data() length:wpw.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bpwB=[dev newBufferWithBytes:bpw.data() length:bpw.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> outBuf=B_((size_t)B*outN*4);
  uint Bu=B; id<MTLBuffer> Bbuf=[dev newBufferWithBytes:&Bu length:4 options:MTLResourceStorageModeShared];

  auto encodeFused=[&](id<MTLCommandBuffer> cb){ id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
    [ce setComputePipelineState:pso];
    id<MTLBuffer> bufs[]={inBuf,wdwB,bdwB,wpwB,bpwB,outBuf,Bbuf};
    for(int i=0;i<7;i++)[ce setBuffer:bufs[i] offset:0 atIndex:i];
    [ce setThreadgroupMemoryLength:Cout*Cin*4 atIndex:0]; [ce dispatchThreads:MTLSizeMake(W,H,B) threadsPerThreadgroup:MTLSizeMake(16,4,1)]; [ce endEncoding]; };

  // ---- correctness (batch 1 slice vs golden) ----
  { id<MTLCommandBuffer> cb=[q commandBuffer]; encodeFused(cb); [cb commit]; [cb waitUntilCompleted]; }
  const float* op=(const float*)outBuf.contents;
  double maxrel=0, maxabs=0; int nbad=0;
  for(size_t i=0;i<outN;i++){ float a=op[i], g=gold[i]; double d=std::fabs(a-g); maxabs=std::max(maxabs,d);
    double r=d/(std::fabs(g)+1e-3); maxrel=std::max(maxrel,r); if(r>1e-2 && d>1e-2) nbad++; }
  std::printf("FUSED correctness vs ORT golden: maxabs=%.4g maxrel=%.4g  bad(>1%%)=%d/%zu\n", maxabs,maxrel,nbad,outN);

  // ---- benchmark fused ----
  for(int i=0;i<10;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; encodeFused(cb); [cb commit]; [cb waitUntilCompleted]; }
  { const int IT=200; auto t=clk::now(); for(int i=0;i<IT;i++){ id<MTLCommandBuffer> cb=[q commandBuffer]; encodeFused(cb); [cb commit]; [cb waitUntilCompleted]; }
    double per=ms(t)/IT; std::printf("FUSED Metal:  %.3f ms/batch(%d) = %.2f us/crop\n", per, B, per*1000.0/B); }

  // ---- MPSGraph version of the SAME block ----
  MPSGraph* g=[MPSGraph new]; MPSGraphDevice* gd=[MPSGraphDevice deviceWithMTLDevice:dev];
  MPSGraphTensor* x=[g placeholderWithShape:@[@(B),@(Cin),@(H),@(W)] dataType:MPSDataTypeFloat32 name:@"x"];
  NSData* dwD=[NSData dataWithBytes:wdw.data() length:wdw.size()*4];
  MPSGraphTensor* wdwT=[g constantWithData:dwD shape:@[@(Cin),@1,@3,@3] dataType:MPSDataTypeFloat32];
  MPSGraphConvolution2DOpDescriptor* dwDesc=[MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1 groups:Cin paddingLeft:1 paddingRight:1 paddingTop:1 paddingBottom:1 paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
  MPSGraphTensor* dwOut=[g convolution2DWithSourceTensor:x weightsTensor:wdwT descriptor:dwDesc name:nil];
  NSData* bdwD=[NSData dataWithBytes:bdw.data() length:bdw.size()*4];
  MPSGraphTensor* bdwT=[g constantWithData:bdwD shape:@[@1,@(Cin),@1,@1] dataType:MPSDataTypeFloat32];
  dwOut=[g additionWithPrimaryTensor:dwOut secondaryTensor:bdwT name:nil];
  NSData* pwD=[NSData dataWithBytes:wpw.data() length:wpw.size()*4];
  MPSGraphTensor* wpwT=[g constantWithData:pwD shape:@[@(Cout),@(Cin),@1,@1] dataType:MPSDataTypeFloat32];
  MPSGraphConvolution2DOpDescriptor* pwDesc=[MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1 groups:1 paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0 paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
  MPSGraphTensor* pwOut=[g convolution2DWithSourceTensor:dwOut weightsTensor:wpwT descriptor:pwDesc name:nil];
  NSData* bpwD=[NSData dataWithBytes:bpw.data() length:bpw.size()*4];
  MPSGraphTensor* bpwT=[g constantWithData:bpwD shape:@[@1,@(Cout),@1,@1] dataType:MPSDataTypeFloat32];
  pwOut=[g additionWithPrimaryTensor:pwOut secondaryTensor:bpwT name:nil];
  // GELU: x*0.5*(1+erf(x/sqrt2))
  MPSGraphTensor* half=[g constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
  MPSGraphTensor* one=[g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
  MPSGraphTensor* isq2=[g constantWithScalar:0.70710678118 dataType:MPSDataTypeFloat32];
  MPSGraphTensor* er=[g erfWithTensor:[g multiplicationWithPrimaryTensor:pwOut secondaryTensor:isq2 name:nil] name:nil];
  MPSGraphTensor* gelu=[g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:pwOut secondaryTensor:half name:nil] secondaryTensor:[g additionWithPrimaryTensor:one secondaryTensor:er name:nil] name:nil];
  MPSGraphExecutable* exe=[g compileWithDevice:gd feeds:@{x:[MPSGraphShapedType.alloc initWithShape:@[@(B),@(Cin),@(H),@(W)] dataType:MPSDataTypeFloat32]} targetTensors:@[gelu] targetOperations:nil compilationDescriptor:nil];
  MPSGraphTensorData* xTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:inBuf shape:@[@(B),@(Cin),@(H),@(W)] dataType:MPSDataTypeFloat32];
  id<MTLBuffer> mgOut=B_((size_t)B*outN*4);
  MPSGraphTensorData* oTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:mgOut shape:@[@(B),@(Cout),@(H),@(W)] dataType:MPSDataTypeFloat32];
  auto encodeMG=[&](MPSCommandBuffer* cb){ [exe encodeToCommandBuffer:cb inputsArray:@[xTD] resultsArray:@[oTD] executionDescriptor:nil]; };
  // warm + validate MG vs golden
  { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; encodeMG(cb); [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
  const float* mp=(const float*)mgOut.contents; double mgrel=0; for(size_t i=0;i<outN;i++){ double r=std::fabs(mp[i]-gold[i])/(std::fabs(gold[i])+1e-3); mgrel=std::max(mgrel,r);}
  std::printf("MPSGraph correctness vs golden: maxrel=%.4g\n", mgrel);
  for(int i=0;i<10;i++){ MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; encodeMG(cb); [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
  { const int IT=200; auto t=clk::now(); for(int i=0;i<IT;i++){ MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; encodeMG(cb); [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
    double per=ms(t)/IT; std::printf("MPSGraph 1 block: %.3f ms/batch(%d) = %.2f us/crop\n", per, B, per*1000.0/B); }
  return 0;
}}
