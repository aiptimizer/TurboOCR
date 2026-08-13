// POC v1 — full GPU-resident recognition on Apple Silicon.
// One MTLCommandBuffer does: warp/resize/normalize N line-crops out of ONE
// source image (Metal compute kernel) -> feed the crops MTLBuffer straight into
// the MPSGraph rec_tiny graph -> logits. NO host round-trips between pre-proc
// and inference (the CUDA-resident-pipeline analogue, on Metal).
//
// Build:
//   xcrun -sdk macosx metal -O -c tools/probes/apple/warp.metal -o build-cpu/warp.air
//   xcrun -sdk macosx metallib build-cpu/warp.air -o build-cpu/warp.metallib
//   clang++ -std=c++17 -ObjC++ -fobjc-arc -O2 tools/probes/apple/mps_rec_v1.mm \
//     -framework Metal -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph -framework Foundation \
//     -o build-cpu/mps_rec_v1
// Run:  ./build-cpu/mps_rec_v1 <export_dir> <batch>

#import <Metal/Metal.h>
#include "mps_rec_build.h"
#include <chrono>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

int main(int argc, char** argv){
 @autoreleasepool{
  std::string dir = argc>1? argv[1] : "rec_export";
  int B = argc>2? atoi(argv[2]) : 37;
  const int H=48, Wc=320, SRCW=400, SRCH=533;
  NSString* d=[NSString stringWithUTF8String:dir.c_str()];

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q=[dev newCommandQueue];

  // ---- warp compute pipeline from the compiled metallib ----
  NSError* err=nil;
  id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/warp.metallib"] error:&err];
  if(!lib){ std::fprintf(stderr,"metallib load failed: %s\n", err.localizedDescription.UTF8String); return 1; }
  id<MTLComputePipelineState> warp=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"warp_crops"] error:&err];
  if(!warp){ std::fprintf(stderr,"pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }

  // ---- synthetic source page as a sampled texture ----
  MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:SRCW height:SRCH mipmapped:NO];
  td.usage=MTLTextureUsageShaderRead;
  id<MTLTexture> srcTex=[dev newTextureWithDescriptor:td];
  { std::vector<uint8_t> px((size_t)SRCW*SRCH*4);
    for(int y=0;y<SRCH;y++)for(int x=0;x<SRCW;x++){ size_t i=((size_t)y*SRCW+x)*4; px[i]=x*255/SRCW; px[i+1]=y*255/SRCH; px[i+2]=128; px[i+3]=255; }
    [srcTex replaceRegion:MTLRegionMake2D(0,0,SRCW,SRCH) mipmapLevel:0 withBytes:px.data() bytesPerRow:SRCW*4]; }

  // ---- per-crop inverse homographies (dst-pixel -> src-pixel), flat 9/crop ----
  std::vector<float> Hm((size_t)B*9);
  float strip=(float)SRCH/B;
  for(int c=0;c<B;c++){ float* h=&Hm[c*9];
    h[0]=(float)SRCW/Wc; h[1]=0;          h[2]=0;
    h[3]=0;              h[4]=strip/H;    h[5]=c*strip;
    h[6]=0;              h[7]=0;          h[8]=1; }
  id<MTLBuffer> Hbuf=[dev newBufferWithBytes:Hm.data() length:Hm.size()*4 options:MTLResourceStorageModeShared];
  uint32_t dims[4]={(uint32_t)B,3,(uint32_t)H,(uint32_t)Wc};
  id<MTLBuffer> dimBuf=[dev newBufferWithBytes:dims length:16 options:MTLResourceStorageModeShared];

  // ---- crops buffer (warp output == rec input) ----
  id<MTLBuffer> cropsBuf=[dev newBufferWithLength:(size_t)B*3*H*Wc*sizeof(float) options:MTLResourceStorageModeShared];

  // ---- build + compile rec_tiny ----
  NSData* jd=[NSData dataWithContentsOfFile:[d stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* G=[NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
  NSData* Wt=[NSData dataWithContentsOfFile:[d stringByAppendingPathComponent:@"weights.bin"]];
  MPSGraph* g=[MPSGraph new];
  RecIO io=buildRecGraph(g, G, (const float*)Wt.bytes, B);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  MPSGraphExecutable* exe=[g compileWithDevice:gdev
      feeds:@{io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32]}
      targetTensors:@[io.output] targetOperations:nil compilationDescriptor:nil];

  long T=io.output.shape.count>1?[io.output.shape[1] longValue]:40, C=[io.output.shape.lastObject longValue];
  id<MTLBuffer> outBuf=[dev newBufferWithLength:(size_t)B*T*C*sizeof(float) options:MTLResourceStorageModeShared];
  MPSGraphTensorData* cropsTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:cropsBuf shape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* outTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:outBuf shape:@[@(B),@(T),@(C)] dataType:MPSDataTypeFloat32];

  std::printf("device: %s | POC v1 GPU-resident: warp %d crops -> rec_tiny, one command buffer\n", dev.name.UTF8String, B);

  auto encodeWarp=[&](id<MTLCommandBuffer> cb){
    id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setComputePipelineState:warp];
    [e setTexture:srcTex atIndex:0];
    [e setBuffer:cropsBuf offset:0 atIndex:0];
    [e setBuffer:Hbuf offset:0 atIndex:1];
    [e setBuffer:dimBuf offset:0 atIndex:2];
    [e dispatchThreads:MTLSizeMake(Wc,H,B) threadsPerThreadgroup:MTLSizeMake(16,8,1)];
    [e endEncoding];
  };
  auto gpums=[&](id<MTLCommandBuffer> cb){ return (cb.GPUEndTime-cb.GPUStartTime)*1000.0; };
  auto bench=[&](const char* tag, void(^body)(id<MTLCommandBuffer>)){
    for(int i=0;i<10;i++){ MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; body(cb); [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
    std::vector<double> t;
    for(int i=0;i<50;i++){ MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; body(cb); [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; t.push_back(gpums(cb.rootCommandBuffer)); }
    std::sort(t.begin(),t.end());
    std::printf("  %-32s GPU: median %.3f ms  min %.3f ms  (%.3f ms/crop)\n", tag, t[t.size()/2], t.front(), t[t.size()/2]/B);
  };

  // (a) warp only  (b) rec only  (c) FUSED warp+rec in one command buffer
  bench("[a] warp only", ^(id<MTLCommandBuffer> cb){ encodeWarp(cb); });
  bench("[b] rec only", ^(id<MTLCommandBuffer> cb){ [exe encodeToCommandBuffer:(MPSCommandBuffer*)cb inputsArray:@[cropsTD] resultsArray:@[outTD] executionDescriptor:nil]; });
  bench("[c] FUSED warp+rec (resident)", ^(id<MTLCommandBuffer> cb){ encodeWarp(cb); [exe encodeToCommandBuffer:(MPSCommandBuffer*)cb inputsArray:@[cropsTD] resultsArray:@[outTD] executionDescriptor:nil]; });

  std::printf("\n  CPU/MLAS reference for %d crops: warp+rec ~= %.1f ms (rec ~1.4 ms/crop + host warp/upload)\n", B, 1.4*B);
  return 0;
 }
}
