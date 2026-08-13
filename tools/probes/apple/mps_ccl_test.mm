// Validate the GPU DB connected-components kernels (db_ccl.metal) against
// OpenCV's connectedComponentsWithStats on the REAL det probability map, and
// time the GPU CCL. Step toward on-GPU box extraction (full residency).

#import <Metal/Metal.h>
#include "mps_rec_build.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "turbo_ocr/analysis/detection/det_postprocess.h"
#include "turbo_ocr/base/geometry/box.h"
#include <chrono>
#include <vector>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

int main(int argc,char** argv){ @autoreleasepool{
  NSString* detDir=[NSString stringWithUTF8String: argc>1?argv[1]:"det_export"];
  const int SZ=640; const float THRESH=0.3f; const int MINAREA=9;
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  NSError* e=nil;
  id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/db_ccl.metallib"] error:&e];
  if(!lib){ std::fprintf(stderr,"metallib: %s\n", e.localizedDescription.UTF8String); return 1; }
  auto pso=[&](NSString* n){ return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e]; };
  id<MTLComputePipelineState> psInit=pso(@"db_init"), psProp=pso(@"db_propagate"), psCompress=pso(@"db_compress"), psBbox=pso(@"db_bbox");

  // --- det -> prob map (reuse the bit-accurate MPSGraph det) ---
  NSData* jd=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* G=[NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
  NSData* W=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"weights.bin"]];
  cv::Mat orig=cv::imread("tests/fixtures/images/png/receipt.png", cv::IMREAD_COLOR);
  std::vector<float> detBuf(3*640*640);
  { cv::Mat di; cv::resize(orig,di,cv::Size(640,640)); cv::Mat bgr[3]; cv::split(di,bgr);
    const float m[3]={0.485f,0.456f,0.406f},sd[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++){ cv::Mat pp(640,640,CV_32F,detBuf.data()+(size_t)c*640*640); bgr[c].convertTo(pp,CV_32F,1.0/(255.0*sd[c]),-m[c]/sd[c]); } }
  NSData* xin=[NSData dataWithBytesNoCopy:detBuf.data() length:detBuf.size()*4 freeWhenDone:NO];
  MPSGraph* g=[MPSGraph new]; RecIO io=buildRecGraph(g,G,(const float*)W.bytes,1);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  id<MTLBuffer> probBuf=[dev newBufferWithLength:(size_t)SZ*SZ*4 options:MTLResourceStorageModeShared];
  MPSGraphTensorData* xTD=[[MPSGraphTensorData alloc] initWithDevice:gdev data:xin shape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* probTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:probBuf shape:@[@1,@1,@(SZ),@(SZ)] dataType:MPSDataTypeFloat32];
  MPSGraphExecutable* exe=[g compileWithDevice:gdev feeds:@{io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[io.output] targetOperations:nil compilationDescriptor:nil];
  { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; [exe encodeToCommandBuffer:cb inputsArray:@[xTD] resultsArray:@[probTD] executionDescriptor:nil]; [cb.rootCommandBuffer commit];[cb.rootCommandBuffer waitUntilCompleted]; }

  // --- GPU CCL ---
  const size_t NPIX=(size_t)SZ*SZ;
  id<MTLBuffer> labelBuf=[dev newBufferWithLength:NPIX*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> changed=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> xmin=[dev newBufferWithLength:NPIX*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> ymin=[dev newBufferWithLength:NPIX*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> xmax=[dev newBufferWithLength:NPIX*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> ymax=[dev newBufferWithLength:NPIX*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> cnt =[dev newBufferWithLength:NPIX*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> psum=[dev newBufferWithLength:NPIX*4 options:MTLResourceStorageModeShared];
  uint32_t dims[2]={SZ,SZ}; id<MTLBuffer> dimBuf=[dev newBufferWithBytes:dims length:8 options:MTLResourceStorageModeShared];
  float th=THRESH; id<MTLBuffer> thBuf=[dev newBufferWithBytes:&th length:4 options:MTLResourceStorageModeShared];

  auto disp=[&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps){
    [enc setComputePipelineState:ps]; [enc dispatchThreads:MTLSizeMake(SZ,SZ,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)]; };
  auto disp1D=[&](id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps){
    [enc setComputePipelineState:ps]; [enc dispatchThreads:MTLSizeMake(SZ,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; };

  // Pre-clear bbox arrays on host BEFORE commit (GPU reads them after commit).
  { uint32_t big=0x7fffffff; memset_pattern4(xmin.contents,&big,NPIX*4); memset_pattern4(ymin.contents,&big,NPIX*4);
    memset(xmax.contents,0,NPIX*4); memset(ymax.contents,0,NPIX*4); memset(cnt.contents,0,NPIX*4); memset(psum.contents,0,NPIX*4); }
  const int K = argc>2? atoi(argv[2]) : 96;  // propagate+compress passes, NO host sync
  int iters=K; double gpu_ms=0, wall=0; auto t0=clk::now();
  // WHOLE CCL in ONE command buffer: init -> K*(propagate,compress) -> bbox. One sync.
  MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
  { id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setBuffer:probBuf offset:0 atIndex:0]; [e setBuffer:labelBuf offset:0 atIndex:1]; [e setBuffer:dimBuf offset:0 atIndex:2]; [e setBuffer:thBuf offset:0 atIndex:3];
    disp(e,psInit); [e endEncoding]; }
  for(int k=0;k<K;k++){
    { id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder]; [e setBuffer:labelBuf offset:0 atIndex:0]; [e setBuffer:dimBuf offset:0 atIndex:1]; [e setBuffer:changed offset:0 atIndex:2]; disp(e,psProp); [e endEncoding]; }
    { id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder]; [e setBuffer:labelBuf offset:0 atIndex:0]; [e setBuffer:dimBuf offset:0 atIndex:1]; [e setBuffer:changed offset:0 atIndex:2]; disp(e,psCompress); [e endEncoding]; }
  }
  { id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
    [e setBuffer:labelBuf offset:0 atIndex:0]; [e setBuffer:dimBuf offset:0 atIndex:1];
    [e setBuffer:xmin offset:0 atIndex:2]; [e setBuffer:ymin offset:0 atIndex:3]; [e setBuffer:xmax offset:0 atIndex:4]; [e setBuffer:ymax offset:0 atIndex:5]; [e setBuffer:cnt offset:0 atIndex:6]; [e setBuffer:probBuf offset:0 atIndex:7]; [e setBuffer:psum offset:0 atIndex:8];
    disp(e,psBbox); [e endEncoding]; }
  [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
  gpu_ms=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0; wall=ms(t0);

  // compact GPU components (host, for validation only)
  const uint32_t* CX0=(const uint32_t*)xmin.contents; const uint32_t* CY0=(const uint32_t*)ymin.contents;
  const uint32_t* CX1=(const uint32_t*)xmax.contents; const uint32_t* CY1=(const uint32_t*)ymax.contents; const uint32_t* CN=(const uint32_t*)cnt.contents;
  int gpu_comps=0; for(size_t r=0;r<NPIX;r++){ if(CN[r]==0)continue; int w=CX1[r]-CX0[r]+1,h=CY1[r]-CY0[r]+1; if(w*h>=MINAREA)gpu_comps++; }

  // --- OpenCV CCL on the same thresholded map (4-connectivity) ---
  cv::Mat pred(SZ,SZ,CV_32F,probBuf.contents), bitmap; cv::threshold(pred,bitmap,THRESH,255,cv::THRESH_BINARY); bitmap.convertTo(bitmap,CV_8U);
  cv::Mat labels,stats,centroids; auto tcv=clk::now();
  int n=cv::connectedComponentsWithStats(bitmap,labels,stats,centroids,4);
  double cv_ms=ms(tcv);
  int cv_comps=0; for(int i=1;i<n;i++){ int a=stats.at<int>(i,cv::CC_STAT_AREA); int w=stats.at<int>(i,cv::CC_STAT_WIDTH),h=stats.at<int>(i,cv::CC_STAT_HEIGHT); if(w*h>=MINAREA && a>0)cv_comps++; }

  const uint32_t* PS=(const uint32_t*)psum.contents;
  int gpu_filt=0; for(size_t r=0;r<NPIX;r++){ if(CN[r]==0)continue; int w=CX1[r]-CX0[r]+1,h=CY1[r]-CY0[r]+1; float mean=(float)PS[r]/1024.0f/(float)CN[r]; int side=std::min(w,h); if(mean>0.4f && side>=3) gpu_filt++; }
  std::printf("det map %dx%d  thresh=%.2f  minArea=%d\n", SZ,SZ,THRESH,MINAREA);
  std::printf("GPU score+size filtered (mean>0.4, side>=3): %d boxes  (host DB-post gives ~44)\n", gpu_filt);
  std::printf("GPU CCL: %d components  (K=%d passes, GPU-exec=%.3f ms, wall=%.2f ms, ONE command buffer)\n", gpu_comps, iters, gpu_ms, wall);
  std::printf("OpenCV : %d components  (4-conn, %.2f ms)\n", cv_comps, cv_ms);
  std::printf("%s\n", gpu_comps==cv_comps? "MATCH ✓" : "MISMATCH — investigate");
  return 0;
}}
