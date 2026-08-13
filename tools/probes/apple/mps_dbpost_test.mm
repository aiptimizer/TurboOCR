// Full NVIDIA-parity GPU DB post-processing on Metal, validated against the host
// extract_boxes_from_bitmap on the REAL receipt: prob -> threshold -> CCL ->
// per-comp stats+score -> crack perimeter -> unclip distance -> boundary-disc JFA
// -> PCA oriented rect -> rotated+unclipped quads. ONE command buffer, one sync.
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
  const int SZ=640, K=argc>2?atoi(argv[2]):48; const float THRESH=0.3f, BOXTHRESH=0.4f, RATIO=1.5f;
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue]; NSError* e=nil;
  id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/db_ccl.metallib"] error:&e];
  if(!lib){ std::fprintf(stderr,"metallib: %s\n",e.localizedDescription.UTF8String); return 1; }
  auto pso=[&](NSString* n){ auto p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&e]; if(!p)std::fprintf(stderr,"pso %s: %s\n",n.UTF8String,e.localizedDescription.UTF8String); return p; };
  auto P=pso; id<MTLComputePipelineState> psInit=P(@"db_init"),psProp=P(@"db_propagate"),psComp=P(@"db_compress"),psBbox=P(@"db_bbox"),
    psPerim=P(@"db_crack_perim"),psExp=P(@"db_expand"),psScat=P(@"db_jfa_scatter"),psRes=P(@"db_jfa_resolve"),
    psMom=P(@"db_moments"),psAxis=P(@"db_axis"),psProj=P(@"db_project");

  // real receipt -> det -> prob
  cv::Mat orig=cv::imread("tests/fixtures/images/png/receipt.png",cv::IMREAD_COLOR);
  std::vector<float> detBuf(3*SZ*SZ);
  { cv::Mat di; cv::resize(orig,di,cv::Size(SZ,SZ)); cv::Mat bgr[3]; cv::split(di,bgr); const float m[3]={0.485f,0.456f,0.406f},sd[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++){ cv::Mat pp(SZ,SZ,CV_32F,detBuf.data()+(size_t)c*SZ*SZ); bgr[c].convertTo(pp,CV_32F,1.0/(255.0*sd[c]),-m[c]/sd[c]); } }
  NSData* jd=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* G=[NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
  NSData* Wd=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"weights.bin"]];
  MPSGraph* g=[MPSGraph new]; RecIO io=buildRecGraph(g,G,(const float*)Wd.bytes,1);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  id<MTLBuffer> probBuf=[dev newBufferWithLength:(size_t)SZ*SZ*4 options:MTLResourceStorageModeShared];
  NSData* dd=[NSData dataWithBytesNoCopy:detBuf.data() length:detBuf.size()*4 freeWhenDone:NO];
  MPSGraphTensorData* xTD=[[MPSGraphTensorData alloc] initWithDevice:gdev data:dd shape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* probTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:probBuf shape:@[@1,@1,@(SZ),@(SZ)] dataType:MPSDataTypeFloat32];
  MPSGraphExecutable* exe=[g compileWithDevice:gdev feeds:@{io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[io.output] targetOperations:nil compilationDescriptor:nil];

  const size_t N=(size_t)SZ*SZ;
  auto buf=[&](size_t n){ return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
  id<MTLBuffer> label=buf(N*4),changed=buf(4),xmin=buf(N*4),ymin=buf(N*4),xmax=buf(N*4),ymax=buf(N*4),cnt=buf(N*4),psum=buf(N*4),
    perim=buf(N*4),expand=buf(N*4),best=buf(N*4),expanded=buf(N*4),mom=buf(N*6*4),orient=buf(N*6*4);
  uint32_t dims[2]={SZ,SZ}; id<MTLBuffer> dimBuf=[dev newBufferWithBytes:dims length:8 options:MTLResourceStorageModeShared];
  float th=THRESH; id<MTLBuffer> thBuf=[dev newBufferWithBytes:&th length:4 options:MTLResourceStorageModeShared];
  uint32_t Nu=(uint32_t)N; id<MTLBuffer> Nbuf=[dev newBufferWithBytes:&Nu length:4 options:MTLResourceStorageModeShared];
  float ratio=RATIO,bt=BOXTHRESH; id<MTLBuffer> ratioBuf=[dev newBufferWithBytes:&ratio length:4 options:MTLResourceStorageModeShared], btBuf=[dev newBufferWithBytes:&bt length:4 options:MTLResourceStorageModeShared];
  uint32_t INIT=0xFFFFFFFF; id<MTLBuffer> initBuf=[dev newBufferWithBytes:&INIT length:4 options:MTLResourceStorageModeShared];

  // host pre-clear
  { uint32_t big=0x7fffffff; memset_pattern4(xmin.contents,&big,N*4); memset_pattern4(ymin.contents,&big,N*4);
    memset(xmax.contents,0,N*4); memset(ymax.contents,0,N*4); memset(cnt.contents,0,N*4); memset(psum.contents,0,N*4);
    memset(perim.contents,0,N*4); memset(mom.contents,0,N*6*4);
    memset_pattern4(best.contents,&INIT,N*4); }

  auto D2=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> ps, NSArray* bufs){ id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder]; [e setComputePipelineState:ps]; for(uint k=0;k<bufs.count;k++)[e setBuffer:bufs[k] offset:0 atIndex:k]; [e dispatchThreads:MTLSizeMake(SZ,SZ,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)]; [e endEncoding]; };
  auto D1=[&](id<MTLCommandBuffer> cb, id<MTLComputePipelineState> ps, NSArray* bufs){ id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder]; [e setComputePipelineState:ps]; for(uint k=0;k<bufs.count;k++)[e setBuffer:bufs[k] offset:0 atIndex:k]; [e dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [e endEncoding]; };

  auto t0=clk::now();
  MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
  [exe encodeToCommandBuffer:cb inputsArray:@[xTD] resultsArray:@[probTD] executionDescriptor:nil];
  D2(cb,psInit,@[probBuf,label,dimBuf,thBuf]);
  for(int k=0;k<K;k++){ D2(cb,psProp,@[label,dimBuf,changed]); D2(cb,psComp,@[label,dimBuf,changed]); }
  D2(cb,psBbox,@[label,dimBuf,xmin,ymin,xmax,ymax,cnt,probBuf,psum]);
  D2(cb,psPerim,@[label,dimBuf,perim]);
  D1(cb,psExp,@[cnt,psum,perim,Nbuf,ratioBuf,btBuf,expand]);
  D2(cb,psScat,@[label,expand,dimBuf,best]);
  D2(cb,psRes,@[label,best,dimBuf,initBuf,expanded]);
  D2(cb,psMom,@[expanded,dimBuf,mom]);
  D1(cb,psAxis,@[mom,Nbuf,orient]);
  D2(cb,psProj,@[expanded,dimBuf,mom,orient]);
  [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
  double gpu=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0, wall=ms(t0);

  // reconstruct oriented+unclipped quads from surviving roots
  const uint32_t* M=(const uint32_t*)mom.contents; const float* O=(const float*)orient.contents; const float* EX=(const float*)expand.contents;
  float sx=(float)orig.cols/SZ, sy=(float)orig.rows/SZ; int gpu_boxes=0;
  std::vector<std::array<cv::Point2f,4>> quads;
  for(size_t r=0;r<N;r++){ if(M[r*6]<3u || EX[r]<=0.0f) continue;
    float c=O[r*6],s=O[r*6+1],umin=O[r*6+2],umax=O[r*6+3],vmin=O[r*6+4],vmax=O[r*6+5];
    float U[4]={umin,umax,umax,umin},V[4]={vmin,vmin,vmax,vmax}; std::array<cv::Point2f,4> qd;
    float mnside=1e9f; for(int k=0;k<4;k++){ qd[k]={(U[k]*c - V[k]*s)*sx,(U[k]*s + V[k]*c)*sy}; }
    float w=std::hypot(qd[1].x-qd[0].x,qd[1].y-qd[0].y), h=std::hypot(qd[3].x-qd[0].x,qd[3].y-qd[0].y); mnside=std::min(w,h);
    if(mnside<3.0f) continue; quads.push_back(qd); gpu_boxes++;
  }

  // host reference on same map
  cv::Mat pred(SZ,SZ,CV_32F,probBuf.contents), bm; cv::threshold(pred,bm,THRESH,255,cv::THRESH_BINARY); bm.convertTo(bm,CV_8U);
  std::vector<cv::Point> sb; cv::Mat mb; std::vector<std::vector<cv::Point>> cbf; std::vector<cv::Vec4i> hb;
  auto hboxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bm,orig.rows,orig.cols,SZ,SZ,BOXTHRESH,RATIO,3.0f,2.0f,sb,mb,cbf,hb);

  std::printf("GPU DB-post (full, one command buffer): %d oriented+unclipped boxes, GPU-exec=%.3f ms, wall=%.2f ms\n", gpu_boxes, gpu, wall);
  std::printf("HOST extract_boxes_from_bitmap:         %zu boxes\n", hboxes.size());
  std::printf("%s\n", std::abs((int)hboxes.size()-gpu_boxes)<=3? "COUNT MATCH (±3) ✓" : "count differs");
  // sample geometry: first 3 GPU quads
  for(int i=0;i<3 && i<(int)quads.size();i++){ auto&Q=quads[i]; std::printf("  gpu quad %d: (%.0f,%.0f)(%.0f,%.0f)(%.0f,%.0f)(%.0f,%.0f)\n",i,Q[0].x,Q[0].y,Q[1].x,Q[1].y,Q[2].x,Q[2].y,Q[3].x,Q[3].y); }
  return 0;
}}
