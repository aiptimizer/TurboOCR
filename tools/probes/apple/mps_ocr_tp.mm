// Throughput harness: run N independent Apple OCR pipelines concurrently so the
// CPU encodes/host-processes image i+1 while the GPU runs image i — hiding the
// per-image submit/sync overhead the profiler exposed (single-stream is 8% GPU
// utilization). Measures aggregate img/s vs the single-stream ~100 img/s.
//
// Build: same as mps_ocr (see report) + this file.
// Run:   ./build-cpu/mps_ocr_tp <image> <det_export> <rec_export> <keys> [threads] [secs]

#import <Metal/Metal.h>
#include "mps_rec_build.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "turbo_ocr/analysis/detection/det_postprocess.h"
#include "turbo_ocr/analysis/recognition/ctc_decode.h"
#include "turbo_ocr/base/geometry/perspective.h"
#include "turbo_ocr/base/geometry/box.h"
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
using clk=std::chrono::steady_clock;
using turbo_ocr::Box;

// One self-contained pipeline (its own graphs, buffers, queue) so N can run in parallel.
struct Pipe {
  id<MTLDevice> dev; id<MTLCommandQueue> q; MPSGraphDevice* gdev;
  id<MTLComputePipelineState> warp; id<MTLTexture> srcTex;
  int DETSZ, RECH, RECW, ow, oh, B; long RT;
  std::vector<float> detBuf;
  MPSGraphExecutable *detExe, *recExe;
  id<MTLBuffer> detOutBuf, Hbuf, cwBuf, dimBuf, cropsBuf, idxBuf, maxBuf;
  MPSGraphTensorData *detTD, *detOutTD, *cropsTD, *idxOutTD, *maxOutTD;
  const std::vector<std::string>* labels;
  cv::Mat orig;

  void detPre(){ cv::Mat di; cv::resize(orig,di,cv::Size(DETSZ,DETSZ)); cv::Mat bgr[3]; cv::split(di,bgr);
    const float m[3]={0.485f,0.456f,0.406f}, s[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++){ cv::Mat p(DETSZ,DETSZ,CV_32F,detBuf.data()+(size_t)c*DETSZ*DETSZ); bgr[c].convertTo(p,CV_32F,1.0/(255.0*s[c]),-m[c]/s[c]); } }

  void runOnce(){  // full pipeline once (det -> host db-post -> warp+rec -> ctc)
    detPre();
    { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[detOutTD] executionDescriptor:nil];
      [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
    cv::Mat pred(DETSZ,DETSZ,CV_32F,detOutBuf.contents), bitmap;
    cv::threshold(pred,bitmap,0.3,255,cv::THRESH_BINARY); bitmap.convertTo(bitmap,CV_8U);
    std::vector<cv::Point> sb; cv::Mat mb; std::vector<std::vector<cv::Point>> cb2; std::vector<cv::Vec4i> hb;
    auto boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bitmap,oh,ow,DETSZ,DETSZ,0.40f,1.5f,3.0f,2.0f,sb,mb,cb2,hb);
    // (homographies precomputed for the fixed image; warp+rec fused)
    { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
      [e setComputePipelineState:warp]; [e setTexture:srcTex atIndex:0];
      [e setBuffer:cropsBuf offset:0 atIndex:0]; [e setBuffer:Hbuf offset:0 atIndex:1];
      [e setBuffer:dimBuf offset:0 atIndex:2]; [e setBuffer:cwBuf offset:0 atIndex:3];
      [e dispatchThreads:MTLSizeMake(RECW,RECH,B) threadsPerThreadgroup:MTLSizeMake(16,8,1)]; [e endEncoding];
      [recExe encodeToCommandBuffer:cb inputsArray:@[cropsTD] resultsArray:@[idxOutTD,maxOutTD] executionDescriptor:nil];
      [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
    const int32_t* idx=(const int32_t*)idxBuf.contents; const float* sc=(const float*)maxBuf.contents;
    for(int i=0;i<B;i++) (void)turbo_ocr::recognition::ctc_greedy_decode(idx+(size_t)i*RT, sc+(size_t)i*RT, (int)RT, *labels);
  }
};

static void buildPipe(Pipe& P, id<MTLDevice> dev, NSString* detDir, NSString* recDir,
                      const cv::Mat& orig, const std::vector<Box>& boxes,
                      const std::vector<std::string>& labels, id<MTLComputePipelineState> warp, id<MTLTexture> srcTex){
  P.dev=dev; P.q=[dev newCommandQueue]; P.gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  P.warp=warp; P.srcTex=srcTex; P.DETSZ=640; P.RECH=48; P.RECW=320; P.orig=orig; P.labels=&labels;
  P.ow=orig.cols; P.oh=orig.rows; P.B=(int)boxes.size();
  // det model
  NSData* dj=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* dG=[NSJSONSerialization JSONObjectWithData:dj options:0 error:nil];
  NSData* dW=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"weights.bin"]];
  MPSGraph* dg=[MPSGraph new]; RecIO dio=buildRecGraph(dg,dG,(const float*)dW.bytes,1);
  P.detExe=[dg compileWithDevice:P.gdev feeds:@{dio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[dio.output] targetOperations:nil compilationDescriptor:nil];
  P.detBuf.resize(3*640*640);
  NSData* dd=[NSData dataWithBytesNoCopy:P.detBuf.data() length:P.detBuf.size()*4 freeWhenDone:NO];
  P.detTD=[[MPSGraphTensorData alloc] initWithDevice:P.gdev data:dd shape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32];
  P.detOutBuf=[dev newBufferWithLength:640*640*4 options:MTLResourceStorageModeShared];
  P.detOutTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:P.detOutBuf shape:@[@1,@1,@640,@640] dataType:MPSDataTypeFloat32];
  // rec model at batch B + argmax head
  NSData* rj=[NSData dataWithContentsOfFile:[recDir stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* rG=[NSJSONSerialization JSONObjectWithData:rj options:0 error:nil];
  NSData* rW=[NSData dataWithContentsOfFile:[recDir stringByAppendingPathComponent:@"weights.bin"]];
  MPSGraph* rg=[MPSGraph new]; RecIO rio=buildRecGraph(rg,rG,(const float*)rW.bytes,P.B);
  MPSGraphTensor* idxT=[rg reductionArgMaximumWithTensor:rio.output axis:2 name:nil];
  MPSGraphTensor* maxT=[rg reductionMaximumWithTensor:rio.output axis:2 name:nil];
  P.RT=[idxT.shape[1] longValue];
  P.recExe=[rg compileWithDevice:P.gdev feeds:@{rio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(rio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[idxT,maxT] targetOperations:nil compilationDescriptor:nil];
  P.cropsBuf=[dev newBufferWithLength:(size_t)P.B*3*48*320*4 options:MTLResourceStorageModeShared];
  P.cropsTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:P.cropsBuf shape:mrb_nums(rio.ishape) dataType:MPSDataTypeFloat32];
  P.idxBuf=[dev newBufferWithLength:(size_t)P.B*P.RT*4 options:MTLResourceStorageModeShared];
  P.maxBuf=[dev newBufferWithLength:(size_t)P.B*P.RT*4 options:MTLResourceStorageModeShared];
  P.idxOutTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:P.idxBuf shape:@[@(P.B),@(P.RT),@1] dataType:MPSDataTypeInt32];
  P.maxOutTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:P.maxBuf shape:@[@(P.B),@(P.RT),@1] dataType:MPSDataTypeFloat32];
  // homographies + content widths
  std::vector<float> Hm((size_t)P.B*9); std::vector<int> cw(P.B);
  for(int i=0;i<P.B;i++){ auto ct=turbo_ocr::compute_crop_transform(boxes[i],48,320); for(int k=0;k<9;k++)Hm[i*9+k]=ct.M_inv[k]; cw[i]=std::min(ct.crop_width,320); }
  P.Hbuf=[dev newBufferWithBytes:Hm.data() length:Hm.size()*4 options:MTLResourceStorageModeShared];
  P.cwBuf=[dev newBufferWithBytes:cw.data() length:cw.size()*4 options:MTLResourceStorageModeShared];
  uint32_t dims[4]={(uint32_t)P.B,3,48,320}; P.dimBuf=[dev newBufferWithBytes:dims length:16 options:MTLResourceStorageModeShared];
}

int main(int argc, char** argv){ @autoreleasepool{
  const char* imgPath=argc>1?argv[1]:"tests/fixtures/images/png/receipt.png";
  NSString* detDir=[NSString stringWithUTF8String:argv[2]], *recDir=[NSString stringWithUTF8String:argv[3]];
  const char* keys=argv[4]; int threads=argc>5?atoi(argv[5]):8; double secs=argc>6?atof(argv[6]):3.0;
  cv::Mat orig=cv::imread(imgPath,cv::IMREAD_COLOR);
  std::vector<std::string> labels={"blank"}; turbo_ocr::recognition::load_label_dict(keys,labels);
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  // warp pipeline + source texture (shared, read-only)
  NSError* e=nil; id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/warp.metallib"] error:&e];
  id<MTLComputePipelineState> warp=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"warp_crops"] error:&e];
  MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:orig.cols height:orig.rows mipmapped:NO];
  td.usage=MTLTextureUsageShaderRead; id<MTLTexture> srcTex=[dev newTextureWithDescriptor:td];
  { std::vector<uint8_t> px((size_t)orig.cols*orig.rows*4);
    for(int y=0;y<orig.rows;y++){ const cv::Vec3b* r=orig.ptr<cv::Vec3b>(y);
      for(int x=0;x<orig.cols;x++){ size_t i=((size_t)y*orig.cols+x)*4; px[i]=r[x][2]; px[i+1]=r[x][1]; px[i+2]=r[x][0]; px[i+3]=255; } }
    [srcTex replaceRegion:MTLRegionMake2D(0,0,orig.cols,orig.rows) mipmapLevel:0 withBytes:px.data() bytesPerRow:orig.cols*4]; }
  // det once (standalone) to get the box set -> B, then build N full pipes.
  NSData* dj=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* dG=[NSJSONSerialization JSONObjectWithData:dj options:0 error:nil];
  NSData* dW=[NSData dataWithContentsOfFile:[detDir stringByAppendingPathComponent:@"weights.bin"]];
  MPSGraph* dg=[MPSGraph new]; RecIO dio=buildRecGraph(dg,dG,(const float*)dW.bytes,1);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  std::vector<float> detBuf(3*640*640);
  { cv::Mat di; cv::resize(orig,di,cv::Size(640,640)); cv::Mat bgr[3]; cv::split(di,bgr);
    const float m[3]={0.485f,0.456f,0.406f},s[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++){ cv::Mat p(640,640,CV_32F,detBuf.data()+(size_t)c*640*640); bgr[c].convertTo(p,CV_32F,1.0/(255.0*s[c]),-m[c]/s[c]); } }
  NSData* dd=[NSData dataWithBytesNoCopy:detBuf.data() length:detBuf.size()*4 freeWhenDone:NO];
  MPSGraphTensorData* dtd=[[MPSGraphTensorData alloc] initWithDevice:gdev data:dd shape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* dr=[dg runWithMTLCommandQueue:[dev newCommandQueue] feeds:@{dio.input:dtd} targetTensors:@[dio.output] targetOperations:nil][dio.output];
  std::vector<float> pm(640*640); [dr.mpsndarray readBytes:pm.data() strideBytes:nil];
  cv::Mat pred(640,640,CV_32F,pm.data()),bm; cv::threshold(pred,bm,0.3,255,cv::THRESH_BINARY); bm.convertTo(bm,CV_8U);
  std::vector<cv::Point> sb; cv::Mat mb; std::vector<std::vector<cv::Point>> cbf; std::vector<cv::Vec4i> hb;
  auto boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bm,orig.rows,orig.cols,640,640,0.40f,1.5f,3.0f,2.0f,sb,mb,cbf,hb);
  std::printf("image %dx%d, %zu boxes, %d threads, %.1fs\n", orig.cols,orig.rows,boxes.size(),threads,secs);

  std::vector<Pipe> pipes(threads);
  for(int i=0;i<threads;i++) buildPipe(pipes[i],dev,detDir,recDir,orig,boxes,labels,warp,srcTex);

  std::atomic<long> total{0}; std::atomic<bool> go{false}, stop{false};
  std::vector<std::thread> ws;
  for(int i=0;i<threads;i++) ws.emplace_back([&,i]{ while(!go.load()){} while(!stop.load()){ pipes[i].runOnce(); total.fetch_add(1,std::memory_order_relaxed); } });
  auto t0=clk::now(); go.store(true);
  std::this_thread::sleep_for(std::chrono::duration<double>(secs)); stop.store(true);
  for(auto& w:ws) w.join();
  double dt=std::chrono::duration<double>(clk::now()-t0).count(); long n=total.load();
  std::printf("AGGREGATE: %ld imgs in %.2fs => %.0f img/s  (single-stream ~100)\n", n, dt, n/dt);
  return 0;
}}
