// Full end-to-end Apple GPU text OCR pipeline (M3 Max), device-resident where it
// matters: MPSGraph detection (DBNet) -> host DB post-process (boxes) -> Metal
// warp kernel (crops on GPU) -> MPSGraph recognition (CRNN) -> host CTC decode.
// Reuses the REAL host code from turbo_ocr_common (extract_boxes_from_bitmap,
// compute_crop_transform, ctc_greedy_decode_raw) so output matches the pipeline.
//
// Build (from the repo root; there is no build_mps_ocr.sh — do NOT use
// `pkg-config --libs opencv4`, it expands to a glued list that breaks the link):
//   clang++ -std=c++20 -ObjC++ -fobjc-arc -O2 -Iinclude -Itools/probes/apple \
//     tools/probes/apple/mps_ocr.mm build-cpu/libturbo_ocr_common.a \
//     -I/opt/homebrew/opt/opencv/include/opencv4 -L/opt/homebrew/opt/opencv/lib \
//     -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
//     -framework Metal -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph -framework Foundation \
//     -o build-cpu/mps_ocr
// The same line builds every other tools/probes/apple/mps_*.mm probe. `mps_rec_build.h`
// below resolves through tools/probes/apple/mps_rec_build.h, a forwarding header to the real
// translator at src/backends/apple/engine/mps_rec_build.h.
// Run:   ./build-cpu/mps_ocr <image> <det_export_dir> <rec_export_dir> <keys.txt>

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
#include <algorithm>
#include <numeric>
#include <functional>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
using turbo_ocr::Box;

// Build an MPSGraph from an export dir; returns graph + io.
struct Model { MPSGraph* g; RecIO io; NSData* weights; };
static Model loadModel(NSString* dir, int B){
  NSData* jd=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* G=[NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
  NSData* W=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights.bin"]];
  MPSGraph* g=[MPSGraph new];
  RecIO io=buildRecGraph(g, G, (const float*)W.bytes, B);
  return {g, io, W};
}

int main(int argc, char** argv){
 @autoreleasepool{
  const char* imgPath = argc>1?argv[1]:"tests/fixtures/images/png/receipt.png";
  NSString* detDir=[NSString stringWithUTF8String: argc>2?argv[2]:"det_export"];
  NSString* recDir=[NSString stringWithUTF8String: argc>3?argv[3]:"rec_export"];
  const char* keys = argc>4?argv[4]:"models/keys_tiny.txt";
  const int DETSZ=640, RECH=48, RECW=320;

  cv::Mat orig=cv::imread(imgPath, cv::IMREAD_COLOR);  // BGR
  if(orig.empty()){ std::fprintf(stderr,"cannot read %s\n",imgPath); return 1; }
  const int oh=orig.rows, ow=orig.cols;
  std::printf("image %s (%dx%d)\n", imgPath, ow, oh);

  // labels: {blank} + dict (matches OrtPaddleRec)
  std::vector<std::string> labels={"blank"};
  turbo_ocr::recognition::load_label_dict(keys, labels);

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q=[dev newCommandQueue];

  // ---- warp pipeline + source texture (original image, RGB) ----
  NSError* e=nil;
  id<MTLLibrary> lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/warp.metallib"] error:&e];
  id<MTLComputePipelineState> warp=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"warp_crops"] error:&e];
  MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:ow height:oh mipmapped:NO];
  td.usage=MTLTextureUsageShaderRead;
  id<MTLTexture> srcTex=[dev newTextureWithDescriptor:td];
  { std::vector<uint8_t> px((size_t)ow*oh*4);
    for(int y=0;y<oh;y++){ const cv::Vec3b* row=orig.ptr<cv::Vec3b>(y);
      for(int x=0;x<ow;x++){ size_t i=((size_t)y*ow+x)*4; px[i]=row[x][2]; px[i+1]=row[x][1]; px[i+2]=row[x][0]; px[i+3]=255; } }
    [srcTex replaceRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0 withBytes:px.data() bytesPerRow:ow*4]; }

  // ---- det model (batch 1) + det preprocess buffer ----
  Model det=loadModel(detDir, 1);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  cv::Mat detIn; cv::resize(orig, detIn, cv::Size(DETSZ,DETSZ));
  std::vector<float> detBuf(3*DETSZ*DETSZ);
  { cv::Mat bgr[3]; cv::split(detIn,bgr);
    const float m[3]={0.485f,0.456f,0.406f}, s[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++){ cv::Mat p(DETSZ,DETSZ,CV_32F, detBuf.data()+(size_t)c*DETSZ*DETSZ);
      bgr[c].convertTo(p, CV_32F, 1.0/(255.0*s[c]), -m[c]/s[c]); } }
  NSData* detData=[NSData dataWithBytesNoCopy:detBuf.data() length:detBuf.size()*4 freeWhenDone:NO];
  MPSGraphTensorData* detTD=[[MPSGraphTensorData alloc] initWithDevice:gdev data:detData shape:mrb_nums(det.io.ishape) dataType:MPSDataTypeFloat32];

  // Compile to an executable + resident output MTLBuffer so we control the
  // command buffer and can read its GPU-execution time (GPUStart/EndTime).
  MPSGraphExecutable* detExe=[det.g compileWithDevice:gdev
      feeds:@{det.io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(det.io.ishape) dataType:MPSDataTypeFloat32]}
      targetTensors:@[det.io.output] targetOperations:nil compilationDescriptor:nil];
  id<MTLBuffer> detOutBuf=[dev newBufferWithLength:(size_t)DETSZ*DETSZ*4 options:MTLResourceStorageModeShared];
  MPSGraphTensorData* detOutTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:detOutBuf shape:@[@1,@1,@(DETSZ),@(DETSZ)] dataType:MPSDataTypeFloat32];
  double g_det_gpu=0, g_det_d2h=0;
  auto runDet=[&](std::vector<float>& outMap){
    MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[detOutTD] executionDescriptor:nil];
    [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
    g_det_gpu=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0;
    auto s=clk::now(); outMap.resize((size_t)DETSZ*DETSZ); memcpy(outMap.data(), detOutBuf.contents, (size_t)DETSZ*DETSZ*4); g_det_d2h=ms_since(s);
  };

  // DB post-process (real host code)
  std::vector<cv::Point> shifted_buf; cv::Mat mask_buf;
  std::vector<std::vector<cv::Point>> contours_buf; std::vector<cv::Vec4i> hier_buf;
  auto detPost=[&](const std::vector<float>& map)->std::vector<Box>{
    cv::Mat pred(DETSZ,DETSZ,CV_32F,(void*)map.data());
    cv::Mat bitmap; cv::threshold(pred, bitmap, 0.3, 255, cv::THRESH_BINARY); bitmap.convertTo(bitmap,CV_8U);
    return turbo_ocr::detection::extract_boxes_from_bitmap(pred, bitmap, oh, ow, DETSZ, DETSZ,
              0.40f, 1.5f, 3.0f, 2.0f, shifted_buf, mask_buf, contours_buf, hier_buf);
  };

  // ---- one det to get box count, build rec graph at that batch ----
  std::vector<float> detMap; runDet(detMap);
  std::vector<Box> boxes=detPost(detMap);
  int B=(int)boxes.size();
  std::printf("detection: %d boxes\n", B);
  if(B==0){ std::printf("no boxes\n"); return 0; }

  Model rec=loadModel(recDir, B);
  // GPU argmax+max over the class axis -> tiny [B,T,1] tensors, so only the token
  // indices + scores cross to host (14 KB) instead of the full [B,T,6906] logits (48 MB).
  MPSGraphTensor* idxT=[rec.g reductionArgMaximumWithTensor:rec.io.output axis:2 name:nil];
  MPSGraphTensor* maxT=[rec.g reductionMaximumWithTensor:rec.io.output axis:2 name:nil];
  // homographies + content widths for the B boxes
  std::vector<float> Hm((size_t)B*9); std::vector<int> cw(B);
  for(int i=0;i<B;i++){ auto ct=turbo_ocr::compute_crop_transform(boxes[i], RECH, RECW);
    for(int k=0;k<9;k++)Hm[i*9+k]=ct.M_inv[k]; cw[i]=std::min(ct.crop_width, RECW); }
  id<MTLBuffer> Hbuf=[dev newBufferWithBytes:Hm.data() length:Hm.size()*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> cwBuf=[dev newBufferWithBytes:cw.data() length:cw.size()*4 options:MTLResourceStorageModeShared];
  uint32_t dims[4]={(uint32_t)B,3,(uint32_t)RECH,(uint32_t)RECW};
  id<MTLBuffer> dimBuf=[dev newBufferWithBytes:dims length:16 options:MTLResourceStorageModeShared];
  id<MTLBuffer> cropsBuf=[dev newBufferWithLength:(size_t)B*3*RECH*RECW*4 options:MTLResourceStorageModeShared];
  MPSGraphTensorData* cropsTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:cropsBuf shape:mrb_nums(rec.io.ishape) dataType:MPSDataTypeFloat32];
  const long RT=[idxT.shape[1] longValue];  // rec time steps (T)
  MPSGraphExecutable* recExe=[rec.g compileWithDevice:gdev
      feeds:@{rec.io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(rec.io.ishape) dataType:MPSDataTypeFloat32]}
      targetTensors:@[idxT,maxT] targetOperations:nil compilationDescriptor:nil];
  id<MTLBuffer> idxBuf=[dev newBufferWithLength:(size_t)B*RT*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> maxBuf=[dev newBufferWithLength:(size_t)B*RT*4 options:MTLResourceStorageModeShared];
  MPSGraphTensorData* idxOutTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:idxBuf shape:@[@(B),@(RT),@1] dataType:MPSDataTypeInt32];
  MPSGraphTensorData* maxOutTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:maxBuf shape:@[@(B),@(RT),@1] dataType:MPSDataTypeFloat32];

  auto encodeWarp=[&](id<MTLCommandBuffer> cb){
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:warp]; [enc setTexture:srcTex atIndex:0];
    [enc setBuffer:cropsBuf offset:0 atIndex:0]; [enc setBuffer:Hbuf offset:0 atIndex:1];
    [enc setBuffer:dimBuf offset:0 atIndex:2]; [enc setBuffer:cwBuf offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(RECW,RECH,B) threadsPerThreadgroup:MTLSizeMake(16,8,1)];
    [enc endEncoding];
  };
  double g_warprec_gpu=0, g_ctc=0;
  auto runRec=[&](std::vector<std::pair<std::string,float>>& out){
    // FUSED: warp compute kernel + rec+argmax executable in ONE command buffer.
    MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    encodeWarp(cb);
    [recExe encodeToCommandBuffer:cb inputsArray:@[cropsTD] resultsArray:@[idxOutTD,maxOutTD] executionDescriptor:nil];
    [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
    g_warprec_gpu=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0;
    auto s=clk::now();
    const int32_t* idx=(const int32_t*)idxBuf.contents; const float* sc=(const float*)maxBuf.contents;
    out.resize(B);
    for(int i=0;i<B;i++) out[i]=turbo_ocr::recognition::ctc_greedy_decode((const int*)(idx+(size_t)i*RT), sc+(size_t)i*RT, (int)RT, labels);
    g_ctc=ms_since(s);
  };

  // ---- run once, print recognized text ----
  std::vector<std::pair<std::string,float>> texts; runRec(texts);
  std::printf("\n--- recognized text (%d lines) ---\n", B);
  int shown=0; for(auto& t:texts){ if(!t.first.empty() && t.second>0.5f){ std::printf("  [%.2f] %s\n", t.second, t.first.c_str()); if(++shown>=12)break; } }

  // ---- benchmark per stage ----
  // det preprocess as a per-image host cost (resize + normalize into detBuf)
  auto detPre=[&](){ cv::Mat di; cv::resize(orig,di,cv::Size(DETSZ,DETSZ)); cv::Mat bgr[3]; cv::split(di,bgr);
    const float m[3]={0.485f,0.456f,0.406f}, s[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++){ cv::Mat p(DETSZ,DETSZ,CV_32F, detBuf.data()+(size_t)c*DETSZ*DETSZ); bgr[c].convertTo(p,CV_32F,1.0/(255.0*s[c]),-m[c]/s[c]); } };

  const int N=30;
  double a_pre=0,a_detgpu=0,a_detd2h=0,a_dbpost=0,a_warp=0,a_ctc=0; std::vector<double> wall;
  std::vector<float> mm; std::vector<Box> bb; std::vector<std::pair<std::string,float>> tt;
  for(int it=-3; it<N; ++it){
    auto t0=clk::now();
    auto s=clk::now(); detPre(); double t_pre=ms_since(s);
    runDet(mm);                             // sets g_det_gpu, g_det_d2h
    s=clk::now(); bb=detPost(mm); double t_db=ms_since(s);
    runRec(tt);                             // sets g_warprec_gpu, g_ctc
    double t_total=ms_since(t0);
    if(it>=0){ a_pre+=t_pre; a_detgpu+=g_det_gpu; a_detd2h+=g_det_d2h; a_dbpost+=t_db; a_warp+=g_warprec_gpu; a_ctc+=g_ctc; wall.push_back(t_total); }
  }
  std::sort(wall.begin(),wall.end());
  auto A=[&](double x){ return x/N; };
  double gpu=A(a_detgpu)+A(a_warp), host=A(a_pre)+A(a_detd2h)+A(a_dbpost)+A(a_ctc), wallmed=wall[wall.size()/2];
  std::printf("\n=== PROFILE (avg of %d iters, %d boxes) ===\n", N, B);
  std::printf("  det preprocess (host)      %.2f ms\n", A(a_pre));
  std::printf("  det GPU-exec               %.2f ms\n", A(a_detgpu));
  std::printf("  det prob-map D2H (host)    %.2f ms\n", A(a_detd2h));
  std::printf("  DB post-process (host)     %.2f ms\n", A(a_dbpost));
  std::printf("  warp+rec+argmax GPU-exec   %.2f ms  (FUSED one cmd buffer)\n", A(a_warp));
  std::printf("  CTC decode (host)          %.2f ms\n", A(a_ctc));
  std::printf("  ---------------------------------------\n");
  std::printf("  GPU-exec total             %.2f ms\n", gpu);
  std::printf("  host-work total            %.2f ms\n", host);
  std::printf("  submit/sync overhead       %.2f ms\n", wallmed-gpu-host);
  std::printf("  WALL total (median)        %.2f ms  => %.1f img/s\n", wallmed, 1000.0/wallmed);
  std::printf("  GPU utilization            %.0f%%  (GPU-exec / wall)\n", 100.0*gpu/wallmed);
  return 0;
 }
}
