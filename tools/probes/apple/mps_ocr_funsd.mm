// FUNSD runner for the Apple GPU pipeline (tiny det+rec in MPSGraph, Metal warp).
// Two detection-post modes selectable at runtime:
//   host : det(GPU) -> extract_boxes_from_bitmap (reference cv geometry) -> warp+rec(GPU)
//   gpu  : det -> 12 Metal DB-post kernels -> emit_quads -> warp+rec, ALL one cmd buffer
// Loops the 50 FUNSD pages, emits recognized words per image as JSON for
// score_funsd.py, and reports GPU-exec throughput + wall throughput.
//
// Run: mps_ocr_funsd <cache_dir> <N> <out.json> <det_export> <rec_export> <keys> [--gpu-dbpost] [--conf T]
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
#include <vector>
#include <string>
#include <algorithm>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
using turbo_ocr::Box;

static std::string jesc(const std::string& s){ std::string o; o.reserve(s.size()+2);
  for(char c: s){ switch(c){ case '"': o+="\\\""; break; case '\\': o+="\\\\"; break;
    case '\n': o+="\\n"; break; case '\r': o+="\\r"; break; case '\t': o+="\\t"; break;
    default: if((unsigned char)c<0x20){char b[8]; std::snprintf(b,sizeof b,"\\u%04x",c); o+=b;} else o+=c; } }
  return o; }

int main(int argc,char** argv){ @autoreleasepool{
  if(argc<7){ std::fprintf(stderr,"usage: %s <cache> <N> <out.json> <det_export> <rec_export> <keys> [--gpu-dbpost] [--conf T]\n",argv[0]); return 2; }
  const std::string cache=argv[1]; const int Nimg=atoi(argv[2]); const std::string outPath=argv[3];
  NSString* detDir=[NSString stringWithUTF8String:argv[4]]; NSString* recDir=[NSString stringWithUTF8String:argv[5]];
  const char* keys=argv[6];
  bool gpuDB=false; float confThr=0.0f; const char* dumpDet=nullptr; const char* dumpCrops=nullptr;
  // DB post-process params default to the TINY tier (det_config.h: thresh 0.2,
  // box_thresh 0.40, unclip 1.4) — NOT the receipt's 0.3/0.4/1.5. Overridable.
  float THRESH=0.2f,BOXTHRESH=0.40f,RATIO=1.4f;
  for(int i=7;i<argc;i++){ if(!strcmp(argv[i],"--gpu-dbpost")) gpuDB=true;
    else if(!strcmp(argv[i],"--conf")&&i+1<argc) confThr=atof(argv[++i]);
    else if(!strcmp(argv[i],"--thresh")&&i+1<argc) THRESH=atof(argv[++i]);
    else if(!strcmp(argv[i],"--boxthresh")&&i+1<argc) BOXTHRESH=atof(argv[++i]);
    else if(!strcmp(argv[i],"--unclip")&&i+1<argc) RATIO=atof(argv[++i]);
    else if(!strcmp(argv[i],"--dump-det")&&i+1<argc) dumpDet=argv[++i];
    else if(!strcmp(argv[i],"--dump-crops")&&i+1<argc) dumpCrops=argv[++i]; }
  const int MAXBOX=512,K=48;

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue]; NSError* e=nil;
  auto libAt=[&](NSString* p){ return [dev newLibraryWithURL:[NSURL fileURLWithPath:p] error:&e]; };
  id<MTLLibrary> dbl=libAt(@"build-cpu/db_ccl.metallib"), wl=libAt(@"build-cpu/warp.metallib");
  auto ps=[&](id<MTLLibrary> L,NSString* n){ auto p=[dev newComputePipelineStateWithFunction:[L newFunctionWithName:n] error:&e]; if(!p)std::fprintf(stderr,"pso %s: %s\n",n.UTF8String,e?e.localizedDescription.UTF8String:"?"); return p; };
  id<MTLComputePipelineState> psInit=ps(dbl,@"db_init"),psProp=ps(dbl,@"db_propagate"),psComp=ps(dbl,@"db_compress"),psBbox=ps(dbl,@"db_bbox"),
    psPerim=ps(dbl,@"db_crack_perim"),psExp=ps(dbl,@"db_expand"),psScat=ps(dbl,@"db_jfa_scatter"),psRes=ps(dbl,@"db_jfa_resolve"),
    psMom=ps(dbl,@"db_moments"),psAxis=ps(dbl,@"db_axis"),psProj=ps(dbl,@"db_project"),psEmit=ps(dbl,@"db_emit_quads"),warp=ps(wl,@"warp_crops");

  std::vector<std::string> labels={"blank"}; turbo_ocr::recognition::load_label_dict(keys,labels);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];

  auto loadG=[&](NSString* dir,int Bb,RecIO& io,MPSGraph** gg){ NSData* j=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"graph.json"]]; NSDictionary* G=[NSJSONSerialization JSONObjectWithData:j options:0 error:nil]; NSData* Wd=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights.bin"]]; MPSGraph* g=[MPSGraph new]; io=buildRecGraph(g,G,(const float*)Wd.bytes,Bb); *gg=g; };
  RecIO dio,rio; MPSGraph *dg,*rg; loadG(detDir,1,dio,&dg); loadG(recDir,MAXBOX,rio,&rg);
  // det input resolution comes from the exported graph ([1,3,H,W]); may be non-square.
  const int SZH=(int)dio.ishape[2], SZW=(int)dio.ishape[3];
  // rec crop size comes from the rec export ([B,3,RECH,RECW]); widen RECW to avoid
  // horizontally compressing long lines (kMaxRecWidth-style aspect preservation).
  const int RECH=(int)rio.ishape[2], RECW=(int)rio.ishape[3];
  if(gpuDB && (SZH!=SZW || (size_t)SZH*SZW>=524288)){ std::fprintf(stderr,"--gpu-dbpost requires square det with H*W<524288 (JFA 19-bit packing); det is %dx%d. Use host DB-post.\n",SZH,SZW); return 2; }
  MPSGraphTensor* idxT=[rg reductionArgMaximumWithTensor:rio.output axis:2 name:nil];
  MPSGraphTensor* maxT=[rg reductionMaximumWithTensor:rio.output axis:2 name:nil]; long RT=[idxT.shape[1] longValue];
  MPSGraphExecutable* detExe=[dg compileWithDevice:gdev feeds:@{dio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[dio.output] targetOperations:nil compilationDescriptor:nil];
  MPSGraphExecutable* recExe=[rg compileWithDevice:gdev feeds:@{rio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(rio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[idxT,maxT] targetOperations:nil compilationDescriptor:nil];

  const size_t N=(size_t)SZH*SZW;
  auto B=[&](size_t n){ return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
  id<MTLBuffer> probBuf=B(N*4),label=B(N*4),changed=B(4),xmin=B(N*4),ymin=B(N*4),xmax=B(N*4),ymax=B(N*4),cnt=B(N*4),psum=B(N*4),
    perim=B(N*4),expand=B(N*4),best=B(N*4),expanded=B(N*4),mom=B(N*6*4),orient=B(N*6*4),
    Hbuf=B(MAXBOX*9*4),cwBuf=B(MAXBOX*4),cropsBuf=B((size_t)MAXBOX*3*RECH*RECW*4),idxBuf=B((size_t)MAXBOX*RT*4),maxBuf=B((size_t)MAXBOX*RT*4),boxCount=B(4);

  // Det input as a LIVE MTLBuffer (not initWithDevice:data:, which snapshots the
  // bytes at init and would feed every image the first one's pixels).
  id<MTLBuffer> detInBuf=B(3*N*4); float* detBuf=(float*)detInBuf.contents;
  MPSGraphTensorData* detTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:detInBuf shape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* probTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:probBuf shape:@[@1,@1,@(SZH),@(SZW)] dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* cropsTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:cropsBuf shape:mrb_nums(rio.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* idxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:idxBuf shape:@[@(MAXBOX),@(RT),@1] dataType:MPSDataTypeInt32];
  MPSGraphTensorData* maxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:maxBuf shape:@[@(MAXBOX),@(RT),@1] dataType:MPSDataTypeFloat32];
  uint32_t dimsHW[2]={(uint32_t)SZW,(uint32_t)SZH}; id<MTLBuffer> dimBuf=[dev newBufferWithBytes:dimsHW length:8 options:0]; // {width,height}
  float th=THRESH,ratio=RATIO,bt=BOXTHRESH; uint32_t Nu=(uint32_t)N,mb=MAXBOX,rhw[2]={(uint32_t)RECH,(uint32_t)RECW},INIT=0xFFFFFFFF,rdims[4]={(uint32_t)MAXBOX,3,(uint32_t)RECH,(uint32_t)RECW};
  id<MTLBuffer> thBuf=[dev newBufferWithBytes:&th length:4 options:0],Nbuf=[dev newBufferWithBytes:&Nu length:4 options:0],ratioBuf=[dev newBufferWithBytes:&ratio length:4 options:0],btBuf=[dev newBufferWithBytes:&bt length:4 options:0],
    initBuf=[dev newBufferWithBytes:&INIT length:4 options:0],mbBuf=[dev newBufferWithBytes:&mb length:4 options:0],rhwBuf=[dev newBufferWithBytes:rhw length:8 options:0],rdimBuf=[dev newBufferWithBytes:rdims length:16 options:0];
  id<MTLBuffer> sclBuf=[dev newBufferWithLength:8 options:0]; // per-image (ow/SZ, oh/SZ)

  auto D2=[&](id<MTLCommandBuffer> cb,id<MTLComputePipelineState> p,NSArray* bufs){id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];[e setComputePipelineState:p];for(uint k=0;k<bufs.count;k++)[e setBuffer:bufs[k] offset:0 atIndex:k];[e dispatchThreads:MTLSizeMake(SZW,SZH,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[e endEncoding];};
  auto D1=[&](id<MTLCommandBuffer> cb,id<MTLComputePipelineState> p,NSArray* bufs,int n){id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];[e setComputePipelineState:p];for(uint k=0;k<bufs.count;k++)[e setBuffer:bufs[k] offset:0 atIndex:k];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];};
  auto clearHost=[&](){ uint32_t big=0x7fffffff; memset_pattern4(xmin.contents,&big,N*4);memset_pattern4(ymin.contents,&big,N*4);
    memset(xmax.contents,0,N*4);memset(ymax.contents,0,N*4);memset(cnt.contents,0,N*4);memset(psum.contents,0,N*4);memset(perim.contents,0,N*4);memset(mom.contents,0,N*6*4);
    memset_pattern4(best.contents,&INIT,N*4); *(uint32_t*)boxCount.contents=0; memset(cwBuf.contents,0,MAXBOX*4); };

  // reusable host DB-post scratch
  std::vector<cv::Point> shifted_buf; cv::Mat mask_buf; std::vector<std::vector<cv::Point>> contours_buf; std::vector<cv::Vec4i> hier_buf;

  std::vector<std::vector<std::string>> allWords(Nimg);
  double sum_gpu=0, sum_wall=0; long total_boxes=0; int truncated=0;

  auto encodeWarpRec=[&](id<MTLCommandBuffer> cb, id<MTLTexture> srcTex){
    id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder]; [ce setComputePipelineState:warp]; [ce setTexture:srcTex atIndex:0];
    [ce setBuffer:cropsBuf offset:0 atIndex:0];[ce setBuffer:Hbuf offset:0 atIndex:1];[ce setBuffer:rdimBuf offset:0 atIndex:2];[ce setBuffer:cwBuf offset:0 atIndex:3];
    [ce dispatchThreads:MTLSizeMake(RECW,RECH,MAXBOX) threadsPerThreadgroup:MTLSizeMake(16,8,1)];[ce endEncoding];
    [recExe encodeToCommandBuffer:cb inputsArray:@[cropsTD] resultsArray:@[idxTD,maxTD] executionDescriptor:nil];
  };
  auto decode=[&](uint count, std::vector<std::string>& words){
    const int32_t* idx=(const int32_t*)idxBuf.contents; const float* sc=(const float*)maxBuf.contents;
    for(uint i=0;i<count;i++){ auto t=turbo_ocr::recognition::ctc_greedy_decode((const int*)(idx+(size_t)i*RT),sc+(size_t)i*RT,(int)RT,labels);
      if(!t.first.empty() && t.second>=confThr) words.push_back(t.first); }
  };

  for(int im=-1; im<Nimg; ++im){                 // -1 = warmup on image 0
    int idx = im<0?0:im;
    char p[512]; std::snprintf(p,sizeof p,"%s/funsd_%03d.png",cache.c_str(),idx);
    cv::Mat orig=cv::imread(p,cv::IMREAD_COLOR); if(orig.empty()){ std::fprintf(stderr,"cannot read %s\n",p); return 1; }
    int ow=orig.cols, oh=orig.rows;
    // src texture (RGB)
    MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:ow height:oh mipmapped:NO]; td.usage=MTLTextureUsageShaderRead;
    id<MTLTexture> srcTex=[dev newTextureWithDescriptor:td];
    { std::vector<uint8_t> px((size_t)ow*oh*4); for(int y=0;y<oh;y++){const cv::Vec3b* r=orig.ptr<cv::Vec3b>(y); for(int x=0;x<ow;x++){size_t i=((size_t)y*ow+x)*4; px[i]=r[x][2];px[i+1]=r[x][1];px[i+2]=r[x][0];px[i+3]=255;}}
      [srcTex replaceRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0 withBytes:px.data() bytesPerRow:ow*4]; }
    // det preprocess into detBuf (stable pointer -> detTD); resize to det graph size (SZW x SZH)
    { cv::Mat di; cv::resize(orig,di,cv::Size(SZW,SZH)); cv::Mat bgr[3]; cv::split(di,bgr); const float m[3]={0.485f,0.456f,0.406f},sd[3]={0.229f,0.224f,0.225f};
      for(int c=0;c<3;c++){cv::Mat pp(SZH,SZW,CV_32F,detBuf+(size_t)c*N); bgr[c].convertTo(pp,CV_32F,1.0/(255.0*sd[c]),-m[c]/sd[c]);} }
    float scl[2]={(float)ow/SZW,(float)oh/SZH}; memcpy(sclBuf.contents,scl,8);

    auto t0=clk::now(); double gpuMs=0; uint count=0;
    std::vector<std::string> words;
    if(gpuDB){
      clearHost();
      MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil];
      D2(cb,psInit,@[probBuf,label,dimBuf,thBuf]);
      for(int k=0;k<K;k++){ D2(cb,psProp,@[label,dimBuf,changed]); D2(cb,psComp,@[label,dimBuf,changed]); }
      D2(cb,psBbox,@[label,dimBuf,xmin,ymin,xmax,ymax,cnt,probBuf,psum]);
      D2(cb,psPerim,@[label,dimBuf,perim]);
      D1(cb,psExp,@[cnt,psum,perim,Nbuf,ratioBuf,btBuf,expand],(int)N);
      D2(cb,psScat,@[label,expand,dimBuf,best]);
      D2(cb,psRes,@[label,best,dimBuf,initBuf,expanded]);
      D2(cb,psMom,@[expanded,dimBuf,mom]);
      D1(cb,psAxis,@[mom,Nbuf,orient],(int)N);
      D2(cb,psProj,@[expanded,dimBuf,mom,orient]);
      D1(cb,psEmit,@[mom,orient,expand,Nbuf,sclBuf,rhwBuf,mbBuf,Hbuf,cwBuf,boxCount],(int)N);
      encodeWarpRec(cb,srcTex);
      [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
      gpuMs=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0;
      count=*(uint32_t*)boxCount.contents; if(count>MAXBOX){truncated++; count=MAXBOX;}
    } else {
      // det in cmd buffer 1
      MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil];
      [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
      double gDet=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0;
      if(dumpDet && im==0){ FILE* df=std::fopen(dumpDet,"wb"); std::fwrite(probBuf.contents,4,N,df); std::fclose(df);
        const float* pm=(const float*)probBuf.contents; double mn=1e9,mx=-1e9,sum=0; size_t hi=0;
        for(size_t k=0;k<N;k++){ float v=pm[k]; mn=std::min(mn,(double)v); mx=std::max(mx,(double)v); sum+=v; if(v>0.2f)hi++; }
        std::printf("DUMP det[%dx%d] -> %s | min=%.4f max=%.4f mean=%.4f >0.2frac=%.4f\n",SZH,SZW,dumpDet,mn,mx,sum/N,(double)hi/N);
        return 0; }
      // host DB-post (reference geometry)
      cv::Mat pred(SZH,SZW,CV_32F,probBuf.contents), bitmap; cv::threshold(pred,bitmap,THRESH,255,cv::THRESH_BINARY); bitmap.convertTo(bitmap,CV_8U);
      std::vector<Box> boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bitmap,oh,ow,SZH,SZW,BOXTHRESH,RATIO,3.0f,2.0f,shifted_buf,mask_buf,contours_buf,hier_buf);
      count=(uint)std::min((size_t)MAXBOX,boxes.size()); if(boxes.size()>MAXBOX)truncated++;
      float* Hm=(float*)Hbuf.contents; int32_t* cwp=(int32_t*)cwBuf.contents;
      for(uint i=0;i<count;i++){ auto ct=turbo_ocr::compute_crop_transform(boxes[i],RECH,RECW); for(int k=0;k<9;k++)Hm[i*9+k]=ct.M_inv[k]; cwp[i]=std::min(ct.crop_width,RECW); }
      // warp+rec in cmd buffer 2
      MPSCommandBuffer* cb2=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      encodeWarpRec(cb2,srcTex);
      [cb2.rootCommandBuffer commit]; [cb2.rootCommandBuffer waitUntilCompleted];
      gpuMs=gDet+(cb2.rootCommandBuffer.GPUEndTime-cb2.rootCommandBuffer.GPUStartTime)*1000.0;
    }
    if(dumpCrops && im==0){
      int show=std::min<uint>(count,40); cv::Mat mont(48*show,RECW,CV_8UC3);
      const float* cp=(const float*)cropsBuf.contents; const size_t plane=(size_t)RECH*RECW;
      for(int i=0;i<show;i++) for(int y=0;y<RECH;y++) for(int x=0;x<RECW;x++){
        size_t bpix=(size_t)i*3*plane+(size_t)y*RECW+x;
        auto U=[&](float v){ return (uchar)std::clamp((int)std::lround((v+1.0f)*127.5f),0,255); };
        mont.at<cv::Vec3b>(i*48+y,x)=cv::Vec3b(U(cp[bpix+2*plane]),U(cp[bpix+plane]),U(cp[bpix])); } // BGR<-(B,G,R planes are R,G,B)
      cv::imwrite(dumpCrops,mont); std::printf("DUMP %d crops -> %s\n",show,dumpCrops);
      // also dump the RAW normalized crop buffer (NCHW float) + count for ORT cross-check
      std::string rawp=std::string(dumpCrops)+".raw"; FILE* rf=std::fopen(rawp.c_str(),"wb");
      uint32_t cc=count; std::fwrite(&cc,4,1,rf); std::fwrite(cropsBuf.contents,4,(size_t)count*3*RECH*RECW,rf); std::fclose(rf);
      std::printf("DUMP rawbuf count=%u -> %s\n",cc,rawp.c_str());
    }
    decode(count,words);
    double wall=ms(t0);
    if(im>=0){ allWords[im]=std::move(words); sum_gpu+=gpuMs; sum_wall+=wall; total_boxes+=count; }
  }

  // write JSON
  FILE* f=std::fopen(outPath.c_str(),"w"); std::fputc('[',f);
  for(int i=0;i<Nimg;i++){ std::fputc('[',f);
    for(size_t k=0;k<allWords[i].size();k++) std::fprintf(f,"\"%s\"%s",jesc(allWords[i][k]).c_str(),k+1<allWords[i].size()?",":"");
    std::fprintf(f,"]%s",i+1<Nimg?",":""); }
  std::fputc(']',f); std::fclose(f);

  std::printf("Apple GPU FUNSD (%s DB-post, tiny models) — N=%d, MAXBOX=%d\n", gpuDB?"GPU":"host", Nimg, MAXBOX);
  std::printf("  avg boxes/img %.1f%s\n", (double)total_boxes/Nimg, truncated?" (some pages truncated!)":"");
  std::printf("  GPU-exec: %.2f ms/img  => %.0f img/s (GPU-bound)\n", sum_gpu/Nimg, 1000.0*Nimg/sum_gpu);
  std::printf("  WALL    : %.2f ms/img  => %.0f img/s (incl host tex upload + %s)\n", sum_wall/Nimg, 1000.0*Nimg/sum_wall, gpuDB?"clears":"cv DB-post");
  std::printf("  wrote %s\n", outPath.c_str());
  return 0;
}}
