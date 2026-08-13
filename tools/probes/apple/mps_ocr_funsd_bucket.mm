// Apple GPU FUNSD runner — WIDTH-BUCKETED recognition with per-bucket BATCH-SIZE
// LADDERS. Each detected line warps at natural aspect into the smallest width
// bucket that fits (minimal padding => rec stays in-distribution => full accuracy),
// and each bucket picks the tightest static batch >= its actual crop count this
// page (so sparse pages run tiny batches — MPSGraph cost scales with batch size).
// det(MPSGraph, live buf) -> host DB-post -> per-bucket {warp(Metal)+rec(MPSGraph)} one cmd buffer -> CTC.
//
// Run: mps_ocr_funsd_bucket <cache> <N> <out.json> <det_export> <rec_base_dir> <keys> [--thresh T --boxthresh T --unclip R --conf C]
#import <Metal/Metal.h>
#include "mps_rec_build.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "turbo_ocr/analysis/detection/det_postprocess.h"
#include "turbo_ocr/analysis/recognition/ctc_decode.h"
#include "turbo_ocr/analysis/recognition/rec_geometry.h"
#include "turbo_ocr/base/geometry/perspective.h"
#include "turbo_ocr/base/geometry/box.h"
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
using turbo_ocr::Box;

static std::string jesc(const std::string& s){ std::string o; o.reserve(s.size()+2);
  for(char c: s){ switch(c){ case '"':o+="\\\"";break; case '\\':o+="\\\\";break; case '\n':o+="\\n";break;
    case '\r':o+="\\r";break; case '\t':o+="\\t";break;
    default: if((unsigned char)c<0x20){char b[8];std::snprintf(b,sizeof b,"\\u%04x",c);o+=b;} else o+=c; } }
  return o; }

int main(int argc,char** argv){ @autoreleasepool{
  if(argc<7){ std::fprintf(stderr,"usage: %s <cache> <N> <out.json> <det_export> <rec_base_dir> <keys> [opts]\n",argv[0]); return 2; }
  const std::string cache=argv[1]; const int Nimg=atoi(argv[2]); const std::string outPath=argv[3];
  NSString* detDir=[NSString stringWithUTF8String:argv[4]]; NSString* recBase=[NSString stringWithUTF8String:argv[5]];
  const char* keys=argv[6];
  float THRESH=0.2f,BOXTHRESH=0.40f,RATIO=1.4f,confThr=0.0f;
  for(int i=7;i<argc;i++){ if(!strcmp(argv[i],"--thresh")&&i+1<argc)THRESH=atof(argv[++i]);
    else if(!strcmp(argv[i],"--boxthresh")&&i+1<argc)BOXTHRESH=atof(argv[++i]);
    else if(!strcmp(argv[i],"--unclip")&&i+1<argc)RATIO=atof(argv[++i]);
    else if(!strcmp(argv[i],"--conf")&&i+1<argc)confThr=atof(argv[++i]); }
  const int RECH=48;
  const int BW[]={320,480,800,1200,1600}; const int NBK=5;
  // per-bucket batch ladders (tightest static batch >= this page's demand)
  const std::vector<std::vector<int>> LADDER={ {16,64,128}, {8,24}, {8,24}, {4,16}, {4,24} };

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue]; NSError* e=nil;
  id<MTLLibrary> wl=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/warp.metallib"] error:&e];
  id<MTLComputePipelineState> warp=[dev newComputePipelineStateWithFunction:[wl newFunctionWithName:@"warp_crops"] error:&e];
  std::vector<std::string> labels={"blank"}; turbo_ocr::recognition::load_label_dict(keys,labels);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  auto BUF=[&](size_t n){ return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
  const bool REC_FP16 = !(getenv("MPS_REC_FP16") && getenv("MPS_REC_FP16")[0]=='0');  // rec in FP16 by default; MPS_REC_FP16=0 disables
  auto loadG=[&](NSString* dir,int Bb,RecIO& io,bool fp16)->MPSGraph*{ NSData* j=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"graph.json"]]; NSDictionary* G=[NSJSONSerialization JSONObjectWithData:j options:0 error:nil]; NSData* Wd=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights.bin"]]; MPSGraph* g=[MPSGraph new]; g.options=MPSGraphOptionsNone; io=buildRecGraph(g,G,(const float*)Wd.bytes,Bb,fp16); return g; };  // OptionsNone: skip the default SynchronizeResults blit (shared buffers)

  // ---- det ----
  RecIO dio; MPSGraph* dg=loadG(detDir,1,dio,false);
  const int SZH=(int)dio.ishape[2], SZW=(int)dio.ishape[3]; const size_t N=(size_t)SZH*SZW;
  MPSGraphExecutable* detExe=[dg compileWithDevice:gdev feeds:@{dio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[dio.output] targetOperations:nil compilationDescriptor:nil];
  id<MTLBuffer> detInBuf=BUF(3*N*4); float* detBuf=(float*)detInBuf.contents;
  id<MTLBuffer> probBuf=BUF(N*4);
  MPSGraphTensorData* detTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:detInBuf shape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* probTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:probBuf shape:@[@1,@1,@(SZH),@(SZW)] dataType:MPSDataTypeFloat32];

  // ---- per-bucket rec (one width, several batch-size executables) ----
  struct Batch { int MB; MPSGraphExecutable* exe; MPSGraphTensorData *cropsTD,*idxTD,*maxTD; };
  struct Bk { int W; long RT; int maxMB; id<MTLBuffer> crops,idx,mx,H,cw,rdim; std::vector<Batch> batches; };
  std::vector<Bk> bk(NBK);
  for(int b=0;b<NBK;b++){ Bk& B=bk[b]; B.W=BW[b]; B.maxMB=*std::max_element(LADDER[b].begin(),LADDER[b].end());
    NSString* dir=[recBase stringByAppendingPathComponent:[NSString stringWithFormat:@"rec_b%d",B.W]];
    // shared buffers sized for the largest batch in the ladder
    RecIO io0; MPSGraph* g0=loadG(dir,B.maxMB,io0,REC_FP16);
    MPSGraphTensor* it0=[g0 reductionArgMaximumWithTensor:io0.output axis:2 name:nil]; B.RT=[it0.shape[1] longValue];
    B.crops=BUF((size_t)B.maxMB*3*RECH*B.W*4); B.idx=BUF((size_t)B.maxMB*B.RT*4); B.mx=BUF((size_t)B.maxMB*B.RT*4);
    B.H=BUF((size_t)B.maxMB*9*4); B.cw=BUF((size_t)B.maxMB*4);
    uint32_t rd[4]={(uint32_t)B.maxMB,3,(uint32_t)RECH,(uint32_t)B.W}; B.rdim=[dev newBufferWithBytes:rd length:16 options:0];
    for(int mb : LADDER[b]){ Batch bt; bt.MB=mb; RecIO io; MPSGraph* g=loadG(dir,mb,io,REC_FP16);
      MPSGraphTensor* idxT=[g reductionArgMaximumWithTensor:io.output axis:2 name:nil];
      MPSGraphTensor* maxT=[g reductionMaximumWithTensor:io.output axis:2 name:nil];
      bt.exe=[g compileWithDevice:gdev feeds:@{io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[idxT,maxT] targetOperations:nil compilationDescriptor:nil];
      bt.cropsTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.crops shape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32];
      bt.idxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.idx shape:@[@(mb),@(B.RT),@1] dataType:MPSDataTypeInt32];
      bt.maxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.mx shape:@[@(mb),@(B.RT),@1] dataType:MPSDataTypeFloat32];
      B.batches.push_back(bt); }
    std::sort(B.batches.begin(),B.batches.end(),[](const Batch&x,const Batch&y){return x.MB<y.MB;});
  }
  auto pickBatch=[&](Bk& B,int nb)->Batch*{ for(auto& bt:B.batches) if(bt.MB>=nb) return &bt; return &B.batches.back(); };

  // warm every executable once (JIT before timing)
  { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil];
    for(int b=0;b<NBK;b++) for(auto& bt:bk[b].batches)[bt.exe encodeToCommandBuffer:cb inputsArray:@[bt.cropsTD] resultsArray:@[bt.idxTD,bt.maxTD] executionDescriptor:nil];
    [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }

  std::vector<cv::Point> sbuf; cv::Mat mbuf; std::vector<std::vector<cv::Point>> cbuf; std::vector<cv::Vec4i> hbuf;
  std::vector<std::vector<std::string>> allWords(Nimg);
  double sum_gpu=0,sum_wall=0; long total_boxes=0; int truncated=0;
  double a_tex=0,a_prep=0,a_det=0,a_db=0,a_homo=0,a_enc=0,a_wait=0,a_dec=0;

  for(int im=-1; im<Nimg; ++im){
    int pidx=im<0?0:im;
    char pth[512]; std::snprintf(pth,sizeof pth,"%s/funsd_%03d.png",cache.c_str(),pidx);
    cv::Mat orig=cv::imread(pth,cv::IMREAD_COLOR); if(orig.empty()){ std::fprintf(stderr,"cannot read %s\n",pth); return 1; }
    int ow=orig.cols,oh=orig.rows;
    auto t0=clk::now(); double gpuMs=0; auto ts=clk::now();
    MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:ow height:oh mipmapped:NO]; td.usage=MTLTextureUsageShaderRead;
    id<MTLTexture> srcTex=[dev newTextureWithDescriptor:td];
    { std::vector<uint8_t> px((size_t)ow*oh*4); for(int y=0;y<oh;y++){const cv::Vec3b* r=orig.ptr<cv::Vec3b>(y); for(int x=0;x<ow;x++){size_t i=((size_t)y*ow+x)*4; px[i]=r[x][2];px[i+1]=r[x][1];px[i+2]=r[x][0];px[i+3]=255;}}
      [srcTex replaceRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0 withBytes:px.data() bytesPerRow:ow*4]; }
    double t_tex=ms(ts); ts=clk::now();
    { cv::Mat di; cv::resize(orig,di,cv::Size(SZW,SZH)); cv::Mat bgr[3]; cv::split(di,bgr); const float m[3]={0.485f,0.456f,0.406f},sd[3]={0.229f,0.224f,0.225f};
      for(int c=0;c<3;c++){cv::Mat pp(SZH,SZW,CV_32F,detBuf+(size_t)c*N); bgr[c].convertTo(pp,CV_32F,1.0/(255.0*sd[c]),-m[c]/sd[c]);} }
    double t_prep=ms(ts); ts=clk::now();
    { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil];
      [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
      gpuMs+=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0; }
    double t_det=ms(ts); ts=clk::now();
    cv::Mat pred(SZH,SZW,CV_32F,probBuf.contents), bm; cv::threshold(pred,bm,THRESH,255,cv::THRESH_BINARY); bm.convertTo(bm,CV_8U);
    std::vector<Box> boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bm,oh,ow,SZH,SZW,BOXTHRESH,RATIO,3.0f,5.0f,sbuf,mbuf,cbuf,hbuf);
    double t_db=ms(ts); ts=clk::now();

    std::vector<std::vector<int>> lists(NBK);
    for(int i=0;i<(int)boxes.size();i++){ int nat=turbo_ocr::recognition::rec_input_width(boxes[i],RECH);
      int b=NBK-1; for(int k=0;k<NBK;k++){ if(nat<=BW[k]){ b=k; break; } } lists[b].push_back(i); }

    // host: build homographies; choose per-bucket batch executable
    int nb[NBK]={0}; Batch* chosen[NBK]={nullptr};
    for(int b=0;b<NBK;b++){ if(lists[b].empty()) continue; Bk& B=bk[b];
      int n=(int)std::min((size_t)B.maxMB,lists[b].size()); if(lists[b].size()>(size_t)B.maxMB)truncated++; nb[b]=n; chosen[b]=pickBatch(B,n);
      float* Hm=(float*)B.H.contents; int32_t* cwp=(int32_t*)B.cw.contents;
      for(int i=0;i<n;i++){ auto ct=turbo_ocr::compute_crop_transform(boxes[lists[b][i]],RECH,B.W); for(int k=0;k<9;k++)Hm[i*9+k]=ct.M_inv[k]; cwp[i]=std::min(ct.crop_width,B.W); } }
    double t_homo=ms(ts); ts=clk::now();
    // GPU: all buckets in one command buffer
    MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    for(int b=0;b<NBK;b++){ if(!nb[b]) continue; Bk& B=bk[b];
      { id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder]; [ce setComputePipelineState:warp]; [ce setTexture:srcTex atIndex:0];
        [ce setBuffer:B.crops offset:0 atIndex:0];[ce setBuffer:B.H offset:0 atIndex:1];[ce setBuffer:B.rdim offset:0 atIndex:2];[ce setBuffer:B.cw offset:0 atIndex:3];
        [ce dispatchThreads:MTLSizeMake(B.W,RECH,nb[b]) threadsPerThreadgroup:MTLSizeMake(16,8,1)];[ce endEncoding]; }
      [chosen[b]->exe encodeToCommandBuffer:cb inputsArray:@[chosen[b]->cropsTD] resultsArray:@[chosen[b]->idxTD,chosen[b]->maxTD] executionDescriptor:nil];
    }
    double t_enc=ms(ts); ts=clk::now();
    [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
    double t_wait=ms(ts); ts=clk::now();
    gpuMs+=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0;
    std::vector<std::string> words;
    for(int b=0;b<NBK;b++){ if(!nb[b]) continue; Bk& B=bk[b];
      const int32_t* idx=(const int32_t*)B.idx.contents; const float* sc=(const float*)B.mx.contents;
      for(int i=0;i<nb[b];i++){ auto t=turbo_ocr::recognition::ctc_greedy_decode((const int*)(idx+(size_t)i*B.RT),sc+(size_t)i*B.RT,(int)B.RT,labels);
        if(!t.first.empty() && t.second>=confThr) words.push_back(t.first); } }
    double t_dec=ms(ts); double wall=ms(t0);
    if(im>=0){ allWords[im]=std::move(words); sum_gpu+=gpuMs; sum_wall+=wall; total_boxes+=boxes.size();
      a_tex+=t_tex; a_prep+=t_prep; a_det+=t_det; a_db+=t_db; a_homo+=t_homo; a_enc+=t_enc; a_wait+=t_wait; a_dec+=t_dec; }
  }

  FILE* f=std::fopen(outPath.c_str(),"w"); std::fputc('[',f);
  for(int i=0;i<Nimg;i++){ std::fputc('[',f);
    for(size_t k=0;k<allWords[i].size();k++) std::fprintf(f,"\"%s\"%s",jesc(allWords[i][k]).c_str(),k+1<allWords[i].size()?",":"");
    std::fprintf(f,"]%s",i+1<Nimg?",":""); }
  std::fputc(']',f); std::fclose(f);
  std::printf("Apple GPU FUNSD (bucketed rec + batch ladders) N=%d det %dx%d\n",Nimg,SZH,SZW);
  std::printf("  avg boxes/img %.1f%s\n",(double)total_boxes/Nimg,truncated?" (bucket truncation)":"");
  std::printf("  GPU-exec(sum): %.2f ms/img => %.0f img/s\n",sum_gpu/Nimg,1000.0*Nimg/sum_gpu);
  std::printf("  WALL         : %.2f ms/img => %.0f img/s\n",sum_wall/Nimg,1000.0*Nimg/sum_wall);
  std::printf("  host ms/img: tex=%.1f prep=%.1f det=%.1f DBpost=%.1f homo=%.1f encode=%.1f commit+wait=%.1f decode=%.1f\n",a_tex/Nimg,a_prep/Nimg,a_det/Nimg,a_db/Nimg,a_homo/Nimg,a_enc/Nimg,a_wait/Nimg,a_dec/Nimg);
  std::printf("  wrote %s\n",outPath.c_str());
  return 0;
}}
