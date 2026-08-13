// Apple GPU FUNSD — BATCHED "all at once" throughput runner. The per-command-
// buffer / per-MPSGraph-executable overhead (~5ms) is FIXED, so paying it per
// image caps throughput. Here it is amortized across the WHOLE set:
//   1) det in batches of DB images  (few big det calls, not N)
//   2) host DB-post per image -> gather EVERY crop of EVERY image, bucketed by width
//   3) warp all crops (per source image) + rec each bucket in a few big batched
//      chunks, ALL in ONE command buffer -> ONE commit/sync for all recognition
// So fixed overhead is O(1) in image count; wall approaches GPU-compute-bound.
//
// Run: mps_ocr_funsd_batch <cache> <N> <out.json> <det_export> <rec_base> <keys> [--thresh T --boxthresh T --unclip R --conf C]
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
    default: if((unsigned char)c<0x20){char b[8];std::snprintf(b,sizeof b,"\\u%04x",c);o+=b;} else o+=c; } } return o; }

int main(int argc,char** argv){ @autoreleasepool{
  if(argc<7){ std::fprintf(stderr,"usage: %s <cache> <N> <out.json> <det_export> <rec_base> <keys> [opts]\n",argv[0]); return 2; }
  const std::string cache=argv[1]; const int Nimg=atoi(argv[2]); const std::string outPath=argv[3];
  NSString* detDir=[NSString stringWithUTF8String:argv[4]]; NSString* recBase=[NSString stringWithUTF8String:argv[5]];
  const char* keys=argv[6];
  float THRESH=0.2f,BOXTHRESH=0.40f,RATIO=1.4f,confThr=0.0f;
  for(int i=7;i<argc;i++){ if(!strcmp(argv[i],"--thresh")&&i+1<argc)THRESH=atof(argv[++i]);
    else if(!strcmp(argv[i],"--boxthresh")&&i+1<argc)BOXTHRESH=atof(argv[++i]);
    else if(!strcmp(argv[i],"--unclip")&&i+1<argc)RATIO=atof(argv[++i]);
    else if(!strcmp(argv[i],"--conf")&&i+1<argc)confThr=atof(argv[++i]); }
  const int RECH=48, DB=1;                         // det batch chunk (buildRecGraph det is batch-1 only for now)
  const int BW[]={320,480,800,1200,1600}; const int NBK=5;
  const int RB[]={2560,512,640,320,640};          // rec batch per bucket (sized to cover whole-set demand => 1 chunk, run concurrently)

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue]; NSError* e=nil;
  id<MTLLibrary> wl=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/warp.metallib"] error:&e];
  id<MTLComputePipelineState> warp=[dev newComputePipelineStateWithFunction:[wl newFunctionWithName:@"warp_crops"] error:&e];
  std::vector<std::string> labels={"blank"}; turbo_ocr::recognition::load_label_dict(keys,labels);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  auto BUF=[&](size_t n){ return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
  auto loadG=[&](NSString* dir,int Bb,RecIO& io)->MPSGraph*{ NSData* j=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"graph.json"]]; NSDictionary* G=[NSJSONSerialization JSONObjectWithData:j options:0 error:nil]; NSData* Wd=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights.bin"]]; MPSGraph* g=[MPSGraph new]; io=buildRecGraph(g,G,(const float*)Wd.bytes,Bb); return g; };

  // ---- det compiled at batch DB ----
  RecIO dio; MPSGraph* dg=loadG(detDir,DB,dio);
  const int SZH=(int)dio.ishape[2], SZW=(int)dio.ishape[3]; const size_t NP=(size_t)SZH*SZW;
  MPSGraphExecutable* detExe=[dg compileWithDevice:gdev feeds:@{dio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[dio.output] targetOperations:nil compilationDescriptor:nil];
  id<MTLBuffer> detInBuf=BUF((size_t)DB*3*NP*4); float* detBuf=(float*)detInBuf.contents;
  id<MTLBuffer> probBuf=BUF((size_t)DB*NP*4);
  MPSGraphTensorData* detTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:detInBuf shape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* probTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:probBuf shape:@[@(DB),@1,@(SZH),@(SZW)] dataType:MPSDataTypeFloat32];

  // ---- per-bucket rec compiled at batch RB[b] ----
  struct Bk { int W,RB; long RT; MPSGraphExecutable* exe; RecIO io;
    id<MTLBuffer> crops,idx,mx,H,cw,rdim; MPSGraphTensorData *cropsTD,*idxTD,*maxTD;
    std::vector<int> imgOf; std::vector<float> Hs; std::vector<int> cws; };  // per-crop record
  std::vector<Bk> bk(NBK);
  for(int b=0;b<NBK;b++){ Bk& B=bk[b]; B.W=BW[b]; B.RB=RB[b];
    NSString* dir=[recBase stringByAppendingPathComponent:[NSString stringWithFormat:@"rec_b%d",B.W]];
    MPSGraph* g=loadG(dir,B.RB,B.io);
    MPSGraphTensor* idxT=[g reductionArgMaximumWithTensor:B.io.output axis:2 name:nil];
    MPSGraphTensor* maxT=[g reductionMaximumWithTensor:B.io.output axis:2 name:nil]; B.RT=[idxT.shape[1] longValue];
    B.exe=[g compileWithDevice:gdev feeds:@{B.io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(B.io.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[idxT,maxT] targetOperations:nil compilationDescriptor:nil];
    B.crops=BUF((size_t)B.RB*3*RECH*B.W*4); B.idx=BUF((size_t)B.RB*B.RT*4); B.mx=BUF((size_t)B.RB*B.RT*4);
    B.H=BUF(0); B.cw=BUF(0); // grown after gather
    uint32_t rd[4]={(uint32_t)B.RB,3,(uint32_t)RECH,(uint32_t)B.W}; B.rdim=[dev newBufferWithBytes:rd length:16 options:0];
    B.cropsTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.crops shape:mrb_nums(B.io.ishape) dataType:MPSDataTypeFloat32];
    B.idxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.idx shape:@[@(B.RB),@(B.RT),@1] dataType:MPSDataTypeInt32];
    B.maxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.mx shape:@[@(B.RB),@(B.RT),@1] dataType:MPSDataTypeFloat32];
  }

  auto t_all=clk::now();
  // ---- load all images: src textures + keep originals ----
  std::vector<id<MTLTexture>> tex(Nimg); std::vector<cv::Mat> imgs(Nimg);
  for(int i=0;i<Nimg;i++){ char pth[512]; std::snprintf(pth,sizeof pth,"%s/funsd_%03d.png",cache.c_str(),i);
    imgs[i]=cv::imread(pth,cv::IMREAD_COLOR); if(imgs[i].empty()){std::fprintf(stderr,"read %s\n",pth);return 1;}
    int ow=imgs[i].cols,oh=imgs[i].rows;
    MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:ow height:oh mipmapped:NO]; td.usage=MTLTextureUsageShaderRead;
    tex[i]=[dev newTextureWithDescriptor:td];
    std::vector<uint8_t> px((size_t)ow*oh*4); for(int y=0;y<oh;y++){const cv::Vec3b* r=imgs[i].ptr<cv::Vec3b>(y); for(int x=0;x<ow;x++){size_t k=((size_t)y*ow+x)*4; px[k]=r[x][2];px[k+1]=r[x][1];px[k+2]=r[x][0];px[k+3]=255;}}
    [tex[i] replaceRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0 withBytes:px.data() bytesPerRow:ow*4]; }

  std::vector<std::vector<std::string>> allWords(Nimg);
  std::vector<cv::Point> sbuf; cv::Mat mbuf; std::vector<std::vector<cv::Point>> cbuf; std::vector<cv::Vec4i> hbuf;
  double det_ms=0;
  // ---- det in chunks of DB + host DB-post + gather crops ----
  for(int c0=0;c0<Nimg;c0+=DB){ int cn=std::min(DB,Nimg-c0);
    // preprocess cn images into detBuf (pad the rest of the batch with last image)
    for(int j=0;j<DB;j++){ int i=std::min(c0+j,Nimg-1); cv::Mat di; cv::resize(imgs[i],di,cv::Size(SZW,SZH)); cv::Mat bgr[3]; cv::split(di,bgr);
      const float m[3]={0.485f,0.456f,0.406f},sd[3]={0.229f,0.224f,0.225f};
      for(int ch=0;ch<3;ch++){cv::Mat pp(SZH,SZW,CV_32F,detBuf+((size_t)j*3+ch)*NP); bgr[ch].convertTo(pp,CV_32F,1.0/(255.0*sd[ch]),-m[ch]/sd[ch]);} }
    MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil];
    [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
    det_ms+=(cb.rootCommandBuffer.GPUEndTime-cb.rootCommandBuffer.GPUStartTime)*1000.0;
    for(int j=0;j<cn;j++){ int i=c0+j; int ow=imgs[i].cols,oh=imgs[i].rows;
      cv::Mat pred(SZH,SZW,CV_32F,(float*)probBuf.contents+(size_t)j*NP), bm; cv::threshold(pred,bm,THRESH,255,cv::THRESH_BINARY); bm.convertTo(bm,CV_8U);
      std::vector<Box> boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bm,oh,ow,SZH,SZW,BOXTHRESH,RATIO,3.0f,5.0f,sbuf,mbuf,cbuf,hbuf);
      for(auto& box:boxes){ int nat=turbo_ocr::recognition::rec_input_width(box,RECH);
        int b=NBK-1; for(int k=0;k<NBK;k++){ if(nat<=BW[k]){ b=k; break; } }
        auto ct=turbo_ocr::compute_crop_transform(box,RECH,bk[b].W);
        bk[b].imgOf.push_back(i); for(int k=0;k<9;k++)bk[b].Hs.push_back(ct.M_inv[k]); bk[b].cws.push_back(std::min(ct.crop_width,bk[b].W)); } }
  }

  // ---- warp ALL crops + rec each bucket in chunks, ONE command buffer ----
  double rec_ms=0; auto tr=clk::now();
  // upload per-bucket H/cw
  for(int b=0;b<NBK;b++){ Bk& B=bk[b]; int T=(int)B.imgOf.size(); if(!T)continue;
    B.H=[dev newBufferWithBytes:B.Hs.data() length:B.Hs.size()*4 options:MTLResourceStorageModeShared];
    B.cw=[dev newBufferWithBytes:B.cws.data() length:B.cws.size()*4 options:MTLResourceStorageModeShared]; }
  // one command buffer per bucket (all its crops), COMMIT ALL WITHOUT WAITING so
  // the GPU overlaps the buckets, then wait once and decode. (RB sized to cover total.)
  std::vector<id<MTLCommandBuffer>> pending(NBK,nil);
  for(int b=0;b<NBK;b++){ Bk& B=bk[b]; int T=(int)B.imgOf.size(); if(!T)continue;
    if(T>B.RB){ std::fprintf(stderr,"bucket %d overflow %d>%d\n",B.W,T,B.RB); T=B.RB; }
    MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    int s=0;
    while(s<T){ int img=B.imgOf[s]; int r=s; while(r<T && B.imgOf[r]==img) r++;
      id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder]; [ce setComputePipelineState:warp]; [ce setTexture:tex[img] atIndex:0];
      [ce setBuffer:B.crops offset:(size_t)s*3*RECH*B.W*4 atIndex:0];
      [ce setBuffer:B.H offset:(size_t)s*9*4 atIndex:1]; [ce setBuffer:B.rdim offset:0 atIndex:2]; [ce setBuffer:B.cw offset:(size_t)s*4 atIndex:3];
      [ce dispatchThreads:MTLSizeMake(B.W,RECH,r-s) threadsPerThreadgroup:MTLSizeMake(16,8,1)]; [ce endEncoding];
      s=r; }
    [B.exe encodeToCommandBuffer:cb inputsArray:@[B.cropsTD] resultsArray:@[B.idxTD,B.maxTD] executionDescriptor:nil];
    [cb.rootCommandBuffer commit];                       // NO wait — let buckets overlap on the GPU
    pending[b]=cb.rootCommandBuffer;
  }
  for(int b=0;b<NBK;b++){ if(!pending[b])continue; Bk& B=bk[b]; int T=std::min((int)B.imgOf.size(),B.RB);
    [pending[b] waitUntilCompleted];
    rec_ms+=(pending[b].GPUEndTime-pending[b].GPUStartTime)*1000.0;
    const int32_t* idx=(const int32_t*)B.idx.contents; const float* sc=(const float*)B.mx.contents;
    for(int i=0;i<T;i++){ auto t=turbo_ocr::recognition::ctc_greedy_decode((const int*)(idx+(size_t)i*B.RT),sc+(size_t)i*B.RT,(int)B.RT,labels);
      if(!t.first.empty() && t.second>=confThr) allWords[B.imgOf[i]].push_back(t.first); }
  }
  double rec_wall=ms(tr); double wall=ms(t_all);

  FILE* f=std::fopen(outPath.c_str(),"w"); std::fputc('[',f);
  for(int i=0;i<Nimg;i++){ std::fputc('[',f);
    for(size_t k=0;k<allWords[i].size();k++) std::fprintf(f,"\"%s\"%s",jesc(allWords[i][k]).c_str(),k+1<allWords[i].size()?",":"");
    std::fprintf(f,"]%s",i+1<Nimg?",":""); }
  std::fputc(']',f); std::fclose(f);
  long tot=0; for(auto& B:bk) tot+=B.imgOf.size();
  std::fprintf(stderr,"PERBK");for(int b=0;b<NBK;b++)std::fprintf(stderr," %d=%zu",BW[b],bk[b].imgOf.size());std::fprintf(stderr,"\n");
  std::printf("Apple GPU FUNSD BATCHED  N=%d det %dx%d  total crops %ld\n",Nimg,SZH,SZW,tot);
  std::printf("  det GPU %.1f ms total | rec GPU %.1f ms total | rec wall %.1f ms\n",det_ms,rec_ms,rec_wall);
  std::printf("  WALL total %.1f ms => %.0f img/s (whole set at once)\n",wall,1000.0*Nimg/wall);
  std::printf("  wrote %s\n",outPath.c_str());
  return 0;
}}
