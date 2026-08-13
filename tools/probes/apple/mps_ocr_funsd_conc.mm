// Apple GPU FUNSD — CONCURRENT multi-stream runner. The rec is submission-latency-
// bound (GPU idle between MPSGraph dispatches; FP16 didn't help), so K worker
// threads SHARE the compiled executables but each owns its command queue + buffers,
// processing disjoint images. Thread B's dispatches fill the GPU while thread A
// waits — overlapping the latency that a single stream leaves idle.
//
// Run: mps_ocr_funsd_conc <cache> <N> <out.json> <det_export> <rec_base> <keys> [--threads K]
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
#include <thread>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
using turbo_ocr::Box;
static std::string jesc(const std::string& s){ std::string o; o.reserve(s.size()+2);
  for(char c: s){ switch(c){ case '"':o+="\\\"";break; case '\\':o+="\\\\";break; case '\n':o+="\\n";break;
    case '\r':o+="\\r";break; case '\t':o+="\\t";break;
    default: if((unsigned char)c<0x20){char b[8];std::snprintf(b,sizeof b,"\\u%04x",c);o+=b;} else o+=c; } } return o; }

int main(int argc,char** argv){ @autoreleasepool{
  if(argc<7){ std::fprintf(stderr,"usage: %s <cache> <N> <out.json> <det_export> <rec_base> <keys> [--threads K]\n",argv[0]); return 2; }
  const std::string cache=argv[1]; const int Nimg=atoi(argv[2]); const std::string outPath=argv[3];
  NSString* detDir=[NSString stringWithUTF8String:argv[4]]; NSString* recBase=[NSString stringWithUTF8String:argv[5]];
  const char* keys=argv[6];
  int K=4; for(int i=7;i<argc;i++) if(!strcmp(argv[i],"--threads")&&i+1<argc) K=atoi(argv[++i]);
  const float THRESH=0.2f,BOXTHRESH=0.40f,RATIO=1.4f; const int RECH=48;
  const int BW[]={320,480,800,1200,1600}; const int NBK=5;
  const std::vector<std::vector<int>> LADDER={ {16,64,128}, {8,24}, {8,24}, {4,16}, {4,24} };
  const bool REC_FP16 = !(getenv("MPS_REC_FP16") && getenv("MPS_REC_FP16")[0]=='0');

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError* e=nil;
  id<MTLLibrary> wl=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/warp.metallib"] error:&e];
  id<MTLComputePipelineState> warp=[dev newComputePipelineStateWithFunction:[wl newFunctionWithName:@"warp_crops"] error:&e];
  std::vector<std::string> labels={"blank"}; turbo_ocr::recognition::load_label_dict(keys,labels);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  auto BUF=[&](size_t n){ return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
  auto loadG=[&](NSString* dir,int Bb,RecIO& io,bool fp16)->MPSGraph*{ NSData* j=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"graph.json"]]; NSDictionary* G=[NSJSONSerialization JSONObjectWithData:j options:0 error:nil]; NSData* Wd=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights.bin"]]; MPSGraph* g=[MPSGraph new]; io=buildRecGraph(g,G,(const float*)Wd.bytes,Bb,fp16); return g; };

  // ---- shared compiled executables ----
  RecIO dio; MPSGraph* dg=loadG(detDir,1,dio,false);
  const int SZH=(int)dio.ishape[2], SZW=(int)dio.ishape[3]; const size_t N=(size_t)SZH*SZW;
  MPSGraphExecutable* detExe=[dg compileWithDevice:gdev feeds:@{dio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[dio.output] targetOperations:nil compilationDescriptor:nil];
  struct BB { int MB; MPSGraphExecutable* exe; std::vector<long> ishape; };
  struct BkS { int W; long RT; int maxMB; std::vector<BB> batches; };
  std::vector<BkS> bks(NBK);
  for(int b=0;b<NBK;b++){ BkS& B=bks[b]; B.W=BW[b]; B.maxMB=*std::max_element(LADDER[b].begin(),LADDER[b].end());
    NSString* dir=[recBase stringByAppendingPathComponent:[NSString stringWithFormat:@"rec_b%d",B.W]];
    RecIO io0; MPSGraph* g0=loadG(dir,B.maxMB,io0,REC_FP16); MPSGraphTensor* it0=[g0 reductionArgMaximumWithTensor:io0.output axis:2 name:nil]; B.RT=[it0.shape[1] longValue];
    for(int mb : LADDER[b]){ RecIO io; MPSGraph* g=loadG(dir,mb,io,REC_FP16);
      MPSGraphTensor* idxT=[g reductionArgMaximumWithTensor:io.output axis:2 name:nil];
      MPSGraphTensor* maxT=[g reductionMaximumWithTensor:io.output axis:2 name:nil];
      MPSGraphExecutable* exe=[g compileWithDevice:gdev feeds:@{io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[idxT,maxT] targetOperations:nil compilationDescriptor:nil];
      B.batches.push_back({mb,exe,io.ishape}); }
    std::sort(B.batches.begin(),B.batches.end(),[](const BB&x,const BB&y){return x.MB<y.MB;});
  }

  std::vector<std::vector<std::string>> allWords(Nimg);
  std::vector<cv::Mat> imgs(Nimg);
  for(int i=0;i<Nimg;i++){ char p[512]; std::snprintf(p,sizeof p,"%s/funsd_%03d.png",cache.c_str(),i); imgs[i]=cv::imread(p,cv::IMREAD_COLOR); if(imgs[i].empty()){std::fprintf(stderr,"read %s\n",p);return 1;} }

  auto worker=[&](int tid){ @autoreleasepool{
    id<MTLCommandQueue> q=[dev newCommandQueue];
    id<MTLBuffer> detIn=BUF(3*N*4); float* detBuf=(float*)detIn.contents; id<MTLBuffer> probBuf=BUF(N*4);
    MPSGraphTensorData* detTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:detIn shape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* probTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:probBuf shape:@[@1,@1,@(SZH),@(SZW)] dataType:MPSDataTypeFloat32];
    struct BkT { id<MTLBuffer> crops,idx,mx,H,cw,rdim; std::vector<MPSGraphTensorData*> cropsTD,idxTD,maxTD; };
    std::vector<BkT> bk(NBK);
    for(int b=0;b<NBK;b++){ BkS& S=bks[b]; BkT& B=bk[b];
      B.crops=BUF((size_t)S.maxMB*3*RECH*S.W*4); B.idx=BUF((size_t)S.maxMB*S.RT*4); B.mx=BUF((size_t)S.maxMB*S.RT*4);
      B.H=BUF((size_t)S.maxMB*9*4); B.cw=BUF((size_t)S.maxMB*4);
      uint32_t rd[4]={(uint32_t)S.maxMB,3,(uint32_t)RECH,(uint32_t)S.W}; B.rdim=[dev newBufferWithBytes:rd length:16 options:0];
      for(auto& bb:S.batches){ B.cropsTD.push_back([[MPSGraphTensorData alloc] initWithMTLBuffer:B.crops shape:mrb_nums(bb.ishape) dataType:MPSDataTypeFloat32]);
        B.idxTD.push_back([[MPSGraphTensorData alloc] initWithMTLBuffer:B.idx shape:@[@(bb.MB),@(S.RT),@1] dataType:MPSDataTypeInt32]);
        B.maxTD.push_back([[MPSGraphTensorData alloc] initWithMTLBuffer:B.mx shape:@[@(bb.MB),@(S.RT),@1] dataType:MPSDataTypeFloat32]); } }
    std::vector<cv::Point> sbuf; cv::Mat mbuf; std::vector<std::vector<cv::Point>> cbuf; std::vector<cv::Vec4i> hbuf;
    auto pick=[&](int b,int nb){ for(size_t j=0;j<bks[b].batches.size();j++) if(bks[b].batches[j].MB>=nb) return (int)j; return (int)bks[b].batches.size()-1; };

    for(int im=tid; im<Nimg; im+=K){
      cv::Mat& orig=imgs[im]; int ow=orig.cols,oh=orig.rows;
      MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:ow height:oh mipmapped:NO]; td.usage=MTLTextureUsageShaderRead;
      id<MTLTexture> srcTex=[dev newTextureWithDescriptor:td];
      { std::vector<uint8_t> px((size_t)ow*oh*4); for(int y=0;y<oh;y++){const cv::Vec3b* r=orig.ptr<cv::Vec3b>(y); for(int x=0;x<ow;x++){size_t k=((size_t)y*ow+x)*4; px[k]=r[x][2];px[k+1]=r[x][1];px[k+2]=r[x][0];px[k+3]=255;}}
        [srcTex replaceRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0 withBytes:px.data() bytesPerRow:ow*4]; }
      { cv::Mat di; cv::resize(orig,di,cv::Size(SZW,SZH)); cv::Mat bgr[3]; cv::split(di,bgr); const float m[3]={0.485f,0.456f,0.406f},sd[3]={0.229f,0.224f,0.225f};
        for(int c=0;c<3;c++){cv::Mat pp(SZH,SZW,CV_32F,detBuf+(size_t)c*N); bgr[c].convertTo(pp,CV_32F,1.0/(255.0*sd[c]),-m[c]/sd[c]);} }
      { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q]; [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil]; [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
      cv::Mat pred(SZH,SZW,CV_32F,probBuf.contents), bm; cv::threshold(pred,bm,THRESH,255,cv::THRESH_BINARY); bm.convertTo(bm,CV_8U);
      std::vector<Box> boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bm,oh,ow,SZH,SZW,BOXTHRESH,RATIO,3.0f,5.0f,sbuf,mbuf,cbuf,hbuf);
      std::vector<std::vector<int>> lists(NBK);
      for(int i=0;i<(int)boxes.size();i++){ int nat=turbo_ocr::recognition::rec_input_width(boxes[i],RECH); int b=NBK-1; for(int k=0;k<NBK;k++) if(nat<=BW[k]){b=k;break;} lists[b].push_back(i); }
      int nb[NBK]={0},bi[NBK]={0};
      for(int b=0;b<NBK;b++){ if(lists[b].empty())continue; BkS& S=bks[b]; BkT& B=bk[b];
        int n=(int)std::min((size_t)S.maxMB,lists[b].size()); nb[b]=n; bi[b]=pick(b,n);
        float* Hm=(float*)B.H.contents; int32_t* cwp=(int32_t*)B.cw.contents;
        for(int i=0;i<n;i++){ auto ct=turbo_ocr::compute_crop_transform(boxes[lists[b][i]],RECH,S.W); for(int k=0;k<9;k++)Hm[i*9+k]=ct.M_inv[k]; cwp[i]=std::min(ct.crop_width,S.W); } }
      MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      for(int b=0;b<NBK;b++){ if(!nb[b])continue; BkS& S=bks[b]; BkT& B=bk[b]; int j=bi[b];
        { id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder]; [ce setComputePipelineState:warp]; [ce setTexture:srcTex atIndex:0];
          [ce setBuffer:B.crops offset:0 atIndex:0];[ce setBuffer:B.H offset:0 atIndex:1];[ce setBuffer:B.rdim offset:0 atIndex:2];[ce setBuffer:B.cw offset:0 atIndex:3];
          [ce dispatchThreads:MTLSizeMake(S.W,RECH,nb[b]) threadsPerThreadgroup:MTLSizeMake(16,8,1)];[ce endEncoding]; }
        [S.batches[j].exe encodeToCommandBuffer:cb inputsArray:@[B.cropsTD[j]] resultsArray:@[B.idxTD[j],B.maxTD[j]] executionDescriptor:nil];
      }
      [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
      std::vector<std::string> words;
      for(int b=0;b<NBK;b++){ if(!nb[b])continue; BkT& B=bk[b]; long RT=bks[b].RT;
        const int32_t* idx=(const int32_t*)B.idx.contents; const float* sc=(const float*)B.mx.contents;
        for(int i=0;i<nb[b];i++){ auto t=turbo_ocr::recognition::ctc_greedy_decode((const int*)(idx+(size_t)i*RT),sc+(size_t)i*RT,(int)RT,labels); if(!t.first.empty()) words.push_back(t.first); } }
      allWords[im]=std::move(words);
    }
  }};

  auto t0=clk::now();
  std::vector<std::thread> ts; for(int t=0;t<K;t++) ts.emplace_back(worker,t);
  for(auto& th:ts) th.join();
  double wall=ms(t0);

  FILE* f=std::fopen(outPath.c_str(),"w"); std::fputc('[',f);
  for(int i=0;i<Nimg;i++){ std::fputc('[',f);
    for(size_t k=0;k<allWords[i].size();k++) std::fprintf(f,"\"%s\"%s",jesc(allWords[i][k]).c_str(),k+1<allWords[i].size()?",":"");
    std::fprintf(f,"]%s",i+1<Nimg?",":""); }
  std::fputc(']',f); std::fclose(f);
  std::printf("Apple GPU FUNSD CONCURRENT K=%d  N=%d  WALL %.1f ms => %.0f img/s\n",K,Nimg,wall,1000.0*Nimg/wall);
  std::printf("  wrote %s\n",outPath.c_str());
  return 0;
}}
