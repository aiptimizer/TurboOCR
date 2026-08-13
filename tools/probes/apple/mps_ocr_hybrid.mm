// Apple GPU+ANE HYBRID OCR. Detection + WIDE-crop recognition on the GPU
// (MPSGraph), NARROW-crop recognition on the Apple Neural Engine (CoreML), so
// the two engines run in parallel. Stage flag HYBRID_OVERLAP=1 runs the ANE rec
// on a separate thread pipelined with GPU work; default is single-threaded (correctness).
//
// Run: mps_ocr_hybrid <cache> <N> <out.json> <det_export> <rec_base> <coreml_dir> <keys>
#import <Metal/Metal.h>
#import <CoreML/CoreML.h>
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
#include <deque>
#include <mutex>
#include <condition_variable>
#include <algorithm>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
using turbo_ocr::Box;
static std::string jesc(const std::string& s){ std::string o; o.reserve(s.size()+2);
  for(char c: s){ switch(c){ case '"':o+="\\\"";break; case '\\':o+="\\\\";break; case '\n':o+="\\n";break;
    case '\r':o+="\\r";break; case '\t':o+="\\t";break;
    default: if((unsigned char)c<0x20){char b[8];std::snprintf(b,sizeof b,"\\u%04x",c);o+=b;} else o+=c; } } return o; }

int main(int argc,char** argv){ @autoreleasepool{
  if(argc<8){ std::fprintf(stderr,"usage: %s <cache> <N> <out.json> <det_export> <rec_base> <coreml_dir> <keys>\n",argv[0]); return 2; }
  const std::string cache=argv[1]; const int Nimg=atoi(argv[2]); const std::string outPath=argv[3];
  NSString* detDir=[NSString stringWithUTF8String:argv[4]]; NSString* recBase=[NSString stringWithUTF8String:argv[5]];
  NSString* cmlDir=[NSString stringWithUTF8String:argv[6]]; const char* keys=argv[7];
  const float THRESH=0.2f,BOXTHRESH=0.40f,RATIO=1.4f; const int RECH=48;
  // Wide GPU buckets consolidated to ONE (1600): the 800/1200/1600 crops are few
  // (~15/img) but 3 separate MPSGraph executables = pure dispatch tax. One wide
  // bucket = one GPU rec executable (crops pad up to 1600).
  const int BW[]={320,480,800,1600}; const int NBK=4;
  const int ANE_MAXW = getenv("ANE_MAXW")?atoi(getenv("ANE_MAXW")):480;
  bool ANE[NBK]; for(int b=0;b<NBK;b++) ANE[b]=(BW[b]<=ANE_MAXW);
  const std::vector<std::vector<int>> ANE_ENUM={ {16,48,96,160}, {8,24,48}, {8,24,48}, {4,16,32} };
  const std::vector<std::vector<int>> GPU_LADDER={ {16,64,128}, {8,24}, {8,24}, {8,32,64} };

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue]; NSError* e=nil;
  id<MTLLibrary> wl=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build-cpu/warp.metallib"] error:&e];
  id<MTLComputePipelineState> warp=[dev newComputePipelineStateWithFunction:[wl newFunctionWithName:@"warp_crops"] error:&e];
  std::vector<std::string> labels={"blank"}; turbo_ocr::recognition::load_label_dict(keys,labels);
  MPSGraphDevice* gdev=[MPSGraphDevice deviceWithMTLDevice:dev];
  auto BUF=[&](size_t n){ return [dev newBufferWithLength:std::max<size_t>(n,16) options:MTLResourceStorageModeShared]; };
  const bool REC_FP16 = !(getenv("MPS_REC_FP16") && getenv("MPS_REC_FP16")[0]=='0');
  auto loadG=[&](NSString* dir,int Bb,RecIO& io,bool fp16)->MPSGraph*{ NSData* j=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"graph.json"]]; NSDictionary* G=[NSJSONSerialization JSONObjectWithData:j options:0 error:nil]; NSData* Wd=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights.bin"]]; MPSGraph* g=[MPSGraph new]; g.options=MPSGraphOptionsNone; io=buildRecGraph(g,G,(const float*)Wd.bytes,Bb,fp16); return g; };
  auto loadANE=[&](int W)->MLModel*{ NSString* p=[cmlDir stringByAppendingPathComponent:[NSString stringWithFormat:@"rec_ane_%d.mlpackage",W]];
    NSError* err=nil; NSURL* c=[MLModel compileModelAtURL:[NSURL fileURLWithPath:p] error:&err];
    if(!c){ std::fprintf(stderr,"ANE compile %d: %s\n",W,err.localizedDescription.UTF8String); return nil; }
    MLModelConfiguration* cfg=[[MLModelConfiguration alloc] init]; cfg.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
    MLModel* m=[MLModel modelWithContentsOfURL:c configuration:cfg error:&err];
    if(!m) std::fprintf(stderr,"ANE load %d: %s\n",W,err.localizedDescription.UTF8String); return m; };

  // ---- det (GPU) ----
  RecIO dio; MPSGraph* dg=loadG(detDir,1,dio,false);
  const int SZH=(int)dio.ishape[2], SZW=(int)dio.ishape[3]; const size_t N=(size_t)SZH*SZW;
  MPSGraphExecutable* detExe=[dg compileWithDevice:gdev feeds:@{dio.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[dio.output] targetOperations:nil compilationDescriptor:nil];
  id<MTLBuffer> detInBuf=BUF(3*N*4); float* detBuf=(float*)detInBuf.contents; id<MTLBuffer> probBuf=BUF(N*4);
  MPSGraphTensorData* detTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:detInBuf shape:mrb_nums(dio.ishape) dataType:MPSDataTypeFloat32];
  MPSGraphTensorData* probTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:probBuf shape:@[@1,@1,@(SZH),@(SZW)] dataType:MPSDataTypeFloat32];
  // Optional CoreML-GPU det (accurate, 3.9<4.7ms MPSGraph). DET_COREML=1 enables.
  const bool DET_COREML = getenv("DET_COREML") && getenv("DET_COREML")[0]=='1';
  MLModel* detModel=nil;
  if(DET_COREML){ NSURL* c=[MLModel compileModelAtURL:[NSURL fileURLWithPath:[cmlDir stringByAppendingPathComponent:@"det_gpu_992.mlpackage"]] error:&e];
    MLModelConfiguration* cfg=[[MLModelConfiguration alloc] init]; cfg.computeUnits=MLComputeUnitsCPUAndGPU;
    detModel=c?[MLModel modelWithContentsOfURL:c configuration:cfg error:&e]:nil;
    if(!detModel){ std::fprintf(stderr,"det CoreML load failed: %s\n",e.localizedDescription.UTF8String); return 1; } }
  auto runDetCoreML=[&](){ NSError* err=nil; NSArray* sh=@[@1,@3,@(SZH),@(SZW)]; NSArray* st=@[@(3*N),@(N),@(SZW),@1];
    MLMultiArray* in=[[MLMultiArray alloc] initWithDataPointer:detBuf shape:sh dataType:MLMultiArrayDataTypeFloat32 strides:st deallocator:nil error:&err];
    MLDictionaryFeatureProvider* fp=[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:in]} error:&err];
    id<MLFeatureProvider> out=[detModel predictionFromFeatures:fp error:&err];
    MLMultiArray* pm=[[out featureValueForName:out.featureNames.anyObject] multiArrayValue];
    [pm getBytesWithHandler:^(const void* bytes, NSInteger sz){ memcpy(probBuf.contents, bytes, std::min((size_t)sz,(size_t)N*4)); }]; };

  // ---- per-bucket state (GPU: MPSGraph ladder; ANE: CoreML model) ----
  struct GBatch { int MB; MPSGraphExecutable* exe; MPSGraphTensorData *cropsTD,*idxTD,*maxTD; };
  struct Bk { int W; bool ane; long RT/*=T*/; int maxMB; id<MTLBuffer> crops,idx,mx,H,cw,rdim;
    std::vector<GBatch> gbatches; MLModel* model; std::vector<int> enums; };
  std::vector<Bk> bk(NBK);
  for(int b=0;b<NBK;b++){ Bk& B=bk[b]; B.W=BW[b]; B.ane=ANE[b]; B.RT=B.W/8;   // rec_tiny T = W/8
    if(B.ane){ B.enums=ANE_ENUM[b]; B.maxMB=B.enums.back(); B.model=loadANE(B.W); if(!B.model) return 1; }
    else { B.maxMB=*std::max_element(GPU_LADDER[b].begin(),GPU_LADDER[b].end());
      NSString* dir=[recBase stringByAppendingPathComponent:[NSString stringWithFormat:@"rec_b%d",B.W]];
      for(int mb : GPU_LADDER[b]){ GBatch bt; bt.MB=mb; RecIO io; MPSGraph* g=loadG(dir,mb,io,REC_FP16);
        MPSGraphTensor* idxT=[g reductionArgMaximumWithTensor:io.output axis:2 name:nil];
        MPSGraphTensor* maxT=[g reductionMaximumWithTensor:io.output axis:2 name:nil]; B.RT=[idxT.shape[1] longValue];
        bt.exe=[g compileWithDevice:gdev feeds:@{io.input:[MPSGraphShapedType.alloc initWithShape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32]} targetTensors:@[idxT,maxT] targetOperations:nil compilationDescriptor:nil];
        // TDs bound below after buffers exist
        B.gbatches.push_back(bt); }
      std::sort(B.gbatches.begin(),B.gbatches.end(),[](const GBatch&x,const GBatch&y){return x.MB<y.MB;}); }
    B.crops=BUF((size_t)B.maxMB*3*RECH*B.W*4); B.idx=BUF((size_t)B.maxMB*B.RT*4); B.mx=BUF((size_t)B.maxMB*B.RT*4);
    B.H=BUF((size_t)B.maxMB*9*4); B.cw=BUF((size_t)B.maxMB*4);
    uint32_t rd[4]={(uint32_t)B.maxMB,3,(uint32_t)RECH,(uint32_t)B.W}; B.rdim=[dev newBufferWithBytes:rd length:16 options:0];
    if(!B.ane) for(auto& bt:B.gbatches){ int mb=bt.MB;
      NSString* dir=[recBase stringByAppendingPathComponent:[NSString stringWithFormat:@"rec_b%d",B.W]]; (void)dir;
      bt.cropsTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.crops shape:@[@(mb),@3,@(RECH),@(B.W)] dataType:MPSDataTypeFloat32];
      bt.idxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.idx shape:@[@(mb),@(B.RT),@1] dataType:MPSDataTypeInt32];
      bt.maxTD=[[MPSGraphTensorData alloc] initWithMTLBuffer:B.mx shape:@[@(mb),@(B.RT),@1] dataType:MPSDataTypeFloat32]; }
  }
  auto pickG=[&](Bk& B,int nb)->GBatch*{ for(auto& bt:B.gbatches) if(bt.MB>=nb) return &bt; return &B.gbatches.back(); };
  auto pickEnum=[&](Bk& B,int nb)->int{ for(int m:B.enums) if(m>=nb) return m; return B.enums.back(); };

  // ANE predict: wrap B.crops (first enumB rows) as MLMultiArray, run, fill idx/score into out
  auto anePredict=[&](Bk& B,int count,std::vector<int32_t>& outIdx,std::vector<float>& outScore){
    int eb=pickEnum(B,count); NSError* err=nil;
    NSArray* shp=@[@(eb),@3,@(RECH),@(B.W)]; NSArray* strd=@[@(3*RECH*B.W),@(RECH*B.W),@(B.W),@1];
    MLMultiArray* arr=[[MLMultiArray alloc] initWithDataPointer:B.crops.contents shape:shp dataType:MLMultiArrayDataTypeFloat32 strides:strd deallocator:nil error:&err];
    MLDictionaryFeatureProvider* fpv=[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:arr]} error:&err];
    id<MLFeatureProvider> o=[B.model predictionFromFeatures:fpv error:&err];
    if(!o){ std::fprintf(stderr,"ANE predict W%d eb%d: %s\n",B.W,eb,err.localizedDescription.UTF8String); return; }
    MLMultiArray* idxA=[[o featureValueForName:@"var_656"] multiArrayValue];
    MLMultiArray* scA =[[o featureValueForName:@"reduce_max_0"] multiArrayValue];
    long T=[idxA.shape[1] longValue];
    outIdx.resize((size_t)count*T); outScore.resize((size_t)count*T);
    // getBytesWithHandler gives a synced, CONTIGUOUS copy — row stride = size/(4*rows).
    [idxA getBytesWithHandler:^(const void* bytes, NSInteger size){ const int32_t* ip=(const int32_t*)bytes;
      long rows=[idxA.shape[0] longValue], rs=(size/4)/std::max(rows,1L);
      for(int i=0;i<count;i++) for(long t=0;t<T;t++) outIdx[(size_t)i*T+t]=ip[i*rs+t]; }];
    [scA getBytesWithHandler:^(const void* bytes, NSInteger size){ const float* sp=(const float*)bytes;
      long rows=[scA.shape[0] longValue], rs=(size/4)/std::max(rows,1L);
      for(int i=0;i<count;i++) for(long t=0;t<T;t++) outScore[(size_t)i*T+t]=sp[i*rs+t]; }];
  };

  // warm ANE + GPU
  { for(int b=0;b<NBK;b++) if(bk[b].ane){ std::vector<int32_t> oi; std::vector<float> os; anePredict(bk[b],1,oi,os); } }
  { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil];
    for(int b=0;b<NBK;b++) if(!bk[b].ane) for(auto& bt:bk[b].gbatches)[bt.exe encodeToCommandBuffer:cb inputsArray:@[bt.cropsTD] resultsArray:@[bt.idxTD,bt.maxTD] executionDescriptor:nil];
    [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }

  std::vector<cv::Point> sbuf; cv::Mat mbuf; std::vector<std::vector<cv::Point>> cbuf; std::vector<cv::Vec4i> hbuf;
  std::vector<std::vector<std::string>> allWords(Nimg);
  double sum_wall=0,a_gpu=0,a_ane=0;
  const bool OVERLAP = !(getenv("HYB_SEQ"));

  // ANE predict from a HOST contiguous crop buffer [count,3,48,W] (for cross-image batching).
  auto anePredictHost=[&](Bk& B,const float* data,int count,std::vector<int32_t>& outIdx,std::vector<float>& outScore){
    int eb=pickEnum(B,count); NSError* err=nil;
    NSArray* shp=@[@(eb),@3,@(RECH),@(B.W)]; NSArray* strd=@[@(3*RECH*B.W),@(RECH*B.W),@(B.W),@1];
    // eb may exceed count; the model needs eb rows — point at a padded buffer if needed
    static thread_local std::vector<float> pad; const float* src=data;
    if(eb>count){ pad.assign((size_t)eb*3*RECH*B.W,0.0f); std::copy(data,data+(size_t)count*3*RECH*B.W,pad.begin()); src=pad.data(); }
    MLMultiArray* arr=[[MLMultiArray alloc] initWithDataPointer:(void*)src shape:shp dataType:MLMultiArrayDataTypeFloat32 strides:strd deallocator:nil error:&err];
    MLDictionaryFeatureProvider* fpv=[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:arr]} error:&err];
    id<MLFeatureProvider> o=[B.model predictionFromFeatures:fpv error:&err];
    if(!o){ std::fprintf(stderr,"ANE host predict W%d: %s\n",B.W,err.localizedDescription.UTF8String); return; }
    MLMultiArray* idxA=[[o featureValueForName:@"var_656"] multiArrayValue];
    MLMultiArray* scA =[[o featureValueForName:@"reduce_max_0"] multiArrayValue];
    long T=[idxA.shape[1] longValue];
    outIdx.resize((size_t)count*T); outScore.resize((size_t)count*T);
    [idxA getBytesWithHandler:^(const void* bytes, NSInteger size){ const int32_t* ip=(const int32_t*)bytes; long rows=[idxA.shape[0] longValue],rs=(size/4)/std::max(rows,1L); for(int i=0;i<count;i++) for(long t=0;t<T;t++) outIdx[(size_t)i*T+t]=ip[i*rs+t]; }];
    [scA getBytesWithHandler:^(const void* bytes, NSInteger size){ const float* sp=(const float*)bytes; long rows=[scA.shape[0] longValue],rs=(size/4)/std::max(rows,1L); for(int i=0;i<count;i++) for(long t=0;t<T;t++) outScore[(size_t)i*T+t]=sp[i*rs+t]; }];
  };

  // ANE bucket indices
  std::vector<int> aneB; for(int b=0;b<NBK;b++) if(bk[b].ane) aneB.push_back(b);
  // producer-consumer for overlap
  struct Payload { int img; std::vector<std::vector<float>> crops; std::vector<int> cnt; };
  std::deque<Payload> queue; std::mutex mu; std::condition_variable cvq; bool prodDone=false;
  std::vector<std::vector<std::string>> gpuWords(Nimg), aneWords(Nimg);

  std::thread aneThread;
  if(OVERLAP) aneThread=std::thread([&]{ @autoreleasepool{
    while(true){ std::vector<Payload> batch;
      { std::unique_lock<std::mutex> lk(mu); cvq.wait(lk,[&]{return !queue.empty()||prodDone;});
        if(queue.empty()&&prodDone) break;
        while(!queue.empty()){ batch.push_back(std::move(queue.front())); queue.pop_front(); } }
      for(size_t ai=0; ai<aneB.size(); ai++){ Bk& B=bk[aneB[ai]]; int stride=3*RECH*B.W;
        std::vector<float> big; std::vector<int> imgOf;
        for(auto& p:batch){ int c=p.cnt[ai]; if(!c)continue; big.insert(big.end(),p.crops[ai].begin(),p.crops[ai].end()); for(int i=0;i<c;i++) imgOf.push_back(p.img); }
        int total=(int)imgOf.size(); int emax=B.enums.back();
        for(int off=0; off<total; ){ int cnt=std::min(emax,total-off);
          std::vector<int32_t> oi; std::vector<float> os; @autoreleasepool{ anePredictHost(B, big.data()+(size_t)off*stride, cnt, oi, os); }
          for(int i=0;i<cnt;i++){ auto t=turbo_ocr::recognition::ctc_greedy_decode((const int*)(oi.data()+(size_t)i*B.RT),os.data()+(size_t)i*B.RT,(int)B.RT,labels);
            if(!t.first.empty()) aneWords[imgOf[off+i]].push_back(t.first); }
          off+=cnt; }
      }
    }
  }});

  // Pre-decode all images + pre-create textures ONCE (removes per-pixel RGBA
  // repack + PNG decode from the timed loop — measures the OCR pipeline itself).
  std::vector<cv::Mat> imgs(Nimg); std::vector<id<MTLTexture>> tex(Nimg);
  for(int i=0;i<Nimg;i++){ char pth[512]; std::snprintf(pth,sizeof pth,"%s/funsd_%03d.png",cache.c_str(),i);
    imgs[i]=cv::imread(pth,cv::IMREAD_COLOR); if(imgs[i].empty()){ std::fprintf(stderr,"read %s\n",pth); return 1; }
    int ow=imgs[i].cols,oh=imgs[i].rows;
    MTLTextureDescriptor* td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:ow height:oh mipmapped:NO]; td.usage=MTLTextureUsageShaderRead;
    tex[i]=[dev newTextureWithDescriptor:td];
    std::vector<uint8_t> px((size_t)ow*oh*4); for(int y=0;y<oh;y++){const cv::Vec3b* r=imgs[i].ptr<cv::Vec3b>(y); for(int x=0;x<ow;x++){size_t k=((size_t)y*ow+x)*4; px[k]=r[x][2];px[k+1]=r[x][1];px[k+2]=r[x][0];px[k+3]=255;}}
    [tex[i] replaceRegion:MTLRegionMake2D(0,0,ow,oh) mipmapLevel:0 withBytes:px.data() bytesPerRow:ow*4]; }

  const int REPEAT=getenv("REPEAT")?atoi(getenv("REPEAT")):1;  // loop the set REPEAT times (for concurrent-throughput measurement)
  auto t_all=clk::now();
  for(int rep=0; rep<REPEAT; ++rep)
  for(int im=(rep?0:-1); im<Nimg; ++im){ @autoreleasepool {
    int pidx=im<0?0:im;
    cv::Mat& orig=imgs[pidx]; id<MTLTexture> srcTex=tex[pidx];
    int ow=orig.cols,oh=orig.rows; auto t0=clk::now();
    { cv::Mat di; cv::resize(orig,di,cv::Size(SZW,SZH)); cv::Mat bgr[3]; cv::split(di,bgr); const float m[3]={0.485f,0.456f,0.406f},sd[3]={0.229f,0.224f,0.225f};
      for(int c=0;c<3;c++){cv::Mat pp(SZH,SZW,CV_32F,detBuf+(size_t)c*N); bgr[c].convertTo(pp,CV_32F,1.0/(255.0*sd[c]),-m[c]/sd[c]);} }
    if(DET_COREML){ runDetCoreML(); }
    else { MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
      [detExe encodeToCommandBuffer:cb inputsArray:@[detTD] resultsArray:@[probTD] executionDescriptor:nil];
      [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted]; }
    cv::Mat pred(SZH,SZW,CV_32F,probBuf.contents), bmt; cv::threshold(pred,bmt,THRESH,255,cv::THRESH_BINARY); bmt.convertTo(bmt,CV_8U);
    std::vector<Box> boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bmt,oh,ow,SZH,SZW,BOXTHRESH,RATIO,3.0f,5.0f,sbuf,mbuf,cbuf,hbuf);
    std::vector<std::vector<int>> lists(NBK);
    for(int i=0;i<(int)boxes.size();i++){ int nat=turbo_ocr::recognition::rec_input_width(boxes[i],RECH); int b=NBK-1; for(int k=0;k<NBK;k++) if(nat<=BW[k]){b=k;break;} lists[b].push_back(i); }
    int nb[NBK]={0}; GBatch* chosen[NBK]={nullptr};
    for(int b=0;b<NBK;b++){ if(lists[b].empty())continue; Bk& B=bk[b];
      int n=(int)std::min((size_t)B.maxMB,lists[b].size()); nb[b]=n; if(!B.ane) chosen[b]=pickG(B,n);
      float* Hm=(float*)B.H.contents; int32_t* cwp=(int32_t*)B.cw.contents;
      for(int i=0;i<n;i++){ auto ct=turbo_ocr::compute_crop_transform(boxes[lists[b][i]],RECH,B.W); for(int k=0;k<9;k++)Hm[i*9+k]=ct.M_inv[k]; cwp[i]=std::min(ct.crop_width,B.W); } }
    // GPU command buffer: warp ALL buckets + rec the GPU (wide) buckets
    auto tg=clk::now();
    MPSCommandBuffer* cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    for(int b=0;b<NBK;b++){ if(!nb[b])continue; Bk& B=bk[b];
      { id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder]; [ce setComputePipelineState:warp]; [ce setTexture:srcTex atIndex:0];
        [ce setBuffer:B.crops offset:0 atIndex:0];[ce setBuffer:B.H offset:0 atIndex:1];[ce setBuffer:B.rdim offset:0 atIndex:2];[ce setBuffer:B.cw offset:0 atIndex:3];
        [ce dispatchThreads:MTLSizeMake(B.W,RECH,nb[b]) threadsPerThreadgroup:MTLSizeMake(16,8,1)];[ce endEncoding]; }
      if(!B.ane){ GBatch* g=chosen[b]; [g->exe encodeToCommandBuffer:cb inputsArray:@[g->cropsTD] resultsArray:@[g->idxTD,g->maxTD] executionDescriptor:nil]; }
    }
    [cb.rootCommandBuffer commit]; [cb.rootCommandBuffer waitUntilCompleted];
    double tgpu=ms(tg);
    // GPU thread: decode WIDE (GPU) buckets inline; hand NARROW crops to the ANE thread.
    auto ta=clk::now();
    std::vector<std::string> gwords;
    for(int b=0;b<NBK;b++){ if(!nb[b])continue; Bk& B=bk[b];
      if(!B.ane){ const int32_t* idx=(const int32_t*)B.idx.contents; const float* sc=(const float*)B.mx.contents;
        for(int i=0;i<nb[b];i++){ auto t=turbo_ocr::recognition::ctc_greedy_decode((const int*)(idx+(size_t)i*B.RT),sc+(size_t)i*B.RT,(int)B.RT,labels); if(!t.first.empty())gwords.push_back(t.first); } } }
    if(im>=0) gpuWords[im]=std::move(gwords);
    if(OVERLAP){
      if(im>=0){ Payload p; p.img=im; p.crops.resize(aneB.size()); p.cnt.resize(aneB.size());
        for(size_t ai=0;ai<aneB.size();ai++){ Bk& B=bk[aneB[ai]]; int c=nb[aneB[ai]]; p.cnt[ai]=c;
          if(c){ p.crops[ai].assign((float*)B.crops.contents,(float*)B.crops.contents+(size_t)c*3*RECH*B.W); } }
        { std::lock_guard<std::mutex> lk(mu); queue.push_back(std::move(p)); } cvq.notify_one(); }
    } else { // sequential ANE (correctness/debug)
      std::vector<std::string> aw;
      for(size_t ai=0;ai<aneB.size();ai++){ Bk& B=bk[aneB[ai]]; int c=nb[aneB[ai]]; if(!c)continue;
        std::vector<int32_t> oi; std::vector<float> os; anePredict(B,c,oi,os);
        for(int i=0;i<c;i++){ auto t=turbo_ocr::recognition::ctc_greedy_decode((const int*)(oi.data()+(size_t)i*B.RT),os.data()+(size_t)i*B.RT,(int)B.RT,labels); if(!t.first.empty())aw.push_back(t.first); } }
      if(im>=0) aneWords[im]=std::move(aw);
    }
    double tane=ms(ta);
    if(im>=0){ sum_wall+=ms(t0); a_gpu+=tgpu; a_ane+=tane; }
  }}
  if(OVERLAP){ { std::lock_guard<std::mutex> lk(mu); prodDone=true; } cvq.notify_one(); aneThread.join(); }
  for(int i=0;i<Nimg;i++){ allWords[i]=std::move(gpuWords[i]); for(auto& w:aneWords[i]) allWords[i].push_back(std::move(w)); }
  double wall=ms(t_all);

  FILE* f=std::fopen(outPath.c_str(),"w"); std::fputc('[',f);
  for(int i=0;i<Nimg;i++){ std::fputc('[',f);
    for(size_t k=0;k<allWords[i].size();k++) std::fprintf(f,"\"%s\"%s",jesc(allWords[i][k]).c_str(),k+1<allWords[i].size()?",":"");
    std::fprintf(f,"]%s",i+1<Nimg?",":""); }
  std::fputc(']',f); std::fclose(f);
  std::printf("HYBRID N=%d ANE_MAXW=%d\n",Nimg,ANE_MAXW);
  std::printf("  GPU-thread/img=%.1fms (=> %.0f img/s GPU-bound)  |  TRUE end-to-end=%.1fms/img => %.0f img/s\n",
    sum_wall/(Nimg*REPEAT),1000.0*Nimg*REPEAT/sum_wall, wall/(Nimg*REPEAT),1000.0*Nimg*REPEAT/wall);
  std::printf("  of GPU-thread: warp+wide-rec GPU cmdbuf=%.1fms/img, CPU rest (det-prep+DBpost+homo+decode+payload-copy)=%.1fms/img\n",
    a_gpu/Nimg, (sum_wall-a_gpu)/Nimg);
  std::printf("  wrote %s\n",outPath.c_str());
  return 0;
}}
