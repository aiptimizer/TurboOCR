// Concurrent CoreML OCR harness for the SMALL/MEDIUM tiers (SVTR-transformer rec).
// Full pipeline entirely on GPU+ANE via CoreML (no CPU inference of the nets):
//   CoreML det (GPU) -> host DB-post -> CPU warp -> batched CoreML rec
//   (narrow buckets 320/480 on ANE, wide 800/1600 on GPU) -> CTC.
// CONC worker threads, each with its OWN MLModel instances, process the image set
// REPEAT times; aggregate throughput = total_images / wall (concurrent metric).
//
// Run: mps_ocr_coreml_conc <cache> <N> <out.json> <tier small|medium> <coreml_dir> <keys>
//   env: CONC (threads, default 1), REPEAT (default 1)
#import <CoreML/CoreML.h>
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
#include <atomic>
#include <algorithm>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
using turbo_ocr::Box;

static std::string jesc(const std::string& s){ std::string o; o.reserve(s.size()+2);
  for(char c: s){ switch(c){ case '"':o+="\\\"";break; case '\\':o+="\\\\";break; case '\n':o+="\\n";break;
    case '\r':o+="\\r";break; case '\t':o+="\\t";break;
    default: if((unsigned char)c<0x20){char b[8];std::snprintf(b,sizeof b,"\\u%04x",c);o+=b;} else o+=c; } } return o; }

// ---- bucket definitions (W, fixed batch, engine) ----
struct BkDef { int W, MB; bool ane; };
static const BkDef BK[] = { {320,48,true}, {480,24,true}, {800,16,false}, {1600,24,false} };
static const int NBK = 4;
static const int RECH = 48, SZH = 992, SZW = 800;
static const size_t DETN = (size_t)SZH*SZW;

// Compile an .mlpackage once -> compiled (.mlmodelc) URL.
static NSURL* compilePkg(NSString* path){ NSError* e=nil;
  NSURL* c=[MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&e];
  if(!c) std::fprintf(stderr,"compile %s: %s\n",path.UTF8String,e.localizedDescription.UTF8String);
  return c; }
static MLModel* loadCompiled(NSURL* c, MLComputeUnits units){ NSError* e=nil;
  MLModelConfiguration* cfg=[[MLModelConfiguration alloc] init]; cfg.computeUnits=units;
  MLModel* m=[MLModel modelWithContentsOfURL:c configuration:cfg error:&e];
  if(!m) std::fprintf(stderr,"load model: %s\n",e.localizedDescription.UTF8String);
  return m; }

// Discover output feature names: classify by dtype. Returns idx(int32) + score(float) names,
// or the single output name for det.
static void outNames(MLModel* m, NSString** idxName, NSString** scoreName, NSString** single){
  NSDictionary<NSString*,MLFeatureDescription*>* od=m.modelDescription.outputDescriptionsByName;
  for(NSString* k in od){ MLFeatureDescription* d=od[k];
    MLMultiArrayDataType dt=d.multiArrayConstraint.dataType;
    if(single) *single=k;
    if(dt==MLMultiArrayDataTypeInt32){ if(idxName)*idxName=k; }
    else { if(scoreName)*scoreName=k; } } }

// Read a CoreML MLMultiArray element (fp16/fp32/int32) at flat (i,t) using its strides.
// Fills a contiguous host [rows*T] fp32 (score) / int32 (idx).
static long readIdxScore(id<MLFeatureProvider> out, NSString* idxName, NSString* scoreName,
                         int count, std::vector<int32_t>& oi, std::vector<float>& os){
  MLMultiArray* idxA=[[out featureValueForName:idxName] multiArrayValue];
  MLMultiArray* scA =[[out featureValueForName:scoreName] multiArrayValue];
  long T=[idxA.shape[1] longValue];
  long is0=[idxA.strides[0] longValue], is1=[idxA.strides[1] longValue];
  long ss0=[scA.strides[0] longValue],  ss1=[scA.strides[1] longValue];
  MLMultiArrayDataType sdt=scA.dataType;
  oi.resize((size_t)count*T); os.resize((size_t)count*T);
  [idxA getBytesWithHandler:^(const void* b, NSInteger){ const int32_t* p=(const int32_t*)b;
    for(int i=0;i<count;i++) for(long t=0;t<T;t++) oi[(size_t)i*T+t]=p[i*is0+t*is1]; }];
  [scA getBytesWithHandler:^(const void* b, NSInteger){
    if(sdt==MLMultiArrayDataTypeFloat16){ const __fp16* p=(const __fp16*)b;
      for(int i=0;i<count;i++) for(long t=0;t<T;t++) os[(size_t)i*T+t]=(float)p[i*ss0+t*ss1]; }
    else { const float* p=(const float*)b;
      for(int i=0;i<count;i++) for(long t=0;t<T;t++) os[(size_t)i*T+t]=p[i*ss0+t*ss1]; } }];
  return T;
}

int main(int argc,char** argv){ @autoreleasepool{
  if(argc<7){ std::fprintf(stderr,"usage: %s <cache> <N> <out.json> <tier> <coreml_dir> <keys>\n",argv[0]); return 2; }
  const std::string cache=argv[1]; const int Nimg=atoi(argv[2]); const std::string outPath=argv[3];
  const std::string tier=argv[4]; NSString* cml=[NSString stringWithUTF8String:argv[5]]; const char* keys=argv[6];
  const int CONC=getenv("CONC")?atoi(getenv("CONC")):1;
  const int REPEAT=getenv("REPEAT")?atoi(getenv("REPEAT")):1;
  const float THRESH=0.2f, BOXTHRESH=0.45f, RATIO=1.4f;
  std::vector<std::string> labels={"blank"}; turbo_ocr::recognition::load_label_dict(keys,labels);

  // ---- compile models ONCE (shared compiled artifacts; per-thread MLModel instances) ----
  NSURL* detC=compilePkg([cml stringByAppendingPathComponent:
      [NSString stringWithFormat:@"det_%s_gpu_992x800.mlpackage",tier.c_str()]]);
  if(!detC) return 1;
  NSURL* recC[NBK];
  for(int b=0;b<NBK;b++){ recC[b]=compilePkg([cml stringByAppendingPathComponent:
      [NSString stringWithFormat:@"rec_%s_%s_%d_b%d.mlpackage",tier.c_str(),BK[b].ane?"ane":"gpu",BK[b].W,BK[b].MB]]);
    if(!recC[b]) return 1; }

  // discover feature names (same across instances)
  NSString* detOut=nil; { MLModel* dm=loadCompiled(detC,MLComputeUnitsCPUAndGPU); outNames(dm,nil,nil,&detOut); }
  NSString* recIdx[NBK]; NSString* recScore[NBK];
  for(int b=0;b<NBK;b++){ NSString* ix=nil; NSString* sc=nil;
    MLModel* rm=loadCompiled(recC[b], BK[b].ane?MLComputeUnitsCPUAndNeuralEngine:MLComputeUnitsCPUAndGPU);
    outNames(rm,&ix,&sc,nil); recIdx[b]=ix; recScore[b]=sc; }

  // ---- load all images once ----
  std::vector<cv::Mat> imgs(Nimg);
  for(int i=0;i<Nimg;i++){ char p[512]; std::snprintf(p,sizeof p,"%s/funsd_%03d.png",cache.c_str(),i);
    imgs[i]=cv::imread(p,cv::IMREAD_COLOR); if(imgs[i].empty()){ std::fprintf(stderr,"read %s\n",p); return 1; } }

  // ---- per-thread worker context ----
  struct Ctx {
    MLModel* det; MLModel* rec[NBK];
    std::vector<float> detIn;                  // [3*992*800]
    std::vector<float> probF32;                // [992*800]
    std::vector<float> recBuf[NBK];            // [MB*3*48*W]
    std::vector<cv::Point> sbuf; cv::Mat mbuf; std::vector<std::vector<cv::Point>> cbuf; std::vector<cv::Vec4i> hbuf;
  };
  auto makeCtx=[&](Ctx& c){
    c.det=loadCompiled(detC,MLComputeUnitsCPUAndGPU);
    for(int b=0;b<NBK;b++){ c.rec[b]=loadCompiled(recC[b], BK[b].ane?MLComputeUnitsCPUAndNeuralEngine:MLComputeUnitsCPUAndGPU);
      c.recBuf[b].assign((size_t)BK[b].MB*3*RECH*BK[b].W,0.0f); }
    c.detIn.assign(3*DETN,0.0f); c.probF32.assign(DETN,0.0f);
  };

  // run ONE image through the full pipeline; append recognized words.
  auto runImage=[&](Ctx& c, const cv::Mat& orig, std::vector<std::string>& words){
    int ow=orig.cols, oh=orig.rows;
    // det preprocess (BGR imagenet)
    { cv::Mat di; cv::resize(orig,di,cv::Size(SZW,SZH)); cv::Mat bgr[3]; cv::split(di,bgr);
      const float m[3]={0.485f,0.456f,0.406f}, sd[3]={0.229f,0.224f,0.225f};
      for(int ch=0;ch<3;ch++){ cv::Mat pp(SZH,SZW,CV_32F,c.detIn.data()+(size_t)ch*DETN);
        bgr[ch].convertTo(pp,CV_32F,1.0/(255.0*sd[ch]),-m[ch]/sd[ch]); } }
    // det predict (CoreML GPU)
    NSError* err=nil;
    NSArray* dsh=@[@1,@3,@(SZH),@(SZW)]; NSArray* dst=@[@(3*DETN),@(DETN),@(SZW),@1];
    MLMultiArray* din=[[MLMultiArray alloc] initWithDataPointer:c.detIn.data() shape:dsh
        dataType:MLMultiArrayDataTypeFloat32 strides:dst deallocator:nil error:&err];
    MLDictionaryFeatureProvider* dfp=[[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:din]} error:&err];
    id<MLFeatureProvider> dout=[c.det predictionFromFeatures:dfp error:&err];
    if(!dout){ std::fprintf(stderr,"det predict: %s\n",err.localizedDescription.UTF8String); return; }
    MLMultiArray* pm=[[dout featureValueForName:detOut] multiArrayValue];
    { long s2=[pm.strides[2] longValue], s3=[pm.strides[3] longValue]; MLMultiArrayDataType dt=pm.dataType;
      [pm getBytesWithHandler:^(const void* b, NSInteger){
        if(dt==MLMultiArrayDataTypeFloat16){ const __fp16* p=(const __fp16*)b;
          for(int y=0;y<SZH;y++) for(int x=0;x<SZW;x++) c.probF32[(size_t)y*SZW+x]=(float)p[y*s2+x*s3]; }
        else { const float* p=(const float*)b;
          for(int y=0;y<SZH;y++) for(int x=0;x<SZW;x++) c.probF32[(size_t)y*SZW+x]=p[y*s2+x*s3]; } }]; }
    // DB post
    cv::Mat pred(SZH,SZW,CV_32F,c.probF32.data()), bmt; cv::threshold(pred,bmt,THRESH,255,cv::THRESH_BINARY); bmt.convertTo(bmt,CV_8U);
    std::vector<Box> boxes=turbo_ocr::detection::extract_boxes_from_bitmap(pred,bmt,oh,ow,SZH,SZW,BOXTHRESH,RATIO,3.0f,5.0f,c.sbuf,c.mbuf,c.cbuf,c.hbuf);
    // bucket by rec width
    std::vector<std::vector<int>> lists(NBK);
    for(int i=0;i<(int)boxes.size();i++){ int nat=turbo_ocr::recognition::rec_input_width(boxes[i],RECH);
      int b=NBK-1; for(int k=0;k<NBK;k++) if(nat<=BK[k].W){ b=k; break; } lists[b].push_back(i); }
    // per bucket: warp -> chunked batched rec -> CTC
    for(int b=0;b<NBK;b++){ if(lists[b].empty()) continue; int W=BK[b].W, MB=BK[b].MB;
      const std::vector<int>& L=lists[b];
      for(size_t off=0; off<L.size(); off+=MB){ int cnt=(int)std::min((size_t)MB,L.size()-off);
        // fill batch buffer [MB,3,48,W]; zero unused rows/pad
        std::fill(c.recBuf[b].begin(),c.recBuf[b].end(),0.0f);
        for(int r=0;r<cnt;r++){ const Box& bx=boxes[L[off+r]];
          auto ctf=turbo_ocr::compute_crop_transform(bx,RECH,W);
          cv::Mat Minv(3,3,CV_32F); for(int k=0;k<9;k++) Minv.at<float>(k/3,k%3)=ctf.M_inv[k];
          cv::Mat crop; cv::warpPerspective(orig,crop,Minv,cv::Size(W,RECH),
              cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
          int cw=std::min(ctf.crop_width,W);
          float* base=c.recBuf[b].data()+(size_t)r*3*RECH*W;
          for(int y=0;y<RECH;y++){ const cv::Vec3b* rp=crop.ptr<cv::Vec3b>(y);
            for(int x=0;x<W;x++){ float R=0,G=0,B=0;
              if(x<cw){ R=rp[x][2]/127.5f-1.0f; G=rp[x][1]/127.5f-1.0f; B=rp[x][0]/127.5f-1.0f; }
              base[0*RECH*W + y*W + x]=R; base[1*RECH*W + y*W + x]=G; base[2*RECH*W + y*W + x]=B; } } }
        // predict
        NSArray* rsh=@[@(MB),@3,@(RECH),@(W)]; NSArray* rst=@[@(3*RECH*W),@(RECH*W),@(W),@1];
        NSError* e2=nil;
        MLMultiArray* rin=[[MLMultiArray alloc] initWithDataPointer:c.recBuf[b].data() shape:rsh
            dataType:MLMultiArrayDataTypeFloat32 strides:rst deallocator:nil error:&e2];
        MLDictionaryFeatureProvider* rfp=[[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:rin]} error:&e2];
        id<MLFeatureProvider> rout=[c.rec[b] predictionFromFeatures:rfp error:&e2];
        if(!rout){ std::fprintf(stderr,"rec b%d predict: %s\n",b,e2.localizedDescription.UTF8String); continue; }
        std::vector<int32_t> oi; std::vector<float> os;
        long T=readIdxScore(rout,recIdx[b],recScore[b],cnt,oi,os);
        for(int r=0;r<cnt;r++){ auto t=turbo_ocr::recognition::ctc_greedy_decode(
            (const int*)(oi.data()+(size_t)r*T), os.data()+(size_t)r*T, (int)T, labels);
          if(!t.first.empty()) words.push_back(t.first); }
      }
    }
  };

  std::vector<std::vector<std::string>> allWords(Nimg);
  // ---- warm each thread's models on image 0 ----
  std::vector<Ctx> ctxs(CONC);
  for(int t=0;t<CONC;t++){ makeCtx(ctxs[t]); std::vector<std::string> w; runImage(ctxs[t],imgs[0],w); }

  // ---- timed concurrent run: each thread does the whole set REPEAT times ----
  std::atomic<long> doneImgs{0};
  auto t_all=clk::now();
  std::vector<std::thread> pool;
  for(int t=0;t<CONC;t++) pool.emplace_back([&,t]{ @autoreleasepool{
    for(int rep=0; rep<REPEAT; ++rep){
      for(int i=0;i<Nimg;i++){ @autoreleasepool{
        std::vector<std::string> w; runImage(ctxs[t],imgs[i],w);
        if(t==0 && rep==0) allWords[i]=std::move(w);
        doneImgs.fetch_add(1,std::memory_order_relaxed);
      }}
    }
  }});
  for(auto& th:pool) th.join();
  double wall=ms(t_all);
  long total=doneImgs.load();

  // ---- write words JSON from thread 0's clean first pass ----
  FILE* f=std::fopen(outPath.c_str(),"w"); std::fputc('[',f);
  for(int i=0;i<Nimg;i++){ std::fputc('[',f);
    for(size_t k=0;k<allWords[i].size();k++) std::fprintf(f,"\"%s\"%s",jesc(allWords[i][k]).c_str(),k+1<allWords[i].size()?",":"");
    std::fprintf(f,"]%s",i+1<Nimg?",":""); }
  std::fputc(']',f); std::fclose(f);

  std::printf("CoreML-conc tier=%s CONC=%d REPEAT=%d  images=%ld in %.0f ms => %.1f img/s (aggregate)\n",
      tier.c_str(),CONC,REPEAT,total,wall,1000.0*total/wall);
  std::printf("  per-stream ~ %.1f img/s ; wrote %s\n", 1000.0*total/wall/std::max(CONC,1), outPath.c_str());
  return 0;
}}
