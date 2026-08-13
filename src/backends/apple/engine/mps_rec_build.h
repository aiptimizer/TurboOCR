// Shared: rebuild rec_tiny (PP-OCRv6 fully-conv CTC) in MPSGraph from the Python
// export (graph.json + weights.bin), wiring by ONNX tensor name.
//
// Two classes of caller:
//   * LIBRARY — turbo_ocr::apple::MpsEngine (src/backends/apple/engine/mps_engine.mm,
//     `#include "apple/engine/mps_rec_build.h"` off -Isrc/backends). This is the server
//     path and the reason the header lives here rather than in tools/.
//   * PROBES — the standalone tools/probes/apple/mps_*.mm harnesses (v0 correctness/
//     throughput, v1 GPU-resident warp->rec, the FUNSD runners, the DB-post
//     tests). They `#include "mps_rec_build.h"` and pick up the one-line
//     forwarding header tools/probes/apple/mps_rec_build.h, which points back here.
#pragma once
#import <Foundation/Foundation.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct RecIO { MPSGraphTensor* input; MPSGraphTensor* output; std::string inName; std::vector<long> ishape; };

// Translation bail-out: an export that names a tensor we never built, or uses
// an op/attribute this translator does not implement.
//
// THROWS, never exit()s. This header started life inside the standalone
// tools/probes/apple/mps_*.mm probes where killing the process was fine, but it is now
// LIBRARY code linked into the Apple backend of the server: a rec export with
// one unhandled op would take the whole process down mid-request, with no way
// for the caller to fall back. Throwing lets MpsEngine::prepare() report a
// failed graph build through the same `return false` every other load failure
// uses (MpsRecognizer::load then declines the stage).
//
// PROBE-VISIBLE CHANGE, stated honestly: no tools/probes/apple/mps_*.mm probe catches this,
// so they still stop right here — but via std::terminate, i.e. SIGABRT / shell
// status 134, where they previously exited 2 (MISSING tensor) or 3 (Slice step
// unsupported / UNHANDLED op). The message TEXT is unchanged (libc++'s
// terminate handler prints what() after its "terminating due to an uncaught
// exception" banner), but the two codes are no longer distinguishable: any
// sweep script branching on `$? == 2` vs `$? == 3` must be re-taught. Do NOT
// reintroduce exit() here — if a probe needs the old codes back, give mrb_fail
// its own exception type carrying an int and catch it in that probe's main().
[[noreturn]] static inline void mrb_fail(const std::string& msg){
  throw std::runtime_error(msg);
}

static inline NSArray<NSNumber*>* mrb_nums(const std::vector<long>& v){
  NSMutableArray* a=[NSMutableArray array]; for(long x:v)[a addObject:@(x)]; return a;
}
static inline std::vector<long> mrb_ints(id v){
  std::vector<long> r; for(NSNumber* n in (NSArray*)v) r.push_back(n.longValue); return r;
}

// Build rec_tiny into `g` from parsed graph.json `G` and raw fp32 weights `wbase`,
// with batch dimension `B`. Returns the input placeholder + output tensor.
static inline RecIO buildRecGraph(MPSGraph* g, NSDictionary* G, const float* wbase, int B, bool fp16=false){
  const MPSDataType SDT = fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;  // scalar/intermediate dtype
  std::unordered_map<std::string, MPSGraphTensor*> T;
  std::unordered_map<std::string, std::vector<long>> wshape;
  std::unordered_map<std::string, size_t> woff;  // name -> float offset into wbase

  for(NSDictionary* w in G[@"initializers"]){
    std::string name=[w[@"name"] UTF8String];
    size_t off=[w[@"offset"] unsignedLongValue]; size_t nb=[w[@"nbytes"] unsignedLongValue];
    std::vector<long> shp=mrb_ints(w[@"shape"]); wshape[name]=shp; woff[name]=off/4;
    NSArray<NSNumber*>* s = shp.empty()? @[@1] : mrb_nums(shp);
    NSData* slice=[NSData dataWithBytesNoCopy:(void*)((const uint8_t*)wbase+off) length:nb freeWhenDone:NO];
    MPSGraphTensor* c=[g constantWithData:slice shape:s dataType:MPSDataTypeFloat32];
    T[name]= fp16 ? [g castTensor:c toType:MPSDataTypeFloat16 name:nil] : c;
  }
  // read a constant initializer's raw float values host-side (for Resize scales)
  auto cvals=[&](id v)->const float*{ auto it=woff.find([v UTF8String]); return it==woff.end()?nullptr:wbase+it->second; };

  // Shape-subgraph evaluator: Shape/Slice/Concat/Reshape that manipulate shapes
  // are constant-folded to host int vectors (static graph), never MPSGraph tensors.
  std::unordered_map<std::string, std::vector<long>> hostInts;  // name -> int vector
  auto asInts=[&](id v)->std::vector<long>{  // resolve a name to its int vector, or {}
    std::string n=[v UTF8String];
    auto h=hostInts.find(n); if(h!=hostInts.end()) return h->second;
    auto w=woff.find(n); if(w!=woff.end()){ auto sh=wshape[n]; long cnt=1; for(long dd:sh)cnt*=dd;
      std::vector<long> r; const float* p=wbase+w->second; for(long i=0;i<cnt;i++)r.push_back((long)llround(p[i])); return r; }
    return {};
  };

  std::string inName=[G[@"input"] UTF8String];
  std::vector<long> ishape=mrb_ints(G[@"input_shape"]); ishape[0]=B;
  MPSGraphTensor* ph=[g placeholderWithShape:mrb_nums(ishape) dataType:MPSDataTypeFloat32 name:@"x"];
  T[inName]= fp16 ? [g castTensor:ph toType:MPSDataTypeFloat16 name:@"xf16"] : ph;  // feed stays FP32; graph runs FP16

  auto get=[&](id v)->MPSGraphTensor*{ std::string n=[v UTF8String];
    auto it=T.find(n); if(it==T.end()) mrb_fail("MISSING tensor '"+n+"'"); return it->second; };

  bool dbg = getenv("MPS_DEBUG");  // pre-commit-allow-getenv (dev-only graph-dump knob, read once at build, not operator config)
  int idx=0;
  auto shp=[&](MPSGraphTensor* tt)->std::string{ if(!tt||!tt.shape)return "(dyn)"; std::string s="["; for(NSNumber* n in tt.shape){s+=std::to_string(n.longValue);s+=",";} s+="]"; return s; };
  for(NSDictionary* n in G[@"nodes"]){
    std::string op=[n[@"op"] UTF8String];
    NSArray* in=n[@"in"]; NSArray* out=n[@"out"]; NSDictionary* at=n[@"attr"];
    MPSGraphTensor* y=nil;
    if(dbg){ std::string s; for(id v in in){ auto it=T.find([v UTF8String]); s+=shp(it!=T.end()?it->second:nil)+" "; } std::fprintf(stderr,"[%d] %-18s in: %s\n",idx,op.c_str(),s.c_str()); }

    // Host-int (shape-subgraph) passthrough. SVTR's transformer blocks route the
    // batch dim through Shape -> Slice -> Squeeze -> Identity -> Unsqueeze ->
    // Concat before it reaches a Reshape target. Those carriers are int vectors,
    // not tensors, so each must forward hostInts or the next op calls get() on a
    // name that was never put in T ("MISSING tensor 'Shape.1'").
    auto hostIn=[&](id v)->bool{ return hostInts.count([v UTF8String])>0; };

    if(op=="Identity"){
      if(hostIn(in[0])) hostInts[[out[0] UTF8String]]=hostInts[[in[0] UTF8String]];
      else y=get(in[0]);
    }
    else if(op=="Relu"){ y=[g reLUWithTensor:get(in[0]) name:nil]; }
    else if(op=="Sub"){ y=[g subtractionWithPrimaryTensor:get(in[0]) secondaryTensor:get(in[1]) name:nil]; }
    else if(op=="Sqrt"){ y=[g squareRootWithTensor:get(in[0]) name:nil]; }
    else if(op=="Pow"){ y=[g powerWithPrimaryTensor:get(in[0]) secondaryTensor:get(in[1]) name:nil]; }
    else if(op=="HardSwish"){
      // ONNX HardSwish: x * max(0, min(1, x/6 + 0.5)) — the fused form of the
      // HardSigmoid(alpha=1/6,beta=0.5) * x pair the older exports emit.
      MPSGraphTensor* x0=get(in[0]);
      MPSGraphTensor* z=[g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:x0 secondaryTensor:[g constantWithScalar:(1.0/6.0) dataType:SDT] name:nil] secondaryTensor:[g constantWithScalar:0.5 dataType:SDT] name:nil];
      MPSGraphTensor* c=[g clampWithTensor:z minValueTensor:[g constantWithScalar:0.0 dataType:SDT] maxValueTensor:[g constantWithScalar:1.0 dataType:SDT] name:nil];
      y=[g multiplicationWithPrimaryTensor:x0 secondaryTensor:c name:nil];
    }
    else if(op=="Add"){ y=[g additionWithPrimaryTensor:get(in[0]) secondaryTensor:get(in[1]) name:nil]; }
    else if(op=="Mul"){ y=[g multiplicationWithPrimaryTensor:get(in[0]) secondaryTensor:get(in[1]) name:nil]; }
    else if(op=="Div"){ y=[g divisionWithPrimaryTensor:get(in[0]) secondaryTensor:get(in[1]) name:nil]; }
    else if(op=="Erf"){ y=[g erfWithTensor:get(in[0]) name:nil]; }
    else if(op=="Softmax"){ long ax=at[@"axis"]?[at[@"axis"] longValue]:-1; y=[g softMaxWithTensor:get(in[0]) axis:ax name:nil]; }
    else if(op=="MatMul"){ y=[g matrixMultiplicationWithPrimaryTensor:get(in[0]) secondaryTensor:get(in[1]) name:nil]; }
    else if(op=="HardSigmoid"){
      double a=at[@"alpha"]?[at[@"alpha"] doubleValue]:0.2, b=at[@"beta"]?[at[@"beta"] doubleValue]:0.5;
      MPSGraphTensor* z=[g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:get(in[0]) secondaryTensor:[g constantWithScalar:a dataType:SDT] name:nil] secondaryTensor:[g constantWithScalar:b dataType:SDT] name:nil];
      y=[g clampWithTensor:z minValueTensor:[g constantWithScalar:0.0 dataType:SDT] maxValueTensor:[g constantWithScalar:1.0 dataType:SDT] name:nil];
    }
    else if(op=="ReduceMean"){
      std::vector<long> ax = at[@"axes"]? mrb_ints(at[@"axes"]) : std::vector<long>{};
      MPSGraphTensor* r=[g meanOfTensor:get(in[0]) axes:mrb_nums(ax) name:nil];
      long keep=at[@"keepdims"]?[at[@"keepdims"] longValue]:1;
      y = keep? r : [g squeezeTensor:r axes:mrb_nums(ax) name:nil];
    }
    else if(op=="Transpose"){ y=[g transposeTensor:get(in[0]) permutation:mrb_nums(mrb_ints(at[@"perm"])) name:nil]; }
    else if(op=="Squeeze"){
      // A squeezed shape-vector stays the same flat int list (host ints are
      // rank-agnostic), so the host chain just carries through.
      if(hostIn(in[0])){ hostInts[[out[0] UTF8String]]=hostInts[[in[0] UTF8String]]; }
      else {
        std::vector<long> ax = at[@"axes"]? mrb_ints(at[@"axes"]) : std::vector<long>{};
        y = ax.empty()? [g squeezeTensor:get(in[0]) name:nil] : [g squeezeTensor:get(in[0]) axes:mrb_nums(ax) name:nil];
      }
    }
    else if(op=="Unsqueeze"){
      if(hostIn(in[0])){ hostInts[[out[0] UTF8String]]=hostInts[[in[0] UTF8String]]; }
      else y=[g expandDimsOfTensor:get(in[0]) axes:mrb_nums(at[@"axes"]?mrb_ints(at[@"axes"]):std::vector<long>{}) name:nil];
    }
    else if(op=="BatchNormalization"){
      double eps=at[@"epsilon"]?[at[@"epsilon"] doubleValue]:1e-5;
      MPSGraphTensor* x0=get(in[0]); NSUInteger rank=x0.shape.count; if(rank<2) rank=4;
      NSMutableArray* ps=[NSMutableArray array]; for(NSUInteger k=0;k<rank;k++)[ps addObject:(k==1?@(-1):@1)];
      auto cN=[&](id v){ return [g reshapeTensor:get(v) withShape:ps name:nil]; };
      y=[g normalizationWithTensor:x0 meanTensor:cN(in[3]) varianceTensor:cN(in[4]) gammaTensor:cN(in[1]) betaTensor:cN(in[2]) epsilon:eps name:nil];
    }
    else if(op=="Conv"){
      std::vector<long> st=at[@"strides"]?mrb_ints(at[@"strides"]):std::vector<long>{1,1};
      std::vector<long> dl=at[@"dilations"]?mrb_ints(at[@"dilations"]):std::vector<long>{1,1};
      std::vector<long> pd=at[@"pads"]?mrb_ints(at[@"pads"]):std::vector<long>{0,0,0,0};
      long grp=at[@"group"]?[at[@"group"] longValue]:1;
      NSString* apc=at[@"auto_pad"];  // SAME_UPPER/SAME_LOWER => compute pads from input+kernel
      if(apc && ([apc isEqualToString:@"SAME_UPPER"]||[apc isEqualToString:@"SAME_LOWER"])){
        MPSGraphTensor* x0=get(in[0]); long inH=[x0.shape[2] longValue], inW=[x0.shape[3] longValue];
        std::vector<long> ks = at[@"kernel_shape"]? mrb_ints(at[@"kernel_shape"])
          : std::vector<long>{wshape[[in[1] UTF8String]][2], wshape[[in[1] UTF8String]][3]};
        bool up=[apc isEqualToString:@"SAME_UPPER"];
        auto same=[&](long i_,long k,long s,long dz,long& b,long& e){ long eff=(k-1)*dz+1; long o=(i_+s-1)/s; long tot=(o-1)*s+eff-i_; if(tot<0)tot=0; b=up?tot/2:tot-tot/2; e=tot-b; };
        long tb,te,lb,le; same(inH,ks[0],st[0],dl[0],tb,te); same(inW,ks[1],st[1],dl[1],lb,le);
        pd={tb,lb,te,le};  // [top,left,bottom,right]
      }
      MPSGraphConvolution2DOpDescriptor* cd=[MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:st[1] strideInY:st[0] dilationRateInX:dl[1] dilationRateInY:dl[0]
        groups:grp paddingLeft:pd[1] paddingRight:pd[3] paddingTop:pd[0] paddingBottom:pd[2]
        paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW
        weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
      MPSGraphTensor* c=[g convolution2DWithSourceTensor:get(in[0]) weightsTensor:get(in[1]) descriptor:cd name:nil];
      if([in count]>2) c=[g additionWithPrimaryTensor:c secondaryTensor:[g reshapeTensor:get(in[2]) withShape:@[@1,@(-1),@1,@1] name:nil] name:nil];
      y=c;
    }
    else if(op=="AveragePool"){
      std::vector<long> ks=mrb_ints(at[@"kernel_shape"]);
      std::vector<long> st=at[@"strides"]?mrb_ints(at[@"strides"]):ks;
      std::vector<long> pd=at[@"pads"]?mrb_ints(at[@"pads"]):std::vector<long>{0,0,0,0};
      MPSGraphPooling2DOpDescriptor* pdd=[MPSGraphPooling2DOpDescriptor
        descriptorWithKernelWidth:ks[1] kernelHeight:ks[0] strideInX:st[1] strideInY:st[0]
        dilationRateInX:1 dilationRateInY:1 paddingLeft:pd[1] paddingRight:pd[3] paddingTop:pd[0] paddingBottom:pd[2]
        paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
      y=[g avgPooling2DWithSourceTensor:get(in[0]) descriptor:pdd name:nil];
    }
    else if(op=="Sigmoid"){ y=[g sigmoidWithTensor:get(in[0]) name:nil]; }
    else if(op=="GlobalAveragePool"){ y=[g meanOfTensor:get(in[0]) axes:@[@2,@3] name:nil]; }  // keepdims -> [N,C,1,1]
    else if(op=="Concat"){
      // Shape concat when EVERY operand is an int vector (a host-int carrier or a
      // small integer initializer) and at least one is host-derived. Testing only
      // in[0] misses the common `Concat(const, const, <dyn dim>, const)` that
      // builds a reshape target — that would concat real tensors and the Reshape
      // consuming it would then resolve an empty target.
      bool anyHost=false, allInt=true;
      for(id v in in){ std::string nm=[v UTF8String];
        if(hostInts.count(nm)) anyHost=true;
        else if(!woff.count(nm)) allInt=false; }
      if(anyHost && allInt){  // shape concat -> host ints
        std::vector<long> r; for(id v in in){ auto iv=asInts(v); r.insert(r.end(),iv.begin(),iv.end()); }
        hostInts[[out[0] UTF8String]]=r;
      } else {
        NSMutableArray* ts=[NSMutableArray array]; for(id v in in)[ts addObject:get(v)];
        y=[g concatTensors:ts dimension:[at[@"axis"] longValue] name:nil];
      }
    }
    else if(op=="Shape"){
      std::vector<long> s; for(NSNumber* nn in get(in[0]).shape)s.push_back(nn.longValue);
      hostInts[[out[0] UTF8String]]=s;
    }
    else if(op=="Slice" && T.count([in[0] UTF8String]) && !hostInts.count([in[0] UTF8String])){
      // DATA slice on a real tensor (SVTR splits the packed [3,B,heads,T,d] QKV
      // with three axis-0 slices). Distinguished from the shape-vector slice
      // below purely by what in[0] resolved to: a graph tensor vs host ints.
      MPSGraphTensor* x=get(in[0]);
      const long rank=(long)x.shape.count;
      std::vector<long> st = at[@"starts"]? mrb_ints(at[@"starts"]) : ([in count]>1?asInts(in[1]):std::vector<long>{});
      std::vector<long> en = at[@"ends"]?   mrb_ints(at[@"ends"])   : ([in count]>2?asInts(in[2]):std::vector<long>{});
      std::vector<long> ax = at[@"axes"]?   mrb_ints(at[@"axes"])   : ([in count]>3?asInts(in[3]):std::vector<long>{});
      std::vector<long> sp = at[@"steps"]?  mrb_ints(at[@"steps"])  : ([in count]>4?asInts(in[4]):std::vector<long>{});
      if(ax.empty()) for(size_t i=0;i<st.size();i++) ax.push_back((long)i);
      MPSGraphTensor* cur=x;
      for(size_t i=0;i<ax.size() && i<st.size() && i<en.size(); i++){
        long a=ax[i]; if(a<0)a+=rank;
        const long dim=[cur.shape[a] longValue];
        long s=st[i], e=en[i];
        // int64 sentinels survive the fp32 weights blob as ~9.2e18; clamp rather
        // than llround them into an out-of-range length.
        if(s<0){ s = (s<-(1L<<40)) ? 0 : s+dim; }
        if(e<0){ e = (e<-(1L<<40)) ? 0 : e+dim; }
        s=std::max(0L,std::min(s,dim)); e=std::max(s,std::min(e,dim));
        if(i<sp.size() && sp[i]!=1) mrb_fail("Slice step "+std::to_string(sp[i])+" unsupported");
        if(s==0 && e==dim) continue;
        cur=[g sliceTensor:cur dimension:a start:s length:(e-s) name:nil];
      }
      y=cur;
    }
    else if(op=="Slice"){  // shape-tensor slice; starts/ends via attrs (opset<10) or inputs (opset10+)
      std::vector<long> data=asInts(in[0]);
      std::vector<long> st = at[@"starts"]? mrb_ints(at[@"starts"]) : ([in count]>1?asInts(in[1]):std::vector<long>{0});
      std::vector<long> en = at[@"ends"]?   mrb_ints(at[@"ends"])   : ([in count]>2?asInts(in[2]):std::vector<long>{(long)data.size()});
      long s0=st.empty()?0:st[0], e0=en.empty()?(long)data.size():en[0];
      if(e0>(long)data.size())e0=data.size(); if(s0<0)s0+=data.size(); if(e0<0)e0+=data.size();
      hostInts[[out[0] UTF8String]]=std::vector<long>(data.begin()+s0, data.begin()+e0);
    }
    else if(op=="Reshape"){
      MPSGraphTensor* x=get(in[0]);
      std::vector<long> xs; for(NSNumber* nn in x.shape)xs.push_back(nn.longValue);
      std::vector<long> tgt=asInts(in[1]);
      long total=1; for(long dd:xs)total*=dd; long known=1; int inferIdx=-1;
      for(size_t i=0;i<tgt.size();i++){ if(tgt[i]==0 && i<xs.size())tgt[i]=xs[i]; if(tgt[i]==-1)inferIdx=(int)i; else known*=tgt[i]; }
      if(inferIdx>=0) tgt[inferIdx]= known? total/known : 0;
      NSMutableArray* sh=[NSMutableArray array]; for(long dd:tgt)[sh addObject:@(dd)];
      y=[g reshapeTensor:x withShape:sh name:nil];
    }
    else if(op=="MaxPool"){
      std::vector<long> ks=mrb_ints(at[@"kernel_shape"]);
      std::vector<long> st=at[@"strides"]?mrb_ints(at[@"strides"]):std::vector<long>{1,1};
      long pl=0,pr=0,pt=0,pb=0;
      NSString* ap=at[@"auto_pad"];
      if(ap && [ap isEqualToString:@"SAME_UPPER"]){  // end-heavy pad to keep size (k>s)
        long th=ks[0]-st[0]; if(th<0)th=0; pt=th/2; pb=th-pt;
        long tw=ks[1]-st[1]; if(tw<0)tw=0; pl=tw/2; pr=tw-pl;
      } else if(at[@"pads"]){ auto pd=mrb_ints(at[@"pads"]); pt=pd[0];pl=pd[1];pb=pd[2];pr=pd[3]; }
      MPSGraphPooling2DOpDescriptor* pdd=[MPSGraphPooling2DOpDescriptor
        descriptorWithKernelWidth:ks[1] kernelHeight:ks[0] strideInX:st[1] strideInY:st[0]
        dilationRateInX:1 dilationRateInY:1 paddingLeft:pl paddingRight:pr paddingTop:pt paddingBottom:pb
        paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
      y=[g maxPooling2DWithSourceTensor:get(in[0]) descriptor:pdd name:nil];
    }
    else if(op=="Resize"){
      // ONNX nearest + asymmetric + floor at INTEGER scale [1,1,S,S] == exact pixel
      // replication: [N,C,H,W] -> [N,C,H,1,W,1] -> broadcast [.,.,H,S,W,S] -> [.,.,H*S,W*S].
      // Static-shape (keeps downstream broadcasts valid) and bit-exact — no resize-op
      // coordinate/rounding ambiguity.
      MPSGraphTensor* x=get(in[0]);
      long N=[x.shape[0] longValue],C=[x.shape[1] longValue],H=[x.shape[2] longValue],Wd=[x.shape[3] longValue];
      const float* sc=cvals(in[2]);
      long sh=(long)llround(sc[2]), sw=(long)llround(sc[3]);
      MPSGraphTensor* a=[g reshapeTensor:x withShape:@[@(N),@(C),@(H),@1,@(Wd),@1] name:nil];
      MPSGraphTensor* b=[g broadcastTensor:a toShape:@[@(N),@(C),@(H),@(sh),@(Wd),@(sw)] name:nil];
      y=[g reshapeTensor:b withShape:@[@(N),@(C),@(H*sh),@(Wd*sw)] name:nil];
    }
    else if(op=="ConvTranspose"){
      MPSGraphTensor* x=get(in[0]);
      std::vector<long> st=at[@"strides"]?mrb_ints(at[@"strides"]):std::vector<long>{1,1};
      std::vector<long> ks=mrb_ints(at[@"kernel_shape"]);
      std::vector<long> pd=at[@"pads"]?mrb_ints(at[@"pads"]):std::vector<long>{0,0,0,0};
      long inH=[x.shape[2] longValue], inW=[x.shape[3] longValue];
      long outH=(inH-1)*st[0]+ks[0]-pd[0]-pd[2];
      long outW=(inW-1)*st[1]+ks[1]-pd[1]-pd[3];
      long Cout=wshape[[in[1] UTF8String]][1];  // ONNX transpose weight [Cin,Cout,kH,kW]
      // MPSGraph transpose conv reads the weight in ONNX-native IOHW order; no swap.
      MPSGraphConvolution2DOpDescriptor* cd=[MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:st[1] strideInY:st[0] dilationRateInX:1 dilationRateInY:1
        groups:(at[@"group"]?[at[@"group"] longValue]:1)
        paddingLeft:pd[1] paddingRight:pd[3] paddingTop:pd[0] paddingBottom:pd[2]
        paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW
        weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
      MPSGraphTensor* c=[g convolutionTranspose2DWithSourceTensor:x weightsTensor:get(in[1])
        outputShape:@[@1,@(Cout),@(outH),@(outW)] descriptor:cd name:nil];
      if([in count]>2) c=[g additionWithPrimaryTensor:c secondaryTensor:[g reshapeTensor:get(in[2]) withShape:@[@1,@(-1),@1,@1] name:nil] name:nil];
      y=c;
    }
    else mrb_fail("UNHANDLED op '"+op+"'");
    if(dbg) std::fprintf(stderr,"      -> %s %s\n", [out[0] UTF8String], shp(y).c_str());
    if(y) T[[out[0] UTF8String]]=y;  // shape-subgraph ops store host ints, not a tensor
    idx++;
  }
  // MPS_OUT=<tensor name> overrides the output tensor (bisection debugging).
  const char* ov=getenv("MPS_OUT");  // pre-commit-allow-getenv (dev-only output-tensor bisection knob, not operator config)
  MPSGraphTensor* outT = (ov && T.count(ov)) ? T[ov] : T[[G[@"output"] UTF8String]];
  if(fp16) outT=[g castTensor:outT toType:MPSDataTypeFloat32 name:@"outf32"];  // back to FP32 for the harness
  return { ph, outT, inName, ishape };
}
