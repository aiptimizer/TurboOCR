// Generic MPSGraph runner: rebuild an exported ONNX model (graph.json+weights.bin
// via tools/modelgen/mps_export_rec.py) in MPSGraph using the shared builder, validate
// bit-accuracy vs the ORT golden, then benchmark batched GPU inference.
// Works for any model the builder covers (rec_tiny, det_tiny, ...).
//
// Build (run from the repo root):
//   clang++ -std=c++17 -ObjC++ -fobjc-arc -O2 -Itools/probes/apple tools/probes/apple/mps_rec.mm \
//     -framework Metal -framework MetalPerformanceShaders \
//     -framework MetalPerformanceShadersGraph -framework Foundation -o build-cpu/mps_rec
// Run:  ./build-cpu/mps_rec <export_dir> [batch]
//
// The shared builder itself now lives at src/backends/apple/engine/mps_rec_build.h
// (it is library code, linked into the Apple backend of the server). The
// `#include "mps_rec_build.h"` below picks up tools/probes/apple/mps_rec_build.h, a one-line
// forwarding header pointing there, so this recipe needs no -Isrc/backends.
// Same for every other tools/probes/apple/mps_*.mm probe.

#import <Metal/Metal.h>
#include "mps_rec_build.h"
#include <chrono>
#include <algorithm>
#include <cmath>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

int main(int argc, char** argv){
 @autoreleasepool{
  std::string dir = argc>1? argv[1] : "rec_export";
  int B = argc>2? atoi(argv[2]) : 1;
  NSString* d=[NSString stringWithUTF8String:dir.c_str()];

  NSData* jd=[NSData dataWithContentsOfFile:[d stringByAppendingPathComponent:@"graph.json"]];
  NSDictionary* G=[NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
  NSData* W=[NSData dataWithContentsOfFile:[d stringByAppendingPathComponent:@"weights.bin"]];

  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> q=[dev newCommandQueue];
  MPSGraph* g=[MPSGraph new];
  std::printf("device: %s | rebuilding %lu nodes, %lu weights (batch=%d)\n", dev.name.UTF8String,
              (unsigned long)[G[@"nodes"] count], (unsigned long)[G[@"initializers"] count], B);
  RecIO io=buildRecGraph(g, G, (const float*)W.bytes, B);
  std::printf("graph built OK\n");

  // input (tile the single golden input to B rows)
  NSData* xin=[NSData dataWithContentsOfFile:[d stringByAppendingPathComponent:@"input.bin"]];
  NSData* xB; if(B==1) xB=xin; else { NSMutableData* m=[NSMutableData data]; for(int i=0;i<B;i++)[m appendData:xin]; xB=m; }
  MPSGraphTensorData* xTD=[[MPSGraphTensorData alloc] initWithDevice:[MPSGraphDevice deviceWithMTLDevice:dev]
                                                                data:xB shape:mrb_nums(io.ishape) dataType:MPSDataTypeFloat32];

  MPSGraphTensorData* r=[g runWithMTLCommandQueue:q feeds:@{io.input:xTD} targetTensors:@[io.output] targetOperations:nil][io.output];
  std::vector<long> os; for(NSNumber* n in r.shape) os.push_back(n.longValue);
  size_t oel=1; for(long v:os) oel*=v;
  std::printf("output shape: ["); for(long v:os)std::printf("%ld ",v); std::printf("]\n");
  if(B==1){
    std::vector<float> mine(oel); [r.mpsndarray readBytes:mine.data() strideBytes:nil];
    const char* gpath=getenv("MPS_GOLDEN");
    NSData* gold=gpath? [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:gpath]]
                      : [NSData dataWithContentsOfFile:[d stringByAppendingPathComponent:@"golden.bin"]];
    const float* gp=(const float*)gold.bytes; size_t gel=gold.length/4;
    if(oel==gel){ double maxabs=0,sum2=0; for(size_t i=0;i<oel;i++){ double e=std::fabs(mine[i]-gp[i]); maxabs=std::max(maxabs,e); sum2+=e*e; }
      std::printf("CORRECTNESS vs ORT golden: max_abs_err=%.5e  rms_err=%.5e  %s\n", maxabs, std::sqrt(sum2/oel), maxabs<1e-2?"PASS":"CHECK"); }
    else std::printf("shape mismatch vs golden (%zu vs %zu)\n", oel, gel);
  }

  for(int i=0;i<10;i++) (void)[g runWithMTLCommandQueue:q feeds:@{io.input:xTD} targetTensors:@[io.output] targetOperations:nil];
  std::vector<double> t;
  for(int i=0;i<50;i++){ auto s=clk::now(); (void)[g runWithMTLCommandQueue:q feeds:@{io.input:xTD} targetTensors:@[io.output] targetOperations:nil]; t.push_back(ms_since(s)); }
  std::sort(t.begin(),t.end());
  std::printf("\nGPU wall (incl host-sync): median %.3f ms  min %.3f ms  (batch=%d, %.3f ms/item)\n", t[t.size()/2], t.front(), B, t[t.size()/2]/B);
  return 0;
 }
}
