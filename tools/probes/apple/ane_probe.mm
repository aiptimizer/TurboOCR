// Stage 1: confirm native CoreML/ANE loading + predict works and measure ANE ms.
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
using clk=std::chrono::steady_clock;
static double ms(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }

int main(int argc,char** argv){ @autoreleasepool{
  // Default derives from $HOME rather than naming one developer's account, so
  // the probe runs for anyone who has the standard package layout.
  std::string dflt = std::string(getenv("HOME") ? getenv("HOME") : ".") +
                     "/.apple_ocr_ml/coreml/rec_ane_320.mlpackage";
  const char* pkg = argc>1?argv[1]:dflt.c_str();
  int B = argc>2?atoi(argv[2]):48;
  int W = argc>3?atoi(argv[3]):320;
  NSError* err=nil;
  NSURL* pkgURL=[NSURL fileURLWithPath:[NSString stringWithUTF8String:pkg]];
  NSURL* compiled=[MLModel compileModelAtURL:pkgURL error:&err];
  if(!compiled){ std::fprintf(stderr,"compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
  std::printf("compiled -> %s\n", compiled.path.UTF8String);
  MLModelConfiguration* cfg=[[MLModelConfiguration alloc] init];
  cfg.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
  MLModel* model=[MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
  if(!model){ std::fprintf(stderr,"load failed: %s\n", err.localizedDescription.UTF8String); return 1; }
  std::printf("loaded model (CPU_AND_NE)\n");

  // input MLMultiArray [B,3,48,W] fp32 NCHW contiguous
  std::vector<float> buf((size_t)B*3*48*W, 0.1f);
  NSArray* shape=@[@(B),@3,@48,@(W)];
  NSArray* strides=@[@(3*48*W),@(48*W),@(W),@1];
  MLMultiArray* arr=[[MLMultiArray alloc] initWithDataPointer:buf.data() shape:shape dataType:MLMultiArrayDataTypeFloat32 strides:strides deallocator:nil error:&err];
  if(!arr){ std::fprintf(stderr,"multiarray failed: %s\n", err.localizedDescription.UTF8String); return 1; }
  MLDictionaryFeatureProvider* fp=[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":[MLFeatureValue featureValueWithMultiArray:arr]} error:&err];

  // warm
  for(int i=0;i<5;i++){ id<MLFeatureProvider> o=[model predictionFromFeatures:fp error:&err]; if(!o){std::fprintf(stderr,"predict err %s\n",err.localizedDescription.UTF8String);return 1;} }
  MLMultiArray* idx=[[[model predictionFromFeatures:fp error:&err] featureValueForName:@"var_656"] multiArrayValue];
  MLMultiArray* sc =[[[model predictionFromFeatures:fp error:&err] featureValueForName:@"reduce_max_0"] multiArrayValue];
  std::printf("out var_656 shape=%s dtype=%ld | reduce_max_0 shape=%s\n",
    idx.shape.description.UTF8String, (long)idx.dataType, sc.shape.description.UTF8String);

  const int N=40; auto t=clk::now();
  for(int i=0;i<N;i++){ (void)[model predictionFromFeatures:fp error:&err]; }
  double per=ms(t)/N;
  std::printf("ANE predict batch=%d W=%d: %.2f ms/batch = %.0f crops/s\n", B, W, per, 1000.0/per*B);
  return 0;
}}
