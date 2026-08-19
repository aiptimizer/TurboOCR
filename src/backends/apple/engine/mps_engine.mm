// MpsEngine implementation (see mps_engine.h).
//
// Wraps the src/backends/apple/engine/mps_rec_build.h translator (shared with the
// standalone tools/probes/apple/mps_*.mm probes): parse graph.json + weights.bin,
// build an MPSGraph per batch size (cached), compile to an MPSGraphExecutable,
// and encodeToCommandBuffer on the caller's queue with MTLBuffer-backed
// MPSGraphTensorData. This is the mps_ocr.mm loadModel + compile + encode path
// (tools/probes/apple/mps_ocr.mm:30-37, 85-93, 132-134) lifted behind IEngine.

#import "apple/engine/mps_engine.h"
#import "apple/support/metal_common.h"
#import "apple/queue/metal_device_queue.h"
#import "apple/support/apple_contention.h"

#import <Foundation/Foundation.h>

#include "apple/engine/mps_rec_build.h" // buildRecGraph + mrb_nums (src/backends is on the -I path)

#include <unordered_map>

namespace turbo_ocr::apple {

// One compiled graph for a given batch size.
struct BuiltGraph {
  MPSGraph *g = nil;
  RecIO io{};
  MPSGraphExecutable *exe = nil;
  MPSGraphTensor *idxT = nil;   // argmax-head targets (nil when disabled)
  MPSGraphTensor *maxT = nil;
  long T = 0;                   // time steps
  long C = 0;                   // classes
};

struct MpsEngine::Impl {
  NSDictionary *graph_json = nil;
  NSData *weights = nil;           // must outlive every built graph (no-copy consts)
  std::string input_name;
  std::string output_name;
  std::vector<long> input_shape;   // template [1,3,H,W] from the export
  std::unordered_map<int, BuiltGraph> by_batch; // batch -> compiled graph
};

// Build + compile (or fetch from cache) the graph for a given batch size.
// Returns nullptr when the export cannot be translated — buildRecGraph throws
// on a tensor it cannot resolve or an op it does not implement (it used to
// exit() the process; see mps_rec_build.h). Nothing is cached on that path, so
// a later call with a different batch size still gets a clean attempt.
static BuiltGraph *ensure_built(MpsEngine::Impl &p, int batch, bool argmax_head,
                                int class_axis, bool fp16, int &T_out, int &C_out) {
  auto it = p.by_batch.find(batch);
  if (it != p.by_batch.end()) { T_out = (int)it->second.T; C_out = (int)it->second.C; return &it->second; }

  BuiltGraph bg;
  bg.g = [MPSGraph new];
  try {
    bg.io = buildRecGraph(bg.g, p.graph_json, (const float *)p.weights.bytes, batch, fp16);
  } catch (const std::exception &e) {
    // Same shape as every other failure in this file: log and report false to
    // the caller, which turns into MpsRecognizer::load() declining the stage
    // (it eagerly prepare()s every ladder rung at load time).
    NSLog(@"[apple] MpsEngine: rec graph build failed at batch %d: %s", batch, e.what());
    return nullptr;
  }

  NSArray<MPSGraphTensor *> *targets;
  if (argmax_head) {
    bg.idxT = [bg.g reductionArgMaximumWithTensor:bg.io.output axis:class_axis name:nil];
    bg.maxT = [bg.g reductionMaximumWithTensor:bg.io.output axis:class_axis name:nil];
    targets = @[ bg.idxT, bg.maxT ];
  } else {
    targets = @[ bg.io.output ];
  }
  // Static I/O shapes -> a fixed executable (mps_ocr.mm:85 / :132).
  bg.exe = [bg.g compileWithDevice:mps_graph_device()
                             feeds:@{bg.io.input : [MPSGraphShapedType.alloc
                                        initWithShape:mrb_nums(bg.io.ishape)
                                             dataType:MPSDataTypeFloat32]}
                     targetTensors:targets
                  targetOperations:nil
             compilationDescriptor:nil];
  // A nil executable is a FAILED build, not a usable one: run() would encode
  // nothing (message-to-nil), submit an empty command buffer and still return
  // true, so the caller would read its untouched output buffers as logits and
  // get plausible-looking garbage instead of "rec forward FAILED". Bail before
  // the emplace below so nothing nil is cached either — a poisoned cache entry
  // would also make every later attempt at this batch size skip the compile.
  if (!bg.exe) {
    NSLog(@"[apple] MpsEngine: graph compile returned nil at batch %d", batch);
    return nullptr;
  }
  // Output dims (from the raw output tensor): [B, T, C].
  if (bg.io.output.shape.count >= 3) {
    bg.T = [bg.io.output.shape[1] longValue];
    bg.C = [bg.io.output.shape[2] longValue];
  }
  T_out = (int)bg.T;
  C_out = (int)bg.C;
  auto res = p.by_batch.emplace(batch, bg);
  return &res.first->second;
}

MpsEngine::MpsEngine() : p_(std::make_unique<Impl>()) {}
MpsEngine::~MpsEngine() = default;

bool MpsEngine::load(const std::string &model_path) {
  @autoreleasepool {
    NSString *dir = [NSString stringWithUTF8String:model_path.c_str()];
    NSData *jd = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"graph.json"]];
    NSData *wd = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"weights.bin"]];
    if (!jd || !wd) {
      NSLog(@"[apple] MpsEngine::load missing graph.json/weights.bin in %s", model_path.c_str());
      return false;
    }
    NSError *err = nil;
    NSDictionary *G = [NSJSONSerialization JSONObjectWithData:jd options:0 error:&err];
    if (!G) { NSLog(@"[apple] graph.json parse failed: %@", err); return false; }
    p_->graph_json = G;
    p_->weights = wd;
    p_->input_name = [G[@"input"] UTF8String];
    p_->output_name = [G[@"output"] UTF8String];
    p_->input_shape = mrb_ints(G[@"input_shape"]);
    input_names_ = {p_->input_name};
    output_names_ = {p_->output_name};
  }
  return true;
}

bool MpsEngine::load_shared(const MpsEngine &src, int input_h, int input_w) {
  if (!src.p_ || !src.p_->graph_json || !src.p_->weights) return false;
  if (src.p_->input_shape.size() < 4 || input_h <= 0 || input_w <= 0) return false;
  @autoreleasepool {
    // SHALLOW copy of the parsed export: the top-level dict is duplicated so
    // input_shape can differ, but "nodes"/"initializers" (the big arrays) and
    // the weights NSData stay the very same ObjC objects — ARC strong refs in
    // both Impls keep them alive for as long as either engine (or any graph
    // built from them: the constants are no-copy views into `weights`) exists.
    NSMutableDictionary *G = [src.p_->graph_json mutableCopy];
    std::vector<long> ishape = src.p_->input_shape;
    ishape[ishape.size() - 2] = input_h;
    ishape[ishape.size() - 1] = input_w;
    G[@"input_shape"] = mrb_nums(ishape);
    p_->graph_json = G;
    p_->weights = src.p_->weights;
    p_->input_name = src.p_->input_name;
    p_->output_name = src.p_->output_name;
    p_->input_shape = std::move(ishape);
    p_->by_batch.clear();
    input_names_ = {p_->input_name};
    output_names_ = src.output_names_;
    fp16_ = src.fp16_;
    argmax_head_ = src.argmax_head_;
    class_axis_ = src.class_axis_;
  }
  return true;
}

void MpsEngine::set_fp16(bool on) {
  if (fp16_ == on) return;
  fp16_ = on;
  if (p_) p_->by_batch.clear(); // cached executables were built at the old dtype
}

void MpsEngine::enable_argmax_head(bool on, int class_axis) {
  argmax_head_ = on;
  class_axis_ = class_axis;
  if (on) output_names_ = {"__argmax_idx", "__argmax_max"};
  else if (p_) output_names_ = {p_->output_name};
  // Force a rebuild: cached graphs targeted the previous head configuration.
  if (p_) p_->by_batch.clear();
}

bool MpsEngine::prepare(int batch) {
  if (!p_ || !p_->graph_json || !p_->weights) return false;
  @autoreleasepool {
    BuiltGraph *bg =
        ensure_built(*p_, batch, argmax_head_, class_axis_, fp16_, time_steps_, num_classes_);
    return bg && bg->exe != nil;
  }
}

std::vector<int> MpsEngine::input_chw() const {
  if (!p_ || p_->input_shape.size() < 4) return {};
  return {(int)p_->input_shape[1], (int)p_->input_shape[2], (int)p_->input_shape[3]};
}

backend::EngineCaps MpsEngine::caps() const {
  backend::EngineCaps c;
  c.io_space = backend::DeviceKind::Metal;
  c.async = true;                 // results ready after queue.synchronize()
  c.caller_owns_outputs = true;   // engine writes into caller MTLBuffers
  c.multi_io = false;             // single-input det/rec/cls
                                  // (TODO(apple-layout-multi-io): layout needs multi-IO)
  c.dynamic_shapes = false;       // one compiled executable per batch size
  c.graph = false;                // MPSGraph plans internally; no external capture
  c.has_profiles = false;
  c.thread_safe_concurrent = false; // one engine per pipeline thread
  c.dtypes = {backend::DType::F32, backend::DType::I32};
  return c;
}

bool MpsEngine::run(const std::vector<backend::DeviceTensor> &inputs,
                    const std::vector<backend::DeviceTensor> &outputs,
                    std::vector<backend::OutputLease> &leases,
                    backend::DeviceQueue &queue) {
  (void)leases; // caller-owns-outputs: no lease
  if (inputs.empty() || inputs[0].shape.empty()) return false;
  if (inputs[0].space != backend::DeviceKind::Metal) {
    NSLog(@"[apple] MpsEngine::run input must be Metal-resident");
    return false;
  }
  @autoreleasepool {
    const int batch = (int)inputs[0].shape[0];
    // Effectively pre-built: MpsRecognizer::load() prepare()s every ladder rung,
    // so reaching an untranslatable graph here means a batch size nobody
    // prepared. Drop the chunk (the stage logs "rec forward FAILED") instead of
    // dereferencing a null build.
    BuiltGraph *bgp = ensure_built(*p_, batch, argmax_head_, class_axis_, fp16_, time_steps_, num_classes_);
    if (!bgp) return false;
    BuiltGraph &bg = *bgp;

    // Bind the input MTLBuffer zero-copy.
    std::size_t in_off = 0;
    id<MTLBuffer> in_buf = resolve_buffer(inputs[0].data, &in_off);
    if (!in_buf) { NSLog(@"[apple] input not a registered Metal buffer"); return false; }
    NSMutableArray<NSNumber *> *in_shape = [NSMutableArray array];
    for (long d : inputs[0].shape) [in_shape addObject:@(d)];
    // MPSGraphTensorData can only bind a buffer at offset 0 (there is no
    // offset-taking initializer). resolve_buffer, however, happily resolves a
    // pointer INTO a registered buffer — so a pooled / sub-allocating Metal
    // allocator would silently bind every tensor to offset 0 and read the wrong
    // memory. Refuse loudly instead of computing the offset and discarding it.
    if (in_off != 0) {
      NSLog(@"[apple] MpsEngine::run: input at offset %zu inside its MTLBuffer; "
            @"MPSGraphTensorData cannot bind a sub-range — allocate this tensor "
            @"as its own buffer", in_off);
      return false;
    }
    MPSGraphTensorData *inTD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                shape:in_shape
                                             dataType:MPSDataTypeFloat32];

    // Bind caller outputs.
    NSMutableArray<MPSGraphTensorData *> *resultTDs = [NSMutableArray array];
    auto bind_out = [&](const backend::DeviceTensor &t, MPSDataType dt) -> bool {
      std::size_t off = 0;
      id<MTLBuffer> b = resolve_buffer(t.data, &off);
      if (!b) return false;
      NSMutableArray<NSNumber *> *sh = [NSMutableArray array];
      for (long d : t.shape) [sh addObject:@(d)];
      [resultTDs addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:sh dataType:dt]];
      return true;
    };

    if (argmax_head_) {
      if (outputs.size() < 2) { NSLog(@"[apple] argmax head needs 2 output buffers"); return false; }
      if (!bind_out(outputs[0], MPSDataTypeInt32)) return false; // indices
      if (!bind_out(outputs[1], MPSDataTypeFloat32)) return false; // scores
    } else {
      if (outputs.empty()) { NSLog(@"[apple] run needs 1 output buffer"); return false; }
      if (!bind_out(outputs[0], MPSDataTypeFloat32)) return false;
    }

    // Encode onto the queue's command buffer (shared with the batch when open).
    auto &mq = as_metal(queue);
    MPSCommandBuffer *cb = mq.acquire_cb();
    TURBO_APPLE_STAT(mps_encode);
    [bg.exe encodeToCommandBuffer:cb
                      inputsArray:@[ inTD ]
                     resultsArray:resultTDs
              executionDescriptor:nil];
    mq.submit_cb(cb); // no-op inside a batch; commits (no wait) otherwise
  }
  return true;
}

} // namespace turbo_ocr::apple
